"""Dataset-local evidence and caption workflows, with configuration-only model specs."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import uuid

from comfy_api.latest import io
from .comfy_input_help import string_input
from .Scripts.artifact_store import ArtifactStore, StoreError
from .Scripts.evidence import Bundle
from .Scripts.caption_files import (caption_document, combine_captions, document_from_tags, edit_caption,
    export_config, plan_caption_export, commit_caption_export)
from .Scripts.persistent_evidence import save_evidence, load_evidence, import_calculations
from .Scripts import workflow_api as api
from .comfy_pipeline_nodes import EVIDENCE, RESULTS
from .comfy_ui_nodes import FILES, OPTIONS

STORE=io.Custom('WDV3_CALCULATION_STORE')
INFERENCE=io.Custom('WDV3_INFERENCE')
PLAN=io.Custom('WDV3_CALCULATION_PLAN')
CAPTIONS=io.Custom('WDV3_CAPTIONS')
EXPORT=io.Custom('WDV3_CAPTION_EXPORT')
CAPTION_PLAN=io.Custom('WDV3_CAPTION_PLAN')
READ=io.Custom('WDV3_CAPTION_READ')
EDIT=io.Custom('WDV3_CAPTION_EDIT')


def open_store(settings):
    return ArtifactStore(settings['dataset_root'],settings.get('store_path'))


def store_request(settings):
    return dict(schema_version=1, dataset_root=settings['dataset_root'],store_path=settings.get('store_path'),
                mode=settings.get('mode','reuse'),refresh=settings.get('refresh',[]))


class WDCalculationStore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CalculationStore',display_name='WDV3 Calculation Store Settings',category='WDV3/Dataset',
            description='Необязательная настройка расширенного маршрута: папка постоянного evidence (.wdv3), режим и выбор пересчёта. Изображения не выбирает: это делает Dataset Files. В обычном графе кэш задаётся прямо в Tagging Settings.',
            inputs=[string_input('dataset_root',default='',display_name='dataset_root'),string_input('store_path',default=''),
                    io.Combo.Input('mode',options=['reuse','reuse_only','refresh'],default='reuse'),
                    string_input('refresh',display_name='refresh_selection',default='',tooltip='Имена источников или стадий, через запятую. Пусто при refresh — пересчитать всё.')],
            outputs=[STORE.Output('calculation_store')])
    @classmethod
    def execute(cls,dataset_root,store_path='',mode='reuse',refresh=''):
        root=Path(dataset_root).expanduser().resolve()
        if not dataset_root.strip() or not root.is_dir(): raise StoreError('dataset_root must exist')
        return io.NodeOutput(dict(dataset_root=str(root),store_path=store_path or None,mode=mode,
                                  refresh=[s.strip() for s in refresh.split(',') if s.strip()]))


class WDInferenceSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3InferenceSettings',display_name='WDV3 Inference Settings',category='WDV3/Dataset',
            description='Необязательный расширенный интерфейс описания моделей без загрузки, для Plan/Compute и агентного API. Для обычного Tag Image / Regions, Collect Evidence, Tag Folder и изменения фильтров не требуется: используйте прежние модели и Tagging Settings.',
            inputs=[string_input('wd_local_path',default=''),string_input('model_id',default='vit'),
                    io.Combo.Input('device',options=['cpu','cuda'],default='cpu'),
                    string_input('sources_json',display_name='region_detectors_json',default='[]',multiline=True,
                      tooltip='Массив источников: name, kind (yolo/sam), model с files/identity и локальными настройками, confidence/classes или SAM prompt, view_mode/padding_px, rules.'),
                    string_input('pinned_identity_json',default='{}',multiline=True,
                      tooltip='Явная сохранённая идентичность WD для планирования без локальных весов. Пустой объект — хешировать локальные файлы.')],
            outputs=[INFERENCE.Output('inference')])
    @classmethod
    def execute(cls,wd_local_path,model_id='vit',device='cpu',sources_json='[]',pinned_identity_json='{}'):
        folder=Path(wd_local_path).expanduser()
        if folder.suffix in ('.safetensors','.bin'): folder=folder.parent
        files={('weights' if name in ('model.safetensors','pytorch_model.bin') else name):str((folder/name).resolve())
               for name in ('model.safetensors','pytorch_model.bin','config.json','selected_tags.csv','preprocess.json') if (folder/name).is_file()}
        wd=dict(local_path=str(folder.resolve()),model_id=model_id,files=files,compute={'device':device},adapter='wdv3-v1')
        identity=json.loads(pinned_identity_json)
        if identity:wd['identity']=identity
        from .comfy_persistent_runtime import normalize_inference
        return io.NodeOutput(normalize_inference(dict(schema_version=1,wd=wd,sources=json.loads(sources_json))))


class WDDatasetFiles(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3DatasetFiles',display_name='WDV3 Dataset Files',category='WDV3/Dataset',
            description='Собирает изображения dataset_root для нового постоянного пайплайна. Исключает .wdv3 и выбранный store до обхода, поэтому сохранённые PNG/маски не попадут в датасет. Подключите files к Plan Calculations.',
            inputs=[STORE.Input('calculation_store'),io.Boolean.Input('recursive',default=True),
                    io.Int.Input('start',default=0,min=0),io.Int.Input('limit',default=0,min=0)],
            outputs=[FILES.Output('files'),io.Int.Output('image_count')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs):return float('nan')
    @classmethod
    def execute(cls,calculation_store,recursive=True,start=0,limit=0):
        store=open_store(calculation_store)
        paths=list(store.scan(recursive=recursive));paths=paths[start:start+limit] if limit else paths[start:]
        if not paths:raise StoreError('No dataset images match folder/range')
        return io.NodeOutput(dict(root=store.dataset_root,paths=paths),len(paths))


class WDPlanCalculations(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3PlanCalculations',display_name='WDV3 Plan Calculations',category='WDV3/Dataset',
            description='Проверяет evidence для IMAGE либо FILES до загрузки моделей. Выдаёт готовые/недостающие/повреждённые стадии и причины. Если геометрия ещё не рассчитана, WD регионов помечается как ожидающий геометрии.',
            inputs=[STORE.Input('calculation_store'),INFERENCE.Input('inference'),
                    io.Image.Input('images',optional=True),FILES.Input('files',optional=True),OPTIONS.Input('options',optional=True),
                    string_input('image_mapping_json',default='[]',multiline=True,tooltip='Для IMAGE: явные относительные имена изображений, по одному на кадр, JSON-массив.')],
            outputs=[PLAN.Output('plan'),io.String.Output('plan_json')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs): return float('nan')
    @classmethod
    def execute(cls,calculation_store,inference,images=None,files=None,options=None,image_mapping_json='[]'):
        if (images is None)==(files is None):raise StoreError('Provide exactly one of images/files')
        request=dict(store_request(calculation_store),inference=inference,options=dict(options or {}),operation_id=uuid.uuid4().hex)
        if files is not None:
            store=open_store(calculation_store)
            paths=files['paths'] if isinstance(files,dict) else files
            request['files']=[Path(p).resolve().relative_to(store.dataset_root).as_posix() for p in paths]
            result=api.plan(request)
            value=dict(request=request,report=result)
        else:
            from .comfy_processing import tensor_frame
            from .comfy_persistent_runtime import PersistentCalculator
            store=open_store(calculation_store);calc=PersistentCalculator(store,inference,options=dict(options or {}))
            mapping=json.loads(image_mapping_json)
            if mapping and len(mapping)!=len(images):raise StoreError('IMAGE mapping count mismatch')
            entries=[]
            try:
                with ExitStack() as stack:
                    frames=[stack.enter_context(tensor_frame(t)) for t in images]
                    for frame in frames:
                        entries.append(calc.plan_image(frame,mode=request['mode'],refresh=request['refresh']))
                    from .comfy_evidence import png
                    value=dict(request=request,report={'schema_version':1,'entries':entries},
                               image_pngs=[png(f) for f in frames],mapping=mapping)
            finally:calc.close()
        return io.NodeOutput(value,json.dumps(value['report'],ensure_ascii=False,indent=2))


class WDComputeMissing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ComputeMissingEvidence',display_name='WDV3 Compute Missing Evidence',category='WDV3/Dataset',
            description='Исполняет план: загружает модели только для отсутствующих стадий, сохраняет raw/PNG/маски в .wdv3 и возвращает evidence для фильтров. При reuse_only модельные вызовы запрещены. Не записывает конечные TXT.',
            inputs=[PLAN.Input('plan')],outputs=[EVIDENCE.Output('evidence'),io.String.Output('run_report'),io.String.Output('evidence_refs')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs): return float('nan')
    @classmethod
    def execute(cls,plan):
        request=plan['request']
        if 'image_pngs' not in plan:
            report=api.execute(request)
            refs=[r['evidence_ref'] for r in report['entries'] if r.get('evidence_ref')]
            if not report['complete']:raise StoreError(json.dumps(report,ensure_ascii=False))
            evidence=api.load(dict(request,artifact_ids=refs))
        else:
            from .comfy_persistent_runtime import PersistentCalculator
            from PIL import Image
            from io import BytesIO
            import base64
            store=api.open_store(request);calc=PersistentCalculator(store,request['inference'],options=request.get('options'))
            refs=[];reports=[]
            try:
                for i,encoded in enumerate(plan['image_pngs']):
                    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                        evidence,row=calc.compute_image(image,mode=request['mode'],refresh=request['refresh'],
                              relative_path=plan['mapping'][i] if plan.get('mapping') else None)
                    if evidence is None:raise StoreError(json.dumps(row,ensure_ascii=False))
                    refs.append(row['evidence_ref']);reports.append(row)
            finally:calc.close()
            report=dict(schema_version=1,entries=reports)
            evidence=api.load(dict(request,artifact_ids=refs))
        return io.NodeOutput(evidence,json.dumps(report,ensure_ascii=False,indent=2),json.dumps(refs))


class WDSaveEvidence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3SaveEvidence',display_name='WDV3 Save Evidence',category='WDV3/Dataset',
            description='Сохраняет готовое полное evidence в dataset-local store. Это raw-артефакт для последующей фильтрации, а не итоговая TXT-подпись. Возвращает переносимый artifact_id.',
            inputs=[STORE.Input('calculation_store'),EVIDENCE.Input('evidence')],outputs=[io.String.Output('artifact_id')],is_output_node=True)
    @classmethod
    def execute(cls,calculation_store,evidence):return io.NodeOutput(save_evidence(open_store(calculation_store),evidence))


class WDLoadEvidence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3LoadEvidence',display_name='WDV3 Load Evidence',category='WDV3/Dataset',
            description='Загружает evidence по artifact_id/JSON-массиву refs либо manifest run_id. Проверяет целостность без весов и моделей. Подключите к Filter Tag Sets/Apply Filters/Inspect; dataset mapping сохраняется.',
            inputs=[STORE.Input('calculation_store'),string_input('artifact_ids',default='',multiline=True),string_input('run_id',default=''),
                    io.Int.Input('start',default=0,min=0),io.Int.Input('limit',default=64,min=1)],
            outputs=[EVIDENCE.Output('evidence')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs):return float('nan')
    @classmethod
    def execute(cls,calculation_store,artifact_ids='',run_id='',start=0,limit=64):
        refs=json.loads(artifact_ids) if artifact_ids.strip().startswith('[') else [s.strip() for s in artifact_ids.split(',') if s.strip()]
        request=dict(store_request(calculation_store),artifact_ids=refs,run_id=run_id,start=start,limit=limit)
        return io.NodeOutput(api.load(request))


class WDImportCalculations(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ImportCalculations',display_name='WDV3 Import Calculations',category='WDV3/Dataset',
            description='Импортирует Bundle evidence либо results_json с полными raw_scores. TXT/merged-only JSON отклоняются. Отчёт capabilities указывает, доступны ли пиксели, маски и пересборка; старый JSON может быть refilter-only.',
            inputs=[STORE.Input('calculation_store'),string_input('calculations_json',default='',multiline=True)],
            outputs=[EVIDENCE.Output('evidence'),io.String.Output('import_report')])
    @classmethod
    def execute(cls,calculation_store,calculations_json):
        store=open_store(calculation_store);report=import_calculations(store,json.loads(calculations_json))
        return io.NodeOutput(load_evidence(store,report['artifact_id']),json.dumps(report,ensure_ascii=False))


class WDLoadCaptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3LoadCaptions',display_name='WDV3 Load Captions',category='WDV3/Captions',
            description='Читает готовые UTF-8 TXT датасета без моделей. Сохраняет bytes/digest, порядок и ручные теги без вероятности. .wdv3 исключена из обхода. Входной разделитель независим от формата итогового текста.',
            inputs=[STORE.Input('calculation_store'),io.Combo.Input('input_format',options=['comma_newline','literal_separator'],default='comma_newline'),
                    string_input('input_separator',default=', '),io.Boolean.Input('escaped_parentheses',default=False),
                    io.Boolean.Input('recursive',default=True),READ.Input('caption_read',optional=True)],outputs=[CAPTIONS.Output('captions'),io.String.Output('preview_json')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs):return float('nan')
    @classmethod
    def execute(cls,calculation_store,input_format='comma_newline',input_separator=', ',escaped_parentheses=False,recursive=True,caption_read=None):
        store=open_store(calculation_store)
        read=caption_read or dict(input_format=input_format,input_separator=input_separator,escaped_parentheses=escaped_parentheses)
        entries=[dict(relative_path=p.relative_to(store.dataset_root).as_posix(),document=caption_document(p.read_bytes(),path=p,read_config=read))
                 for p in store.scan(suffixes=['.txt'],recursive=recursive)]
        return io.NodeOutput(entries,json.dumps(entries,ensure_ascii=False,indent=2))


class WDEditCaptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3EditCaptions',display_name='WDV3 Edit Captions',category='WDV3/Captions',
            description='Применяет упорядоченные add_before/add_after/remove/remove_regex/replace/sort к подписям в памяти. Remove сильнее add в одной операции. Без записи TXT и без моделей; provenance и история сохраняются.',
            inputs=[CAPTIONS.Input('captions'),string_input('operations_json',default='[]',multiline=True),EDIT.Input('caption_edit',optional=True)],
            outputs=[CAPTIONS.Output('captions'),io.String.Output('preview_json')])
    @classmethod
    def execute(cls,captions,operations_json='[]',caption_edit=None):
        operations=caption_edit if caption_edit is not None else json.loads(operations_json)
        result=[dict(e,document=edit_caption(e['document'],operations)) for e in captions]
        return io.NodeOutput(result,json.dumps(result,ensure_ascii=False,indent=2))


class WDCombineCaptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CombineCaptions',display_name='WDV3 Combine Captions',category='WDV3/Captions',
            description='Преобразует готовые results в подписи либо дополняет существующие captions. Старые пользовательские теги сохраняются, новые дедуплицируются; фильтры модели не удаляют старые теги. Нужен dataset-relative mapping результатов.',
            inputs=[RESULTS.Input('results'),CAPTIONS.Input('captions',optional=True)],outputs=[CAPTIONS.Output('captions')])
    @classmethod
    def execute(cls,results,captions=None):
        new=api.captions_from_results(results)
        old={Path(e['relative_path']).with_suffix('.txt').as_posix():e for e in captions or []}
        combined=[]
        for entry in new:
            key=Path(entry['relative_path']).with_suffix('.txt').as_posix()
            previous=old.get(key)
            combined.append(dict(entry,document=combine_captions(previous['document'],entry['document']) if previous else entry['document']))
        return io.NodeOutput(combined)


class WDCaptionExportSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CaptionExportSettings',display_name='WDV3 Caption Export Settings',category='WDV3/Captions',
            description='Задаёт запись подписей: error/skip/append/overwrite, соседний image.txt (default), image.png.txt либо explicit mapping. Формат вывода берётся из options; чтение старого назначения задаётся отдельно. Настройка не пишет файлы.',
            inputs=[io.Combo.Input('write_mode',options=['error','skip','append','overwrite'],default='error'),
                    io.Combo.Input('naming',options=['stem_txt','image_extension_txt','explicit'],default='stem_txt'),
                    io.Boolean.Input('in_place',default=True),string_input('target_root',default=''),
                    string_input('mapping_json',default='{}',multiline=True),
                    io.Combo.Input('input_format',options=['comma_newline','literal_separator'],default='comma_newline'),
                    string_input('input_separator',default=', '),io.Boolean.Input('escaped_parentheses',default=False),OPTIONS.Input('options',optional=True)],
            outputs=[EXPORT.Output('caption_export')])
    @classmethod
    def execute(cls,write_mode='error',naming='stem_txt',in_place=True,target_root='',mapping_json='{}',input_format='comma_newline',input_separator=', ',escaped_parentheses=False,options=None):
        return io.NodeOutput(export_config(write_mode=write_mode,naming=naming,in_place=in_place,target_root=target_root or None,
             mapping=json.loads(mapping_json),options=dict(options or {}),read_config=dict(input_format=input_format,input_separator=input_separator,escaped_parentheses=escaped_parentheses)))


class WDPlanCaptionExport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3PlanCaptionExport',display_name='WDV3 Plan Caption Export',category='WDV3/Captions',
            description='Строит preview diff и проверяет коллизии, границы путей, режим записи и текущие digest TXT. Файлы не меняются. Сохранение — отдельный финальный Save Captions.',
            inputs=[STORE.Input('calculation_store'),CAPTIONS.Input('captions'),EXPORT.Input('caption_export'),string_input('operation_id',default='',tooltip='Пусто — автоматический ID конкретного плана. Явный ID для агента/повторения: другое содержимое с тем же ID отклоняется.')],
            outputs=[CAPTION_PLAN.Output('caption_plan'),io.String.Output('preview_json')])
    @classmethod
    def fingerprint_inputs(cls,**kwargs):return float('nan')
    @classmethod
    def execute(cls,calculation_store,captions,caption_export,operation_id=''):
        result=plan_caption_export(open_store(calculation_store),captions,operation_id=operation_id or 'auto',config=caption_export)
        if not operation_id:
            from .Scripts.artifact_store import canonical_bytes,sha256
            result['operation_id']='caption-'+sha256(canonical_bytes(result))
        return io.NodeOutput(dict(store=calculation_store,plan=result),json.dumps(result,ensure_ascii=False,indent=2))


class WDSaveCaptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3SaveCaptions',display_name='WDV3 Save Captions',category='WDV3/Captions',
            description='Последний этап: безопасно записывает подготовленные TXT по плану. Проверяет внешние правки, использует атомарную запись и журнал operation_id. Повтор возобновляет запись, не inference. Отчёт: written/skipped/unchanged/conflict/failed.',
            inputs=[CAPTION_PLAN.Input('caption_plan')],outputs=[io.String.Output('export_report')],is_output_node=True)
    @classmethod
    def fingerprint_inputs(cls,**kwargs):return float('nan')
    @classmethod
    def execute(cls,caption_plan):
        report=commit_caption_export(open_store(caption_plan['store']),caption_plan['plan'])
        text=json.dumps(report,ensure_ascii=False,indent=2)
        return io.NodeOutput(text,ui={'text':[text]})


class WDCaptionReadSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CaptionReadSettings',display_name='WDV3 Caption Read Settings',category='WDV3/Captions',
            description='Переносимые правила чтения TXT для Load Captions и pipeline-пресета. Запятые/переводы строк либо явный непробельный literal separator; скобки декодируются только по переключателю.',
            inputs=[io.Combo.Input('input_format',options=['comma_newline','literal_separator'],default='comma_newline'),
                    string_input('input_separator',default=', '),io.Boolean.Input('escaped_parentheses',default=False)],outputs=[READ.Output('caption_read')])
    @classmethod
    def execute(cls,input_format='comma_newline',input_separator=', ',escaped_parentheses=False):
        from .Scripts.caption_files import parse_caption
        value=dict(input_format=input_format,input_separator=input_separator,escaped_parentheses=escaped_parentheses)
        parse_caption('',**value)
        return io.NodeOutput(value)


class WDCaptionEditSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CaptionEditSettings',display_name='WDV3 Caption Edit Settings',category='WDV3/Captions',
            description='Переносимая упорядоченная цепочка редактирования TXT. Подключите к Edit Captions либо Save Pipeline Config. Проверяет операции без чтения файлов и моделей.',
            inputs=[string_input('operations_json',default='[]',multiline=True)],outputs=[EDIT.Output('caption_edit')])
    @classmethod
    def execute(cls,operations_json='[]'):
        value=json.loads(operations_json)
        edit_caption(document_from_tags([]),value)
        return io.NodeOutput(value)


NODE_CLASSES=[WDCalculationStore,WDInferenceSettings,WDDatasetFiles,WDPlanCalculations,WDComputeMissing,WDSaveEvidence,
              WDLoadEvidence,WDImportCalculations,WDLoadCaptions,WDEditCaptions,WDCombineCaptions,
              WDCaptionExportSettings,WDPlanCaptionExport,WDSaveCaptions,WDCaptionReadSettings,WDCaptionEditSettings]
