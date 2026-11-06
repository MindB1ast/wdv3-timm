"""Explicit raw evidence, filtering, merging, inspection and report stages."""
from contextlib import ExitStack
import json
from pathlib import Path
import folder_paths
from comfy_api.latest import io
from .comfy_input_help import string_input

from .comfy_ui_nodes import MODEL, OPTIONS, DETECTORS, SAM_MODEL, SAM_SOURCES, RULES, progress
from .comfy_processing import tensor_frame
from .comfy_evidence import collect, apply_filters, outputs, merge_results, cached_segment
from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle

EVIDENCE=io.Custom('WDV3_EVIDENCE')
RESULTS=io.Custom('WDV3_RESULTS')
REGIONS=io.Custom('WDV3_REGIONS')
REPORT=io.Custom('WDV3_REPORT')
TAG_SETS=io.Custom('WDV3_TAG_SETS')


class WDCollectEvidence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CollectEvidence',display_name='WDV3 Collect Evidence',category='WDV3/Pipeline',
            description="Распознаёт и сохраняет исходные оценки WD до пользовательских порогов отдельно для полного кадра и YOLO/SAM-регионов. Evidence позволяет менять фильтры и постобработку без нового инференса при прежней геометрии и моделях. Выход regions содержит геометрию для Inspect.",
            inputs=[io.Image.Input('images',
                        tooltip='Исходное изображение или батч IMAGE, например из Load Image. Для SAM Segment и тегирования подключайте одни и те же изображения в одинаковом порядке, чтобы маски соответствовали картинкам.'),MODEL.Input('wd_model',
                        tooltip='Модель WD из Load Tagger Model. Она распознаёт теги на полном изображении и вырезанных регионах; YOLO/SAM только находят эти регионы.'),OPTIONS.Input('options',optional=True,
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'),
                    DETECTORS.Input('detectors',optional=True,
                        tooltip='Источники YOLO из YOLO Detector / Combine YOLO Detectors / Load Pipeline Config. Их прямоугольные вырезки тегируются WD вместе с полным изображением; фон внутри вырезок сохраняется.'),SAM_MODEL.Input('sam_model',optional=True,
                        tooltip='Настройки SAM из Load SAM 3.1 или Load Pipeline Config. Подключайте вместе с запросами SAM. SAM ищет объекты по тексту; для получения тегов также нужна модель WD.'),
                    SAM_SOURCES.Input('sources', display_name='sam_queries',optional=True,
                        tooltip='Текстовые запросы SAM и настройки регионов из SAM Text / Region Settings, Combine SAM Queries или Load Pipeline Config. Подключайте вместе с sam_model. Каждый источник сохраняет собственные пороги, режим вырезки и правила.')],
            outputs=[EVIDENCE.Output('evidence',
                        tooltip='Все исходные оценки WD для полного изображения и регионов, до применения порогов. Подключите к Apply Filters для настройки правил без повторного инференса или к evidence у Inspect для анализа.'),REGIONS.Output('regions',
                        tooltip='Геометрия областей YOLO/SAM и данные исходных изображений, без оценок WD. Подключите к regions у Inspect для просмотра найденных областей.')])
    @classmethod
    def execute(cls,images,wd_model,sam_batch_size=1,options=None,detectors=None,sam_model=None,sources=None):
        sources=sources or None
        if (sam_model is None)!=(sources is None): raise ValueError('SAM needs both sam_model and sources')
        from .comfy_config import unique_names
        unique_names([c.name for c in (detectors[0] if detectors else [])]+[s['name'] for s in (sources or [])])
        with ExitStack() as stack:
            frames=[stack.enter_context(tensor_frame(t)) for t in images]
            from .comfy_automatic_cache import cached_evidence
            evidence=cached_evidence(frames,wd_model,options=options,detectors=detectors,sam_model=sam_model,sources=sources,progress=progress)
            if evidence is None:
                regions=cached_segment(frames,wd_model,sam_model,sources,(options or {}).get('sam_batch_size',sam_batch_size),progress,low_memory_mode=(options or {}).get('low_memory_mode',True)) if sources else None
                evidence=collect(frames,wd_model,options=options,detectors=detectors,sam_regions=regions,progress=progress)
        geometry=evidence.data()
        for frame in geometry['frames']:
            frame.pop('full_image',None)
            for rs in frame['detectors'].values():
                for r in rs:
                    for category in ['general','character','ratings']:r.pop(category,None)
        return io.NodeOutput(evidence,Bundle.pack('regions',geometry))


class WDApplyFilters(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ApplyFilters',display_name='WDV3 Apply Filters',category='WDV3/Pipeline',
            description="Применяет к evidence пороги и правила источников, затем сразу объединяет теги в общий caption. Выдаёт results, tags и results_json. Для вставки отдельных операций перед объединением используйте Filter Tag Sets и Merge Tag Sets.",
            inputs=[EVIDENCE.Input('evidence',
                        tooltip='Сырые оценки WD до порогов из Collect Evidence. В Apply Filters можно менять правила без повторного распознавания. В Inspect этот вход показывает исходные оценки; подключите только один вход стадии.'),OPTIONS.Input('options',optional=True,
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'),
                    DETECTORS.Input('detectors',optional=True,
                        tooltip='Новые правила YOLO для уже собранных evidence. При подключении detectors или запросы SAM передайте полный набор источников, совпадающий с Collect Evidence по именам. Параметры детекции менять нельзя — для этого заново соберите evidence.'),SAM_SOURCES.Input('sources', display_name='sam_queries',optional=True,
                        tooltip='Новые правила SAM для уже найденных регионов. При подключении запросы SAM или detectors передайте полный набор исходных источников с теми же именами. Изменение prompt, масок или вырезок требует повторного Collect Evidence.')],
            outputs=[RESULTS.Output('results',
                        tooltip='Отфильтрованные результаты с решениями по тегам и их происхождением. Подключите к results у Inspect или к left/right у Merge Results.'),io.String.Output('tags',is_output_list=True,
                        tooltip='Готовый текст тегов: отдельная строка на каждое изображение, с выбранным форматированием. Можно передать в текстовый промпт или tags у Save Result вместе с results_json из той же ветки.'),io.String.Output('results_json',is_output_list=True,
                        tooltip='Подробный JSON для каждого изображения: теги полного кадра и регионов, оценки и происхождение итоговых тегов. Подключите к Results Viewer или Save Result. Это содержимое JSON, а не путь к файлу.')])
    @classmethod
    def execute(cls,evidence,options=None,detectors=None,sources=None):
        configs=None
        if detectors is not None or sources is not None:
            configs=list(detectors[0]) if detectors else []
            configs += [DetectorConfig(s['name'],'sam3',**s.get('rules',{})) for s in (sources or [])]
        results=apply_filters(evidence,options,configs);tags,details=outputs(results)
        return io.NodeOutput(results,tags,details)


class WDMergeResults(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3MergeResults',display_name='WDV3 Merge Results',category='WDV3/Pipeline',
            description="Объединяет результаты нескольких ветвей одного набора изображений, сохраняя исходные оценки, регионы и происхождение тегов. Общие настройки и цепочки постобработки должны быть совместимы. Одинаковые теги общего caption получают максимальную допустимую оценку.",
            inputs=[RESULTS.Input('left',
                        tooltip='Первая ветка результатов из Apply Filters или Merge Results. Обе ветки должны относиться к тем же изображениям и совместимым настройкам текста. Теги объединяются с сохранением происхождения.'),RESULTS.Input('right',
                        tooltip='Вторая ветка результатов для тех же изображений. При совпадении тегов выбирается наибольшая оценка; данные регионов и происхождение сохраняются.')],
            outputs=[RESULTS.Output('results',
                        tooltip='Отфильтрованные результаты с решениями по тегам и их происхождением. Подключите к results у Inspect или к left/right у Merge Results.'),io.String.Output('tags',is_output_list=True,
                        tooltip='Готовый текст тегов: отдельная строка на каждое изображение, с выбранным форматированием. Можно передать в текстовый промпт или tags у Save Result вместе с results_json из той же ветки.'),io.String.Output('results_json',is_output_list=True,
                        tooltip='Подробный JSON для каждого изображения: теги полного кадра и регионов, оценки и происхождение итоговых тегов. Подключите к Results Viewer или Save Result. Это содержимое JSON, а не путь к файлу.')])
    @classmethod
    def execute(cls,left,right):
        result=merge_results(left,right);tags,details=outputs(result)
        return io.NodeOutput(result,tags,details)


class WDInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3Inspect',display_name='WDV3 Inspect',category='WDV3/Pipeline',is_output_node=True,
            description="Показывает одну выбранную стадию: regions, исходные evidence, локальные tag_sets или итоговые results. Доступны изображения, области, оценки, причины решений и история before/after. Подключите ровно один вход стадии; report можно сохранить через Export Report.",
            inputs=[RESULTS.Input('results',optional=True,
                        tooltip='Отфильтрованные результаты из Apply Filters или Merge Results. В Inspect показывает принятые/отклонённые теги и их источники; подключите только один вход стадии.'),EVIDENCE.Input('evidence',optional=True,
                        tooltip='Сырые оценки WD до порогов из Collect Evidence. В Apply Filters можно менять правила без повторного распознавания. В Inspect этот вход показывает исходные оценки; подключите только один вход стадии.'),REGIONS.Input('regions',optional=True,
                        tooltip='Геометрия найденных областей из Collect Evidence, без оценок тегов. В Inspect подключите только этот вход, чтобы изучить расположение регионов.'),
                    TAG_SETS.Input('tag_sets', optional=True, tooltip='Локальные принятые теги до merge: before/after и причины удаления избыточности. Подключите только один вход стадии.')],
            outputs=[REPORT.Output('report',
                        tooltip='Отчёт по выбранной стадии с изображениями и подробными данными. Подключите к report у Export Report для сохранения JSON, HTML и, для отфильтрованных результатов, TXT.'),io.Image.Output('images',is_output_list=True,
                        tooltip='Исходные изображения выбранной стадии, отдельный элемент на изображение. Для просмотра рамок и подробностей используйте встроенный просмотрщик этой ноды.'),io.String.Output('tags',is_output_list=True,
                        tooltip='Текст тегов выбранной стадии, по одному элементу на изображение. Итоговые теги доступны при подключении results; в evidence показаны сырые теги до фильтрации, а regions не содержит тегов.')])
    @classmethod
    def execute(cls,results=None,evidence=None,regions=None,tag_sets=None):
        from .comfy_reports import make_report,preview_report
        values=[v for v in (results,evidence,regions,tag_sets) if v is not None]
        if len(values)!=1: raise ValueError('Connect exactly one Inspect stage: results, evidence, regions or tag_sets')
        report=make_report(values[0]);rows,images,tags=preview_report(report,folder_paths.get_temp_directory())
        return io.NodeOutput(report,images,tags,ui={'wdv3_results':rows,'wdv3_total':[len(rows)]})


class WDExportReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ExportReport',display_name='WDV3 Export Report',category='WDV3/Output',
            description="Сохраняет report из Inspect как автономный HTML с изображениями, подробный JSON и/или TXT. TXT общего caption доступен для итоговой стадии results; для локальных tag_sets, evidence и regions выбирайте HTML/JSON. Каждый экспорт получает отдельную папку.",
            is_output_node=True,not_idempotent=True,
            inputs=[REPORT.Input('report',
                        tooltip='Отчёт из Inspect с изображениями и данными выбранной стадии. Для сохранения итогового TXT создавайте его из results; evidence подходит для анализа исходных оценок.'),string_input('output_directory',default=str(Path(folder_paths.get_output_directory())/'wdv3'),
                        tooltip='Папка сохранения на машине, где работает ComfyUI; при необходимости создаётся автоматически. Это папка, а не имя файла.'),
                    string_input('prefix',default='report',
                        tooltip='Имя отчёта: буквы, цифры, пробелы, _ и -. Для каждого экспорта создаётся отдельная папка wdv3-<имя>-<уникальный ID> внутри output_directory.'),io.Boolean.Input('save_json',default=True,
                        tooltip='Сохранить подробные данные в JSON: оценки, регионы и сведения о результате. Формат результата и формат отчёта Inspect различаются.'),
                    io.Boolean.Input('save_html',default=True,
                        tooltip='Сохранить интерактивный HTML-отчёт с картинками, регионами, поиском и оценками. Открывайте report.html в браузере; при переносе сохраняйте рядом папку assets.'),io.Boolean.Input('save_txt',default=True,
                        tooltip='Сохранить итоговые теги в текстовый файл TXT. Для Export Report нужен отчёт по results из Apply Filters или Merge Results; сырые evidence и геометрия regions не содержат итоговых тегов.')],
            outputs=[io.String.Output('json_path',
                        tooltip='Полный путь к сохранённому JSON. Пустая строка, если сохранение JSON выключено.'),io.String.Output('html_path',
                        tooltip='Полный путь к HTML-отчёту. Откройте в браузере; для переноса сохраните рядом assets. Пустая строка, если HTML выключен.'),io.String.Output('txt_path',
                        tooltip='Путь к tags.txt: одна строка итоговых тегов на изображение. Для этого формата Inspect должен получать results. Пустая строка, если TXT выключен.')])
    @classmethod
    def execute(cls,report,output_directory,prefix,save_json,save_html,save_txt):
        from .comfy_reports import export_report
        paths=export_report(report,output_directory,prefix,save_json,save_html,save_txt)
        return io.NodeOutput(paths.get('json',''),paths.get('html',''),paths.get('txt',''),ui={'text':list(paths.values())})


NODE_CLASSES=[WDCollectEvidence,WDApplyFilters,WDMergeResults,WDInspect,WDExportReport]

from .comfy_postprocessing_nodes import NODE_CLASSES as POSTPROCESSING_CLASSES
NODE_CLASSES += POSTPROCESSING_CLASSES
