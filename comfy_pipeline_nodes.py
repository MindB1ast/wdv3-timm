"""Explicit raw evidence, filtering, merging, inspection and report stages."""
from contextlib import ExitStack
import json
from pathlib import Path
import folder_paths
from comfy_api.latest import io

from .comfy_ui_nodes import MODEL, OPTIONS, DETECTORS, SAM_MODEL, SAM_SOURCES, RULES, progress
from .comfy_processing import tensor_frame
from .comfy_evidence import collect, apply_filters, outputs, merge_results, cached_segment
from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle

EVIDENCE=io.Custom('WDV3_EVIDENCE')
RESULTS=io.Custom('WDV3_RESULTS')
REGIONS=io.Custom('WDV3_REGIONS')
REPORT=io.Custom('WDV3_REPORT')


class WDCollectEvidence(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3CollectEvidence',display_name='WDV3 Collect Evidence',category='WDV3/Pipeline',
            description='Collect all WD scores before thresholds. Reuses cached inference when only rules change.',
            inputs=[io.Image.Input('images',
                        tooltip='Исходное изображение или батч IMAGE, например из Load Image. Для SAM Segment и тегирования подключайте одни и те же изображения в одинаковом порядке, чтобы маски соответствовали картинкам.'),MODEL.Input('wd_model',
                        tooltip='Модель WD из Load Tagger Model. Она распознаёт теги на полном изображении и вырезанных регионах; YOLO/SAM только находят эти регионы.'),OPTIONS.Input('options',optional=True,
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'),
                    DETECTORS.Input('detectors',optional=True,
                        tooltip='Источники YOLO из YOLO Detector / Combine YOLO Detectors / Load Pipeline Config. Их прямоугольные вырезки тегируются WD вместе с полным изображением; фон внутри вырезок сохраняется.'),SAM_MODEL.Input('sam_model',optional=True,
                        tooltip='Настройки SAM из Load SAM 3.1 или Load Pipeline Config. Подключайте вместе с sources. SAM ищет объекты по тексту; для получения тегов также нужна модель WD.'),
                    SAM_SOURCES.Input('sources',optional=True,
                        tooltip='Текстовые запросы SAM и настройки регионов из SAM Text / Region Settings, Combine SAM Sources или Load Pipeline Config. Подключайте вместе с sam_model. Каждый источник сохраняет собственные пороги, режим вырезки и правила.'),io.Int.Input('sam_batch_size',default=1,min=1,max=16,
                        tooltip='Сколько исходных изображений SAM сегментирует одновременно. Начните с 1; увеличение требует больше памяти. Это не число объектов на изображении и не батч WD.')],
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
            regions=cached_segment(frames,wd_model,sam_model,sources,(options or {}).get('sam_batch_size',sam_batch_size),progress) if sources else None
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
            inputs=[EVIDENCE.Input('evidence',
                        tooltip='Сырые оценки WD до порогов из Collect Evidence. В Apply Filters можно менять правила без повторного распознавания. В Inspect этот вход показывает исходные оценки; подключите только один вход стадии.'),OPTIONS.Input('options',optional=True,
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'),
                    DETECTORS.Input('detectors',optional=True,
                        tooltip='Новые правила YOLO для уже собранных evidence. При подключении detectors или sources передайте полный набор источников, совпадающий с Collect Evidence по именам. Параметры детекции менять нельзя — для этого заново соберите evidence.'),SAM_SOURCES.Input('sources',optional=True,
                        tooltip='Новые правила SAM для уже найденных регионов. При подключении sources или detectors передайте полный набор исходных источников с теми же именами. Изменение prompt, масок или вырезок требует повторного Collect Evidence.')],
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
            description='Connect exactly one stage. Inspect raw scores, filtered decisions, regions and source attribution.',
            inputs=[RESULTS.Input('results',optional=True,
                        tooltip='Отфильтрованные результаты из Apply Filters или Merge Results. В Inspect показывает принятые/отклонённые теги и их источники; подключите только один вход стадии.'),EVIDENCE.Input('evidence',optional=True,
                        tooltip='Сырые оценки WD до порогов из Collect Evidence. В Apply Filters можно менять правила без повторного распознавания. В Inspect этот вход показывает исходные оценки; подключите только один вход стадии.'),REGIONS.Input('regions',optional=True,
                        tooltip='Геометрия найденных областей из Collect Evidence, без оценок тегов. В Inspect подключите только этот вход, чтобы изучить расположение регионов.')],
            outputs=[REPORT.Output('report',
                        tooltip='Отчёт по выбранной стадии с изображениями и подробными данными. Подключите к report у Export Report для сохранения JSON, HTML и, для отфильтрованных результатов, TXT.'),io.Image.Output('images',is_output_list=True,
                        tooltip='Исходные изображения выбранной стадии, отдельный элемент на изображение. Для просмотра рамок и подробностей используйте встроенный просмотрщик этой ноды.'),io.String.Output('tags',is_output_list=True,
                        tooltip='Текст тегов выбранной стадии, по одному элементу на изображение. Итоговые теги доступны при подключении results; в evidence показаны сырые теги до фильтрации, а regions не содержит тегов.')])
    @classmethod
    def execute(cls,results=None,evidence=None,regions=None):
        from .comfy_reports import make_report,preview_report
        values=[v for v in (results,evidence,regions) if v is not None]
        if len(values)!=1: raise ValueError('Connect exactly one Inspect stage: results, evidence or regions')
        report=make_report(values[0]);rows,images,tags=preview_report(report,folder_paths.get_temp_directory())
        return io.NodeOutput(report,images,tags,ui={'wdv3_results':rows,'wdv3_total':[len(rows)]})


class WDExportReport(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='WDV3ExportReport',display_name='WDV3 Export Report',category='WDV3/Output',
            is_output_node=True,not_idempotent=True,
            inputs=[REPORT.Input('report',
                        tooltip='Отчёт из Inspect с изображениями и данными выбранной стадии. Для сохранения итогового TXT создавайте его из results; evidence подходит для анализа исходных оценок.'),io.String.Input('output_directory',default=str(Path(folder_paths.get_output_directory())/'wdv3'),
                        tooltip='Папка сохранения на машине, где работает ComfyUI; при необходимости создаётся автоматически. Это папка, а не имя файла.'),
                    io.String.Input('prefix',default='report',
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
