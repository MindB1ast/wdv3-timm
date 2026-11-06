"""ComfyUI V3 nodes for editable policies, SAM, folders and result inspection."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io
from .comfy_input_help import string_input
import folder_paths
from comfy.utils import ProgressBar

from .comfy_nodes import ROOT, model_directory
from .comfy_processing import (file_frame, scan_folder,
                               segment_frames, tag_folder, tensor_frame)
from .Scripts.tag_utils import parse_tags, parse_additions, parse_regex, REGEX_RULE_FIELDS
from .Scripts.evidence import Bundle
from .Scripts.preprocess import GLOBAL_EXCLUDED_REGION_TAGS

RULES = io.Custom("WDV3_RULES")
OPTIONS = io.Custom("WDV3_OPTIONS")
DETECTORS = io.Custom("WDV3_DETECTORS")
SAM_MODEL = io.Custom("WDV3_SAM_MODEL")
SAM_SOURCES = io.Custom("WDV3_SAM_SOURCES")
SAM_REGIONS = io.Custom("WDV3_SAM_REGIONS")
FILES = io.Custom("WDV3_FILES")
MODEL = io.Custom("WDV3_MODEL")

LITERAL_HELP = (' Буквальные имена: blue eyes = blue_eyes. Список через запятые или переводы строк; '
                'пробел внутри имени не разделяет теги. Regex-метасимволы здесь буквальные.')
REGEX_HELP = ('Regex: один Python-шаблон на строку, полное совпадение с именем с _, '
              'с учётом регистра. Например .*_eyes. Пробелы и запятые сохраняются. '
              'Обычные теги вводите в отдельное буквальное поле.')


def literal_input(name, **kwargs):
    kwargs['tooltip'] = kwargs.get('tooltip', '') + LITERAL_HELP
    return string_input(name, **kwargs)


def regex_inputs(fields):
    return [string_input(field + '_regex', default='', multiline=True,
                            tooltip=field + ': ' + REGEX_HELP) for field in fields]


def progress(done, total):
    import comfy.model_management as management
    management.throw_exception_if_processing_interrupted()
    ProgressBar(total).update_absolute(done, total)


class WDTaggingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3Options", display_name="WDV3 Tagging Settings", category="WDV3/Settings",
            description="Задаёт общие параметры тегирования: пороги WD, удаляемые теги, prefix/suffix, формат текста и размеры батчей. Подключите options к нодам обработки и фильтрации. Настройки можно сохранить через Save Pipeline Config.",
            inputs=[io.Float.Input("general_threshold", default=.35, min=0, max=1, step=.01,
                        tooltip='Порог уверенности WD для обычных тегов: проходят оценки строго выше этого значения. Ниже — больше тегов и ошибок; выше — меньше, но увереннее. Для регионов можно задать отдельный порог в Region Tag Rules.'),
                    io.Float.Input("character_threshold", default=.75, min=0, max=1, step=.01,
                        tooltip='Порог WD для имён персонажей, а не для обнаружения людей. Ниже — больше предполагаемых персонажей; выше — меньше ложных имён. Для регионов используйте Region Tag Rules.'),
                    io.Int.Input("image_batch_size", default=2, min=1, max=64,
                        tooltip='Сколько полных изображений WD обрабатывает одновременно. Больше — обычно быстрее, но требует больше памяти; при нехватке памяти уменьшите до 1. Размеры батчей SAM и регионов задаются отдельно.'),
                    io.Int.Input("region_batch_size", default=8, min=1, max=128,
                        tooltip='Сколько вырезанных областей YOLO/SAM WD тегирует одновременно. При нехватке памяти уменьшите. Не ограничивает число найденных объектов и не задаёт размер батча SAM.'),
                    io.Boolean.Input("replace_underscores", default=True,
                        tooltip='Заменять подчёркивания пробелами в итоговом тексте: blue_eyes → blue eyes. Имена тегов в подробных данных остаются каноническими.'),
                    io.Boolean.Input("escape_parentheses", default=True,
                        tooltip='Экранировать скобки в итоговом тексте для использования в промпте: hero_(series) → hero_\\(series\\). Отключите, если текст нужен без экранирования.'),
                    literal_input("remove_tags", default="", multiline=True,
                        tooltip='Теги, которые нужно удалить из итогового результата независимо от источника. Разделяйте запятыми или переносами строк, например blurry, watermark. Удаление имеет приоритет над ручным добавлением.'),
                    literal_input("prefix_tags", default="",
                        tooltip='Вручную добавить теги в начало итогового текста, через запятую. Они не требуют распознавания моделью; глобальное удаление и ограничения источников продолжают действовать.'), literal_input("suffix_tags", default="",
                        tooltip='Вручную добавить теги в конец итогового текста, через запятую. Они не требуют распознавания моделью; глобальное удаление и ограничения источников продолжают действовать.'),
                    *regex_inputs(['remove_tags']),
                    string_input('tag_separator', default=', ',
                        tooltip='Точная строка между тегами: , или пробел или |. Пробелы сохраняются. Пусто — без разделителя. На ввод правил не влияет.'),
                    io.Int.Input('sam_batch_size', default=1, min=1, max=16,
                        tooltip='Сколько исходных изображений SAM сегментирует одновременно. Это не число областей или источников. Сохраняется в options pipeline-конфига.'),
                    io.Combo.Input('cache_mode',options=['reuse','reuse_only','refresh','off'],default='reuse',optional=True,display_name='calculation_mode',
                        tooltip='reuse = automatic: reuse saved calculations, compute only missing stages. reuse_only = saved calculations only, never run inference. refresh = recalculate. off = do not use persistent calculations. Saving TXT/JSON is controlled separately in Tag Folder.'),
                    string_input('cache_dataset_root',default='',optional=True,display_name='dataset_folder_override',
                        tooltip='Tag Folder: leave empty; calculations are stored automatically beside images in Image Folder/.wdv3. Override only to reuse another dataset folder. Accepts the dataset folder or its .wdv3 folder. This is independent of the TXT/JSON output directory. For IMAGE input, choose a dataset folder once.'),
                    string_input('cache_refresh',default='',optional=True,display_name='refresh_selection',
                        tooltip='Только для refresh: имена детекторов или стадии через запятую. Пусто — всё. Для изменения фильтров используйте reuse.'),
                    io.Boolean.Input('low_memory_mode',default=True,optional=True,
                        tooltip='On: release WD weights from RAM before SAM; reload WD only when inference is needed. Uses less peak RAM but can be slower. Off: keep WD weights resident to avoid reloading; requires enough RAM for WD and SAM together. SAM remains isolated in a worker in both modes. Cached results are unchanged.')],
            outputs=[OPTIONS.Output("options",
                        tooltip='Глобальные настройки тегирования. Подключите к options у Tag Image / Regions, Tag Folder, Collect Evidence или Apply Filters; также можно сохранить через Save Pipeline Config.')])

    @classmethod
    def execute(cls, **kwargs):
        from .comfy_config import validate_config
        if 'remove_tags_regex' in kwargs:
            kwargs['remove_tags_regex'] = parse_regex(kwargs['remove_tags_regex'])
        validate_config({'schema_version':1,'options':kwargs})
        return io.NodeOutput(Bundle.pack('rules',kwargs))


class WDRegionRules(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3RegionRules", display_name="WDV3 Region Tag Rules", category="WDV3/Settings",
            description="Задаёт правила тегов YOLO/SAM-источника и необязательный дочерний стек детекторов. Подключите детей к child_detectors либо child_sam_model + child_sources, затем rules — к родительскому YOLO Detector или SAM Text / Region Settings. Дети ищут внутри исходного bbox родителя. Буквальные теги и regex задаются раздельно.",
            inputs=[io.Float.Input("general_threshold", default=.35, min=0, max=1, step=.01,
                        tooltip='Порог WD для обычных тегов только в регионах этого источника. Проходят оценки строго выше порога; меньшее значение добавляет больше тегов и ошибок. Порог полного изображения задаётся в Tagging Settings.'),
                    io.Float.Input("character_threshold", default=.75, min=0, max=1, step=.01,
                        tooltip='Порог WD для имён персонажей только в регионах этого источника. Это не уверенность детектора и не класс person. Выше — меньше ложных имён.'),
                    literal_input("remove_from_full", default="", multiline=True,
                        tooltip='Убрать эти теги из вклада полного изображения, если этот источник нашёл хотя бы один регион. Например, перенести определение 1girl на регионы. Другие источники всё ещё могут добавить эти теги.'),
                    literal_input("remove_from_regions", default="", multiline=True,
                        tooltip='Не принимать эти теги из регионов данного источника. Запятые или переносы строк. Полное изображение и другие источники сохраняют свои теги; удаление побеждает add_to_regions.'),
                    literal_input("add_to_regions", default="", multiline=True, tooltip='Добавить теги в каждый найденный регион этого источника. Формат: person или person:0.9, через запятую/перенос строки. Без оценки назначается 1.0; это ручная оценка, не уверенность WD. Если регионов нет, ничего не добавится. Ограничения и удаление продолжают действовать.'),
                    literal_input("require_full_confirmation", default="", multiline=True,
                        tooltip='Принимать перечисленные теги из регионов только если они также прошли порог на полном изображении. Например, full_body не попадёт в итог лишь из-за маленького кропа. Запятые или переносы строк.'),
                    literal_input("only_from_source", default="", multiline=True,
                        tooltip='Закрепить эти теги за данным источником: полное изображение и источники, не объявившие этот тег своим, не смогут добавить его в итог. Если источник не нашёл/не подтвердил тег, запасного варианта с полного изображения нет.'),
                    literal_input("allowed_tags", default="", multiline=True,
                        tooltip='Разрешить из этого источника только перечисленные теги, включая ручные добавления. Пусто — без такого ограничения. Например, для лица оставьте blue_eyes, smile. Запятые или переносы строк.'),
                    literal_input('full_image_only_tags', default=', '.join(sorted(GLOBAL_EXCLUDED_REGION_TAGS)),multiline=True,
                        tooltip='Эти теги нельзя добавлять из регионов данного источника: они описывают фон и композицию всей картинки. Они могут прийти с полного изображения. Очистите поле, чтобы разрешить их из этого источника.'),
                    *regex_inputs(REGEX_RULE_FIELDS),
                    DETECTORS.Input('child_detectors', optional=True, tooltip='Дочерний стек обычных YOLO Detector / Combine YOLO Detectors. Ищет внутри каждого исходного bbox этого источника, без маски или заливки родителя.'),
                    SAM_MODEL.Input('child_sam_model', optional=True, tooltip='Модель дочернего SAM из Load SAM 3.1; подключите вместе с child_sources.'),
                    SAM_SOURCES.Input('child_sources', optional=True, display_name='child_sam_queries', tooltip='Дочерние запросы SAM с собственными обычными настройками и rules. Для следующих уровней подключите детей в их Region Tag Rules.'),
                    io.Combo.Input('child_tag_route', options=['separate', 'merge_to_parent'], default='separate', optional=True, tooltip='Оставить дочерние области отдельными либо добавить их теги непосредственному родителю.'),
                    io.Boolean.Input('tag_enabled', default=True, optional=True, tooltip='Тегировать этот источник через WD. Выключите для промежуточного детектора, который только находит области для детей.'),
                    io.Custom('WDV3_POSTPROCESSING').Input('local_operations', optional=True),
                    io.Custom('WDV3_POSTPROCESSING').Input('aggregate_operations', optional=True)],
            outputs=[RULES.Output("rules",
                        tooltip='Правила и необязательный дочерний стек одного источника. Подключите к rules родительского YOLO Detector или SAM Text / Region Settings.')])

    @classmethod
    def execute(cls, general_threshold, character_threshold, remove_from_full,
                remove_from_regions, add_to_regions, require_full_confirmation,
                only_from_source="", allowed_tags="", full_image_only_tags=None,
                child_detectors=None, child_sam_model=None, child_sources=None,
                child_tag_route='separate', tag_enabled=True,
                local_operations=None, aggregate_operations=None, **regex_rules):
        rules={"region_gen_threshold": general_threshold, "region_char_threshold": character_threshold,
            "remove_tags_from_full": parse_tags(remove_from_full),
            "remove_tags_from_region": parse_tags(remove_from_regions),
            "add_tags_to_region": parse_additions(add_to_regions),
            "exclude_from_region": parse_tags(require_full_confirmation),
            "only_from_source": parse_tags(only_from_source), "allowed_tags": parse_tags(allowed_tags),
            "full_image_only_tags": None if full_image_only_tags is None else parse_tags(full_image_only_tags)}
        from .comfy_config import validate_rules
        unknown = set(regex_rules) - {field + '_regex' for field in REGEX_RULE_FIELDS}
        if unknown:
            raise ValueError('Unknown regex rules: ' + ', '.join(sorted(unknown)))
        rules.update({field + '_regex': parse_regex(regex_rules.get(field + '_regex', []))
                      for field in REGEX_RULE_FIELDS})
        from .comfy_nested import child_settings
        from .Scripts.region_rule_defaults import child_route_default, tag_enabled_default
        child_tag_route = child_route_default(child_tag_route)
        tag_enabled = tag_enabled_default(tag_enabled)
        children = child_settings(child_detectors, child_sam_model, child_sources,
            tag_route=child_tag_route, tag_enabled=tag_enabled,
            local_operations=local_operations, aggregate_operations=aggregate_operations)
        if children is not None:
            rules['children'] = children
        validate_rules(rules)
        return io.NodeOutput(Bundle.pack('rules',rules))


class WDYOLODetector(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, model, custom_model_path='', **kwargs):
        from .comfy_evidence import digest, file_stamp
        candidates = [Path(custom_model_path)] if custom_model_path.strip() else [
            directory / model for directory in (model_directory('yolo'), ROOT/'models'/'yolo',
                Path(folder_paths.models_dir)/'ultralytics'/'bbox',Path(folder_paths.models_dir)/'ultralytics'/'segm')]
        return digest([file_stamp(path) for path in candidates])

    @classmethod
    def define_schema(cls):
        known = ["person_yolov8m-seg.pt", "face_yolov9c.pt", "hand_yolov9c.pt", "yolov8n.pt"]
        for directory in (model_directory("yolo"), ROOT / "models" / "yolo",
                          Path(folder_paths.models_dir) / "ultralytics" / "bbox",
                          Path(folder_paths.models_dir) / "ultralytics" / "segm"):
            if directory.is_dir():
                known.extend(p.name for p in directory.glob("*.pt"))
        return io.Schema(node_id="WDV3YOLODetector", display_name="WDV3 YOLO Detector", category="WDV3/Regions",
            description="Настраивает один именованный YOLO-источник: модель, классы, confidence и правила тегов регионов. Поиск областей выполняется при тегировании или Collect Evidence; сами теги распознаёт WD. Несколько источников можно связать через previous_detectors или Combine YOLO Detectors.",
            inputs=[io.Combo.Input("model", options=list(dict.fromkeys(known)),
                        tooltip='Веса YOLO для поиска объектов. person/face/hand — специализированные модели; yolov8n — общие классы COCO. Используются прямоугольные вырезки bbox, даже для весов с суффиксом -seg. custom_model_path заменяет выбор.'),
                    string_input("source_name", default="person", display_name="detector_name",
                        tooltip='Уникальное имя источника регионов в общей схеме, например person_detail или face. По нему показываются регионы и происхождение тегов в отчётах. Имена YOLO и SAM не должны совпадать.'),
                    io.Float.Input("confidence", default=.35, min=0, max=1, step=.01,
                        tooltip='Порог уверенности обнаружения объекта (0–1). Выше — меньше сомнительных объектов, ниже — больше находок и ложных срабатываний. Это не порог уверенности тегов WD.'),
                    string_input("classes", default="0", tooltip='Числовые ID классов YOLO через запятую; пусто — все классы. Для COCO-модели 0 = person; у специализированных моделей значения могут отличаться. Это не текстовый запрос SAM.'),
                    string_input("custom_model_path", default="", tooltip='Полный путь к весам YOLO (.pt). Если заполнен, используется вместо model. Оставьте пустым для выбора из списка.'),
                    RULES.Input("rules", optional=True,
                        tooltip='Подключите rules из Region Tag Rules: пороги WD, удаление и добавление тегов применятся к регионам этого источника. Без подключения используются стандартные правила.'), DETECTORS.Input("previous_detectors", optional=True,
                        tooltip='Необязательно: detectors от другой ноды YOLO. Текущий источник будет добавлен к списку. Для отдельных веток используйте Combine YOLO Detectors; имена источников должны быть уникальны.'),
                    io.Combo.Input('device', options=['auto','cpu','cuda'], default='auto', optional=True,
                        tooltip='Где выполнять вычисления: auto — устройство, выбранное окружением; cpu — процессор; cuda — видеокарта NVIDIA с доступной CUDA. При нехватке видеопамяти выберите cpu или уменьшите батч.'),
                    io.Boolean.Input('download_missing', default=True, optional=True,
                        tooltip='Разрешить скачивание недостающих файлов модели. Выключите для работы только с локальными файлами: если их нет или комплект неполон, выполнение завершится ошибкой.')],
            outputs=[DETECTORS.Output("detectors",
                        tooltip='Список источников YOLO с индивидуальными настройками. Подключите к detectors у ноды тегирования или Collect Evidence; объединить ветки можно через Combine YOLO Detectors.')])

    @classmethod
    def execute(cls, model, source_name, confidence, classes, custom_model_path, rules=None, previous_detectors=None,
                device='auto', download_missing=True):
        from .Scripts.configs import DetectorConfig
        choices = cls.define_schema().inputs[0].options
        if model not in choices:
            raise ValueError("Unknown YOLO selection")
        name = source_name.strip()
        if not name:
            raise ValueError("source_name cannot be empty")
        ids = [int(text.strip()) for text in classes.split(",") if text.strip()]
        if any(i < 0 for i in ids):
            raise ValueError("Class IDs must be nonnegative")
        path = custom_model_path.strip() or model
        if not custom_model_path.strip():
            for directory in (model_directory("yolo"), ROOT / "models" / "yolo",
                              Path(folder_paths.models_dir) / "ultralytics" / "bbox",
                              Path(folder_paths.models_dir) / "ultralytics" / "segm"):
                if (directory / model).is_file():
                    path = str(directory / model)
                    break
        if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable; select cpu or auto')
        configs, models = previous_detectors if previous_detectors is not None else ([], {})
        if name in {cfg.name for cfg in configs}:
            raise ValueError("Detector source names must be unique")
        config = DetectorConfig(name, path, confidence=confidence, classes=ids,
                                device=device, download_missing=download_missing, **(rules or {}))
        # The existing dataclass turns None into class 0; an empty selection explicitly means all.
        config.classes = ids or None
        models = dict(models)
        return io.NodeOutput(([*configs, config], models))


class WDSAMLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMLoader", display_name="WDV3 Load SAM 3.1", category="WDV3/SAM",
            description="Задаёт модель SAM 3.1, устройство, локальный путь и разрешение скачивания. Подключите sam_model вместе с текстовыми запросы SAM к сегментации или тегированию. Веса загружаются при выполнении сегментации; эту конфигурацию можно сохранить в пресете.",
            inputs=[io.Combo.Input("model", options=["unsloth/sam3.1"],
                        tooltip='Модель сегментации SAM 3.1 (unsloth/sam3.1). Находит объекты по тексту из запросы SAM; теги распознаёт отдельная модель WD. Недостающие файлы загружаются при запуске сегментации, если это разрешено.'),
                    io.Combo.Input("device", options=["auto", "cpu", "cuda"],
                        tooltip='Где выполнять вычисления: auto — устройство, выбранное окружением; cpu — процессор; cuda — видеокарта NVIDIA с доступной CUDA. При нехватке видеопамяти выберите cpu или уменьшите батч.'),
                    io.Boolean.Input("download_missing", default=True,
                        tooltip='Разрешить скачивание недостающих файлов модели. Выключите для работы только с локальными файлами: если их нет или комплект неполон, выполнение завершится ошибкой.'),
                    string_input("local_model_path", default="", tooltip='Папка SAM 3.1 с весами и сопутствующими JSON/токенизатором либо путь к model.safetensors в ней. Пусто — стандартная папка models/sam3/unsloth-sam3.1 ComfyUI или локальный models/sam3 проекта, если там есть веса.')],
            outputs=[SAM_MODEL.Output("sam_model",
                        tooltip='Настройки модели SAM. Подключите к sam_model вместе с запросами SAM из SAM Text / Region Settings; отдельная модель WD нужна для распознавания тегов.')])

    @classmethod
    def execute(cls, model, device, download_missing, local_model_path):
        return io.NodeOutput(sam_model_settings(model, device, download_missing, local_model_path))


def sam_model_settings(model, device, download_missing, local_model_path):
    from .Scripts.sam_runtime import resolve_device
    if model != "unsloth/sam3.1":
        raise ValueError("Unsupported SAM model")
    selected = resolve_device(device, torch)
    default = Path(folder_paths.models_dir) / "sam3" / "unsloth-sam3.1"
    if not local_model_path.strip() and (ROOT / "models" / "sam3" / "model.safetensors").is_file():
        default = ROOT / "models" / "sam3"
    return Bundle.pack('model', {"checkpoint": local_model_path.strip() or str(default), "model_id": model,
            "backend": "transformers", "device": selected, "allow_download": download_missing,
            "loader_options": dict(model=model, device=device, download_missing=download_missing,
                                   local_model_path=local_model_path)})


class WDSAMSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMSource", display_name="WDV3 SAM Text / Region Settings", category="WDV3/SAM",
            description="Описывает один именованный текстовый запрос SAM и настройки его регионов: отбор масок, вид вырезки, padding и правила тегов. Подключите запросы SAM вместе с sam_model. Текст prompt используется для поиска объектов; для распознавания тегов регионов служит WD.",
            inputs=[string_input("source_name", default="sam_person", display_name="sam_query_name",
                        tooltip='Уникальное имя источника регионов в общей схеме, например person_detail или face. По нему показываются регионы и происхождение тегов в отчётах. Имена YOLO и SAM не должны совпадать.'), string_input("prompt", default="person",
                        tooltip='Какой объект должен найти SAM: короткое описание, например person, face или cup. Лучше начать с английского названия. Для нескольких независимых запросов создайте отдельные SAM Sources и объедините их; это не список итоговых тегов.'),
                    io.Float.Input("confidence", default=.5, min=0, max=1, step=.01,
                        tooltip='Минимальная уверенность SAM, что найденная область соответствует текстовому prompt. Проходят оценки строго выше порога. Выше — меньше сомнительных объектов. Форму маски задаёт mask_threshold, а отбор тегов WD — Region Tag Rules.'),
                    io.Float.Input("mask_threshold", default=.5, min=0, max=1, step=.01,
                        tooltip='Порог включения пикселя в маску SAM: остаются вероятности строго выше значения. Выше — маска обычно меньше и строже, ниже — захватывает больше краёв и фона. Меняет форму маски и её bbox, а не уверенность тегов WD.'),
                    io.Int.Input("min_mask_area_px", default=32, min=1, max=100000000,
                        tooltip='Минимальная площадь маски в пикселях исходного изображения. Более мелкие маски отбрасываются. Уменьшите для небольших лиц/предметов; увеличьте для удаления мелкого шума. Это площадь маски, а не прямоугольника bbox.'),
                    io.Int.Input("max_regions", default=32, min=1, max=256,
                        tooltip='Максимум регионов этого SAM-источника на одно изображение. После фильтрации и удаления одинаковых масок остаются самые уверенные. Это не размер батча.'),
                    io.Combo.Input("view_mode", options=["bbox", "mask"],
                        tooltip='Какой фрагмент получит WD для тегирования региона. mask — вырезка объекта по маске SAM: внутри минимального прямоугольника вокруг маски пиксели вне объекта заливаются белым, поэтому исходный фон меньше влияет на теги. bbox — минимальный прямоугольник вокруг той же маски целиком: исходный фон и соседние объекты внутри сохраняются и могут влиять на теги. padding_px добавляет контекст; в mask он также расширяет видимую область вокруг маски. Сама найденная SAM-маска от выбора режима не меняется.'),
                    io.Int.Input("padding_px", default=0, min=0, max=4096,
                        tooltip='Отступ в пикселях исходного изображения вокруг bbox маски, с обрезкой по краям картинки. 0 — плотная вырезка. В bbox добавляет исходный фон; в mask также расширяет видимую маску на этот отступ, сохраняя немного окружения объекта. Остальное в mask заливается белым.'),
                    RULES.Input("rules", optional=True,
                        tooltip='Подключите rules из Region Tag Rules: пороги WD, удаление и добавление тегов применятся к регионам этого источника. Без подключения используются стандартные правила.'), SAM_SOURCES.Input("previous_sources", display_name="previous_sam_queries", optional=True,
                        tooltip='Необязательно: запросы SAM от другой SAM Text / Region Settings. Текущий запрос добавится к списку. Для отдельных веток используйте Combine SAM Queries; имена источников должны быть уникальны.')],
            outputs=[SAM_SOURCES.Output("sources",
                        tooltip='Список запросов SAM с настройками масок, вырезок и правилами тегов. Подключите к входу «sam_queries» вместе с sam_model. Несколько веток объединяются через Combine SAM Queries.')])

    @classmethod
    def execute(cls, source_name, prompt, confidence, mask_threshold, min_mask_area_px,
                max_regions, view_mode, padding_px, rules=None, previous_sources=None):
        sources = list(previous_sources or [])
        name = source_name.strip()
        if not name or not prompt.strip():
            raise ValueError("Source name and SAM prompt must be nonempty")
        if name in {s["name"] for s in sources}:
            raise ValueError("SAM source names must be unique")
        sources.append(dict(name=name, prompt=prompt.strip(), confidence=confidence, mask_threshold=mask_threshold,
            min_mask_area_px=min_mask_area_px, max_regions=max_regions, view_mode=view_mode,
            padding_px=padding_px, rules=dict(rules or {})))
        from .comfy_config import validate_config
        sources=validate_config({'schema_version':1,'sam_sources':sources})['sam_sources']
        return io.NodeOutput(sources)


class WDSAMSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMSegment", display_name="WDV3 SAM Segment Images", category="WDV3/SAM",
            description="Находит области SAM на входных изображениях по заданным текстовым источникам. Выдаёт sam_regions с отдельными объектами и combined_mask для каждого изображения. Готовые области можно передать в Tag Image / Regions вместе с теми же исходными изображениями.",
            inputs=[io.Image.Input("images",
                        tooltip='Исходное изображение или батч IMAGE, например из Load Image. Для SAM Segment и тегирования подключайте одни и те же изображения в одинаковом порядке, чтобы маски соответствовали картинкам.'), SAM_MODEL.Input("sam_model",
                        tooltip='Настройки SAM из Load SAM 3.1 или Load Pipeline Config. Подключайте вместе с запросами SAM. SAM ищет объекты по тексту; для получения тегов также нужна модель WD.'), SAM_SOURCES.Input("sources", display_name='sam_queries',
                        tooltip='Текстовые запросы SAM и настройки регионов из SAM Text / Region Settings, Combine SAM Queries или Load Pipeline Config. Подключайте вместе с sam_model. Каждый источник сохраняет собственные пороги, режим вырезки и правила.'),
                    io.Int.Input("sam_batch_size", default=1, min=1, max=16,
                        tooltip='Сколько исходных изображений SAM сегментирует одновременно. Начните с 1; увеличение требует больше памяти. Это не число объектов на изображении и не батч WD.')],
            outputs=[SAM_REGIONS.Output("sam_regions",
                        tooltip='Маски SAM и данные регионов для каждого исходного изображения, со всеми источниками. Подключите к sam_regions у Tag Image / Regions или Results Viewer; передавайте те же изображения в том же порядке. Это не обычный выход MASK.'), io.Mask.Output("combined_mask",
                        tooltip='Одна объединённая маска на изображение исходного размера: 1 (белое) — попало хотя бы в один принятый регион любого источника, 0 (чёрное) — фон. Подключите к нодам с входом MASK. Отдельные объекты и источники доступны в sam_regions.')])

    @classmethod
    def execute(cls, images, sam_model, sources, sam_batch_size):
        if sam_model is None or not sources:
            raise ValueError("The selected config needs a SAM model and at least one SAM source")
        with ExitStack() as stack:
            frames = [stack.enter_context(tensor_frame(t)) for t in images]
            records = segment_frames(frames, sam_model, sources, sam_batch_size, progress)
        masks = []
        for record in records:
            mask = np.zeros((record["size"][1], record["size"][0]), dtype=np.float32)
            for rows in record["regions"].values():
                for row in rows:
                    mask[row["mask"]] = 1
            masks.append(torch.from_numpy(mask))
        return io.NodeOutput(records, torch.stack(masks))


class WDLoadFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3LoadFolder", display_name="WDV3 Image Folder", category="WDV3/Batch",
            description="Собирает пути изображений выбранной папки с учётом вложенных каталогов, начального индекса и лимита. Выдаёт files и число изображений. Подключите files к Tag Folder для пакетного тегирования.",
            inputs=[string_input("directory", default=folder_paths.get_input_directory(),
                        tooltip='Папка с исходными изображениями на машине, где работает ComfyUI. Нода собирает пути; для обработки подключите files к Tag Folder. Оригиналы не изменяются.'),
                    io.Boolean.Input("recursive", default=False,
                        tooltip='Искать изображения также во всех подпапках. Если выключено, берутся только файлы непосредственно в directory.'),
                    io.Int.Input("start_index", default=0, min=0, max=10000000,
                        tooltip='С какого изображения в отсортированном списке начинать: 0 — первое. Удобно для обработки папки по частям; применяется до limit.'),
                    io.Int.Input("limit", default=0, min=0, max=10000000, tooltip='Сколько изображений взять после start_index. 0 — все оставшиеся подходящие изображения.')],
            outputs=[FILES.Output("files",
                        tooltip='Пути к выбранным изображениям, без загрузки всей папки в память. Подключите к files у Tag Folder.'), io.Int.Output("image_count",
                        tooltip='Количество изображений после применения recursive, start_index и limit. Позволяет проверить размер выбранной части папки до тегирования.')], not_idempotent=True)

    @classmethod
    def execute(cls, directory, recursive, start_index, limit):
        files = scan_folder(directory, recursive, start_index, limit)
        return io.NodeOutput(files, len(files["paths"]))


class WDTagFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3TagFolder", display_name="WDV3 Tag Folder", category="WDV3/Batch",
            description="Тегирует изображения из files полным WD и подключёнными YOLO/SAM-источниками. Для каждого изображения формирует общий caption и подробный JSON; при включении сохранения записывает TXT/JSON в выбранную папку. Общие параметры берутся из options.",
            is_output_node=True,
            inputs=[FILES.Input("files",
                        tooltip='Список файлов из Image Folder. Изображения читаются группами, поэтому папка может содержать картинки разных размеров.'), MODEL.Input("wd_model",
                        tooltip='Модель WD из Load Tagger Model. Она распознаёт теги на полном изображении и вырезанных регионах; YOLO/SAM только находят эти регионы.'), OPTIONS.Input("options",
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'),
                    io.Boolean.Input("save_txt", default=True,
                        tooltip='Сохранить итоговые теги в текстовый файл TXT. Для Export Report нужен отчёт по results из Apply Filters или Merge Results; сырые evidence и геометрия regions не содержат итоговых тегов.'), io.Boolean.Input("save_json", default=True,
                        tooltip='Сохранить подробные данные в JSON: оценки, регионы и сведения о результате. Формат результата и формат отчёта Inspect различаются.'),
                    string_input("output_directory", default=str(Path(folder_paths.get_output_directory()) / "wdv3"),
                        tooltip='Папка сохранения на машине, где работает ComfyUI; при необходимости создаётся автоматически. Это папка, а не имя файла.'),
                    io.Boolean.Input("overwrite", default=False,
                        tooltip='Разрешить заменять существующие файлы с теми же именами. Если выключено, при совпадении имён будет ошибка: выберите другое имя или папку.'),
                    io.Int.Input("sam_batch_size", default=1, min=1, max=16,
                        tooltip='Сколько исходных изображений SAM сегментирует одновременно. Начните с 1; увеличение требует больше памяти. Это не число объектов на изображении и не батч WD.'),
                    DETECTORS.Input("detectors", optional=True,
                        tooltip='Источники YOLO из YOLO Detector / Combine YOLO Detectors / Load Pipeline Config. Их прямоугольные вырезки тегируются WD вместе с полным изображением; фон внутри вырезок сохраняется.'), SAM_MODEL.Input("sam_model", optional=True,
                        tooltip='Настройки SAM из Load SAM 3.1 или Load Pipeline Config. Подключайте вместе с запросами SAM. SAM ищет объекты по тексту; для получения тегов также нужна модель WD.'),
                    SAM_SOURCES.Input("sources", display_name='sam_queries', optional=True,
                        tooltip='Текстовые запросы SAM и настройки регионов из SAM Text / Region Settings, Combine SAM Queries или Load Pipeline Config. Подключайте вместе с sam_model. Каждый источник сохраняет собственные пороги, режим вырезки и правила.')],
            outputs=[io.String.Output("tags", is_output_list=True,
                        tooltip='Готовый текст тегов: отдельная строка на каждое изображение, с выбранным форматированием. Можно передать в текстовый промпт или tags у Save Result вместе с results_json из той же ветки.'), io.String.Output("results_json", is_output_list=True,
                        tooltip='Подробный JSON для каждого изображения: теги полного кадра и регионов, оценки и происхождение итоговых тегов. Подключите к Results Viewer или Save Result. Это содержимое JSON, а не путь к файлу.')])

    @classmethod
    def execute(cls, files, wd_model, options, save_txt, save_json, output_directory, overwrite,
                sam_batch_size, detectors=None, sam_model=None, sources=None):
        if sources and sam_model is None:
            raise ValueError("SAM sources need sam_model")
        tags, details = tag_folder(files, wd_model, options, detectors=detectors, sam_settings=sam_model if sources else None,
            sources=sources, sam_batch_size=sam_batch_size, output_directory=output_directory,
            save_txt=save_txt, save_json=save_json, overwrite=overwrite, progress=progress)
        return io.NodeOutput(tags, details)


class WDSaveResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SaveResult", display_name="WDV3 Save Result", category="WDV3/Output",
            description="Сохраняет один выбранный результат из results_json в TXT и/или JSON. result_index выбирает изображение в батче. Необязательный вход tags позволяет записать точный текст того же результата с его форматированием.",
            is_output_node=True, is_input_list=True, not_idempotent=True,
            inputs=[string_input("results_json", force_input=True,
                        tooltip='Подробные результаты из выхода results_json ноды тегирования, Apply Filters или Merge Results: один JSON на изображение. Подключайте напрямую; это не путь к файлу и не простой список тегов.'),
                    string_input("output_directory", default=str(Path(folder_paths.get_output_directory()) / "wdv3"),
                        tooltip='Папка сохранения на машине, где работает ComfyUI; при необходимости создаётся автоматически. Это папка, а не имя файла.'),
                    string_input("filename", default="result",
                        tooltip='Имя сохраняемого результата без папки и расширения, например result_01. Допустимы буквы, цифры, пробелы, _ и -; первый символ — буква или цифра. Расширения .txt и .json добавляются автоматически.'),
                    io.Int.Input("result_index", default=0, min=0, max=10000000,
                        tooltip='Какое изображение из батча сохранить: 0 — первое, 1 — второе и т. д. За один запуск сохраняется один выбранный результат; для всей папки используйте Tag Folder.'),
                    io.Boolean.Input("save_txt", default=True,
                        tooltip='Сохранить итоговые теги в текстовый файл TXT. Для Export Report нужен отчёт по results из Apply Filters или Merge Results; сырые evidence и геометрия regions не содержат итоговых тегов.'), io.Boolean.Input("save_json", default=True,
                        tooltip='Сохранить подробные данные в JSON: оценки, регионы и сведения о результате. Формат результата и формат отчёта Inspect различаются.'),
                    io.Boolean.Input("overwrite", default=False,
                        tooltip='Разрешить заменять существующие файлы с теми же именами. Если выключено, при совпадении имён будет ошибка: выберите другое имя или папку.'),
                    string_input("tags", force_input=True, optional=True,
                        tooltip='Необязательно: tags из той же ноды и ветки, что и results_json. Сохраняет точное форматирование текста, включая пробелы и экранирование скобок. Без подключения текст берётся из JSON.')],
            outputs=[io.String.Output("txt_path",
                        tooltip='Полный путь к сохранённому TXT. Пустая строка, если сохранение TXT выключено.'), io.String.Output("json_path",
                        tooltip='Полный путь к сохранённому JSON. Пустая строка, если сохранение JSON выключено.')])

    @classmethod
    def execute(cls, results_json, output_directory, filename, result_index, save_txt, save_json, overwrite, tags=None):
        from .comfy_export import save_result
        txt_path, json_path = save_result(results_json, output_directory[0], filename[0],
            result_index=result_index[0], tags=tags, save_txt=save_txt[0], save_json=save_json[0], overwrite=overwrite[0])
        return io.NodeOutput(txt_path, json_path, ui={"text": [f"Saved: {path}" for path in (txt_path, json_path) if path]})


class WDPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3Preview", display_name="WDV3 Results Viewer", category="WDV3",
            description="Displays results_json from Tag Folder, Tag Image / Regions, Merge Tag Sets, Apply Filters or Merge Results: captions, image and region tags, scores and provenance. For Tag Folder, original images are read from result paths automatically. For Tag Image / Regions, connect the original images for previews. No inference or file saving is performed by this viewer.",
            is_output_node=True, is_input_list=True,
            inputs=[string_input("results_json", force_input=True,
                        tooltip='Connect results_json from Tag Folder, Tag Image / Regions, Merge Tag Sets, Apply Filters or Merge Results. Tag Folder results include original image paths, so images is optional. Use results_json, not tags or a file path.'),
                    io.Int.Input("preview_limit", default=16, min=0, max=256,
                        tooltip='0 = show all results. A positive value shows only the first N results. Affects viewing only, not inference or saving. Showing all images in a large dataset can use more memory.'),
                    io.Image.Input("images", optional=True,
                        tooltip='Для результатов Tag Image / Regions подключите тот же исходный батч IMAGE: появится предпросмотр картинки и рамок. Для Tag Folder просмотрщик читает оригиналы по путям из results_json.'), SAM_REGIONS.Input("sam_regions", optional=True,
                        tooltip='Готовые маски и регионы из SAM Segment Images для этих же изображений и в том же порядке. В Tag Image / Regions используйте либо этот вход, либо пару sam_model + запросы SAM. Для просмотра масок подключите также к Results Viewer.')], outputs=[])

    @classmethod
    def execute(cls, results_json, preview_limit, images=None, sam_regions=None):
        summaries = []
        tensors = [tensor for batch in (images or []) for tensor in batch]
        records = [record for group in (sam_regions or []) for record in group]
        total = len(results_json)
        visible = results_json if preview_limit[0] == 0 else results_json[:preview_limit[0]]
        for index, encoded in enumerate(visible):
            detail = json.loads(encoded)
            image_index = detail.get("image_index", index)
            frame = (tensor_frame(tensors[image_index]) if image_index < len(tensors) else
                     file_frame(detail["image_path"]) if detail.get("image_path") else None)
            assets = {"image": None, "region_masks": {}}
            if frame is not None:
                from .comfy_preview import preview_assets
                try:
                    record = records[image_index] if image_index < len(records) else None
                    assets = preview_assets(frame, folder_paths.get_temp_directory(), record)
                finally:
                    frame.close()
            summaries.append({"detail": detail, **assets})
        return io.NodeOutput(ui={"wdv3_results": summaries, "wdv3_total": [total]})


class WDCombineDetectors(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3CombineDetectors", display_name="WDV3 Combine YOLO Detectors", category="WDV3/Regions",
            description="Собирает несколько независимо настроенных YOLO-источников в один detectors для тегирования или Collect Evidence. Имена источников должны быть уникальны. Новые входы появляются по мере подключения.",
            inputs=[io.Autogrow.Input("detectors", template=io.Autogrow.TemplatePrefix(
                DETECTORS.Input("detector",
                        tooltip='Подключите detectors от отдельно настроенной YOLO-ноды. При подключении появляется следующий вход. Все источники объединяются; их имена должны быть уникальны.'), prefix="detector", min=1, max=64))],
            outputs=[DETECTORS.Output("detectors",
                        tooltip='Список источников YOLO с индивидуальными настройками. Подключите к detectors у ноды тегирования или Collect Evidence; объединить ветки можно через Combine YOLO Detectors.')])

    @classmethod
    def execute(cls, detectors):
        from .comfy_config import combine_detectors
        return io.NodeOutput(combine_detectors(detectors.values()))


class WDCombineSAMSources(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3CombineSAMSources", display_name="WDV3 Combine SAM Queries", category="WDV3/SAM",
            description="Собирает несколько SAM-промптов и настроек регионов в общий запросы SAM для одной SAM-модели. Каждый источник сохраняет своё имя, параметры вырезки и правила тегов. Подключите результат вместе с sam_model.",
            inputs=[io.Autogrow.Input("sources", display_name="sam_queries", template=io.Autogrow.TemplatePrefix(
                SAM_SOURCES.Input("source",
                        tooltip='Подключите запросы SAM от отдельно настроенного SAM-запроса. При подключении появляется следующий вход. Все запросы используют общую модель SAM, сохраняя свои настройки и уникальные имена.'), prefix="source", min=1, max=64))],
            outputs=[SAM_SOURCES.Output("sources",
                        tooltip='Список запросов SAM с настройками масок, вырезок и правилами тегов. Подключите к входу «sam_queries» вместе с sam_model. Несколько веток объединяются через Combine SAM Queries.')])

    @classmethod
    def execute(cls, sources):
        from .comfy_config import combine_sources
        return io.NodeOutput(combine_sources(sources.values()))


def preset_directory():
    return Path(folder_paths.get_user_directory()) / "default" / "wdv3" / "configs"


def preset_files():
    paths = {}
    for label, directory in (("Example", ROOT / "configs"), ("Saved", preset_directory())):
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                paths[f"{label}: {path.stem}"] = path
    return paths


class WDSaveConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SaveConfig", display_name="WDV3 Save Pipeline Config", category="WDV3/Settings",
            description="Сохраняет подключённые настройки тегирования, YOLO/SAM-источники, конфигурацию SAM и цепочку постобработки в переносимый JSON-пресет. Настройки берутся из конфигурационных нод. Сохранённый файл открывается через Load Pipeline Config.",
            is_output_node=True,
            inputs=[string_input("preset_name", default="my_pipeline",
                        tooltip='Имя JSON-пресета настроек, например my_pipeline. Сохраняется в пользовательской папке ComfyUI default/wdv3/configs. Затем выберите его в Load Pipeline Config.'), io.Boolean.Input("overwrite", default=False,
                        tooltip='Разрешить заменять существующие файлы с теми же именами. Если выключено, при совпадении имён будет ошибка: выберите другое имя или папку.'),
                    OPTIONS.Input("options", optional=True,
                        tooltip='Настройки из Tagging Settings или Load Pipeline Config. Задают глобальные пороги WD, формат текста и батчи. В Tag Image / Regions эти настройки задаются только через options; без подключения используются стандартные значения.'), DETECTORS.Input("detectors", optional=True,
                        tooltip='Источники YOLO из YOLO Detector / Combine YOLO Detectors / Load Pipeline Config. Их прямоугольные вырезки тегируются WD вместе с полным изображением; фон внутри вырезок сохраняется.'),
                    SAM_SOURCES.Input("sources", display_name='sam_queries', optional=True,
                        tooltip='Текстовые запросы SAM и настройки регионов из SAM Text / Region Settings, Combine SAM Queries или Load Pipeline Config. Подключайте вместе с sam_model. Каждый источник сохраняет собственные пороги, режим вырезки и правила.'), SAM_MODEL.Input("sam_model", optional=True,
                        tooltip='Настройки SAM из Load SAM 3.1 или Load Pipeline Config. Подключайте вместе с запросами SAM. SAM ищет объекты по тексту; для получения тегов также нужна модель WD.'),
                    io.Custom('WDV3_POSTPROCESSING').Input('postprocessing', optional=True,
                        tooltip='Цепочка из Postprocessing Config или Load Pipeline Config. Настройки без изображений и инференса.'),
                    io.Custom('WDV3_CALCULATION_STORE').Input('calculation_store',optional=True),
                    io.Custom('WDV3_INFERENCE').Input('inference',optional=True),
                    io.Custom('WDV3_CAPTION_READ').Input('caption_read',optional=True),
                    io.Custom('WDV3_CAPTION_EDIT').Input('caption_edit',optional=True),
                    io.Custom('WDV3_CAPTION_EXPORT').Input('caption_export',optional=True)],
            outputs=[io.String.Output("config_path",
                        tooltip='Полный путь к JSON-пресету. Загрузите его через Load Pipeline Config; после создания нового пресета может понадобиться обновить ComfyUI.')])

    @classmethod
    def execute(cls, preset_name, overwrite, options=None, detectors=None, sources=None, sam_model=None, postprocessing=None, **dataset_settings):
        from .comfy_config import config_data, save_config
        if options is None and detectors is None and sources is None and sam_model is None and postprocessing is None and not any(v is not None for v in dataset_settings.values()):
            raise ValueError("Connect settings, YOLO detectors or SAM sources to save")
        path = save_config(preset_directory(), preset_name, config_data(options, detectors, sources, sam_model, postprocessing, **dataset_settings), overwrite)
        return io.NodeOutput(str(path), ui={"text": [f"Saved pipeline config: {path}"]})


class WDLoadConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3LoadConfig", display_name="WDV3 Load Pipeline Config", category="WDV3/Settings",
            description="Открывает JSON-пресет и выдаёт options, detectors, запросы SAM, sam_model и postprocessing. Подключите нужные выходы к соответствующим этапам обработки. Пресет содержит настройки, а WD-модель выбирается отдельно в Load Tagger Model.",
            inputs=[io.Combo.Input("preset", options=list(preset_files()) or ["No saved configs"],
                        tooltip='Сохранённый или примерный JSON-пресет. Подключите его выходы к ноде тегирования. Модели и изображения не хранятся в пресете; если новый файл не появился в списке, обновите ComfyUI.')],
            outputs=[OPTIONS.Output("options",
                        tooltip='Глобальные настройки тегирования. Подключите к options у Tag Image / Regions, Tag Folder, Collect Evidence или Apply Filters; также можно сохранить через Save Pipeline Config.'), DETECTORS.Output("detectors",
                        tooltip='Список источников YOLO с индивидуальными настройками. Подключите к detectors у ноды тегирования или Collect Evidence; объединить ветки можно через Combine YOLO Detectors.'), SAM_SOURCES.Output("sources",
                        tooltip='Список запросов SAM с настройками масок, вырезок и правилами тегов. Подключите к входу «sam_queries» вместе с sam_model. Несколько веток объединяются через Combine SAM Queries.'),
                     SAM_MODEL.Output("sam_model",
                        tooltip='Настройки модели SAM. Подключите к sam_model вместе с запросами SAM из SAM Text / Region Settings; отдельная модель WD нужна для распознавания тегов.'),
                     io.Custom('WDV3_POSTPROCESSING').Output('postprocessing',
                        tooltip='Сохранённая цепочка локальных операций для Filter Tag Sets. Старый пресет возвращает пустую цепочку.'),
                     io.Custom('WDV3_CALCULATION_STORE').Output('calculation_store'),
                     io.Custom('WDV3_INFERENCE').Output('inference'),
                     io.Custom('WDV3_CAPTION_READ').Output('caption_read'),
                     io.Custom('WDV3_CAPTION_EDIT').Output('caption_edit'),
                     io.Custom('WDV3_CAPTION_EXPORT').Output('caption_export')])

    @classmethod
    def fingerprint_inputs(cls, preset):
        import hashlib
        path = preset_files().get(preset)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else "missing"

    @classmethod
    def execute(cls, preset):
        from .comfy_config import config_values
        path = preset_files().get(preset)
        if path is None:
            raise ValueError("Unknown preset. Refresh ComfyUI to update the preset dropdown.")
        data = json.loads(path.read_text(encoding="utf-8"))
        options, detectors, sources, model = config_values(data)
        from .Scripts.tag_postprocessing import validate_chain
        return io.NodeOutput(Bundle.pack('rules',options), detectors, sources, sam_model_settings(**model) if model else None,
                             validate_chain(data.get('postprocessing', [])),
                             *(data.get(key) for key in ('calculation_store','inference','caption_read','caption_edit','caption_export')))


NODE_CLASSES = [WDTaggingOptions, WDRegionRules, WDYOLODetector, WDSAMLoader, WDSAMSource,
                WDSAMSegment, WDLoadFolder, WDTagFolder, WDPreview, WDCombineDetectors, WDCombineSAMSources,
                WDSaveConfig, WDLoadConfig, WDSaveResult]

from .comfy_pipeline_nodes import NODE_CLASSES as PIPELINE_CLASSES
NODE_CLASSES += PIPELINE_CLASSES
from .comfy_persistent_nodes import NODE_CLASSES as DATASET_CLASSES
NODE_CLASSES += DATASET_CLASSES
