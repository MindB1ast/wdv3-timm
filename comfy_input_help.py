"""Input examples shared by legacy and V3 string fields; no execution changes."""
import hashlib
import json


def _json(value):
    return json.dumps(value, ensure_ascii=False)


_categories = {'general': ['1girl'], 'character': [], 'ratings': ['general']}
_scores = {'general': {'1girl': .95}, 'character': {}, 'ratings': {'general': .9}}
_calculation = [{
    'image_id': 'image_001', 'image_size': [512, 512], 'image_path': 'C:/dataset/image_001.png',
    'model': {'model_id': 'example', 'labels_count': 2, 'categories': _categories,
              'labels_sha256': hashlib.sha256(json.dumps(_categories, sort_keys=True, ensure_ascii=False).encode()).hexdigest()},
    'full_image': {'raw_scores': _scores}, 'detectors': {}, 'source_settings': [], 'geometry': {},
}]

EXAMPLES = {
    'remove_tags': 'watermark, blurry',
    'prefix_tags': 'highres',
    'suffix_tags': 'dataset_sample',
    'remove_from_full': 'blue_eyes, green_eyes',
    'remove_from_regions': 'watermark, blurry',
    'add_to_regions': 'person, red_hair:0.9',
    'require_full_confirmation': 'full_body, cowboy_shot',
    'only_from_source': 'blue_eyes, smile',
    'allowed_tags': 'blue_eyes, green_eyes, smile',
    'full_image_only_tags': 'solo, scenery',
    'protected_tags': 'skirt, dress',
    'sources': 'person_detector, face_detector',
    'source_name': 'person_detector',
    'prompt': 'person wearing a red shirt',
    'classes': '0',
    'directory': 'C:/dataset/images',
    'dataset_root': 'C:/dataset',
    'cache_dataset_root': 'C:/dataset',
    'store_path': 'C:/dataset/.wdv3',
    'output_directory': 'C:/dataset/tagging_output',
    'target_root': 'C:/dataset/captions',
    'custom_model_path': 'C:/models/yolo/person_yolov8s-seg.pt',
    'local_model_path': 'C:/models/my_model',
    'wd_local_path': 'C:/models/taggers/wd-eva02-large-tagger-v3',
    'repo_id': 'SmilingWolf/wd-eva02-large-tagger-v3',
    'model_id': 'SmilingWolf/wd-eva02-large-tagger-v3',
    'filename': 'image_001',
    'prefix': 'dataset_check',
    'preset_name': 'my_tagging_pipeline',
    'tags': '1girl, blue_eyes, long_hair',
    'cache_refresh': 'person_detector, wd_full',
    'refresh': 'person_detector, wd_full',
    'operation_id': 'caption-export-001',
    'run_id': '0123456789abcdef0123456789abcdef',
    'artifact_ids': _json(['0123456789abcdef' * 4]),
    'pinned_identity_json': _json({'weights_sha256': '0123456789abcdef' * 4}),
    'sources_json': _json([{'name': 'person_detector', 'kind': 'yolo',
        'model': {'files': {'weights': 'C:/models/yolo/person_yolov8s-seg.pt'}},
        'confidence': .4, 'classes': [0], 'rules': {'region_gen_threshold': .3}}]),
    'image_mapping_json': _json(['image_001.png', 'subfolder/image_002.jpg']),
    'mapping_json': _json({'image_001.png': 'captions/image_001.txt'}),
    'calculations_json': _json(_calculation),
    'results_json': _json({'image_path': 'C:/dataset/image_001.png', 'detectors': {},
        'full_image': _scores, 'merged_tags': dict(_scores, caption='1girl', taglist='1girl')}),
    'operations_json': _json([{'add_before': ['highres'], 'remove': ['watermark'], 'sort': 'preserve'}]),
    'relations_json': _json([{'specific': 'miniskirt', 'general': 'skirt'}]),
    'regex_relations_json': _json([{'id': 'colors', 'specific': r'(?:black|white)_(?P<base>.+)',
                                    'general': r'\g<base>'}]),
    'postprocessing_json': _json([{'schema_version': 1, 'operation': 'remove_redundant',
                                  'textual_enabled': True, 'scope': 'all'}]),
    'detectors_json': _json([{'name': 'person_detector', 'model_path': 'person_yolov8s-seg.pt',
                              'confidence': .4, 'classes': [0], 'region_gen_threshold': .3,
                              'region_char_threshold': .8}]),
}

NOTES = {
    'dataset_root': 'Папка датасета с изображениями.',
    'store_path': 'Пусто — папка .wdv3 внутри dataset_root.',
    'wd_local_path': 'Локальная папка с весами WD, конфигурацией и словарём тегов.',
    'model_id': 'Идентификатор WD-модели или поддерживаемый псевдоним, например vit.',
    'artifact_ids': 'Это пример формата. Используйте ID существующих расчётов из Save Evidence или отчёта Compute Missing.',
    'run_id': 'Это пример формата. Возьмите run_id существующего сохранённого запуска, не создавайте произвольный.',
    'pinned_identity_json': 'Это пример структуры. Подставьте настоящую identity сохранённой WD-модели; {} отключает явную привязку.',
    'image_mapping_json': 'Относительные пути внутри датасета; число имён должно совпадать с числом кадров IMAGE.',
    'calculations_json': 'Пример маленького refilter-only расчёта. Для своих изображений вставьте полный JSON с raw_scores из экспорта.',
    'operations_json': 'Упорядоченный JSON-массив операций редактирования подписей.',
    'mapping_json': 'JSON-объект: относительный путь изображения → относительный путь TXT.',
}


def input_tooltip(name, tooltip=''):
    if name in ('tag_separator', 'input_separator'):
        example = ('Примеры ввода без кавычек: ", " — запятая и пробел; " | " — символ | с пробелами. '
                   + ('Для вывода можно указать " " — один пробел или пустую строку.'
                      if name == 'tag_separator' else
                      'Поле используется при input_format=literal_separator; разделитель не может состоять только из пробелов.'))
    elif name.endswith('_regex'):
        example = 'Пример ввода (два отдельных шаблона, по одному на строку):\n(?:blue|green)_eyes\n.*_skirt'
    else:
        example = 'Пример ввода: ' + EXAMPLES[name]
    return '\n'.join(part for part in (tooltip.strip(), NOTES.get(name, ''), example) if part)


def string_spec(name, options):
    return ('STRING', {**options, 'tooltip': input_tooltip(name, options.get('tooltip', ''))})


def string_input(name, **kwargs):
    from comfy_api.latest import io
    kwargs['tooltip'] = input_tooltip(name, kwargs.get('tooltip', ''))
    return io.String.Input(name, **kwargs)
