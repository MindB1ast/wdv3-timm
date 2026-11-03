"""Read-only prompt validation. Never executes nodes or allocates model weights."""
import json
import math
from pathlib import Path


def _input_spec(schema, name):
    entries = {**schema.get('required', {}), **schema.get('optional', {})}
    if name in entries:
        return entries[name]
    root = name.split('.', 1)[0]
    spec = entries.get(root)
    if spec and spec[0] == 'COMFY_AUTOGROW_V3' and '.' in name:
        nested = spec[1]['template']['input']['required']
        return next(iter(nested.values()))
    return None


def _compatible(received, expected):
    if isinstance(expected, list):
        expected = 'COMBO'
    try:
        from comfy_execution.validation import validate_node_input
    except ImportError:  # Standalone contract tests without ComfyUI installed.
        return received == expected or received == '*' or expected == '*'
    return validate_node_input(received, expected)


def _validate_scalar(name, value, spec):
    kind, options = spec[0], spec[1] if len(spec) > 1 else {}
    choices = kind if isinstance(kind, list) else options.get('options') if kind == 'COMBO' else None
    if choices is not None and value not in choices:
        raise ValueError(f'{name}: недопустимый выбор {value!r}')
    if kind in ('INT', 'FLOAT'):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name}: требуется конечное число')
        if kind == 'INT' and not isinstance(value, int):
            raise ValueError(f'{name}: требуется целое число')
        if 'min' in options and value < options['min'] or 'max' in options and value > options['max']:
            raise ValueError(f'{name}: значение вне допустимого диапазона')
    if kind == 'BOOLEAN' and not isinstance(value, bool):
        raise ValueError(f'{name}: требуется boolean')
    if kind == 'STRING' and not isinstance(value, str):
        raise ValueError(f'{name}: требуется строка')


def _filename(value, field, *, max_length=120, reserved=True):
    if not value or len(value) > max_length or not value[0].isalnum() or any(
            not (char.isalnum() or char in ' _-') for char in value):
        raise ValueError(f'{field}: используйте буквы, цифры, пробел, _ и -, без пути или расширения')
    names = {'CON', 'PRN', 'AUX', 'NUL', *[f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)]}
    if reserved and value.upper() in names:
        raise ValueError(f'{field}: имя зарезервировано Windows')


def _sam_pair(connected):
    if 'sam_regions' in connected and {'sam_model', 'sources'} & connected:
        raise ValueError('Используйте либо готовые sam_regions, либо sam_model + sources, не оба способа.')
    if ('sources' in connected) != ('sam_model' in connected):
        missing = 'sam_model (Load SAM 3.1)' if 'sources' in connected else 'sources (SAM Text / Region Settings)'
        raise ValueError(f'Для SAM подключите {missing}, либо отключите оба SAM-входа. YOLO не обязателен.')


def _offline_model(node_id, values):
    """Only check filesystem metadata when downloads are explicitly disabled."""
    if values.get('download_missing', True) is not False:
        return
    from .comfy_nodes import ROOT, model_directory
    if node_id == 'WDV3ModelLoader' and 'model' in values:
        from .Scripts.tagger_catalog import model_path, resolve_model_id
        repo = resolve_model_id(values.get('repo_id', '').strip() or values['model'])
        local = values.get('local_model_path', '').strip()
        paths = [Path(local).expanduser()] if local else [
            model_path(ROOT / 'models' / 'taggers', repo), model_path(model_directory('taggers'), repo)]
        paths = [p.parent if p.suffix in ('.safetensors', '.bin') else p for p in paths]
        required = ['config.json', 'selected_tags.csv']
        if not repo.startswith('SmilingWolf/'):
            required.append('preprocess.json')
        if not any(all((p / name).is_file() for name in required) and any(
                (p / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')) for p in paths):
            raise ValueError('WD model: локальные файлы неполны; проверьте путь или включите download_missing.')
    elif node_id == 'WDV3YOLODetector' and 'model' in values:
        import folder_paths
        local = values.get('custom_model_path', '').strip()
        paths = [Path(local).expanduser()] if local else [directory / values['model'] for directory in (
            model_directory('yolo'), ROOT / 'models' / 'yolo',
            Path(folder_paths.models_dir) / 'ultralytics' / 'bbox',
            Path(folder_paths.models_dir) / 'ultralytics' / 'segm')]
        if not any(p.is_file() for p in paths):
            raise ValueError('YOLO model: локальный файл отсутствует; проверьте путь или включите download_missing.')
    elif node_id == 'WDV3SAMLoader':
        import folder_paths
        local = values.get('local_model_path', '').strip()
        paths = [Path(local).expanduser()] if local else [
            ROOT / 'models' / 'sam3', Path(folder_paths.models_dir) / 'sam3' / 'unsloth-sam3.1']
        paths = [p.parent if p.is_file() else p for p in paths]
        if not any(any((p / name).is_file() for name in ('model.safetensors', 'model.safetensors.index.json'))
                   and (p / 'config.json').is_file() and
                   any((p / name).is_file() for name in ('processor_config.json', 'preprocessor_config.json'))
                   and ((p / 'tokenizer.json').is_file() or
                        ((p / 'vocab.json').is_file() and (p / 'merges.txt').is_file()))
                   for p in paths):
            raise ValueError('SAM model: локальные файлы неполны; проверьте путь или включите download_missing.')


def validate_preflight(node_id, schema, input_types=None, values=None):
    """Return True or a user-facing validation error; linked values stay unknown."""
    received = dict(input_types or {})
    values = dict(values or {})
    connected = set(received)
    roots = {name.split('.', 1)[0] for name in connected}
    known = {name: value for name, value in values.items() if name not in roots}
    try:
        # input_types opts out of Comfy's default linked-type check; preserve it explicitly.
        for name, actual in received.items():
            spec = _input_spec(schema, name)
            if spec is None or not _compatible(actual, spec[0]):
                raise ValueError(f'{name}: несовместимый тип {actual!r}')
        # **kwargs opts out of native bounds/choices checks, so validate those too.
        for name, value in known.items():
            spec = _input_spec(schema, name)
            if spec is not None:
                _validate_scalar(name, value, spec)
        if node_id in ('WDV3Tagger', 'WDV3TagFolder', 'WDV3CollectEvidence'):
            _sam_pair(roots)
        if node_id == 'WDV3Inspect' and len(roots & {'results', 'evidence', 'regions'}) != 1:
            raise ValueError('Inspect: подключите ровно один вход — results, evidence или regions.')
        if node_id == 'WDV3SaveConfig' and not roots & {'options', 'detectors', 'sources', 'sam_model'}:
            raise ValueError('Save Pipeline Config: подключите хотя бы один набор настроек.')
        if node_id in ('WDV3CombineDetectors', 'WDV3CombineSAMSources') and not connected:
            raise ValueError('Combine: подключите хотя бы один источник.')
        if node_id in ('WDV3SaveResult', 'WDV3ExportReport'):
            formats = ('save_txt', 'save_json') + (('save_html',) if node_id == 'WDV3ExportReport' else ())
            if all(name in known for name in formats) and not any(known[name] for name in formats):
                raise ValueError('Выберите хотя бы один формат сохранения.')
        if 'output_directory' in known:
            if node_id != 'WDV3TagFolder' or known.get('save_txt', True) or known.get('save_json', True):
                directory = known['output_directory']
                if not directory.strip():
                    raise ValueError('output_directory: укажите каталог сохранения.')
                if Path(directory).exists() and not Path(directory).is_dir():
                    raise ValueError('output_directory: указан файл вместо папки.')
        if node_id == 'WDV3SaveResult' and 'filename' in known:
            _filename(known['filename'].strip(), 'filename')
        if node_id == 'WDV3SaveConfig' and 'preset_name' in known:
            _filename(known['preset_name'].strip(), 'preset_name', max_length=80)
        if node_id == 'WDV3ExportReport' and 'prefix' in known:
            prefix = known['prefix']
            if not prefix.strip() or any(not (c.isalnum() or c in ' _-') for c in prefix):
                raise ValueError('prefix: используйте буквы, цифры, пробел, _ и -.')
        if node_id == 'WDV3Options':
            from .comfy_config import validate_config
            from .Scripts.tag_utils import parse_regex
            options = dict(known)
            if 'remove_tags_regex' in options:
                options['remove_tags_regex'] = parse_regex(options['remove_tags_regex'])
            validate_config({'schema_version': 1, 'options': options})
        if node_id == 'WDV3RegionRules':
            from .Scripts.tag_utils import TagMatcher, parse_additions
            for name, value in known.items():
                if name.endswith('_regex'):
                    TagMatcher(patterns=value, field=name)
            if 'add_to_regions' in known:
                parse_additions(known['add_to_regions'])
        if node_id in ('WDV3YOLODetector', 'WDV3SAMSource') and 'source_name' in known:
            from .comfy_config import unique_names
            unique_names([known['source_name'].strip()])
        if node_id == 'WDV3SAMSource' and 'prompt' in known and not known['prompt'].strip():
            raise ValueError('prompt: текстовый запрос SAM не может быть пустым.')
        if node_id == 'WDV3YOLODetector' and 'classes' in known:
            ids = [int(text.strip()) for text in known['classes'].split(',') if text.strip()]
            if any(value < 0 for value in ids):
                raise ValueError('classes: ID классов должны быть неотрицательными целыми числами.')
        if node_id == 'WDV3ModelLoader' and ('repo_id' in known or 'model' in known):
            from .Scripts.tagger_catalog import resolve_model_id
            resolve_model_id(known.get('repo_id', '').strip() or known.get('model', 'vit'))
        if node_id in ('WDV3ModelLoader', 'WDV3YOLODetector', 'WDV3SAMLoader'):
            if known.get('device') == 'cuda':
                import torch
                if not torch.cuda.is_available():
                    raise ValueError('CUDA недоступна: выберите cpu или auto.')
            # Defer paths/selection supplied by upstream nodes instead of guessing their values.
            resource_fields = {'model', 'download_missing', 'local_model_path', 'custom_model_path', 'repo_id'}
            if not resource_fields & roots:
                _offline_model(node_id, known)
        if node_id == 'WDV3LoadFolder' and 'directory' in known:
            if not known['directory'].strip() or not Path(known['directory']).expanduser().is_dir():
                raise ValueError('directory: папка изображений не существует.')
        if node_id == 'WDV3DetectorLoader' and 'detectors_json' in known:
            from .comfy_config import validate_config
            data = json.loads(known['detectors_json'])
            if not isinstance(data, list):
                raise ValueError('detectors_json: требуется JSON-массив источников YOLO.')
            validate_config({'schema_version': 1, 'detectors': data})
        if node_id == 'WDV3LoadConfig' and 'preset' in known:
            from .comfy_ui_nodes import preset_files
            from .comfy_config import validate_config
            path = preset_files().get(known['preset'])
            if path is None:
                raise ValueError('preset: пресет не найден; обновите список ComfyUI.')
            validate_config(json.loads(path.read_text(encoding='utf-8')))
    except (ValueError, TypeError, KeyError, OSError) as error:
        return f'{node_id}: {error}'
    return True


def register_preflight(classes):
    """Register both public node APIs; each validation runs before prompt execution."""
    for node_id, cls in classes.items():
        if not node_id.startswith('WDV3') or cls.__dict__.get('_wdv3_preflight'):
            continue
        def validate(node_cls, input_types=None, **kwargs):
            if getattr(node_cls, 'INPUT_IS_LIST', False):
                input_types = input_types[0] if isinstance(input_types, list) else input_types
                kwargs = {name: value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
                          for name, value in kwargs.items()}
            return validate_preflight(node_cls._wdv3_preflight, node_cls.INPUT_TYPES(), input_types, kwargs)
        name = 'validate_inputs' if hasattr(cls, 'define_schema') else 'VALIDATE_INPUTS'
        setattr(cls, name, classmethod(validate))
        cls._wdv3_preflight = node_id
