"""API graph construction and result contracts for real ComfyUI CPU tests."""
import json
from pathlib import Path

from PIL import Image

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'comfy_cpu'
MANIFEST = json.loads((FIXTURES / 'cases.json').read_text(encoding='utf8'))
CASES = MANIFEST['cases']


def build_prompt(scenario, catalog, models, output):
    case = CASES[scenario]
    graph = {}

    def add(kind, **values):
        defaults = {}
        for name, spec in catalog[kind]['input']['required'].items():
            options = spec[1] if len(spec) > 1 else {}
            if 'default' in options:
                defaults[name] = options['default']
            elif isinstance(spec[0], list):
                defaults[name] = spec[0][0]
        key = str(len(graph) + 1)
        graph[key] = {'class_type': kind, 'inputs': {**defaults, **values}}
        return [key, 0]

    evidence_inputs = {
        'images': add('LoadImage', image=MANIFEST['image']),
        'wd_model': add('WDV3ModelLoader', model='vit', device='cpu',
                        download_missing=False, local_model_path=str(models / 'wd')),
    }
    sam = {}
    if case['sam']:
        sam['sam_model'] = add('WDV3SAMLoader', model='unsloth/sam3.1', device='cpu',
                               download_missing=False, local_model_path=str(models / 'sam'))
        sam['sources'] = add('WDV3SAMSource', source_name='sam', prompt=MANIFEST['prompt'],
                             confidence=.5, mask_threshold=.5, min_mask_area_px=32,
                             max_regions=1, view_mode='mask', padding_px=0)
    if case['yolo']:
        yolo = {}
        if sam:
            yolo['rules'] = add('WDV3RegionRules', child_sam_model=sam['sam_model'],
                                child_sources=sam['sources'], child_tag_route='separate')
        evidence_inputs['detectors'] = add(
            'WDV3YOLODetector', model='yolov8n.pt', source_name='yolo', confidence=.35,
            classes=str(MANIFEST['class_id']), custom_model_path=str(models / 'yolo' / 'yolo11n.pt'),
            device='cpu', download_missing=False, **yolo,
        )
    else:
        evidence_inputs.update(sam)
    evidence = add('WDV3CollectEvidence', **evidence_inputs)
    report = add('WDV3Inspect', regions=[evidence[0], 1])
    export = add('WDV3ExportReport', report=report, output_directory=str(output), prefix=scenario,
                 save_json=True, save_html=False, save_txt=False)
    return graph, export[0]


def inside(box, bounds):
    x, y, right, bottom = box
    a, b, c, d = bounds
    return a <= x < right <= c and b <= y < bottom <= d


def check_report(scenario, report_path, size=None):
    case = CASES[scenario]
    size = list(MANIFEST['size'] if size is None else size)
    report = json.loads(report_path.read_text(encoding='utf8'))
    frame, = report['frames']
    if frame['image_size'] != size:
        raise ValueError('Unexpected frame size')
    detections = frame['detectors']
    bounds = [0, 0, *size]
    parents = detections.get('yolo', [])
    if case['yolo']:
        if not parents or any(row.get('class_id') != MANIFEST['class_id'] or
                              not inside(row['bbox'], bounds) for row in parents):
            raise ValueError('YOLO must detect the expected class with a valid bbox')
    if not case['sam']:
        return
    children = detections.get('yolo/sam' if case['yolo'] else 'sam', [])
    if not children:
        raise ValueError('SAM returned no regions')
    parent_boxes = {row['region_uid']: row['bbox'] for row in parents if row.get('region_uid')}
    for child in children:
        asset = (report_path.parent / child['mask_asset']).resolve()
        if not asset.is_relative_to(report_path.parent.resolve()):
            raise ValueError('Mask asset is outside the report')
        with Image.open(asset) as image:
            if image.format != 'PNG' or list(image.size) != size:
                raise ValueError('SAM mask must be a full-frame PNG')
            box = image.convert('L').getbbox()
        if box is None:
            raise ValueError('Empty SAM mask')
        if list(box) != child['bbox']:
            raise ValueError('SAM mask and bbox disagree')
        if case['yolo']:
            parent = parent_boxes.get(child.get('parent_region_uid'))
            if parent is None:
                raise ValueError('SAM mask has no YOLO parent')
            x, y = parent[:2]
            local = child.get('local_bbox')
            if (parent[:2] == [0, 0] or child.get('detector_input_bbox') != parent or
                    not inside(box, parent) or not local or
                    tuple(value + offset for value, offset in zip(local, (x, y, x, y))) != box):
                raise ValueError('SAM mask does not map into an offset YOLO crop')
