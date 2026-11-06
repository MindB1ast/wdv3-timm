"""Lossless evidence storage/import, independent of ComfyUI and model loaders."""
import base64
from copy import deepcopy
from io import BytesIO
import math

from .artifact_store import StoreError, computation_key
from .evidence import Bundle


def validate_evidence(value):
    if not isinstance(value, Bundle) or value.kind != 'evidence':
        raise StoreError('Expected evidence Bundle, not TXT or merged results')
    data = value.data()
    model = data.get('model', {})
    if not model.get('labels_count') or not model.get('labels_sha256'):
        raise StoreError('Missing model dictionary identity/completeness metadata')
    if not isinstance(data.get('frames'), list) or not data['frames']:
        raise StoreError('Evidence requires non-empty frames')
    categories = ('general', 'character', 'ratings')
    expected_names = model.get('categories')
    reference = None
    for frame in data['frames']:
        if not frame.get('image_id') or len(frame.get('image_size', [])) != 2:
            raise StoreError('Missing image identity/size')
        if set(frame.get('detectors', {})) != {s['name'] for s in frame.get('source_settings', [])}:
            raise StoreError('Source settings/regions mismatch')
        for rows in frame['detectors'].values():
            ids = [r['region_id'] for r in rows]
            if len(set(ids)) != len(ids):
                raise StoreError('Duplicate local region_id')
            for row in rows:
                box = row.get('bbox', [])
                if len(box) != 4 or not all(isinstance(x, (int,float)) and math.isfinite(x) for x in box):
                    raise StoreError('Invalid region bbox')
                if not (0 <= box[0] < box[2] <= frame['image_size'][0] and 0 <= box[1] < box[3] <= frame['image_size'][1]):
                    raise StoreError('Region bbox outside image')
        for record in [frame['full_image']] + [r for rows in frame['detectors'].values() for r in rows]:
            names = {}
            for cat in categories:
                scores = record.get(cat)
                if not isinstance(scores, dict) or not all(isinstance(t, str) and isinstance(s, (int,float)) and not isinstance(s,bool) and math.isfinite(s) and 0 <= s <= 1 for t,s in scores.items()):
                    raise StoreError('Missing/invalid complete raw category: ' + cat)
                names[cat] = sorted(scores)
            if sum(len(v) for v in names.values()) != model['labels_count']:
                raise StoreError('raw_scores dictionary is incomplete')
            if reference is not None and names != reference:
                raise StoreError('raw_scores dictionary differs between records')
            if expected_names is not None and names != {k:sorted(v) for k,v in expected_names.items()}:
                raise StoreError('raw_scores mismatch with model dictionary')
            reference = names
    return data


def _assets(value, assets, *, restore=False, store=None):
    if isinstance(value, list):
        return [_assets(v, assets, restore=restore, store=store) for v in value]
    if isinstance(value, dict):
        if restore and set(value) == {'wdv3_blob'}:
            assert store is not None
            return base64.b64encode(store.get_blob(value['wdv3_blob'])).decode('ascii')
        result = {}
        for key, item in value.items():
            if not restore and key in ('image_png', 'mask_png') and item:
                try:
                    raw = base64.b64decode(item, validate=True)
                    from PIL import Image
                    with Image.open(BytesIO(raw)) as image:
                        if image.format != 'PNG':
                            raise StoreError('Evidence assets must be PNG')
                        image.verify()
                except Exception as exc:
                    raise StoreError('Invalid lossless PNG asset') from exc
                name = str(len(assets)) + '-' + key
                assets[name] = raw
                result[key] = {'wdv3_asset': name}
            else:
                result[key] = _assets(item, assets, restore=restore, store=store)
        return result
    return value


def _link_assets(value, refs):
    if isinstance(value, list):
        return [_link_assets(v, refs) for v in value]
    if isinstance(value, dict):
        if set(value) == {'wdv3_asset'}:
            return {'wdv3_blob': refs[value['wdv3_asset']]}
        return {k:_link_assets(v, refs) for k,v in value.items()}
    return value


def capabilities(data):
    frames = data['frames']
    pixels = all(f.get('image_png') for f in frames)
    masks = all(r.get('mask_png') for f in frames for s in f.get('geometry', {}).get('sam', [])
                for r in f['detectors'].get(s['name'], []))
    return dict(refilter=True, inspect_pixels=bool(pixels), recrop_masks=bool(pixels and masks),
                resume_inference=bool(pixels and masks and data['model'].get('identity')))


def save_evidence(store, bundle, *, run_id=None, dependencies=()):
    data = validate_evidence(bundle)
    assets = {}
    payload = _assets(data, assets)
    refs = {k:store.put_blob(v) for k,v in assets.items()}
    payload = _link_assets(payload, refs)
    key = computation_key('evidence_snapshot', dict(evidence=payload,dependencies=list(dependencies)))
    claim = store.claim(key)
    if claim['status'] == 'reused':
        return claim['artifact_id']
    if claim['status'] == 'waiting':
        raise StoreError('Evidence publication waiting for active owner')
    return store.publish(claim, 'evidence', dict(evidence=payload, capabilities=capabilities(data)),
                         blobs=assets, run_id=run_id,dependencies=dependencies)


def load_evidence(store, artifact_id):
    record = store.load(artifact_id)
    if record['stage'] != 'evidence':
        raise StoreError('Artifact is not evidence')
    data = _assets(record['payload']['evidence'], {}, restore=True, store=store)
    bundle = Bundle.pack('evidence', data)
    validate_evidence(bundle)
    return bundle


def import_calculations(store, value):
    if isinstance(value, Bundle):
        bundle = value
    elif isinstance(value, dict) and value.get('kind') == 'evidence':
        import json
        bundle = Bundle.from_json(json.dumps(value))
    else:
        frames = deepcopy(value if isinstance(value, list) else [value])
        if not frames or not all(isinstance(f,dict) and f.get('full_image') and f.get('model') for f in frames):
            raise StoreError('Supported: evidence Bundle or results_json with complete raw_scores')
        model = frames[0]['model']
        for frame in frames:
            if frame['model'] != model:
                raise StoreError('Imported results have different model identities')
            for record in [frame['full_image']] + [r for rows in frame.get('detectors', {}).values() for r in rows]:
                raw = record.get('raw_scores')
                if raw is None:
                    raise StoreError('No complete raw_scores; merged-only JSON is not evidence')
                for cat in ('general','character','ratings'):
                    if cat not in raw:
                        raise StoreError('raw_scores missing ' + cat)
                    record[cat] = raw[cat]
                for key in ('raw_scores','decisions','caption','taglist','postprocessing_history'):
                    record.pop(key, None)
            frame['source_settings'] = frame.get('source_settings', frame.get('settings', {}).get('sources', []))
            for key in ('merged_tags','decisions','stage','settings'):
                frame.pop(key, None)
        bundle = Bundle.pack('evidence', dict(model=model, frames=frames))
    data = validate_evidence(bundle)
    ref = save_evidence(store, bundle)
    return dict(artifact_id=ref, capabilities=capabilities(data))


def rebind_paths(store, bundle, mapping):
    """Mapping by frame index is export metadata; no inferred filename from pixels."""
    data = bundle.data()
    from PIL import Image
    from .inference_plan import pixel_identity
    from .workflow_api import normalized_image
    for index, relative in mapping.items():
        frame = data['frames'][int(index)]
        path = store.resolve_dataset_path(relative)
        if not path.is_file():
            raise StoreError('Missing source asset: ' + relative)
        with normalized_image(path) as image:
            expected = frame.get('pixel_identity')
            if expected is None and frame.get('image_png'):
                with Image.open(BytesIO(base64.b64decode(frame['image_png']))) as stored:
                    expected = pixel_identity(stored)
            if expected is None or expected != pixel_identity(image):
                raise StoreError('Source pixel digest mismatch or unavailable identity: ' + relative)
        frame['relative_path'] = relative
        frame['image_path'] = str(path)
    return Bundle.pack('evidence', data)
