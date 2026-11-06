"""Compile ordinary Region Tag Rules child stacks into a private cascade plan."""
from copy import deepcopy
from urllib.parse import quote

from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle
from .Scripts.tag_postprocessing import validate_chain
from .comfy_cascade import validate_plan, collect_cascade


def has_nested(detectors=None, sources=None):
    return any(getattr(c, '_children', None) is not None for c in (detectors[0] if detectors else [])) or any(
        s.get('rules', {}).get('children') is not None for s in (sources or []))


def child_settings(detectors=None, sam_model=None, sources=None, *, tag_route='separate',
                   tag_enabled=True, local_operations=None, aggregate_operations=None):
    if (sam_model is None) != (not sources):
        raise ValueError('Child SAM needs child_sam_model + child_sources')
    if tag_route not in ('separate', 'merge_to_parent') or type(tag_enabled) is not bool:
        raise ValueError('Invalid child route or tag_enabled')
    if not detectors and not sources and tag_enabled and not local_operations and not aggregate_operations:
        return None
    model = sam_model.data() if isinstance(sam_model, Bundle) else deepcopy(sam_model)
    value = dict(schema_version=1, detectors=[c.to_dict() for c in (detectors[0] if detectors else [])],
                 sam_model=model, sam_sources=deepcopy(sources or []), tag_route=tag_route,
                 tag_enabled=tag_enabled, local_operations=validate_chain(local_operations or []),
                 aggregate_operations=validate_chain(aggregate_operations or []))
    validate_children(value)
    return value


def validate_children(value):
    """A temporary ordinary root validates child structure without any model load."""
    if not isinstance(value, dict) or set(value) - {'schema_version', 'detectors', 'sam_model', 'sam_sources',
            'tag_route', 'tag_enabled', 'local_operations', 'aggregate_operations'}:
        raise ValueError('Invalid nested Region Tag Rules children')
    if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('Nested rules require schema_version=1')
    build_plan(detectors=([DetectorConfig('__validation_root__', 'validation.pt', children=value)], {}))
    return value


def build_plan(detectors=None, sam_model=None, sources=None):
    configs = detectors[0] if detectors else []
    model = sam_model.data() if isinstance(sam_model, Bundle) else sam_model
    roots = dict(detectors=[c.to_dict() for c in configs], sam_model=deepcopy(model), sam_sources=deepcopy(sources or []))
    stages, active = [], set()

    def walk(yolo, sam_sources, sam_settings, parent=None, route='separate', depth=1):
        if depth > 8 and (yolo or sam_sources):
            raise ValueError('Nested detector max_depth exceeded')
        if not isinstance(yolo, list) or not isinstance(sam_sources, list):
            raise ValueError('Child detectors and sam_sources must be arrays')
        if bool(sam_sources) != (sam_settings is not None):
            raise ValueError('Nested SAM requires model/source pair')
        names = []
        for kind, row in [('yolo', r) for r in yolo] + [('sam', s) for s in sam_sources]:
            if not isinstance(row, dict):
                raise ValueError('Nested detector must be an object')
            name = row.get('name')
            if not isinstance(name, str) or not name.strip() or name == 'full' or name in names:
                raise ValueError('Nested source names must be nonempty and unique within a stack')
            names.append(name)
            stage_id = name if parent is None else parent + '/' + quote(name, safe='')
            metadata = row.get('children') if kind == 'yolo' else row.get('rules', {}).get('children')
            if metadata is not None:
                if not isinstance(metadata, dict) or id(metadata) in active:
                    raise ValueError('Nested detector cycle or invalid children')
                if type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1:
                    raise ValueError('Nested rules require schema_version=1')
                if set(metadata) - {'schema_version', 'detectors', 'sam_model', 'sam_sources', 'tag_route',
                                     'tag_enabled', 'local_operations', 'aggregate_operations'}:
                    raise ValueError('Unknown nested settings')
                if metadata.get('tag_route', 'separate') not in ('separate', 'merge_to_parent'):
                    raise ValueError('Invalid child tag_route')
                active.add(id(metadata))
            settings = metadata or {}
            stage = dict(stage_id=stage_id, parent_stage=parent, kind=kind, tag_route=route,
                         tag_enabled=settings.get('tag_enabled', True),
                         local_operations=settings.get('local_operations', []),
                         aggregate_operations=settings.get('aggregate_operations', []))
            if kind == 'yolo':
                stage['detector'] = {k: deepcopy(v) for k, v in row.items() if k != 'children'}
            else:
                source = {k: deepcopy(v) for k, v in row.items() if k != 'rules'}
                source['rules'] = {k: deepcopy(v) for k, v in row.get('rules', {}).items() if k != 'children'}
                stage.update(sam_source=source, sam_model=deepcopy(sam_settings))
            stages.append(stage)
            if metadata is not None:
                walk(metadata.get('detectors', []), metadata.get('sam_sources', []), metadata.get('sam_model'),
                     stage_id, metadata.get('tag_route', 'separate'), depth+1)
                active.remove(id(metadata))
    walk(roots['detectors'], roots['sam_sources'], roots['sam_model'])
    return validate_plan(dict(schema_version=1, stages=stages)).data(), roots


def collect_nested(frames, runtime, *, detectors=None, sam_model=None, sources=None,
                   sam_regions=None, options=None, image_paths=None, progress=None):
    if sam_regions is not None:
        if len(sam_regions) != len(frames):
            raise ValueError('SAM and nested IMAGE counts differ')
        sam_model = sam_model or sam_regions[0].get('model_settings')
    plan, roots = build_plan(detectors, sam_model, sources)
    adapter = None
    if sam_regions is not None:
        from .comfy_cascade import DetectorAdapters
        from .comfy_processing import frame_digest
        from .comfy_evidence import digest
        delegate = DetectorAdapters(runtime, options or {})
        records = {record['digest']: record for record in sam_regions}
        for image, record in zip(frames, sam_regions):
            if frame_digest(image) != record['digest'] or list(image.size) != record['size']:
                raise ValueError('SAM regions belong to a different nested input image')
        class Prefetched:
            def identity(self, stage):
                if stage.parent_stage is None and stage.kind == 'sam':
                    return digest([[key, [{'bbox': row['bbox'], 'mask': row['mask'].tobytes().hex()}
                        for row in record['regions'][stage.stage_id]]] for key, record in sorted(records.items())])
                return delegate.identity(stage)
            def __call__(self, images, stage):
                if stage.parent_stage is None and stage.kind == 'sam':
                    return [[dict(row, confidence=row['segmentation_score'])
                        for row in records[frame_digest(image)]['regions'][stage.stage_id]] for image in images]
                return delegate(images, stage)
            def finish_detection(self):
                delegate.finish_detection()
        adapter = Prefetched()
    evidence = collect_cascade(frames, runtime, plan, options=options, image_paths=image_paths,
                               progress=progress, detect=adapter)
    value = evidence.data()
    for frame in value['frames']:
        frame['cascade'].update(nested_rules=True, roots=deepcopy(roots))
        for rows in frame['detectors'].values():
            for row in rows:
                row['stage_path'] = row['stage_id'].split('/')
    return Bundle.pack('evidence', value)


def filter_configs(frame, configs):
    """Expand connected ordinary root settings to the saved internal source paths."""
    roots = deepcopy(frame['cascade']['roots'])
    changes = {c.name: c for c in configs}
    expected = {r['name'] for r in roots['detectors']} | {s['name'] for s in roots['sam_sources']}
    if set(changes) != expected:
        # Full internal configurations are also supported for backend callers.
        return configs
    roots['detectors'] = [changes[r['name']].to_dict() for r in roots['detectors']]
    for source in roots['sam_sources']:
        source['rules'] = {k: v for k, v in changes[source['name']].to_dict().items() if k not in ('name', 'model_path')}
    selected, _ = build_plan(([DetectorConfig.from_dict(r) for r in roots['detectors']], {}),
                              roots['sam_model'], roots['sam_sources'])
    before = deepcopy(frame['cascade']['plan'])
    after = deepcopy(selected)
    for value in (before, after):
        for stage in value['stages']:
            stage.pop('tag_route')
            stage.pop('local_operations')
            stage.pop('aggregate_operations')
            if stage['kind'] == 'yolo':
                from .comfy_evidence import geometry_config
                stage['detector'] = geometry_config(DetectorConfig.from_dict(stage['detector']))
            else:
                stage['sam_source'].pop('rules', None)
    if before != after:
        raise ValueError('Nested detection/view settings changed; collect new evidence')
    frame['cascade'].update(plan=selected, roots=roots, routed=False)
    return [DetectorConfig.from_dict(s['detector']) if s['kind'] == 'yolo' else
            DetectorConfig(s['stage_id'], 'sam3', **s['sam_source']['rules']) for s in selected['stages']]
