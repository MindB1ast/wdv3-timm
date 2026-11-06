"""Portable finite cascade plans. Validation never loads detector weights."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

from .configs import DetectorConfig
from .sam_config import RegionViewConfig, finite_number, positive_integer
from .tag_postprocessing import validate_chain


@dataclass(frozen=True)
class CascadeStage:
    stage_id: str
    parent_stage: str | None
    kind: Literal['yolo', 'sam']
    detector: dict | None = None
    sam_model: dict | None = None
    sam_source: dict | None = None
    tag_enabled: bool = True
    tag_route: Literal['separate', 'merge_to_parent'] = 'separate'
    wd_view: dict | None = None
    local_operations: tuple = ()
    aggregate_operations: tuple = ()


@dataclass(frozen=True)
class CascadePlan:
    stages: tuple[CascadeStage, ...]
    schema_version: int = 1
    max_depth: int = 8
    max_regions_per_parent: int = 64
    max_regions_per_image: int = 4096

    def data(self) -> dict:
        value = asdict(self)
        value['stages'] = [asdict(stage) for stage in self.stages]
        for stage in value['stages']:
            for field in ('local_operations', 'aggregate_operations'):
                stage[field] = list(stage[field])
        return value


def _name(value, field):
    if not isinstance(value, str) or not value.strip() or value == 'full':
        raise ValueError(f'{field}: expected nonempty name other than full')
    return value


def compile_plan(value: dict | CascadePlan) -> CascadePlan:
    """Check the entire graph and return stable topological order before I/O."""
    if isinstance(value, CascadePlan):
        value = value.data()
    if not isinstance(value, dict) or set(value) - set(CascadePlan.__dataclass_fields__):
        raise ValueError('Unknown cascade plan fields')
    if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('Cascade requires schema_version=1')
    limits = {key: positive_integer(value.get(key, default), key) for key, default in
              [('max_depth', 8), ('max_regions_per_parent', 64), ('max_regions_per_image', 4096)]}
    rows = value.get('stages')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Cascade stages must be a nonempty array')
    stages = {}
    for row in deepcopy(rows):
        if not isinstance(row, dict) or set(row) - set(CascadeStage.__dataclass_fields__):
            raise ValueError('Unknown cascade stage fields')
        name = _name(row.get('stage_id'), 'stage_id')
        if name in stages:
            raise ValueError('Duplicate stage_id: ' + name)
        parent = row.get('parent_stage')
        if parent is not None:
            _name(parent, 'parent_stage')
        kind = row.get('kind')
        if kind not in ('yolo', 'sam'):
            raise ValueError('Cascade kind must be yolo or sam')
        if type(row.get('tag_enabled', True)) is not bool:
            raise ValueError('tag_enabled must be boolean')
        if row.get('tag_route', 'separate') not in ('separate', 'merge_to_parent'):
            raise ValueError('Invalid tag_route')
        if parent is None and row.get('tag_route', 'separate') != 'separate':
            raise ValueError('Root stage requires separate route')
        if kind == 'yolo':
            if not isinstance(row.get('detector'), dict) or row.get('sam_model') or row.get('sam_source'):
                raise ValueError('YOLO stage needs exactly one detector')
            config = DetectorConfig.from_dict({**row['detector'], 'name': name})
            if not isinstance(config.model_path, str) or not config.model_path.strip():
                raise ValueError('YOLO model_path is required')
            finite_number(config.confidence, 'confidence', maximum=1)
            if ((config.classes is not None and (not isinstance(config.classes, list) or any(type(c) is not int or c < 0 for c in config.classes)))
                    or type(config.download_missing) is not bool or config.device not in ('auto', 'cpu', 'cuda')):
                raise ValueError('Invalid YOLO classes/device/download_missing')
            row['detector'] = asdict(config)
        else:
            if row.get('detector') or not isinstance(row.get('sam_model'), dict) or not row['sam_model']:
                raise ValueError('SAM stage needs model settings and one source')
            source = row.get('sam_source')
            if not isinstance(source, dict) or not isinstance(source.get('prompt'), str) or not source['prompt'].strip():
                raise ValueError('SAM source needs prompt')
            source['name'] = name
            for key, default in [('confidence', .5), ('mask_threshold', .5)]:
                source[key] = finite_number(source.get(key, default), key, maximum=1)
            for key, default in [('min_mask_area_px', 1), ('max_regions', 64)]:
                source[key] = positive_integer(source.get(key, default), key)
            source.setdefault('view_mode', 'bbox')
            source.setdefault('padding_px', 0)
            source.setdefault('rules', {})
            DetectorConfig(name, 'sam3', **source['rules'])
            RegionViewConfig(mode=source['view_mode'], padding_value=source['padding_px'])
        view = row.get('wd_view')
        if view is None:
            source = row.get('sam_source') or {}
            view = dict(mode=source.get('view_mode', 'bbox'), padding_value=source.get('padding_px', 0))
        row['wd_view'] = asdict(RegionViewConfig(**view))
        row['wd_view']['background_rgb'] = list(row['wd_view']['background_rgb'])
        for field in ('local_operations', 'aggregate_operations'):
            row[field] = tuple(validate_chain(list(row.get(field, []))))
        row.setdefault('parent_stage', None)
        stages[name] = CascadeStage(**row)
    for stage in stages.values():
        if stage.parent_stage is not None and stage.parent_stage not in stages:
            raise ValueError('Missing parent_stage: ' + stage.parent_stage)
    ordered, depths = [], {}
    pending = set(stages)
    while pending:
        ready = sorted(n for n in pending if stages[n].parent_stage is None or stages[n].parent_stage in depths)
        if not ready:
            raise ValueError('Cascade cycle: ' + ' -> '.join(sorted(pending)))
        for name in ready:
            parent = stages[name].parent_stage
            depths[name] = 1 if parent is None else depths[parent] + 1
            if depths[name] > limits['max_depth']:
                raise ValueError('Cascade max_depth exceeded at ' + name)
            ordered.append(stages[name])
            pending.remove(name)
    return CascadePlan(tuple(ordered), **limits)


def combine_plans(*plans):
    """Share identical ancestors, reject conflicting definitions or limits."""
    normalized = [compile_plan(plan).data() for plan in plans]
    if not normalized:
        raise ValueError('At least one cascade plan required')
    result = deepcopy(normalized[0])
    stages = {row['stage_id']: row for row in result['stages']}
    for plan in normalized[1:]:
        if any(result[key] != plan[key] for key in ('max_depth', 'max_regions_per_parent', 'max_regions_per_image')):
            raise ValueError('Conflicting cascade limits')
        for row in plan['stages']:
            name = row['stage_id']
            if name in stages and stages[name] != row:
                raise ValueError('Conflicting cascade stage: ' + name)
            stages[name] = row
    result['stages'] = list(stages.values())
    return compile_plan(result)


def region_uid(image_id: str, stage_id: str, parent_uid: str | None, local_index: int) -> str:
    return hashlib.sha256(json.dumps([image_id, stage_id, parent_uid, local_index],
                                    ensure_ascii=False).encode()).hexdigest()
