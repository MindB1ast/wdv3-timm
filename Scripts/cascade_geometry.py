"""Cascade coordinates and tree identities in the canonical source frame."""
from copy import deepcopy

import numpy as np
from PIL import Image

from .cascade import compile_plan, region_uid
from .mask_ops import mask_bbox
from .regions import rectangle_geometry


def detector_crop(frame, parent_bbox=None):
    """Always select source pixels. Parent masks and WD views are irrelevant."""
    bbox = rectangle_geometry('', '', 0, frame.size,
                              parent_bbox or [0, 0, *frame.size])['bbox']
    return frame.crop(tuple(bbox)), bbox


def map_detection(image_id, stage_id, parent, local_index, size, input_bbox, detection):
    """Adapters already undo resize/letterbox; only source translation remains."""
    x, y, right, bottom = input_bbox
    local = rectangle_geometry(image_id, stage_id, local_index,
                               (right-x, bottom-y), detection['bbox'])['bbox']
    bbox = [local[0]+x, local[1]+y, local[2]+x, local[3]+y]
    mask = detection.get('mask')
    source_mask = None
    if mask is not None:
        if mask.shape != (bottom-y, right-x) or mask.dtype != np.bool_:
            raise ValueError('SAM mask must match actual detector input')
        source_mask = np.zeros((size[1], size[0]), dtype=bool)
        source_mask[y:bottom, x:right] = mask
        exact = mask_bbox(source_mask)
        if exact is None:
            raise ValueError('Empty SAM mask')
        if exact != bbox:
            raise ValueError('SAM mask and bbox disagree')
    parent_uid = parent['region_uid'] if parent else None
    row = {key: deepcopy(value) for key, value in detection.items() if key not in ('mask', 'bbox')}
    row.update(region_id=local_index, region_uid=region_uid(image_id, stage_id, parent_uid, local_index),
               parent_region_uid=parent_uid, stage_id=stage_id, source_id=stage_id,
               depth=parent['depth']+1 if parent else 1, local_bbox=local, bbox=bbox,
               stage_path=(parent.get('stage_path', []) if parent else []) + [stage_id],
               image_size=list(size), local_detection_index=local_index,
               detector_input_bbox=list(input_bbox),
               detector_input_transform=[[1, 0, x], [0, 1, y], [0, 0, 1]],
               relation='detected_within_parent_input' if parent else 'full_image_input')
    if source_mask is not None:
        row['mask'] = source_mask
    return row


def validate_tree(frame):
    """Validate saved hierarchy; flat legacy frames adapt to independent roots."""
    plan = frame.get('cascade', {}).get('plan')
    stages = {s.stage_id: s for s in compile_plan(plan).stages} if plan else None
    seen = {}
    for source, rows in frame['detectors'].items():
        for row in rows:
            uid = row.get('region_uid')
            if not uid or uid in seen:
                raise ValueError('Missing or duplicate region_uid')
            if row.get('stage_id') != source:
                raise ValueError('Region stage/source mismatch')
            seen[uid] = row
    for uid, row in seen.items():
        source = row['stage_id']
        if stages is not None and source not in stages:
            raise ValueError('Unknown region stage: ' + source)
        parent_uid = row.get('parent_region_uid')
        parent = seen.get(parent_uid)
        if parent_uid is not None and parent is None:
            raise ValueError('Missing parent_region_uid')
        expected_stage = stages[source].parent_stage if stages is not None else None
        if stages is not None and (parent['stage_id'] if parent else None) != expected_stage:
            raise ValueError('Region parent stage mismatch')
        if row['depth'] != (parent['depth']+1 if parent else 1):
            raise ValueError('Invalid region depth')
        if uid != region_uid(frame['image_id'], source, parent_uid, row['local_detection_index']):
            raise ValueError('Region UID does not match frame/stage/parent')
    return frame


def adapt_flat_frame(frame):
    result = deepcopy(frame)
    if result.get('cascade'):
        return validate_tree(result)
    for source, rows in result['detectors'].items():
        for row in rows:
            index = row['region_id']
            row.update(region_uid=region_uid(result['image_id'], source, None, index),
                       parent_region_uid=None, stage_id=source, depth=1,
                       local_detection_index=index, tag_route='separate')
    return validate_tree(result)
