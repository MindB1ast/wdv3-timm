"""Lazy cascade collection through existing YOLO/SAM/WD adapters."""
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib

import numpy as np
from PIL import Image

from .Scripts.cascade import compile_plan
from .Scripts.cascade_geometry import detector_crop, map_detection, validate_tree
from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle
from .Scripts.region_tagging import RegionTagger
from .Scripts.regions import prepare_region_view, sha256_file
from .Scripts.sam_config import RegionViewConfig, positive_integer
from .comfy_evidence import digest, interrupt, model_identity, png, detector_path


class CascadeExecutionError(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def validate_plan(value):
    from .comfy_config import validate_config, validate_rules
    plan = compile_plan(value)
    for stage in plan.stages:
        if stage.kind == 'yolo':
            validate_config(dict(schema_version=1, detectors=[stage.detector]))
        else:
            validate_config(dict(schema_version=1, sam_sources=[stage.sam_source]))
            validate_rules(stage.sam_source['rules'])
            model = stage.sam_model
            if model.get('device', 'auto') not in ('auto', 'cpu', 'cuda'):
                raise ValueError('Invalid SAM cascade device')
            if not isinstance(model.get('checkpoint', ''), str) or not model.get('checkpoint'):
                raise ValueError('SAM cascade needs resolved checkpoint settings')
    return plan


def _weights(path):
    path = detector_path(path)
    if path.is_dir():
        return [(p.name, sha256_file(p)) for p in sorted(path.iterdir()) if p.is_file()]
    return sha256_file(path) if path.is_file() else str(path)


class DetectorAdapters:
    def __init__(self, runtime, options):
        self.runtime, self.options, self.models = runtime, options, {}
        self.sam_workers = {}
        self.sam_stack = ExitStack()

    def sam_worker(self, python, config, **kwargs):
        from .Scripts.sam3_adapter import SamWorker
        key = digest([config.checkpoint, config.device, config.allow_download,
                      config.backend, config.model_id, config.revision])
        if key not in self.sam_workers:
            self.sam_workers[key] = self.sam_stack.enter_context(SamWorker(python, config, **kwargs))
        worker = self.sam_workers[key]
        # The JSONL protocol sends sources with each request; the initialized
        # model stays unchanged while response validation uses this stage's source.
        worker.config = config
        return nullcontext(worker)

    def finish_detection(self):
        self.sam_stack.close()
        self.sam_workers.clear()

    def identity(self, stage):
        if stage.kind == 'yolo':
            c = stage.detector
            return {k: c[k] for k in ('model_path', 'confidence', 'classes', 'device', 'download_missing')} | {
                'weights': _weights(c['model_path'])}
        return dict(model=stage.sam_model, weights=_weights(stage.sam_model['checkpoint']),
                    source={k: v for k, v in stage.sam_source.items()
                            if k not in ('name', 'rules', 'view_mode', 'padding_px')})

    def __call__(self, images, stage):
        if stage.kind == 'sam':
            from .comfy_processing import segment_frames
            if self.options.get('low_memory_mode', True) and hasattr(self.runtime, 'release_weights'):
                self.runtime.release_weights()
            records = segment_frames(images, stage.sam_model, [stage.sam_source],
                self.options.get('sam_batch_size', 1), worker_factory=self.sam_worker)
            return [[dict(row, confidence=row['segmentation_score'])
                     for row in record['regions'][stage.stage_id]] for record in records]
        from .Scripts.model import load_yolo_model
        config = DetectorConfig.from_dict(stage.detector)
        key = digest([config.model_path, _weights(config.model_path)])
        if key not in self.models:
            path = detector_path(config.model_path)
            model = load_yolo_model(str(path), path.parent, allow_download=config.download_missing).to('cpu')
            self.models[key] = model
            self.models[digest([config.model_path, _weights(config.model_path)])] = model
        model = self.models[key]
        device = config.device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output = []
        try:
            for image in images:
                interrupt()
                results = model(image, conf=config.confidence, classes=config.classes, device=device, verbose=False)
                output.append([dict(bbox=box.xyxy[0].tolist(), confidence=float(box.conf[0]),
                                    class_id=int(box.cls[0]))
                               for result in results if result.boxes is not None for box in result.boxes])
        finally:
            model.to('cpu')
        return output


def collect_cascade(frames, runtime, plan, *, options=None, image_paths=None,
                    detect=None, score=None, cache=None, progress=None):
    """Bound parent and WD batches. Optional adapters support deterministic API tests."""
    plan = validate_plan(plan)
    options = dict(options or {})
    from .comfy_config import validate_config
    validate_config(dict(schema_version=1, options=options))
    if not frames or (image_paths is not None and len(image_paths) != len(frames)):
        raise ValueError('Cascade requires matching nonempty images and paths')
    batch = positive_integer(options.get('region_batch_size', 8), 'region_batch_size')
    adapters = detect or DetectorAdapters(runtime, options)
    metadata = model_identity(runtime)
    if score is None:
        def score(images):
            model = runtime.model
            try:
                model.to(runtime.device)
                return RegionTagger(model, runtime.transform, runtime.labels, runtime.device).score_batch(images)
            finally:
                model.to('cpu')
    if cache is None:
        cache = None if options.get('cache_mode') in ('off', 'refresh') else getattr(runtime, '_cascade_cache', None)
        if cache is None:
            cache = runtime._cascade_cache = OrderedDict()
    def reuse(key, compute):
        interrupt()
        if key not in cache:
            if options.get('cache_mode') == 'reuse_only':
                raise ValueError('Nested cascade calculation missing from process cache; reuse_only prohibits inference')
            cache[key] = deepcopy(compute())
            while len(cache) > 1024:
                cache.pop(next(iter(cache)))
        return deepcopy(cache[key])
    output = []
    for frame_index, image in enumerate(frames):
        interrupt()
        if progress:
            progress(frame_index, len(frames))
        image_id = hashlib.sha256(image.tobytes()).hexdigest()
        if image_paths is not None:
            image_id = digest([str(image_paths[frame_index]), image_id])
        else:
            image_id = digest([frame_index, list(image.size), image_id])
        detail = dict(image_id=image_id, image_size=list(image.size), image_png=png(image),
                      detectors={}, geometry={}, source_settings=[],
                      cascade=dict(schema_version=1, plan=plan.data(), routed=False, report=[]))
        if image_paths is not None:
            detail['image_path'] = str(image_paths[frame_index])
        count = 0
        try:
            for stage in plan.stages:
                rows = detail['detectors'][stage.stage_id] = []
                config = (DetectorConfig.from_dict(stage.detector) if stage.kind == 'yolo' else
                          DetectorConfig(stage.stage_id, 'sam3', **stage.sam_source['rules']))
                detail['source_settings'].append(asdict(config))
                parents = detail['detectors'][stage.parent_stage] if stage.parent_stage else [None]
                if not parents:
                    detail['cascade']['report'].append(dict(stage=stage.stage_id, status='skipped', reason='no_parents'))
                    continue
                # Identity hashing is delayed until an actual parent exists.
                identity = adapters.identity(stage) if hasattr(adapters, 'identity') else (
                    {key: stage.detector[key] for key in ('model_path', 'confidence', 'classes', 'device')}
                    if stage.kind == 'yolo' else dict(model=stage.sam_model,
                        source={k: v for k, v in stage.sam_source.items()
                                if k not in ('rules', 'view_mode', 'padding_px')}))
                for start in range(0, len(parents), batch):
                    interrupt()
                    for parent in parents[start:start+batch]:
                        with detector_crop(image, parent['bbox'] if parent else None)[0] as crop:
                            input_bbox = parent['bbox'] if parent else [0, 0, *image.size]
                            key = digest(['cascade-detect-v1', crop.size, hashlib.sha256(crop.tobytes()).hexdigest(), identity])
                            detections = reuse(key, lambda: adapters([crop], stage)[0])
                        valid, discarded = [], 0
                        for local_index, detection in enumerate(detections):
                            try:
                                row = map_detection(image_id, stage.stage_id, parent, local_index,
                                                    image.size, input_bbox, detection)
                            except ValueError:
                                discarded += 1
                                continue
                            valid.append(row)
                        valid.sort(key=lambda row: (-float(row.get('confidence', 0)), row['bbox'], row['local_detection_index']))
                        found = len(valid)
                        remaining = max(0, plan.max_regions_per_image-count)
                        valid = valid[:min(plan.max_regions_per_parent, remaining)]
                        # Stable local indices after canonical sorting and limiting.
                        for index, row in enumerate(valid):
                            from .Scripts.cascade import region_uid
                            row.update(local_detection_index=index, region_id=len(rows),
                                       region_uid=region_uid(image_id, stage.stage_id, row['parent_region_uid'], index),
                                       tag_enabled=stage.tag_enabled, tag_route=stage.tag_route)
                        count += len(valid)
                        rows.extend(valid)
                        detail['cascade']['report'].append(dict(stage=stage.stage_id,
                            parent_region_uid=parent['region_uid'] if parent else None,
                            status='truncated' if found > len(valid) else 'empty' if not valid else 'complete',
                            found=found, kept=len(valid), discarded=discarded, truncated=found-len(valid)))
            # All SAM calls finish before WD scoring, including low-memory release.
            if hasattr(adapters, 'finish_detection'):
                adapters.finish_detection()
            for stage in plan.stages:
                rows = detail['detectors'][stage.stage_id]
                for start in range(0, len(rows), batch):
                    with ExitStack() as stack:
                        crops, misses, keys = [], [], []
                        for index, row in enumerate(rows[start:start+batch], start):
                            mask = row.pop('mask', None)
                            if mask is not None:
                                with Image.fromarray(mask.astype(np.uint8)*255, 'L') as saved:
                                    row['mask_png'] = png(saved)
                            crop, view = prepare_region_view(image, row, RegionViewConfig(**stage.wd_view), mask)
                            stack.enter_context(crop)
                            row['view'] = view
                            row['general'], row['character'], row['ratings'] = {}, {}, {}
                            if not stage.tag_enabled:
                                continue
                            key = digest(['cascade-wd-v1', metadata, crop.size, hashlib.sha256(crop.tobytes()).hexdigest()])
                            keys.append((index, key))
                            if key not in cache:
                                crops.append(crop)
                                misses.append(key)
                        if crops:
                            if options.get('cache_mode') == 'reuse_only':
                                raise ValueError('Nested cascade WD scores missing; reuse_only prohibits inference')
                            interrupt()
                            scored = score(crops)
                            if len(scored) != len(crops):
                                raise ValueError('WD batch count mismatch')
                            for key, raw in zip(misses, scored):
                                cache[key] = deepcopy(raw)
                        for index, key in keys:
                            rows[index].update(deepcopy(cache[key]))
                        while len(cache) > 1024:
                            cache.pop(next(iter(cache)))
                for row in rows:
                    row.pop('mask', None)
            key = digest(['cascade-wd-full-v1', metadata, image.size, hashlib.sha256(image.tobytes()).hexdigest()])
            detail['full_image'] = reuse(key, lambda: score([image])[0])
            validate_tree(detail)
            output.append(detail)
        except Exception as error:
            # ComfyUI cancellation must remain cancellation, not a detector failure.
            if type(error).__name__ == 'InterruptProcessingException':
                raise
            detail['cascade']['report'].append(dict(stage=stage.stage_id, status='failed', reason=str(error)))
            partial = Bundle.pack('evidence', dict(model=metadata, frames=output, failed_frame={
                'image_id': image_id, 'cascade': detail['cascade']}))
            raise CascadeExecutionError(f'Cascade failed at {stage.stage_id}: {error}', partial) from error
        finally:
            if hasattr(adapters, 'finish_detection'):
                adapters.finish_detection()
    return Bundle.pack('evidence', dict(model=metadata, frames=output))
