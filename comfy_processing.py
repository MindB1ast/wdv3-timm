"""Inference and folder processing shared by ComfyUI nodes, without UI imports."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps
import torch

from .Scripts.configs import DetectorConfig
from .Scripts.preprocess import merge_tags_from_regions, NumpyEncoder
from .Scripts.region_tagging import RegionTagger
from .Scripts.regions import prepare_region_view
from .Scripts.sam_config import RegionViewConfig
from .Scripts.tag_utils import apply_tag_rules, enforce_owners
from .Scripts.wdv3_timm import pil_ensure_rgb


def frame_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def tensor_frame(tensor):
    if tensor.ndim != 3 or tensor.shape[-1] not in (3,4) or any(s<1 for s in tensor.shape):
        raise ValueError('Expected image [H,W,3/4]')
    pixels = (tensor.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    with Image.fromarray(pixels) as original:
        rgb = pil_ensure_rgb(original)
        try:
            return rgb.copy()
        finally:
            if rgb is not original:
                rgb.close()


def file_frame(path):
    with Image.open(path) as original, ImageOps.exif_transpose(original) as oriented:
        rgb = pil_ensure_rgb(oriented)
        try:
            return rgb.copy()
        finally:
            if rgb is not oriented:
                rgb.close()


def sam_folder(checkpoint, default_folder, allow_download):
    """Download only the normal weights and companion files, directly into models/."""
    from .Scripts.sam_transformers import METADATA_FILES
    folder = Path(checkpoint) if checkpoint.strip() else Path(default_folder)
    if folder.suffix == ".safetensors":
        if folder.name != "model.safetensors":
            raise ValueError("Use unquantized model.safetensors for Transformers SAM 3.1")
        folder = folder.parent
    required = ["model.safetensors", "config.json", "tokenizer_config.json"]
    ready = all((folder / name).is_file() for name in required)
    ready = ready and any((folder / name).is_file() for name in ("processor_config.json", "preprocessor_config.json"))
    ready = ready and ((folder / "tokenizer.json").is_file() or all(
        (folder / name).is_file() for name in ("vocab.json", "merges.txt")))
    if not ready:
        if not allow_download:
            raise FileNotFoundError(f"Incomplete SAM 3.1 folder: {folder}. Enable download_missing.")
        from huggingface_hub import snapshot_download
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id="unsloth/sam3.1", local_dir=str(folder),
                          allow_patterns=[*METADATA_FILES, "model.safetensors"])
    return folder.resolve()


def segment_frames(frames, settings, sources, batch_size=1, progress=None, *, worker_factory=None):
    """Use the existing isolated SAM worker; release all CPU/GPU weights on exit."""
    if batch_size < 1 or not sources:
        raise ValueError("SAM needs positive batch_size and at least one source")
    from .Scripts.sam3_adapter import SamWorker
    from .Scripts.sam_config import SamConfig, SamSourceConfig
    from .Scripts.regions import read_mask, resolve_asset
    from .comfy_evidence import interrupt
    from .comfy_config import validate_config
    sources=validate_config({'schema_version':1,'sam_sources':sources})['sam_sources']
    interrupt()
    if settings.get("model_id") == "unsloth/sam3.1":
        settings = dict(settings)
        folder = sam_folder(settings["checkpoint"], settings["checkpoint"], settings.get("allow_download", False))
        settings.update(checkpoint=str(folder), allow_download=False)
    config=SamConfig(checkpoint=settings.get('checkpoint',''),device=settings.get('device','auto'),
        allow_download=settings.get('allow_download',False),backend=settings.get('backend','auto'),
        model_id=settings.get('model_id'),revision=settings.get('revision'),
        sources=tuple(SamSourceConfig(source_id=s['name'],name=s['name'],prompt=s['prompt'],
            confidence=s['confidence'],mask_threshold=s['mask_threshold'],min_mask_area_px=s['min_mask_area_px'],
            max_regions=s['max_regions']) for s in sources))
    def cancelled():
        interrupt()
        return False
    output = []
    try:
        import folder_paths
        temporary_root=Path(folder_paths.get_temp_directory())
    except ImportError:
        temporary_root=Path(__file__).resolve().parent/'temp'
    temporary_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wdv3-sam-',dir=temporary_root) as temporary, \
         (worker_factory or SamWorker)(sys.executable,config,timeout=3600,cancelled=cancelled) as worker:
        directory=Path(temporary)
        for start in range(0,len(frames),batch_size):
            interrupt()
            if progress:progress(start,len(frames))
            chunk=frames[start:start+batch_size]
            images=[]
            for offset,image in enumerate(chunk):
                image_id=f'frame_{start+offset}'
                path=directory/(image_id+'.png');image.save(path)
                images.append(dict(image_id=image_id,path=str(path),size=list(image.size)))
            segmented=worker.segment(images,directory)
            by_image={row['image_id']:{'digest':frame_digest(image),'size':list(image.size),
                'model_settings':dict(settings),'sources':sources,'regions':{}} for row,image in zip(images,chunk)}
            for record in segmented:
                rows=[]
                for geometry in record['regions']:
                    mask_path=resolve_asset(directory,geometry['mask_path'])
                    mask=read_mask(mask_path,tuple(by_image[record['image_id']]['size']))
                    rows.append(dict(bbox=geometry['bbox'],segmentation_score=geometry['segmentation_score'],mask=mask))
                    mask_path.unlink()
                by_image[record['image_id']]['regions'][record['source_id']]=rows
            output.extend(by_image[row['image_id']] for row in images)
            for row in images:Path(row['path']).unlink()
    return output


def sam_tag_regions(image, record, tagger, batch_size):
    if record["size"] != list(image.size) or record["digest"] != frame_digest(image):
        raise ValueError("SAM regions belong to a different image; connect the same images to both nodes")
    output = {}
    for source in record["sources"]:
        rules = DetectorConfig(source["name"], "sam3", **source["rules"])
        view = RegionViewConfig(mode=source["view_mode"], padding_value=source["padding_px"])
        rows = record["regions"][source["name"]]
        regions = []
        for start in range(0, len(rows), batch_size):
            with ExitStack() as stack:
                crops, metadata = [], []
                for row in rows[start:start + batch_size]:
                    geometry = {"image_size": list(image.size), "bbox": row["bbox"]}
                    crop, meta = prepare_region_view(image, geometry, view, row["mask"])
                    crops.append(stack.enter_context(crop))
                    metadata.append(meta)
                for offset, tagged in enumerate(tagger.tag_batch(crops, rules)):
                    row = rows[start + offset]
                    regions.append({**tagged, "region_id": start + offset, "bbox": row["bbox"],
                                    "segmentation_score": row["segmentation_score"],
                                    "prompt": source["prompt"], "view": metadata[offset]})
        output[source["name"]] = regions
    return output


def tag_frames(frames, runtime, *, detectors=None, sam_regions=None, options=None,
               progress=None, image_paths=None):
    from .comfy_evidence import collect, apply_filters, outputs
    evidence = collect(frames, runtime, detectors=detectors, sam_regions=sam_regions,
                       options=options, image_paths=image_paths, progress=progress)
    results = apply_filters(evidence, options)
    texts, encoded = outputs(results)
    return texts, [json.loads(row) for row in encoded]


def json_result(detail):
    return json.dumps(detail, cls=NumpyEncoder, ensure_ascii=False, indent=2)


def scan_folder(directory, recursive, start, limit):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    def candidates():
        for directory,folders,names in os.walk(root,followlinks=False):
            folders[:]=[name for name in folders if name!='.wdv3']
            yield from (Path(directory)/name for name in names)
            if not recursive:folders.clear()
    files = sorted((p for p in candidates()
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")),
                   key=lambda p: p.relative_to(root).as_posix().casefold())
    files = files[start:start + limit] if limit else files[start:]
    if not files:
        raise ValueError("No images match the folder/range")
    return {"root": root, "paths": files}


def tag_folder(files, runtime, options, *, detectors=None, sam_settings=None, sources=None,
               sam_batch_size=1, output_directory=None, save_txt=False, save_json=False,
               overwrite=False, progress=None):
    sam_batch_size = options.get('sam_batch_size', sam_batch_size)
    paths = files["paths"]
    root = Path(files["root"])
    destinations = {}
    if save_txt or save_json:
        out = Path(output_directory).expanduser().resolve()
        used = set()
        for path in paths:
            targets = {}
            for suffix, enabled in [(".txt", save_txt), (".json", save_json)]:
                if enabled:
                    # Keep original extension: a.jpg and a.png cannot collide.
                    target = out / (str(path.relative_to(root)) + suffix)
                    if target in used:
                        raise FileExistsError(f"Output paths collide: {target}")
                    if target.exists() and not overwrite:
                        raise FileExistsError(f"Result file already exists: {target}. In Tag Folder enable overwrite "
                                              "to update saved TXT/JSON, or disable save_txt and save_json to view results only. "
                                              "This controls file saving, not inference; cache_mode=reuse_only still prevents inference.")
                    if target.resolve() == path.resolve():
                        raise ValueError("Output must not overwrite an input image")
                    targets[suffix] = target
                    used.add(target)
            destinations[path] = targets
    batch_size = options.get("image_batch_size", 4)
    if batch_size < 1:
        raise ValueError("image_batch_size must be positive")
    tags, details = [], []
    for start in range(0, len(paths), batch_size):
        if progress:
            progress(start, len(paths))
        chunk = paths[start:start + batch_size]
        with ExitStack() as stack:
            frames = [stack.enter_context(file_frame(path)) for path in chunk]
            from .comfy_evidence import cached_segment,apply_filters,outputs
            from .comfy_automatic_cache import cached_evidence
            raw=cached_evidence(frames,runtime,options=options,detectors=detectors,sam_model=sam_settings,sources=sources,
                                image_paths=chunk,dataset_root=root,progress=progress)
            if raw is not None:
                texts,encoded=outputs(apply_filters(raw,options))
                records=[json.loads(row) for row in encoded]
            else:
                regions = cached_segment(frames, runtime, sam_settings, sources, sam_batch_size, progress, low_memory_mode=options.get('low_memory_mode',True)) if sam_settings else None
                texts, records = tag_frames(frames, runtime, detectors=detectors, sam_regions=regions,
                                           options=options, image_paths=chunk, progress=progress)
        for offset, (path, text, record) in enumerate(zip(chunk, texts, records)):
            record["image_index"] = start + offset
            encoded = json_result(record)
            for suffix, destination in destinations.get(path, {}).items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects existing data, including races after preflight.
                with destination.open("w" if overwrite else "x", encoding="utf-8") as stream:
                    stream.write(text if suffix == ".txt" else encoded)
            tags.append(text)
            details.append(encoded)
    if progress:
        progress(len(paths), len(paths))
    return tags, details
