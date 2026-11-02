"""Inference and folder processing shared by ComfyUI nodes, without UI imports."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .Scripts.configs import DetectorConfig
from .Scripts.preprocess import merge_tags_from_regions, NumpyEncoder
from .Scripts.region_tagging import RegionTagger
from .Scripts.regions import prepare_region_view
from .Scripts.sam_config import RegionViewConfig
from .Scripts.sam_masks import select_masks
from .Scripts.tag_utils import apply_tag_rules
from .Scripts.wdv3_timm import pil_ensure_rgb


def frame_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def tensor_frame(tensor):
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


def segment_frames(frames, settings, sources, batch_size=1, progress=None):
    """Reuse the project's SAM backend and mask selection; release weights after the stage."""
    if batch_size < 1 or not sources:
        raise ValueError("SAM needs positive batch_size and at least one source")
    from .Scripts.sam_runtime import create_backend
    if settings.get("model_id") == "unsloth/sam3.1":
        settings = dict(settings)
        folder = sam_folder(settings["checkpoint"], settings["checkpoint"], settings.get("allow_download", False))
        settings.update(checkpoint=str(folder), allow_download=False)
    backend = create_backend(settings)
    output = []
    try:
        backend.load()
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                if progress:
                    progress(start, len(frames))
                chunk = frames[start:start + batch_size]
                state = backend._encode_images(chunk)
                records = [{"digest": frame_digest(image), "size": list(image.size),
                            "sources": sources, "regions": {}} for image in chunk]
                for source in sources:
                    predictions = backend._predict_source(state, source, len(chunk))
                    for index, image in enumerate(chunk):
                        candidates = backend._candidates(predictions, index, image.size, source["confidence"])
                        try:
                            rows, _ = select_masks(candidates, source, image.size)
                        finally:
                            candidates.close()
                        records[index]["regions"][source["name"]] = rows
                    del predictions
                del state
                output.extend(records)
        return output
    finally:
        backend.offload()


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
    from .comfy_nodes import tag_regions
    options = options or {}
    general = options.get("general_threshold", .35)
    character = options.get("character_threshold", .75)
    region_batch = options.get("region_batch_size", 8)
    image_batch = options.get("image_batch_size", 4)
    if region_batch < 1 or image_batch < 1:
        raise ValueError("Batch sizes must be positive")
    if sam_regions is not None and len(sam_regions) != len(frames):
        raise ValueError("SAM and WD image counts differ")
    configs, models = detectors if detectors is not None else ([], {})
    all_configs = list(configs)
    if sam_regions:
        all_configs.extend(DetectorConfig(s["name"], "sam3", **s["rules"]) for s in sam_regions[0]["sources"])
    if len({c.name for c in all_configs}) != len(all_configs):
        raise ValueError("YOLO/SAM source names must be unique")
    tagger = RegionTagger(runtime.model, runtime.transform, runtime.labels, runtime.device, general, character)
    characters = [runtime.labels.names[i] for i in runtime.labels.character]
    tags, details = [], []
    try:
        if torch.device(runtime.device).type != "cpu":
            import comfy.model_management as management
            management.free_memory(sum(p.numel() * p.element_size() for p in runtime.model.parameters()) * 1.3,
                                   runtime.device)
        runtime.model.to(runtime.device)
        for start in range(0, len(frames), image_batch):
            if progress:
                progress(start, len(frames))
            chunk = frames[start:start + image_batch]
            full = tagger.tag_batch(chunk, DetectorConfig("full", ""))
            for offset, (image, tagged) in enumerate(zip(chunk, full)):
                index = start + offset
                detected = tag_regions(image, tagger, configs, models, region_batch)
                if sam_regions is not None:
                    detected.update(sam_tag_regions(image, sam_regions[index], tagger, region_batch))
                detail = {"image_index": index, "full_image": tagged, "detectors": detected}
                if image_paths is not None:
                    detail["image_path"] = str(image_paths[index])
                merged = apply_tag_rules(merge_tags_from_regions(detail, all_configs),
                    remove=options.get("remove_tags", ""), before=options.get("prefix_tags", ""),
                    after=options.get("suffix_tags", ""), character_names=characters)
                text = merged["taglist"] if options.get("replace_underscores", True) else merged["caption"]
                if options.get("escape_parentheses", True):
                    text = text.replace("(", r"\(").replace(")", r"\)")
                tags.append(text)
                details.append({**detail, "merged_tags": merged})
        if progress:
            progress(len(frames), len(frames))
        return tags, details
    finally:
        runtime.model.to("cpu")


def json_result(detail):
    return json.dumps(detail, cls=NumpyEncoder, ensure_ascii=False, indent=2)


def scan_folder(directory, recursive, start, limit):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    files = sorted((p for p in (root.rglob("*") if recursive else root.iterdir())
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")),
                   key=lambda p: p.relative_to(root).as_posix().casefold())
    files = files[start:start + limit] if limit else files[start:]
    if not files:
        raise ValueError("No images match the folder/range")
    return {"root": root, "paths": files}


def tag_folder(files, runtime, options, *, detectors=None, sam_settings=None, sources=None,
               sam_batch_size=1, output_directory=None, save_txt=False, save_json=False,
               overwrite=False, progress=None):
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
                    if target in used or (target.exists() and not overwrite):
                        raise FileExistsError(f"Output exists or collides: {target}")
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
            regions = segment_frames(frames, sam_settings, sources, sam_batch_size, progress) if sam_settings else None
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
