"""Canonical geometry and repeatable RGB region views, without model inference."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .sam_config import RegionViewConfig
from .mask_ops import Mask, mask_bbox, mask_digest
from .types import PathInput, RegionGeometry, RegionViewMetadata

def sha256_file(path: PathInput) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_image(path: PathInput) -> tuple[Image.Image, dict]:
    """Return owned RGB pixels and a pixel-coordinate EXIF transform."""
    with Image.open(path) as original:
        width, height = original.size
        orientation = int(original.getexif().get(274, 1))
        matrices = {
            1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            2: [[-1, 0, width-1], [0, 1, 0], [0, 0, 1]],
            3: [[-1, 0, width-1], [0, -1, height-1], [0, 0, 1]],
            4: [[1, 0, 0], [0, -1, height-1], [0, 0, 1]],
            5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            6: [[0, -1, height-1], [1, 0, 0], [0, 0, 1]],
            7: [[0, -1, height-1], [-1, 0, width-1], [0, 0, 1]],
            8: [[0, 1, 0], [-1, 0, width-1], [0, 0, 1]],
        }
        with ImageOps.exif_transpose(original) as transposed:
            rgb = transposed.convert("RGB")
        return rgb, {"original_size": [width, height], "exif_orientation": orientation,
                     "original_to_canonical": matrices.get(orientation, matrices[1]),
                     "coordinate_system": "canonical_pixels_half_open_xyxy"}


def read_mask(path: PathInput, size: tuple[int, int]) -> Mask:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != size:
            raise ValueError("Mask must be L PNG matching canonical image size")
        pixels = np.asarray(image)
        if not np.isin(pixels, [0, 255]).all():
            raise ValueError("Mask must contain only 0 and 255")
        return pixels == 255


def resolve_asset(root: PathInput, relative: str) -> Path:
    directory = Path(root).resolve()
    candidate = (directory / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(directory):
        raise ValueError("Asset path must remain inside its export directory")
    return candidate


def rectangle_geometry(image_id: str, source_id: str, region_id: int,
                       size: tuple[int, int], bbox) -> RegionGeometry:
    """Adapt a YOLO box to canonical integer bounds without inventing a mask."""
    width, height = size
    values = [float(v) for v in bbox]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid bbox")
    x1, y1, x2, y2 = values
    bounds = [max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(width, math.ceil(x2)), min(height, math.ceil(y2))]
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("Empty bbox")
    return {"image_id": image_id, "source_id": source_id, "region_id": region_id,
            "image_size": list(size), "bbox": bounds}


def dilate_rectangle(mask: Mask, px: int, py: int) -> Mask:
    """Rectangular binary dilation with bounded work even for a huge radius."""
    height, width = mask.shape
    px, py = min(px, width), min(py, height)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    xs, ys = np.arange(width), np.arange(height)
    left, right = np.maximum(xs-px, 0), np.minimum(xs+px+1, width)
    top, bottom = np.maximum(ys-py, 0), np.minimum(ys+py+1, height)
    counts = (integral[bottom[:, None], right] - integral[top[:, None], right]
              - integral[bottom[:, None], left] + integral[top[:, None], left])
    return counts > 0


def prepare_region_view(image: Image.Image, geometry: RegionGeometry,
                        config: RegionViewConfig, mask: Mask | None = None
                        ) -> tuple[Image.Image, RegionViewMetadata]:
    """Select pixels; does not mutate geometry/mask or resize for any model."""
    if image.mode != "RGB" or list(image.size) != geometry["image_size"]:
        raise ValueError("RGB frame must match canonical geometry")
    x1, y1, x2, y2 = geometry["bbox"]
    width, height = image.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox outside canonical frame")
    if mask is not None:
        if mask.shape != (height, width) or mask_bbox(mask) != geometry["bbox"]:
            raise ValueError("Mask and bbox disagree")
    if config.mode == "mask" and mask is None:
        raise ValueError("mask mode requires an original mask")
    factor_x, factor_y = (x2-x1, y2-y1) if config.padding_unit == "fraction" else (1, 1)
    px, py = math.ceil(factor_x*config.padding_value), math.ceil(factor_y*config.padding_value)
    crop_bbox = [max(0, x1-px), max(0, y1-py), min(width, x2+px), min(height, y2+py)]
    a, b, c, d = crop_bbox
    crop = image.crop((a, b, c, d))
    if config.mode == "mask" and mask is not None:
        visible = dilate_rectangle(mask, px, py)
        a, b, c, d = crop_bbox
        pixels = np.array(crop)
        pixels[~visible[b:d, a:c]] = config.background_rgb
        crop.close()
        crop = Image.fromarray(pixels)
    content = hashlib.sha256(image.tobytes()).hexdigest()
    metadata: RegionViewMetadata = {
        "mode": config.mode, "padding_unit": config.padding_unit,
        "padding_value": config.padding_value, "padding_px": [px, py],
        "crop_bbox": crop_bbox, "offset": [-crop_bbox[0], -crop_bbox[1]],
        "background_rgb": list(config.background_rgb), "transform_version": 1, "fingerprint": "",
    }
    metadata["fingerprint"] = hashlib.sha256(json.dumps(
        [content, {key: value for key, value in geometry.items() if key != "mask_path"},
         mask_digest(mask) if mask is not None else None, metadata],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return crop, metadata


def prepare_saved_region(root: PathInput, canonical_path: str, geometry: RegionGeometry,
                         config: RegionViewConfig) -> tuple[Image.Image, RegionViewMetadata]:
    with Image.open(resolve_asset(root, canonical_path)) as frame:
        with frame.convert("RGB") as rgb:
            mask = None
            if "mask_path" in geometry:
                mask = read_mask(resolve_asset(root, geometry["mask_path"]), rgb.size)
                if mask_digest(mask) != geometry.get("mask_sha256"):
                    raise ValueError("Saved mask checksum mismatch")
            return prepare_region_view(rgb, geometry, config, mask)
