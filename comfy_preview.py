"""Clean preview images and selectable mask assets in original-image coordinates."""
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from .comfy_processing import frame_digest


def preview_assets(frame, directory, sam_record=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "wdv3-preview-" + uuid.uuid4().hex
    descriptor = lambda filename: {"filename": filename, "subfolder": "", "type": "temp"}
    with frame.copy() as preview:
        preview.thumbnail((1200, 1200))
        filename = prefix + ".png"
        preview.save(directory / filename)
        size = list(preview.size)
    masks = {}
    if sam_record is not None:
        if sam_record["digest"] != frame_digest(frame) or sam_record["size"] != list(frame.size):
            raise ValueError("Preview SAM regions belong to another image")
        for source, regions in sam_record["regions"].items():
            for index, row in enumerate(regions):
                filename = f"{prefix}-mask-{len(masks)}.png"
                with Image.fromarray(row["mask"].astype(np.uint8) * 255) as mask:
                    with mask.resize(tuple(size), Image.Resampling.NEAREST) as resized:
                        resized.save(directory / filename)
                masks[f"{source}:{index}"] = descriptor(filename)
    return {"image": descriptor(prefix + ".png"), "image_size": list(frame.size),
            "preview_size": size, "region_masks": masks}
