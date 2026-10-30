"""Deterministic subprocess fixture for protocol/batch integration acceptance."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))
from mask_ops import mask_bbox, mask_digest

for line in sys.stdin:
    request = json.loads(line)
    response = {"protocol_version": 1, "request_id": request["request_id"], "status": "ok"}
    if request["action"] == "init":
        mode = Path(request["model"]["checkpoint"]).stem
        response["provenance"] = {"backend": "fake", "checkpoint_sha256": "fixture"}
        print("fake stderr log", file=sys.stderr, flush=True)
    elif request["action"] == "segment":
        if mode == "timeout":
            time.sleep(30)
        if mode == "exit":
            sys.exit(2)
        if mode == "malformed":
            print("not json", flush=True)
            continue
        if mode == "mismatch":
            response["request_id"] = "wrong"
        if mode == "version":
            response["protocol_version"] = 999
        if mode == "oom":
            response.update(status="error", error="CUDA out of memory")
        records = []
        for image_index, image in enumerate(request["images"]):
            width, height = image["size"]
            for source_index, source in enumerate(request["sources"]):
                mask = np.zeros((height, width), dtype=bool)
                mask[1:max(2, height//2), 1:max(2, width//2)] = True
                filename = f"mask-{image_index}-{source_index}.png"
                with Image.fromarray(mask.astype(np.uint8)*255) as pixels:
                    pixels.save(Path(request["directory"]) / filename)
                geometry = {"image_id": image["image_id"], "source_id": source["source_id"],
                            "region_id": 0, "image_size": [width, height], "bbox": mask_bbox(mask),
                            "mask_path": filename, "mask_sha256": mask_digest(mask),
                            "prompt": source["prompt"], "segmentation_score": .9}
                if mode == "bad_size":
                    geometry["image_size"] = [1, 1]
                if mode == "bad_prompt":
                    geometry["prompt"] = "wrong"
                if mode == "escape":
                    geometry["mask_path"] = "../outside.png"
                records.append({"image_id": image["image_id"], "source_id": source["source_id"],
                                "status": "ok", "regions": [] if source["prompt"] == "nothing" else [geometry],
                                "diagnostics": {"selected": 0 if source["prompt"] == "nothing" else 1}})
        response["records"] = records[:-1] if mode == "incomplete" else records
    print(json.dumps(response), flush=True)
