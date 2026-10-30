"""Real CUDA acceptance. No fake backend and no implied success when unavailable."""
import argparse
import copy
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scripts import BatchTagging, ScriptOptions
from Scripts.region_export import json_text, load_region_record
from Scripts.sam_config import load_sam_config


def compare(first, second):
    assert len(first[0]) == len(second[0])
    for a, b in zip(first[0], second[0]):
        assert a["image_id"] == b["image_id"]
        for source in a["detectors"]:
            left, right = a["detectors"][source], b["detectors"][source]
            assert len(left) == len(right), source
            for x, y in zip(left, right):
                assert x["bbox"] == y["bbox"]
                gx, gy = x["geometry"], y["geometry"]
                assert gx.get("mask_sha256") == gy.get("mask_sha256")
                if "segmentation_score" in gx:
                    np.testing.assert_allclose(gx["segmentation_score"], gy["segmentation_score"], rtol=1e-4, atol=1e-5)
                for category in ("general", "character", "ratings"):
                    assert set(x[category]) == set(y[category])
                    np.testing.assert_allclose(list(x[category].values()), [y[category][k] for k in x[category]],
                                               rtol=1e-4, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Folder with at least two images of different sizes")
    parser.add_argument("--sam-config", required=True, type=Path)
    parser.add_argument("--sam-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU acceptance requires CUDA in the WD interpreter; a CPU run is not a GPU smoke")
    config = load_sam_config(args.sam_config)
    assert len(config.sources) >= 2, "Smoke requires multiple prompts"
    data = json.loads(args.sam_config.read_text(encoding="utf-8"))
    data["model"]["checkpoint"] = config.checkpoint
    sizes = set()
    for path in args.images.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            with Image.open(path) as image:
                sizes.add(image.size)
    assert len(sizes) >= 2, "Smoke requires different image sizes"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "platform": platform.platform(), "wd_python": sys.version,
              "score_rtol": 1e-4, "score_atol": 1e-5, "runs": []}
    report_path = args.output / "smoke.json"
    if report_path.exists():
        raise FileExistsError("Choose a fresh output folder for the smoke report")
    try:
        with tempfile.TemporaryDirectory(dir=args.output) as scratch:
            config_path = Path(scratch) / "sources.json"
            for mode in ("mask", "bbox"):
                reference = None
                for batch_size in (1, 2):
                    configured = copy.deepcopy(data)
                    for source in configured["sources"]:
                        source["view_mode"] = mode
                        source["padding"] = {"unit": "fraction", "value": .15}
                    config_path.write_text(json_text(configured), encoding="utf-8")
                    started = time.monotonic()
                    result = BatchTagging(ScriptOptions(args.images, sam_config=config_path,
                        sam_python=args.sam_python, sam_batch_size=batch_size,
                        region_batch_size=1 if batch_size == 1 else 3, regions_output_dir=args.output,
                        recursive=True, save_txt=False, output_file=None))
                    assert result and len(result[0]) >= 2
                    assert any(r for detail in result[0] for regions in detail["detectors"].values() for r in regions
                               if "mask_path" in r["geometry"]), "No masks found; supply suitable prompts/images"
                    root = Path(result[0][0]["export_root"])
                    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
                    if not manifest["sam"]["device"].startswith("cuda"):
                        raise RuntimeError("GPU acceptance requires SAM on CUDA; configure model.device=cuda")
                    records = [load_region_record(root, detail["image_id"]) for detail in result[0]]
                    report["runs"].append({"mode": mode, "batch_size": batch_size,
                        "seconds": time.monotonic()-started, "root": str(root),
                        "sam": manifest["sam"], "wd": manifest["settings"]["wd"],
                        "metrics": [record["diagnostics"] for record in records]})
                    if reference is not None:
                        compare(reference, result)
                    reference = result
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_path.write_text(json_text(report), encoding="utf-8")


if __name__ == "__main__":
    main()
