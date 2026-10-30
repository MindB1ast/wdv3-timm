"""Opt-in SAM folder orchestration with per-image commits and bounded pixels."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile

import torch
from PIL import Image

from .configs import DetectorConfig, ScriptOptions
from .detection import load_detectors_config
from .region_export import RegionRunExport, image_identifier, json_text
from .region_tagging import RegionTagger
from .regions import canonical_image, prepare_region_view, read_mask, rectangle_geometry, resolve_asset, sha256_file
from .sam3_adapter import SamWorker
from .sam_config import RegionViewConfig, load_sam_config, sam_python_executable, validate_batch_options
from .types import BatchResult, DetailedResult


def _region_inputs(frame, geometries, view, masks_root):
    for geometry in geometries:
        mask = (read_mask(resolve_asset(masks_root, geometry["mask_path"]), frame.size)
                if "mask_path" in geometry else None)
        crop, metadata = prepare_region_view(frame, geometry, view, mask)
        yield crop, geometry, metadata


def _yolo_geometries(frame, image_id, detector, model, device):
    results = model(frame, conf=detector.confidence, classes=detector.classes, device=str(device))
    geometries = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                geometries.append(rectangle_geometry(image_id, detector.name, len(geometries),
                                                      frame.size, box.xyxy[0].tolist()))
    return geometries


def run_sam_batch(options: ScriptOptions, *, cancelled=None) -> BatchResult | None:
    # Kept lazy so importing Scripts/legacy notebooks does not start a worker.
    from .main import _load_batch_resources, _postprocess_result, _report_image, _save_batch_results
    validate_batch_options(options)
    assert options.sam_config is not None
    yolo_configs = load_detectors_config(options.detectors_config)
    config = load_sam_config(options.sam_config, [detector.name for detector in yolo_configs])
    folder = Path(options.ImageFolder).resolve()
    output_root = Path(options.regions_output_dir).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    if output_root == folder:
        raise ValueError("regions_output_dir must differ from the input folder")
    output_is_inside = output_root.is_relative_to(folder)
    search = folder.rglob("*") if options.recursive else folder.glob("*")
    files = sorted((p for p in search if p.is_file()
                    and (not output_is_inside or not p.resolve().is_relative_to(output_root))
                    and p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                   key=lambda p: p.relative_to(folder).as_posix())
    if not files:
        return None
    cancelled = cancelled or (lambda: False)
    detailed_results, merged_results = [], []
    with SamWorker(sam_python_executable(options), config, options.sam_timeout, cancelled=cancelled) as worker:
        print(f"SAM backend={worker.provenance.get('backend')}, device={worker.provenance.get('device')}")
        resources = _load_batch_resources(options)
        yolo_rules = list(resources.detectors)
        resources.detectors.extend(source.rules for source in config.sources)
        resources.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tagger = RegionTagger(resources.model, resources.transform, resources.labels, resources.device,
                              options.gen_threshold, options.char_threshold)
        wd_identity = tagger.identity()
        export = RegionRunExport(options.regions_output_dir, config, worker.provenance,
                                 {"wd": wd_identity, "options": asdict(options)})
        print(f"SAM export: {export.root}")
        try:
            for start in range(0, len(files), options.sam_batch_size):
                if cancelled():
                    raise KeyboardInterrupt("SAM batch cancelled")
                with tempfile.TemporaryDirectory(prefix="portion-", dir=export.root / ".staging") as portion:
                    directory = Path(portion)
                    active = []
                    orientation = {}
                    for path in files[start:start+options.sam_batch_size]:
                        relative = path.relative_to(folder).as_posix()
                        # Read errors are per-image; model/protocol/storage errors remain fatal.
                        try:
                            digest = sha256_file(path)
                            frame, transform = canonical_image(path)
                        except (OSError, ValueError) as error:
                            failed_id = image_identifier(relative, "unreadable")
                            export.status(failed_id, "error", source=relative, reason=f"{type(error).__name__}: {error}")
                            print(f"SAM read error: {relative}: {error}")
                            continue
                        image_id = image_identifier(relative, digest)
                        canonical_path = directory / f"{image_id}.png"
                        with frame:
                            frame.save(canonical_path)
                            active.append({"image_id": image_id, "path": str(canonical_path),
                                           "size": list(frame.size), "source_path": str(path),
                                           "source_sha256": digest})
                        orientation[image_id] = transform
                        export.status(image_id, "read", source=relative, source_sha256=digest)
                    if not active:
                        continue
                    resources.model.to("cpu")
                    for model in resources.yolo_models.values():
                        model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    records = worker.segment(active, directory)
                    worker.offload()
                    for image in active:
                        export.status(image["image_id"], "segmented", source=image["source_path"])
                    print(f"SAM segmented {min(start+len(active), len(files))}/{len(files)}; saved {len(detailed_results)}")
                    resources.model.to(resources.device)
                    if resources.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(resources.device)
                    for image in active:
                        if cancelled():
                            raise KeyboardInterrupt("SAM batch cancelled")
                        image_id = image["image_id"]
                        with Image.open(image["path"]) as frame:
                            full = tagger.tag_batch([frame], DetectorConfig("full", "wd"))[0]
                            result: DetailedResult = {"image_path": image["source_path"], "image_id": image_id,
                                "full_image": full, "detectors": {}}
                            diagnostics = {"sam_batch_metrics": worker.last_metrics}
                            for source in config.sources:
                                record = next(r for r in records if r["image_id"] == image_id
                                              and r["source_id"] == source.source_id)
                                result["detectors"][source.name] = tagger.tag_regions(
                                    _region_inputs(frame, record["regions"], source.view, directory),
                                    source.rules, options.region_batch_size)
                                diagnostics[source.name] = record["diagnostics"]
                            for detector in yolo_rules:
                                model = resources.yolo_models.get(detector.model_path)
                                if model is None:
                                    continue
                                geometries = _yolo_geometries(frame, image_id, detector, model, resources.device)
                                result["detectors"][detector.name] = tagger.tag_regions(
                                    _region_inputs(frame, geometries, RegionViewConfig(), directory),
                                    detector, options.region_batch_size)
                        result["tagging_fingerprint"] = hashlib.sha256(json_text([
                            image["source_sha256"], wd_identity, [asdict(rule) for rule in resources.detectors],
                            [r.get("view") for regions in result["detectors"].values() for r in regions],
                        ]).encode()).hexdigest()
                        export.status(image_id, "tagged", source=image["source_path"])
                        if resources.device.type == "cuda":
                            diagnostics["wd_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(resources.device)
                        detailed, merged = _postprocess_result(result, resources, options)
                        export.commit(image_id, image["path"], detailed, merged, directory,
                                      diagnostics, {**orientation[image_id], "source_sha256": image["source_sha256"]})
                        detailed_results.append(detailed)
                        merged_results.append(merged)
                        _report_image(detailed, merged, options)
                        print(f"SAM saved {len(detailed_results)}/{len(files)}")
            _save_batch_results(detailed_results, merged_results, options.output_file)
            has_errors = any(row["status"] == "error" for row in export.manifest["images"].values())
            export.finish("complete_with_errors" if has_errors else "complete")
        except BaseException as error:
            export.finish("cancelled" if isinstance(error, KeyboardInterrupt) else "error",
                          f"{type(error).__name__}: {error}")
            raise
        finally:
            resources.model.to("cpu")
            for model in resources.yolo_models.values():
                model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return detailed_results, merged_results
