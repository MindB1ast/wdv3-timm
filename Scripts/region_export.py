"""Per-image commit and portable lossless assets for SAM runs."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from .io_utils import atomic_write_text
from .regions import prepare_saved_region, resolve_asset, sha256_file
from .sam_config import RegionViewConfig, SamConfig
from .types import DetailedResult, MergedResult, PathInput


def json_text(data) -> str:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Not serializable: {type(value).__name__}")
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=convert)


def image_identifier(relative_path: str, content_hash: str) -> str:
    return hashlib.sha256((relative_path + "\0" + content_hash).encode("utf-8")).hexdigest()[:24]


class RegionRunExport:
    def __init__(self, directory: PathInput, config: SamConfig, provenance: dict, settings: dict):
        self.root = Path(directory).resolve() / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        (self.root / "images").mkdir()
        (self.root / ".staging").mkdir()
        self.manifest = {"schema_version": 1, "run_id": self.root.name, "status": "running",
                         "sam": provenance, "settings": settings, "sources": asdict(config),
                         "images": {}}
        self._write(self.manifest)

    def _write(self, manifest):
        text = json_text(manifest)
        for attempt in range(5):
            try:
                atomic_write_text(self.root / "manifest.json", text)
                return
            except PermissionError as error:
                # Windows scanners/readers can briefly deny replacement even
                # after our stream is closed. Never weaken an eventual failure.
                if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 4:
                    raise
                time.sleep(.05 * (attempt+1))

    def status(self, image_id: str, state: str, **fields):
        updated = deepcopy(self.manifest)
        updated["images"][image_id] = {"status": state, **fields}
        self._write(updated)
        self.manifest = updated

    def commit(self, image_id: str, canonical: PathInput, detailed: DetailedResult,
               merged: MergedResult, masks_root: PathInput, diagnostics: dict, orientation: dict):
        stage = self.root / ".staging" / uuid.uuid4().hex
        stage.mkdir()
        target = self.root / "images" / image_id
        if target.exists():
            raise ValueError("Image already committed")
        stored = deepcopy(detailed)
        stored.pop("export_root", None)
        prefix = f"images/{image_id}"
        stored["canonical_path"] = f"{prefix}/canonical.png"
        try:
            shutil.copyfile(canonical, stage / "canonical.png")
            mask_index = 0
            for regions in stored["detectors"].values():
                for region in regions:
                    geometry = region.get("geometry")
                    if geometry is None or "mask_path" not in geometry:
                        continue
                    filename = f"mask-{mask_index}.png"
                    shutil.copyfile(resolve_asset(masks_root, geometry["mask_path"]), stage / filename)
                    geometry["mask_path"] = f"{prefix}/{filename}"
                    mask_index += 1
            record = {"schema_version": 1, "detailed": stored, "merged": merged,
                      "orientation": orientation, "diagnostics": diagnostics,
                      "canonical_sha256": sha256_file(stage / "canonical.png")}
            atomic_write_text(stage / "record.json", json_text(record))
            os.replace(stage, target)
            self.status(image_id, "complete", record=f"{prefix}/record.json")
        finally:
            if stage.exists():
                # Only our newly allocated staging directory, never a caller-supplied path.
                if not stage.resolve().is_relative_to((self.root / ".staging").resolve()):
                    raise ValueError("Invalid staging directory")
                shutil.rmtree(stage)
        detailed.update(stored)
        detailed["export_root"] = str(self.root)

    def finish(self, state="complete", reason=None):
        updated = deepcopy(self.manifest)
        updated["status"] = state
        if reason:
            updated["reason"] = reason
        self._write(updated)
        self.manifest = updated


def load_region_record(root: PathInput, image_id: str) -> dict:
    directory = Path(root).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported region manifest version")
    entry = manifest["images"][image_id]
    if entry["status"] != "complete":
        raise ValueError("Image has not been committed")
    record = json.loads(resolve_asset(directory, entry["record"]).read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported image record version")
    detail = record["detailed"]
    if sha256_file(resolve_asset(directory, detail["canonical_path"])) != record["canonical_sha256"]:
        raise ValueError("Canonical image checksum mismatch")
    detail["export_root"] = str(directory)
    return record


def saved_region_input(root: PathInput, image_id: str, source_name: str,
                       region_id: int, view: RegionViewConfig):
    record = load_region_record(root, image_id)
    detail = record["detailed"]
    region = next(r for r in detail["detectors"][source_name] if r["region_id"] == region_id)
    return prepare_saved_region(root, detail["canonical_path"], region["geometry"], view)
