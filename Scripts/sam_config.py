"""Validated opt-in SAM configuration, independent of the SAM runtime."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Literal

from .configs import DetectorConfig, ScriptOptions
from .types import PathInput


def finite_number(value, name: str, minimum=0.0, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: out of range")
    return value


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}: expected a positive integer")
    return value


@dataclass(frozen=True)
class RegionViewConfig:
    mode: Literal["mask", "bbox"] = "bbox"
    padding_unit: Literal["px", "fraction"] = "px"
    padding_value: float = 0.0
    background_rgb: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self):
        if self.mode not in ("mask", "bbox") or self.padding_unit not in ("px", "fraction"):
            raise ValueError("view: invalid mode or padding unit")
        finite_number(self.padding_value, "padding.value")
        if len(self.background_rgb) != 3 or any(
            isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255
            for c in self.background_rgb
        ):
            raise ValueError("background_rgb: expected three integers in 0..255")


@dataclass(frozen=True)
class SamSourceConfig:
    source_id: str
    name: str
    prompt: str
    confidence: float = 0.5
    mask_threshold: float = 0.5
    min_mask_area_px: int = 1
    max_regions: int = 64
    view: RegionViewConfig = field(default_factory=RegionViewConfig)
    rules: DetectorConfig = field(default_factory=lambda: DetectorConfig("sam", "sam3"))


@dataclass(frozen=True)
class SamConfig:
    checkpoint: str
    device: str
    allow_download: bool
    sources: tuple[SamSourceConfig, ...]
    backend: str = "auto"
    model_id: str | None = None
    revision: str | None = None


def validate_batch_options(options: ScriptOptions) -> None:
    positive_integer(options.sam_batch_size, "sam_batch_size")
    positive_integer(options.region_batch_size, "region_batch_size")
    if finite_number(options.sam_timeout, "sam_timeout") == 0:
        raise ValueError("sam_timeout must be positive")


def sam_python_executable(options: ScriptOptions) -> str:
    """Use the running notebook/CLI interpreter unless explicitly overridden."""
    return str(options.sam_python) if options.sam_python is not None else sys.executable


def load_sam_config(path: PathInput, detector_names=()) -> SamConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("SAM config requires schema_version=1")
    model = data.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    model_id = model.get("model_id")
    revision = model.get("revision")
    for name, value in [("model_id", model_id), ("revision", revision)]:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"model.{name} must be nonempty text")
    checkpoint = model.get("checkpoint", "")
    if not isinstance(checkpoint, str) or (not checkpoint.strip() and model_id is None):
        raise ValueError("model.checkpoint or model.model_id is required")
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = config_path.parent / checkpoint_path
    device = model.get("device", "auto")
    if not isinstance(device, str) or not (device in ("auto", "cpu", "cuda") or (
        device.startswith("cuda:") and device[5:].isdigit()
    )):
        raise ValueError("device must be auto, cpu, cuda or cuda:N")
    backend = model.get("backend", "auto")
    if backend not in ("auto", "native", "transformers"):
        raise ValueError("backend must be auto, native or transformers")
    allow_download = model.get("allow_download", False)
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be boolean")
    entries = data.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sources must be a nonempty list")
    names, ids, sources = set(detector_names), set(), []
    rule_fields = (
        "remove_tags_from_full", "remove_tags_from_region", "add_tags_to_region",
        "exclude_from_region", "region_gen_threshold", "region_char_threshold",
    )
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Each SAM source must be an object")
        for key in ("source_id", "name", "prompt"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"source.{key} must be nonempty text")
        if item.get("backend", "sam3") != "sam3":
            raise ValueError("Only backend=sam3 is supported")
        if item["name"] in names or item["source_id"] in ids:
            raise ValueError("Duplicate SAM source_id/name or conflict with YOLO name")
        names.add(item["name"])
        ids.add(item["source_id"])
        padding = item.get("padding", {})
        if not isinstance(padding, dict):
            raise ValueError("padding must be an object")
        background = item.get("background_rgb", [255, 255, 255])
        if not isinstance(background, list) or len(background) != 3:
            raise ValueError("background_rgb requires three integers")
        view = RegionViewConfig(
            item.get("view_mode", "bbox"), padding.get("unit", "px"),
            finite_number(padding.get("value", 0), "padding.value"),
            (background[0], background[1], background[2]),
        )
        rules_data = {key: item[key] for key in rule_fields if key in item}
        for key in ("region_gen_threshold", "region_char_threshold"):
            if key in rules_data and rules_data[key] is not None:
                finite_number(rules_data[key], key, maximum=1)
        for key in ("remove_tags_from_full", "remove_tags_from_region", "exclude_from_region"):
            if key in rules_data and (not isinstance(rules_data[key], list) or
                                     not all(isinstance(t, str) for t in rules_data[key])):
                raise ValueError(f"{key}: expected list of tags")
        if "add_tags_to_region" in rules_data:
            if not isinstance(rules_data["add_tags_to_region"], dict):
                raise ValueError("add_tags_to_region must be an object")
            for tag, score in rules_data["add_tags_to_region"].items():
                if not isinstance(tag, str):
                    raise ValueError("Tag names must be strings")
                finite_number(score, "assigned score", maximum=1)
        sources.append(SamSourceConfig(
            item["source_id"], item["name"], item["prompt"],
            finite_number(item.get("confidence", .5), "confidence", maximum=1),
            finite_number(item.get("mask_threshold", .5), "mask_threshold", maximum=1),
            positive_integer(item.get("min_mask_area_px", 1), "min_mask_area_px"),
            positive_integer(item.get("max_regions", 64), "max_regions"), view,
            DetectorConfig(item["name"], "sam3", **rules_data),
        ))
    resolved = str(checkpoint_path.resolve()) if checkpoint_path is not None else ""
    return SamConfig(resolved, device, allow_download, tuple(sources), backend, model_id, revision)
