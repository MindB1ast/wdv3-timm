"""Portable pipeline presets: settings only, never model objects or image data."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .Scripts.configs import DetectorConfig
from .Scripts.sam_config import finite_number, positive_integer


DEFAULT_SAM_MODEL = {"model": "unsloth/sam3.1", "device": "auto", "download_missing": True,
                     "local_model_path": ""}


def unique_names(names):
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Source names must be nonempty text")
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique across YOLO and SAM")


def combine_detectors(groups):
    configs, models = [], {}
    for group in groups:
        rows, loaded = group
        configs.extend(rows)
        for path, model in loaded.items():
            models.setdefault(path, model)
    unique_names([config.name for config in configs])
    return configs, models


def combine_sources(groups):
    sources = [deepcopy(source) for group in groups for source in group]
    unique_names([source["name"] for source in sources])
    return sources


def validate_config(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("WDV3 config requires schema_version=1")
    data = deepcopy(data)
    options = data.setdefault("options", {})
    allowed = {"general_threshold", "character_threshold", "image_batch_size", "region_batch_size",
               "replace_underscores", "escape_parentheses", "remove_tags", "prefix_tags", "suffix_tags"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unknown tagging settings in config")
    for name, value in options.items():
        if name.endswith("threshold"):
            finite_number(value, name, maximum=1)
        elif name.endswith("batch_size"):
            positive_integer(value, name)
        elif name in ("replace_underscores", "escape_parentheses"):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        elif not isinstance(value, str):
            raise ValueError(f"{name} must be text")
    detectors = data.setdefault("detectors", [])
    sources = data.setdefault("sam_sources", [])
    if not isinstance(detectors, list) or not isinstance(sources, list):
        raise ValueError("detectors and sam_sources must be arrays")
    names = []
    for row in detectors:
        config = DetectorConfig(**row)
        names.append(config.name)
        if not isinstance(config.model_path, str) or not config.model_path.strip():
            raise ValueError("YOLO model_path must be nonempty")
        finite_number(config.confidence, "YOLO confidence", maximum=1)
        classes = row.get("classes")
        if classes is not None and (not isinstance(classes, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in classes)):
            raise ValueError("YOLO classes must be nonnegative integers or null")
        validate_rules(row)
    for source in sources:
        names.append(source["name"])
        if not isinstance(source["prompt"], str) or not source["prompt"].strip():
            raise ValueError("SAM prompt must be nonempty")
        for name in ("confidence", "mask_threshold"):
            finite_number(source[name], name, maximum=1)
        for name in ("min_mask_area_px", "max_regions"):
            positive_integer(source[name], name)
        padding = source["padding_px"]
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding_px must be a nonnegative integer")
        if source["view_mode"] not in ("bbox", "mask"):
            raise ValueError("SAM view_mode must be bbox or mask")
        validate_rules(source.get("rules", {}))
    unique_names(names)
    model = data.get("sam_model")
    if model is None and sources:
        model = deepcopy(DEFAULT_SAM_MODEL)
    if model is not None:
        if not isinstance(model, dict) or set(model) - DEFAULT_SAM_MODEL.keys():
            raise ValueError("Invalid SAM model settings")
        model = {**DEFAULT_SAM_MODEL, **model}
        if model["model"] != "unsloth/sam3.1" or model["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("Unsupported SAM model or device")
        if not isinstance(model["download_missing"], bool) or not isinstance(model["local_model_path"], str):
            raise ValueError("Invalid SAM download setting or local path")
    data["sam_model"] = model
    return data


def validate_rules(rules):
    if not isinstance(rules, dict):
        raise ValueError("Region rules must be an object")
    for name in ("region_gen_threshold", "region_char_threshold"):
        if rules.get(name) is not None:
            finite_number(rules[name], name, maximum=1)
    for name in ("remove_tags_from_full", "remove_tags_from_region", "exclude_from_region"):
        value = rules.get(name)
        if value is not None and (not isinstance(value, list) or any(not isinstance(tag, str) for tag in value)):
            raise ValueError(f"{name} must be a list of tags")
    for name, value in (rules.get("add_tags_to_region") or {}).items():
        if not isinstance(name, str):
            raise ValueError("Tag name must be text")
        finite_number(value, "Assigned tag score", maximum=1)


def config_data(options=None, detectors=None, sources=None, sam_model=None):
    configs = detectors[0] if detectors is not None else []
    return validate_config({"schema_version": 1, "options": options or {},
                            "detectors": [asdict(config) for config in configs],
                            "sam_sources": sources or [],
                            "sam_model": sam_model.get("loader_options") if sam_model is not None else None})


def config_values(data):
    data = validate_config(data)
    configs = []
    for row in data["detectors"]:
        config = DetectorConfig(**row)
        config.classes = row.get("classes")
        configs.append(config)
    return data["options"], (configs, {}), data["sam_sources"], data["sam_model"]


def save_config(directory, name, data, overwrite=False):
    name = name.strip()
    if not name or len(name) > 80 or not name[0].isalnum() or any(
        not (c.isalnum() or c in " _-") for c in name):
        raise ValueError("Preset name: letters, digits, spaces, _ and - only")
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]}:
        raise ValueError("This preset name is reserved on Windows")
    data = validate_config(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".json")
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path
