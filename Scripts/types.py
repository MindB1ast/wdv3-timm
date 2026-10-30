"""Static contracts for existing dictionaries; no runtime conversion or validation."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Literal, NotRequired, TypeAlias, TypedDict

import numpy as np
from PIL import Image
from torch import Tensor

PathInput: TypeAlias = str | Path
Score: TypeAlias = float | np.float16 | np.float32 | np.float64
Scores: TypeAlias = dict[str, Score]
TagInput: TypeAlias = str | Iterable[str] | None
Category: TypeAlias = Literal["general", "character"]
CATEGORIES: tuple[Category, Category] = ("general", "character")
ImageTransform: TypeAlias = Callable[[Image.Image], Tensor]
Crop: TypeAlias = tuple[Image.Image, list[float], str]
TagSets: TypeAlias = dict[str, set[str]]
TagSamples: TypeAlias = Mapping[str, TagInput]


class ModelProvenance(TypedDict):
    """Existing model evidence; source may name a full frame or region."""

    source: str
    model_score: Score


class ManualProvenance(TypedDict):
    """Existing manually assigned score without a model prediction."""

    source: str
    assigned_score: Score


Provenance: TypeAlias = dict[str, ModelProvenance | ManualProvenance]


class WDTagResult(TypedDict):
    """Authoritative score maps and their existing textual representations."""

    caption: str
    taglist: str
    general: Scores
    character: Scores
    ratings: Scores
    provenance: NotRequired[Provenance]


class RegionResult(WDTagResult):
    """Detector crop; bbox is [x1, y1, x2, y2] in original image pixels."""

    region_id: int
    bbox: list[float]
    geometry: NotRequired["RegionGeometry"]
    view: NotRequired["RegionViewMetadata"]


class RegionGeometry(TypedDict):
    """Canonical half-open geometry; paths are relative to the run export."""

    image_id: str
    source_id: str
    region_id: int
    image_size: list[int]
    bbox: list[int]
    mask_path: NotRequired[str]
    mask_sha256: NotRequired[str]
    segmentation_score: NotRequired[float]
    prompt: NotRequired[str]
    model_bbox: NotRequired[list[float]]


class RegionViewMetadata(TypedDict):
    """Pixel selection before the independent WD transform."""

    mode: Literal["mask", "bbox"]
    padding_unit: Literal["px", "fraction"]
    padding_value: float
    padding_px: list[int]
    crop_bbox: list[int]
    offset: list[int]
    background_rgb: list[int]
    transform_version: int
    fingerprint: str


class DetailedResult(TypedDict):
    """Full frame and independent regional evidence."""

    image_path: PathInput
    full_image: WDTagResult
    detectors: dict[str, list[RegionResult]]
    image_id: NotRequired[str]
    export_root: NotRequired[str]
    canonical_path: NotRequired[str]
    tagging_fingerprint: NotRequired[str]


class MergedResult(TypedDict):
    """Merged tags, optionally accompanied by a saved TXT path."""

    image_path: PathInput
    merged_tags: WDTagResult
    txt_path: NotRequired[str]


BatchResult: TypeAlias = tuple[list[DetailedResult], list[MergedResult]]


class SampleMetrics(TypedDict):
    """Integer evidence counts and unrounded sample metrics."""

    TP: int
    FP: int
    FN: int
    precision: float
    recall: float
    f1_score: float


Metrics = TypedDict("Metrics", {"Precision": float, "Recall": float, "F1-score": float})


class MetricDiagnostics(TypedDict):
    """Optional coverage, evidence counts and explicit evaluation policy."""

    expected: int
    processed: int
    coverage: float
    missing_ids: list[str]
    TP: int
    FP: int
    FN: int
    F1_definition: str
    metric_policy_version: str
    epsilon: float
