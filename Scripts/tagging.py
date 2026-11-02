"""Run tagging inference on full images and detector crops."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

from .configs import DetectorConfig
from .detection import extract_regions_with_detector
from .region_tagging import RegionTagger
from .sam_config import positive_integer
from .tag_utils import format_result
from .types import (Crop, DetailedResult, ImageTransform, PathInput,
                    RegionResult, WDTagResult)
from .wdv3_timm import LabelData, get_tags
from .tagger_preprocessing import prepare_tensor


# Preserve the existing notebook calling convention.
# pylint: disable-next=R0913,R0917
def tag_image(
    img: Image.Image,
    model: nn.Module,
    transform: ImageTransform,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
    torch_device: torch.device | None,
) -> WDTagResult:
    """
    Тегирует изображение с помощью модели.

    Args:
        img: Изображение для тегирования
        model: Модель для тегирования
        transform: Функция преобразования
        labels: Данные о тегах
        gen_threshold: Порог для общих тегов
        char_threshold: Порог для тегов персонажей
        torch_device: Устройство для вычислений

    Returns:
        Dict с результатами тегирования
    """
    input_tensor = prepare_tensor(img, transform)
    with torch.inference_mode():
        input_batch = input_tensor.unsqueeze(0).to(torch_device)
        output = model(input_batch)
        output = F.sigmoid(output).to("cpu")
    if output.ndim != 2 or output.shape != (1, len(labels.names)):
        raise ValueError('Tagger output shape does not match image/labels')
    tagged = get_tags(
        probs=output[0], labels=labels, gen_threshold=gen_threshold, char_threshold=char_threshold
    )
    return {"caption": tagged[0], "taglist": tagged[1], "ratings": tagged[2],
            "character": tagged[3], "general": tagged[4]}


@dataclass(frozen=True)
class _TaggingRuntime:
    """Shared tagging dependencies and fallback thresholds for regional inference."""

    model: nn.Module
    transform: ImageTransform
    labels: LabelData
    gen_threshold: float
    char_threshold: float
    torch_device: torch.device | None
    region_batch_size: int


def _tag_detector_regions(
    regions: list[Crop], detector: DetectorConfig, runtime: _TaggingRuntime,
) -> list[RegionResult]:
    """Tag detached crops and close them even when inference fails."""
    tagger = RegionTagger(
        runtime.model, runtime.transform, runtime.labels,
        runtime.torch_device or torch.device("cpu"),
        runtime.gen_threshold, runtime.char_threshold,
    )
    results: list[RegionResult] = []
    try:
        for start in range(0, len(regions), runtime.region_batch_size):
            chunk = regions[start:start + runtime.region_batch_size]
            tagged = tagger.tag_batch([image for image, _, _ in chunk], detector)
            for offset, ((_, bbox, _), raw) in enumerate(zip(chunk, tagged)):
                results.append({"region_id": start + offset, "bbox": bbox, **raw})
    finally:
        for image, _, _ in regions:
            image.close()
    return results


def _tag_all_regions(
    img_path: PathInput, runtime: _TaggingRuntime, detectors: Sequence[DetectorConfig],
    yolo_models: Mapping[str, YOLO],
) -> dict[str, list[RegionResult]]:
    """Run each available detector with its own regional thresholds."""
    results: dict[str, list[RegionResult]] = {}
    for detector in detectors:
        yolo_model = yolo_models.get(detector.model_path)
        if yolo_model is None:
            continue
        regions = extract_regions_with_detector(img_path, detector, yolo_model)
        results[detector.name] = _tag_detector_regions(regions, detector, runtime)
    return results


# Preserve the existing notebook calling convention.
# pylint: disable-next=R0913,R0917
def process_image_with_multiple_detectors(
    img_path: PathInput,
    tagger_model: nn.Module,
    transform: ImageTransform,
    labels: LabelData,
    detectors: Sequence[DetectorConfig],
    yolo_models: Mapping[str, YOLO],
    gen_threshold: float,
    char_threshold: float,
    torch_device: torch.device | None,
    *,
    region_batch_size: int = 8,
) -> DetailedResult:
    """Tag the full image and independent crops, always closing owned images."""
    positive_integer(region_batch_size, "region_batch_size")
    runtime = _TaggingRuntime(
        tagger_model, transform, labels, gen_threshold, char_threshold, torch_device,
        region_batch_size,
    )
    with Image.open(img_path) as image:
        full = tag_image(
            image, tagger_model, transform, labels, gen_threshold, char_threshold, torch_device
        )
    return {
        "image_path": img_path,
        "full_image": format_result(full),
        "detectors": _tag_all_regions(img_path, runtime, detectors, yolo_models),
    }
