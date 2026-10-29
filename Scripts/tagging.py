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
from .tag_utils import apply_tag_rules, format_result
from .types import (Crop, DetailedResult, ImageTransform, PathInput,
                    RegionResult, WDTagResult)
from .wdv3_timm import LabelData, get_tags, pil_ensure_rgb, pil_pad_square


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
    img = pil_ensure_rgb(img)
    img_padded = pil_pad_square(img)
    input_tensor = transform(img_padded)
    input_tensor = input_tensor[[2, 1, 0], :, :]
    with torch.inference_mode():
        input_batch = input_tensor.unsqueeze(0).to(torch_device)
        output = model(input_batch)
        output = F.sigmoid(output).to("cpu")
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


def _tag_detector_regions(
    regions: list[Crop], detector: DetectorConfig, runtime: _TaggingRuntime,
) -> list[RegionResult]:
    """Tag detached crops and close them even when inference fails."""
    gen_threshold = (
        detector.region_gen_threshold
        if detector.region_gen_threshold is not None
        else runtime.gen_threshold
    )
    char_threshold = (
        detector.region_char_threshold
        if detector.region_char_threshold is not None
        else runtime.char_threshold
    )
    results: list[RegionResult] = []
    character_names = [runtime.labels.names[index] for index in runtime.labels.character]
    try:
        for index, (image, bbox, _) in enumerate(regions):
            raw = tag_image(
                image,
                runtime.model,
                runtime.transform,
                runtime.labels,
                gen_threshold,
                char_threshold,
                runtime.torch_device,
            )
            filtered = apply_tag_rules(
                raw,
                remove=detector.remove_tags_from_region,
                add=detector.add_tags_to_region,
                character_names=character_names,
            )
            results.append({"region_id": index, "bbox": bbox, **filtered})
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
) -> DetailedResult:
    """Tag the full image and independent crops, always closing owned images."""
    runtime = _TaggingRuntime(
        tagger_model, transform, labels, gen_threshold, char_threshold, torch_device
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
