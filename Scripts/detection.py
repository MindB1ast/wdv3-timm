"""Load detector configurations and extract original-resolution regions."""

# yolo funcs
import json
import os
from typing import List

from PIL import Image
from ultralytics import YOLO

from .configs import DetectorConfig
from .types import Crop, PathInput


def load_detectors_config(config_path: PathInput) -> List[DetectorConfig]:
    """Read independent detector settings, reporting malformed entries."""
    if not os.path.exists(config_path):
        print(f"Файл конфигурации детекторов не найден: {config_path}")
        print("YOLO-детекторы не будут использоваться")
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка в формате конфигурационного файла: {e}")
        return []

    if isinstance(config_data, dict) and "sources" in config_data:
        raise ValueError(
            f"{config_path}: это конфигурация SAM 3, а не YOLO. "
            "Передайте её через ScriptOptions(sam_config=...), "
            "а detectors_config оставьте для detectors.json с YOLO-детекторами. "
            "Для SAM также задайте sam_python — Python отдельного SAM-окружения."
        )
    if not isinstance(config_data, list):
        raise ValueError(f"{config_path}: конфигурация YOLO должна быть JSON-массивом детекторов")

    detectors = []
    for item in config_data:
        try:
            detector = DetectorConfig(
                name=item["name"],
                model_path=item["model_path"],
                confidence=item.get("confidence", 0.25),
                classes=item.get("classes", [0]),
                remove_tags_from_full=item.get("remove_tags_from_full", []),
                remove_tags_from_region=item.get("remove_tags_from_region", []),
                add_tags_to_region=item.get("add_tags_to_region", {}),
                exclude_from_region=item.get("exclude_from_region", []),
                # specific_excluded_tags=item.get("specific_excluded_tags", []),
                region_gen_threshold=item.get("region_gen_threshold"),
                region_char_threshold=item.get("region_char_threshold"),
            )
            detectors.append(detector)
        except KeyError as e:
            print(f"Ошибка в конфигурации детектора: отсутствует обязательное поле {e}")
        except (TypeError, ValueError) as e:
            print(f"Ошибка обработки конфигурации детектора: {e}")

    return detectors


def extract_regions_with_detector(
    img_path: PathInput, detector: DetectorConfig, yolo_model: YOLO
) -> list[Crop]:
    """
    Использует YOLO для детектирования объектов и вырезает области в исходном разрешении.

    Args:
        img_path: Путь к изображению
        detector: Конфигурация детектора
        yolo_model: Загруженная модель YOLO

    Returns:
        List of tuples (cropped_image, bbox, detector_name)
    """
    regions = []
    with Image.open(img_path) as image:
        results = yolo_model(img_path, conf=detector.confidence, classes=detector.classes)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                bbox = box.xyxy[0].tolist()
                regions.append((image.crop(tuple(bbox)), bbox, detector.name))
    return regions
