#yolo funcs
import os
import json
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from .configs import DetectorConfig
import numpy as np



def load_detectors_config(config_path: str) -> List[DetectorConfig]:
    if not os.path.exists(config_path):
        print(f"Файл конфигурации детекторов не найден: {config_path}")
        print("YOLO-детекторы не будут использоваться")
        return []

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка в формате конфигурационного файла: {e}")
        return []

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
                #specific_excluded_tags=item.get("specific_excluded_tags", []),
                region_gen_threshold=item.get("region_gen_threshold"),
                region_char_threshold=item.get("region_char_threshold"),
                use_min_side=item.get("use_min_side", False)
            )
            detectors.append(detector)
        except KeyError as e:
            print(f"Ошибка в конфигурации детектора: отсутствует обязательное поле {e}")
        except Exception as e:
            print(f"Ошибка обработки конфигурации детектора: {e}")

    return detectors


def extract_regions_with_detector(image, detector, detector_config):
    """
    Извлекает регионы из изображения с помощью YOLO детектора.

    Args:
        image: PIL Image или путь к изображению
        detector: YOLO модель
        detector_config: Конфигурация детектора

    Returns:
        List[Dict]: Список словарей с информацией о регионах
    """
    # Убеждаемся, что у нас есть объект PIL.Image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise TypeError("image должен быть путем к файлу или объектом PIL.Image")

    # Получаем предсказания
    # Для моделей сегментации используем только параметр classes
    if hasattr(detector, 'model') and hasattr(detector.model, 'task') and detector.model.task == 'segment':
        results = detector(image, classes=detector_config.classes)[0]
    else:
        # Для обычных моделей детекции используем все параметры
        results = detector(image, conf=detector_config.confidence, classes=detector_config.classes)[0]

    print(f"Детектор {detector_config.name}: найдено {len(results.boxes)} объектов")
    if len(results.boxes) == 0:
        print(f"Параметры детекции: confidence={detector_config.confidence}, classes={detector_config.classes}")

    regions = []
    for box in results.boxes:
        # Получаем координаты бокса
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Получаем уверенность и класс
        if hasattr(box, 'conf'):
            confidence = float(box.conf[0])
        else:
            # Для моделей сегментации используем значение из конфига
            confidence = detector_config.confidence

        if hasattr(box, 'cls'):
            class_id = int(box.cls[0])
        else:
            class_id = 0  # Для моделей сегментации используем класс по умолчанию

        # Определяем размер квадрата для кропа
        width = x2 - x1
        height = y2 - y1
        if detector_config.use_min_side:
            size = min(width, height)
        else:
            size = max(width, height)

        # Вычисляем центр бокса
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Вычисляем координаты квадратного региона
        half_size = size / 2
        square_x1 = max(0, center_x - half_size)
        square_y1 = max(0, center_y - half_size)
        square_x2 = min(image.width, center_x + half_size)
        square_y2 = min(image.height, center_y + half_size)

        # Корректируем размер, если вышли за границы
        if square_x1 == 0:
            square_x2 = min(image.width, size)
        if square_y1 == 0:
            square_y2 = min(image.height, size)
        if square_x2 == image.width:
            square_x1 = max(0, image.width - size)
        if square_y2 == image.height:
            square_y1 = max(0, image.height - size)

        # Вырезаем регион
        region = image.crop((square_x1, square_y1, square_x2, square_y2))

        regions.append({
            'region': np.array(region),
            'confidence': confidence,
            'class_id': class_id,
            'bbox': [x1, y1, x2, y2]
        })

    return regions