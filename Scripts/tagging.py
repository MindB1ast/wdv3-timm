from .wdv3_timm import pil_ensure_rgb, pil_pad_square, get_tags, LabelData
from .configs import DetectorConfig
from PIL import Image
from torch import Tensor, nn
from typing import Callable, Dict, List, Any
import torch
from ultralytics import YOLO
from torch.nn import functional as F
from .detection import extract_regions_with_detector
from .preprocess import process_tags


def tag_image(img: Image.Image, model: nn.Module, transform: Callable, labels: LabelData,
             gen_threshold: float, char_threshold: float, torch_device: torch.device) -> Dict:
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
    # Обработка изображения
    img = pil_ensure_rgb(img)
    img_padded = pil_pad_square(img)
    input_tensor = transform(img_padded)
    input_tensor = input_tensor[[2, 1, 0], :, :]  # RGB -> BGR

    # Инференс
    with torch.inference_mode():
        input_batch = input_tensor.unsqueeze(0).to(torch_device)
        output = model(input_batch)
        output = F.sigmoid(output).to("cpu")

    # Получаем теги
    caption, taglist, ratings, character, general = get_tags(
        probs=output[0],
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )

    return {
        "caption": caption,
        "taglist": taglist,
        "ratings": ratings,
        "character": character,
        "general": general,
    }


def process_image_with_multiple_detectors(
    img_path: str,
    tagger_model: nn.Module,
    transform: Callable,
    labels: LabelData,
    detectors: List[DetectorConfig],
    yolo_models: Dict[str, YOLO],
    gen_threshold: float,  # Из ScriptOptions
    char_threshold: float,  # Из ScriptOptions
    torch_device: torch.device
) -> Dict:
    """
    Обрабатывает изображение с помощью нескольких детекторов и теггера.

    Args:
        img_path: Путь к изображению
        tagger_model: Модель тегирования
        transform: Трансформация для изображения
        labels: Данные о тегах
        detectors: Список конфигураций детекторов
        yolo_models: Словарь с загруженными моделями YOLO
        gen_threshold: Порог для общих тегов
        char_threshold: Порог для тегов персонажей
        torch_device: Устройство PyTorch

    Returns:
        Dict: Результаты тегирования
    """
    # Загружаем изображение
    img = Image.open(img_path).convert('RGB')

    # Тегируем полное изображение
    full_image_tags = tag_image(
        img, tagger_model, transform, labels,
        gen_threshold, char_threshold, torch_device
    )

    # Собираем все теги из регионов
    all_detector_results = {}

    # Обрабатываем каждый детектор
    for detector in detectors:
        # Получаем модель YOLO для этого детектора
        yolo_model = yolo_models.get(detector.model_path)
        if not yolo_model:
            print(f"Модель YOLO для детектора {detector.name} (путь: {detector.model_path}) не найдена")
            continue

        # Используем пороги из DetectorConfig или из ScriptOptions
        region_gen_threshold = detector.region_gen_threshold if detector.region_gen_threshold is not None else gen_threshold
        region_char_threshold = detector.region_char_threshold if detector.region_char_threshold is not None else char_threshold

        # Исправляем порядок параметров: image, yolo_model, detector_config
        regions = extract_regions_with_detector(img, yolo_model, detector)
        detector_results = []

        for region in regions:
            region_img = region['region']
            bbox = region['bbox']
            confidence = region['confidence']

            region_tags = tag_image(
                Image.fromarray(region_img), tagger_model, transform, labels,
                region_gen_threshold, region_char_threshold, torch_device
            )

            # Обрабатываем теги региона
            region_tags["general"] = process_tags(region_tags["general"], detector.remove_tags_from_region)
            region_tags["character"] = process_tags(region_tags["character"], detector.remove_tags_from_region)

            # Добавляем теги из конфигурации
            for tag, conf in detector.add_tags_to_region.items():
                if tag in region_tags["character"] or any(tag == labels.names[i] for i in labels.character):
                    region_tags["character"][tag] = conf
                else:
                    region_tags["general"][tag] = conf

            # Формируем список тегов
            combined_tags = list(region_tags["general"].keys()) + list(region_tags["character"].keys())
            region_tags["taglist"] = ", ".join(combined_tags)

            # Добавляем теги региона с учетом уверенности детектора
            detector_results.append({
                "region_id": len(detector_results),
                "bbox": bbox,
                "confidence": confidence,
                "caption": region_tags["caption"],
                "taglist": region_tags["taglist"],
                "ratings": region_tags["ratings"],
                "character": region_tags["character"],
                "general": region_tags["general"]
            })

        # Добавляем результаты детектора в общий список
        all_detector_results[detector.name] = detector_results

    # Формируем итоговый результат
    result = {
        "image_path": img_path,
        "full_image": full_image_tags,
        "detectors": all_detector_results
    }

    return result