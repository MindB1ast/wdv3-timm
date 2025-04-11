#yolo funcs
import os
import json
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from .configs import DetectorConfig



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
                region_char_threshold=item.get("region_char_threshold")
            )
            detectors.append(detector)
        except KeyError as e:
            print(f"Ошибка в конфигурации детектора: отсутствует обязательное поле {e}")
        except Exception as e:
            print(f"Ошибка обработки конфигурации детектора: {e}")

    return detectors


def extract_regions_with_detector(img_path: str, detector: DetectorConfig, yolo_model: YOLO) -> List[Tuple[Image.Image, List[float], str]]:
    """
    Использует YOLO для детектирования объектов и вырезает квадратные области в исходном разрешении.
    
    Args:
        img_path: Путь к изображению
        detector: Конфигурация детектора
        yolo_model: Загруженная модель YOLO
        
    Returns:
        List of tuples (cropped_image, bbox, detector_name)
    """
    # Загрузка изображения
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # Запускаем YOLO детекцию с параметрами из конфигурации детектора
    results = yolo_model(img_path, conf=detector.confidence, classes=detector.classes)
    
    # Извлекаем обнаруженные области
    regions = []
    
    # Проверяем, есть ли какие-либо результаты
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        # Извлекаем боксы из результатов YOLO
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Получаем координаты бокса
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Преобразуем прямоугольник в квадрат, сохраняя центр обнаруженного объекта
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Определяем размер квадрата по большей стороне
                square_size = max(box_width, box_height)
                
                # Вычисляем центр бокса
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Вычисляем новые координаты для квадратного бокса
                new_x1 = center_x - square_size / 2
                new_y1 = center_y - square_size / 2
                new_x2 = center_x + square_size / 2
                new_y2 = center_y + square_size / 2
                
                # Убеждаемся, что квадрат не выходит за границы изображения
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(img_width, new_x2)
                new_y2 = min(img_height, new_y2)
                
                # Корректируем размер, если квадрат оказался за границами изображения
                current_width = new_x2 - new_x1
                current_height = new_y2 - new_y1
                
                # Если после обрезки по границе изображения получился не квадрат, 
                # подгоняем размер по минимальной стороне
                if current_width != current_height:
                    min_side = min(current_width, current_height)
                    
                    if current_width > min_side:
                        diff = current_width - min_side
                        new_x1 += diff / 2
                        new_x2 -= diff / 2
                    elif current_height > min_side:
                        diff = current_height - min_side
                        new_y1 += diff / 2
                        new_y2 -= diff / 2
                
                # Обрезаем изображение по новым координатам
                cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
                
                # Добавляем обрезанное изображение, координаты и имя детектора в список
                regions.append((cropped_img, [new_x1, new_y1, new_x2, new_y2], detector.name))
    
    return regions