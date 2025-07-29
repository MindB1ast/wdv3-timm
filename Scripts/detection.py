#yolo funcs
import os
import json
from typing import List, Tuple, Union
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


def extract_regions_with_detector(
    img_path: str,
    detector: DetectorConfig,
    yolo_model: YOLO,
    make_square: bool = True,
    padding: Union[int, float] = 0  # int = пиксели, float = доля от размера bbox (например, 0.1 = 10%)
) -> List[Tuple[Image.Image, List[float], str]]:
    from PIL import ImageOps

    img = Image.open(img_path)
    width, height = img.size

    results = yolo_model(img_path, conf=detector.confidence, classes=detector.classes)
    regions = []

    if len(results) > 0 and hasattr(results[0], 'boxes'):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_width = x2 - x1
                box_height = y2 - y1

                # ─── 📌 Добавление padding ─────
                if isinstance(padding, float):
                    pad_x = box_width * padding
                    pad_y = box_height * padding
                else:
                    pad_x = pad_y = padding

                x1_pad = max(0, int(x1 - pad_x))
                y1_pad = max(0, int(y1 - pad_y))
                x2_pad = min(width, int(x2 + pad_x))
                y2_pad = min(height, int(y2 + pad_y))

                cropped_img = img.crop((x1_pad, y1_pad, x2_pad, y2_pad))

                # ─── 📌 Приведение к квадрату ─────
                if make_square:
                    crop_w, crop_h = cropped_img.size
                    side = max(crop_w, crop_h)
                    delta_w = side - crop_w
                    delta_h = side - crop_h
                    padding_tuple = (
                        delta_w // 2, delta_h // 2,
                        delta_w - (delta_w // 2), delta_h - (delta_h // 2)
                    )
                    cropped_img = ImageOps.expand(cropped_img, padding_tuple, fill=(0, 0, 0))  # Чёрный фон

                regions.append((cropped_img, [x1_pad, y1_pad, x2_pad, y2_pad], detector.name))

    return regions