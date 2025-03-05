from typing import Dict, List, Set
import json
from .configs import DetectorConfig
import numpy as np


def process_tags(tags_dict: Dict[str, float], tags_to_remove: List[str]) -> Dict[str, float]:
    """
    Удаляет указанные теги из словаря тегов.
    
    Args:
        tags_dict: Словарь тегов и их уверенностей
        tags_to_remove: Список тегов для удаления
        
    Returns:
        Обновленный словарь тегов
    """
    return {tag: conf for tag, conf in tags_dict.items() if tag not in tags_to_remove}


def merge_tags_from_regions(image_result: Dict, detectors: List[DetectorConfig]) -> Dict:
    # Теги из полной картинки
    full_general = image_result["full_image"]["general"]
    full_character = image_result["full_image"]["character"]
    full_tags_set = set(full_general.keys()) | set(full_character.keys())
    
    # Глобальный список запрещённых тегов
    # Удаляет теги которые появились в обнаруженой области если они есть
    #Для артефактов приближения определяя данные теги из полного изображения
    GLOBAL_EXCLUDED_REGION_TAGS = {"close-up", "cowboy_shot", "portrait", "upper_body", "full_body", "eye_focus", "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "solo", "solo_focus", "1boy",
    "2boys", "3boys", "4boys", "5boys", "6+boys", "male_focus", "other_focus",  "1other", "2others",
    "3others","4others", "5others", "6+others", "ass_focus",  "animal_focus", "foot_focus", "text_focus",
    "hip_focus", "vehicle_focus", "food_focus", "mecha_focus", "pokemon_focus", "reflection_focus", 
    "breast_focus", "back_focus", "hand_focus", "object_focus", "crotch_focus", "pectoral_focus",
    "weapon_focus","armpit_focus", "monster_focus", "pussy_focus", "creature_focus", "flower_focus", 
    "hair_focus","footwear_focus", "navel_focus", "sky_focus", "plant_focus", "thigh_focus", "mouth_focus", 
    "leg_focus",
    "ear_focus", "kanji_focus", "neck_focus", "penis_focus", "simple_background", "white_background",
    "blue_background", "aqua_background", "black_background", "brown_background", "green_background", 
    "grey_background", "orange_background", "pink_background", "purple_background", "red_background",
    "colorful_background", "gradient_background", "halftone_background", "monochrome_background", 
     "rainbow_background", "heaven_condition", "two-tone_background", "argyle_background",
     "checkered_background", "cross_background", "dithered_background", "sky", "blue_sky", "night", 
     "night_sky", "dotted_background", "grid_background", "honeycomb_background", "lace_background",
     "marble_background", "mosaic_background", "patterned_background", "plaid_background",
     "polka_dot_background", "spiral_background", "splatter_background", "striped_background", 
     "diagonal-striped_background", "sunburst_background", "triangle_background", "abstract_background", 
     "blurry_background", "bright_background", "dark_background", "drama_layer"}
    
    # Инициализируем результирующие теги
    merged_general = {}
    merged_character = full_character.copy()  # Пока из полной картинки
    merged_tags = []  # Единый список тегов для caption и taglist
    
    # Initialize exclude_from_full as an empty set outside of the loop
    exclude_from_full = set()

    # Проходим по всем детекторам и их областям
    for detector_name, regions in image_result["detectors"].items():
        detector_config = next((d for d in detectors if d.name == detector_name), None)
        
        if detector_config is None:
            continue
            
        # Теги для исключения из полной картинки
        exclude_from_full = set(detector_config.remove_tags_from_full)# | set(detector_config.specific_excluded_tags)
        
        for region in regions:
            region_general = region["general"]
            region_character = region["character"]
            
            # Фильтруем теги области
            filtered_general = {}
            for tag, conf in region_general.items():
                if tag in detector_config.remove_tags_from_region:
                    continue
                if (tag in detector_config.exclude_from_region or tag in GLOBAL_EXCLUDED_REGION_TAGS) and tag not in full_tags_set:
                    continue
                filtered_general[tag] = conf
            
            # Обновляем merged_general
            for tag, conf in filtered_general.items():
                if tag not in merged_general or conf > merged_general[tag]:
                    merged_general[tag] = conf
            
            # Обновляем merged_character
            for tag, conf in region_character.items():
                if tag not in merged_character or conf > merged_character[tag]:
                    merged_character[tag] = conf
            
            # Собираем теги из области в единый список
            region_tags = list(filtered_general.keys()) + list(region_character.keys())
            merged_tags.extend(region_tags)
    
    # Добавляем теги из полной картинки, исключая указанные
    for tag, conf in full_general.items():
        if tag not in exclude_from_full:
            if tag not in merged_general or conf > merged_general[tag]:
                merged_general[tag] = conf
                merged_tags.append(tag)

    # Формируем caption и taglist из одного источника
    unique_merged_tags = sorted(set(merged_tags))  # Убираем дубликаты и сортируем
    merged_caption = ", ".join(unique_merged_tags)  # С подчёркиваниями
    merged_taglist = ", ".join(tag.replace("_", " ") for tag in unique_merged_tags)  # Без подчёркиваний

    return {
        "caption": merged_caption,
        "taglist": merged_taglist,
        "general": merged_general,
        "character": merged_character,
        "ratings": image_result["full_image"]["ratings"].copy()
    }


# Создаем кастомный энкодер для обработки numpy.float32
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# Сохранение в Json результатов
def save_results_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Результаты сохранены в {output_file}")