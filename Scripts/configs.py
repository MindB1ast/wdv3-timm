#Файл для классов конфигов
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
from simple_parsing import field as sp_field




#класс конфига для настройки раьоты yolo моделей + параметров работы теггера в найденой области
@dataclass
class DetectorConfig:
    name: str #Как будет отображатся название модели в результате
    model_path: str #Путь до помдели
    confidence: float = 0.25  # Confedence для определения областей
    classes: List[int] = None # Классы определяемые моделью, если может в несколько
    remove_tags_from_full: List[str] = None # Для удаления тегов с предсказания с полной картинки при использовании этой модели
    remove_tags_from_region: List[str] = None # Удаление тегов из предсказаний в области определенной этой моделью
    add_tags_to_region: Dict[str, float] = None # Добавляет теги в этот регион
    exclude_from_region: List[str] = None 
    #specific_excluded_tags: List[str] = None
    region_gen_threshold: Optional[float] = None  # Порог для общих тегов в регионах
    region_char_threshold: Optional[float] = None  # Порог для тегов персонажей в регионах
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = [0]
        if self.remove_tags_from_full is None:
            self.remove_tags_from_full = []
        if self.remove_tags_from_region is None:
            self.remove_tags_from_region = []
        if self.add_tags_to_region is None:
            self.add_tags_to_region = {}
        if self.exclude_from_region is None:
            self.exclude_from_region = []
#        if self.specific_excluded_tags is None:
#            self.specific_excluded_tags = []


# Пример конфига для настройки yolo моделей и обработки рещультата областей, должен быть в формате json и передан в BatchWork параметром detectors_config, пример конфига detectors.json
EXAMPLE_DETECTORS_CONFIG = [
    {
        "name": "person_detector", 
        "model_path": "yolov8n.pt",
        "confidence": 0.25,
        "classes": [0],
        "remove_tags_from_full": ["1girl", "1boy"], #Удалить теги полной области при нахождении моделью области
        "remove_tags_from_region": ["multiple views"], # удалить теги из найденой области
        "add_tags_to_region": {"person": 1.0}, #Добавление тега в теги области при нахождении области на картинки 
        "exclude_from_region": ["close-up", "cowboy_shot", "full_body"], 
        #"specific_excluded_tags": [],
        "region_gen_threshold": 0.4, 
        "region_char_threshold": 0.8
    },
    {
        "name": "face_detector",
        "model_path": "yolov8n-face.pt",
        "confidence": 0.3,
        "classes": [0],
        "remove_tags_from_full": ["face"],
        "remove_tags_from_region": ["full body"],
        "add_tags_to_region": {"face": 1.0},
        "exclude_from_region": ["1girl", "portrait", "upper_body"],
        "specific_excluded_tags": ["blue_eyes", "green_eyes", "yellow_eyes", "smile"],
        "region_gen_threshold": 0.5, 
        "region_char_threshold": 0.85
    }
]



@dataclass
class ScriptOptions:
    ImageFolder: Path = sp_field(positional=True)
    model: str = sp_field(default="vit")
    gen_threshold: float = sp_field(default=0.35)
    char_threshold: float = sp_field(default=0.75)
    batch: int = sp_field(default=1)
    recursive: bool = sp_field(default=False)
    model_folder: Path = sp_field(default="./models/taggers/")  # Для моделей тегирования
    yolo_model_dir: Path = sp_field(default="./models/yolo/")    # Для YOLO моделей
    detectors_config: str = sp_field(default="detectors.json")
    output_file: str = sp_field(default="results.json")
    save_txt: bool = sp_field(default=True)      # Сохранять ли теги в TXT файлы
    append_txt: bool = sp_field(default=True)    # Добавлять ли теги к существующим TXT файлам


