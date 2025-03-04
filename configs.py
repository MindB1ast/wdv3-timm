#Файл для классов конфигов
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
from simple_parsing import field as sp_field


@dataclass
class DetectorConfig:
    name: str
    model_path: str
    confidence: float = 0.25  # Confedence для определения областей
    classes: List[int] = None # Классы определяемые моделью, если может в несколько
    remove_tags_from_full: List[str] = None # Для удаления тегов с предсказания с полной картинки при использовании этой модели
    remove_tags_from_region: List[str] = None # Удаление тегов из предсказаний в области определенной этой моделью
    add_tags_to_region: Dict[str, float] = None # Добавляет теги в этот регион
    exclude_from_region: List[str] = None 
    specific_excluded_tags: List[str] = None
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
        if self.specific_excluded_tags is None:
            self.specific_excluded_tags = []


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


