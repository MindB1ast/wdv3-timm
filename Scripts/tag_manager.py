from typing import Dict, List, Tuple, Set, Optional
import os
import json
from pathlib import Path
import re

class TagManager:
    def __init__(self):
        self.tag_stats = {}  # Статистика использования тегов
        self.image_tags = {}  # Теги для каждого изображения
    
    def build_index_from_results(self, results):
        """
        Строит индекс тегов из результатов обработки
        
        Args:
            results: Результаты обработки изображений
        """
        self.tag_stats = {}
        self.image_tags = {}
        
        for image_data in results:
            image_path = image_data["image_path"]
            merged_tags = image_data["merged_tags"]
            
            # Собираем все теги из taglist
            tags = [tag.strip() for tag in merged_tags["taglist"].split(",")]
            
            # Добавляем изображение в индекс
            self.image_tags[image_path] = tags
            
            # Обновляем статистику
            for tag in tags:
                if tag not in self.tag_stats:
                    self.tag_stats[tag] = {"count": 0, "images": []}
                
                self.tag_stats[tag]["count"] += 1
                self.tag_stats[tag]["images"].append(image_path)
    
    def find_images_with_tag(self, tag: str) -> List[str]:
        """
        Возвращает список изображений с указанным тегом
        
        Args:
            tag: Искомый тег
            
        Returns:
            Список путей к изображениям с этим тегом
        """
        if tag in self.tag_stats:
            return self.tag_stats[tag]["images"]
        return []
    
    def replace_tag(self, old_tag: str, new_tag: str, results) -> Tuple[List, int]:
        """
        Заменяет один тег на другой во всех изображениях
        
        Args:
            old_tag: Тег для замены
            new_tag: Новый тег
            results: Результаты обработки
            
        Returns:
            Обновленные результаты и количество замен
        """
        count = 0
        modified_results = []
        
        for image_data in results:
            modified_image_data = image_data.copy()
            merged_tags = modified_image_data["merged_tags"]
            
            # Замена в taglist (с пробелами)
            if old_tag in merged_tags["taglist"]:
                merged_tags["taglist"] = re.sub(
                    r'\b' + re.escape(old_tag) + r'\b', 
                    new_tag, 
                    merged_tags["taglist"]
                )
                count += 1
            
            # Замена в caption (с подчёркиваниями)
            old_tag_underscore = old_tag.replace(" ", "_")
            new_tag_underscore = new_tag.replace(" ", "_")
            
            if old_tag_underscore in merged_tags["caption"]:
                merged_tags["caption"] = re.sub(
                    r'\b' + re.escape(old_tag_underscore) + r'\b', 
                    new_tag_underscore, 
                    merged_tags["caption"]
                )
            
            modified_results.append(modified_image_data)
        
        return modified_results, count
    
    def sort_tags(self, results, alphabetical: bool = True, importance: Optional[List[str]] = None) -> List:
        """
        Сортирует теги по алфавиту или по важности
        
        Args:
            results: Результаты обработки
            alphabetical: Сортировать ли по алфавиту
            importance: Список тегов по важности (сначала самые важные)
            
        Returns:
            Обновленные результаты
        """
        modified_results = []
        
        for image_data in results:
            modified_image_data = image_data.copy()
            merged_tags = modified_image_data["merged_tags"]
            
            # Получаем списки тегов
            taglist = [tag.strip() for tag in merged_tags["taglist"].split(",") if tag.strip()]
            caption = [tag.strip() for tag in merged_tags["caption"].split(",") if tag.strip()]
            
            # Сортируем теги
            if alphabetical:
                taglist.sort()
                caption.sort()
            
            # Сортируем по важности (нужные теги выносим в начало)
            if importance:
                # Сначала создаем словарь с индексами важности
                importance_dict = {tag: i for i, tag in enumerate(importance)}
                
                # Сортируем taglist: сначала по важности, потом по алфавиту
                taglist.sort(key=lambda x: (importance_dict.get(x, len(importance)), x))
                
                # То же самое для caption, но с учетом подчеркиваний
                importance_underscore = [tag.replace(" ", "_") for tag in importance]
                importance_dict_underscore = {tag: i for i, tag in enumerate(importance_underscore)}
                caption.sort(key=lambda x: (importance_dict_underscore.get(x, len(importance)), x))
            
            # Обновляем данные
            merged_tags["taglist"] = ", ".join(taglist)
            merged_tags["caption"] = ", ".join(caption)
            
            modified_results.append(modified_image_data)
        
        return modified_results
    
    def get_top_tags(self, limit: int = 50) -> List[Tuple[str, int]]:
        """
        Возвращает самые частые теги
        
        Args:
            limit: Максимальное количество тегов
            
        Returns:
            Список кортежей (тег, количество)
        """
        return sorted(
            [(tag, data["count"]) for tag, data in self.tag_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def save_changes_to_txt(self, results, append: bool = True):
        """
        Сохраняет изменения в TXT файлы рядом с изображениями
        
        Args:
            results: Обновленные результаты
            append: Добавлять ли к существующим файлам
        """
        for image_data in results:
            image_path = image_data["image_path"]
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            
            tags = image_data["merged_tags"]["taglist"]
            
            if append and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read().strip()
                
                # Перезаписываем файл
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(tags)
                
                print(f"Обновлены теги в: {txt_path}")
            else:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(tags)
                
                print(f"Созданы теги в: {txt_path}")
        
        return True

    @staticmethod
    def filter_results_by_tag(results, tag: str, include: bool = True) -> List:
        """
        Фильтрует результаты по наличию тега
        
        Args:
            results: Результаты обработки
            tag: Тег для фильтрации
            include: True - только с тегом, False - только без тега
            
        Returns:
            Отфильтрованные результаты
        """
        filtered = []
        
        for image_data in results:
            tags = image_data["merged_tags"]["taglist"].split(", ")
            has_tag = tag in tags
            
            if (include and has_tag) or (not include and not has_tag):
                filtered.append(image_data)
        
        return filtered