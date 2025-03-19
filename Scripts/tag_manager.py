from typing import Dict, List, Tuple, Set, Optional, Union
import os
import json
from pathlib import Path
import re
import glob


class TagManager:
    def __init__(self, txt_folder: Optional[str] = None):
        """
        Инициализация менеджера тегов
        
        Args:
            txt_folder: Путь к папке с TXT файлами (опционально)
        """
        self.tag_stats = {}  # Статистика использования тегов
        self.image_tags = {}  # Теги для каждого изображения
        
        # Загружаем теги из указанной папки, если она предоставлена
        if txt_folder:
            self.load_from_folder(txt_folder)
    
    def load_from_folder(self, folder_path: str, recursive: bool = False) -> int:
        """
        Загружает теги из всех TXT файлов в указанной папке
        
        Args:
            folder_path: Путь к папке с TXT файлами
            recursive: Рекурсивный поиск в подпапках
            
        Returns:
            Количество загруженных файлов
        """
        folder = Path(folder_path).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Директория не найдена: {folder}")
        
        # Очищаем текущие данные
        self.tag_stats = {}
        self.image_tags = {}
        
        # Получаем список TXT файлов
        if recursive:
            txt_files = list(folder.rglob("*.txt"))
        else:
            txt_files = list(folder.glob("*.txt"))
        
        count = 0
        for txt_path in txt_files:
            try:
                # Проверяем, существует ли соответствующее изображение
                img_base = txt_path.stem
                img_dir = txt_path.parent
                
                # Проверяем наличие изображения с разными расширениями
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_img = img_dir / f"{img_base}{ext}"
                    if potential_img.exists():
                        img_path = str(potential_img)
                        break
                
                # Если изображение не найдено, используем путь TXT-файла
                if not img_path:
                    img_path = str(txt_path)
                
                # Читаем теги из файла
                with open(txt_path, 'r', encoding='utf-8') as f:
                    tags_content = f.read().strip()
                
                # Разбиваем строку на отдельные теги
                tags = [tag.strip() for tag in tags_content.split(",") if tag.strip()]
                
                # Добавляем в индекс
                self.image_tags[str(txt_path)] = {
                    "image_path": img_path,
                    "txt_path": str(txt_path),
                    "tags": tags
                }
                
                # Обновляем статистику
                for tag in tags:
                    if tag not in self.tag_stats:
                        self.tag_stats[tag] = {"count": 0, "files": []}
                    
                    self.tag_stats[tag]["count"] += 1
                    self.tag_stats[tag]["files"].append(str(txt_path))
                
                count += 1
            except Exception as e:
                print(f"Ошибка при обработке файла {txt_path}: {e}")
        
        print(f"Загружено {count} файлов с тегами")
        return count
    
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
            txt_path = image_data.get("txt_path", os.path.splitext(image_path)[0] + ".txt")
            merged_tags = image_data["merged_tags"]
            
            # Собираем все теги из taglist
            tags = [tag.strip() for tag in merged_tags["taglist"].split(",") if tag.strip()]
            
            # Добавляем изображение в индекс
            self.image_tags[txt_path] = {
                "image_path": image_path,
                "txt_path": txt_path,
                "tags": tags
            }
            
            # Обновляем статистику
            for tag in tags:
                if tag not in self.tag_stats:
                    self.tag_stats[tag] = {"count": 0, "files": []}
                
                self.tag_stats[tag]["count"] += 1
                self.tag_stats[tag]["files"].append(txt_path)
    
    def find_files_with_tag(self, tag: str) -> List[str]:
        """
        Возвращает список файлов с указанным тегом
        
        Args:
            tag: Искомый тег
            
        Returns:
            Список путей к файлам с этим тегом
        """
        if tag in self.tag_stats:
            return self.tag_stats[tag]["files"]
        return []
    
    def add_tags(self, add_tags_before: str = "", add_tags_after: str = "", files: List[str] = None) -> int:
        """
        Добавляет теги в начало и/или конец списка тегов в указанных файлах
        
        Args:
            add_tags_before: Теги для добавления в начало (через запятую)
            add_tags_after: Теги для добавления в конец (через запятую)
            files: Список путей к файлам (если None, обрабатываются все файлы)
            
        Returns:
            Количество измененных файлов
        """
        if not add_tags_before and not add_tags_after:
            return 0
        
        # Подготовка списков тегов
        before_tags = [tag.strip() for tag in add_tags_before.split(",") if tag.strip()]
        after_tags = [tag.strip() for tag in add_tags_after.split(",") if tag.strip()]
        
        # Если список файлов не указан, используем все файлы
        if files is None:
            files = list(self.image_tags.keys())
        
        count = 0
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                original_tags = file_data["tags"].copy()
                
                # Добавляем теги в начало
                new_tags = before_tags.copy()
                # Добавляем существующие теги, но избегаем дубликатов
                for tag in original_tags:
                    if tag not in new_tags:
                        new_tags.append(tag)
                
                # Добавляем теги в конец
                for tag in after_tags:
                    if tag not in new_tags:
                        new_tags.append(tag)
                
                # Обновляем данные, если теги изменились
                if new_tags != original_tags:
                    file_data["tags"] = new_tags
                    count += 1
        
        # Обновляем статистику, если были изменения
        if count > 0:
            self._update_tag_stats()
        
        return count
    
    def remove_tags(self, tags_to_remove: Union[str, List[str]], files: List[str] = None) -> int:
        """
        Удаляет указанные теги из всех файлов
        
        Args:
            tags_to_remove: Теги для удаления (строка через запятую или список)
            files: Список путей к файлам (если None, обрабатываются все файлы)
            
        Returns:
            Количество измененных файлов
        """
        # Преобразуем строку в список, если нужно
        if isinstance(tags_to_remove, str):
            remove_list = [tag.strip() for tag in tags_to_remove.split(",") if tag.strip()]
        else:
            remove_list = [tag.strip() for tag in tags_to_remove if tag.strip()]
        
        if not remove_list:
            return 0
        
        # Если список файлов не указан, используем все файлы
        if files is None:
            files = list(self.image_tags.keys())
        
        count = 0
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                original_tags = file_data["tags"].copy()
                
                # Удаляем указанные теги
                new_tags = [tag for tag in original_tags if tag.lower() not in [t.lower() for t in remove_list]]
                
                # Обновляем данные, если теги изменились
                if new_tags != original_tags:
                    file_data["tags"] = new_tags
                    count += 1
        
        # Обновляем статистику, если были изменения
        if count > 0:
            self._update_tag_stats()
        
        return count
    
    def replace_tag(self, old_tag: str, new_tag: str, files: List[str] = None) -> int:
        """
        Заменяет один тег на другой в указанных файлах
        
        Args:
            old_tag: Тег для замены
            new_tag: Новый тег
            files: Список путей к файлам (если None, обрабатываются все файлы)
            
        Returns:
            Количество измененных файлов
        """
        if not old_tag or not new_tag or old_tag == new_tag:
            return 0
        
        # Если список файлов не указан, проверяем все файлы с этим тегом
        if files is None:
            files = self.find_files_with_tag(old_tag)
        
        count = 0
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                original_tags = file_data["tags"].copy()
                
                # Заменяем тег
                new_tags = []
                for tag in original_tags:
                    if tag.lower() == old_tag.lower():
                        if new_tag not in new_tags:  # Избегаем дубликатов
                            new_tags.append(new_tag)
                    else:
                        new_tags.append(tag)
                
                # Обновляем данные, если теги изменились
                if new_tags != original_tags:
                    file_data["tags"] = new_tags
                    count += 1
        
        # Обновляем статистику, если были изменения
        if count > 0:
            self._update_tag_stats()
        
        return count
    
    def sort_tags(self, alphabetical: bool = True, importance: Optional[List[str]] = None, files: List[str] = None) -> int:
        """
        Сортирует теги по алфавиту или по важности
        
        Args:
            alphabetical: Сортировать ли по алфавиту
            importance: Список тегов по важности (сначала самые важные)
            files: Список путей к файлам (если None, обрабатываются все файлы)
            
        Returns:
            Количество измененных файлов
        """
        # Если список файлов не указан, используем все файлы
        if files is None:
            files = list(self.image_tags.keys())
        
        count = 0
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                tags = file_data["tags"].copy()
                
                # Сортируем теги
                if alphabetical:
                    tags.sort()
                
                # Сортируем по важности (нужные теги выносим в начало)
                if importance:
                    # Сначала создаем словарь с индексами важности
                    importance_dict = {tag.lower(): i for i, tag in enumerate(importance)}
                    
                    # Сортируем: сначала по важности, потом по алфавиту
                    tags.sort(key=lambda x: (importance_dict.get(x.lower(), len(importance)), x))
                
                # Обновляем данные
                if tags != file_data["tags"]:
                    file_data["tags"] = tags
                    count += 1
        
        return count
    
    def _update_tag_stats(self):
        """
        Обновляет статистику использования тегов на основе текущих данных
        """
        self.tag_stats = {}
        
        for file_path, file_data in self.image_tags.items():
            for tag in file_data["tags"]:
                if tag not in self.tag_stats:
                    self.tag_stats[tag] = {"count": 0, "files": []}
                
                self.tag_stats[tag]["count"] += 1
                self.tag_stats[tag]["files"].append(file_path)
    
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
    
    def save_changes(self, files: List[str] = None) -> int:
        """
        Сохраняет изменения в TXT файлы
        
        Args:
            files: Список путей к файлам (если None, сохраняются все файлы)
            
        Returns:
            Количество сохраненных файлов
        """
        # Если список файлов не указан, используем все файлы
        if files is None:
            files = list(self.image_tags.keys())
        
        count = 0
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                txt_path = file_data["txt_path"]
                tags = file_data["tags"]
                
                # Формируем строку с тегами
                tags_str = ", ".join(tags)
                
                # Сохраняем в файл
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(tags_str)
                    count += 1
                except Exception as e:
                    print(f"Ошибка при сохранении файла {txt_path}: {e}")
    
        print(f"Сохранено {count} файлов")
        return count

    def remove_redundant_subtags(self, files: List[str] = None) -> int:
        """
        Удаляет избыточные теги, которые являются подмножеством других тегов:
        Например:
        - blue sky + sky -> удаляет sky
        - white shirt + shirt -> удаляет shirt
        - collared shirt + shirt -> удаляет shirt
        
        Args:
            files: Список путей к файлам (если None, обрабатываются все файлы)
            
        Returns:
            Количество удаленных избыточных тегов
        """
        # Если список файлов не указан, используем все файлы
        if files is None:
            files = list(self.image_tags.keys())
        
        # Создаем функцию для определения подтегов
        def get_subtags(tags):
            redundant = []
            for tag in tags:
                for other_tag in tags:
                    if tag != other_tag and tag in other_tag and tag.split()[0] not in ["1", "2", "3", "4"]:
                        # Проверяем, что это действительно подтег (а не просто часть слова)
                        if f" {tag} " in f" {other_tag} " or other_tag.endswith(f" {tag}") or other_tag.startswith(f"{tag} "):
                            redundant.append(tag)
                            break
            return redundant
        
        # Обрабатываем указанные файлы
        total_removed = 0
        modified_files = 0
        
        for file_path in files:
            if file_path in self.image_tags:
                file_data = self.image_tags[file_path]
                original_tags = file_data["tags"].copy()
                tags = original_tags.copy()
                
                redundant_tags = get_subtags(tags)
                
                if redundant_tags:
                    print(f"В файле {file_path} найдены избыточные теги: {redundant_tags}")
                    for tag in redundant_tags:
                        if tag in tags:
                            tags.remove(tag)
                            total_removed += 1
                    
                    # Обновляем данные, если теги изменились
                    if tags != original_tags:
                        file_data["tags"] = tags
                        modified_files += 1
        
        # Обновляем статистику, если были изменения
        if total_removed > 0:
            self._update_tag_stats()
        
        print(f"Всего удалено {total_removed} избыточных подтегов в {modified_files} файлах")
        return total_removed