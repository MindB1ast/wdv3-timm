import os

# Процесс сохранения результатов в txt файл, отдельный файл потому что вероятно добавлю сортировку/другие штуки для сохранения тегов

def save_tags_to_txt(image_path, merged_tags, append_tags=True):
    """
    Сохраняет объединенные теги в TXT файл рядом с изображением.
    
    Args:
        image_path: Путь к изображению
        merged_tags: Словарь с объединенными тегами
        append_tags: Если True, добавляет новые теги к существующим; иначе перезаписывает
        
    Returns:
        str: Путь к созданному/обновленному TXT файлу
    """
    # Получаем путь для TXT файла (заменяем расширение на .txt)
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    
    # Получаем список тегов из объединенных результатов
    # Используем taglist, где теги разделены пробелами (без подчеркиваний)
    tags_to_save = merged_tags["taglist"].split(", ")
    
    # Если нужно добавить к существующим тегам
    if append_tags and os.path.exists(txt_path):
        try:
            # Читаем существующие теги
            with open(txt_path, 'r', encoding='utf-8') as f:
                existing_content = f.read().strip()
                
            # Разделяем существующие теги (они могут быть разделены запятыми, запятыми с пробелами, или новыми строками)
            if ", " in existing_content:
                existing_tags = [tag.strip() for tag in existing_content.split(", ")]
            elif "," in existing_content:
                existing_tags = [tag.strip() for tag in existing_content.split(",")]
            else:
                existing_tags = [tag.strip() for tag in existing_content.split("\n") if tag.strip()]
            
            # Объединяем старые и новые теги, убираем дубликаты
            existing_tags_set = set(existing_tags)
            new_tags = [tag for tag in tags_to_save if tag not in existing_tags_set]
            
            # Если есть новые теги для добавления
            if new_tags:
                all_tags = existing_tags + new_tags
                # Сортируем теги для удобства
                all_tags.sort()
                
                # Сохраняем обновленный список
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(", ".join(all_tags))
                
                print(f"Добавлено {len(new_tags)} новых тегов в {txt_path}")
            else:
                print(f"Новых тегов не обнаружено для {txt_path}")
                
        except Exception as e:
            print(f"Ошибка при добавлении тегов к {txt_path}: {e}")
            # В случае ошибки создадим новый файл с тегами
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(", ".join(tags_to_save))
            print(f"Создан новый файл с тегами: {txt_path}")
    else:
        # Просто создаем новый файл с тегами
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(tags_to_save))
        print(f"Создан файл с тегами: {txt_path}")
    
    return txt_path



#Для добавления тегов в начале или конце
def add_tag_to_merged(tags_for_image, add_tags_before: str = "", add_tags_after: str = ""):
    if len(tags_for_image) != 2:
        print('Нет результатов для изображения, или не верные данные')
        return None

    results = tags_for_image

    # Обработка для taglist: просто чистим лишние пробелы и оставляем как есть
    before_tags = ', '.join([s.strip() for s in add_tags_before.split(',') if s.strip()])
    after_tags = ', '.join([s.strip() for s in add_tags_after.split(',') if s.strip()])

    # Обработка для caption: заменяем пробелы на подчёркивания в каждом теге
    caption_before = ', '.join([s.strip().replace(' ', '_') for s in add_tags_before.split(',') if s.strip()])
    caption_after = ', '.join([s.strip().replace(' ', '_') for s in add_tags_after.split(',') if s.strip()])

    for image_data in results[1]:
        merged = image_data['merged_tags']
        
        # Обновляем taglist: добавляем перед и после исходного списка
        new_taglist = merged['taglist']
        if before_tags:
            new_taglist = before_tags + ", " + new_taglist
        if after_tags:
            new_taglist = new_taglist + ", " + after_tags
        merged['taglist'] = new_taglist.strip(', ')
        
        # Обновляем caption: используем обработанные подчеркиванием теги, разделённые запятыми
        new_caption = merged['caption']
        if caption_before:
            new_caption = caption_before + ", " + new_caption
        if caption_after:
            new_caption = new_caption + ", " + caption_after
        merged['caption'] = new_caption.strip(', ')

    return results
