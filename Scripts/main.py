from pathlib import Path
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
import glob
import os
from .wdv3_timm import MODEL_REPO_MAP
from .configs import ScriptOptions
from .model import ensure_model_folder, download_model_files, load_model_local_or_remote, load_labels_local_or_remote, load_yolo_model
from .detection import load_detectors_config
from .tagging import process_image_with_multiple_detectors
from .preprocess import merge_tags_from_regions, save_results_to_json
import matplotlib as plt
from .visualization import view_image_results
from .savingProc import save_tags_to_txt



def BatchTagging(opts: ScriptOptions):
    """
    Выполняет тегирование пакета изображений с использованием моделей тегирования и детекции.
    
    Args:
        opts (ScriptOptions): Параметры для выполнения тегирования. См. configs.py для деталей.
    
    Returns:
        tuple: Кортеж из двух списков (all_results, all_merged_results)
        
        1. all_results: Список с детальной информацией о каждом изображении
           [
               {
                   "image_path": str,      # Путь к обработанному изображению
                   "full_image": {         # Теги для всего изображения
                       "caption": str,     # Теги через запятую с подчеркиваниями
                       "taglist": str,     # Теги через запятую с пробелами
                       "general": list,    # Значения уверенности для общих тегов
                       "character": list,  # Значения уверенности для тегов персонажей
                       "ratings": list     # Значения уверенности для рейтинговых тегов
                   },
                   "detectors": {          # Результаты от каждого детектора
                       "detector_name": [  # Список областей, найденных детектором
                           {
                               "region_id": int,      # ID области (нумерация с 0)
                               "bbox": [x, y, w, h],  # Координаты области
                               "caption": str,        # Теги области с подчеркиваниями
                               "taglist": str,        # Теги области с пробелами
                               "general": list,       # Значения уверенности (общие)
                               "character": list,     # Значения уверенности (персонажи)
                               "ratings": list        # Значения уверенности (рейтинги)
                           },
                           # ... другие области
                       ],
                       # ... другие детекторы
                   }
               },
               # ... другие изображения
           ]
        
        2. all_merged_results: Список с объединенными тегами для каждого изображения
           [
               {
                   "image_path": str,      # Путь к обработанному изображению
                   "merged_tags": {        # Объединенные теги со всех детекторов
                       "caption": str,     # Теги через запятую с подчеркиваниями
                       "taglist": str,     # Теги через запятую с пробелами
                       "general": list,    # Значения уверенности для общих тегов
                       "character": list,  # Значения уверенности для тегов персонажей
                       "ratings": list     # Значения уверенности для рейтинговых тегов
                   },
                   "txt_path": str         # Путь к сохраненному TXT файлу (если save_txt=True)
               },
               # ... другие изображения
           ]
    
    Примеры использования:
        # Получение пути к 5-му изображению
        image_path = result[0][4]["image_path"]
        
        # Получение тегов для всего 3-го изображения
        full_image_tags = result[0][2]["full_image"]["taglist"]
        
        # Получение объединенных тегов для 7-го изображения
        merged_tags = result[1][6]["merged_tags"]["taglist"]
        merged_caption = result[1][6]["merged_tags"]["caption"]
        
        # Получение областей, обнаруженных детектором "face" в 1-м изображении
        face_regions = result[0][0]["detectors"].get("face", [])
    """
    if opts.model not in MODEL_REPO_MAP:
        print(f"Доступные модели: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Неизвестная модель: {opts.model}")
    
    repo_id = MODEL_REPO_MAP[opts.model]
    image_folder = Path(opts.ImageFolder).resolve()
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {image_folder}")
    
    # Создаем папку для моделей тегирования, если она не существует
    model_folder = ensure_model_folder(Path(opts.model_folder))
    
    # Создаем папку для моделей YOLO, если она не существует
    yolo_model_dir = ensure_model_folder(Path(opts.yolo_model_dir))
    
    # Загружаем или скачиваем файлы модели тегирования
    download_model_files(repo_id, model_folder)
    
    # Загружаем модель из локальной папки или из Hub
    print(f"Загрузка модели тегирования '{opts.model}' из '{repo_id}'...")
    tagger_model = load_model_local_or_remote(repo_id, model_folder)
    
    # Загружаем теги из локальной папки или из Hub
    print("Загрузка списка тегов...")
    labels = load_labels_local_or_remote(repo_id, model_folder)
    
    print("Создание трансформации данных...")
    #Тест новой трансформации данных
    transform = create_transform(**resolve_data_config(tagger_model.pretrained_cfg, model=tagger_model))

    # Загружаем конфигурацию детекторов
    print(f"Загрузка конфигурации детекторов из {opts.detectors_config}...")
    detectors = load_detectors_config(opts.detectors_config)
    
    # Загружаем все уникальные модели YOLO
    yolo_models = {}
    for detector in detectors:
        if detector.model_path not in yolo_models:
            print(f"Загрузка модели YOLO {detector.model_path}...")
            try:
                # Используем новую функцию для загрузки YOLO модели
                yolo_models[detector.model_path] = load_yolo_model(detector.model_path, yolo_model_dir)
            except Exception as e:
                print(f"Ошибка загрузки модели YOLO {detector.model_path}: {e}")
                print(f"Детектор {detector.name} будет пропущен")
    
    # Получаем список изображений
    if opts.recursive:
        image_files = list(image_folder.rglob("*.jpg")) + list(image_folder.rglob("*.jpeg")) + list(image_folder.rglob("*.png"))
    else:
        image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")) + list(image_folder.glob("*.png"))
    
    if not image_files:
        print("Изображения не найдены в указанной директории.")
        return
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tagger_model = tagger_model.to(torch_device)
    
    all_results = []
    all_merged_results = []
    total_images = len(image_files)
    
    print(f"Всего изображений для обработки: {total_images}")
    
    # Подготавливаем списки для управления тегами
    add_tags_before = opts.add_tags_before.strip()
    add_tags_after = opts.add_tags_after.strip()
    remove_tags = [tag.strip() for tag in opts.remove_tags.split(',')] if opts.remove_tags else []
    
    for i, img_path in enumerate(image_files):
        print(f"\nОбработка изображения {i+1}/{total_images}: {img_path}")
        
        # Обрабатываем изображение со всеми детекторами
        result = process_image_with_multiple_detectors(
            str(img_path),
            tagger_model,
            transform,
            labels,
            detectors,
            yolo_models,
            opts.gen_threshold,
            opts.char_threshold,
            torch_device
        )
        
        if result:
            all_results.append(result)
            
            # Объединяем теги из всех источников
            merged_tags = merge_tags_from_regions(result, detectors)
            
            # Дополнительная обработка тегов: удаление и добавление
            if remove_tags:
                # Удаляем указанные теги
                tags_list = merged_tags["taglist"].split(", ")
                caption_list = merged_tags["caption"].split(", ")
                
                # Фильтруем теги в taglist (с пробелами)
                filtered_tags = [tag for tag in tags_list if tag.lower() not in [t.lower() for t in remove_tags]]
                # Фильтруем теги в caption (с подчёркиваниями)
                filtered_caption = [tag for tag in caption_list if tag.lower() not in [t.lower().replace(' ', '_') for t in remove_tags]]
                
                merged_tags["taglist"] = ", ".join(filtered_tags)
                merged_tags["caption"] = ", ".join(filtered_caption)
            
            # Добавляем теги в начало и конец
            if add_tags_before or add_tags_after:
                # Обработка для taglist: просто чистим лишние пробелы и оставляем как есть
                before_tags = ', '.join([s.strip() for s in add_tags_before.split(',') if s.strip()])
                after_tags = ', '.join([s.strip() for s in add_tags_after.split(',') if s.strip()])
                
                # Обработка для caption: заменяем пробелы на подчёркивания в каждом теге
                caption_before = ', '.join([s.strip().replace(' ', '_') for s in add_tags_before.split(',') if s.strip()])
                caption_after = ', '.join([s.strip().replace(' ', '_') for s in add_tags_after.split(',') if s.strip()])
                
                # Обновляем taglist
                new_taglist = merged_tags['taglist']
                if before_tags:
                    new_taglist = before_tags + ", " + new_taglist
                if after_tags:
                    new_taglist = new_taglist + ", " + after_tags
                merged_tags['taglist'] = new_taglist.strip(', ')
                
                # Обновляем caption
                new_caption = merged_tags['caption']
                if caption_before:
                    new_caption = caption_before + ", " + new_caption
                if caption_after:
                    new_caption = new_caption + ", " + caption_after
                merged_tags['caption'] = new_caption.strip(', ')
            
            # Добавляем информацию о результатах для текущего изображения
            merged_result = {
                "image_path": result["image_path"],
                "merged_tags": merged_tags
            }
            all_merged_results.append(merged_result)
            
            # Если включена опция сохранения в TXT, сохраняем теги в отдельный файл
            if opts.save_txt:
                txt_path = save_tags_to_txt(
                    result["image_path"], 
                    merged_tags, 
                    append_tags=opts.append_txt,
                    add_tags_before=opts.add_tags_before,
                    add_tags_after=opts.add_tags_after
                )
                # Добавляем путь к TXT файлу в результаты
                merged_result["txt_path"] = txt_path
            
            # Выводим информацию о результатах
            print("--------")
            print(f"Путь к изображению: {result['image_path']}")
            print("\nРезультаты для полного изображения:")
            print(f"Описание: {result['full_image']['caption']}")
            print(f"Теги: {result['full_image']['taglist']}")
            
            # Выводим информацию о найденных областях для каждого детектора
            for detector_name, regions in result["detectors"].items():
                if regions:
                    print(f"\nДетектор {detector_name}: найдено областей: {len(regions)}")
                    for region in regions:
                        print(f"  Область {region['region_id']+1}, координаты: {region['bbox']}")
                        print(f"  Описание: {region['caption']}")
                        print(f"  Теги: {region['taglist']}")
                else:
                    print(f"\nДетектор {detector_name}: областей не найдено")
            
            # Выводим объединенные теги
            print("\nОбъединенные теги:")
            print(f"Описание: {merged_tags['caption']}")
            print(f"Теги: {merged_tags['taglist']}")
            
            # Если сохранили в TXT, показываем информацию
            if opts.save_txt:
                if opts.append_txt:
                    print(f"Теги добавлены/обновлены в: {merged_result['txt_path']}")
                else:
                    print(f"Теги сохранены в: {merged_result['txt_path']}")
            
            print("--------")
    
    print(f"\nОбработка завершена. Обработано изображений: {len(all_results)}/{total_images}")
    
    # Сохраняем результаты в файл
    if opts.output_file:
        save_results_to_json(all_results, opts.output_file)
        
        # Также сохраняем объединенные результаты
        merged_output = os.path.splitext(opts.output_file)[0] + "_merged" + os.path.splitext(opts.output_file)[1]
        save_results_to_json(all_merged_results, merged_output)
    
    return all_results, all_merged_results

