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
from .savingProc import save_tags_to_txt
from tqdm import tqdm
from ultralytics.utils import LOGGER



def vprint(*args, verbose=True, **kwargs):
    if verbose:
        print(*args, **kwargs)

def make_vprint(verbose):
    return lambda *args, **kwargs: print(*args, **kwargs) if verbose else None


def BatchTagging(opts: ScriptOptions):
    vprint = make_vprint(opts.verbose)
    if not opts.verbose:
        LOGGER.setLevel("ERROR")  # Или "CRITICAL"

    if opts.model not in MODEL_REPO_MAP:
        vprint(f"Доступные модели: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Неизвестная модель: {opts.model}")

    repo_id = MODEL_REPO_MAP[opts.model]
    image_folder = Path(opts.ImageFolder).resolve()
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {image_folder}")

    model_folder = ensure_model_folder(Path(opts.model_folder))
    yolo_model_dir = ensure_model_folder(Path(opts.yolo_model_dir))

    download_model_files(repo_id, model_folder)

    vprint(f"Загрузка модели тегирования '{opts.model}' из '{repo_id}'...")
    tagger_model = load_model_local_or_remote(repo_id, model_folder)

    vprint("Загрузка списка тегов...")
    labels = load_labels_local_or_remote(repo_id, model_folder)

    vprint("Создание трансформации данных...")
    transform = create_transform(**resolve_data_config(tagger_model.pretrained_cfg, model=tagger_model))

    vprint(f"Загрузка конфигурации детекторов из {opts.detectors_config}...")
    detectors = load_detectors_config(opts.detectors_config)

    yolo_models = {}
    for detector in detectors:
        if detector.model_path not in yolo_models:
            vprint(f"Загрузка модели YOLO {detector.model_path}...")
            try:
                yolo_models[detector.model_path] = load_yolo_model(detector.model_path, yolo_model_dir)
            except Exception as e:
                vprint(f"Ошибка загрузки модели YOLO {detector.model_path}: {e}")
                vprint(f"Детектор {detector.name} будет пропущен")

    if opts.recursive:
        image_files = list(image_folder.rglob("*.jpg")) + list(image_folder.rglob("*.jpeg")) + list(image_folder.rglob("*.png"))
    else:
        image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")) + list(image_folder.glob("*.png"))

    if not image_files:
        vprint("Изображения не найдены в указанной директории.")
        return

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tagger_model = tagger_model.to(torch_device)

    all_results = []
    all_merged_results = []
    total_images = len(image_files)

    vprint(f"Всего изображений для обработки: {total_images}")

    add_tags_before = opts.add_tags_before.strip()
    add_tags_after = opts.add_tags_after.strip()
    remove_tags = [tag.strip() for tag in opts.remove_tags.split(',')] if opts.remove_tags else []

    for i, img_path in enumerate(tqdm(image_files, desc="Обработка изображений")):
        vprint(f"\nОбработка изображения {i+1}/{total_images}: {img_path}")

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
            merged_tags = merge_tags_from_regions(result, detectors)

            if remove_tags:
                tags_list = merged_tags["taglist"].split(", ")
                caption_list = merged_tags["caption"].split(", ")
                filtered_tags = [tag for tag in tags_list if tag.lower() not in [t.lower() for t in remove_tags]]
                filtered_caption = [tag for tag in caption_list if tag.lower() not in [t.lower().replace(' ', '_') for t in remove_tags]]
                merged_tags["taglist"] = ", ".join(filtered_tags)
                merged_tags["caption"] = ", ".join(filtered_caption)

            if add_tags_before or add_tags_after:
                before_tags = ', '.join([s.strip() for s in add_tags_before.split(',') if s.strip()])
                after_tags = ', '.join([s.strip() for s in add_tags_after.split(',') if s.strip()])
                caption_before = ', '.join([s.strip().replace(' ', '_') for s in add_tags_before.split(',') if s.strip()])
                caption_after = ', '.join([s.strip().replace(' ', '_') for s in add_tags_after.split(',') if s.strip()])

                new_taglist = merged_tags['taglist']
                if before_tags:
                    new_taglist = before_tags + ", " + new_taglist
                if after_tags:
                    new_taglist = new_taglist + ", " + after_tags
                merged_tags['taglist'] = new_taglist.strip(', ')

                new_caption = merged_tags['caption']
                if caption_before:
                    new_caption = caption_before + ", " + new_caption
                if caption_after:
                    new_caption = new_caption + ", " + caption_after
                merged_tags['caption'] = new_caption.strip(', ')

            merged_result = {
                "image_path": result["image_path"],
                "merged_tags": merged_tags
            }
            all_merged_results.append(merged_result)

            if opts.save_txt:
                txt_path = save_tags_to_txt(
                    result["image_path"],
                    merged_tags,
                    append_tags=opts.append_txt,
                    add_tags_before=opts.add_tags_before,
                    add_tags_after=opts.add_tags_after
                )
                merged_result["txt_path"] = txt_path

            vprint("--------")
            vprint(f"Путь к изображению: {result['image_path']}")
            vprint("\nРезультаты для полного изображения:")
            vprint(f"Описание: {result['full_image']['caption']}")
            vprint(f"Теги: {result['full_image']['taglist']}")

            for detector_name, regions in result["detectors"].items():
                if regions:
                    vprint(f"\nДетектор {detector_name}: найдено областей: {len(regions)}")
                    for region in tqdm(regions, desc=f"{detector_name}: регионы", leave=False):
                        vprint(f"  Область {region['region_id']+1}, координаты: {region['bbox']}")
                        vprint(f"  Описание: {region['caption']}")
                        vprint(f"  Теги: {region['taglist']}")
                else:
                    vprint(f"\nДетектор {detector_name}: областей не найдено")

            vprint("\nОбъединенные теги:")
            vprint(f"Описание: {merged_tags['caption']}")
            vprint(f"Теги: {merged_tags['taglist']}")

            if opts.save_txt:
                if opts.append_txt:
                    vprint(f"Теги добавлены/обновлены в: {merged_result['txt_path']}")
                else:
                    vprint(f"Теги сохранены в: {merged_result['txt_path']}")

            vprint("--------")

    vprint(f"\nОбработка завершена. Обработано изображений: {len(all_results)}/{total_images}")

    if opts.output_file:
        save_results_to_json(all_results, opts.output_file)
        merged_output = os.path.splitext(opts.output_file)[0] + "_merged" + os.path.splitext(opts.output_file)[1]
        save_results_to_json(all_merged_results, merged_output)

    return all_results, all_merged_results
