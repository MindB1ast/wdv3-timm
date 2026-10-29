"""Public batch orchestration for full-image and regional tagging."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
#from timm.data import create_transform, resolve_data_config
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform

from .configs import ScriptOptions, DetectorConfig
from .detection import load_detectors_config
from .model import (
    download_model_files,
    ensure_model_folder,
    load_labels_local_or_remote,
    load_model_local_or_remote,
    load_yolo_model,
)
from .preprocess import merge_tags_from_regions, save_results_to_json
from .savingProc import save_tags_to_txt
from .tag_utils import apply_tag_rules
from .tagging import process_image_with_multiple_detectors
from .wdv3_timm import MODEL_REPO_MAP


@dataclass
class _BatchResources:
    """Loaded resources shared by all images in one batch."""

    model: torch.nn.Module
    transform: object
    labels: object
    detectors: list
    yolo_models: dict
    device: object = None


def _load_yolo_models(detectors:List[DetectorConfig], model_directory:Path):
    """Load each model once and report expected resource failures per detector."""
    models = {}
    for detector in detectors:
        if detector.model_path in models:
            continue
        print(f"Загрузка модели YOLO {detector.model_path}...")
        try:
            models[detector.model_path] = load_yolo_model(detector.model_path, model_directory)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
            print(f"Ошибка загрузки модели YOLO {detector.model_path}: {error}")
            print(f"Детектор {detector.name} будет пропущен")
    return models


def _load_batch_resources(opts:ScriptOptions):
    """Load model resources independently of file iteration and output formatting."""
    repo_id = MODEL_REPO_MAP[opts.model]
    model_folder = ensure_model_folder(Path(opts.model_folder))
    yolo_folder = ensure_model_folder(Path(opts.yolo_model_dir))
    download_model_files(repo_id, model_folder)
    print(f"Загрузка модели тегирования '{opts.model}' из '{repo_id}'...")
    model = load_model_local_or_remote(repo_id, model_folder)
    print("Загрузка списка тегов...")
    labels = load_labels_local_or_remote(repo_id, model_folder)
    print("Создание трансформации данных...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print(f"Загрузка конфигурации детекторов из {opts.detectors_config}...")
    detectors = load_detectors_config(opts.detectors_config)
    return _BatchResources(
        model, transform, labels, detectors, _load_yolo_models(detectors, yolo_folder)
    )


def _process_image(image_path:Path, resources:_BatchResources, opts:ScriptOptions):
    """Run inference and postprocess a single image using unchanged public APIs."""
    result = process_image_with_multiple_detectors(
        str(image_path),
        resources.model,
        resources.transform,
        resources.labels,
        resources.detectors,
        resources.yolo_models,
        opts.gen_threshold,
        opts.char_threshold,
        resources.device,
    )
    if not result:
        return None
    character_names = [
        resources.labels.names[index] for index in getattr(resources.labels, "character", [])
    ]
    merged = apply_tag_rules(
        merge_tags_from_regions(result, resources.detectors),
        remove=opts.remove_tags,
        before=opts.add_tags_before.strip(),
        after=opts.add_tags_after.strip(),
        character_names=character_names,
    )
    combined = {"image_path": result["image_path"], "merged_tags": merged}
    if opts.save_txt:
        combined["txt_path"] = save_tags_to_txt(
            result["image_path"],
            merged,
            append_tags=opts.append_txt,
            add_tags_before=opts.add_tags_before,
            add_tags_after=opts.add_tags_after,
        )
    return result, combined


def _save_batch_results(results, merged_results, output_file):
    """Keep detailed and merged JSON exports separate."""
    if output_file:
        save_results_to_json(results, output_file)
        stem, extension = os.path.splitext(output_file)
        save_results_to_json(merged_results, stem + "_merged" + extension)


# Preserve the existing notebook calling convention.
# pylint: disable-next=C0103
def BatchTagging(opts: ScriptOptions):
    """Run a batch, retaining the legacy public name and two-list result."""
    if opts.model not in MODEL_REPO_MAP:
        print(f"Доступные модели: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Неизвестная модель: {opts.model}")
    image_folder = Path(opts.ImageFolder).resolve()
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {image_folder}")
    resources = _load_batch_resources(opts)
    search = image_folder.rglob if opts.recursive else image_folder.glob
    image_files = [path for extension in ("*.jpg", "*.jpeg", "*.png") for path in search(extension)]
    if not image_files:
        print("Изображения не найдены в указанной директории.")
        return None
    resources.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resources.model = resources.model.to(resources.device)
    results, merged_results = [], []
    print(f"Всего изображений для обработки: {len(image_files)}")
    for index, image_path in enumerate(image_files):
        print(f"\nОбработка изображения {index + 1}/{len(image_files)}: {image_path}")
        processed = _process_image(image_path, resources, opts)
        if processed:
            detailed, merged = processed
            results.append(detailed)
            merged_results.append(merged)
            _report_image(detailed, merged, opts)
    print(f"\nОбработка завершена. Обработано изображений: {len(results)}/{len(image_files)}")
    _save_batch_results(results, merged_results, opts.output_file)
    return results, merged_results


def _report_image(result, merged_result, opts):
    """Print the same full-image, regional and merged notebook summary."""
    merged_tags = merged_result["merged_tags"]
    print("--------")
    print(f"Путь к изображению: {result['image_path']}")
    print("\nРезультаты для полного изображения:")
    print(f"Описание: {result['full_image']['caption']}")
    print(f"Теги: {result['full_image']['taglist']}")
    for detector_name, regions in result["detectors"].items():
        if regions:
            print(f"\nДетектор {detector_name}: найдено областей: {len(regions)}")
            for region in regions:
                print(f"  Область {region['region_id'] + 1}, координаты: {region['bbox']}")
                print(f"  Описание: {region['caption']}")
                print(f"  Теги: {region['taglist']}")
        else:
            print(f"\nДетектор {detector_name}: областей не найдено")
    print("\nОбъединенные теги:")
    print(f"Описание: {merged_tags['caption']}")
    print(f"Теги: {merged_tags['taglist']}")
    if opts.save_txt:
        if opts.append_txt:
            print(f"Теги добавлены/обновлены в: {merged_result['txt_path']}")
        else:
            print(f"Теги сохранены в: {merged_result['txt_path']}")
    print("--------")
