"""Public batch orchestration for full-image and regional tagging."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from ultralytics import YOLO

from .configs import DetectorConfig, ScriptOptions
from .detection import load_detectors_config
from .model import ensure_model_folder, load_yolo_model
from .tagger_catalog import resolve_model_id
from .tagger_resources import load_tagger
from .tagger_preprocessing import transform_metadata
from .preprocess import merge_tags_from_regions, save_results_to_json
from .savingProc import save_tags_to_txt
from .sam_config import positive_integer
from .tag_utils import apply_tag_rules, parse_tags
from .tagging import process_image_with_multiple_detectors
from .types import (BatchResult, DetailedResult, ImageTransform, MergedResult,
                    PathInput)
from .wdv3_timm import LabelData


@dataclass
class _BatchResources:
    """Loaded resources shared by all images in one batch."""

    model: torch.nn.Module
    transform: ImageTransform
    labels: LabelData
    detectors: list[DetectorConfig]
    yolo_models: dict[str, YOLO]
    device: torch.device | None = None


def _load_yolo_models(
    detectors: List[DetectorConfig], model_directory: Path,
) -> dict[str, YOLO]:
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


def _load_batch_resources(opts: ScriptOptions) -> _BatchResources:
    """Load model resources independently of file iteration and output formatting."""
    print(f"Загрузка конфигурации детекторов из {opts.detectors_config}...")
    detectors = load_detectors_config(opts.detectors_config)
    repo_id = resolve_model_id(opts.model)
    model_folder = ensure_model_folder(Path(opts.model_folder))
    yolo_folder = ensure_model_folder(Path(opts.yolo_model_dir))
    print(f"Загрузка модели тегирования '{opts.model}' из '{repo_id}'...")
    model, labels, transform, _ = load_tagger(repo_id, model_folder)
    return _BatchResources(
        model, transform, labels, detectors, _load_yolo_models(detectors, yolo_folder)
    )


def _process_image(
    image_path: Path, resources: _BatchResources, opts: ScriptOptions,
) -> tuple[DetailedResult, MergedResult] | None:
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
        region_batch_size=opts.region_batch_size,
    )
    if not result:
        return None
    result['model_profile'] = transform_metadata(resources.transform)
    return _postprocess_result(result, resources, opts)


def _postprocess_result(
    result: DetailedResult, resources: _BatchResources, opts: ScriptOptions,
) -> tuple[DetailedResult, MergedResult]:
    """Use the same merge and TXT rules for legacy and SAM image results."""
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
    from .tag_utils import enforce_owners
    merged = enforce_owners(merged, resources.detectors)
    combined: MergedResult = {"image_path": result["image_path"], "merged_tags": merged}
    if opts.save_txt:
        combined["txt_path"] = save_tags_to_txt(
            result["image_path"],
            merged,
            append_tags=opts.append_txt,
            add_tags_before=", ".join(t for t in parse_tags(opts.add_tags_before)
                                      if t in merged['general'] or t in merged['character']),
            add_tags_after=", ".join(t for t in parse_tags(opts.add_tags_after)
                                     if t in merged['general'] or t in merged['character']),
        )
    return result, combined


def _save_batch_results(
    results: list[DetailedResult], merged_results: list[MergedResult],
    output_file: PathInput | None,
) -> None:
    """Keep detailed and merged JSON exports separate."""
    if output_file:
        save_results_to_json(results, output_file)
        stem, extension = os.path.splitext(output_file)
        save_results_to_json(merged_results, stem + "_merged" + extension)


# Preserve the existing notebook calling convention.
# pylint: disable-next=C0103
def BatchTagging(opts: ScriptOptions) -> BatchResult | None:
    """Run a batch, retaining the legacy public name and two-list result."""
    positive_integer(opts.region_batch_size, "region_batch_size")
    resolve_model_id(opts.model)
    image_folder = Path(opts.ImageFolder).resolve()
    if not image_folder.is_dir():
        raise FileNotFoundError(f"Директория не найдена: {image_folder}")
    if opts.sam_config is not None:
        from .sam_pipeline import run_sam_batch
        return run_sam_batch(opts)
    search = image_folder.rglob if opts.recursive else image_folder.glob
    image_files = [path for extension in ("*.jpg", "*.jpeg", "*.png") for path in search(extension)]
    if not image_files:
        # Validate config without loading models, including a SAM config in the YOLO field.
        load_detectors_config(opts.detectors_config)
        print("Изображения не найдены в указанной директории.")
        return None
    resources = _load_batch_resources(opts)
    resources.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resources.model = resources.model.to(resources.device)
    results: list[DetailedResult] = []
    merged_results: list[MergedResult] = []
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


def _report_image(
    result: DetailedResult, merged_result: MergedResult, opts: ScriptOptions,
) -> None:
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
            print(f"Теги добавлены/обновлены в: {merged_result.get('txt_path')}")
        else:
            print(f"Теги сохранены в: {merged_result.get('txt_path')}")
    print("--------")
