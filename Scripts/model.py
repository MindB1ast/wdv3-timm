"""Load local tagging models and explicitly download missing resources."""

# Получение моделей/загрузка моделей
import os
from pathlib import Path
from typing import Dict

import requests
import safetensors.torch
import timm
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from torch import nn
from ultralytics import YOLO

from .wdv3_timm import LabelData, load_labels_csv, load_labels_hf


def ensure_model_folder(folder_path: Path) -> Path:
    """Создает папку для моделей, если она не существует."""
    if not folder_path.exists():
        print(f"Создание папки для моделей: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def download_model_files(repo_id: str, model_folder: Path) -> Dict[str, Path]:
    """Загружает файлы модели wd1.4 в локальную папку."""
    # Создаем подпапку для конкретной модели (например models/wd-vit-tagger-v3)
    model_name = repo_id.split("/")[-1]
    model_specific_folder = model_folder / model_name
    ensure_model_folder(model_specific_folder)

    # Проверяем, есть ли уже safetensors
    has_safetensors = (model_specific_folder / "model.safetensors").exists()

    # Файлы, которые нужно загрузить
    files_to_download = [
        "config.json",
        "selected_tags.csv",  # Для тегов
    ]

    # Добавляем pytorch_model.bin только если нет safetensors
    if not has_safetensors:
        files_to_download.append("pytorch_model.bin")
        files_to_download.append(
            "model.safetensors"
        )  # Пробуем загрузить safetensors, если нет локально

    downloaded_files = {}
    for file in files_to_download:
        try:
            local_file_path = model_specific_folder / file
            # Если файл уже существует, пропускаем загрузку
            if local_file_path.exists():
                print(f"Файл {file} уже существует в {model_specific_folder}")
                downloaded_files[file] = local_file_path
                continue

            # Загружаем файл, если он не существует
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=str(model_specific_folder),
                local_files_only=False,
            )
            downloaded_files[file] = Path(downloaded_file)
            print(f"Загружен файл: {downloaded_file}")
        except HfHubHTTPError as e:
            print(f"Не удалось загрузить {file}: {e}")
            # Некоторые файлы могут не существовать, это нормально
            continue

    return downloaded_files


def load_model_local_or_remote(repo_id: str, model_folder: Path) -> nn.Module:
    """Load a WD architecture with locally cached weights when available."""
    model_name = repo_id.split("/")[-1]
    model_specific_folder = model_folder / model_name
    model = timm.create_model("hf-hub:" + repo_id).eval()

    local_safetensors_path = model_specific_folder / "model.safetensors"

    if local_safetensors_path.exists():
        print(f"Loading model from {local_safetensors_path} using safetensors")
        state_dict = safetensors.torch.load_file(local_safetensors_path)
    else:
        print(f"Loading model from Hugging Face Hub for {repo_id}...")
        state_dict = timm.models.load_state_dict_from_hf(repo_id)
        ensure_model_folder(model_specific_folder)
        local_save_path = model_specific_folder / "model.safetensors"
        print(f"Saving model to {local_save_path} using safetensors")
        safetensors.torch.save_file(state_dict, local_save_path)
        state_dict = safetensors.torch.load_file(local_save_path)

    model.load_state_dict(state_dict)
    return model


def load_labels_local_or_remote(repo_id: str, model_folder: Path) -> LabelData:
    """Загружает теги из локальной папки или с Hugging Face Hub."""
    model_name = repo_id.split("/")[-1]
    model_specific_folder = model_folder / model_name
    local_tags_path = model_specific_folder / "selected_tags.csv"

    if local_tags_path.exists():
        print(f"Загрузка тегов из локального файла: {local_tags_path}")
        try:
            # Используем возможность передачи пути к CSV файлу
            # Адаптируем для локального использования
            return load_labels_csv(local_tags_path)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
            print(f"Ошибка при загрузке локальных тегов: {e}")
            print("Попытка загрузки из Hugging Face Hub...")

    # Если локальных тегов нет или загрузка не удалась, загружаем из Hugging Face Hub
    print(f"Загрузка тегов из Hugging Face Hub для {repo_id}...")
    labels = load_labels_hf(repo_id=repo_id)

    # Если папка для модели существует, но CSV файла нет, можно скачать его
    # через hf_hub_download и сохранить локально
    if model_specific_folder.exists() and not local_tags_path.exists():
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="selected_tags.csv",
                local_dir=str(model_specific_folder),
                local_files_only=False,
            )
            print(f"Файл тегов сохранен локально в {local_tags_path}")
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
            print(f"Ошибка при сохранении файла тегов локально: {e}")

    return labels


_STANDARD_YOLO_MODELS = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
)
_KNOWN_YOLO_MODELS = {
    "hand_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov9c.pt",
    "face_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt",
    ("person_yolov8m-seg.pt"): (
        "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt"
    ),
}


def _local_yolo_path(model_path: str, model_directory: Path) -> Path | None:
    """Resolve existing custom weights before attempting network access."""
    if os.path.isfile(model_path):
        return Path(model_path)
    candidate = model_directory / model_path
    if candidate.exists():
        return candidate
    for extension in (".pt", ".pth", ".weights"):
        candidate = model_directory / (model_path + extension)
        if not model_path.endswith(extension) and candidate.exists():
            return candidate
    return None


def _download_known_yolo(model_path: str, model_directory: Path) -> YOLO:
    """Download known weights with a bounded timeout and closed HTTP response."""
    destination = model_directory / model_path
    print(f"Downloading {model_path} from known source...")
    with requests.get(_KNOWN_YOLO_MODELS[model_path], stream=True, timeout=30) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=8192):
                stream.write(chunk)
    print(f"Model saved to {destination}")
    return YOLO(str(destination))


def _remote_yolo_model(model_path: str, model_directory: Path) -> YOLO | None:
    """Resolve the explicitly supported standard, known-URL, and Hub backends."""
    if any(model_path.startswith(name) for name in _STANDARD_YOLO_MODELS):
        print(f"Standard model {model_path} not found locally. Attempting to download...")
        model = YOLO(model_path)
        destination = model_directory / model_path
        model.save(str(destination.parent))
        print(f"Model saved to {destination}")
        return model
    if model_path in _KNOWN_YOLO_MODELS:
        return _download_known_yolo(model_path, model_directory)
    if "/" in model_path:
        repo_id, filename = model_path.rsplit("/", 1)
        print(f"Attempting to download model from Hugging Face: repo={repo_id}, file={filename}")
        ensure_model_folder(model_directory)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_directory),
            local_files_only=False,
        )
        print(f"Model downloaded to {downloaded}")
        return YOLO(downloaded)
    return None


def load_yolo_model(model_path: str, yolo_model_dir: Path) -> YOLO:
    """Load existing custom weights first, then try supported download backends."""
    local_path = _local_yolo_path(model_path, yolo_model_dir)
    if local_path is not None:
        print(f"Loading YOLO model from local path: {local_path}")
        return YOLO(str(local_path))
    try:
        model = _remote_yolo_model(model_path, yolo_model_dir)
        if model is not None:
            return model
    except (
        requests.RequestException,
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
    ) as error:
        print(f"Error loading YOLO model {model_path}: {error}")
    raise FileNotFoundError(
        f"Could not find or download model: {model_path}. "
        f"Please place the model file manually in {yolo_model_dir}"
    )
