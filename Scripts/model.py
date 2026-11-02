"""Load local tagging models and explicitly download missing resources."""

# Получение моделей/загрузка моделей
import os
from pathlib import Path
from typing import Dict

import requests
from huggingface_hub import hf_hub_download
from torch import nn
from ultralytics import YOLO

from .wdv3_timm import LabelData, load_labels_csv


def ensure_model_folder(folder_path: Path) -> Path:
    """Создает папку для моделей, если она не существует."""
    if not folder_path.exists():
        print(f"Создание папки для моделей: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def download_model_files(repo_id: str, model_folder: Path) -> Dict[str, Path]:
    """Ensure the shared tagger bundle without fetching duplicate weight formats."""
    from .tagger_resources import ensure_resources, RESOURCE_FILES
    target = ensure_resources(repo_id, model_folder)
    return {name: target / name for name in RESOURCE_FILES if (target / name).is_file()}


def load_model_local_or_remote(repo_id: str, model_folder: Path) -> nn.Module:
    """Load the network from the shared complete resource bundle."""
    from .tagger_resources import ensure_resources, load_network
    return load_network(ensure_resources(repo_id, model_folder))


def load_labels_local_or_remote(repo_id: str, model_folder: Path) -> LabelData:
    """Read the selected bundle's own label dictionary."""
    from .tagger_resources import ensure_resources
    return load_labels_csv(ensure_resources(repo_id, model_folder) / 'selected_tags.csv')


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
        ensure_model_folder(model_directory)
        destination = model_directory / model_path
        model = YOLO(str(destination))
        if not destination.is_file():
            model.save(str(destination))
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


def load_yolo_model(model_path: str, yolo_model_dir: Path, *, allow_download=True) -> YOLO:
    """Load existing custom weights first, then try supported download backends."""
    local_path = _local_yolo_path(model_path, yolo_model_dir)
    if local_path is not None:
        print(f"Loading YOLO model from local path: {local_path}")
        return YOLO(str(local_path))
    if not allow_download:
        raise FileNotFoundError(f"YOLO weights not found: {model_path}. Enable download_missing or choose a local model in {yolo_model_dir}")
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
