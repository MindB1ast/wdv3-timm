#Получение моделей/загрузка моделей
import os
from pathlib import Path
import timm
import torch
from torch import nn
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from timm.data import create_transform, resolve_data_config
import safetensors.torch
from typing import Dict
from ultralytics import YOLO
from .wdv3_timm import load_labels_hf, LabelData


def ensure_model_folder(folder_path: Path) -> Path:
    """Создает папку для моделей, если она не существует."""
    if not folder_path.exists():
        print(f"Создание папки для моделей: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def download_model_files(repo_id: str, model_folder: Path) -> Dict[str, Path]:
    """Загружает файлы модели wd1.4 в локальную папку."""
    # Создаем подпапку для конкретной модели (например models/wd-vit-tagger-v3)
    model_name = repo_id.split('/')[-1]
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
        files_to_download.append("model.safetensors")  # Пробуем загрузить safetensors, если нет локально
    
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
                local_files_only=False
            )
            downloaded_files[file] = Path(downloaded_file)
            print(f"Загружен файл: {downloaded_file}")
        except HfHubHTTPError as e:
            print(f"Не удалось загрузить {file}: {e}")
            # Некоторые файлы могут не существовать, это нормально
            continue
    
    return downloaded_files


def load_model_local_or_remote(repo_id: str, model_folder: Path) -> nn.Module:
    model_name = repo_id.split('/')[-1]
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
    model_name = repo_id.split('/')[-1]
    model_specific_folder = model_folder / model_name
    local_tags_path = model_specific_folder / "selected_tags.csv"
    
    if local_tags_path.exists():
        print(f"Загрузка тегов из локального файла: {local_tags_path}")
        try:
            # Используем возможность передачи пути к CSV файлу
            # Адаптируем для локального использования
            df: pd.DataFrame = pd.read_csv(local_tags_path, usecols=["name", "category"])
            tag_data = LabelData(
                names=df["name"].tolist(),
                rating=list(np.where(df["category"] == 9)[0]),
                general=list(np.where(df["category"] == 0)[0]),
                character=list(np.where(df["category"] == 4)[0]),
            )
            return tag_data
        except Exception as e:
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
                local_files_only=False
            )
            print(f"Файл тегов сохранен локально в {local_tags_path}")
        except Exception as e:
            print(f"Ошибка при сохранении файла тегов локально: {e}")
    
    return labels

from huggingface_hub import hf_hub_download

def load_yolo_model(model_path: str, yolo_model_dir: Path) -> YOLO:
    """
    Loads a YOLO model from a specified path, prioritizing custom models.
    
    Args:
        model_path: Path or name of the YOLO model
        yolo_model_dir: Directory for YOLO models
        
    Returns:
        Loaded YOLO model
    """
    # First check if this is an absolute path
    if os.path.isfile(model_path):
        print(f"Loading YOLO model from specified path: {model_path}")
        return YOLO(model_path)
    
    # Check if the model is in the specified yolo_model_dir
    local_model_path = yolo_model_dir / model_path
    if local_model_path.exists():
        print(f"Loading YOLO model from local directory: {local_model_path}")
        return YOLO(str(local_model_path))
    
    # For custom models, check if it's in various common extensions
    extensions = ['.pt', '.pth', '.weights']
    for ext in extensions:
        if not model_path.endswith(ext):
            potential_path = yolo_model_dir / (model_path + ext)
            if potential_path.exists():
                print(f"Loading YOLO model from local directory: {potential_path}")
                return YOLO(str(potential_path))
    
    # If not found locally, try downloading depending on the model source
    standard_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                     'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    
    # Словарь известных моделей и их URL
    known_models = {
        # Модели YOLOv8-face
        "hand_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov9c.pt",
        "face_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt",
        "person_yolov8m-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
        #"face_yolov8l.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8l.pt",
        # Другие известные модели
       # "hand_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
        # Можно добавить и другие модели
    }
    
    # Check if it's a standard model (from ultralytics)
    if any(model_path.startswith(model) for model in standard_models):
        print(f"Standard model {model_path} not found locally. Attempting to download...")
        try:
            # Try to load the model from the cloud (ultralytics)
            model = YOLO(model_path)
            # Save the model locally for future use
            save_path = yolo_model_dir / model_path
            model.save(str(save_path.parent))
            print(f"Model saved to {save_path}")
            return model
        except Exception as e:
            print(f"Error loading standard YOLO model {model_path}: {e}")
    
    # Проверяем, есть ли модель в нашем словаре известных моделей
    elif model_path in known_models:
        import requests
        
        print(f"Downloading {model_path} from known source...")
        save_path = yolo_model_dir / model_path
        
        try:
            response = requests.get(known_models[model_path], stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Model saved to {save_path}")
            return YOLO(str(save_path))
        except Exception as e:
            print(f"Error downloading known model {model_path}: {e}")
    
    # Для моделей Hugging Face со специфическим форматом ('repo_id/filename')
    elif '/' in model_path:
        try:
            # Разбираем путь на репозиторий и имя файла
            parts = model_path.split('/')
            repo_id = '/'.join(parts[:-1])  # все, кроме последней части
            filename = parts[-1]            # последняя часть
            
            print(f"Attempting to download model from Hugging Face: repo={repo_id}, file={filename}")
            
            # Создаем директорию для сохранения
            ensure_model_folder(yolo_model_dir)
            
            # Загружаем модель из Hugging Face
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(yolo_model_dir),
                local_files_only=False
            )
            
            print(f"Model downloaded to {downloaded_file}")
            return YOLO(downloaded_file)
        except Exception as e:
            print(f"Error downloading model from Hugging Face {model_path}: {e}")
    
    raise FileNotFoundError(f"Could not find or download model: {model_path}. "
                           f"Please place the model file manually in {yolo_model_dir}")