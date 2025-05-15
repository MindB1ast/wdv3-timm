"""
Файл с информацией о доступных YOLO моделях.
Содержит словарь имен моделей и их URL для загрузки.
"""

# Словарь известных моделей и их URL
KNOWN_MODELS = {
    # Модели YOLOv8-face
    "hand_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov9c.pt",
    "face_yolov9c.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov9c.pt",
    "person_yolov8m-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
    # Здесь можно добавить другие модели
}

# Стандартные модели YOLOv8 и YOLOv5
STANDARD_MODELS = [
    'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
    'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
]