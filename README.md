# WD Tagger with Region Detection

## Overview

This project is an image tagging tool that enhances tag accuracy by using region detection techniques. It's designed to improve tagging for machine learning models, based on Waifu Diffusion (WD) models, by addressing limitations in latent space resolution.

## Key Features

- Support for multiple pre-trained models:
  - ViT 
  - SwinV2
  - ConvNeXT
  - EVA02 Large

- Flexible region detection with YOLO models
- Customizable tag thresholds
- Ability to add or remove tags for specific regions
- Batch processing
- Optional text file output for tags
- Solves the problem with wrong tags caused by detection (for example tags close-up, portrait when using face detection)

## Installation

```bash
git clone https://github.com/MindB1ast/wdv3-timm.git
cd wdv3-timm
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage
Usage example in BatchWork.ipynb

### Basic Example

```python
from Scripts import ScriptOptions, BatchTagging

params = ScriptOptions(
    ImageFolder="./TestPic/",
    model='big',  # Options: convnext, swinv2, big, vit
    gen_threshold=0.35,  # Confidence for general tags (confidence when tagging full picture, for threshold when tagging detected area change config)
    char_threshold=0.75,  
    batch=2, 
    recursive=False, 
    save_txt=True,   
    append_txt=True  
)

result = BatchTagging(params)
```

### Configuration Options 

#### ScriptOptions Parameters

- `ImageFolder`: Path to the directory containing images
- `model`: Tagging model to use (vit, swinv2, convnext, big)
- `gen_threshold`: Confidence threshold for general tags (default: 0.35)
- `char_threshold`: Confidence threshold for character tags (default: 0.75)
- `batch`: Number of images to process simultaneously
- `recursive`: Process images in subdirectories
- `save_txt`: Save tags to text files
- `append_txt`: Append tags to existing text files

### Custom Models and Detectors

1. Place yours Yolo models in the `models/` directory
2. Configure detectors in `detectors.json`

#### Detector Configuration Example

```json
[
  {
    "name": "person_detector", 
    "model_path": "person_yolov8s-seg.pt",
    "confidence": 0.35,
    "classes": [0],
    "remove_tags_from_full": ["tag1", "tag2"],
    "remove_tags_from_region": [],
    "add_tags_to_region": {},
    "exclude_from_region": [],
    "region_gen_threshold": 0.25,
    "region_char_threshold": 0.8
  }
]
```

## Advanced Usage

### Visualization

You can use the `view_image_results()` function to visualize detection results:

```python
from Scripts.visualization import view_image_results

view_image_results(result, image_index=0, visualize=True)
```

![изображение](https://github.com/user-attachments/assets/a67ced19-8c70-4ed0-87da-100f5b5a0545)




Сводка метрик (в %) для 12 изображений
| Метод              | Precision | Recall  | F1-score |
|--------------------|-----------|---------|----------|
| Объединенные теги(c yolo) | 72.72     | 71.75   | 70.09    |
| Полное изображение(без yolo) | 76.23     | 60.31   | 62.89    |

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
