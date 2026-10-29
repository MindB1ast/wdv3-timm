"""Inspect image regions with Plotly and export frames using Pillow."""

import os
import random

import numpy as np
import plotly.express as px
from PIL import Image, ImageDraw


def _save_static(pixels, detection_results, colors, output_path):
    """Save boxes and labels with Pillow without a browser/static renderer."""
    with Image.fromarray(pixels).convert("RGB") as canvas:
        draw = ImageDraw.Draw(canvas)
        for name, regions in detection_results["detectors"].items():
            color = colors[name]
            for region in regions:
                bbox = region["bbox"]
                draw.rectangle(bbox, outline=color, width=3)
                draw.text(
                    (bbox[0] + 3, bbox[1] + 3), f"{name} ({region['region_id'] + 1})", fill=color
                )
        canvas.save(output_path)


def visualize_detection_results(image_path, detection_results, output_path=None, show=True):
    """
    Визуализирует результаты детекции YOLO-моделей с использованием plotly
    с интерактивным отображением тегов для каждого бокса.

    Args:
        image_path: Путь к исходному изображению
        detection_results: Результаты детекции из BatchTagging
        output_path: Путь для сохранения изображения с боксами (опционально)
        show: Показать результат (True) или только сохранить (False)

    Returns:
        plotly.graph_objs._figure.Figure: Фигура с отрисованными боксами
    """
    with Image.open(image_path) as img:
        img_np = np.array(img)
    fig = px.imshow(img_np)
    fig.update_layout(dragmode="pan")
    colors = {}
    for detector_name, regions in detection_results["detectors"].items():
        if detector_name not in colors:
            colors[detector_name] = "".join(
                [
                    "rgb(",
                    format(random.randint(50, 255), ""),
                    ", ",
                    format(random.randint(50, 255), ""),
                    ", ",
                    format(random.randint(50, 255), ""),
                    ")",
                ]
            )
        color = colors[detector_name]
        for region in regions:
            _draw_region(fig, detector_name, region, color)
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False}, margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    if output_path:
        _save_static(img_np, detection_results, colors, output_path)
        print(f"Визуализация сохранена в: {output_path}")
    if show:
        fig.show()
    return fig


def view_image_results(
    results, image_index, visualize=True, save_visualization=False, verbose=False
):
    """
    Просматривает результаты обработки конкретного изображения

    Args:
        results: Результат выполнения BatchTagging (кортеж из двух списков)
        image_index: Индекс изображения (начиная с 0)
        visualize: Визуализировать боксы детекторов
        save_visualization: Сохранять визуализацию в файл

    Returns:
        dict: Информация о тегах и детекции для выбранного изображения
    """
    if not results or len(results) != 2:
        print("Ошибка: неверный формат результатов")
        return None

    all_results, merged_results = results

    if image_index < 0 or image_index >= len(merged_results):
        print(
            "".join(
                [
                    "Ошибка: индекс ",
                    format(image_index, ""),
                    " вне диапазона. Доступно ",
                    format(len(merged_results), ""),
                    " изображений",
                ]
            )
        )
        return None

    # Получаем результаты для выбранного изображения
    image_result = all_results[image_index]
    merged_result = merged_results[image_index]

    image_path = image_result["image_path"]

    # Визуализируем результаты детекции, если требуется
    if visualize:
        output_path = None
        if save_visualization:
            output_path = os.path.splitext(image_path)[0] + "_detection.jpg"

        visualize_detection_results(image_path, image_result, output_path)

    if verbose:
        # Выводим основную информацию
        print(f"Информация о изображении #{image_index + 1}: {image_path}")

        print("\nОбъединенные теги:")
        print(merged_result["merged_tags"]["taglist"])
        print()
        print("Теги без включения yolo моделей:")
        print(image_result["full_image"]["taglist"])

        # Преобразуем строки тегов в списки, убирая лишние пробелы
        merged_tags = [tag.strip() for tag in merged_result["merged_tags"]["taglist"].split(",")]
        yolo_excluded_tags = [
            tag.strip() for tag in image_result["full_image"]["taglist"].split(",")
        ]

        # Вычисляем разницу: теги, присутствующие в объединённых, но отсутствующие в yolo
        diff_tags = set(merged_tags) - set(yolo_excluded_tags)

        print("\nРазница тегов (присутствуют в объединенных, но отсутствуют в тегах без yolo):")
        print(", ".join(sorted(diff_tags)))
    return None


def _draw_region(fig, detector_name, region, color):
    """Add a region box and its tag hover annotation."""
    x1, y1, x2, y2 = region["bbox"]
    region_tags = region.get("taglist", "")
    if region_tags:
        tags_text = region_tags.replace(", ", "<br>")
    else:
        tags_text = "Нет тегов"
    fig.add_shape(
        type="rect",
        x0=x1,
        y0=y1,
        x1=x2,
        y1=y2,
        line={"color": color, "width": 3},
        name=f"{detector_name} ({region['region_id'] + 1})",
    )
    fig.add_annotation(
        x=x1,
        y=y1,
        text=f"{detector_name} ({region['region_id'] + 1})",
        showarrow=False,
        font={"color": "white"},
        align="left",
        bordercolor=color,
        borderwidth=2,
        borderpad=4,
        bgcolor=color,
        opacity=0.8,
        hovertext=f"Теги:<br>{tags_text}",
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Arial"},
    )
