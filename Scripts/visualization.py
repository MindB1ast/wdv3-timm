"""Inspect image regions with Plotly and export frames using Pillow."""

import os
import random
import copy
import base64
from html import escape
from io import BytesIO

import numpy as np
import plotly.express as px
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .types import BatchResult, DetailedResult, PathInput, RegionResult
from .region_figure import HoverMaskFigure


def _png_source(image: Image.Image) -> str:
    """Embed pixels independently of the notebook's Plotly typed-array support."""
    with BytesIO() as stream:
        image.save(stream, format="PNG")
        return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _save_static(
    pixels: NDArray[np.uint8], detection_results: DetailedResult,
    colors: dict[str, str], output_path: PathInput,
) -> None:
    """Save boxes and labels with Pillow without a browser/static renderer."""
    with Image.fromarray(pixels).convert("RGB") as canvas:
        draw = ImageDraw.Draw(canvas)
        for name, regions in detection_results["detectors"].items():
            color = colors[name]
            for region in regions:
                bbox = region["bbox"]
                draw.rectangle(bbox, outline=color, width=3)
                if "view" in region:
                    draw.rectangle(region["view"]["crop_bbox"], outline="yellow", width=1)
                draw.text(
                    (bbox[0] + 3, bbox[1] + 3), f"{name} ({region['region_id'] + 1})", fill=color
                )
        canvas.save(output_path)


def visualize_detection_results(
    image_path: PathInput, detection_results: DetailedResult,
    output_path: PathInput | None = None, show: bool = True,
) -> Figure:
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
    if "canonical_path" in detection_results and "export_root" in detection_results:
        from .regions import resolve_asset
        image_path = resolve_asset(detection_results["export_root"], detection_results["canonical_path"])
    with Image.open(image_path) as img, img.convert("RGB") as rgb:
        img_np = np.array(rgb)
    fig = px.imshow(img_np, binary_string=True)
    has_masks = any("mask_path" in region.get("geometry", {})
                    for regions in detection_results["detectors"].values() for region in regions)
    if has_masks and "export_root" in detection_results:
        fig = HoverMaskFigure(fig)
    fig.update_traces(hoverinfo="skip", selector={"type": "image"})
    fig.update_layout(dragmode="pan", hovermode="closest", showlegend=False)
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
            mask_index = _add_hover_mask(fig, region, detection_results.get("export_root"),
                                         (img_np.shape[1], img_np.shape[0]))
            _draw_region(fig, detector_name, region, color, mask_index)
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
    results: BatchResult | None, image_index: int, visualize: bool = True,
    save_visualization: bool = False, verbose: bool = False,
) -> None:
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


def _draw_region(
    fig: Figure, detector_name: str, region: RegionResult, color: str,
    mask_index: int | None = None,
) -> None:
    """Add a region box and its tag hover annotation."""
    x1, y1, x2, y2 = region["bbox"]
    if "view" in region:
        a, b, c, d = region["view"]["crop_bbox"]
        fig.add_shape(type="rect", x0=a, y0=b, x1=c, y1=d,
                      line={"color": "yellow", "width": 1, "dash": "dash"})
    region_tags = region.get("taglist", "")
    if region_tags:
        tags_text = escape(region_tags).replace(", ", "<br>")
    else:
        tags_text = "Нет тегов"
    if "geometry" in region:
        geometry = region["geometry"]
        tags_text += "<br>prompt: " + escape(geometry.get("prompt", ""))
        tags_text += "<br>segmentation: " + str(geometry.get("segmentation_score", "n/a"))
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
        text=f"{escape(detector_name)} ({region['region_id'] + 1})",
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
    if mask_index is not None:
        fig.add_trace(go.Scatter(
            x=[x1, x2, x2, x1, x1], y=[y1, y1, y2, y2, y1],
            mode="lines", line={"color": color, "width": 3},
            fill="toself", fillcolor="rgba(0,0,0,0)", hoveron="fills+points",
            name=f"{detector_name} ({region['region_id'] + 1})", showlegend=False,
            hoverinfo="text", text=f"{escape(detector_name)}<br>Теги:<br>{tags_text}",
            meta={"sam_mask_index": mask_index, "sam_bbox": [x1, y1, x2, y2]},
        ))


def _add_hover_mask(fig: Figure, region: RegionResult, root, size) -> int | None:
    """Use PNG alpha, never a full-frame RGBA Image trace over the source."""
    from .regions import read_mask, resolve_asset
    geometry = region.get("geometry")
    if root is None or geometry is None or "mask_path" not in geometry:
        return None
    mask = read_mask(resolve_asset(root, geometry["mask_path"]), size)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = [0, 200, 120, 255]
    index = len(fig.layout.images)
    with Image.fromarray(rgba) as overlay:
        fig.add_layout_image(source=overlay, xref="x", yref="y", x=-.5, y=-.5,
            sizex=size[0], sizey=size[1], xanchor="left", yanchor="top", sizing="stretch",
            opacity=.3, visible=False, layer="above")
    return index


def inspect_saved_region(root, image_id: str, source_name: str, region_id: int,
                         view=None, *, show=True, output_path=None) -> Figure:
    """Inspect geometry and the actual RGB input before the WD resize."""
    from .region_export import load_region_record, saved_region_input
    from .regions import resolve_asset
    from .sam_config import RegionViewConfig
    record = load_region_record(root, image_id)
    detail = record["detailed"]
    region: RegionResult = next(r for r in detail["detectors"][source_name] if r["region_id"] == region_id)
    if "geometry" not in region or "view" not in region:
        raise ValueError("Saved region has no geometry/view metadata")
    if view is None:
        original = region["view"]
        red, green, blue = original["background_rgb"]
        view = RegionViewConfig(original["mode"], original["padding_unit"], original["padding_value"],
                                (red, green, blue))
    crop, metadata = saved_region_input(root, image_id, source_name, region_id, view)
    with crop, Image.open(resolve_asset(root, detail["canonical_path"])) as original_frame:
        frame = original_frame.convert("RGB")
        with frame:
            selected_region = copy.deepcopy(region)
            selected_region["view"] = metadata
            title = (f"{source_name}: {region['geometry'].get('prompt', '')}; "
                     f"score={region['geometry'].get('segmentation_score', 'n/a')}; "
                     f"{view.mode}, padding={view.padding_value} {view.padding_unit}")
            fig = HoverMaskFigure(make_subplots(rows=1, cols=2, subplot_titles=(escape(title), "RGB input before WD resize")))
            fig.add_trace(go.Image(source=_png_source(frame)), row=1, col=1)
            fig.add_trace(go.Image(source=_png_source(crop)), row=1, col=2)
            fig.update_traces(hoverinfo="skip", selector={"type": "image"})
            mask_index = _add_hover_mask(fig, selected_region, root, frame.size)
            _draw_region(fig, source_name, selected_region, "red", mask_index)
            fig.update_layout(showlegend=False, hovermode="closest")
            if output_path:
                draw = ImageDraw.Draw(frame)
                draw.rectangle(region["bbox"], outline="red", width=2)
                draw.rectangle(metadata["crop_bbox"], outline="yellow", width=1)
                with Image.new("RGB", (frame.width+crop.width, max(frame.height, crop.height)+28), "white") as panel:
                    panel.paste(frame, (0, 28))
                    panel.paste(crop, (frame.width, 28))
                    ImageDraw.Draw(panel).text((4, 4), title, fill="black")
                    panel.save(output_path)
    if show:
        fig.show()
    return fig
