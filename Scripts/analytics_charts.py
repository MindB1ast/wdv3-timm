"""Single-method and aggregate Plotly analytics figures."""

from typing import Dict, Set

import pandas as pd
import plotly.graph_objects as go

from .analytics_stats import calculate_average_tag_count
from .analytics_text import wrap_text


def create_interactive_image_metrics_chart(
    error_df: pd.DataFrame, method_name: str, n: int = 10
) -> go.Figure:
    """Build a metric chart with preserved controls and hover details."""
    worst_images = error_df.nsmallest(n, "f1_score")
    fig = go.Figure()
    _add_metric_bars(fig, worst_images, "image", None)
    fig.update_layout(
        title=f"Худшие {n} изображений: {method_name}",
        xaxis_title="Изображение",
        yaxis_title="Значение метрики",
        barmode="group",
        height=500,
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": list(
                    [
                        {
                            "args": [{"visible": [True, False, False]}],
                            "label": "F1 Score",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, True, False]}],
                            "label": "Precision",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, False, True]}],
                            "label": "Recall",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [True, True, True]}],
                            "label": "All Metrics",
                            "method": "update",
                        },
                    ]
                ),
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
    )
    _add_hover_points(
        fig,
        worst_images,
        "image",
        [
            "".join(
                [
                    "\n            <b>Изображение:</b> ",
                    format(row["image"], ""),
                    "<br>\n            <b>F1 Score:</b> ",
                    format(row["f1_score"], "".join([".3f"])),
                    "<br>\n            <b>Precision:</b> ",
                    format(row["precision"], "".join([".3f"])),
                    "<br>\n            <b>Recall:</b> ",
                    format(row["recall"], "".join([".3f"])),
                    "<br>\n            <b>Правильные теги:</b> ",
                    format(wrap_text(row["correct_tags"], width=50), ""),
                    "<br>\n            <b>Ложноположительные теги:</b> ",
                    format(wrap_text(row["false_positives"], width=50), ""),
                    "<br>\n            <b>Ложноотрицательные теги:</b> ",
                    format(wrap_text(row["false_negatives"], width=50), ""),
                    "\n            ",
                ]
            )
            for _, row in worst_images.iterrows()
        ],
        None,
    )
    return fig


def create_interactive_tag_metrics_chart(
    tag_df: pd.DataFrame, method_name: str, n: int = 20
) -> go.Figure:
    """Создает интерактивный график с метриками для тегов с использованием Plotly."""
    filtered_df = tag_df[tag_df["total_ground_truth"] >= 3]
    worst_tags = filtered_df.nsmallest(n, "f1_score")
    fig = go.Figure()
    _add_metric_bars(fig, worst_tags, "tag", None)
    fig.update_layout(
        title=f"Худшие {n} тегов: {method_name} (мин. 3 вхождения)",
        xaxis_title="Значение метрики",
        yaxis_title="Тег",
        barmode="group",
        height=max(500, n * 25),
        hovermode="y unified",
    )
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": list(
                    [
                        {
                            "args": [{"visible": [True, False, False]}],
                            "label": "F1 Score",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, True, False]}],
                            "label": "Precision",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, False, True]}],
                            "label": "Recall",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [True, True, True]}],
                            "label": "All Metrics",
                            "method": "update",
                        },
                    ]
                ),
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ]
    )
    _add_hover_points(
        fig,
        worst_tags,
        "tag",
        [
            "".join(
                [
                    "\n            <b>Тег:</b> ",
                    format(row["tag"], ""),
                    "<br>\n            <b>F1 Score:</b> ",
                    format(row["f1_score"], "".join([".3f"])),
                    "<br>\n            <b>Precision:</b> ",
                    format(row["precision"], "".join([".3f"])),
                    "<br>\n            <b>Recall:</b> ",
                    format(row["recall"], "".join([".3f"])),
                    "<br>\n            <b>True Positives:</b> ",
                    format(row["true_positives"], ""),
                    "<br>\n            <b>False Positives:</b> ",
                    format(row["false_positives"], ""),
                    "<br>\n            <b>False Negatives:</b> ",
                    format(row["false_negatives"], ""),
                    "<br>\n            <b>Всего в Ground Truth:</b> ",
                    format(row["total_ground_truth"], ""),
                    "<br>\n            <b>Всего предсказано:</b> ",
                    format(row["total_predicted"], ""),
                    "\n            ",
                ]
            )
            for _, row in worst_tags.iterrows()
        ],
        None,
    )
    return fig


def create_combined_metrics_chart(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Создает единый интерактивный график для сравнения метрик обоих методов."""
    methods = list(metrics.keys())
    metric_types = list(metrics[methods[0]].keys())
    fig = go.Figure()
    for metric in metric_types:
        fig.add_trace(
            go.Bar(
                x=methods,
                y=[metrics[method][metric] * 100 for method in methods],
                name=metric,
                marker_color={"Precision": "#FAA43A", "Recall": "#60BD68", "F1-score": "#5DA5DA"}[
                    metric
                ],
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="Сравнение методов тегирования",
        xaxis_title="Метод",
        yaxis_title="Значение (%)",
        barmode="group",
        height=500,
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": list(
                    [
                        {
                            "args": [{"visible": [True, False, False]}],
                            "label": "Precision",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, True, False]}],
                            "label": "Recall",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [False, False, True]}],
                            "label": "F1 Score",
                            "method": "update",
                        },
                        {
                            "args": [{"visible": [True, True, True]}],
                            "label": "All Metrics",
                            "method": "update",
                        },
                    ]
                ),
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top",
            }
        ],
    )
    return fig


def create_interactive_tag_count_chart(
    ground_truth: Dict[str, Set[str]], method_tags: Dict[str, Dict[str, Set[str]]]
) -> go.Figure:
    """
    Создает интерактивный график для сравнения количества тегов в разных методах.

    Args:
        ground_truth: Словарь с эталонными тегами
        method_tags: Словарь с тегами для каждого метода

    Returns:
        Plotly фигуру с графиком
    """
    gt_stats = calculate_average_tag_count(ground_truth)
    method_stats = {
        method: calculate_average_tag_count(tags) for method, tags in method_tags.items()
    }
    methods = ["Ground Truth"] + list(method_stats.keys())
    avg_counts = [gt_stats["avg"]] + [stats["avg"] for stats in method_stats.values()]
    median_counts = [gt_stats["median"]] + [stats["median"] for stats in method_stats.values()]
    min_counts = [gt_stats["min"]] + [stats["min"] for stats in method_stats.values()]
    max_counts = [gt_stats["max"]] + [stats["max"] for stats in method_stats.values()]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=methods,
            y=avg_counts,
            name="Среднее",
            marker_color="#3498db",
            hovertemplate="<b>%{x}</b><br>Среднее: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=methods,
            y=median_counts,
            name="Медиана",
            marker_color="#2ecc71",
            hovertemplate="<b>%{x}</b><br>Медиана: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=methods,
            y=min_counts,
            mode="markers",
            name="Минимум",
            marker={"symbol": "triangle-down", "size": 12, "color": "#e74c3c"},
            hovertemplate="<b>%{x}</b><br>Минимум: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=methods,
            y=max_counts,
            mode="markers",
            name="Максимум",
            marker={"symbol": "triangle-up", "size": 12, "color": "#9b59b6"},
            hovertemplate="<b>%{x}</b><br>Максимум: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Статистика по количеству тегов",
        xaxis_title="Метод",
        yaxis_title="Количество тегов",
        barmode="group",
        height=500,
        hovermode="x unified",
    )
    return fig


def _visibility(methods, counts, selected, metrics):
    """Build visibility flags for metric bars and their method's hover traces."""
    return [
        flag
        for index, method in enumerate(methods)
        for flag in (list(metrics) if index == selected else [False] * 3)
        + [index == selected] * counts[method]
    ]


def _visibility_buttons(methods, counts):
    """Keep method selection and metric selection consistent across charts."""
    method_buttons = [
        {
            "args": [{"visible": _visibility(methods, counts, index, (True, False, False))}],
            "label": method,
            "method": "update",
        }
        for index, method in enumerate(methods)
    ]
    metric_buttons = [
        {
            "args": [{"visible": _visibility(methods, counts, 0, flags)}],
            "label": label,
            "method": "update",
        }
        for label, flags in [
            ("F1 Score", (True, False, False)),
            ("Precision", (False, True, False)),
            ("Recall", (False, False, True)),
            ("All Metrics", (True, True, True)),
        ]
    ]
    return (method_buttons, metric_buttons)


def _supported_tags(frame, count):
    """Select tag rows with the same minimum support as the notebook charts."""
    return frame[frame["total_ground_truth"] >= 3].nsmallest(count, "f1_score")


def create_combined_image_metrics_chart(
    error_dfs: Dict[str, pd.DataFrame], n: int = 10
) -> go.Figure:
    """Compare per-image metrics with method and metric selection."""
    fig = go.Figure()
    methods = list(error_dfs.keys())
    for method in methods:
        worst_images = error_dfs[method].nsmallest(n, "f1_score")
        _add_metric_bars(
            fig, worst_images, "image", {"method": method, "selected": method == methods[0]}
        )
        _add_hover_points(
            fig,
            worst_images,
            "image",
            [
                "".join(
                    [
                        "\n                <b>Изображение:</b> ",
                        format(row["image"], ""),
                        "<br>\n                <b>Метод:</b> ",
                        format(method, ""),
                        "<br>\n                <b>F1 Score:</b> ",
                        format(row["f1_score"], "".join([".3f"])),
                        "<br>\n                <b>Precision:</b> ",
                        format(row["precision"], "".join([".3f"])),
                        "<br>\n                <b>Recall:</b> ",
                        format(row["recall"], "".join([".3f"])),
                        "<br>\n                <b>Правильные теги:</b> ",
                        format(wrap_text(row["correct_tags"], width=50), ""),
                        "<br>\n                <b>Ложноположительные теги:</b> ",
                        format(wrap_text(row["false_positives"], width=50), ""),
                        "<br>\n                <b>Ложноотрицательные теги:</b> ",
                        format(wrap_text(row["false_negatives"], width=50), ""),
                        "\n                ",
                    ]
                )
                for _, row in worst_images.iterrows()
            ],
            {"method": method, "selected": method == methods[0]},
        )
    counts = {method: len(error_dfs[method].nsmallest(n, "f1_score")) for method in methods}
    method_buttons, metric_buttons = _visibility_buttons(methods, counts)
    fig.update_layout(
        title=f"Худшие {n} изображений по методам",
        xaxis_title="Изображение",
        yaxis_title="Значение метрики",
        barmode="group",
        height=500,
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": method_buttons,
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            },
            {
                "type": "buttons",
                "direction": "right",
                "buttons": metric_buttons,
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.5,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            },
        ],
        annotations=[
            {
                "text": "Выберите метод:",
                "x": 0.1,
                "y": 1.12,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12},
            },
            {
                "text": "Выберите метрику:",
                "x": 0.5,
                "y": 1.12,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12},
            },
        ],
    )
    return fig


def create_combined_tag_metrics_chart(tag_dfs: Dict[str, pd.DataFrame], n: int = 20) -> go.Figure:
    """Compare per-tag metrics with method and metric selection."""
    fig = go.Figure()
    methods = list(tag_dfs.keys())
    for method in methods:
        filtered_df = tag_dfs[method][tag_dfs[method]["total_ground_truth"] >= 3]
        worst_tags = filtered_df.nsmallest(n, "f1_score")
        _add_metric_bars(
            fig, worst_tags, "tag", {"method": method, "selected": method == methods[0]}
        )
        _add_hover_points(
            fig,
            worst_tags,
            "tag",
            [
                "".join(
                    [
                        "\n                <b>Тег:</b> ",
                        format(row["tag"], ""),
                        "<br>\n                <b>Метод:</b> ",
                        format(method, ""),
                        "<br>\n                <b>F1 Score:</b> ",
                        format(row["f1_score"], "".join([".3f"])),
                        "<br>\n                <b>Precision:</b> ",
                        format(row["precision"], "".join([".3f"])),
                        "<br>\n                <b>Recall:</b> ",
                        format(row["recall"], "".join([".3f"])),
                        "<br>\n                <b>True Positives:</b> ",
                        format(row["true_positives"], ""),
                        "<br>\n                <b>False Positives:</b> ",
                        format(row["false_positives"], ""),
                        "<br>\n                <b>False Negatives:</b> ",
                        format(row["false_negatives"], ""),
                        "<br>\n                <b>Всего в Ground Truth:</b> ",
                        format(row["total_ground_truth"], ""),
                        "<br>\n                <b>Всего предсказано:</b> ",
                        format(row["total_predicted"], ""),
                        "\n                ",
                    ]
                )
                for _, row in worst_tags.iterrows()
            ],
            {"method": method, "selected": method == methods[0]},
        )
    counts = {method: len(_supported_tags(tag_dfs[method], n)) for method in methods}
    method_buttons, metric_buttons = _visibility_buttons(methods, counts)
    fig.update_layout(
        title=f"Худшие {n} тегов по методам (мин. 3 вхождения)",
        xaxis_title="Значение метрики",
        yaxis_title="Тег",
        barmode="group",
        height=max(500, n * 25),
        hovermode="y unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": method_buttons,
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.05,
                "yanchor": "top",
            },
            {
                "type": "buttons",
                "direction": "right",
                "buttons": metric_buttons,
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.5,
                "xanchor": "left",
                "y": 1.05,
                "yanchor": "top",
            },
        ],
        annotations=[
            {
                "text": "Выберите метод:",
                "x": 0.1,
                "y": 1.03,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12},
            },
            {
                "text": "Выберите метрику:",
                "x": 0.5,
                "y": 1.03,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12},
            },
        ],
    )
    return fig


def _add_metric_bars(fig, rows, kind, state=None):
    """Draw metric bars once for single-method and comparison figures."""
    for metric, label, color in [
        ("f1_score", "F1 Score", "#5DA5DA"),
        ("precision", "Precision", "#FAA43A"),
        ("recall", "Recall", "#60BD68"),
    ]:
        identity_axis, value_axis = ("y", "x") if kind == "tag" else ("x", "y")
        properties = {
            identity_axis: rows[kind],
            value_axis: rows[metric],
            "name": f"{label} ({state['method']})" if state is not None else label,
            "marker_color": color,
            "hovertemplate": "<b>%{"
            + identity_axis
            + "}</b><br>"
            + ("F1" if metric == "f1_score" else label)
            + ": %{"
            + value_axis
            + ":.3f}<extra></extra>",
        }
        if kind == "tag":
            properties["orientation"] = "h"
        if state is not None:
            properties["visible"] = state["selected"] if metric == "f1_score" else False
        fig.add_trace(go.Bar(**properties))


def _add_hover_points(fig, rows, kind, texts, state=None):
    """Add unchanged contextual hover strings using one trace constructor."""
    for (_, row), text in zip(rows.iterrows(), texts):
        properties = {
            "x": [row["f1_score"]] if kind == "tag" else [row["image"]],
            "y": [row["tag"]] if kind == "tag" else [row["f1_score"]],
            "mode": "markers",
            "marker": {"size": 0},
            "hoverinfo": "text",
            "hovertext": text,
            "showlegend": False,
        }
        if state is not None:
            properties["visible"] = state["selected"]
        fig.add_trace(go.Scatter(**properties))
