"""Shared improvement and deterioration reports for notebook evaluation."""

import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display

_IMPROVEMENT = {"direction": 1, "text": {}}
_DETERIORATION = {
    "direction": -1,
    "text": {
        (
            "\n    Анализирует, какие теги показали наибольшее улучшение по количеству найд"
            "енных ранее пропущенных тегов\n    (уменьшение false negatives) при использова"
            "нии метода объединенных тегов.\n    "
        ): (
            "\n    Анализирует, какие теги показали наибольшее ухудшение по количеству ложн"
            "оположительных результатов\n    (увеличение false positives) при использовании"
            " метода объединенных тегов.\n    "
        ),
        "found_tags_improvement": "added_incorrect_tags",
        "full_fn": "full_fp",
        "merged_fn": "merged_fp",
        "Найдено пропущенных тегов": "Добавлено некорректных тегов",
        "Пропущено (полное)": "Некорректных (полное)",
        "Пропущено (объединенные)": "Некорректных (объединенные)",
        "false_negatives": "false_positives",
        (
            "<h2 style='color:#1a5276;'>Теги с наибольшим улучшением по количеству найденны"
            "х тегов (объединенные vs полное изображение)</h2>"
        ): (
            "<h2 style='color:#922B21;'>Теги с наибольшим количеством добавленных некоррект"
            "ных тегов (объединенные vs полное изображение)</h2>"
        ),
        "Количество найденных пропущенных тегов": "Количество добавленных некорректных тегов",
        "#4CAF50": "#E74C3C",
        ("<b>%{y}</b><br>Найдено пропущенных тегов: %{x}<extra></extra>"): (
            "<b>%{y}</b><br>Добавлено некорректных тегов: %{x}<extra></extra>"
        ),
        "#FAA43A": "#EC7063",
        "#60BD68": "#F1948A",
        (" тегов с наибольшим числом найденных пропущенных тегов"): (
            " тегов с наибольшим числом добавленных некорректных тегов"
        ),
        ("<br>\n                <b>Найдено пропущенных тегов:</b> "): (
            "<br>\n                <b>Добавлено некорректных тегов:</b> "
        ),
        ("<br>\n                <b>Пропущено (полное):</b> "): (
            "<br>\n                <b>Некорректных (полное):</b> "
        ),
        ("<br>\n                <b>Пропущено (объединенные):</b> "): (
            "<br>\n                <b>Некорректных (объединенные):</b> "
        ),
    },
}


def _collect_changes(full_image_tag_perf, merged_tags_tag_perf, min_occurrences, style):
    """Collect comparable tag deltas with the selected false-negative/positive policy."""
    full_df = full_image_tag_perf.set_index("tag")
    merged_df = merged_tags_tag_perf.set_index("tag")
    common_tags = set(full_df.index) & set(merged_df.index)
    changes = []
    for tag in common_tags:
        full_metrics = full_df.loc[tag]
        merged_metrics = merged_df.loc[tag]
        if full_metrics["total_ground_truth"] < min_occurrences:
            continue
        full_count = full_metrics[style["text"].get("false_negatives", "false_negatives")]
        merged_count = merged_metrics[style["text"].get("false_negatives", "false_negatives")]
        error_change = style["direction"] * (full_count - merged_count)
        changes.append(
            {
                "tag": tag,
                style["text"].get("found_tags_improvement", "found_tags_improvement"): error_change,
                "full_precision": full_metrics["precision"],
                "full_recall": full_metrics["recall"],
                "full_f1": full_metrics["f1_score"],
                "merged_precision": merged_metrics["precision"],
                "merged_recall": merged_metrics["recall"],
                "merged_f1": merged_metrics["f1_score"],
                "occurrences": full_metrics["total_ground_truth"],
                style["text"].get("full_fn", "full_fn"): full_count,
                style["text"].get("merged_fn", "merged_fn"): merged_count,
                "precision_change": merged_metrics["precision"] - full_metrics["precision"],
                "recall_change": merged_metrics["recall"] - full_metrics["recall"],
                "f1_change": merged_metrics["f1_score"] - full_metrics["f1_score"],
            }
        )
    changes_df = pd.DataFrame(changes)
    changes_df = changes_df.sort_values(
        style["text"].get("found_tags_improvement", "found_tags_improvement"), ascending=False
    )
    return changes_df


def _display_changes(changes_df, top_n, style):
    """Display the numeric change table without altering its result frame."""
    display(
        HTML(
            style["text"].get(
                (
                    "<h2 style='color:#1a5276;'>Теги с наибольшим улучшением по количеству "
                    "найденных тегов (объединенные vs полное изображение)</h2>"
                ),
                (
                    "<h2 style='color:#1a5276;'>Теги с наибольшим улучшением по количеству "
                    "найденных тегов (объединенные vs полное изображение)</h2>"
                ),
            )
        )
    )
    display_df = changes_df.head(top_n).copy()
    for col in ["precision_change", "recall_change", "f1_change"]:
        display_df[col] = display_df[col].map(lambda x: f"{x * 100:+.2f}%")
    for col in [
        "full_precision",
        "full_recall",
        "full_f1",
        "merged_precision",
        "merged_recall",
        "merged_f1",
    ]:
        display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")
    column_mapping = {
        "tag": "Тег",
        style["text"].get("found_tags_improvement", "found_tags_improvement"): style["text"].get(
            "Найдено пропущенных тегов", "Найдено пропущенных тегов"
        ),
        "precision_change": "Изменение Precision",
        "recall_change": "Изменение Recall",
        "f1_change": "Изменение F1",
        "full_precision": "Precision (полное)",
        "full_recall": "Recall (полное)",
        "full_f1": "F1 (полное)",
        "merged_precision": "Precision (объединенные)",
        "merged_recall": "Recall (объединенные)",
        "merged_f1": "F1 (объединенные)",
        "occurrences": "Вхождений в GT",
        style["text"].get("full_fn", "full_fn"): style["text"].get(
            "Пропущено (полное)", "Пропущено (полное)"
        ),
        style["text"].get("merged_fn", "merged_fn"): style["text"].get(
            "Пропущено (объединенные)", "Пропущено (объединенные)"
        ),
    }
    display_df = display_df.rename(columns=column_mapping)
    display_cols_order = [
        "Тег",
        style["text"].get("Найдено пропущенных тегов", "Найдено пропущенных тегов"),
        "Вхождений в GT",
        style["text"].get("Пропущено (полное)", "Пропущено (полное)"),
        style["text"].get("Пропущено (объединенные)", "Пропущено (объединенные)"),
        "Изменение Precision",
        "Изменение Recall",
        "Изменение F1",
        "Precision (полное)",
        "Recall (полное)",
        "F1 (полное)",
        "Precision (объединенные)",
        "Recall (объединенные)",
        "F1 (объединенные)",
    ]
    display_df = display_df[display_cols_order]
    display(HTML(display_df.to_html(classes="table table-striped table-hover", index=False)))


def _plot_changes(changes_df, top_n, style):
    """Render change bars with the same labels and colors as the notebook."""
    top_changes = changes_df.head(top_n)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=top_changes["tag"],
            x=top_changes[style["text"].get("found_tags_improvement", "found_tags_improvement")],
            name=style["text"].get("Найдено пропущенных тегов", "Найдено пропущенных тегов"),
            marker_color=style["text"].get("#4CAF50", "#4CAF50"),
            orientation="h",
            hovertemplate=style["text"].get(
                "<b>%{y}</b><br>Найдено пропущенных тегов: %{x}<extra></extra>",
                "<b>%{y}</b><br>Найдено пропущенных тегов: %{x}<extra></extra>",
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            y=top_changes["tag"],
            x=top_changes["precision_change"] * 100,
            name="Изменение Precision",
            marker_color=style["text"].get("#FAA43A", "#FAA43A"),
            orientation="h",
            visible="legendonly",
            hovertemplate="<b>%{y}</b><br>Изменение Precision: %{x:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=top_changes["tag"],
            x=top_changes["recall_change"] * 100,
            name="Изменение Recall",
            marker_color=style["text"].get("#60BD68", "#60BD68"),
            orientation="h",
            visible="legendonly",
            hovertemplate="<b>%{y}</b><br>Изменение Recall: %{x:.2f}%<extra></extra>",
        )
    )
    for _, row in top_changes.iterrows():
        fig.add_trace(
            go.Scatter(
                y=[row["tag"]],
                x=[row[style["text"].get("found_tags_improvement", "found_tags_improvement")]],
                mode="markers",
                marker={"size": 0},
                hoverinfo="text",
                hovertext="".join(
                    [
                        "\n                <b>Тег:</b> ",
                        format(row["tag"], ""),
                        "<br>\n                <b>Вхождений в GT:</b> ",
                        format(row["occurrences"], ""),
                        format(
                            style["text"].get(
                                "<br>\n                <b>Найдено пропущенных тегов:</b> ",
                                "<br>\n                <b>Найдено пропущенных тегов:</b> ",
                            ),
                            "",
                        ),
                        format(
                            row[
                                style["text"].get(
                                    "found_tags_improvement", "found_tags_improvement"
                                )
                            ],
                            "",
                        ),
                        format(
                            style["text"].get(
                                "<br>\n                <b>Пропущено (полное):</b> ",
                                "<br>\n                <b>Пропущено (полное):</b> ",
                            ),
                            "",
                        ),
                        format(row[style["text"].get("full_fn", "full_fn")], ""),
                        format(
                            style["text"].get(
                                "<br>\n                <b>Пропущено (объединенные):</b> ",
                                "<br>\n                <b>Пропущено (объединенные):</b> ",
                            ),
                            "",
                        ),
                        format(row[style["text"].get("merged_fn", "merged_fn")], ""),
                        "<br>\n                <hr>\n                <b>Изменение Precision:</b> ",
                        format(row["precision_change"] * 100, "".join([".2f"])),
                        "%<br>\n                <b>Изменение Recall:</b> ",
                        format(row["recall_change"] * 100, "".join([".2f"])),
                        "%<br>\n                <b>Изменение F1:</b> ",
                        format(row["f1_change"] * 100, "".join([".2f"])),
                        (
                            "%<br>\n                <hr>\n                <b>Полное изображ"
                            "ение:</b><br>\n                Precision: "
                        ),
                        format(row["full_precision"] * 100, "".join([".2f"])),
                        "%<br>\n                Recall: ",
                        format(row["full_recall"] * 100, "".join([".2f"])),
                        "%<br>\n                F1: ",
                        format(row["full_f1"] * 100, "".join([".2f"])),
                        (
                            "%<br>\n                <hr>\n                <b>Объединенные т"
                            "еги:</b><br>\n                Precision: "
                        ),
                        format(row["merged_precision"] * 100, "".join([".2f"])),
                        "%<br>\n                Recall: ",
                        format(row["merged_recall"] * 100, "".join([".2f"])),
                        "%<br>\n                F1: ",
                        format(row["merged_f1"] * 100, "".join([".2f"])),
                        "%\n                ",
                    ]
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="".join(
            [
                "Топ ",
                format(top_n, ""),
                format(
                    style["text"].get(
                        " тегов с наибольшим числом найденных пропущенных тегов",
                        " тегов с наибольшим числом найденных пропущенных тегов",
                    ),
                    "",
                ),
            ]
        ),
        xaxis_title=style["text"].get(
            "Количество найденных пропущенных тегов", "Количество найденных пропущенных тегов"
        ),
        yaxis_title="Тег",
        height=max(500, top_n * 25),
        hovermode="y unified",
    )
    fig.show()


def _make_change_analyzer(style, name, description):
    """Create a named public report function with the unchanged five-argument API."""

    def analyze_tag_changes(
        full_image_tag_perf: pd.DataFrame,
        merged_tags_tag_perf: pd.DataFrame,
        top_n: int = 20,
        min_occurrences: int = 3,
        interactive: bool = True,
    ) -> pd.DataFrame:
        """Analyze one configured direction using the shared reporting pipeline."""
        changes = _collect_changes(
            full_image_tag_perf, merged_tags_tag_perf, min_occurrences, style
        )
        _display_changes(changes, top_n, style)
        if interactive:
            _plot_changes(changes, top_n, style)
        return changes

    analyze_tag_changes.__name__ = name
    analyze_tag_changes.__qualname__ = name
    analyze_tag_changes.__doc__ = description
    return analyze_tag_changes


analyze_tag_improvements = _make_change_analyzer(
    _IMPROVEMENT,
    "analyze_tag_improvements",
    "Report recovered tags using the existing notebook table and chart.",
)
analyze_tag_deteriorations = _make_change_analyzer(
    _DETERIORATION,
    "analyze_tag_deteriorations",
    "Report additional incorrect tags using the existing table and chart.",
)
