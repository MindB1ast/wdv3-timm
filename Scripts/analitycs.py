"""Notebook evaluation, tables, and compatible chart exports."""

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display

from .analytics_changes import analyze_tag_deteriorations, analyze_tag_improvements
from .analytics_charts import (
    create_combined_metrics_chart,
    create_interactive_image_metrics_chart,
    create_interactive_tag_count_chart,
    create_interactive_tag_metrics_chart,
)
from .analytics_comparison_charts import (
    create_combined_image_metrics_chart,
    create_combined_tag_metrics_chart,
)
from .analytics_stats import calculate_average_tag_count
from .analytics_text import wrap_text
from .evaluation import aligned_samples, ground_truth_tags
from .evaluation import model_tags as extract_model_tags
from .evaluation import tag_metrics as evaluate_tags

__all__ = [
    "load_ground_truth_tags",
    "load_model_tags",
    "filter_known_tags",
    "calculate_tag_metrics",
    "analyze_tagging_errors",
    "analyze_tag_performance",
    "display_metrics_summary",
    "create_interactive_image_metrics_chart",
    "create_interactive_tag_metrics_chart",
    "display_tags_inline",
    "display_worst_performing_images",
    "wrap_text",
    "display_worst_performing_tags",
    "create_combined_metrics_chart",
    "create_combined_image_metrics_chart",
    "create_combined_tag_metrics_chart",
    "calculate_average_tag_count",
    "display_tag_count_statistics",
    "create_interactive_tag_count_chart",
    "analyze_tag_improvements",
    "analyze_tag_deteriorations",
    "compare_tagging_methods",
]


def load_ground_truth_tags(txt_folder, recursive=False, image_root=None):
    """Load labels, optionally preserving recursive dataset identities."""
    return ground_truth_tags(txt_folder, recursive=recursive, image_root=image_root)


def load_model_tags(results, use_merged=True, image_root=None):
    """Extract predictions without silently overwriting duplicate basenames."""
    return extract_model_tags(results, use_merged=use_merged, image_root=image_root)


def filter_known_tags(tags: Set[str], model_labels_path: str) -> Set[str]:
    """Фильтрует теги на основе известных модели тегов."""
    model_labels = pd.read_csv(model_labels_path)
    known_tags = set(model_labels["name"].str.replace("_", " ").tolist())

    return {tag for tag in tags if tag in known_tags}


def calculate_tag_metrics(ground_truth, model_tags, *, return_diagnostics=False):
    """Evaluate all labels; optionally return coverage and the F1 definition."""
    return evaluate_tags(ground_truth, model_tags, return_diagnostics=return_diagnostics)


def analyze_tagging_errors(
    ground_truth: Dict[str, Set[str]], model_tags: Dict[str, Set[str]], output_file: str = None
) -> pd.DataFrame:
    """Analyzes tagging errors between ground truth and model predictions."""
    # Find common images in ground truth and model results
    ground_truth, model_tags = aligned_samples(ground_truth, model_tags)
    common_images = sorted(ground_truth)

    # Prepare data for analysis
    results = []

    for img in common_images:
        true_tags = ground_truth[img]
        pred_tags = model_tags[img]

        # Find false positives (predicted but not in ground truth)
        false_positives = pred_tags - true_tags

        # Find false negatives (in ground truth but not predicted)
        false_negatives = true_tags - pred_tags

        # Find true positives (correctly predicted)
        true_positives = true_tags & pred_tags

        # Calculate per-image metrics
        precision = len(true_positives) / len(pred_tags) if pred_tags else 1.0
        recall = len(true_positives) / len(true_tags) if true_tags else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append(
            {
                "image": img,
                "ground_truth_tags": ", ".join(sorted(true_tags)),
                "predicted_tags": ", ".join(sorted(pred_tags)),
                "correct_tags": ", ".join(sorted(true_positives)),
                "false_positives": ", ".join(sorted(false_positives)),
                "false_negatives": ", ".join(sorted(false_negatives)),
                "num_false_positives": len(false_positives),
                "num_false_negatives": len(false_negatives),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by F1 score (worst performing images first)
    if not df.empty:
        df = df.sort_values("f1_score")

    # Save to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Error analysis saved to {output_file}")

    return df


def analyze_tag_performance(
    ground_truth: Dict[str, Set[str]], model_tags: Dict[str, Set[str]], output_file: str = None
) -> pd.DataFrame:
    """Analyzes performance of individual tags across all images."""
    # Find common images
    ground_truth, model_tags = aligned_samples(ground_truth, model_tags)
    common_images = sorted(ground_truth)

    # Get all unique tags
    all_tags = set()
    for img in common_images:
        all_tags.update(ground_truth[img])
        all_tags.update(model_tags[img])

    # Track metrics for each tag
    tag_metrics = {}

    for tag in all_tags:
        # Initialize counters
        tp, fp, fn = 0, 0, 0

        for img in common_images:
            # Tag is in ground truth
            if tag in ground_truth[img]:
                # Tag is also predicted (true positive)
                if tag in model_tags[img]:
                    tp += 1
                # Tag is not predicted (false negative)
                else:
                    fn += 1
            # Tag is not in ground truth but is predicted (false positive)
            elif tag in model_tags[img]:
                fp += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics
        tag_metrics[tag] = {
            "tag": tag,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_ground_truth": tp + fn,
            "total_predicted": tp + fp,
        }

    # Create DataFrame
    df = pd.DataFrame(list(tag_metrics.values()))

    # Sort by lowest F1 score (worst performing tags first)
    if not df.empty:
        df = df.sort_values("f1_score")

    # Save to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Tag performance analysis saved to {output_file}")

    return df


def display_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Отображает сводку метрик для всех методов в формате таблицы."""
    # Создаем DataFrame для метрик
    metrics_df = pd.DataFrame(metrics).T * 100  # Переводим в проценты для лучшей читаемости
    metrics_df = metrics_df.round(2)  # Округляем до 2 знаков после запятой

    # Добавляем столбец сортировки и сортируем по F1-score
    metrics_df = metrics_df.sort_values("F1-score", ascending=False)

    # Отображаем метрики с заголовком
    display(HTML("<h2 style='color:#2c3e50;'>Сводка метрик (в %)</h2>"))
    display(HTML(metrics_df.to_html(classes="table table-striped table-hover")))

    return metrics_df


def display_tags_inline(tags_string):
    """Отображает теги в строку без переносов на новую строку."""
    if not tags_string:
        return "Нет"
    return tags_string


def display_worst_performing_images(error_df: pd.DataFrame, method_name: str, n: int = 5) -> None:
    """Отображает детальную информацию о худших изображениях в формате HTML таблицы."""
    worst_images = error_df.nsmallest(n, "f1_score")

    # Форматируем теги для вывода в строку
    formatted_df = worst_images.copy()

    # Создаем HTML таблицу с метриками для каждого изображения
    display(HTML(f"<h3 style='color:#2c3e50;'>Худшие {n} изображений: {method_name}</h3>"))

    # Выбираем и переименовываем колонки для отображения
    display_cols = {
        "image": "Изображение",
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "num_false_positives": "Кол-во FP",
        "num_false_negatives": "Кол-во FN",
    }

    metrics_df = formatted_df[display_cols.keys()].rename(columns=display_cols)
    metrics_df[["F1", "Precision", "Recall"]] = metrics_df[["F1", "Precision", "Recall"]].apply(
        lambda column: column.map(lambda value: f"{value:.2f}")
    )

    display(HTML(metrics_df.to_html(classes="table table-striped table-hover", index=False)))

    # Для каждого изображения создаем таблицу с тегами, но не разбиваем их на строки
    for _, row in formatted_df.iterrows():
        display(HTML(f"<h4 style='color:#2c3e50;'>Изображение: {row['image']}</h4>"))

        # Создаем таблицу для тегов
        tags_data = [
            {"Категория": "Правильные теги", "Теги": display_tags_inline(row["correct_tags"])},
            {
                "Категория": "Ложноположительные теги",
                "Теги": display_tags_inline(row["false_positives"]),
            },
            {
                "Категория": "Ложноотрицательные теги",
                "Теги": display_tags_inline(row["false_negatives"]),
            },
        ]
        tags_df = pd.DataFrame(tags_data)

        # Задаем стиль для таблицы, чтобы теги не переносились
        display(
            HTML("""
        <style>
        .tag-table td {
            white-space: nowrap;
            overflow: auto;
            max-width: 500px;
        }
        </style>
        """)
        )

        display(HTML(tags_df.to_html(classes="table table-striped tag-table", index=False)))


def display_worst_performing_tags(tag_df: pd.DataFrame, method_name: str, n: int = 5) -> None:
    """Отображает детальную информацию о худших тегах в формате HTML таблицы."""
    # Фильтруем теги, которые встречаются хотя бы 3 раза
    filtered_df = tag_df[tag_df["total_ground_truth"] >= 3]
    worst_tags = filtered_df.nsmallest(n, "f1_score")

    # Создаем HTML таблицу
    display(
        HTML(f"<h3 style='color:#2c3e50;'>Худшие {n} тегов: {method_name} (мин. 3 вхождения)</h3>")
    )

    # Выбираем и переименовываем колонки для отображения
    display_cols = {
        "tag": "Тег",
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "true_positives": "TP",
        "false_positives": "FP",
        "false_negatives": "FN",
        "total_ground_truth": "Всего в GT",
        "total_predicted": "Всего предсказано",
    }

    display_df = worst_tags[display_cols.keys()].rename(columns=display_cols)
    display_df[["F1", "Precision", "Recall"]] = display_df[["F1", "Precision", "Recall"]].apply(
        lambda column: column.map(lambda value: f"{value:.2f}")
    )

    display(HTML(display_df.to_html(classes="table table-striped table-hover", index=False)))


def display_tag_count_statistics(
    ground_truth: Dict[str, Set[str]], method_tags: Dict[str, Dict[str, Set[str]]]
) -> None:
    """
    Отображает статистику по количеству тегов для ground truth и каждого метода.

    Args:
        ground_truth: Словарь с эталонными тегами
        method_tags: Словарь с тегами для каждого метода
    """
    # Вычисляем статистику для ground truth
    gt_stats = calculate_average_tag_count(ground_truth)

    # Вычисляем статистику для каждого метода
    method_stats = {
        method: calculate_average_tag_count(tags) for method, tags in method_tags.items()
    }

    # Подготавливаем DataFrame для отображения
    stats_data = []

    # Добавляем строку для ground truth
    stats_data.append(
        {
            "Метод": "Ground Truth",
            "Среднее кол-во тегов": f"{gt_stats['avg']:.2f}",
            "Медиана": f"{gt_stats['median']}",
            "Минимум": f"{gt_stats['min']}",
            "Максимум": f"{gt_stats['max']}",
        }
    )

    # Добавляем строки для каждого метода
    for method, stats in method_stats.items():
        stats_data.append(
            {
                "Метод": method,
                "Среднее кол-во тегов": f"{stats['avg']:.2f}",
                "Медиана": f"{stats['median']}",
                "Минимум": f"{stats['min']}",
                "Максимум": f"{stats['max']}",
            }
        )

    # Создаем и отображаем таблицу
    stats_df = pd.DataFrame(stats_data)
    display(HTML("<h3 style='color:#2c3e50;'>Статистика по количеству тегов</h3>"))
    display(HTML(stats_df.to_html(classes="table table-striped table-hover", index=False)))

    # Создаем столбчатую диаграмму для среднего количества тегов
    methods = [row["Метод"] for row in stats_data]
    avg_counts = [float(row["Среднее кол-во тегов"]) for row in stats_data]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, avg_counts, color=["#3498db", "#2ecc71", "#e74c3c"])
    plt.title("Среднее количество тегов по методам", fontsize=14)
    plt.ylabel("Количество тегов", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Добавляем значения над столбцами
    for column in bars:
        height = column.get_height()
        plt.text(
            column.get_x() + column.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()
    plt.show()


# Preserve the existing notebook calling convention.
# pylint: disable-next=R0913,R0917
def compare_tagging_methods(
    image_folder: str,
    txt_folder: str,
    results: Tuple,
    model_labels_path: str,
    filter_tags: bool = True,
    num_worst_examples: int = 5,
    num_top_improved: int = 20,
    num_top_deteriorated: int = 20,
    visualize: bool = True,
    interactive: bool = True,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compare tagging methods using the unchanged legacy notebook signature."""
    return _compare_methods(
        _ComparisonInput(image_folder, txt_folder, results, model_labels_path),
        _ComparisonOptions(
            filter_tags,
            num_worst_examples,
            num_top_improved,
            num_top_deteriorated,
            visualize,
            interactive,
        ),
    )


@dataclass(frozen=True)
class _ComparisonInput:
    """Dataset paths and predictions for one notebook comparison."""

    image_folder: str
    txt_folder: str
    results: tuple
    labels_path: str


@dataclass(frozen=True)
class _ComparisonOptions:
    """Presentation options independent of evaluation data."""

    filter_tags: bool
    num_worst_examples: int
    num_top_improved: int
    num_top_deteriorated: int
    visualize: bool
    interactive: bool


def _comparison_intro(truth, methods, interactive):
    """Show tag counts and the explicit historical F1 definition."""
    print("F1-score: harmonic mean of samples-averaged Precision and Recall")
    display(HTML("<h1 style='color:#1a5276;'>Сравнение методов тегирования</h1>"))
    display_tag_count_statistics(truth, methods)
    if interactive:
        display(
            HTML("<h3 style='color:#2c3e50;'>Интерактивная статистика по количеству тегов</h3>")
        )
        create_interactive_tag_count_chart(truth, methods).show()


def _evaluate_comparison_method(truth, predictions, prefix):
    """Evaluate every labelled sample and write method-specific CSV tables."""
    metrics, diagnostics = calculate_tag_metrics(truth, predictions, return_diagnostics=True)
    label = "Full-image" if prefix == "full_image" else "Merged"
    print(label + " coverage:", diagnostics)
    errors = analyze_tagging_errors(truth, predictions, prefix + "_errors.csv")
    performance = analyze_tag_performance(truth, predictions, prefix + "_tag_performance.csv")
    return metrics, errors, performance


def _comparison_details(errors, performance, count):
    """Render the same per-image and per-tag details for both methods."""
    names = {"Полное изображение": "полного изображения", "Объединенные теги": "объединенных тегов"}
    for method, suffix in names.items():
        display(HTML("<h2 style='color:#1a5276;'>Подробный анализ метода " + suffix + "</h2>"))
        display_worst_performing_images(errors[method], "Метод " + suffix, count)
        display_worst_performing_tags(performance[method], "Метод " + suffix, count)


def _comparison_charts(metrics, errors, performance, options):
    """Show optional combined figures without growing the evaluation function."""
    if not (options.visualize and options.interactive):
        return
    display(HTML("<h2 style='color:#1a5276;'>Интерактивные графики для сравнения методов</h2>"))
    for label, figure in [
        ("Сравнение общих метрик", create_combined_metrics_chart(metrics)),
        (
            "Сравнение по худшим изображениям",
            create_combined_image_metrics_chart(errors, options.num_worst_examples),
        ),
        ("Сравнение по худшим тегам", create_combined_tag_metrics_chart(performance, 20)),
    ]:
        display(HTML("<h3 style='color:#2c3e50;'>" + label + "</h3>"))
        figure.show()


def _compare_methods(inputs, options):
    """Compose validated loading, evaluation, presentation and change analysis."""
    truth = load_ground_truth_tags(
        inputs.txt_folder, recursive=True, image_root=inputs.image_folder
    )
    methods = {
        "Полное изображение": load_model_tags(
            inputs.results, use_merged=False, image_root=inputs.image_folder
        ),
        "Объединенные теги": load_model_tags(
            inputs.results, use_merged=True, image_root=inputs.image_folder
        ),
    }
    if options.filter_tags:
        truth = {key: filter_known_tags(tags, inputs.labels_path) for key, tags in truth.items()}
        methods = {
            method: {
                key: filter_known_tags(tags, inputs.labels_path) for key, tags in values.items()
            }
            for method, values in methods.items()
        }
    _comparison_intro(truth, methods, options.interactive)
    metrics, errors, performance = {}, {}, {}
    for method, predictions in methods.items():
        prefix = "full_image" if method == "Полное изображение" else "merged_tags"
        metrics[method], errors[method], performance[method] = _evaluate_comparison_method(
            truth, predictions, prefix
        )
    display_metrics_summary(metrics)
    _comparison_details(errors, performance, options.num_worst_examples)
    _comparison_charts(metrics, errors, performance, options)
    improved = analyze_tag_improvements(
        performance["Полное изображение"],
        performance["Объединенные теги"],
        top_n=options.num_top_improved,
        min_occurrences=3,
        interactive=options.interactive,
    )
    worsened = analyze_tag_deteriorations(
        performance["Полное изображение"],
        performance["Объединенные теги"],
        top_n=options.num_top_deteriorated,
        min_occurrences=3,
        interactive=options.interactive,
    )
    return metrics, errors, performance, improved, worsened
