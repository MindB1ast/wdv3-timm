import os
import pandas as pd
from typing import List, Dict, Set, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML

def load_ground_truth_tags(txt_folder: str) -> Dict[str, Set[str]]:
    """Загружает теги из txt файлов в указанной папке."""
    ground_truth = {}
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            img_name = os.path.splitext(txt_file)[0]
            with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8') as f:
                tags = {tag.strip() for tag in f.read().replace('_', ' ').split(',')}
                ground_truth[img_name] = tags
    return ground_truth

def load_model_tags(results, use_merged: bool = True) -> Dict[str, Set[str]]:
    """Извлекает теги из результатов модели."""
    model_tags = {}

    # Работаем с merged_tags или full_image
    for result in results[1] if use_merged else results[0]:
        img_name = os.path.splitext(os.path.basename(result['image_path']))[0]

        # Извлекаем теги
        tags_str = result['merged_tags']['taglist'] if use_merged else result['full_image']['taglist']
        tags = {tag.strip() for tag in tags_str.split(',')}

        model_tags[img_name] = tags

    return model_tags

def filter_known_tags(tags: Set[str], model_labels_path: str) -> Set[str]:
    """Фильтрует теги на основе известных модели тегов."""
    model_labels = pd.read_csv(model_labels_path)
    known_tags = set(model_labels['name'].str.replace('_', ' ').tolist())

    return {tag for tag in tags if tag in known_tags}

def calculate_tag_metrics(ground_truth: Dict[str, Set[str]],
                         model_tags: Dict[str, Set[str]]) -> Dict[str, float]:
    """Вычисляет метрики Precision, Recall и F1-score для тегов."""
    # Находим общие изображения в ground truth и результатах модели
    common_images = set(ground_truth.keys()) & set(model_tags.keys())

    # Подготавливаем списки для вычисления метрик
    y_true, y_pred = [], []

    # Создаем словарь для всех уникальных тегов
    all_tags = set()
    for tags in ground_truth.values():
        all_tags.update(tags)
    for tags in model_tags.values():
        all_tags.update(tags)

    # Преобразуем теги в бинарные векторы
    for img in common_images:
        img_true_tags = ground_truth[img]
        img_pred_tags = model_tags[img]

        # Создаем бинарный вектор для каждого изображения
        img_true_vec = [1 if tag in img_true_tags else 0 for tag in sorted(all_tags)]
        img_pred_vec = [1 if tag in img_pred_tags else 0 for tag in sorted(all_tags)]

        y_true.append(img_true_vec)
        y_pred.append(img_pred_vec)

    # Вычисляем метрики для каждого тега
    precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
    recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #f1 = f1_score(y_true, y_pred, average='samples', zero_division=0) #убрал потому что почему то не корректно считал

    return {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

def analyze_tagging_errors(
    ground_truth: Dict[str, Set[str]],
    model_tags: Dict[str, Set[str]],
    output_file: str = None
) -> pd.DataFrame:
    """Analyzes tagging errors between ground truth and model predictions."""
    # Find common images in ground truth and model results
    common_images = set(ground_truth.keys()) & set(model_tags.keys())

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

        results.append({
            'image': img,
            'ground_truth_tags': ', '.join(sorted(true_tags)),
            'predicted_tags': ', '.join(sorted(pred_tags)),
            'correct_tags': ', '.join(sorted(true_positives)),
            'false_positives': ', '.join(sorted(false_positives)),
            'false_negatives': ', '.join(sorted(false_negatives)),
            'num_false_positives': len(false_positives),
            'num_false_negatives': len(false_negatives),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by F1 score (worst performing images first)
    df = df.sort_values('f1_score')

    # Save to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Error analysis saved to {output_file}")

    return df

def analyze_tag_performance(
    ground_truth: Dict[str, Set[str]],
    model_tags: Dict[str, Set[str]],
    output_file: str = None
) -> pd.DataFrame:
    """Analyzes performance of individual tags across all images."""
    # Find common images
    common_images = set(ground_truth.keys()) & set(model_tags.keys())

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
            'tag': tag,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_ground_truth': tp + fn,
            'total_predicted': tp + fp
        }

    # Create DataFrame
    df = pd.DataFrame(list(tag_metrics.values()))

    # Sort by lowest F1 score (worst performing tags first)
    df = df.sort_values('f1_score')

    # Save to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Tag performance analysis saved to {output_file}")

    return df

def display_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Отображает сводку метрик для всех методов в формате таблицы."""
    # Создаем DataFrame для метрик
    metrics_df = pd.DataFrame(metrics).T * 100  # Переводим в проценты для лучшей читаемости
    metrics_df = metrics_df.round(2)  # Округляем до 2 знаков после запятой

    # Добавляем столбец сортировки и сортируем по F1-score
    metrics_df = metrics_df.sort_values('F1-score', ascending=False)

    # Отображаем метрики с заголовком
    display(HTML("<h2 style='color:#2c3e50;'>Сводка метрик (в %)</h2>"))
    display(HTML(metrics_df.to_html(classes='table table-striped table-hover')))

    return metrics_df

def create_interactive_image_metrics_chart(error_df: pd.DataFrame, method_name: str, n: int = 10) -> go.Figure:
    worst_images = error_df.nsmallest(n, 'f1_score')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=worst_images['image'],
        y=worst_images['f1_score'],
        name='F1 Score',
        marker_color='#5DA5DA',
        hovertemplate='<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=worst_images['image'],
        y=worst_images['precision'],
        name='Precision',
        marker_color='#FAA43A',
        hovertemplate='<b>%{x}</b><br>Precision: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=worst_images['image'],
        y=worst_images['recall'],
        name='Recall',
        marker_color='#60BD68',
        hovertemplate='<b>%{x}</b><br>Recall: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Худшие {n} изображений: {method_name}',
        xaxis_title='Изображение',
        yaxis_title='Значение метрики',
        barmode='group',
        height=500,
        hovermode='x unified',
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(args=[{"visible": [True, False, False]}], label="F1 Score", method="update"),
                    dict(args=[{"visible": [False, True, False]}], label="Precision", method="update"),
                    dict(args=[{"visible": [False, False, True]}], label="Recall", method="update"),
                    dict(args=[{"visible": [True, True, True]}], label="All Metrics", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    for i, row in worst_images.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['image']],
            y=[row['f1_score']],
            mode='markers',
            marker=dict(size=0),
            hoverinfo='text',
            hovertext=f"""
            <b>Изображение:</b> {row['image']}<br>
            <b>F1 Score:</b> {row['f1_score']:.3f}<br>
            <b>Precision:</b> {row['precision']:.3f}<br>
            <b>Recall:</b> {row['recall']:.3f}<br>
            <b>Правильные теги:</b> {wrap_text(row['correct_tags'], width=50)}<br>
            <b>Ложноположительные теги:</b> {wrap_text(row['false_positives'], width=50)}<br>
            <b>Ложноотрицательные теги:</b> {wrap_text(row['false_negatives'], width=50)}
            """,
            showlegend=False
        ))

    return fig

def create_interactive_tag_metrics_chart(tag_df: pd.DataFrame, method_name: str, n: int = 20) -> go.Figure:
    """Создает интерактивный график с метриками для тегов с использованием Plotly."""
    # Фильтруем теги, которые встречаются хотя бы 3 раза
    filtered_df = tag_df[tag_df['total_ground_truth'] >= 3]

    # Получаем n худших тегов
    worst_tags = filtered_df.nsmallest(n, 'f1_score')

    # Создаем фигуру с тремя метриками
    fig = go.Figure()

    # Добавляем столбцы для каждой метрики
    fig.add_trace(go.Bar(
        y=worst_tags['tag'],
        x=worst_tags['f1_score'],
        name='F1 Score',
        marker_color='#5DA5DA',
        orientation='h',
        hovertemplate='<b>%{y}</b><br>F1: %{x:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        y=worst_tags['tag'],
        x=worst_tags['precision'],
        name='Precision',
        marker_color='#FAA43A',
        orientation='h',
        hovertemplate='<b>%{y}</b><br>Precision: %{x:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        y=worst_tags['tag'],
        x=worst_tags['recall'],
        name='Recall',
        marker_color='#60BD68',
        orientation='h',
        hovertemplate='<b>%{y}</b><br>Recall: %{x:.3f}<extra></extra>'
    ))

    # Настраиваем макет графика
    fig.update_layout(
        title=f'Худшие {n} тегов: {method_name} (мин. 3 вхождения)',
        xaxis_title='Значение метрики',
        yaxis_title='Тег',
        barmode='group',
        height=max(500, n*25),  # Динамическая высота в зависимости от количества тегов
        hovermode='y unified'
    )

    # Добавляем переключатель для метрик
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{"visible": [True, False, False]}],
                        label="F1 Score",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False]}],
                        label="Precision",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False, True]}],
                        label="Recall",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True, True]}],
                        label="All Metrics",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Добавляем подсказки при наведении
    for i, row in worst_tags.iterrows():
        fig.add_trace(go.Scatter(
            y=[row['tag']],
            x=[row['f1_score']],
            mode='markers',
            marker=dict(size=0),
            hoverinfo='text',
            hovertext=f"""
            <b>Тег:</b> {row['tag']}<br>
            <b>F1 Score:</b> {row['f1_score']:.3f}<br>
            <b>Precision:</b> {row['precision']:.3f}<br>
            <b>Recall:</b> {row['recall']:.3f}<br>
            <b>True Positives:</b> {row['true_positives']}<br>
            <b>False Positives:</b> {row['false_positives']}<br>
            <b>False Negatives:</b> {row['false_negatives']}<br>
            <b>Всего в Ground Truth:</b> {row['total_ground_truth']}<br>
            <b>Всего предсказано:</b> {row['total_predicted']}
            """,
            showlegend=False
        ))

    return fig

def display_tags_inline(tags_string):
    """Отображает теги в строку без переносов на новую строку."""
    if not tags_string:
        return "Нет"
    return tags_string

def display_worst_performing_images(error_df: pd.DataFrame, method_name: str, n: int = 5) -> None:
    """Отображает детальную информацию о худших изображениях в формате HTML таблицы."""
    worst_images = error_df.nsmallest(n, 'f1_score')

    # Форматируем теги для вывода в строку
    formatted_df = worst_images.copy()

    # Создаем HTML таблицу с метриками для каждого изображения
    display(HTML(f"<h3 style='color:#2c3e50;'>Худшие {n} изображений: {method_name}</h3>"))

    # Выбираем и переименовываем колонки для отображения
    display_cols = {
        'image': 'Изображение',
        'f1_score': 'F1',
        'precision': 'Precision',
        'recall': 'Recall',
        'num_false_positives': 'Кол-во FP',
        'num_false_negatives': 'Кол-во FN'
    }

    metrics_df = formatted_df[display_cols.keys()].rename(columns=display_cols)
    metrics_df[['F1', 'Precision', 'Recall']] = metrics_df[['F1', 'Precision', 'Recall']].map(lambda x: f"{x:.2f}")

    display(HTML(metrics_df.to_html(classes='table table-striped table-hover', index=False)))

    # Для каждого изображения создаем таблицу с тегами, но не разбиваем их на строки
    for idx, row in formatted_df.iterrows():
        display(HTML(f"<h4 style='color:#2c3e50;'>Изображение: {row['image']}</h4>"))

        # Создаем таблицу для тегов
        tags_data = [
            {"Категория": "Правильные теги", "Теги": display_tags_inline(row['correct_tags'])},
            {"Категория": "Ложноположительные теги", "Теги": display_tags_inline(row['false_positives'])},
            {"Категория": "Ложноотрицательные теги", "Теги": display_tags_inline(row['false_negatives'])}
        ]
        tags_df = pd.DataFrame(tags_data)

        # Задаем стиль для таблицы, чтобы теги не переносились
        display(HTML("""
        <style>
        .tag-table td {
            white-space: nowrap;
            overflow: auto;
            max-width: 500px;
        }
        </style>
        """))

        display(HTML(tags_df.to_html(classes='table table-striped tag-table', index=False)))

def wrap_text(text: str, width: int = 50) -> str:
    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))


def display_worst_performing_tags(tag_df: pd.DataFrame, method_name: str, n: int = 5) -> None:
    """Отображает детальную информацию о худших тегах в формате HTML таблицы."""
    # Фильтруем теги, которые встречаются хотя бы 3 раза
    filtered_df = tag_df[tag_df['total_ground_truth'] >= 3]
    worst_tags = filtered_df.nsmallest(n, 'f1_score')

    # Создаем HTML таблицу
    display(HTML(f"<h3 style='color:#2c3e50;'>Худшие {n} тегов: {method_name} (мин. 3 вхождения)</h3>"))

    # Выбираем и переименовываем колонки для отображения
    display_cols = {
        'tag': 'Тег',
        'f1_score': 'F1',
        'precision': 'Precision',
        'recall': 'Recall',
        'true_positives': 'TP',
        'false_positives': 'FP',
        'false_negatives': 'FN',
        'total_ground_truth': 'Всего в GT',
        'total_predicted': 'Всего предсказано'
    }

    display_df = worst_tags[display_cols.keys()].rename(columns=display_cols)
    display_df[['F1', 'Precision', 'Recall']] = display_df[['F1', 'Precision', 'Recall']].map(lambda x: f"{x:.2f}")

    display(HTML(display_df.to_html(classes='table table-striped table-hover', index=False)))

def create_combined_metrics_chart(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """Создает единый интерактивный график для сравнения метрик обоих методов."""
    # Подготавливаем данные для графика
    methods = list(metrics.keys())
    metric_types = list(metrics[methods[0]].keys())

    fig = go.Figure()

    for metric in metric_types:
        fig.add_trace(go.Bar(
            x=methods,
            y=[metrics[method][metric] * 100 for method in methods],  # переводим в проценты
            name=metric,
            marker_color={'Precision': '#FAA43A', 'Recall': '#60BD68', 'F1-score': '#5DA5DA'}[metric],
            hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title='Сравнение методов тегирования',
        xaxis_title='Метод',
        yaxis_title='Значение (%)',
        barmode='group',
        height=500,
        hovermode='x unified',
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(args=[{"visible": [True, False, False]}], label="Precision", method="update"),
                    dict(args=[{"visible": [False, True, False]}], label="Recall", method="update"),
                    dict(args=[{"visible": [False, False, True]}], label="F1 Score", method="update"),
                    dict(args=[{"visible": [True, True, True]}], label="All Metrics", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    return fig

def create_combined_image_metrics_chart(error_dfs: Dict[str, pd.DataFrame], n: int = 10) -> go.Figure:
    """Создает единый интерактивный график для сравнения метрик изображений с возможностью переключения между методами."""
    fig = go.Figure()

    # Получаем худшие изображения для каждого метода
    methods = list(error_dfs.keys())

    # Добавляем следы для каждого метода
    for method in methods:
        worst_images = error_dfs[method].nsmallest(n, 'f1_score')

        # Добавляем F1 Score
        fig.add_trace(go.Bar(
            x=worst_images['image'],
            y=worst_images['f1_score'],
            name=f'F1 Score ({method})',
            marker_color='#5DA5DA',
            visible=(method == methods[0]),  # первый метод видимый по умолчанию
            hovertemplate='<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>'
        ))

        # Добавляем Precision
        fig.add_trace(go.Bar(
            x=worst_images['image'],
            y=worst_images['precision'],
            name=f'Precision ({method})',
            marker_color='#FAA43A',
            visible=False,  # скрыты по умолчанию
            hovertemplate='<b>%{x}</b><br>Precision: %{y:.3f}<extra></extra>'
        ))

        # Добавляем Recall
        fig.add_trace(go.Bar(
            x=worst_images['image'],
            y=worst_images['recall'],
            name=f'Recall ({method})',
            marker_color='#60BD68',
            visible=False,  # скрыты по умолчанию
            hovertemplate='<b>%{x}</b><br>Recall: %{y:.3f}<extra></extra>'
        ))

        # Добавляем подсказки при наведении
        for i, row in worst_images.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['image']],
                y=[row['f1_score']],
                mode='markers',
                marker=dict(size=0),
                hoverinfo='text',
                visible=(method == methods[0]),  # первый метод видимый по умолчанию
                hovertext=f"""
                <b>Изображение:</b> {row['image']}<br>
                <b>Метод:</b> {method}<br>
                <b>F1 Score:</b> {row['f1_score']:.3f}<br>
                <b>Precision:</b> {row['precision']:.3f}<br>
                <b>Recall:</b> {row['recall']:.3f}<br>
                <b>Правильные теги:</b> {wrap_text(row['correct_tags'], width=50)}<br>
                <b>Ложноположительные теги:</b> {wrap_text(row['false_positives'], width=50)}<br>
                <b>Ложноотрицательные теги:</b> {wrap_text(row['false_negatives'], width=50)}
                """,
                showlegend=False
            ))

    # Создаем кнопки для переключения между методами
    method_buttons = []
    for i, method in enumerate(methods):
        # Вычисляем, какие следы должны быть видимыми для каждого метода
        visibilities = []
        for j in range(len(methods)):
            # Для каждого метода у нас 4 следа (3 метрики + подсказки)
            if j == i:
                visibilities.extend([True, False, False] + [True] * len(error_dfs[method].nsmallest(n, 'f1_score')))
            else:
                visibilities.extend([False, False, False] + [False] * len(error_dfs[list(error_dfs.keys())[j]].nsmallest(n, 'f1_score')))

        method_buttons.append(
            dict(args=[{"visible": visibilities}], label=method, method="update")
        )

    # Создаем кнопки для переключения между метриками для активного метода
    metric_buttons = []

    # F1 Score
    f1_visibilities = []
    for i, method in enumerate(methods):
        # Для активного метода показываем только F1 Score
        if i == 0:  # первый метод активен по умолчанию
            f1_visibilities.extend([True, False, False] + [True] * len(error_dfs[method].nsmallest(n, 'f1_score')))
        else:
            f1_visibilities.extend([False, False, False] + [False] * len(error_dfs[method].nsmallest(n, 'f1_score')))
    metric_buttons.append(dict(args=[{"visible": f1_visibilities}], label="F1 Score", method="update"))

    # Precision
    prec_visibilities = []
    for i, method in enumerate(methods):
        if i == 0:  # первый метод активен по умолчанию
            prec_visibilities.extend([False, True, False] + [True] * len(error_dfs[method].nsmallest(n, 'f1_score')))
        else:
            prec_visibilities.extend([False, False, False] + [False] * len(error_dfs[method].nsmallest(n, 'f1_score')))
    metric_buttons.append(dict(args=[{"visible": prec_visibilities}], label="Precision", method="update"))

    # Recall
    recall_visibilities = []
    for i, method in enumerate(methods):
        if i == 0:  # первый метод активен по умолчанию
            recall_visibilities.extend([False, False, True] + [True] * len(error_dfs[method].nsmallest(n, 'f1_score')))
        else:
            recall_visibilities.extend([False, False, False] + [False] * len(error_dfs[method].nsmallest(n, 'f1_score')))
    metric_buttons.append(dict(args=[{"visible": recall_visibilities}], label="Recall", method="update"))

    # All metrics
    all_visibilities = []
    for i, method in enumerate(methods):
        if i == 0:  # первый метод активен по умолчанию
            all_visibilities.extend([True, True, True] + [True] * len(error_dfs[method].nsmallest(n, 'f1_score')))
        else:
            all_visibilities.extend([False, False, False] + [False] * len(error_dfs[method].nsmallest(n, 'f1_score')))
    metric_buttons.append(dict(args=[{"visible": all_visibilities}], label="All Metrics", method="update"))

    # Настраиваем макет
    fig.update_layout(
        title=f'Худшие {n} изображений по методам',
        xaxis_title='Изображение',
        yaxis_title='Значение метрики',
        barmode='group',
        height=500,
        hovermode='x unified',
        # Добавляем две группы кнопок
        updatemenus=[
            # Кнопки методов
            dict(
                type="buttons",
                direction="right",
                buttons=method_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            # Кнопки метрик
            dict(
                type="buttons",
                direction="right",
                buttons=metric_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(text="Выберите метод:", x=0.1, y=1.12, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12)),
            dict(text="Выберите метрику:", x=0.5, y=1.12, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12))
        ]
    )

    return fig

def create_combined_tag_metrics_chart(tag_dfs: Dict[str, pd.DataFrame], n: int = 20) -> go.Figure:
    """Создает единый интерактивный график для сравнения метрик тегов с возможностью переключения между методами."""
    fig = go.Figure()

    # Получаем методы
    methods = list(tag_dfs.keys())

    # Для каждого метода
    for method in methods:
        # Фильтруем теги, которые встречаются хотя бы 3 раза
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags = filtered_df.nsmallest(n, 'f1_score')

        # Добавляем F1 Score
        fig.add_trace(go.Bar(
            y=worst_tags['tag'],
            x=worst_tags['f1_score'],
            name=f'F1 Score ({method})',
            marker_color='#5DA5DA',
            orientation='h',
            visible=(method == methods[0]),  # первый метод видимый по умолчанию
            hovertemplate='<b>%{y}</b><br>F1: %{x:.3f}<extra></extra>'
        ))

        # Добавляем Precision
        fig.add_trace(go.Bar(
            y=worst_tags['tag'],
            x=worst_tags['precision'],
            name=f'Precision ({method})',
            marker_color='#FAA43A',
            orientation='h',
            visible=False,  # скрыты по умолчанию
            hovertemplate='<b>%{y}</b><br>Precision: %{x:.3f}<extra></extra>'
        ))

        # Добавляем Recall
        fig.add_trace(go.Bar(
            y=worst_tags['tag'],
            x=worst_tags['recall'],
            name=f'Recall ({method})',
            marker_color='#60BD68',
            orientation='h',
            visible=False,  # скрыты по умолчанию
            hovertemplate='<b>%{y}</b><br>Recall: %{x:.3f}<extra></extra>'
        ))

        # Добавляем подсказки при наведении
        for i, row in worst_tags.iterrows():
            fig.add_trace(go.Scatter(
                y=[row['tag']],
                x=[row['f1_score']],
                mode='markers',
                marker=dict(size=0),
                hoverinfo='text',
                visible=(method == methods[0]),  # первый метод видимый по умолчанию
                hovertext=f"""
                <b>Тег:</b> {row['tag']}<br>
                <b>Метод:</b> {method}<br>
                <b>F1 Score:</b> {row['f1_score']:.3f}<br>
                <b>Precision:</b> {row['precision']:.3f}<br>
                <b>Recall:</b> {row['recall']:.3f}<br>
                <b>True Positives:</b> {row['true_positives']}<br>
                <b>False Positives:</b> {row['false_positives']}<br>
                <b>False Negatives:</b> {row['false_negatives']}<br>
                <b>Всего в Ground Truth:</b> {row['total_ground_truth']}<br>
                <b>Всего предсказано:</b> {row['total_predicted']}
                """,
                showlegend=False
            ))

    # Создаем кнопки для переключения между методами
    method_buttons = []
    for i, method in enumerate(methods):
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags_count = len(filtered_df.nsmallest(n, 'f1_score'))

        # Вычисляем, какие следы должны быть видимыми для каждого метода
        visibilities = []
        for j, m in enumerate(methods):
            filtered_df_j = tag_dfs[m][tag_dfs[m]['total_ground_truth'] >= 3]
            worst_tags_count_j = len(filtered_df_j.nsmallest(n, 'f1_score'))

            # Для каждого метода у нас 4 следа (3 метрики + подсказки)
            if j == i:
                visibilities.extend([True, False, False] + [True] * worst_tags_count_j)
            else:
                visibilities.extend([False, False, False] + [False] * worst_tags_count_j)

        method_buttons.append(
            dict(args=[{"visible": visibilities}], label=method, method="update")
        )

    # Создаем кнопки для переключения между метриками для активного метода
    metric_buttons = []

    # F1 Score (для первого метода по умолчанию)
    f1_visibilities = []
    for i, method in enumerate(methods):
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags_count = len(filtered_df.nsmallest(n, 'f1_score'))

        if i == 0:  # первый метод активен по умолчанию
            f1_visibilities.extend([True, False, False] + [True] * worst_tags_count)
        else:
            f1_visibilities.extend([False, False, False] + [False] * worst_tags_count)

    metric_buttons.append(dict(args=[{"visible": f1_visibilities}], label="F1 Score", method="update"))

    # Precision
    prec_visibilities = []
    for i, method in enumerate(methods):
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags_count = len(filtered_df.nsmallest(n, 'f1_score'))

        if i == 0:  # первый метод активен по умолчанию
            prec_visibilities.extend([False, True, False] + [True] * worst_tags_count)
        else:
            prec_visibilities.extend([False, False, False] + [False] * worst_tags_count)

    metric_buttons.append(dict(args=[{"visible": prec_visibilities}], label="Precision", method="update"))

    # Recall
    recall_visibilities = []
    for i, method in enumerate(methods):
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags_count = len(filtered_df.nsmallest(n, 'f1_score'))

        if i == 0:  # первый метод активен по умолчанию
            recall_visibilities.extend([False, False, True] + [True] * worst_tags_count)
        else:
            recall_visibilities.extend([False, False, False] + [False] * worst_tags_count)

    metric_buttons.append(dict(args=[{"visible": recall_visibilities}], label="Recall", method="update"))

    # All metrics
    all_visibilities = []
    for i, method in enumerate(methods):
        filtered_df = tag_dfs[method][tag_dfs[method]['total_ground_truth'] >= 3]
        worst_tags_count = len(filtered_df.nsmallest(n, 'f1_score'))

        if i == 0:  # первый метод активен по умолчанию
            all_visibilities.extend([True, True, True] + [True] * worst_tags_count)
        else:
            all_visibilities.extend([False, False, False] + [False] * worst_tags_count)

    metric_buttons.append(dict(args=[{"visible": all_visibilities}], label="All Metrics", method="update"))

    # Настраиваем макет
    fig.update_layout(
        title=f'Худшие {n} тегов по методам (мин. 3 вхождения)',
        xaxis_title='Значение метрики',
        yaxis_title='Тег',
        barmode='group',
        height=max(500, n*25),  # Динамическая высота
        hovermode='y unified',
        # Добавляем две группы кнопок
        updatemenus=[
            # Кнопки методов
            dict(
                type="buttons",
                direction="right",
                buttons=method_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.05,
                yanchor="top"
            ),
            # Кнопки метрик
            dict(
                type="buttons",
                direction="right",
                buttons=metric_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.05,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(text="Выберите метод:", x=0.1, y=1.03, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12)),
            dict(text="Выберите метрику:", x=0.5, y=1.03, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12))
        ]
    )

    return fig

def calculate_average_tag_count(tags_dict: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Вычисляет среднее, минимальное и максимальное количество тегов в словаре тегов.

    Args:
        tags_dict: Словарь, где ключи - имена изображений, а значения - множества тегов

    Returns:
        Словарь со статистикой по количеству тегов
    """
    if not tags_dict:
        return {"avg": 0, "min": 0, "max": 0, "median": 0}

    # Собираем количество тегов для каждого изображения
    tag_counts = [len(tags) for tags in tags_dict.values()]

    # Вычисляем статистику
    avg_count = sum(tag_counts) / len(tag_counts)
    min_count = min(tag_counts)
    max_count = max(tag_counts)
    median_count = sorted(tag_counts)[len(tag_counts) // 2]

    return {
        "avg": avg_count,
        "min": min_count,
        "max": max_count,
        "median": median_count
    }

def display_tag_count_statistics(ground_truth: Dict[str, Set[str]],
                                method_tags: Dict[str, Dict[str, Set[str]]]) -> None:
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
        method: calculate_average_tag_count(tags)
        for method, tags in method_tags.items()
    }

    # Подготавливаем DataFrame для отображения
    stats_data = []

    # Добавляем строку для ground truth
    stats_data.append({
        "Метод": "Ground Truth",
        "Среднее кол-во тегов": f"{gt_stats['avg']:.2f}",
        "Медиана": f"{gt_stats['median']}",
        "Минимум": f"{gt_stats['min']}",
        "Максимум": f"{gt_stats['max']}"
    })

    # Добавляем строки для каждого метода
    for method, stats in method_stats.items():
        stats_data.append({
            "Метод": method,
            "Среднее кол-во тегов": f"{stats['avg']:.2f}",
            "Медиана": f"{stats['median']}",
            "Минимум": f"{stats['min']}",
            "Максимум": f"{stats['max']}"
        })

    # Создаем и отображаем таблицу
    stats_df = pd.DataFrame(stats_data)
    display(HTML("<h3 style='color:#2c3e50;'>Статистика по количеству тегов</h3>"))
    display(HTML(stats_df.to_html(classes='table table-striped table-hover', index=False)))

    # Создаем столбчатую диаграмму для среднего количества тегов
    methods = [row["Метод"] for row in stats_data]
    avg_counts = [float(row["Среднее кол-во тегов"]) for row in stats_data]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, avg_counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Среднее количество тегов по методам', fontsize=14)
    plt.ylabel('Количество тегов', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.show()

def create_interactive_tag_count_chart(ground_truth: Dict[str, Set[str]],
                                     method_tags: Dict[str, Dict[str, Set[str]]]) -> go.Figure:
    """
    Создает интерактивный график для сравнения количества тегов в разных методах.

    Args:
        ground_truth: Словарь с эталонными тегами
        method_tags: Словарь с тегами для каждого метода

    Returns:
        Plotly фигуру с графиком
    """
    # Вычисляем статистику для ground truth
    gt_stats = calculate_average_tag_count(ground_truth)

    # Вычисляем статистику для каждого метода
    method_stats = {
        method: calculate_average_tag_count(tags)
        for method, tags in method_tags.items()
    }

    # Подготавливаем данные для графика
    methods = ["Ground Truth"] + list(method_stats.keys())
    avg_counts = [gt_stats["avg"]] + [stats["avg"] for stats in method_stats.values()]
    median_counts = [gt_stats["median"]] + [stats["median"] for stats in method_stats.values()]
    min_counts = [gt_stats["min"]] + [stats["min"] for stats in method_stats.values()]
    max_counts = [gt_stats["max"]] + [stats["max"] for stats in method_stats.values()]

    # Создаем фигуру
    fig = go.Figure()

    # Добавляем столбцы для среднего
    fig.add_trace(go.Bar(
        x=methods,
        y=avg_counts,
        name="Среднее",
        marker_color="#3498db",
        hovertemplate="<b>%{x}</b><br>Среднее: %{y:.2f}<extra></extra>"
    ))

    # Добавляем столбцы для медианы
    fig.add_trace(go.Bar(
        x=methods,
        y=median_counts,
        name="Медиана",
        marker_color="#2ecc71",
        hovertemplate="<b>%{x}</b><br>Медиана: %{y}<extra></extra>"
    ))

    # Добавляем scatter points для минимума и максимума
    fig.add_trace(go.Scatter(
        x=methods,
        y=min_counts,
        mode="markers",
        name="Минимум",
        marker=dict(
            symbol="triangle-down",
            size=12,
            color="#e74c3c"
        ),
        hovertemplate="<b>%{x}</b><br>Минимум: %{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=methods,
        y=max_counts,
        mode="markers",
        name="Максимум",
        marker=dict(
            symbol="triangle-up",
            size=12,
            color="#9b59b6"
        ),
        hovertemplate="<b>%{x}</b><br>Максимум: %{y}<extra></extra>"
    ))

    # Настраиваем макет
    fig.update_layout(
        title="Статистика по количеству тегов",
        xaxis_title="Метод",
        yaxis_title="Количество тегов",
        barmode="group",
        height=500,
        hovermode="x unified"
    )

    return fig

def analyze_tag_improvements(
    full_image_tag_perf: pd.DataFrame,
    merged_tags_tag_perf: pd.DataFrame,
    top_n: int = 20,
    min_occurrences: int = 3,
    interactive: bool = True
) -> pd.DataFrame:
    """
    Анализирует, какие теги показали наибольшее улучшение по количеству найденных ранее пропущенных тегов
    (уменьшение false negatives) при использовании метода объединенных тегов.
    """
    # Set index for joining
    full_df = full_image_tag_perf.set_index('tag')
    merged_df = merged_tags_tag_perf.set_index('tag')

    # Find common tags with minimum occurrences
    common_tags = set(full_df.index) & set(merged_df.index)

    # Create list to store improvement data
    improvements = []

    for tag in common_tags:
        # Get metrics for both methods
        full_metrics = full_df.loc[tag]
        merged_metrics = merged_df.loc[tag]

        # Only consider tags that appear at least min_occurrences times in ground truth
        if full_metrics['total_ground_truth'] < min_occurrences:
            continue

        # Calculate absolute values of false negatives (missed tags)
        full_fn = full_metrics['false_negatives']
        merged_fn = merged_metrics['false_negatives']

        # Calculate how many previously missed tags are now found
        found_tags_improvement = full_fn - merged_fn

        # Store improvement data
        improvements.append({
            'tag': tag,
            'found_tags_improvement': found_tags_improvement,  # Абсолютное значение улучшения
            'full_precision': full_metrics['precision'],
            'full_recall': full_metrics['recall'],
            'full_f1': full_metrics['f1_score'],
            'merged_precision': merged_metrics['precision'],
            'merged_recall': merged_metrics['recall'],
            'merged_f1': merged_metrics['f1_score'],
            'occurrences': full_metrics['total_ground_truth'],  # Общее число тегов в ground truth
            'full_fn': full_fn,  # Количество пропущенных тегов в методе полного изображения
            'merged_fn': merged_fn,  # Количество пропущенных тегов в методе объединенных тегов
            'precision_change': merged_metrics['precision'] - full_metrics['precision'],
            'recall_change': merged_metrics['recall'] - full_metrics['recall'],
            'f1_change': merged_metrics['f1_score'] - full_metrics['f1_score']
        })

    # Create DataFrame and sort by number of newly found tags
    improvements_df = pd.DataFrame(improvements)
    improvements_df = improvements_df.sort_values('found_tags_improvement', ascending=False)

    # Display results
    display(HTML("<h2 style='color:#1a5276;'>Теги с наибольшим улучшением по количеству найденных тегов (объединенные vs полное изображение)</h2>"))

    # Format for display
    display_df = improvements_df.head(top_n).copy()

    # Добавляем процентные метрики для наглядности
    for col in ['precision_change', 'recall_change', 'f1_change']:
        display_df[col] = display_df[col].map(lambda x: f"{x*100:+.2f}%")

    for col in ['full_precision', 'full_recall', 'full_f1', 'merged_precision', 'merged_recall', 'merged_f1']:
        display_df[col] = display_df[col].map(lambda x: f"{x*100:.2f}%")

    # Rename columns for display
    column_mapping = {
        'tag': 'Тег',
        'found_tags_improvement': 'Найдено пропущенных тегов',
        'precision_change': 'Изменение Precision',
        'recall_change': 'Изменение Recall',
        'f1_change': 'Изменение F1',
        'full_precision': 'Precision (полное)',
        'full_recall': 'Recall (полное)',
        'full_f1': 'F1 (полное)',
        'merged_precision': 'Precision (объединенные)',
        'merged_recall': 'Recall (объединенные)',
        'merged_f1': 'F1 (объединенные)',
        'occurrences': 'Вхождений в GT',
        'full_fn': 'Пропущено (полное)',
        'merged_fn': 'Пропущено (объединенные)'
    }

    display_df = display_df.rename(columns=column_mapping)
    display_cols_order = [
        'Тег', 'Найдено пропущенных тегов', 'Вхождений в GT',
        'Пропущено (полное)', 'Пропущено (объединенные)',
        'Изменение Precision', 'Изменение Recall', 'Изменение F1',
        'Precision (полное)', 'Recall (полное)', 'F1 (полное)',
        'Precision (объединенные)', 'Recall (объединенные)', 'F1 (объединенные)'
    ]
    display_df = display_df[display_cols_order]

    display(HTML(display_df.to_html(classes='table table-striped table-hover', index=False)))

    # Update interactive visualization if requested
    if interactive:
        # Create data for top N improved tags
        top_improved = improvements_df.head(top_n)

        # Create figure
        fig = go.Figure()

        # Add trace for found tags improvement (primary metric)
        fig.add_trace(go.Bar(
            y=top_improved['tag'],
            x=top_improved['found_tags_improvement'],
            name='Найдено пропущенных тегов',
            marker_color='#4CAF50',  # Green for improvement
            orientation='h',
            hovertemplate='<b>%{y}</b><br>Найдено пропущенных тегов: %{x}<extra></extra>'
        ))

        # Add traces for each improvement metric
        fig.add_trace(go.Bar(
            y=top_improved['tag'],
            x=top_improved['precision_change'] * 100,  # Convert to percentage
            name='Изменение Precision',
            marker_color='#FAA43A',
            orientation='h',
            visible='legendonly',  # Hide by default
            hovertemplate='<b>%{y}</b><br>Изменение Precision: %{x:.2f}%<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            y=top_improved['tag'],
            x=top_improved['recall_change'] * 100,  # Convert to percentage
            name='Изменение Recall',
            marker_color='#60BD68',
            orientation='h',
            visible='legendonly',  # Hide by default
            hovertemplate='<b>%{y}</b><br>Изменение Recall: %{x:.2f}%<extra></extra>'
        ))

        # Add hover info for each tag
        for i, row in top_improved.iterrows():
            fig.add_trace(go.Scatter(
                y=[row['tag']],
                x=[row['found_tags_improvement']],  # Use found_tags_improvement for positioning
                mode='markers',
                marker=dict(size=0),
                hoverinfo='text',
                hovertext=f"""
                <b>Тег:</b> {row['tag']}<br>
                <b>Вхождений в GT:</b> {row['occurrences']}<br>
                <b>Найдено пропущенных тегов:</b> {row['found_tags_improvement']}<br>
                <b>Пропущено (полное):</b> {row['full_fn']}<br>
                <b>Пропущено (объединенные):</b> {row['merged_fn']}<br>
                <hr>
                <b>Изменение Precision:</b> {row['precision_change']*100:.2f}%<br>
                <b>Изменение Recall:</b> {row['recall_change']*100:.2f}%<br>
                <b>Изменение F1:</b> {row['f1_change']*100:.2f}%<br>
                <hr>
                <b>Полное изображение:</b><br>
                Precision: {row['full_precision']*100:.2f}%<br>
                Recall: {row['full_recall']*100:.2f}%<br>
                F1: {row['full_f1']*100:.2f}%<br>
                <hr>
                <b>Объединенные теги:</b><br>
                Precision: {row['merged_precision']*100:.2f}%<br>
                Recall: {row['merged_recall']*100:.2f}%<br>
                F1: {row['merged_f1']*100:.2f}%
                """,
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            title=f'Топ {top_n} тегов с наибольшим числом найденных пропущенных тегов',
            xaxis_title='Количество найденных пропущенных тегов',
            yaxis_title='Тег',
            height=max(500, top_n*25),
            hovermode='y unified'
        )

        fig.show()

    return improvements_df

def analyze_tag_deteriorations(
    full_image_tag_perf: pd.DataFrame,
    merged_tags_tag_perf: pd.DataFrame,
    top_n: int = 20,
    min_occurrences: int = 3,
    interactive: bool = True
) -> pd.DataFrame:
    """
    Анализирует, какие теги показали наибольшее ухудшение по количеству ложноположительных результатов
    (увеличение false positives) при использовании метода объединенных тегов.
    """
    # Set index for joining
    full_df = full_image_tag_perf.set_index('tag')
    merged_df = merged_tags_tag_perf.set_index('tag')

    # Find common tags with minimum occurrences
    common_tags = set(full_df.index) & set(merged_df.index)

    # Create list to store deterioration data
    deteriorations = []

    for tag in common_tags:
        # Get metrics for both methods
        full_metrics = full_df.loc[tag]
        merged_metrics = merged_df.loc[tag]

        # Only consider tags that appear at least min_occurrences times in ground truth
        if full_metrics['total_ground_truth'] < min_occurrences:
            continue

        # Calculate absolute values of false positives (incorrect tags)
        full_fp = full_metrics['false_positives']
        merged_fp = merged_metrics['false_positives']

        # Calculate how many more incorrect tags were added
        added_incorrect_tags = merged_fp - full_fp

        # Store deterioration data
        deteriorations.append({
            'tag': tag,
            'added_incorrect_tags': added_incorrect_tags,  # Абсолютное значение ухудшения
            'full_precision': full_metrics['precision'],
            'full_recall': full_metrics['recall'],
            'full_f1': full_metrics['f1_score'],
            'merged_precision': merged_metrics['precision'],
            'merged_recall': merged_metrics['recall'],
            'merged_f1': merged_metrics['f1_score'],
            'occurrences': full_metrics['total_ground_truth'],  # Общее число тегов в ground truth
            'full_fp': full_fp,  # Количество неверных тегов в методе полного изображения
            'merged_fp': merged_fp,  # Количество неверных тегов в методе объединенных тегов
            'precision_change': merged_metrics['precision'] - full_metrics['precision'],
            'recall_change': merged_metrics['recall'] - full_metrics['recall'],
            'f1_change': merged_metrics['f1_score'] - full_metrics['f1_score']
        })

    # Create DataFrame and sort by number of added incorrect tags (descending)
    deteriorations_df = pd.DataFrame(deteriorations)
    deteriorations_df = deteriorations_df.sort_values('added_incorrect_tags', ascending=False)

    # Display results
    display(HTML("<h2 style='color:#922B21;'>Теги с наибольшим количеством добавленных некорректных тегов (объединенные vs полное изображение)</h2>"))

    # Format for display
    display_df = deteriorations_df.head(top_n).copy()

    # Добавляем процентные метрики для наглядности
    for col in ['precision_change', 'recall_change', 'f1_change']:
        display_df[col] = display_df[col].map(lambda x: f"{x*100:+.2f}%")

    for col in ['full_precision', 'full_recall', 'full_f1', 'merged_precision', 'merged_recall', 'merged_f1']:
        display_df[col] = display_df[col].map(lambda x: f"{x*100:.2f}%")

    # Rename columns for display
    column_mapping = {
        'tag': 'Тег',
        'added_incorrect_tags': 'Добавлено некорректных тегов',
        'precision_change': 'Изменение Precision',
        'recall_change': 'Изменение Recall',
        'f1_change': 'Изменение F1',
        'full_precision': 'Precision (полное)',
        'full_recall': 'Recall (полное)',
        'full_f1': 'F1 (полное)',
        'merged_precision': 'Precision (объединенные)',
        'merged_recall': 'Recall (объединенные)',
        'merged_f1': 'F1 (объединенные)',
        'occurrences': 'Вхождений в GT',
        'full_fp': 'Некорректных (полное)',
        'merged_fp': 'Некорректных (объединенные)'
    }

    display_df = display_df.rename(columns=column_mapping)
    display_cols_order = [
        'Тег', 'Добавлено некорректных тегов', 'Вхождений в GT',
        'Некорректных (полное)', 'Некорректных (объединенные)',
        'Изменение Precision', 'Изменение Recall', 'Изменение F1',
        'Precision (полное)', 'Recall (полное)', 'F1 (полное)',
        'Precision (объединенные)', 'Recall (объединенные)', 'F1 (объединенные)'
    ]
    display_df = display_df[display_cols_order]

    display(HTML(display_df.to_html(classes='table table-striped table-hover', index=False)))

    # Update interactive visualization if requested
    if interactive:
        # Create data for top N deteriorated tags
        top_deteriorated = deteriorations_df.head(top_n)

        # Create figure
        fig = go.Figure()

        # Add trace for added incorrect tags (primary metric)
        fig.add_trace(go.Bar(
            y=top_deteriorated['tag'],
            x=top_deteriorated['added_incorrect_tags'],
            name='Добавлено некорректных тегов',
            marker_color='#E74C3C',  # Red for deterioration
            orientation='h',
            hovertemplate='<b>%{y}</b><br>Добавлено некорректных тегов: %{x}<extra></extra>'
        ))

        # Add traces for each change metric
        fig.add_trace(go.Bar(
            y=top_deteriorated['tag'],
            x=top_deteriorated['precision_change'] * 100,  # Convert to percentage
            name='Изменение Precision',
            marker_color='#EC7063',
            orientation='h',
            visible='legendonly',  # Hide by default
            hovertemplate='<b>%{y}</b><br>Изменение Precision: %{x:.2f}%<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            y=top_deteriorated['tag'],
            x=top_deteriorated['recall_change'] * 100,  # Convert to percentage
            name='Изменение Recall',
            marker_color='#F1948A',
            orientation='h',
            visible='legendonly',  # Hide by default
            hovertemplate='<b>%{y}</b><br>Изменение Recall: %{x:.2f}%<extra></extra>'
        ))

        # Add hover info for each tag
        for i, row in top_deteriorated.iterrows():
            fig.add_trace(go.Scatter(
                y=[row['tag']],
                x=[row['added_incorrect_tags']],  # Use added_incorrect_tags for positioning
                mode='markers',
                marker=dict(size=0),
                hoverinfo='text',
                hovertext=f"""
                <b>Тег:</b> {row['tag']}<br>
                <b>Вхождений в GT:</b> {row['occurrences']}<br>
                <b>Добавлено некорректных тегов:</b> {row['added_incorrect_tags']}<br>
                <b>Некорректных (полное):</b> {row['full_fp']}<br>
                <b>Некорректных (объединенные):</b> {row['merged_fp']}<br>
                <hr>
                <b>Изменение Precision:</b> {row['precision_change']*100:.2f}%<br>
                <b>Изменение Recall:</b> {row['recall_change']*100:.2f}%<br>
                <b>Изменение F1:</b> {row['f1_change']*100:.2f}%<br>
                <hr>
                <b>Полное изображение:</b><br>
                Precision: {row['full_precision']*100:.2f}%<br>
                Recall: {row['full_recall']*100:.2f}%<br>
                F1: {row['full_f1']*100:.2f}%<br>
                <hr>
                <b>Объединенные теги:</b><br>
                Precision: {row['merged_precision']*100:.2f}%<br>
                Recall: {row['merged_recall']*100:.2f}%<br>
                F1: {row['merged_f1']*100:.2f}%
                """,
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            title=f'Топ {top_n} тегов с наибольшим числом добавленных некорректных тегов',
            xaxis_title='Количество добавленных некорректных тегов',
            yaxis_title='Тег',
            height=max(500, top_n*25),
            hovermode='y unified'
        )

        fig.show()

    return deteriorations_df



def compare_tagging_methods(
    image_folder: str,
    txt_folder: str,
    results: Tuple,
    model_labels_path: str,
    filter_tags: bool = True,
    num_worst_examples: int = 5,
    num_top_improved: int = 20,  # Параметр для топ улучшенных тегов
    num_top_deteriorated: int = 20,  # Добавленный параметр для топ ухудшенных тегов
    visualize: bool = True,
    interactive: bool = True
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Сравнивает методы тегирования: полное изображение vs объединенные теги.

    Args:
        image_folder (str): Путь к папке с изображениями
        txt_folder (str): Путь к папке с txt файлами ground truth
        results (tuple): Результаты BatchTagging
        model_labels_path (str): Путь к CSV файлу с тегами модели
        filter_tags (bool): Фильтровать теги по известным модели
        num_worst_examples (int): Количество худших примеров для вывода
        num_top_improved (int): Количество тегов с наибольшим улучшением для отображения
        num_top_deteriorated (int): Количество тегов с наибольшим ухудшением для отображения
        visualize (bool): Выполнять ли визуализацию результатов
        interactive (bool): Использовать ли интерактивные графики Plotly

    Returns:
        Tuple содержащий:
        - Словарь с общими метриками для каждого метода
        - Словарь с DataFrame ошибок для каждого метода
        - Словарь с DataFrame производительности тегов для каждого метода
        - DataFrame с улучшенными тегами
        - DataFrame с ухудшенными тегами
    """
    # Загружаем ground truth теги
    ground_truth = load_ground_truth_tags(txt_folder)

    # Загружаем теги для полного изображения и объединенные теги
    full_image_tags = load_model_tags(results, use_merged=False)
    merged_tags = load_model_tags(results, use_merged=True)

    if filter_tags:
        ground_truth = {
            img: filter_known_tags(tags, model_labels_path)
            for img, tags in ground_truth.items()
        }
        full_image_tags = {
            img: filter_known_tags(tags, model_labels_path)
            for img, tags in full_image_tags.items()
        }
        merged_tags = {
            img: filter_known_tags(tags, model_labels_path)
            for img, tags in merged_tags.items()
        }

    # Подготавливаем словари для хранения результатов
    metrics = {}
    error_dfs = {}
    tag_perf_dfs = {}

    # Начинаем анализ
    display(HTML("<h1 style='color:#1a5276;'>Сравнение методов тегирования</h1>"))

    # Отображаем статистику по количеству тегов
    method_tags_dict = {
        "Полное изображение": full_image_tags,
        "Объединенные теги": merged_tags
    }
    display_tag_count_statistics(ground_truth, method_tags_dict)

    if interactive:
        # Создаем интерактивный график для количества тегов
        fig_tag_counts = create_interactive_tag_count_chart(ground_truth, method_tags_dict)
        display(HTML("<h3 style='color:#2c3e50;'>Интерактивная статистика по количеству тегов</h3>"))
        fig_tag_counts.show()

    # Обрабатываем метод полного изображения
    full_image_metrics = calculate_tag_metrics(ground_truth, full_image_tags)
    metrics["Полное изображение"] = full_image_metrics

    # Обрабатываем метод объединенных тегов
    merged_metrics = calculate_tag_metrics(ground_truth, merged_tags)
    metrics["Объединенные теги"] = merged_metrics

    # Отображаем сводку метрик
    metrics_df = display_metrics_summary(metrics)

    # Анализ ошибок для обоих методов
    full_image_errors = analyze_tagging_errors(
        ground_truth,
        full_image_tags,
        "full_image_errors.csv"
    )
    error_dfs["Полное изображение"] = full_image_errors

    merged_tags_errors = analyze_tagging_errors(
        ground_truth,
        merged_tags,
        "merged_tags_errors.csv"
    )
    error_dfs["Объединенные теги"] = merged_tags_errors

    # Анализ производительности тегов для обоих методов
    full_image_tag_perf = analyze_tag_performance(
        ground_truth,
        full_image_tags,
        "full_image_tag_performance.csv"
    )
    tag_perf_dfs["Полное изображение"] = full_image_tag_perf

    merged_tags_tag_perf = analyze_tag_performance(
        ground_truth,
        merged_tags,
        "merged_tags_tag_performance.csv"
    )
    tag_perf_dfs["Объединенные теги"] = merged_tags_tag_perf

    # Выводим информацию о методах по отдельности
    display(HTML("<h2 style='color:#1a5276;'>Подробный анализ метода полного изображения</h2>"))
    display_worst_performing_images(full_image_errors, "Метод полного изображения", num_worst_examples)
    display_worst_performing_tags(full_image_tag_perf, "Метод полного изображения", num_worst_examples)

    display(HTML("<h2 style='color:#1a5276;'>Подробный анализ метода объединенных тегов</h2>"))
    display_worst_performing_images(merged_tags_errors, "Метод объединенных тегов", num_worst_examples)
    display_worst_performing_tags(merged_tags_tag_perf, "Метод объединенных тегов", num_worst_examples)

    # Визуализация с комбинированными графиками
    if visualize and interactive:
        display(HTML("<h2 style='color:#1a5276;'>Интерактивные графики для сравнения методов</h2>"))

        # 1. Общий график сравнения метрик обоих методов
        fig_metrics = create_combined_metrics_chart(metrics)
        display(HTML("<h3 style='color:#2c3e50;'>Сравнение общих метрик</h3>"))
        fig_metrics.show()

        # 2. Комбинированный график для изображений
        fig_images = create_combined_image_metrics_chart(error_dfs, num_worst_examples)
        display(HTML("<h3 style='color:#2c3e50;'>Сравнение по худшим изображениям</h3>"))
        fig_images.show()

        # 3. Комбинированный график для тегов
        fig_tags = create_combined_tag_metrics_chart(tag_perf_dfs, 20)  # показываем больше тегов в комбинированном графике
        display(HTML("<h3 style='color:#2c3e50;'>Сравнение по худшим тегам</h3>"))
        fig_tags.show()

   # Analyze and display tags with the most improvement
    tag_improvements = analyze_tag_improvements(
        full_image_tag_perf=full_image_tag_perf,
        merged_tags_tag_perf=merged_tags_tag_perf,
        top_n=num_top_improved,
        min_occurrences=3,
        interactive=interactive
    )

    # Analyze and display tags with the most deterioration
    tag_deteriorations = analyze_tag_deteriorations(
        full_image_tag_perf=full_image_tag_perf,
        merged_tags_tag_perf=merged_tags_tag_perf,
        top_n=num_top_deteriorated,
        min_occurrences=3,
        interactive=interactive
    )

    return metrics, error_dfs, tag_perf_dfs, tag_improvements, tag_deteriorations