from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


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
    import plotly.express as px
    import numpy as np
    from PIL import Image
    import random
    import os

    # Загружаем изображение
    img = Image.open(image_path)
    img_np = np.array(img)

    # Создаем фигуру с изображением
    fig = px.imshow(img_np)
    
    # Переключаем режим перемещения на "панорамирование" (drag and drop)
    fig.update_layout(dragmode='pan')
    
    # Словарь цветов для разных детекторов
    colors = {}
    
    # Проходим по всем детекторам
    for detector_name, regions in detection_results['detectors'].items():
        # Если детектора еще нет в словаре цветов, генерируем для него случайный цвет
        if detector_name not in colors:
            colors[detector_name] = f'rgb({random.randint(50, 255)}, {random.randint(50, 255)}, {random.randint(50, 255)})'
        
        # Цвет для текущего детектора
        color = colors[detector_name]
        
        # Проходим по всем регионам/боксам детектора
        for region in regions:
            # Получаем координаты бокса
            x1, y1, x2, y2 = region["bbox"]
            
            # Получаем теги для текущего региона и вставляем разрывы строк
            region_tags = region.get('taglist', '')
            if region_tags:
                # Если taglist хранится как строка, заменяем разделитель на разрыв строки
                tags_text = region_tags.replace(', ', '<br>')
            else:
                tags_text = "Нет тегов"
            
            # Добавляем прямоугольник на фигуру
            fig.add_shape(
                type='rect',
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(color=color, width=3),
                name=f"{detector_name} ({region['region_id']+1})"
            )
            
            # Добавляем подпись с hovertext (теги будут видны при наведении)
            fig.add_annotation(
                x=x1, y=y1,
                text=f"{detector_name} ({region['region_id']+1})",
                showarrow=False,
                font=dict(color='white'),
                align='left',
                bordercolor=color,
                borderwidth=2,
                borderpad=4,
                bgcolor=color,
                opacity=0.8,
                # hovertext с тегами, где теги будут перенесены на новую строку
                hovertext=f"Теги:<br>{tags_text}",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
            )
    
    # Настраиваем оси и отображение
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Сохраняем результат, если указан путь
    if output_path:
        fig.write_image(output_path)
        print(f"Визуализация сохранена в: {output_path}")
    
    # Показываем изображение, если требуется
    if show:
        fig.show()
    
    return fig


def view_image_results(results, image_index, visualize=True, save_visualization=False, verbose=False):
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
        print(f"Ошибка: индекс {image_index} вне диапазона. Доступно {len(merged_results)} изображений")
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

    if verbose == True:
        # Выводим основную информацию
        print(f"Информация о изображении #{image_index + 1}: {image_path}")
        
        print("\nОбъединенные теги:")
        print(merged_result["merged_tags"]["taglist"])
        print()
        print("Теги без включения yolo моделей:")
        print(image_result["full_image"]["taglist"])
        
        # Преобразуем строки тегов в списки, убирая лишние пробелы
        merged_tags = [tag.strip() for tag in merged_result["merged_tags"]["taglist"].split(',')]
        yolo_excluded_tags = [tag.strip() for tag in image_result["full_image"]["taglist"].split(',')]
        
        # Вычисляем разницу: теги, присутствующие в объединённых, но отсутствующие в yolo
        diff_tags = set(merged_tags) - set(yolo_excluded_tags)
        
        print("\nРазница тегов (присутствуют в объединенных, но отсутствуют в тегах без yolo):")
        print(', '.join(sorted(diff_tags)))

    """
    return {
        "original_result": image_result,
        "merged_result": merged_result,
        "image_path": image_path
    }
    """