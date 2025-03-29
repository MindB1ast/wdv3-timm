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


def extract_region_images(image_path, detection_results, output_dir=None, save=True):
    """
    Извлекает изображения из обнаруженных областей и сохраняет их в отдельные файлы.
    
    Args:
        image_path: Путь к исходному изображению
        detection_results: Результаты детекции из BatchTagging
        output_dir: Директория для сохранения вырезанных изображений (если None, 
                   будет создана поддиректория рядом с оригинальным изображением)
        save: Сохранять изображения (True) или только возвращать их (False)
    
    Returns:
        dict: Словарь с вырезанными изображениями в формате:
              {
                 "detector_name": [
                    {
                        "region_id": id,
                        "bbox": [x1, y1, x2, y2],
                        "image": PIL.Image,
                        "saved_path": путь_к_сохраненному_файлу (если save=True)
                    },
                    ...
                 ],
                 ...
              }
    """
    from PIL import Image
    import os
    
    # Загружаем изображение
    img = Image.open(image_path)
    
    # Если output_dir не указан, создаем папку рядом с изображением
    if output_dir is None:
        base_dir = os.path.dirname(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(base_dir, f"{img_name}_regions")
    
    # Создаем директорию, если она не существует
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Словарь для хранения вырезанных изображений
    extracted_regions = {}
    
    # Проходим по всем детекторам
    for detector_name, regions in detection_results['detectors'].items():
        extracted_regions[detector_name] = []
        
        # Проходим по всем регионам/боксам детектора
        for region in regions:
            # Получаем координаты бокса
            x1, y1, x2, y2 = region["bbox"]
            
            # Вырезаем область
            region_img = img.crop((x1, y1, x2, y2))
            
            # Имя файла для сохранения
            if save:
                region_filename = f"{detector_name}_{region['region_id'] + 1}.png"
                save_path = os.path.join(output_dir, region_filename)
                
                # Сохраняем изображение
                region_img.save(save_path)
                
                print(f"Сохранено изображение области {region['region_id'] + 1} детектора {detector_name}: {save_path}")
            else:
                save_path = None
            
            # Добавляем информацию в словарь
            extracted_regions[detector_name].append({
                "region_id": region["region_id"],
                "bbox": region["bbox"],
                "image": region_img,
                "saved_path": save_path
            })
    
    return extracted_regions


def visualize_detection_results_highres(image_path, detection_results, output_path=None, line_width=3, font_size=20):
    """
    Визуализирует результаты детекции YOLO-моделей с использованием PIL
    с сохранением изображения в полном разрешении.
    
    Args:
        image_path: Путь к исходному изображению
        detection_results: Результаты детекции из BatchTagging
        output_path: Путь для сохранения изображения с боксами
        line_width: Толщина линий рамок
        font_size: Размер шрифта для надписей
    
    Returns:
        PIL.Image: Изображение с отрисованными боксами
    """
    import os
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    # Загружаем изображение
    img = Image.open(image_path)
    
    # Создаем копию, чтобы не изменять оригинал
    draw_img = img.copy()
    
    # Создаем объект для рисования
    draw = ImageDraw.Draw(draw_img)
    
    # Пытаемся загрузить шрифт (если не получится, будет использован шрифт по умолчанию)
    try:
        # Пробуем разные шрифты, которые могут быть в системе
        possible_fonts = [
            "arial.ttf", "Arial.ttf",
            "DejaVuSans.ttf", "dejavu-sans.ttf",
            "NotoSans-Regular.ttf", "Roboto-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ]
        
        font = None
        for font_name in possible_fonts:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except IOError:
                continue
        
        if font is None:
            # Если ничего не нашли, используем шрифт по умолчанию
            font = ImageFont.load_default()
    except Exception:
        # Если что-то пошло не так, используем шрифт по умолчанию
        font = ImageFont.load_default()
    
    # Словарь цветов для разных детекторов
    colors = {}
    
    # Проходим по всем детекторам
    for detector_name, regions in detection_results['detectors'].items():
        # Если детектора еще нет в словаре цветов, генерируем для него случайный цвет
        if detector_name not in colors:
            colors[detector_name] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        
        # Цвет для текущего детектора
        color = colors[detector_name]
        
        # Проходим по всем регионам/боксам детектора
        for region in regions:
            # Получаем координаты бокса
            x1, y1, x2, y2 = region["bbox"]
            
            # Рисуем прямоугольник
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # Получаем название региона
            region_name = f"{detector_name} ({region['region_id']+1})"
            
            # Рисуем фон для текста
            text_width, text_height = draw.textbbox((0, 0), region_name, font=font)[2:]
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Рисуем текст
            draw.text((x1 + 2, y1 - text_height - 2), region_name, fill="white", font=font)
    
    # Если указан путь, сохраняем результат
    if output_path:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Сохраняем в высоком разрешении
        draw_img.save(output_path, quality=95)
        print(f"Визуализация в высоком разрешении сохранена в: {output_path}")
    
    return draw_img


def enhanced_view_image_results(results, image_index, visualize=True, save_visualization=False, 
                              extract_regions=False, high_res=True, verbose=False):
    """
    Улучшенная версия функции просмотра результатов обработки изображения
    с возможностью извлечения областей и визуализации в высоком разрешении.
    
    Args:
        results: Результат выполнения BatchTagging (кортеж из двух списков)
        image_index: Индекс изображения (начиная с 0)
        visualize: Визуализировать боксы детекторов
        save_visualization: Сохранять визуализацию в файл
        extract_regions: Извлекать и сохранять области в отдельные файлы
        high_res: Использовать визуализацию в высоком разрешении
        verbose: Подробный вывод информации
    
    Returns:
        dict: Информация о тегах и детекции для выбранного изображения
    """
    import os
    
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
    
    # Извлекаем области, если требуется
    extracted_regions = None
    if extract_regions:
        print("\nИзвлечение областей из изображения...")
        extracted_regions = extract_region_images(image_path, image_result)
    
    # Визуализируем результаты детекции, если требуется
    visualization_result = None
    if visualize:
        output_path = None
        if save_visualization:
            base_dir = os.path.dirname(image_path)
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if high_res:
                output_path = os.path.join(base_dir, f"{img_name}_detection_highres.jpg")
            else:
                output_path = os.path.join(base_dir, f"{img_name}_detection.jpg")
        
        if high_res:
            print("\nСоздание визуализации в высоком разрешении...")
            visualization_result = visualize_detection_results_highres(
                image_path, image_result, output_path
            )
        else:
            print("\nСоздание стандартной визуализации...")
            visualization_result = visualize_detection_results(
                image_path, image_result, output_path, show=True
            )

    if verbose:
        # Выводим основную информацию
        print(f"\nИнформация о изображении #{image_index + 1}: {image_path}")
        
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
        "image_path": image_path,
        "extracted_regions": extracted_regions,
        "visualization": visualization_result
    }
        """