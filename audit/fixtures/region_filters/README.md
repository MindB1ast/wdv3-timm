# Пользовательские примеры Region Tag Rules

Зафиксированы 16 результатов из `ComfyUI-Shared/output/wdv3/filters_test` и исходный
`filters_test.json`. Пользователь последовательно менял настройки одного графа.
В `source_workflow.json` хранится сохранённая версия, а `workflow.json` показывает
все варианты одновременно с одним общим Collect Evidence. У каждой ветки есть
комментарий с ожидаемым эффектом, собственные правила и отдельное сохранение.

Откройте `workflow.json` в ComfyUI. Новые результаты направлены в
`ComfyUI-Shared/output/wdv3/filters_test_demo`. Для сравнения доступны записи
полного кадра, шести регионов person и общий caption.

## Запуск без моделей

Из корня проекта в PowerShell:

```powershell
& .\.venv\Scripts\python.exe audit/run_region_filter_regressions.py
& .\.venv\Scripts\python.exe -m unittest audit.test_saved_region_filters
```

Первая команда печатает отдельный PASS/FAIL для каждого файла и возвращает
ненулевой код при несовпадении. Вторая включает те же примеры в unittest discovery.
Для одного примера доступен `--case`, например:

```powershell
& .\.venv\Scripts\python.exe audit/run_region_filter_regressions.py --case girl_only_from_source_regex
```

`evidence.json.gz` содержит реальные исходные оценки WD, полученные одним
Collect Evidence на том же изображении с локальными WD/YOLO-моделями. Пиксели
изображения исключены из этой тестовой фикстуры. Размеры, bbox, модельные оценки
и настройки сохранены. Демонстрационный граф получает пиксели обычным Load Image.
Обычные регрессии загружают зафиксированные оценки и запрещают вызовы загрузчиков
моделей, сбора evidence и WD-инференса. Контрольные суммы проверяются до обработки.

Проверяются все поля пользовательского JSON, включая отдельные категории full и
каждого региона, bbox и caption. В manifest дополнительно закреплены ожидаемые
добавления и удаления относительно baseline. Форматирование JSON допускается.
Обычный запуск не обновляет фикстуры и эталоны.

## Восстановленные настройки

История всех прежних значений отсутствует в сохранённом workflow. Варианты
восстановлены по именам файлов и различиям результатов. Regex могут иметь другие
эквивалентные записи, чем использовал пользователь. Источник восстановления указан
для каждого случая в manifest. Для ornament шаблон `.*ornament.*` также проверен
по истории ComfyUI. Первые 14 восстановленных вариантов дали точное совпадение
разобранных JSON при повторном запуске реальных backend-нод с локальными моделями.
Проверка штатным валидатором ComfyUI прошла для первоначальных 14 выходов графа.
Повторение без моделей и интеграционный unittest первоначально прошли все 14 случаев.
Пробное добавление лишнего тега было обнаружено сравнением с эталоном.
После добавления двух пороговых примеров полный запуск без моделей прошёл 16/16 случаев.

| Пример | Изменённое поле |
| --- | --- |
| filter_test_default | Базовые пустые списки |
| 1girl_remove_from_regions | remove_from_regions |
| multiple_girls_remove_from_full | remove_from_full |
| test_added_to_regions | add_to_regions |
| pigeon-toed_require_full_confirmation | require_full_confirmation |
| height_chart_only_from_source | only_from_source |
| 1girl_allowed_tags | allowed_tags |
| 1girl_full_region_tags_only | full_image_only_tags |
| black_remove_from_region_regex | remove_tags_from_region_regex |
| girl_regex_remove_from_full | remove_tags_from_full_regex |
| pigeon_require_full_confirmation_regex | exclude_from_region_regex, подпись require_full_confirmation_regex |
| girl_only_from_source_regex | only_from_source_regex |
| eye_allowed_tags_regex | allowed_tags_regex |
| ornament_full_image_only_tags_regex | full_image_only_tags_regex |
| 1_general_treshold | general_threshold = 1 |
| 1_character_treshold | character_threshold = 1 |

Поле full_image_only_tags в baseline намеренно пустое. В остальных вариантах
меняется только указанное поле. Обе категории тегов сохраняются, оценки и история
отключены. Добавленные пороговые примеры проверяют выключение соответствующей
категории региональных тегов порогом 1 при сохранении другой категории и тегов
полного кадра. Примеры уменьшения порога и точного равенства оценки порогу
этим набором отдельно не представлены.
