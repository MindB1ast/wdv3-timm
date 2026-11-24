# Различимые исходы Region Tag Rules

Откройте `workflow.json` в ComfyUI как отдельный workflow. Он собран из
закреплённой копии `WDV3 Region Tag Rules Tests`: исходный граф и 16 исторических
эталонов остались неизменными. Граф содержит 26 подписанных веток в двух блоках:
один `person` и два параллельных источника `person_A` / `person_B`.
В каждом блоке один Collect Evidence и независимые ветки Apply Filters.

Для запуска на холсте нужны исходное изображение в ComfyUI Input и локальные
WD/YOLO-веса из исходного графа. Загрузки моделей из сети отключены. Выходной
путь на этой установке — `ComfyUI-Shared/output/wdv3/filter_outcomes_demo`.
При переносе измените `output_directory` в нодах сохранения нового графа.

Без моделей:

```powershell
& .\.venv\Scripts\python.exe -B audit/run_region_filter_outcomes.py
& .\.venv\Scripts\python.exe -B audit/run_region_filter_outcomes.py --case only_from_source_blocks_other_source_caption
& .\.venv\Scripts\python.exe -B audit/run_region_filter_outcomes.py --output-dir temp/region-filter-outcomes
& .\.venv\Scripts\python.exe -B audit/build_region_filter_outcomes_workflow.py --check
```

Без `--output-dir` фактические JSON временные; указанный каталог должен быть
новым. `manifest.json` задаёт ожидаемое наличие или отсутствие конкретных тегов,
локальный белый список и происхождение ключевых тегов. Общие принятые теги
полного кадра сравниваются с историческим эталоном. Имя сценария — основа
имени `.regions.json` и название ветки, например
`remove_from_full_allows_other_source_caption.regions.json`.

Из 29 автосценариев три проверяют точное равенство порогу на искусственных
оценках `0.49`, `0.50`, `0.51` для обеих категорий. Эти три не входят в граф,
где модель выдаёт собственные оценки. Автопроверки также входят в
`audit.test_region_filter_outcomes`.
