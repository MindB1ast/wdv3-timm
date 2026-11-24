# Регрессионные проверки

## GitHub Actions

[CI](../.github/workflows/ci.yml) запускается на push в `main` и на pull request
в `main`. `Python regressions (CPU)` проверяет правила тегов, каскады,
постобработку, экспорт и два набора эталонных сценариев. `Validate OpenSpec`
проверяет основные спецификации отдельно. Для этих задач не нужны ComfyUI,
модельные веса или секреты.

Эквивалентный локальный запуск из корня проекта (Bash, в активном
окружении Python 3.11, с Node.js 22 для OpenSpec):

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r requirements.txt
export HF_HUB_OFFLINE=1 MPLBACKEND=Agg PYTHONIOENCODING=utf-8
python -B -m unittest \
  audit.test_saved_region_filters \
  audit.test_region_filter_outcomes \
  audit.test_conditional_tags \
  audit.test_folder_postprocessing \
  audit.test_comfy_export \
  audit.test_cascade_tags \
  audit.test_tag_policies \
  audit.test_tag_regex
npx --yes --package=@fission-ai/openspec@1.15.0 -- openspec validate --specs
```

CI сравнивает результаты на сохранённых данных и не проверяет реальный
инференс WD/YOLO/SAM. Прогоны сохранённых workflow с реальными моделями
по-прежнему запускаются вручную, как описано ниже.
Для побайтовых SHA-256 используйте Git checkout: `.gitattributes` сохраняет
CRLF у двух исторических JSON-эталонов и на Linux.

Исходная ревизия 2026-10-29: 30 проверок, 18 passed и 12 нарушений.
Исторические свидетельства остаются в docs/audit/2026-10-29.
После исправлений исходные 30 контрактов проходят без изменения assertions.
Дополнительные проверки охватывают сбои атомарной записи, метрики, идентификаторы,
ручные теги, настоящую визуализацию и освобождение файлов.

Запуск из корня с Python 3.11 и зависимостями проекта:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python audit/run_checks.py --report temp/regression-results.json
```

Локальное восстановленное окружение: temp/apply-runtime/Scripts/python.exe.
Исходные 30 отдельно: добавить --pattern test_current_behavior.py.
По умолчанию запускаются все test*.py. Все временные записи изолированы в temp;
оригиналы изображений и подписей не перезаписываются. Весовые модели в unit-тестах
не загружаются; реальный CPU-прогон записывается отдельно.

Итог: 50/50 (30 исходных + 20 дополнительных); подробная приёмка и ограничения:
`docs/verification/fix-audited-tagging-defects/README.md`.
Реальные модели: `audit/run_integration.py`; ограниченные мутации: `audit/run_mutations.py`.

Persistent evidence и готовые TXT: `audit/test_artifact_store.py`,
`test_inference_plan.py`, `test_persistent_runtime.py`, `test_persistent_evidence.py`,
`test_caption_files.py`, `test_workflow_api.py`. Все включены в обычный discovery.
Проверка настоящего Windows junction требует запуска вне sandbox, создаёт только
временные каталоги проекта. Сквозные команды: `audit/run_persistent_real.py` и
`audit/run_persistent_acceptance.py --comfy-root <ComfyUI-root>`.
Результаты и ограничения: `docs/verification/persistent-evidence/README.md`.


## Эталонные пользовательские графы

Снимки графов и ожидаемые файлы лежат в `audit/fixtures/saved_workflows`.
Файлы `.regions.json` сравниваются по разобранному содержимому JSON, поэтому отступы
и переносы строк допустимы. TXT сравниваются побайтово.
В наборе полный кадр, одиночный YOLO, YOLO с дочерним детектором,
параллельные источники, два варианта каскада, SAM и два варианта Tag Folder. Сохранённый результат
дочернего YOLO взят из ранее зафиксированного эталона демонстрации.
Изображения берутся из каталога input и проверяются по SHA-256.
Весовые файлы используются из существующей локальной установки и проекта.

Запуск из корня проекта для ComfyUI Desktop на Windows:

```powershell
$desktop = 'C:/Users/annak/AppData/Local/Comfy-Desktop'
$comfy = "$desktop/ComfyUI-Installs/ComfyUI (1)/ComfyUI"
& "$comfy/.venv/Scripts/python.exe" -B audit/run_saved_workflow_regressions.py --comfy-root $comfy --input-dir "$desktop/ComfyUI-Shared/input" --folder-input "$desktop/ComfyUI-Shared/output/wdv3/folder_tagging"
```

Можно добавить `--case sam_test` или повторить `--case` для нескольких графов.
`--models-dir` переопределяет каталог моделей ComfyUI. Для сравнения уже
созданных артефактов используйте `--compare-dir <каталог-прогона>` вместо
`--comfy-root`. Код выхода 1 означает ошибку выполнения, недостающий файл
или отличие от эталона. Новые файлы находятся в `temp/saved-workflows-*`.
Сравнение TXT и компактного JSON побайтовое. Большой single_yolo_test.json
проверяется по SHA-256 канонического JSON с исключением трёх служебных полей
download_missing, которые меняет тест. Эталоны обновляются отдельно от прогона.

Проверка выполняет backend-ноды сохранённых графов с реальными моделями.
Для нод, которые только показывают данные на канве, отдельная проверка —
`run_viewer_layout.py` и тесты выбора регионов. Валидацию подключений и схем
проверяет `run_preflight_acceptance.py`. Быстрые тесты в обычном discovery
по-прежнему покрывают отдельные правила, порядок операций, запись файлов
и кэш без загрузки тяжёлых моделей.

Случай folder_tagging сохраняет только TXT для изображения в корне, согласно
сохранённой настройке recursive=false. folder_tagging_recursive меняет только
recursive и проверяет оба TXT, включая folder/tag_managing1.png.txt.
Входные изображения копируются в temp после проверки SHA-256; перезапись
оригиналов и использование существующего кэша исключены.

Пользовательские примеры фильтров закреплены в `audit/fixtures/region_filters`.
Команда `& .\.venv\Scripts\python.exe audit/run_region_filter_regressions.py`
проверяет все 16 результатов на одних сохранённых оценках без моделей,
с отдельным PASS/FAIL по каждому файлу. Те же примеры включены в unittest
через `audit.test_saved_region_filters`. Демонстрационный `workflow.json`
показывает все варианты независимыми подписанными ветками от одного evidence.
В README фикстуры описаны восстановленные настройки и ограничения покрытия.

Отдельный [граф различимых исходов](fixtures/region_filter_outcomes/README.md)
не меняет эти 16 эталонов. `audit/run_region_filter_outcomes.py` проверяет
29 сценариев без моделей, включая два источника, происхождение тегов в caption
и строгое сравнение с порогом. `--output-dir` сохраняет результаты с именами
из нового manifest.

Отдельные эталоны ещё нужны для экспортируемых SAM-масок. Пороговые примеры
general_threshold и character_threshold проверяют значение 1 для регионов.
