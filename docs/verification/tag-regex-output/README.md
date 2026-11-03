# Приёмка regex и формата тегов

Дата: 2026-11-03. Change: `add-explicit-tag-regex-and-output-format`.

## Проверки

- `temp/apply-runtime/Scripts/python.exe audit/run_checks.py --report temp/regex-regression-results.json`: 188/188, без failures/errors/skips. Вывод: `temp/regex-regression-output.txt`. Исходные изображения/подписи не перезаписываются.
- `node audit/test_rule_presets.mjs`: перенос regex без канонизации, сброс отсутствующих шаблонов старого пресета, сохранение буквальных списков и разделителя.
- `node audit/test_tag_format.mjs`: формат текста viewer, порядок, экранирование, пустой/пробельный/произвольный разделитель.
- `node audit/test_preview_limit.mjs`, `node audit/test_region_selection.mjs`, `node audit/test_tag_origin.mjs`: регрессии frontend проходят.
- `.venv/Scripts/python.exe audit/check_types.py --report-dir temp/regex-types312`: положительные контракты без ошибок/предупреждений; отрицательные контракты дают только ожидаемые диагностики.
- `openspec validate add-explicit-tag-regex-and-output-format --strict`: change валиден.

Старый Python 3.11 из `temp/apply-runtime` при отдельном запуске typecheck сообщает `reportPrivateImportUsage` для существующего импорта `huggingface_hub.utils.validate_repo_id` в неизменённом `Scripts/tagger_catalog.py`. В основном окружении проекта Python 3.12.15 проверка полностью проходит; изменение импорта не требуется.

## Реальные схемы и исполнение ComfyUI

Скрипт `audit/run_tag_regex_acceptance.py` запускается Python установленного ComfyUI, с `-B` и `--comfy-root <каталог ComfyUI>`. Схемы и классы `io` настоящие; для воспроизводимых проверок числа вызовов WD и двух YOLO используются небольшие детерминированные тестовые модели. Сервис HTTP и браузер не запускались.

Проверены Region Tag Rules → YOLO/SAM-передача правил, Tagging Settings → Collect Evidence → две политики Apply Filters → Inspect/Results Viewer → Export Report/Save Result, а также Tag Folder. Число вызовов после двух фильтраций остаётся WD=3, eyes=1, face=1; исходное evidence неизменно. TXT всех потребителей и выход Inspect совпадают: `blue_eyes | hero_\(series\) | long_hair`. Отдельно проверены старые позиционные виджеты поставляемого `examples/comfyui_wdv3.json`: при defaults текст сохраняет `, ` и замену `_` пробелами.

Сводка: [acceptance.json](acceptance.json). Результаты и автономный HTML сохраняются в `temp/tag-regex-acceptance/`; точный путь HTML есть в сводке. Python ComfyUI — 3.13.12. Изменения frontend проверены отдельными Node.js-тестами; отдельная визуальная браузерная приёмка не выполнялась.

## Notebook и настоящие модели

В основном окружении Python 3.12.15/PyTorch 2.6.0+cpu выполнены первые две существующие ячейки `ManagingTags.ipynb` через `audit/run_comfy_notebook_acceptance.py` на отдельной копии изображения. Путь итоговой сводки перенаправлен в этот каталог при исполнении, исторический отчёт не заменялся.

Использованы локальные WD ViT v3, person_yolov8s-seg и face_yolov9c, без скачивания. Публичный `BatchTagging` возвращает прежние списки detailed/merged и сохраняет TXT/JSON. SHA256 исходных файлов проверены до/после, изменений нет. Отдельно модули merge из `HEAD` загружены в изолированный пакет `temp/regex-baseline`: на полученных настоящих notebook-свидетельствах результат полностью совпал, включая scores, provenance, caption и taglist.

Сводка: [notebook.json](notebook.json). GPU не требовался и не проверялся.

## Уточнение интерфейса: настройки только в options

После уточнения пользователя из схемы Tag Image / Regions удалены все скалярные
поля, включая SAM batch. Tagging Settings хранит весь набор настроек, включая
`sam_batch_size`; pipeline-конфиг переносит его в options. Python API прежних
прямых вызовов сохранён, но эти параметры не публикуются в UI.

Перегенерированы workflow-примеры с центральной Tagging Settings. Для старого
UI-workflow frontend переносит локальные значения в новую Settings, когда
options не был подключён; при подключённом Settings значения сохраняются,
а для переноса прежнего нестандартного SAM batch создаётся отдельная копия.
Связь с Load Pipeline Config сохраняется; SAM batch задаётся теперь в options
самого пресета, default 1. `node audit/test_tagger_options.mjs` проверяет перенос
значений, связи, отсутствие изменения общего Settings и повторную загрузку.

После изменения выполнены 41/41 профильных Python-проверок (вывод
`temp/tagger-options-tests.txt`), тесты миграции и пресетов Node.js, strict
валидация change, а также повторный прогон настоящих схем/нод ComfyUI через
`audit/run_tag_regex_acceptance.py`. Этот прогон проверяет отсутствие любых
скалярных настроек у теггера и выполнение без options со стандартными значениями.
