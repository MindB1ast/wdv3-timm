# Приёмка контекстных рекомендаций WDV3

Дата: 2026-11-03. Change: `add-contextual-node-connection-suggestions`.

## Автоматические проверки

Реальные legacy/V3 definitions экспортированы из установленного ComfyUI 0.43.2
Python 3.13.12 без inference и скачивания:

```powershell
& '<Python установленного ComfyUI>' -B audit/export_connection_catalog.py --comfy-root '<каталог ComfyUI>'
node audit/test_connection_catalog.mjs temp/connection-node-defs.json
```

Результат: 22 ноды, 12 пользовательских типов, у всех есть производитель и
потребитель. Нормализованные контракты точно совпали с проверяемым fixture
`audit/fixtures/connection_nodes.json`. Полная матрица — `docs/connection-types.md`.
Серверные type IDs, порядок выходов и вычислительные модули не изменялись.

Все 10 Node.js-наборов проходят:

```powershell
node audit/test_connection_catalog.mjs
node audit/test_connection_suggestions.mjs
node audit/test_connection_menu.mjs
node audit/test_connection_creation.mjs
node audit/test_preview_limit.mjs
node audit/test_region_selection.mjs
node audit/test_rule_presets.mjs
node audit/test_tag_origin.mjs
node audit/test_tag_format.mjs
node audit/test_tagger_options.mjs
```

Новые проверки охватывают приоритеты, limit=1, native fallback/search, оба
направления, точные multi-output-порты, отмену, занятый вход, rollback неудачного
создания, current slot name после изменения индексов, новые динамические входы,
повторную регистрацию и цепочку обработчиков другого расширения. Проверки
создания используют тестовый контракт graph/node/canvas и подтверждают пары
beforeChange/afterChange; реальное undo/redo остаётся частью ручной приёмки.

Python-регрессия:

```powershell
temp/apply-runtime/Scripts/python.exe -m unittest audit.test_comfy_nodes audit.test_comfy_config audit.test_comfy_evidence
```

Результат: 21/21, вывод `temp/connection-python-checks.txt`. Для записи служебных
файлов при запуске установлены YOLO_CONFIG_DIR и MPLCONFIGDIR в `temp/` проекта.
`node --check` новых frontend-модулей, `git diff --check` и
`openspec validate add-contextual-node-connection-suggestions --strict` проходят.
TXT/store-ноды соседнего change не реализованы и не нужны этим проверкам;
добавление будущего типа/ноды проверяется синтетическим definition.

## Ручная приёмка — ожидается

По прямому указанию пользователя computer use не используется. До этого указания
была открыта временная отдельная вкладка ComfyUI, но проверка реализации в ней
не выполнялась. Результаты браузерной приёмки, реальные жесты меню и снимки
не заявляются проверенными.

Адаптер написан по API установленного frontend 1.57.4: ContextMenu,
showConnectionMenu/showSearchBox, getExtraSlotMenuOptions и стандартные graph
change hooks. Проверку фактического интерфейса выполняет пользователь по
[manual-checklist.md](manual-checklist.md); задача 3.1 остаётся незавершённой.
Особенно нужно подтвердить drag-to-empty, right-click меню, Vue/canvas-режим
пользователя и настоящее undo/redo. После результата обновить этот раздел.

## Исправление sam_batch_size при восстановлении

Пользователь сообщил об ошибке преобразования sam_batch_size в INT. Добавлена
нормализация этого виджета при создании/восстановлении Tagging Settings и перед
отправкой: пустое или некорректное значение становится 1, допустимое целое
сохраняется, границы соответствуют схеме 1..16. Текстовый разделитель не меняется.
`node audit/test_tagging_settings.mjs`, `node audit/test_tagger_options.mjs` и
`node audit/test_rule_presets.mjs` проходят; результат повторного запуска в UI
ожидается от пользователя. Computer use для исправления не применялся.

## Зависимости SAM-входов Tag Image / Regions

После сообщения пользователя `sources` скрывается, пока не подключён sam_model;
существующая связь sources не удаляется при отключении модели. Backend продолжает
явно отклонять неполную пару, теперь с инструкцией подключения. YOLO не скрывает
sam_regions: это независимые источники. Проверены plain full-image без YOLO/SAM
и ошибки неполной SAM-пары: 13/13 Python-тестов comfy_nodes/comfy_processing,
вывод `temp/tagger-sam-dependencies.txt`. Проходят Node.js-тесты
`test_tagger_input_state`, `test_connection_creation`, `test_connection_menu`.
Ручной результат после Ctrl+F5 ожидается от пользователя.
