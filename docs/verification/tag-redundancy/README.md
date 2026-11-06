# Приёмка add-scope-local-tag-redundancy

Дата: 2026-11-03. Все операции выполнялись через код/CLI, без computer use.

## Результат

- Новый `Bundle('tag_sets')` отделяет локальные принятые теги от глобального merge.
- Без операций старый Apply Filters даёт прежние frames/JSON/decisions. Два
  эталона `audit/fixtures/tag-sets-legacy.json` получены из предыдущей реализации
  `apply_filters` коммита b3423d1 на фиксированных full/YOLO/SAM-свидетельствах,
  включая ручные теги, whitelist, regex и форматирование. Тесты не требуют git.
- Точные отношения и независимая эвристика меняют только general внутри одного
  набора. Проверены защита, непринятый свидетель, одинаковые region_id разных
  источников, пустые регионы, два кадра и два региона с 1girl.
- Нормализованная цепочка переносится отдельно от options. Исходные evidence,
  геометрия и raw_scores не меняются. Merge Results повторяет ту же цепочку и
  отклоняет несовместимые политики вместо восстановления удалённых тегов.
- Inspect, HTML и JSON сохраняют причины и before/after; общий caption/TXT
  остаётся результатом стадии Merge Tag Sets. До merge TXT отчёта отклоняется.

## Автоматизированные проверки

1. `.venv/Scripts/python.exe -B audit/run_checks.py --report temp/redundancy-regression.json`
   — **199/199 passed**, 0 failures/errors/skips. Полный вывод:
   `temp/redundancy-regression.txt`. Новый профильный модуль:
   `audit/test_tag_postprocessing.py` (9 тестов).
2. `node --test audit/test_*.mjs` — **12/12 passed**. Каталог обновлён из
   реальных схем: 26 нод, 14 custom types, для каждого есть производитель и
   потребитель; новые подключения работают в обоих направлениях.
3. `.venv/Scripts/python.exe -B audit/check_types.py --report-dir temp/redundancy-types-final`
   — 32 положительных файла, 0 ошибок/предупреждений; отрицательные примеры
   дали только ожидаемые 3 и 2 диагностики. Новые чистые модули включены в pyright.
4. `.venv/Scripts/python.exe -B audit/run_comfy_notebook_acceptance.py`
   — прежние ячейки ManagingTags:0/1, реальный CPU WD ViT и YOLO, временная
   копия изображения, прежний BatchTagging и два списка результатов. TXT/JSON
   записаны; исходные файлы и подписи сохранили хеши. Вывод:
   `temp/redundancy-notebook.txt`, сводка: `notebook.json`.
5. Пример Python из `docs/tag-postprocessing.md` выполнен на фиксированных
   evidence без моделей: локальный skirt удалён, full skirt сохранён.
6. `openspec validate add-scope-local-tag-redundancy --strict` и
   `git diff --check` — успешно.

## Установленный ComfyUI

Python 3.13.12 из установленного ComfyUI, CPU. Команды:

```powershell
& '<ComfyUI>/.venv/Scripts/python.exe' -u -B audit/run_tag_redundancy_acceptance.py --comfy-root '<ComfyUI>'
& '<ComfyUI>/.venv/Scripts/python.exe' -u -B audit/run_preflight_acceptance.py --comfy-root '<ComfyUI>'
& '<ComfyUI>/.venv/Scripts/python.exe' -B audit/export_connection_catalog.py --comfy-root '<ComfyUI>' --output temp/redundancy/connection-nodes.json
node audit/test_connection_catalog.mjs temp/redundancy/connection-nodes.json
```

Используются настоящие V3/legacy-схемы, Python API execute и штатный
`execution.validate_prompt`. HTTP-сервис и браузер не запускаются. В тестовом
окружении валидатор синхронных нод исполняется без создания asyncio-loop:
Windows создаёт локальный сокет даже без сетевой работы, а sandbox его ограничивает.

Сводка: `acceptance.json`. Фиксированные свидетельства содержат полный кадр,
YOLO-регионы clothes и SAM-регион face с маской. Loaders WD/YOLO/SAM и сбор
новых evidence запрещены заглушками. Проверены изменение защиты, локальная
очистка, replay через сохранённый пресет, экспорт и сохранность raw. **Ноль
модельных вызовов** на стадии постобработки.

Настоящий prompt-валидатор принимает новую цепочку и отклоняет цикл отношений
до исполнения. Общий preflight-прогон: **26 схем, 26 ошибочных конфигураций**,
проверки обязательных/связанных входов, типов и отложенных значений;
**ноль исполнений нод**.

Генератор создал `examples/comfyui_wdv3_postprocessing.json`; проверены
уникальные node ID, существование концов каждой связи, индексы слотов и типы
всех примеров, включая стандартный wildcard PreviewAny. Индексы первых четырёх
выходов Load Pipeline Config сохранены.

## Границы проверки

Сам inference в смешанном full/YOLO/SAM-сценарии намеренно заменён готовыми
свидетельствами: проверяется CPU-политика и отсутствие повторных вычислений.
Реальные веса отдельно использовались для notebook-регрессии.
Визуальная браузерная приёмка не проводилась и не отмечена выполненной.
Долговременный кэш, TXT-редактор, условные правила и каскады сюда не входят.
После обновления расширения нужен полный перезапуск ComfyUI.

## Уточнение regex-семейств (2026-11-03)

Добавлены независимые regex_enabled/regex_relations_json в конце прежних
виджетов, optional для совместимости старых API-prompts. Имена правил, шаблоны,
подстановки, свидетели и content digest сохраняются в истории. Literal-пары
не меняют смысл. Пример workflow теперь использует семейства цветов/длины юбок.

- Профильный запуск `unittest audit.test_tag_postprocessing audit.test_comfy_config audit.test_tag_regex`:
  **26/26 passed**, включая 11 тестов локальной постобработки. Проверены полные
  имена, группы/подстановки, защита, отсутствующие цели, self-match, циклы,
  неизменность исходника и defaults старых политик.
- Typecheck `temp/redundancy-regex-types`: 32 файла без ошибок/предупреждений;
  отрицательные проверки дают только ожидаемые диагностики.
- Повторная приёмка установленного ComfyUI: regex-семейства, replay после
  сохранения/загрузки пресета, merge с сохранением цепочки; ноль модельных вызовов.
  Сводка `acceptance.json` содержит regex_families и regex_preset_roundtrip.
- Штатный preflight: 26 схем, 28 ошибочных конфигураций, включая некомпилируемый
  regex и неизвестную группу подстановки. Старый prompt без новых полей принят.
- Каталог и предложения подключений: 4/4 профильных Node.js-проверки прошли.
- Строгая проверка OpenSpec пройдена; браузер/GUI не использовались.
