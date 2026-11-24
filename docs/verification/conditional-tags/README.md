# Проверка условной постобработки

2026-11-08. Change `add-conditional-tag-postprocessing`.

- `audit.test_conditional_tags`: 5 тестов строгой схемы, локальности, ALL/ANY/NONE,
  принятого снимка, буквальных метасимволов, character/manual, нескольких причин,
  противоположных удалений, порядка операций, неизменности Bundle и совместимости merge.
- Профильный запуск: 37 тестов (`test_conditional_tags`, `test_tag_postprocessing`,
  `test_comfy_evidence`, `test_comfy_config`, `test_postprocessing_cost`) прошли.
  Legacy golden caption без операции сохранены.
- `audit/run_conditional_acceptance.py --comfy-root <ComfyUI>` выполняется Python
  окружения установленного ComfyUI. Проверены реальные схемы, Python API,
  сохранение/загрузка preset, Inspect и JSON/HTML/TXT экспорт, штатный валидатор
  корректной цепочки и отклонение malformed JSON/противоречий.
- WD/YOLO/SAM loaders и входы инференса подменены функциями, выбрасывающими ошибку.
  Вызовов — 0. Исходный evidence неизменен. Оба порядка условной очистки и
  избыточности дают ожидаемую разницу skirt, проходят merge, Inspect и экспорт.
- Генератор проверяет типы и ссылки JSON workflow. Пример conditional содержит
  источник clothes и правило apron/dress перед merge. Другой регион/full
  сохраняет dress в общем caption.
- Контекстные предложения проверены в обоих направлениях по каталогу реальных
  схем; существующий `audit/test_connection_suggestions.mjs` прошёл.
- `openspec validate add-conditional-tag-postprocessing --strict` прошёл.

Машиночитаемый результат: `acceptance.json`. Пути отчётов в нём ведут в локальный
`temp/conditional-acceptance`. Браузерная проверка не выполнялась.


## Tag Folder: 2026-11-09

Добавлен необязательный вход postprocessing. Цепочка применяется до merge и записи
в путях с кэшем и без кэша. `audit.test_folder_postprocessing` проверяет несколько
батчей, смену порядка операций, эквивалентность явному пайплайну, raw/историю,
TXT/JSON и вложенные имена, защиту overwrite, раннюю валидацию и пустую цепочку.
Профильный запуск (folder_postprocessing, tag_postprocessing, comfy_processing,
conditional_tags, comfy_config): 32 теста прошли. Реальная схема и штатный
валидатор ComfyUI принимают Postprocessing Config → Tag Folder; API ноды и
генерация `comfyui_wdv3_folder_postprocessing.json` проверяются acceptance-скриптом.
Браузер не запускался.
