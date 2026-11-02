# Аудит выполнения ComfyUI change

> Исторический аудит до реализации оставшихся задач. Описанные ниже пробелы
> относятся к состоянию начала работы. Актуальные результаты находятся в
> [итоговой приёмке](README.md) и tasks.md change.

Дата: 2026-11-02. Change: add-comfyui-regional-tagging-nodes.

Проверены текущие исходники, регрессии и существующие результаты запусков.
Отчёт различает реализацию в ComfyUI, общие утилиты/notebook и запланированный контракт.
Наличие старого артефакта не считается новым сквозным прогоном.

Итог после уточнения объёма пользователем: 18 пунктов выполнено, 12 выполнено частично, 3 не реализовано. Ручной фидбек (прежний 5.4) исключён из текущего change, а не выполнен.
Все 15 открытых checkbox остаются открытыми: для закрытия составного пункта
должны выполняться все его критерии. Описания выше не заменяют требования.

## Выполненные группы

- 1.1–1.5: регистрация, зависимости, локальный WD, общее ядро, IMAGE-батчи и согласованный merge. Код: __init__.py, comfy_nodes.py, comfy_processing.py, Scripts/region_tagging.py, Scripts/preprocess.py.
- 2.1–2.7: YOLO/SAM-источники, встроенный SAM, скачивание SAM, обычные поля правил, полные пресеты и папочные батчи. Код: comfy_ui_nodes.py, comfy_config.py, comfy_processing.py.
- 3.1–3.6: Viewer, выбор регионов, preview_limit, происхождение merged-тегов, Save Result и примеры. Код: web/, comfy_preview.py, comfy_export.py, примеры examples/comfyui_wdv3*.json.

Реальные предыдущие результаты: temp/comfy-multi-sam-result.json,
temp/comfy-multi-yolo-result.json, temp/comfy-batch-results/,
temp/comfy-save-result-api/. Это локальные диагностические артефакты;
их наличие само по себе не закрывает расширенные сценарии раздела 6.

## Повторно выполненные проверки

70 тестов Python прошли:

```powershell
$env:YOLO_CONFIG_DIR=Join-Path (Get-Location) 'temp/ultralytics-comfy-tests'
$env:PYTHONPATH=Join-Path (Get-Location) 'audit'
.venv/Scripts/python.exe -m unittest audit.test_comfy_nodes audit.test_comfy_processing audit.test_comfy_config audit.test_comfy_preview audit.test_comfy_export audit.test_yolo_batch audit.test_sam_auto audit.test_sam_backend audit.test_fixes audit.test_typed_contracts
node audit/test_region_selection.mjs
node audit/test_preview_limit.mjs
node audit/test_tag_origin.mjs
```

Все три JS-проверки прошли. Эти регрессии используют небольшие фикстуры;
они не являются новым реальным прогоном всех весов, GPU, браузерного UI или notebook.
V3-ноды дополнительно подтверждены ранее выполненными прогонами ComfyUI;
обычная проектная .venv без comfy_api проверяет вычислительное ядро и legacy-регистрацию.

## Открытые пункты по фактической реализации

### 4.1

Частично: есть TypedDict-контракты, идентичность SAM-кадра и schema_version конфигов/экспорта SAM. Единого версионированного неизменяемого контракта всех стадий ComfyUI нет.

### 4.2

Не реализовано: RegionTagger вызывает get_tags с порогами до выдачи результата; отклонённые scores и причины не сохраняются.

### 4.3

Не реализовано: правила входят в вызов tag_frames/RegionTagger. Fingerprint конфигов и сохранённой геометрии не является кэшем исходных WD-оценок.

### 4.4

Не реализовано: в DetectorConfig, Region Tag Rules и merge нет исключительных владельцев с omit/allow_full; require_full_confirmation — другое правило.

### 4.5

Частично: обычные поля, JSON-пресеты и Save/Load Pipeline Config есть. Браузерного импорта выбранного detectors.json и полного набора ролевых пресетов нет.

### 4.6

Частично: merge сортирует источники и сохраняет максимум и provenance победителя. Нет объединения произвольных сохранённых свидетельств со всеми конкурирующими источниками; Combine отвергает повтор имени, а не дедуплицирует ветвь.

### 4.7

Частично: контракт описан в design.md. Рабочий JSON-контракт class_label/text и его round-trip без score не подтверждены реализацией и тестами.

### 5.1

Частично: WD поддерживает auto/cpu/cuda, проверку CUDA и локальные config/веса; YOLO поддерживает локальный путь и ленивую загрузку. У WD/YOLO нет общего download_missing и полного контракта метаданных; YOLO-инференс использует CPU.

### 5.2

Частично: auto выбирает устройство ComfyUI, WD переносится на CPU в finally, SAM offload проверен, папочный прогресс проверяет прерывание. Это не полное управление ресурсами; IMAGE-теггер не передаёт progress и GPU-проверки нет.

### 5.3

Частично: Viewer показывает принятые теги, области, источники и оценки, Added from regions и выбор кликом. Полных исходных стадий, отклонённых тегов с причинами, поиска и пагинации нет.

### 5.4 — исключено из объёма

По решению пользователя ручная разметка и UI фидбека относятся к отдельной возможной работе после пайплайна, только если потребуется для агентского пайплайна. Это не оставшаяся задача текущего change.

### 5.5

Частично: Save Result и Tag Folder сохраняют TXT/JSON, есть provenance. HTML в notebook-визуализации и отдельный SAM RegionRunExport не являются полной ComfyUI-нодой отчёта.

### 5.6

Частично: Save Result проверен через ComfyUI API без браузера. Полного отчёта пока нет, поэтому проверка всего требования невозможна.

### 6.1

Частично: сохранены реальные ComfyUI-результаты Multi SAM/YOLO и двух кадров папки. Есть прежние notebook-прогоны четырёх изображений. Полного нового ComfyUI-отчёта приёмки всех случаев helmet/eyelashes нет.

### 6.2

Частично: есть сведения о версиях и реальные GUI/API-прогоны; полноценная матрица не оформлена; фильтрация без инференса отсутствует.

### 6.3

Частично: notebook-контракты покрыты регрессиями, предыдущие реальные notebook-прогоны сохранены. Итоговая повторная приёмка после всех ещё незавершённых этапов и проверка отключения пакета не выполнены.

## Ключевые основания

- Scripts/wdv3_timm.py:get_tags и Scripts/region_tagging.py:tag_batch отсекают scores до возврата данных.
- comfy_nodes.py:WDTagger.tag передаёт настройки прямо в вычисления; отдельной Apply Filters нет.
- Scripts/preprocess.py:_merge_source сохраняет источник победившего максимума; истории всех решений нет.
- comfy_config.py:combine_detectors/combine_sources проверяют уникальность имён, а не объединяют свидетельства.
- comfy_nodes.py:WDModelLoader.INPUT_TYPES содержит model/device, но не флаг скачивания. tag_regions передаёт YOLO device=cpu.
- comfy_ui_nodes.py:progress проверяет прерывание; Tag Folder передаёт callback, legacy IMAGE-теггер — нет.
- comfy_processing.py:tag_frames возвращает WD на CPU в finally; tests проверяют cleanup и SAM offload на ошибке.
- web/results.js содержит просмотр принятых scores и источников; сериализуемого feedback, поиска и режима rejected нет.
- comfy_export.py сохраняет подробный JSON и итоговую TXT-строку; HTML и feedback не реализованы.
- Scripts/region_export.py и Scripts/region_figure.py реализуют отдельные SAM/notebook-возможности, которые не подключены как полный ComfyUI Export Report.
- docs/verification/typed-contracts/notebook.json и docs/verification/notebook-setup/ описывают прежние notebook-проверки, не финальную приёмку текущего change.

Архивирование по результатам аудита преждевременно. Код и требования в рамках
этого аудита не изменялись; уточнены только статусы и основания в tasks.md.
