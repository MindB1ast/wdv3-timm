# Приёмка add-typed-tagging-contracts-and-stable-metrics

Реализованы типы текущего WD-пайплайна и единая политика complete-labels-epsilon-v1. Пользовательские настройки моделей/фильтров и формы вызовов сохранены. Сводка: [summary.json](summary.json). Использование, охват checker и точные формулы: [typed-tagging.md](../../typed-tagging.md).

## Результаты

- Исходные 55 регрессий пройдены до изменений; после добавления восьми проверок проходят 63/63 в Python 3.11.17 и 3.12.15, без skip/xfail.
- Pyright 1.1.414: 15 проверяемых файлов, ноль ошибок; отдельный отрицательный набор выдаёт ровно три ожидаемые диагностики. Модуль типов и положительный потребитель проверяются в strict. Оформление графиков не объявляется полностью строго типизированным.
- Pylint 4.1.3: 10.00/10 в обоих окружениях, без новых отключений правил или повышения порогов.
- Три изолированные подмены обнаружены конкретными AssertionError: epsilon=0, единица при пустоте, mean(F_i) вместо исторической агрегации. [mutations.json](mutations.json) содержит пути импортов, конкретные падения и отдельные errors. При epsilon=0 дополнительные ZeroDivisionError — следствие подменённой арифметики; доказательством обнаружения служит именно численное AssertionError для правильного непустого примера.
- Реальные сохранённые ячейки ManagingTags/Analitycs выполнены в зарегистрированном ядре на четырёх копиях PNG/JPG с локальными WD ViT v3 и YOLO, CPU. Контрольный sample сравнен с результатами предшествующей notebook-приёмки: теги, scores, provenance, bbox совпадают. Созданы JSON/TXT/PNG/JPG, Plotly MIME; сравнение возвращает пять элементов. Переопределены только пути копий, доступная модель/конфигурация и флаги показа. Эталонные TXT находятся в отдельном каталоге копий.
- Публичные способы вызова проверены через AST без аннотаций: [api-comparison.json](api-comparison.json). Столбцы четырёх CSV совпадают с baseline: [csv-compatibility.json](csv-compatibility.json). Контрольные суммы оригинальных изображений/TXT и исторических audit/verification-отчётов сохранены: [integrity.json](integrity.json).

## Численные изменения

[numerical-deltas.json](numerical-deltas.json) рекурсивно сравнивает графики/таблицы, включая декодированные Plotly bdata. Изменились 78 числовых значений; максимальная разница 7.9999999514e-7 в процентной шкале. Подписи, форматы, прочие значения, порядок графического примера совпадают. Исходный analytics-before.json сохранён; новая проверка использует отдельный analytics-epsilon.json. Пустой пример меняет P/R/F1 с 1 на 0 и переходит к худшим строкам. Отдельные отчёты показывают прежний и текущий порядок. В близких оценках иная длина знаменателя может менять ранжирование; округление применяется только при отображении.

TP/FP/FN не изменились. Epsilon добавлен только к знаменателям P/R/F1, не к N/coverage. Исторический F1 остаётся гармоническим средним усреднённых P/R, что проверено на различающем примере. Версия и epsilon добавлены только в optional diagnostics и видимый notebook-отчёт. Для сопоставления разных политик нужно пересчитать обе стороны.

## Команды

```powershell
.\.venv\Scripts\python.exe audit/run_checks.py --report docs/verification/typed-contracts/tests312.json
.\temp\apply-runtime\Scripts\python.exe audit/run_checks.py --report docs/verification/typed-contracts/tests311.json
.\.venv\Scripts\python.exe audit/check_types.py --report-dir docs/verification/typed-contracts/types312
.\temp\apply-runtime\Scripts\python.exe audit/check_types.py --report-dir docs/verification/typed-contracts/types311
.\.venv\Scripts\python.exe -m pylint --persistent=n Scripts
.\temp\apply-runtime\Scripts\python.exe -m pylint --persistent=n Scripts
openspec validate add-typed-tagging-contracts-and-stable-metrics --strict
```

requirements-dev.txt фиксирует версии checker/stubs. Скрипты notebook/мутационной приёмки сохранены рядом для воспроизводимости в этом локальном окружении. В Python 3.11 проверочное окружение использует существующий .pth с runtime-зависимостями; NumPy закреплён на совместимой 1.26.4. Небольшое расширение охвата analytics_charts.py уточняет только тип аргумента существующей функции графика, без изменения её построения. Два локальных pyright-ignore документируют runtime-класс NumPy 1.26 и намеренный доступ теста к приватному контейнеру; контракты собственных данных не отключены.

## Сценарии и границы

| Требование | Доказательство |
|---|---|
| Поля региона, scores, optional-поля | assert_type в positive.py и три конкретные negative-диагностики |
| Ресурсы и LabelData | assert_type для nn.Module, PIL→Tensor, names и device; проверки str/Path/None |
| Совместимые контейнеры и пустой каталог | исходные batch-регрессии, JSON round-trip, notebook |
| Полный эталон: TP/FP/FN, пустые наборы | независимый Decimal-эталон и таблицы в test_typed_contracts.py |
| Правильный непустой пример с epsilon | численная проверка с допуском 1e-12 и мутация epsilon=0 |
| Исторический F1 и покрытие | различающий пример, coverage=0.5, мутация mean(F_i) |
| Пустая выборка/чужие ID | сохранённые ValueError-регрессии |
| Версия/epsilon/CSV/return shape | notebook.log, csv-compatibility.json, diagnostics-проверки |
| Изоляция данных и воспроизводимость | integrity.json, types311/types312 и project settings |

SAM, VLM, новые классификаторы, ComfyUI и агенты не реализуются этим change. Существующая фабрика аналитики не переписывалась. Для ComfyUI пригодны BatchResult/WDTagResult/RegionResult и версия оценщика; задачи ComfyUI остаются невыполненными. Дельта сохраняет сценарии завершённой ревизии. Архивирование отдельно: сначала fix-audited-tagging-defects, затем этот change. Основные спеки автоматически не синхронизируются.
