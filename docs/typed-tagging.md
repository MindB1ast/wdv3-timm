# Типы результатов и политика метрик

Публичные вызовы и контейнеры сохранены. `BatchTagging(options)` возвращает `BatchResult | None`: кортеж двух списков либо None, если изображений нет. Типы описаны в `Scripts/types.py`; они не проверяют данные при выполнении и не преобразуют сохранённый JSON.

## Работа с результатом

```python
from Scripts import BatchTagging, ScriptOptions

result = BatchTagging(ScriptOptions(ImageFolder="images"))
if result is not None:
    detailed, merged = result
    if merged:
        tags = merged[0]["merged_tags"]  # WDTagResult
        caption = tags["caption"]  # str
        for tag, confidence in tags["general"].items():
            print(tag, confidence)
        if "provenance" in tags:
            print(tags["provenance"])
    for image in detailed:
        for regions in image["detectors"].values():
            for region in regions:
                print(region["bbox"], region["taglist"])
```

`WDTagResult` обязательно содержит caption, taglist и три карты general/character/ratings; provenance необязателен. У региона обязательны region_id и bbox: список четырёх координат `[x1, y1, x2, y2]` в исходных пикселях. Объединённая запись имеет необязательный txt_path. Проверяйте наличие optional-полей перед доступом. Модельное происхождение содержит source/model_score, ручное — source/assigned_score; значения score не округляются и не преобразуются. Форматирование сохраняет дополнительные поля региона и тип расширенного результата.

Ресурсы сохраняют nn.Module, LabelData, callable PIL→Tensor, YOLO, DetectorConfig и torch.device | None. Конфигурация принимает str/Path, поддержанные None-сентинелы остаются прежними. На границе timm применяется локальный cast только преобразования PIL→Tensor. Словари собственных результатов не стираются до Any/object. Единственное подавление в production касается NumPy 1.26: float32 является runtime-классом, хотя stub записывает generic alias; оно не отключает проверку score. Статический пример отдельно разрешает доступ к приватному контейнеру для его проверки.

## Проверка и редактор

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe audit/check_types.py
.\.venv\Scripts\python.exe -m pylint --persistent=n Scripts
```

Pyright проверяет перечисленные в pyrightconfig.json модули текущего пайплайна в standard, типы и положительный потребитель — в strict. Неперерабатываемые модули оформления графиков не объявлены строго типизированными. Отрицательный пример запускается отдельно: проверяются именно три ожидаемые ошибки поля/score/optional; ошибки импортов не считаются успехом.

В VS Code выберите `.venv` и ядро `wdv3-timm (.venv Python 3.12)`. Проектные variableTypes/functionReturnTypes включены в `.vscode/settings.json`: например, `caption` в примере получает подсказку str. Если подсказки скрыты интерфейсом, включите команду `View: Toggle Inlay Hints`. Глобальные настройки не меняются. Python 3.11 также поддержан; синтаксис `type` из Python 3.12 не используется.

## Метрики complete-labels-epsilon-v1

Эталон полон в оцениваемом словаре. TP — пересечение, FP — лишние предсказания, FN — пропуски. Отсутствие с обеих сторон не создаёт ошибок. Для каждого изображения и суммарных счётчиков каждого тега:

```text
epsilon = 1e-8
P_i = TP_i / (TP_i + FP_i + epsilon)
R_i = TP_i / (TP_i + FN_i + epsilon)
F_i = 2*P_i*R_i / (P_i + R_i + epsilon)
P = mean(P_i)
R = mean(R_i)
F1-score = 2*P*R / (P + R + epsilon)
```

Сводный F1 сохраняет исторический порядок: сначала средние P/R, затем их гармоническое среднее. Это не mean(F_i). Epsilon не добавляется к N, coverage или количеству тегов. Обе пустые стороны дают 0/0/0 и нулевые счётчики. Правильный непустой пример получает значения чуть ниже 1 без внутреннего округления. Пустая выборка и чужие ID по-прежнему ошибки; пропущенное предсказание считается пустым.

Обычный результат сохраняет Precision/Recall/F1-score. `calculate_tag_metrics(..., return_diagnostics=True)` дополнительно возвращает metric_policy_version, epsilon, F1_definition и прежние TP/FP/FN/coverage. Notebook показывает эту политику; CSV-столбцы и кортеж пяти результатов сравнения сохранены. Исторические отчёты не переписываются: для сравнения разных версий пересчитайте обе стороны одной политикой. Нулевые пустые строки и почти равные числа могут изменить порядок худших примеров.
