# Вложенные детекторы через Region Tag Rules

Дочерний стек подключается прямо к **Region Tag Rules** родительского источника.
Используются обычные YOLO Detector, Combine YOLO Detectors, Load SAM 3.1,
SAM Text / Region Settings и Combine SAM Queries.

## Пример: человек → лицо → глаза

1. Настройте обычный детектор глаз.
2. Подключите его выход detectors к **child_detectors** в Region Tag Rules лица.
3. Выход rules подключите к обычному детектору лица.
4. Детектор лица подключите к **child_detectors** в Region Tag Rules человека.
5. Rules человека подключите к обычному детектору человека.
6. Детектор человека подключите к обычному **Tag Image / Regions** либо **Collect Evidence**.

В граф не добавляются Cascade Stage, Collect Cascade Evidence, Route или ручные
stage_id/parent_stage. При нескольких детекторах в дочернем стеке они независимо
запускаются внутри каждого найденного родителя. Имена должны быть уникальны
в одном стеке; одинаковые имена под разными родителями допускаются.

Для дочернего SAM подключите к Region Tag Rules пару **child_sam_model** и
**child_sam_queries** (API-ключ child_sources). Можно одновременно подключить
child_detectors и SAM-пару. Чтобы получить следующий уровень, добавьте детей
в Region Tag Rules самого дочернего YOLO/SAM.

## Настройки Region Tag Rules

| Настройка | Назначение |
|---|---|
| child_detectors | Обычный YOLO-стек, запускаемый в bbox этого источника |
| child_sam_model + child_sources | Модель и обычные запросы дочернего SAM |
| child_tag_route = separate | Оставить дочерние области отдельными |
| child_tag_route = merge_to_parent | Дополнить теги непосредственного родителя |
| tag_enabled | Собственное WD-тегирование этого источника; false оставляет геометрию для детей |
| local_operations | Необязательная цепочка над собственными принятыми тегами |
| aggregate_operations | Необязательная цепочка после приёма тегов детей |

Все обычные пороги, фильтры, классы и модели остаются независимыми.
Значения родителя не наследуются. Без детей и новых операций правила сохраняют
прежние defaults и прежний формат JSON.

При загрузке старого workflow пустые новые виджеты child_tag_route/tag_enabled
автоматически получают separate/true. Явные merge_to_parent и false сохраняются;
тот же fallback действует при отправке графа, проверке и исполнении через API.

## Пиксели и результаты

Детектор ребёнка получает **исходные пиксели прямоугольного bbox родителя**.
Маска SAM, заливка, padding и resize WD-входа родителя туда не попадают.
WD-вырезка ребёнка использует только его собственную маску и настройки заливки.
Bbox округляется наружу и ограничивается исходным каноническим кадром; EXIF
нормализуется файловым загрузчиком. Адаптеры возвращают bbox после отмены своего
resize/letterbox; смещения переводят дочерние bbox/маски в исходный кадр.

Области сохраняют region_uid, parent_region_uid, stage_id, depth,
local_detection_index, исходные bbox/маски, detector_input_bbox/transform и
stage_path. Эти идентификаторы технические и генерируются автоматически.
Вложенность означает detected_within_parent_input, а не доказанное part_of.
Фоновая деталь внутри bbox не становится автоматически частью родителя.

Собственные raw_scores и принятые теги остаются отдельно от aggregate.
Route history хранит исходный источник, регион, путь получателей и причины отказа.
Ratings детей не передаются. Пороги WD родителя не применяются повторно к оценкам
ребёнка; remove/allowed родителя сохраняются. only_from_source, композиционные
ограничения и подтверждение full проверяются по исходному источнику.
Повторная сборка начинает с собственных наборов и не удваивает вклад.

**Tag Image / Regions и Tag Folder** исполняют всё автоматически.
Для локальных операций используйте обычную цепочку:
**Collect Evidence → Filter Tag Sets → операции → Merge Tag Sets → Inspect**.
Merge сам передаёт теги по сохранённой политике child_tag_route.

В Results Viewer/Inspect панель занимает всю ширину ноды и сохраняет её при
переключении изображений. Картинка масштабируется по доступной ширине с сохранением
пропорций; высота ноды определяет видимую область панели с прокруткой.

## Сохранение и API

Save/Load Pipeline Config сохраняет вложенность внутри обычных правил children,
через прежние detectors/sam_sources. Отдельного порта каскадного плана нет.
Минимальный пример переносимого конфига:

```json
{
  "schema_version": 1,
  "detectors": [{
    "name": "body", "model_path": "person_yolov8s-seg.pt",
    "children": {
      "schema_version": 1, "tag_route": "merge_to_parent",
      "detectors": [{
        "name": "face", "model_path": "face_yolov9c.pt",
        "children": {
          "schema_version": 1, "tag_route": "separate",
          "detectors": [{"name": "eyes", "model_path": "eyes.pt"}]
        }
      }]
    }
  }]
}
```

У children также есть необязательные sam_model/sam_sources, tag_enabled,
local_operations/aggregate_operations. В SAM-источнике children хранится в rules.
Вместо ввода JSON обычный workflow формирует его через Region Tag Rules.

Внутренний компилятор проверяет конечное дерево, циклы, дубли соседей и глубину
до loaders. Лимиты исполнения: 8 уровней, 64 области на родителя, 4096 на кадр.
Рабочие порции WD ограничены region_batch_size. При отсутствии родителей
дочерние loaders/inference не вызываются. Старый eager-loader, вручную подключённый
раньше по графу, мог уже загрузить веса.

Отчёт различает complete, empty, skipped/no_parents, truncated и failed;
found/kept/discarded/truncated объясняют отсечения. При ошибке
CascadeExecutionError.partial сохраняет завершённые кадры. Отмена передаётся наружу.
SAM worker переиспользуется в фазе детекции кадра и закрывается перед WD.

Каскад использует кэш процесса; смена политики не требует инференса, изменение
bbox инвалидирует зависимые этапы, а смена только родительской заливки — нет.
reuse_only запрещает новый инференс при отсутствии расчётов в процессе.
Дисковое планирование вложенных вычислений пока вне scope. Прежние плоские
workflow продолжают использовать существующий постоянный кэш как раньше.
Сохранённое evidence можно повторно фильтровать и объединять без моделей.

Примеры обычных графов: examples/comfyui_wdv3_cascade_separate.json и
examples/comfyui_wdv3_cascade_merge.json. **eyes.pt — заполнитель:** замените его
на свою локальную модель глаз. Примеры генерируются examples/build_comfy_workflows.py.
Проверки: audit/test_nested_detectors.py, audit/test_cascade*.py и
 audit/run_cascade_acceptance.py. Интерактивный UI и реальные веса отдельно не проверялись.
