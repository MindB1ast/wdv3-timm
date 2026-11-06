# Проверка вложенных детекторов через Region Tag Rules

Проверено 2026-11-06, change add-cascaded-region-processing. Этот отчёт заменяет
прежнюю приёмку отдельного интерфейса Cascade Stage.

## Результат

- Старые публичные Cascade Stage/Combine/Plan/Collect/Route и тип WDV3_CASCADE_PLAN
  сняты. Отдельный порт плана в Save/Load Config убран. Несвязанные изменения
  рабочего дерева не откатывались.
- Region Tag Rules принимает обычный дочерний YOLO-стек и/или SAM-пару. Дальнейшие
  уровни задаются такими же rules у дочернего источника; ручные ID не вводятся.
- В реальном Python API установленного ComfyUI проверены Region Tag Rules,
  YOLO Detector, SAM Loader/Source, Tag Image / Regions, Collect Evidence,
  Filter/Merge Tag Sets и общий backend Tag Folder.
- Проверены три уровня, два родителя, разные пути при одинаковых именах детей,
  автоматический merge, оба режима передачи, независимость настроек, собственная
  SAM-заливка ребёнка и исходный фон дочернего поиска внутри bbox родителя.
- Вложенность проходит JSON и обычный Save/Load Pipeline Config; старые плоские
  настройки сохраняют прежнюю форму. Evidence, исходные оценки и дерево переносимы.
- Кэш и модели переиспользуются; пределы и отмена/ошибки сохраняют диагностику.
  Два SAM-этапа одной модели используют один worker в фазе детекции кадра;
  перед WD он закрывается. reuse_only запрещает новый инференс.
- Inspect/JSON/HTML сохраняют дерево, собственные теги/агрегаты и происхождение.
  Примеры с merge/separate состоят из обычных нод, без Cascade/Route.

## Свидетельства

- tests.json: 25/25 профильных тестов из audit.test_cascade,
  audit.test_cascade_runtime, audit.test_cascade_tags, audit.test_nested_detectors.
- regression.json: 53/53 тестов audit.test_comfy_evidence,
  audit.test_tag_postprocessing, audit.test_comfy_config, audit.test_comfy_processing,
  audit.test_artifact_store, audit.test_persistent_evidence, audit.test_sam_worker.
- acceptance.json: audit/run_cascade_acceptance.py --comfy-root <ComfyUI>,
  выполненный Python установленного ComfyUI. Обычные схемы/API, три уровня,
  смешанный стек, папка, пресет, отчёты, валидный prompt и примеры прошли.
- audit/run_preflight_acceptance.py --comfy-root <ComfyUI>: PASS, 42 схемы,
  28 некорректных конфигураций и дополнительные проверки child_detectors,
  обязательной SAM-пары и неверного типа. Ноды не исполнялись при валидации.
- python audit/check_types.py --report-dir docs/verification/cascaded-regions/types:
  PASS, 40 положительных файлов без ошибок; отрицательные примеры выдали ровно
  ожидаемые диагностики.
- node audit/test_connection_catalog.mjs: PASS, 42 ноды, 22 типа.
- node audit/test_connection_suggestions.mjs и node --check web/results.js: PASS.
- Исправление пустых новых виджетов: node audit/test_region_rule_defaults.mjs
  проверяет восстановление и сериализацию, сохранение merge_to_parent/false и
  неизменность полей тегов. run_preflight_acceptance и run_cascade_acceptance
  дополнены пустыми/null значениями и проверкой явных значений через реальные API.
- openspec validate add-cascaded-region-processing --strict: PASS.

## Исправление сжатия просмотрщика

`audit/run_viewer_native_layout.py --browser <Chrome.exe>`: PASS, 24 сценария
в настоящем frontend установленного ComfyUI с нативным DOM-widget mount.
Проверены Results Viewer и Inspect: повторное переключение четырёх изображений
(портрет, альбом, квадрат, небольшой портрет), выбранная нода и фокус dropdown,
resize до 1000 и 520 px, масштаб 75%, 100% и 65%.
Измерения: `viewer-native-layout.json`; воспроизведение до исправления:
`viewer-native-layout-before.json`.

Причина: WidgetLegacy бокового инспектора присваивает общему объекту виджета
ширину своего canvas (233 px). Нативный graph mount использует это значение
вместо ширины ноды 760 px: панель сжималась до 159.75 px при масштабе 75%.
Теперь ширина панели следует текущей ширине ноды, даже при записи ширины
боковым инспектором; современный computeLayoutSize управляет высотой,
computeSize остаётся только для старого frontend.

Браузер запускается в отдельном профиле через локальный proxy только для GET:
POST/PUT/DELETE не передаются, пользовательский граф и настройки не записываются,
модели не запускаются, изображения синтетические. Прямое управление открытым
Comfy Desktop и ручная приёмка пользователем не выполнялись. При масштабе 50%
этот frontend штатно скрывает DOM-виджеты; проверка видимой панели использует 65%.

Прежний `audit/run_viewer_layout.py` (12 сценариев с изолированным flex mount)
не воспроизводил проблему native layout и сам по себе не подтверждает исправление.

`node audit/test_region_selection.mjs`, `node audit/test_preview_limit.mjs` и
`node --check web/results.js`: PASS.

## Границы проверки

Реальные веса WD/YOLO/SAM и GPU в этом прогоне не запускались. Использованы
детерминированные тестовые детекторы/WD и штатный SAM-адаптер с тестовым worker.
Автоматизированная браузерная проверка описана выше; ручная визуальная приёмка
пользователем не выполнялась.
Каскад использует кэш процесса; дисковое планирование вложенных вычислений
остаётся вне scope. Прежний постоянный кэш плоского графа сохранён.
eyes.pt в примерах заменяется локальной моделью глаз пользователя.
