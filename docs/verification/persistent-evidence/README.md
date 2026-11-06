# Приёмка persistent evidence и caption files — 2026-11-04

## Отложенные веса и пустой путь — 2026-11-05

В локальной истории двух запусков обнаружены reuse и пустой cache_dataset_root:
дисковое evidence тогда не создавалось. В логах повторялись загрузки WD и YOLO.
Убрано молчаливое отключение кэша IMAGE: reuse/reuse_only/refresh без папки теперь
сообщают ошибку до inference. Tag Folder продолжает брать root из Image Folder.

Load Tagger Model теперь читает словарь/preprocessing без создания WD-сети.
DeferredWDRuntime загружает сеть только при необходимости score; идентичность
проверяется по содержимому ресурсов без materialization tensor weights.
Legacy YOLO Loader тоже откладывает веса. Ключи обычного кэша local-files-v1
отличаются от прежних state_dict-ключей; старые refs остаются доступны Load Evidence.

`audit/test_deferred_runtime.py`, loader/profile/resource tests и автоматический
кэш: 18/18. Проверены полный hit со свежим loader, запрет load_network/score_batch,
пустой путь, reuse_only miss, refresh и изменение весов; отдельный финальный прогон
проверяет очистку уже загруженного runtime при изменении файла весов. Основная
регрессия — 245/245, frontend — 13/13, типы — 38 файлов без ошибок/предупреждений.
Реальные схемы/node API ComfyUI прошли, notebook cells 0/1 прошли с сохранностью
оригиналов (`deferred-notebook.json`). Браузерное управление не применялось.

`audit/run_automatic_cache_real.py` проверен на локальной WD ViT v3: первый расчёт,
новый metadata-only loader и новый процесс с изменёнными фильтрами и запрещёнными
load_network/score_batch. На повторе 0 загрузок весов и 0 inference, raw неизменны;
`automatic-real.json` содержит актуальное подтверждение. Само чтение/хеширование
локальных файлов не считается загрузкой сети и остаётся частью проверки совместимости.

## Уточнение обычного графа — 2026-11-05

Постоянный кэш подключён внутри существующих Tag Image / Regions, Collect Evidence
и Tag Folder. Настройки cache_mode/cache_dataset_root/cache_refresh добавлены в конец
Tagging Settings как optional, старые API-промпты и ключи сохранены. Отдельные
Inference Settings/Plan/Compute-ноды для этого маршрута не нужны.

`audit/test_automatic_cache.py`: 6/6 — прежний граф, другой runtime и фильтры,
отдельный процесс, папка из Image Folder, режимы reuse_only/refresh, отсутствие
повторных YOLO/SAM-вызовов и переиспользование переданных масок.
`audit/run_automatic_cache_real.py`: настоящая локальная WD ViT v3, новый runtime,
изменённые фильтры, запрещённый повторный score_batch, неизменные raw_scores — PASS.
Свидетельства: `automatic-real.json`. Загрузка модели остаётся возможной upstream.

`audit/run_persistent_acceptance.py` дополнительно проверяет реальный Collect Evidence
с новыми options и новым runtime без inference. Обновлены все примеры; обычный
маршрут — `examples/comfyui_wdv3_automatic_cache.json`. Основная регрессия — 240/240;
добавленная отдельная SAM-проверка входит в профильные 6/6. Frontend — 13/13,
включая labelPorts, меню и сохранность внутренних имён/связей. Pyright — 38 файлов,
0 ошибок/предупреждений; отрицательные контракты без изменений. Браузерные жесты
не выполнялись, подписи проверены на схемах и чистых frontend-функциях.

Реализованы постоянное dataset-local хранилище, независимые стадии и выборочный
досчёт, загрузка/импорт исходных оценок, операции над готовыми TXT, безопасный
финальный экспорт, общий API/CLI и отдельные ноды ComfyUI. Старые notebook-вызовы,
Save Result и Tag Folder сохраняют прежние параметры и defaults.

## Проверенное поведение

- Локальное `.wdv3`: SQLite, контрольные суммы, отдельные lossless PNG/маски,
  версии публикаций, manifest со страницами и перенос dataset_root.
- Несколько subprocess: один владелец стадии, waiting/reuse второго, heartbeat,
  восстановление после истечения lease, fencing устаревшей публикации.
- Ключи по содержимому: одинаковые path/size/mtime не скрывают новые пиксели/веса;
  перенос и размер батча не меняют готовую совместимую стадию.
- Полный hit не вызывает исполнителей. Новый источник/изображение досчитывается
  отдельно. Crop WD группируются ограниченными батчами, одинаковые пиксели
  переиспользуются между источниками. Padding/view отделены от геометрии SAM.
- Evidence после выхода из процесса фильтруется с запрещёнными model loaders.
  Import старого results_json использует raw_scores и честно объявляет refilter-only,
  если PNG/маски отсутствуют. Проверяются dictionary completeness и mapping pixels.
- UTF-8/BOM, неизвестные ручные теги без score, явные разделители, ordered edits,
  отдельный remove_regex, дедупликация, приоритет remove и происхождение.
- error/skip/append/overwrite, unchanged bytes, сопоставление подпапок, коллизии
  stem и регистра Windows, настоящий junction-выход за экспортную границу.
- Персональный OS-lock назначения, конкурентное создание, внешняя правка,
  сбои до/после replace, повтор operation_id и возобновление без повторного append.
- Только существующий TXT в skip-only API не требует inference; отдельный запрос
  evidence проверяется независимо. Запись TXT остаётся явным финальным действием.
- Policy/Format snapshots и refs разделены: смена разделителя использует прежнюю
  policy-публикацию; новые версии не меняют raw.

## Команды и результаты

Основное окружение: `.venv/Scripts/python.exe`, Python 3.12.15, CPU.

```powershell
.venv/Scripts/python.exe audit/run_checks.py --report temp/persistent-regression.json
.venv/Scripts/python.exe -m unittest audit.test_persistent_runtime audit.test_workflow_api audit.test_inference_plan audit.test_artifact_store -v
.venv/Scripts/python.exe audit/check_types.py --report-dir temp/persistent-types
node --test audit/test*.mjs
openspec validate add-persistent-evidence-and-caption-file-workflows --strict
```

Основная регрессия: 233 passed, 0 failed/errors/skipped. После последних изменений
storage/planner/runtime/API дополнительно проверены профильным прогоном (19/19),
CLI и продолжение записи папки — ещё 4/4; импорт и alpha-policy — 3/3. Отдельный
тест проверяет рабочее батчирование вырезок и reuse при изменении batch size.
Frontend: 12/12. Pyright: 38 файлов, 0 ошибок/предупреждений; 3 и 2 ожидаемых
диагностики отрицательных контрактов сохранены. Git diff whitespace check пройден.
Для создания настоящего временного Windows junction тесты запускаются вне sandbox;
цель и ссылка находятся только в `temp` проекта.

```powershell
.venv/Scripts/python.exe -B audit/run_persistent_real.py
.venv/Scripts/python.exe -B audit/run_comfy_notebook_acceptance.py
```

Реальные локальные веса: WD ViT v3, person_yolov8s-seg и face_yolov9c, без скачивания.
Сначала один файл с WD/YOLO, затем отдельный процесс с изменёнными правилами и
запрещёнными loaders, соседний итоговый TXT, после этого новое изображение и ветвь.
Первый WD и прежняя геометрия первого изображения переиспользованы. Counter/ref
свидетельства — `real-models.json`. Тестовый датасет изолирован в temp и удалён;
его refs документируют проверку и не являются refs пользовательского датасета.

Настоящие клетки ManagingTags.ipynb 0/1 и публичный BatchTagging прошли на CPU;
исходные изображения и TXT проверены по хешам, без ComfyUI API в notebook-процессе.

## ComfyUI

```powershell
<ComfyUI-python> -B audit/run_persistent_acceptance.py --comfy-root <ComfyUI-root>
```

Использован Python 3.13 установленного ComfyUI. Проверены 42 реальные схемы и node API,
штатный validate_prompt, FILES/IMAGE mapping, evidence → существующие фильтры/merge,
preview/Save Captions, новые поля пресетов и сохранение прежних индексов.
Все четыре режима и custom separator дают побайтово одинаковый результат UI API/CLI.
Проверены resume после сбоя после replace и конфликт внешней правки.
`comfyui.json` содержит итоги. Каталог портов проверен frontend-тестами:
42 ноды, 22 custom types, для каждого есть производитель и потребитель.
Связи всех поставляемых примеров проверены по реальным типам/индексам.

## Границы проверки

Computer use и браузерные жесты не выполнялись. «UI» здесь означает реальную схему
и node API установленного ComfyUI, а не автоматизированную браузерную приёмку.
Реальный SAM inference заново не запускался: сохранение масок, bbox/mask/padding,
смена prompt и независимость SAM/WD проверены фикстурами; рабочий адаптер использует
прежний SAM backend. Поддерживаются локальные процессы одного компьютера, без
сетевых файловых систем/распределённых workers и без автоматического GC.
Legacy eager loader при его явном подключении может загрузить модель upstream;
новые Inference Settings/Load Evidence работают без такой зависимости.
