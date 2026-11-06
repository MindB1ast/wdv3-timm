# Постоянное evidence

## Обычный граф — без новых обязательных нод

Для обычного Tag Folder достаточно выбрать папку в Image Folder, оставить
`calculation_mode=reuse` и пустой `dataset_folder_override`. Первый запуск считает,
повторный использует сохранённые расчёты. Для обновления уже записанных TXT/JSON
в Tag Folder включите `overwrite`; для просмотра без записи выключите `save_txt`
и `save_json`. Чтобы гарантированно запретить любой новый инференс, выберите
`calculation_mode=reuse_only`. Отсутствующие расчёты тогда вызывают понятную ошибку
с фактическим путём к кэшу, а не запускают модели.

Для изменения фильтров **Inference Settings не нужна**. Оставьте существующие
Load Tagger Model, YOLO/SAM-настройки и Tag Image / Regions либо Collect Evidence.
В подключённой **Tagging Settings** укажите:

1. «calculation_mode» (`cache_mode`) = reuse: автоматически брать готовое и досчитывать недостающее.
2. Для Tag Folder оставьте «dataset_folder_override» (`cache_dataset_root`) пустым.
3. Меняйте фильтры и запускайте тот же граф: совместимые raw-оценки загружаются с диска,
   а inference пропускается. Новое изображение или недостающая ветвь досчитываются.

Для Tag Folder папка автоматически берётся из существующего Image Folder, если
поле пути к кэшу пусто. Если путь переопределён, можно указать папку датасета
или саму `.wdv3`: повторная вложенная `.wdv3` не создаётся. Папка сохранения TXT/JSON
не меняет место хранения расчётов. Для IMAGE путь нужно указать один раз: тензор не содержит
исходного имени файла. Пустой путь для IMAGE при reuse/reuse_only/refresh сообщает
ошибку до inference: выберите папку либо явно cache_mode=off.

refresh пересчитывает выбранные имена детекторов/стадий из поля «refresh_selection»;
пустое поле означает всё. off отключает постоянный кэш. Все эти параметры задаются
в Tagging Settings и не дублируются в нодах выполнения. Старые API/presets без
cache_mode сохраняют прежнее поведение. Пример: `examples/comfyui_wdv3_automatic_cache.json`.

Load Tagger Model читает словарь и preprocessing без выделения WD-сети. Совместимость
проверяется по содержимому локальных файлов; веса загружаются только на miss/refresh.
Legacy Load YOLO Detectors тоже выдаёт настройки без загрузки весов.
Если SAM Segment Images стоит отдельно, его запуск
также не отменяется downstream-флагом; WD-оценки по готовым маскам переиспользуются.
Чтобы работать вообще без загрузчиков моделей, используйте отдельный Load Evidence.

Старые evidence-снимки не удаляются и по-прежнему загружаются через Load Evidence.
Автоматические записи с прежней state_dict-идентичностью могут потребовать разового
досчёта при переходе на новую файловую идентичность. После первого сохранения новый
loader использует кэш и после перезапуска процесса без загрузки весов.

Видимые подписи теперь различают «sam_queries», «detector_names»
и «region_detectors_json». Внутренние sources/previous_sources и индексы связей
сохранены, поэтому переименования не ломают старые workflow и API.

## Необязательный расширенный маршрут

Calculation Store Settings задаёт место хранения и режим, **не выбирает входные
картинки**. Dataset Files отдельно выбирает пути. Inference Settings, Plan и Compute
Missing Evidence нужны для явного планирования, ленивых моделей и агентного API.
Они не обязательны для обычного тегирования и смены фильтров.

Новая dataset-ветка использует явно выбранный `dataset_root`. По умолчанию
`ArtifactStore(dataset_root)` размещает данные в `<dataset_root>/.wdv3`.
Старые ноды сами это хранилище не создают. При ошибке доступа другой каталог
автоматически не выбирается. TXT остаются отдельным конечным результатом.

## Формат хранилища v1

- `store.json`: schema_version=1 и стабильный store_id.
- `index.sqlite3`: завершённые артефакты, claims, операции и страницы коллекций.
- `objects/<sha256>/data`: неизменяемые JSON и бинарные blobs, включая PNG масок.
- `runs/<run_id>/manifest.json`: запрос и ссылка на упорядоченные страницы индекса.

`computation_key` — SHA-256 канонического JSON `{stage, version, inputs}`.
Порядок ключей словаря не влияет на hash; содержание и версия влияют. NaN запрещён.
`artifact_id` — SHA-256 конкретного JSON артефакта, содержащего computation_key,
stage, run_id, publication_id, generation, зависимости, refs blobs и payload.
`run_id` обозначает исполнение; повторный refresh создаёт новую публикацию,
оставляя доступными предыдущие refs. Одни и те же бинарные bytes дедуплицируются.
Неизвестные версии отклоняются, автоматической миграции нет.

Сначала writer записывает sibling temp, flush/fsync, атомарно публикует blobs
и проверяет контрольные суммы. Затем транзакция SQLite публикует запись.
Незарегистрированные остатки не являются cache hit. Чтение проверяет также
контрольные суммы зависимостей; повреждение возвращается как invalidated.
Автоматическая очистка артефактов не выполняется.

## Несколько локальных процессов

Claim имеет уникального owner, монотонную generation и срок expires. Пока lease
действует, второй процесс получает waiting либо готовый проверенный ref.
`heartbeat` продлевает только действующее владение. После истечения другой процесс
получает новую generation; устаревший writer не может опубликовать результат.
Сбой до публикации может потребовать повторить незавершённую работу.
SQLite координирует процессы одного компьютера на локальной файловой системе;
сетевые диски и распределённые исполнители не входят в контракт.

## Датасет и коллекции

Manifest хранит относительный `relative_path` отдельно от image/content identity.
Два пути могут ссылаться на один расчёт, сохраняя отдельные будущие назначения TXT.
`write_manifest` принимает итератор, `iter_entries(page_size=64)` читает страницы
без загрузки всей коллекции или PNG в память. Перенос датасета вместе с `.wdv3`
сохраняет refs и mapping, а исходные файлы разрешаются относительно нового root.
Пути за пределами root и выход через symlink отклоняются.

`scan` исключает `.wdv3` и фактический store до рекурсивного обхода как для
изображений, так и для TXT. Служебные PNG и маски не становятся новым датасетом.

## Совместимость стадий

Пиксели идентифицируются по нормализованному RGB, размеру и версии decode/EXIF/alpha.
Модель — по содержимому весов, словаря, профиля подготовки и версии адаптера либо
явному сохранённому pin. Путь, size/mtime и batch size не доказывают совместимость.
План не вызывает loaders/download/inference. Неразрешённая локально идентичность
возвращает unresolved_identity, в том числе в reuse_only; pin не ищется в сети.

| Изменение | Необходимые стадии |
| --- | --- |
| Порог/regex/ownership | policy и зависимый текст |
| Разделитель/замена `_`/скобки | только text |
| Новое изображение | стадии новых пикселей |
| Новый источник | его geometry и отсутствующие WD crop |
| Padding/bbox/mask | новые фактические crop-пиксели и WD; geometry сохранена |
| Confidence/prompt/геометрические пороги | geometry затронутого источника |
| Веса/словарь/preprocessing WD | WD full/crop, без повторной geometry |
| Batch size/перенос файла | inference не требуется |

PolicySnapshot исключает форматирование и размеры батчей. FormatSnapshot хранит
разделитель, замену `_` и скобки отдельно. План — упорядоченный DAG с computation_key,
image_id/source, dependencies, status/reason/ref. Режим refresh имеет явный выбор
стадий/источников и инвалидирует зависимые результаты; старые публикации доступны.

## Evidence и импорт

Save Evidence сохраняет полные категории оценок, model dictionary identity,
source snapshots, image_id/размер, пиксели PNG и lossless маски отдельными blobs.
Load Evidence проверяет всё содержимое и собирает Bundle для существующих
Filter Tag Sets → операции → Merge Tag Sets либо Apply Filters и Inspect.

Импорт принимает Bundle evidence v1 или один/список текущих results_json с полными
raw_scores (general/character/ratings). Проверяются число меток, одинаковый словарь
записей и категории, model metadata, соответствие источников и региональные bbox/IDs.
Новый формат дополнительно сохраняет имена каждой категории для проверки словаря.
Merged-only JSON, TXT и HTML-отчёты не становятся evidence.

Capabilities объявляются явно: refilter, inspect_pixels, recrop_masks, resume_inference.
Старый results_json без PNG/масок остаётся refilter-only; загрузка не придумывает assets.
Для экспорта отдельно задаётся mapping. Привязка исходных изображений проверяет
фактические нормализованные пиксели относительно сохранённого PNG/pixel_identity;
отсутствие источника или несовпадение дают диагностику, не запрещая чистый refilter.

## Как запускать в ComfyUI

После обновления перезапустите backend ComfyUI и обновите страницу. Новые готовые
графы в `examples`:

- `comfyui_wdv3_incremental_folder.json`: Calculation Store Settings → Dataset Files
  и Inference Settings → Plan Calculations → Compute Missing Evidence → Filter Tag Sets
  → Remove Redundant Tags → Merge Tag Sets → Combine Captions → Plan Caption Export
  → Save Captions. Укажите dataset_root и локальную папку WD. Store по умолчанию .wdv3.
- `comfyui_wdv3_cold_restart_refilter.json`: Load Evidence вместо вычислений.
  Вставьте artifact_id/массив refs из Compute Missing Evidence либо manifest run_id.
  Загрузчики моделей не нужны. Load Evidence позволяет start/limit для страниц коллекции.
- `comfyui_wdv3_caption_txt_only.json`: Load Captions → Edit Captions → Plan/Save.
- `comfyui_wdv3_caption_append.json`: evidence → новые теги плюс готовые captions.

Plan Calculations принимает ровно один IMAGE или FILES. Для IMAGE сопоставление имён
задаётся явно JSON-массивом image_mapping_json. Оно необходимо для соседних TXT.
Для папки используйте Dataset Files: он исключает store до обхода. Планировщик также
отсекает служебные пути в FILES, полученных из прежнего Image Folder.

Повторный запуск в reuse проверяет артефакты и загружает модели только для misses.
reuse_only при недостающих расчётах возвращает причины, без скрытого inference.
refresh выбирает стадии (`wd_full`, `wd_crop`, `yolo_geometry`, `sam_geometry`)
или имена источников. Пустой refresh-selection означает полный пересчёт.
Маски/геометрия отделены от view; WD работает с содержимым вырезки, поэтому одинаковые
пиксели могут делить оценки в разных областях. Готовые raw и старые версии сохраняются.
Рабочий region_batch_size ограничивает группы отсутствующих WD-вырезок, а не ключ кэша.

Inference Settings описывает локальную WD-папку с весами/config/selected_tags.csv
и, когда требуется, preprocess.json. Поле sources_json содержит ленивые источники,
не объекты загруженных моделей. Пример YOLO (замените пути):

```json
[{"kind":"yolo","name":"body","confidence":0.3,"classes":[0],
  "model":{"local_path":"C:/models/person.pt","files":{"weights":"C:/models/person.pt"},
           "adapter":"yolo-v1","compute":{"device":"cpu"}},
  "rules":{"region_gen_threshold":0.35},"view_mode":"bbox","padding_px":0}]
```

SAM-источник имеет kind=sam, prompt, confidence/mask_threshold, min_mask_area_px,
max_regions, view_mode/padding_px и rules. Его model.files перечисляет weights и
локальные config/processor/tokenizer-файлы; model.settings — обычные настройки
SAM backend (model_id, checkpoint, backend, device, allow_download=false).
Порог геометрии/prompt меняет SAM-ключ; padding и view_mode его не меняют.
План не скачивает недостающие файлы. Saved pinned_identity_json берётся из
`model.identity` своего evidence; Load Evidence требует только ref и store.

Пресеты сохраняют необязательные calculation_store, inference, caption_read,
caption_edit, caption_export. Первые пять выходов Load Pipeline Config сохраняют
прежние индексы; новые выходы добавлены справа. Старые пресеты не активируют store.

## API/CLI для агента

Общий API находится в `Scripts.workflow_api`; CLI выводит только JSON в stdout,
обычные сообщения библиотек направляет в stderr:

```powershell
python -m Scripts.workflow_cli plan --request request.json
python -m Scripts.workflow_cli execute --request request.json
python -m Scripts.workflow_cli status --request status.json
python -m Scripts.workflow_cli load --request evidence.json
python -m Scripts.workflow_cli import --request import.json
python -m Scripts.workflow_cli refilter --request policy.json
python -m Scripts.workflow_cli caption-plan --request captions.json
python -m Scripts.workflow_cli caption-commit --request commit.json
```

Каждый запрос имеет schema_version=1, dataset_root, необязательный store_path.
plan/execute дополнительно: inference, files (относительные пути; без files — scan),
mode, refresh, options; execute требует operation_id. status требует operation_id.
load принимает artifact_id/artifact_ids либо run_id и start/limit; refilter — те же
refs, options, sources (снимки правил) и postprocessing. import принимает calculations
(Bundle JSON либо один/список results_json). caption-plan принимает entries вида
`{"relative_path":"sub/image.png","document":CaptionDocument}`, caption_export и
operation_id; caption-commit принимает plan, полученный из preview. Редактор и
CaptionDocument доступны через `Scripts.caption_files`, без моделей и UI.

Ответ: `{schema_version:1,ok:true,result:...}` или `{ok:false,error:{code,message}}`.
Exit codes: 0 — успешный ответ, 1 — некорректный запрос/ошибка, 2 — неполный план,
неполное вычисление либо conflict/failed при записи. Отчёт сохраняет run_id,
computation_key, stage/source/image_id, parent_refs, artifact_id и status/reason.

operation_id связывается с содержимым запроса и входных файлов. Повтор подтверждённой
операции возвращает прежний результат; новый запрос/изображения требуют нового ID.
Продолжение частичного refresh сохраняет уже завершённые записи. В UI пустой
operation_id экспорта автоматически идентифицирует конкретный план: после изменения
правил не требуется вручную менять ID. Явный ID с иным планом вызывает конфликт.

Skip существующего TXT отсекает модели только при `outputs=["captions"]` и
caption_export.write_mode=skip. При отдельном запросе evidence проверка независима.
Сами вычисления не записывают подписи; preview/commit остаются явным финальным этапом.
