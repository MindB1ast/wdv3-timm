# SAM 3: области и контекст

SAM 3 включается через `ScriptOptions.sam_config`; без этого поля старый WD/YOLO-путь не импортирует SAM и сохраняет свои форматы. VLM будет отдельным change.

Конфигурация — `sam_sources.example.json`, отдельная от `detectors.json`. Пути checkpoint разрешаются относительно файла конфигурации. Каждый источник имеет уникальные `source_id` и `name` (имя не должно совпадать с YOLO), непустой текстовый `prompt` и собственные правила WD. Промпт не добавляет предсказанный тег автоматически.

В Python передавайте эти файлы разными параметрами:

```python
params = ScriptOptions(
    ImageFolder="./pictures",
    detectors_config="detectors.json",           # только YOLO
    sam_config="sam_sources.example.json",       # SAM 3
    save_txt=False,
)
result = BatchTagging(params)
```

Если SAM-конфиг передан как `detectors_config`, загрузчик выдаёт явную ошибку с правильным параметром до загрузки весов. Без `sam_python` используется Python текущего ноутбука/команды. При необходимости `sam_python` явно указывает отдельное окружение.

### Автовыбор и загрузка

Пример уже настроен на предоставленный пользователем `unsloth/sam3.1`:

```json
"model": {
  "checkpoint": "models/sam3/model.safetensors",
  "model_id": "unsloth/sam3.1",
  "device": "auto",
  "backend": "auto",
  "allow_download": true
}
```

`device=auto` проверяет доступную CUDA пробным tensor; если она недоступна, выбирает CPU. На ноутбуке с AMD и обычным CPU PyTorch выбирается CPU. Явные `cpu`, `cuda`, `cuda:N` сохраняются; ошибка CUDA не маскируется, если она выбрана явно. Автовыбор не устанавливает драйверы и не меняет библиотеки.

`backend=auto` читает ключи safetensors/локальный config: Transformers SAM 3/3.1 направляется в `Sam3Model`, native Meta — в официальный загрузчик. `.pt` остаётся native; этот backend всё ещё требует CUDA, автоматически конвертировать его веса в Transformers нельзя. Явные `backend=native`/`transformers` проверяются на совместимость с найденными весами.

Существующий `model.safetensors` не скачивается повторно. Если рядом нет config, processor или tokenizer, они берутся из `model_id` в кэше либо скачиваются при `allow_download=true`. Существующие файлы не заменяются. Один checkpoint без этих метаданных недостаточен для Transformers. Неизвестный исходный репозиторий не угадывается: задайте `model_id` явно. Вместо файла можно указать полный локальный каталог модели или только `model_id` для загрузки в HF-кэш. `revision` позволяет закрепить ревизию репозитория.

При `allow_download=false` используются только локальные файлы/кэш, без сети. Скачивание модели не означает автоматическую установку Python-пакетов.

### Форматы весов

Native backend использует архитектуру `facebookresearch/sam3`. Поддерживаются официальный `sam3.pt` и native `.safetensors` с исходными ключами этой архитектуры, в том числе с префиксом `detector.`. Native safetensors загружаются со строгой проверкой всех ключей и размеров; случайно инициализированные недостающие слои не допускаются. Для этого нужен пакет `safetensors`.

Файл Hugging Face Transformers `model.safetensors` с ключами `detr_decoder.*`/`vision_encoder.*` имеет другую схему весов и автоматически выбирает Transformers backend. CPU-инференс выполняется в float32; native библиотека SAM/CUDA для него не нужны. Неожиданные/недостающие ключи и неверные размеры при загрузке приводят к отказу. Квантованные FP8/INT8-файлы не поддерживаются этим загрузчиком; используйте обычный `model.safetensors`. Исходные файлы весов не конвертируются и не изменяются автоматически.

| Поле | Умолчание / смысл |
|---|---|
| `model.device` | `auto`; также `cpu`, `cuda`, `cuda:N` |
| `model.backend` | `auto`; также `native`, `transformers` |
| `model.model_id` | исходный HF-репозиторий для недостающих файлов |
| `model.revision` | необязательная закреплённая ревизия HF |
| `model.allow_download` | `false`: использовать только локальный checkpoint |
| `confidence` | 0.5, отбор результатов сегментации |
| `mask_threshold` | 0.5, порог вероятности пикселя маски |
| `min_mask_area_px` | 1, минимальная площадь бинарной маски |
| `max_regions` | 64, максимум экземпляров на источник и изображение |
| `view_mode` | `bbox`; также `mask` |
| `padding.unit` / `value` | `px` / 0; альтернативная единица `fraction` |
| `background_rgb` | `[255,255,255]`, фон вне видимой маски |
| `region_gen_threshold`, `region_char_threshold` | общие пороги WD, если поле не задано |

Поддерживаются прежние правила `remove_tags_from_full`, `remove_tags_from_region`, `add_tags_to_region`, `exclude_from_region`. Score сегментации и вероятность тега WD — разные величины.

`sam_python` необязателен: по умолчанию используется текущий Python; можно явно выбрать другое окружение. `sam_batch_size` (по умолчанию 1) ограничивает число кадров сегментации, `region_batch_size` (8) — размер tensor batch регионов WD. Оба положительные; прежний `batch` не переопределяется. `sam_timeout` (300 секунд) — предельное ожидание одного запроса worker; для медленного CPU его можно увеличить. `regions_output_dir` по умолчанию `./region_runs`.

## Геометрия

`bbox` — минимальная осевая рамка непустых пикселей исходной маски, `[x1,y1,x2,y2]`, правая и нижняя границы исключаются. Пиксель `(99,79)` имеет bbox `[99,79,100,80]`. Маска хранится в каноническом RGB-кадре после применения EXIF orientation; исходный размер и преобразование записываются отдельно. `crop_bbox` — фактическая вырезка с контекстом, ограниченная кадром.

Padding `px` округляется вверх. Для `fraction` отступ по X равен `ceil(width*value)`, по Y — `ceil(height*value)`. Например, 100×40 и 0.1 дают 10 пикселей слева/справа и 4 сверху/снизу.

- `bbox`: все исходные RGB-пиксели внутри `crop_bbox`.
- `mask`: видимая маска расширяется прямоугольной окрестностью с радиусами padding по X/Y; вне неё — `background_rgb`. Нулевой padding оставляет только исходную маску. Исходная сохранённая маска всегда неизменна.

WD квадратный padding и resize применяются после подготовки области; они не добавляют контекст исходной сцены. Метаданные view сохраняют offset, отступы, цвет и fingerprint; их достаточно для повторения выбора пикселей по сохранённому кадру и маске.

## Зависимости Transformers / CPU

В Python, выбранном для SAM, установите:

```powershell
python -m pip install -r requirements-sam-transformers.txt
```

Это отдельный список зависимостей; PyTorch устанавливается отдельно для вашего железа. Проверенный CPU API: Python 3.12.15, torch 2.6.0+cpu, Transformers 5.22.1. NumPy ограничен `<2` для совместимости с текущим notebook-стеком. Для сохранения старых pins WD можно использовать отдельное окружение и `sam_python`; устанавливать Transformers поверх старого закреплённого HF-стека без проверки совместимости не требуется.

CPU использует float32: только веса пользовательского SAM 3.1 занимают около 3.4 GB ОЗУ, рабочие tensors требуют дополнительной памяти. Скорость и полный расход памяти этой модели на ноутбуке ещё не измерены. CPU smoke с небольшой реальной SAM-архитектурой проверяет API и протокол, а не качество или скорость полного checkpoint.

## Зависимости native CUDA backend

Официальный backend требует Python ≥3.12, PyTorch ≥2.7 с CUDA ≥12.6 и совместимую NVIDIA GPU. Не обновляйте автоматически окружение WD с его `torch==2.6.0`. Создайте отдельное окружение, установите совместимый CUDA PyTorch по официальной инструкции и SAM из исходников:

```powershell
py -3.12 -m venv temp/sam3-runtime
git clone https://github.com/facebookresearch/sam3.git temp/sam3-source
# Выберите и сохраните ревизию для своего GPU smoke:
git -C temp/sam3-source rev-parse HEAD
temp/sam3-runtime/Scripts/python.exe -m pip install -e temp/sam3-source
```

Команда установки SAM предполагает, что совместимый CUDA PyTorch уже установлен в это окружение. Проверенная GPU-ревизия в этом репозитории пока не установлена; используемая версия и хеши критичных исходников записываются в каждый manifest. Перед объявлением совместимости выполните GPU smoke из `examples/sam3_smoke.py`. Фактически проверенные платформы перечислены в отчёте приёмки.

Веса `facebook/sam3` требуют доступа через Hugging Face. Разместите checkpoint по пути из конфига. С `allow_download=false` backend не вызывает HF-загрузку. С `true`, если файла нет, разрешена явная загрузка `sam3.pt` через штатный HF-кэш и авторизацию окружения. Токен не передаётся через JSON и не сохраняется в manifest. `sam_python` — полный путь к Python этого окружения. Worker запускается локально, без HTTP-сервера; `preflight` проверяет окружение до обработки изображений. Отсутствующая GPU, несовместимый torch и ошибка модели не являются пустой детекцией.

Основания: [официальная установка](https://github.com/facebookresearch/sam3#installation), [image processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py). Версия API фиксируется хешами исходников в manifest; будущая версия upstream требует повторной проверки.

## Экспорт и ошибки

Каждый запуск создаёт `regions_output_dir/<run_id>/manifest.json`. В `images/<image_id>/` находятся `canonical.png`, отдельные lossless `mask-N.png` (0/255) и `record.json`. `image_id` зависит от относительного пути и содержимого исходника. Все ссылки на assets относительные, поэтому каталог можно перенести целиком. Исходная маска сохраняется независимо от режима и padding. PNG-вырезки для WD воспроизводятся по кадру/маске/view, а не обязательно дублируются на диске.

Каталог `regions_output_dir` исключается из рекурсивного поиска исходных изображений, в том числе при повторном запуске. Он может быть подпапкой входного каталога, но не самим входным каталогом.

`manifest.json` содержит настройки, provenance SAM/WD и статусы изображений: `read`, `segmented`, `tagged`, `complete`, `error`. Запись становится `complete` только после сохранения всех файлов. Сбой может оставить неподтверждённые файлы без `complete`; их нельзя считать готовым результатом. Итоговый статус запуска — `complete`, `complete_with_errors`, `cancelled` или `error`. Manifest обновляется после каждого изображения; автоматическое возобновление не реализовано.

Повреждённый входной файл пропускается с записью причины. Отказ worker, нарушение протокола, OOM или ошибка сохранения останавливают запуск и сохраняют уже завершённые результаты. При OOM уменьшите `sam_batch_size`/`region_batch_size`. `Ctrl+C` отменяет текущий запрос и завершает worker; программная отмена доступна через `run_sam_batch(options, cancelled=callback)`. Тайм-аут задаётся `sam_timeout`.

Новые маски/JSON никогда не пишутся поверх исходников. Старое `save_txt=True` по-прежнему явно включает существующий TXT-экспорт рядом с исходниками; SAM-пример использует `save_txt=False`. `output_file=None` отключает дополнительный legacy JSON, но каталог SAM-экспорта сохраняется всегда.

## Повторное использование и будущий ComfyUI

`load_region_record(root, image_id)` проверяет завершённость и checksum кадра. `saved_region_input(...)` восстанавливает RGB-вход; `retag_saved_region(...)` возвращает новый региональный WD-результат без SAM и без изменения старого экспорта. Для повторения исходных фильтров передайте `rules=DetectorConfig(...)`; без `rules` применяются только пороги переданного `RegionTagger`. Все варианты сохраняют исходный region ID, mask и bbox. Изменение view даёт новый fingerprint.

Контракт для будущего `WD_REGIONS`: канонический кадр + `RegionGeometry` + `RegionViewMetadata`, отдельно от результатов текстовой модели. `mask_path` разрешается относительно корня экспорта; `bbox` не заменяется `crop_bbox`. Адаптер нод должен вызвать `prepare_region_view`, а не реализовывать свою геометрию. Ноды, VLM и UI ComfyUI в этом change не входят.

## Примеры запуска

Из корня репозитория в WD-окружении:

```powershell
python examples/sam3_batch.py ./pictures --sam-python ./temp/sam3-runtime/Scripts/python.exe --sam-config ./sam_sources.example.json --output ./region_runs --sam-batch 2 --region-batch 8 --show
```

Реальная GPU-приёмка (минимум два изображения разных размеров, несколько prompts, подходящие локальные веса):

```powershell
python examples/sam3_smoke.py ./pictures --sam-python ./temp/sam3-runtime/Scripts/python.exe --sam-config ./sam_sources.example.json --output ./temp/sam3-gpu-acceptance
```

Smoke выполняет `mask`/`bbox` с batch 1/2, проверяет точное совпадение масок и bbox, совпадение состава тегов и оценки с `rtol=1e-4`, `atol=1e-5`. Пустые результаты на всём наборе не считаются достаточной проверкой. В отчёте — версии, хеши исходников/весов, время и peak GPU memory сегментации/WD. Используйте новый каталог для каждого smoke.

GPU smoke требует CUDA как у WD, так и у SAM; CPU-запуск не выдаётся за GPU-приёмку.
Модель SAM загружается из файлов один раз при старте worker. Команды JSONL `load` и
`offload` перемещают существующую модель между настроенным устройством и CPU.
После каждой порции SAM освобождает embeddings и predictions, включая ошибку
декодирования, и выгружает модель. Основной процесс дополнительно ждёт ответа
`offload` перед загрузкой WD на GPU. Перед следующей порцией WD и YOLO
перемещаются на CPU, затем SAM возвращается на своё устройство.

Программное сравнение контекста без повторной сегментации:

```python
from Scripts.region_export import saved_region_input
from Scripts.sam_config import RegionViewConfig
from Scripts.visualization import inspect_saved_region

crop, metadata = saved_region_input(run_root, image_id, "sam_eyes", 0,
                                   RegionViewConfig("mask", "fraction", 0.2))
try:
    display(crop)  # RGB до transform модели; можно передать своему адаптеру
finally:
    crop.close()
inspect_saved_region(run_root, image_id, "sam_eyes", 0,
                     RegionViewConfig("bbox", "px", 16))
```

В интерактивном просмотре исходный кадр отображается без заливки. При наведении внутрь
SAM bounding box появляется только соответствующая маска с непрозрачностью 30%; при
уходе курсора она скрывается. Если под курсором несколько SAM-рамок, выбирается
наименьшая по площади: вложенные и пересекающиеся области доступны независимо от
порядка их отрисовки. Координаты учитывают масштабирование и панорамирование;
при перетаскивании маска скрывается. Обычные YOLO-рамки не включают маски. Это поведение
сохраняется при `figure.show()`, выводе фигуры в Jupyter и `figure.write_html(...)`.
Статические PNG/JPEG сохраняются с рамками, без наложения масок; RGB-вход для WD
в `inspect_saved_region` по-прежнему показывает фактическую вырезку для тегирования.

SAM-график в Jupyter размещается в отдельном iframe со встроенным Plotly. Обычный PNG
исходного кадра остаётся видимым под ним, если интерактивный JavaScript не выполняется.
Для обновления уже открытого ядра без повторного расчёта `result`:

```python
import importlib
import Scripts.region_figure as rf
import Scripts.visualization as vis

importlib.reload(rf)
importlib.reload(vis)
vis.view_image_results(result, 0)
```
