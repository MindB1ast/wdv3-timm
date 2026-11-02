# Проверка модельных профилей — 2026-11-02

Дополнительная проверка списка в работающем ComfyUI: до перезапуска его
`/object_info/WDV3ModelLoader` возвращал только прежние пять aliases, несмотря
на обновлённые файлы через junction. После перезапуска проверено 31 значение,
сортировка по Macro F1@0.40 и fallback по дате публикации. Default `vit`
сохранён. Подтверждение API: [live-catalog.json](live-catalog.json).
Шесть профильных проверок каталога/ComfyUI прошли; Pyright каталога без ошибок.

Реализован общий загрузчик и preprocessing для notebook и ComfyUI. Каталог
содержит все 21 модели dbv4-full ranklist, пять дополнительных dbv4-full из
организации AnimeTimm и пять прежних WD v3: всего 31 selectable model.
Manifest также хранит остальные репозитории обоих авторов с объяснением исключения.

## Результаты

- Полный regression suite: **178/178 passed**, без skipped и ошибок.
- Pyright: **0 ошибок** в положительном контракте; negative и SAM-negative
  выдали только ожидаемые диагностики. Проверено Python 3.12.15, Torch 2.6.0+cpu,
  timm 1.0.30; зависимости установленного окружения не заменялись.
- Прочитаны и проверены реальные config/preprocess для **24 из 31** моделей;
  семь остальных возвращают HTTP 403. Снимки привязаны к Hub revision.
- RGB/RGBA прямоугольные входы совпали с независимым воспроизведением
  опубликованной Pillow/torchvision-цепочки **без различий тензоров**.
- Реальные WD ViT, CAFormer B36 и ConvNeXtV2 Huge прошли CPU inference,
  сравнение полного кадра с двумя регионами и сравнение с ComfyUI.
  Загрузка и inference повторены с запрещённой сетью.

| Модель | Полный кадр | Региональный batch | Цвет |
| --- | --- | --- | --- |
| SmilingWolf/wd-vit-tagger-v3 | `[1,3,448,448]` | `[2,3,448,448]` | BGR |
| animetimm/caformer_b36.dbv4-full | `[1,3,384,384]` | `[2,3,384,384]` | RGB |
| animetimm/convnextv2_huge.dbv4-full | `[1,3,512,512]` | `[2,3,512,512]` | RGB |

Сырые результаты real-model проверки, фактические профили и revision находятся
в [smoke.json](smoke.json). Для старого вручную установленного WD-комплекта
Hub revision неизвестна; записан SHA-256 фактически загруженного файла весов.
Численный допуск сравнения реальных score: `rtol=1e-5, atol=1e-6`.

## Покрытие требований

| Контракт | Проверка |
| --- | --- |
| Полный каталог, aliases, прямой repo ID, отсутствие запросов при показе списка | `test_tagger_catalog.py` |
| Сериализация, полные C/H/W, порядок операций и fingerprint | `test_tagger_profile.py` |
| Размеры 512/384/448, синтетические 224 и прямоугольный вход, несовместимый профиль | `test_tagger_preprocessing.py` |
| Цвета, нормализация, RGBA, отсутствие двойного padding/BGR, full/region parity | `test_tagger_preprocessing.py`, real smoke |
| Локальные комплекты, один формат весов, namespaces, доступ/404, словарь | `test_tagger_resources.py`, `test_comfy_loaders.py` |
| Старые notebook/SAM-вызовы и возврат двух списков, геометрия областей | существующие `test_current_behavior.py`, `test_sam_pipeline.py`, `test_sam_regions.py` |
| Сохранённые входы ComfyUI, repo ID override, model_info, обновление runtime | `test_tagger_comfy.py` |
| Смена профиля/словаря инвалидирует оценки, смена фильтров не запускает inference | `test_tagger_comfy.py`, существующие `test_comfy_evidence.py` |

## Ограничения внешнего доступа

HTTP 403 препятствует чтению файлов семи моделей: MobileNetV3 `.r224`, MobileViT,
ConvFormer S36/B36, RepViT M2.3, EVA Giant и ViT GiantOpt. Имена и ответы записаны
в `audit/fixtures/tagger-profiles/access.json`. Для `.r224` сервер явно сообщил
ожидание одобрения автора. Все семь остаются в каталоге и используют свои файлы
при предоставлении доступа; они не выдаются за модели с проверенным профилем.
Синтетическая проверка 224×224 не подменяет проверку реальных файлов `.r224`.

Real-model smoke охватывает три указанных checkpoint, а не все 31 веса.
Остальные доступные профили проверены без скачивания весов; улучшение качества
тегов, CUDA и запуск настоящей сегментационной модели этим change не оцениваются.

## Повторение

Из корня репозитория с зависимостями проекта:

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path temp/yolo-settings).Path
$env:MPLCONFIGDIR = (Resolve-Path temp/matplotlib-cache).Path
python audit/run_checks.py --report temp/tagger-regression.json
python audit/check_types.py --report-dir temp/tagger-types
python audit/run_tagger_smoke.py
openspec validate add-metadata-driven-tagger-models --strict
```

Smoke использует только полные локальные комплекты; веса Huge и CAFormer уже
скачаны в `models/taggers/animetimm/`. Скрипт не пишет TXT рядом с исходниками.
Обновить pinned fixtures после предоставления доступа можно через
`python audit/fetch_tagger_profiles.py`; веса при этом не скачиваются.
