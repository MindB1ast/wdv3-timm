# Windows: native crash при загрузке WD

В Desktop-логе последним событием был запуск Tag Folder. Fatal stack: `torch/storage.py:471` → `safetensors/torch.py:360` → `Scripts/tagger_resources.py:load_network`. Windows Application Error 1000 указал `torch_cpu.dll`, код `0xc0000005`. Это падение при чтении весов WD, а не ошибка импорта на старте.

Изменение: на Windows `load_weight_state` выбирает `safetensors.torch.load_file(..., device='cpu', backend='pread')` до открытия checkpoint. В старых версиях API без аргумента backend используется чтение bytes через `safetensors.torch.load`. Native access violation невозможно перехватить обычным try/except. Файлы весов не меняются.

Подтверждение API: [Safetensors Torch API](https://huggingface.co/docs/safetensors/api/torch). Аналогичный стек и проверенный non-mmap обход опубликованы в [ComfyUI issue #15424](https://github.com/Comfy-Org/ComfyUI/issues/15424). Совпадение стека указывает на этот путь загрузки; сообщение Desktop само по себе не доказывает отсутствие библиотеки.

Реальная проверка выполнена отдельным процессом Python окружения ComfyUI: Python 3.13.12, Torch 2.12.1+cpu, safetensors 0.8.0, Windows. `animetimm/convnextv2_huge.dbv4-full`, checkpoint около 2.58 GiB: сеть создана, 692617532 параметра загружены со строгой проверкой ключей и размеров; выборочные значения конечны. Время загрузки 40.477 с, процесс завершился с кодом 0.

Запуск проверки: `audit/check_windows_weight_loading.py --model-folder models/taggers/animetimm/convnextv2_huge.dbv4-full --report temp/windows-weight-loading.json` через Python установленного ComfyUI. Проверяется загрузка сети, не полный пользовательский workflow и не inference. После перезапуска через Desktop нужно повторить исходный workflow.

Целевые unit-проверки: выбор pread на Windows, совместимость со старой API, обычный reader на других ОС, а также существующие resource/deferred-runtime контракты. Обновления или переустановки PyTorch/ComfyUI для этого обхода не выполнялись.
