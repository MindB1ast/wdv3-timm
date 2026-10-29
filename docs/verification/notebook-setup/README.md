# Окружение ноутбуков Python 3.12

В существующую `.venv` проекта установлен полный набор из `requirements.txt`,
включая matplotlib, pandas, numpy, PyTorch/torchvision/torchaudio, timm,
ultralytics, scikit-learn, Pillow, Plotly, nbformat, ipykernel и ipywidgets.
Добавлен pip, чтобы `%pip` работал из ноутбука. Проверка зависимостей:
88 установленных пакетов совместимы; полный список в `installed.txt`.

Python: `.venv/Scripts/python.exe`, версия 3.12.15.
В requirements сохранены прежние pandas/scikit-learn для Python 3.11,
а для Python 3.12 указаны pandas 2.2.3 и scikit-learn 1.5.2.
Удалены повторные записи зависимостей, явно добавлены пакеты Jupyter/виджетов.

В обоих ноутбуках сохранено ядро **wdv3-timm (.venv Python 3.12)**,
идентификатор `wdv3-timm-venv`. Код, результаты и счётчики ячеек не менялись;
это подтверждено сравнением с резервными копиями в `temp/notebook-setup`.
В `.vscode/settings.json` выбран интерпретатор `.venv`.
У зарегистрированного ядра YOLO_CONFIG_DIR и MPLCONFIGDIR находятся внутри
project/temp и не требуют изменения глобальных настроек Ultralytics.

Проверка выполнена настоящими ячейками отдельного процесса IPython/Jupyter:

- Все импорты обоих ноутбуков прошли.
- ScriptOptions и BatchTagging выполнили WD ViT v3 + YOLO на копии изображения.
- Созданы два JSON, TXT и PNG с регионами внутри temp.
- Выведены notebook MIME-результаты Matplotlib, Plotly и ipywidgets.
- Все 55 регрессий прошли на том же Python 3.12 без ошибок и пропусков.

Результаты: `kernel.json`, `rendering.json`, `tests.json`, `notebooks.json`.
Проверочное выполнение использовало копии данных и не запускало произвольные
пользовательские ячейки массового редактирования тегов.

В VS Code нужно перезапустить ранее запущенное ядро. Если выбор ещё старый,
выбрать **wdv3-timm (.venv Python 3.12)** и выполнить Restart Kernel.

Повторная установка в эту среду:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip check --python .venv/Scripts/python.exe
.\.venv\Scripts\python.exe audit/run_checks.py
```
