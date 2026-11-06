"""Report completed inference measurements without running any more models."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'


def main():
    rows=json.loads((OUT/'per-image.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert len(rows)==manifest['images']==108
    lines=['# Test_Yolo_WD: результаты прогона','',
           f"108 изображений из [MindB1ast/Test_Yolo_WD](https://huggingface.co/datasets/{manifest['repo']}/tree/{manifest['revision']}). Ревизия `{manifest['revision']}`.",'',
           'Запуск: функции notebook (`Scripts.main._process_image`) в отдельном Python-скрипте, CPU, 8 потоков. Сам notebook целиком и граф на сервере ComfyUI не исполнялись. Это время backend, без очереди и интерфейса ComfyUI.','',
           'WD: `big` / `SmilingWolf/wd-eva02-large-tagger-v3`, полный кадр 0.35/0.75. YOLO: `person_yolov8s-seg.pt`, `face_yolov9c.pt`, confidence 0.4, classes [0]. Пороги областей person 0.45/0.8, face 0.35/0.85. Остальные правила из текущего `detectors.json` без изменений; копия находится рядом с отчётом. Батч областей 8, файловый батч 2. SAM и удаление избыточных тегов выключены.','',
           'Основные метрики ограничивают ручные теги словарём WD, как `filter_tags=True` в notebook. Пробелы/подчёркивания нормализованы. Precision, Recall и F1 усреднены по изображениям; для текущей формулы проекта отдельно сохранён harmonic(mean P, mean R).','',
           '| Категория | N | P полного, % | R полного, % | F1 полного, % | P с регионами, % | R с регионами, % | F1 с регионами, % | Δ F1, п.п. |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary={}
    for category in ('All','NoDetection','Normal','Wide_shot'):
        subset=rows if category=='All' else [r for r in rows if r['category']==category]
        data=dict(n=len(subset),known={},all_reference={})
        for scope,key in [('known','known'),('all_reference','all')]:
            for mode in ('full','regional'):
                values={k:statistics.mean(r['metrics'][key][mode][k] for r in subset) for k in ('precision','recall','f1')}
                p,r=values['precision'],values['recall'];values['f1_project_harmonic_mean_PR']=2*p*r/(p+r+1e-8)
                data[scope][mode]=values
        data.update(full_seconds=sum(r['timing']['full_only_seconds'] for r in subset),
                    regional_seconds=sum(r['timing']['regional_seconds'] for r in subset))
        summary[category]=data
        a,b=data['known']['full'],data['known']['regional']
        values=[a['precision'],a['recall'],a['f1'],b['precision'],b['recall'],b['f1'],b['f1']-a['f1']]
        lines.append('| '+category+' | '+str(data['n'])+' | '+' | '.join(f'{v*100:.2f}' for v in values)+' |')
    lines+=['','| Категория | Полный кадр, с | С регионами, с | Полный, с/изобр. | С регионами, с/изобр. |',
            '|---|---:|---:|---:|---:|']
    for category,d in summary.items():
        a,b,n=d['full_seconds'],d['regional_seconds'],d['n']
        lines.append(f'| {category} | {a:.2f} | {b:.2f} | {a/n:.2f} | {b/n:.2f} |')
    lines+=['',f"Загрузка моделей отдельно: WD {manifest['wd_load_seconds']:.2f} с, оба YOLO {sum(manifest['yolo_load_seconds'].values()):.2f} с. Таблица исключает загрузку моделей, запись кэша и дополнительное копирование сырых scores. Первый инференс включён; контроль полного кадра выполнялся после регионального прохода. Открытый ComfyUI не останавливался, поэтому время зависит также от давления на RAM.",'',
            '| Категория | F1 полного по всем ручным тегам, % | F1 с регионами по всем ручным тегам, % |','|---|---:|---:|']
    for category,d in summary.items():
        lines.append(f"| {category} | {d['all_reference']['full']['f1']*100:.2f} | {d['all_reference']['regional']['f1']*100:.2f} |")
    unchanged=all(__import__('hashlib').sha256((ROOT/'temp/readme-benchmark/dataset'/p).read_bytes()).hexdigest()==v for p,v in manifest['reference_hashes'].items())
    assert unchanged
    lines+=['', 'Исходные изображения и TXT не изменены: SHA-256 до/после совпадают. Четыре TXT без изображения исключены. Набор из 108 изображений отличается от исторической таблицы README на 12 изображений.','',
            'Подробные результаты каждого изображения — `per-image.json`; настройки, версии и SHA-256 — `manifest.json`; конфиг детекторов — `detectors.json`. Прогон: `.venv/Scripts/python.exe audit/run_readme_benchmark.py`; этот отчёт без инференса: `.venv/Scripts/python.exe audit/report_readme_benchmark.py`. Датасет находится в `temp/readme-benchmark/dataset`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(dict(categories=summary,reference_unchanged=unchanged),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
