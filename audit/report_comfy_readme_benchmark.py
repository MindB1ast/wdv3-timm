"""Read actual ComfyUI job histories and saved node outputs; no inference."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
OUT=BASE/'comfyui-pipeline'


def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    OUT=parser.parse_args().output_dir.resolve()
    submitted=json.loads((OUT/'submitted.json').read_text(encoding='utf-8'))
    def history(run):
        with urlopen('http://127.0.0.1:8188/history/'+run['prompt_id'],timeout=30) as response:
            data=json.load(response).get(run['prompt_id'])
        if not data:return dict(run,status='pending')
        status=data['status'];messages=status.get('messages',[])
        timestamps={name:item.get('timestamp') for name,item in messages}
        start=timestamps.get('execution_start');end=timestamps.get('execution_success')
        return dict(run,status=status.get('status_str'),completed=status.get('completed'),
                    start_ms=start,end_ms=end,seconds=(end-start)/1000 if start is not None and end is not None else None,
                    node_classes={k:v['class_type'] for k,v in data['prompt'][2].items()},
                    errors=[item for name,item in messages if name in ('execution_error','execution_interrupted')])
    with ThreadPoolExecutor(max_workers=4) as pool:histories=list(pool.map(history,submitted['runs']))
    (OUT/'history-summary.json').write_text(json.dumps(histories,ensure_ascii=False,indent=2),encoding='utf-8')
    counts={status:sum(r['status']==status for r in histories) for status in sorted({r['status'] for r in histories})}
    files=len(list((OUT/'tags').glob('*.txt')))
    print(json.dumps(dict(status_counts=counts,txt_files=files),ensure_ascii=False),flush=True)
    if not all(r['status']=='success' and r.get('completed') for r in histories):return
    from Scripts.tag_names import parse_tags
    from audit.run_readme_benchmark import metric,dump
    originals=json.loads((BASE/'per-image.json').read_text(encoding='utf-8'))
    expected=json.loads((BASE/'copy1-region-thresholds/per-image.json').read_text(encoding='utf-8'))
    results=[]
    for index,original in enumerate(originals):
        predicted=set(parse_tags((OUT/'tags'/f'image_{index:03d}.txt').read_text(encoding='utf-8')))
        truth=set(original['truth']);known_truth=truth-set(original['unknown_reference'])
        results.append(dict(image=original['image'],category=original['category'],tags=sorted(predicted),
                            matches_previous=predicted==set(expected[index]['tags']),
                            metrics=dict(known=metric(known_truth,predicted),all=metric(truth,predicted))))
    summary={}
    for category in ('All','NoDetection','Normal','Wide_shot'):
        rows=results if category=='All' else [r for r in results if r['category']==category]
        summary[category]=dict(n=len(rows),**{scope:{k:statistics.mean(r['metrics'][scope][k] for r in rows)
                                               for k in ('precision','recall','f1')} for scope in ('known','all')})
    seconds=sum(r['seconds'] for r in histories)
    wall=(max(r['end_ms'] for r in histories)-min(r['start_ms'] for r in histories))/1000
    node_types=sorted({n for r in histories for n in r['node_classes'].values()})
    assert not any('ModelLoader' in n or 'CollectEvidence' in n or 'ComputeMissing' in n for n in node_types)
    output=dict(categories=summary,graphs_completed=len(histories),graph_execution_seconds=seconds,
                first_start_to_last_finish_seconds=wall,images_matching_previous=sum(r['matches_previous'] for r in results),
                mismatches=[r['image'] for r in results if not r['matches_previous']],
                node_classes=node_types,inference_calls=0,
                config='copy1-region-thresholds/effective-detectors.json',
                note='Actual local ComfyUI server queue, cached evidence, region WD general thresholds 0.3; YOLO confidence stays 0.4.')
    dump(OUT/'summary.json',output);dump(OUT/'per-image.json',results)
    lines=['# Результаты пайплайна ComfyUI','',
           'Фактическое исполнение на локальном сервере ComfyUI `127.0.0.1:8188`, 108 изображений, 54 графа по 2 изображения. Граф: Calculation Store → Load Evidence → Filter Tag Sets → Merge Tag Sets → Save Result; настройки подключены через Tagging Settings и Load YOLO Detectors. Все запуски завершены успешно.','',
           'Конфиг: WD full 0.35/0.75; WD person 0.3/0.8; WD face 0.3/0.85; confidence YOLO 0.4. SAM и удаление избыточных тегов выключены. Применены только пороги тегов областей из Copy1, как в предыдущем пересчёте. Граф загружает готовое evidence из `.wdv3`; загрузки моделей и инференса в нём нет.','',
           '| Категория | N | Precision, % | Recall, % | F1, % |','|---|---:|---:|---:|---:|']
    for category,d in summary.items():
        v=d['known'];lines.append(f"| {category} | {d['n']} | {v['precision']*100:.2f} | {v['recall']*100:.2f} | {v['f1']*100:.2f} |")
    lines+=['','F1 — среднее по изображениям; ручные теги ограничены словарём WD. Метрики по всем ручным тегам также сохранены в `summary.json`.', '',
            f'Суммарное время исполнения графов по серверным execution_start/execution_success: **{seconds:.2f} с**. От старта первого графа до завершения последнего, включая интервалы отправки через MCP и ожидание очереди: **{wall:.2f} с**. Время не является замером повторного инференса.', '',
            f"Совпадение итоговых тегов с предыдущим пересчётом из кэша: **{output['images_matching_previous']}/108**. Ручные TXT не перезаписывались; результаты нод сохранены отдельно в `tags/`, соответствие исходным изображениям — в `image-map.json`.",'',
            'API-графы: `workflows/`; ID выполненных заданий: `submitted.json`; статусы и серверные времена: `history-summary.json`; метрики: `summary.json`; теги по изображениям: `per-image.json`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
