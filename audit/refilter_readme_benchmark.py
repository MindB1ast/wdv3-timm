"""Threshold-only comparison on saved raw evidence; never performs inference."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
OUT=BASE/'copy1-region-thresholds'


def main():
    from Scripts.artifact_store import ArtifactStore
    from Scripts.configs import DetectorConfig
    from Scripts.preprocess import merge_tags_from_regions
    from Scripts.tag_utils import apply_tag_rules,restrict_allowed_tags,format_result,enforce_owners
    from Scripts.tag_names import parse_tags
    from audit.run_readme_benchmark import metric,names,dump
    store=ArtifactStore(ROOT/'temp/readme-benchmark/dataset')
    manifest=json.loads((BASE/'cache-manifest.json').read_text(encoding='utf-8'))
    entries={r['relative_path']:r['artifact_id'] for r in store.iter_entries(manifest['run_id'])}
    old_json=json.loads((BASE/'detectors.json').read_text(encoding='utf-8'))
    copy_json=json.loads((ROOT/'detectors-Copy1.json').read_text(encoding='utf-8'))
    effective=deepcopy(old_json)
    for old in effective:
        updated=next(c for c in copy_json if c['name']==old['name'])
        for field in ('region_gen_threshold','region_char_threshold'):old[field]=updated[field]
    old_configs=[DetectorConfig.from_dict(row) for row in old_json]
    configs=[DetectorConfig.from_dict(row) for row in effective]
    dump(OUT/'effective-detectors.json',effective)
    dump(OUT/'input-detectors-Copy1.json',copy_json)
    originals=json.loads((BASE/'per-image.json').read_text(encoding='utf-8'))
    def filtered(frame,settings):
        characters=list(frame['full_image']['character'])
        def record(raw,config=None):
            gen=.35 if config is None or config.region_gen_threshold is None else config.region_gen_threshold
            char=.75 if config is None or config.region_char_threshold is None else config.region_char_threshold
            row=dict(general={t:s for t,s in raw['general'].items() if s>gen},
                     character={t:s for t,s in raw['character'].items() if s>char},ratings=dict(raw['ratings']))
            if config is not None:
                row=restrict_allowed_tags(apply_tag_rules(row,remove=config.remove_tags_from_region,
                                            add=config.add_tags_to_region,character_names=characters),config)
            return format_result(row)
        detail=dict(image_path=frame.get('image_path',''),full_image=record(frame['full_image']),
                    detectors={c.name:[dict(record(r,c),region_id=r['region_id'],bbox=r['bbox'])
                                       for r in frame['detectors'][c.name]] for c in settings})
        return enforce_owners(apply_tag_rules(merge_tags_from_regions(detail,settings),character_names=characters),settings)
    rows=[];start=time.perf_counter();filter_seconds=0;read_seconds=0
    for index,original in enumerate(originals):
        t=time.perf_counter()
        saved=store.load(entries[original['image']])['payload']['evidence']
        frame=saved['frames'][0]
        read_seconds+=time.perf_counter()-t
        assert names(filtered(frame,old_configs))==set(original['regional']),original['image']
        t=time.perf_counter();predicted=names(filtered(frame,configs));elapsed=time.perf_counter()-t
        filter_seconds+=elapsed
        known=set(parse_tags(t for tags in saved['model']['categories'].values() for t in tags))
        truth=set(original['truth'])
        rows.append(dict(image=original['image'],category=original['category'],tags=sorted(predicted),
                         previous=original['metrics'],filter_seconds=elapsed,
                         metrics={scope:metric(truth & known if scope=='known' else truth,predicted) for scope in ('known','all')}))
        if (index+1)%20==0:print('REFILTER',index+1,'/',len(originals),flush=True)
    total=time.perf_counter()-start
    summary={}
    for category in ('All','NoDetection','Normal','Wide_shot'):
        subset=rows if category=='All' else [r for r in rows if r['category']==category]
        summary[category]=dict(n=len(subset),**{scope:{k:statistics.mean(r['metrics'][scope][k] for r in subset)
                                    for k in ('precision','recall','f1')} for scope in ('known','all')},
            previous_f1=statistics.mean(r['previous']['known']['regional']['f1'] for r in subset))
    result=dict(categories=summary,seconds=total,read_seconds=read_seconds,filter_seconds=filter_seconds,
                inference_calls=0,weight_loads=0,baseline_parity_images=len(rows),
                note='Only region WD thresholds from Copy1 applied. YOLO confidence retained at cached 0.4, as requested.')
    dump(OUT/'per-image.json',rows);dump(OUT/'summary.json',result)
    lines=['# Copy1: пороги тегов областей из кэша','',
           '108 изображений. Применены только `region_gen_threshold` и `region_char_threshold` из `detectors-Copy1.json`; confidence YOLO сохранён 0.4 по уточнению пользователя. Person WD general: 0.45 → 0.3; face WD general: 0.35 → 0.3. Полный кадр и пороги персонажей не изменялись.','',
           '| Категория | N | F1 до, % | F1 после, % | Precision после, % | Recall после, % |','|---|---:|---:|---:|---:|---:|']
    for category,d in summary.items():
        p=d['known'];lines.append(f"| {category} | {d['n']} | {d['previous_f1']*100:.2f} | {p['f1']*100:.2f} | {p['precision']*100:.2f} | {p['recall']*100:.2f} |")
    lines+=['',f'Чтение кэша: {read_seconds:.2f} с; применение новых порогов и слияние: {filter_seconds:.2f} с. Весь цикл, включая проверку прежних результатов: {total:.2f} с. Загрузок весов: 0; инференса: 0. Проверено совпадение прежних результатов на всех 108 изображениях.', '',
            'F1 — среднее по изображениям, эталон ограничен словарём WD, как в исходном сравнении. Метрики по всем ручным тегам также сохранены в `summary.json`. Это пересчёт через функции notebook, без построения подробных диагностик ComfyUI. Конфиг этого сравнения — `effective-detectors.json`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
