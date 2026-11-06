"""Build bounded API graphs for replaying benchmark evidence on the live ComfyUI."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
OUT=BASE/'comfyui-pipeline'


def main():
    rows=json.loads((BASE/'per-image.json').read_text(encoding='utf-8'))
    manifest=json.loads((BASE/'cache-manifest.json').read_text(encoding='utf-8'))
    settings=json.loads((BASE/'copy1-region-thresholds/effective-detectors.json').read_text(encoding='utf-8'))
    for config in settings:config['model_path']=str((ROOT/'models/yolo'/config['model_path']).resolve())
    detectors=json.dumps(settings,ensure_ascii=False)
    options=dict(general_threshold=.35,character_threshold=.75,image_batch_size=2,region_batch_size=8,
                 replace_underscores=False,escape_parentheses=False,remove_tags='',prefix_tags='',suffix_tags='',
                 remove_tags_regex='',tag_separator=', ',sam_batch_size=1,cache_mode='reuse_only',
                 cache_dataset_root=str(ROOT/'temp/readme-benchmark/dataset'),cache_refresh='')
    jobs=[]
    for start in range(0,len(rows),2):
        graph={
            '1':dict(class_type='WDV3CalculationStore',inputs=dict(dataset_root=options['cache_dataset_root'],store_path='',mode='reuse_only',refresh='')),
            '2':dict(class_type='WDV3LoadEvidence',inputs=dict(calculation_store=['1',0],artifact_ids='',run_id=manifest['run_id'],start=start,limit=2)),
            '3':dict(class_type='WDV3Options',inputs=options),
            '4':dict(class_type='WDV3DetectorLoader',inputs=dict(detectors_json=detectors)),
            '5':dict(class_type='WDV3FilterTagSets',inputs=dict(evidence=['2',0],options=['3',0],detectors=['4',0])),
            '6':dict(class_type='WDV3MergeTagSets',inputs=dict(tag_sets=['5',0]))}
        for offset in range(min(2,len(rows)-start)):
            graph[str(7+offset)]=dict(class_type='WDV3SaveResult',inputs=dict(results_json=['6',2],tags=['6',1],
                output_directory=str(OUT/'tags'),filename=f'image_{start+offset:03d}',result_index=offset,
                save_txt=True,save_json=False,overwrite=False))
        path=OUT/'workflows'/f'batch_{start:03d}.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
        jobs.append(dict(workflow_path=str(path),start=start,count=min(2,len(rows)-start)))
    (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    (OUT/'image-map.json').write_text(json.dumps([dict(image=r['image'],file=f'image_{i:03d}.txt') for i,r in enumerate(rows)],ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(jobs=len(jobs),first=jobs[0]),indent=2))


if __name__=='__main__':main()
