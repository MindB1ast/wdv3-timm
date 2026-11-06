"""Profile one cached frame through the same backend stages as the ComfyUI graph."""
import cProfile
import importlib
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COMFY=Path(r'C:\Users\annak\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI')
BASE=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
OUT=BASE/'comfyui-pipeline'
sys.path[:0]=[str(ROOT),str(COMFY)]
sys.argv=['profile_comfy_readme','--cpu']
os.environ.update(HF_HUB_OFFLINE='1',MPLBACKEND='Agg',YOLO_CONFIG_DIR=str(ROOT/'temp/readme-benchmark/yolo'))


def main():
    from Scripts import workflow_api as api
    from Scripts.configs import DetectorConfig
    from Scripts.artifact_store import ArtifactStore
    backend=api.runtime_module()
    evidence=importlib.import_module(backend.__package__+'.comfy_evidence')
    export=importlib.import_module(backend.__package__+'.comfy_export')
    store=ArtifactStore(ROOT/'temp/readme-benchmark/dataset')
    manifest=json.loads((BASE/'cache-manifest.json').read_text(encoding='utf-8'))
    entry=next(r for r in store.iter_entries(manifest['run_id']) if r['relative_path']=='Wide_shot/0SxBXLP.png')
    config=json.loads((OUT/'workflows/batch_000.json').read_text(encoding='utf-8'))
    options=dict(config['3']['inputs'],remove_tags_regex=[])
    settings=[DetectorConfig.from_dict(c) for c in json.loads(config['4']['inputs']['detectors_json'])]
    profiler=cProfile.Profile();times={}
    def stage(name,fn,*args,**kwargs):
        t=time.perf_counter();value=profiler.runcall(fn,*args,**kwargs);times[name]=time.perf_counter()-t
        return value
    raw=stage('load',api.load,dict(dataset_root=str(store.dataset_root),artifact_id=entry['artifact_id']))
    filtered=stage('filter_tag_sets',evidence.filter_tag_sets,raw,options,settings)
    merged=stage('merge_tag_sets',evidence.merge_tag_sets,filtered)
    tags,encoded=stage('outputs',evidence.outputs,merged)
    # Validate and save through the same Save Result implementation, in a separate scratch folder.
    stage('save_result',export.save_result,encoded,str(ROOT/'temp/readme-benchmark/profile-output'),'sample',
          tags=tags,result_index=0,save_txt=True,save_json=False,overwrite=True)
    frame=merged.data()['frames'][0]
    counts=dict(labels=raw.data()['model']['labels_count'],regions=sum(map(len,frame['detectors'].values())),
                decisions=len(frame['decisions']),accepted_tags=len(frame['merged_tags']['general'])+len(frame['merged_tags']['character']))
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
    (OUT/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    summary=dict(image=entry['relative_path'],python=sys.version,profiled_stage_seconds=times,counts=counts,
                 note='One cached frame; cProfile adds overhead. No models loaded and no inference called.')
    (OUT/'profile.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(stream.getvalue())


if __name__=='__main__':main()
