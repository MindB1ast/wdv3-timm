"""Pinned hand-labelled dataset: notebook parity, category accuracy, stage timing and cache."""
import contextlib
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WORK=ROOT/'temp/readme-benchmark'
DATA=WORK/'dataset'
OUT=ROOT/'docs/benchmarks/2026-11-05-test-yolo-wd'
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',MPLBACKEND='Agg',
                  YOLO_CONFIG_DIR=str(WORK/'yolo'),MPLCONFIGDIR=str(WORK/'matplotlib'))


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
                               default=lambda item:item.item()),encoding='utf-8')
    os.replace(temp,path)


def metric(truth,predicted):
    tp=len(truth & predicted);fp=len(predicted-truth);fn=len(truth-predicted)
    p=tp/(tp+fp+1e-8);r=tp/(tp+fn+1e-8)
    return dict(precision=p,recall=r,f1=2*p*r/(p+r+1e-8),TP=tp,FP=fp,FN=fn)


def aggregate(rows,method,known=True):
    values=[row['metrics']['known' if known else 'all'][method] for row in rows]
    p=statistics.mean(v['precision'] for v in values);r=statistics.mean(v['recall'] for v in values)
    tp=sum(v['TP'] for v in values);fp=sum(v['FP'] for v in values);fn=sum(v['FN'] for v in values)
    return dict(n=len(values),precision=p,recall=r,f1_mean_per_image=statistics.mean(v['f1'] for v in values),
                f1_project_harmonic_mean_PR=2*p*r/(p+r+1e-8),TP=tp,FP=fp,FN=fn,
                micro=metric(set(range(tp+fn)),set(range(tp))|set(range(tp+fn,tp+fn+fp))))


def names(record):return set(record['general'])|set(record['character'])


def redundant_names(tags):
    redundant=set()
    for tag in tags:
        parts=tag.split('_')
        for start in range(len(parts)):
            for end in range(start+1,len(parts)+1):
                sub='_'.join(parts[start:end])
                if sub!=tag and sub in tags:redundant.add(sub)
    return sorted(redundant)


def main():
    global OUT
    if '--smoke' in sys.argv:OUT=WORK/'smoke-results'
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from Scripts import ScriptOptions
    batch=importlib.import_module('Scripts.main')
    tagging=importlib.import_module('Scripts.tagging')
    from Scripts.region_tagging import RegionTagger
    from Scripts.tagger_resources import load_tagger
    from Scripts.model import load_yolo_model
    from Scripts.detection import load_detectors_config
    from Scripts.tag_names import parse_tags
    from Scripts.tagger_preprocessing import transform_metadata
    from Scripts.workflow_api import runtime_module,normalized_image
    from Scripts.artifact_store import ArtifactStore
    from Scripts.inference_plan import model_identity,pixel_identity

    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=(json.loads((WORK/'remote-manifest.json').read_text(encoding='utf-8'))
              if (WORK/'remote-manifest.json').is_file() else
              dict(repo_id='MindB1ast/Test_Yolo_WD',revision='f1b1f4e62bfdcff1a52a674281d7013964b35677'))
    files=sorted(p for p in DATA.rglob('*') if p.is_file() and '.cache' not in p.parts and '.wdv3' not in p.parts
                 and p.suffix.lower() in ('.png','.jpg','.jpeg','.txt'))
    before={p.relative_to(DATA).as_posix():digest(p) for p in files}
    images=[p for p in files if p.suffix.lower()!='.txt']
    assert images and all(p.with_suffix('.txt').is_file() for p in images)
    paired={p.with_suffix('.txt') for p in images}
    orphan=[p.relative_to(DATA).as_posix() for p in files if p.suffix.lower()=='.txt' and p not in paired]
    if '--smoke' in sys.argv:
        images=[next(p for p in images if p.parent.name==c) for c in ('NoDetection','Normal','Wide_shot')]
    refs={p.relative_to(DATA).as_posix():set(parse_tags(p.with_suffix('.txt').read_text(encoding='utf-8-sig'))) for p in images}
    torch.set_num_threads(min(8,psutil.cpu_count(logical=False) or 1))
    torch.manual_seed(20261105)
    options=ScriptOptions(ImageFolder=DATA,model='big',gen_threshold=.35,char_threshold=.75,batch=2,recursive=True,
         save_txt=False,append_txt=False,output_file=None,model_folder=ROOT/'models/taggers',
         yolo_model_dir=ROOT/'models/yolo',detectors_config=ROOT/'detectors.json')
    configs=load_detectors_config(options.detectors_config)
    assert len(configs)==2
    dump(OUT/'detectors.json',json.loads((ROOT/'detectors.json').read_text()))
    meta=dict(schema_version=1,started_utc=datetime.now(timezone.utc).isoformat(),repo=snapshot['repo_id'],revision=snapshot['revision'],
         images=len(images),categories={c:sum(p.parent.name==c for p in images) for c in ('NoDetection','Normal','Wide_shot')},
         orphan_labels=orphan,options={k:str(v) if isinstance(v,Path) else v for k,v in asdict(options).items()},
         environment=dict(python=sys.version,torch=torch.__version__,device='cpu',threads=torch.get_num_threads(),
                          cpu_count=psutil.cpu_count(),ram_total=psutil.virtual_memory().total,
                          ram_available_start=psutil.virtual_memory().available),
         f1_note='Main historical-style F1 is mean per-image F1. Current project harmonic(mean P, mean R) and micro metrics also provided.',
         reference_hashes=before)
    dump(OUT/'manifest.json',meta)
    print('DATASET',meta['categories'],'orphan TXT',len(orphan),flush=True)
    start=time.perf_counter()
    model,labels,transform,folder=load_tagger('big',ROOT/'models/taggers',allow_download=False)
    wd_load=time.perf_counter()-start
    yolo={};yolo_load={}
    for config in configs:
        start=time.perf_counter()
        yolo[config.model_path]=load_yolo_model(config.model_path,ROOT/'models/yolo',allow_download=False).to('cpu')
        yolo_load[config.name]=time.perf_counter()-start
    model=model.cpu().eval()
    known=set(parse_tags(labels.names))
    resources=batch._BatchResources(model,transform,labels,configs,yolo,torch.device('cpu'))
    full_resources=batch._BatchResources(model,transform,labels,[],{},torch.device('cpu'))
    meta.update(model_profile=transform_metadata(transform),wd_load_seconds=wd_load,yolo_load_seconds=yolo_load,
                model_files={str(folder.relative_to(ROOT))+'/model.safetensors':digest(folder/'model.safetensors'),
                             **{str((ROOT/'models/yolo'/c.model_path).relative_to(ROOT)):digest(ROOT/'models/yolo'/c.model_path) for c in configs}})
    dump(OUT/'manifest.json',meta)
    print('LOADED seconds WD',round(wd_load,2),'YOLO',yolo_load,flush=True)

    # Prepare the exact ordinary-cache identities once. Scores are ingested from the notebook run, not recomputed.
    backend=runtime_module();package=backend.__package__
    native=importlib.import_module(package+'.comfy_nodes')
    native_evidence=importlib.import_module(package+'.comfy_evidence')
    native_cache=importlib.import_module(package+'.comfy_automatic_cache')
    lazy=native.DeferredWDRuntime(labels,transform,torch.device('cpu'),'SmilingWolf/wd-eva02-large-tagger-v3',folder,None)
    wd_identity=lazy.persistent_identity
    branches=[]
    for c in configs:
        path=(ROOT/'models/yolo'/c.model_path).resolve()
        spec=dict(local_path=str(path),files={'weights':str(path)},adapter='legacy-yolo-v1',compute={'device':c.device})
        pin=model_identity(spec)
        spec.pop('files');spec['identity']=pin
        branches.append(dict(kind='yolo',name=c.name,confidence=c.confidence,classes=c.classes,
                             rules={k:v for k,v in asdict(c).items() if k not in ('name','model_path','confidence','classes')},model=spec))
    inference=dict(schema_version=1,wd=dict(model_id=lazy.model_id,identity=wd_identity),sources=branches)
    store=ArtifactStore(DATA)
    raw_full={};raw_crops={};crop_scores={};current={}
    state=dict(mode='',stage='',source='',capture_seconds=0,times={},counts={})
    def clean(row):return {cat:{tag:float(v) for tag,v in row[cat].items()} for cat in ('general','character','ratings')}
    def full_hook(module,inputs,output):
        if state['mode']=='regional' and state['stage']=='full':
            from Scripts.wdv3_timm import get_tags
            t=time.perf_counter();value=get_tags(output[0].detach().sigmoid().cpu(),labels,-1,-1)
            raw_full.clear();raw_full.update(dict(ratings=value[2],character=value[3],general=value[4]))
            state['capture_seconds']+=time.perf_counter()-t
    original_full=tagging.tag_image;original_extract=tagging.extract_regions_with_detector
    original_crop=tagging._tag_detector_regions;original_scores=RegionTagger.score_batch
    def timed_full(*args,**kwargs):
        state['stage']='full';t=time.perf_counter();overhead=state['capture_seconds']
        try:return original_full(*args,**kwargs)
        finally:state['times']['wd_full']=time.perf_counter()-t-(state['capture_seconds']-overhead);state['stage']=''
    def timed_extract(path,config,detector):
        t=time.perf_counter();result=original_extract(path,config,detector)
        state['times']['detect_'+config.name]=time.perf_counter()-t
        state['counts'][config.name]=len(result)
        return result
    def timed_crop(inputs,config,rt):
        state['stage']='crop';state['source']=config.name;t=time.perf_counter();overhead=state['capture_seconds']
        try:return original_crop(inputs,config,rt)
        finally:
            state['times']['wd_crops_'+config.name]=time.perf_counter()-t-(state['capture_seconds']-overhead)
            state['stage']=''
    def capture_scores(tagger,inputs):
        result=original_scores(tagger,inputs)
        if state['mode']=='regional':
            t=time.perf_counter();raw_crops.setdefault(state['source'],[]).extend(clean(r) for r in result)
            state['capture_seconds']+=time.perf_counter()-t
        return result
    class Ingest(backend.PersistentCalculator):
        def _score(self,action,deps):return clean(raw_full)
        def _wd_scores(self,inputs):return [crop_scores[pixel_identity(image)] for image in inputs]
    def saved_geometry(image,source):
        return {'regions':[dict(region_id=i,bbox=row['bbox']) for i,row in enumerate(current['detectors'][source['name']])]}

    records=[];cache_entries=[];ingest_seconds=0;cache_errors=[]
    hook=model.register_forward_hook(full_hook)
    try:
        with patch.object(tagging,'tag_image',timed_full),patch.object(tagging,'extract_regions_with_detector',timed_extract), \
             patch.object(tagging,'_tag_detector_regions',timed_crop),patch.object(RegionTagger,'score_batch',capture_scores):
            for index,path in enumerate(images):
                key=path.relative_to(DATA).as_posix();truth=refs[key]
                state.update(mode='regional',times={},counts={},capture_seconds=0);raw_full.clear();raw_crops.clear()
                start=time.perf_counter();detail,merged=batch._process_image(path,resources,options)
                regional_wall=time.perf_counter()-start;regional_times=dict(state['times']);counts=dict(state['counts'])
                overhead=state['capture_seconds'];current=detail
                predicted=names(merged['merged_tags']);full_pred=names(detail['full_image'])
                state.update(mode='full_only',times={},counts={},capture_seconds=0)
                start=time.perf_counter();_,control=batch._process_image(path,full_resources,options)
                full_wall=time.perf_counter()-start
                assert names(control['merged_tags'])==full_pred,('full-only parity',key)
                added=predicted-full_pred;removed=full_pred-predicted
                row=dict(image=key,category=path.parent.name,truth=sorted(truth),full=sorted(full_pred),regional=sorted(predicted),
                    unknown_reference=sorted(truth-known),textual_redundancy_in_reference=redundant_names(truth),
                    gt_no_humans='no_humans' in truth,
                    metrics={scope:{name:metric(truth & known if scope=='known' else truth,tags)
                                   for name,tags in [('full',full_pred),('regional',predicted)]} for scope in ('known','all')},
                    changes=dict(correct_added=sorted(added & truth),incorrect_added=sorted(added-truth),
                                 correct_removed=sorted(removed-truth),harmful_removed=sorted(removed & truth)),
                    detections=counts,timing=dict(full_only_seconds=full_wall,regional_seconds=regional_wall-overhead,
                                                instrumented_wall_seconds=regional_wall,capture_seconds=overhead,stages=regional_times),
                    ram_available=psutil.virtual_memory().available,rss=psutil.Process().memory_info().rss)
                # No new model call: publish raw full/crop scores observed above as ordinary stage artifacts.
                t=time.perf_counter()
                try:
                    with normalized_image(path) as image:
                        crop_scores.clear()
                        for name,regions in detail['detectors'].items():
                            assert len(regions)==len(raw_crops.get(name,[]))
                            for region,raw in zip(regions,raw_crops.get(name,[])):
                                with image.crop(tuple(region['bbox'])) as crop:crop_scores[pixel_identity(crop)]=raw
                        calculator=Ingest(store,inference,hooks={'yolo':saved_geometry})
                        try:
                            raw,report=calculator.compute_image(image,relative_path=key)
                            assert report['complete'],report
                            cache_entries.append(dict(relative_path=key,artifact_id=report['evidence_ref']))
                        finally:calculator.close()
                except Exception as exc:
                    cache_errors.append(dict(image=key,error=str(exc)))
                row['cache_ingest_seconds']=time.perf_counter()-t;ingest_seconds+=row['cache_ingest_seconds']
                records.append(row);dump(OUT/'per-image.json',records)
                print('PROGRESS',index+1,'/',len(images),key,'full',round(full_wall,3),'regional',round(row['timing']['regional_seconds'],3),
                      'regions',counts,'cache_errors',len(cache_errors),flush=True)
    finally:hook.remove()
    dump(OUT/'cache-ingest-errors.json',cache_errors)
    if cache_entries:dump(OUT/'cache-manifest.json',store.write_manifest(cache_entries,request={'benchmark_revision':snapshot['revision']}))

    # The real ordinary graph reads every compatible stage without loading a network or invoking inference.
    cache_rows=[]
    if '--check-cache' in sys.argv and not cache_errors:
        blocked=[]
        def forbidden(*args,**kwargs):blocked.append('called');raise AssertionError('cache pass must not load/infer')
        native_resources=importlib.import_module(package+'.Scripts.tagger_resources')
        native_models=importlib.import_module(package+'.Scripts.model')
        with patch.object(native_resources,'load_network',forbidden),patch.object(native_models,'load_yolo_model',forbidden), \
             patch.object(RegionTagger,'score_batch',forbidden):
            for start in range(0,len(images),options.batch):
                chunk=images[start:start+options.batch]
                with contextlib.ExitStack() as stack:
                    frames=[stack.enter_context(normalized_image(path)) for path in chunk]
                    opts=dict(cache_mode='reuse_only',cache_dataset_root=str(DATA),general_threshold=.35,character_threshold=.75,
                              region_batch_size=options.region_batch_size)
                    t=time.perf_counter();raw=native_cache.cached_evidence(frames,lazy,options=opts,detectors=(configs,{}),image_paths=chunk)
                    read=time.perf_counter()-t
                    t=time.perf_counter();result=native_evidence.apply_filters(raw,opts)
                    filtering=time.perf_counter()-t
                    for offset,frame in enumerate(result.data()['frames']):
                        original=records[start+offset]
                        parity=names(frame['merged_tags'])==set(original['regional'])
                        cache_rows.append(dict(image=original['image'],read_seconds=read/len(chunk),filter_seconds=filtering/len(chunk),parity=parity))
                print('CACHE',min(start+options.batch,len(images)),'/',len(images),round(read+filtering,3),'seconds',flush=True)
        assert not blocked and lazy._network is None
    dump(OUT/'cache-timing.json',cache_rows)

    from collections import Counter,defaultdict
    summary={}
    rng=np.random.default_rng(20261105)
    for category in ['All','NoDetection','Normal','Wide_shot']:
        subset=records if category=='All' else [r for r in records if r['category']==category]
        delta=np.array([r['metrics']['known']['regional']['f1']-r['metrics']['known']['full']['f1'] for r in subset])
        resampled=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
        summary[category]=dict(n=len(subset),known={m:aggregate(subset,m) for m in ('full','regional')},
            all_reference={m:aggregate(subset,m,False) for m in ('full','regional')},
            delta_f1_mean=float(delta.mean()),paired_bootstrap_95ci=[float(v) for v in np.quantile(resampled,[.025,.975])],
            improved=sum(d>1e-8 for d in delta),worsened=sum(d< -1e-8 for d in delta),unchanged=sum(abs(d)<=1e-8 for d in delta),
            full_seconds=sum(r['timing']['full_only_seconds'] for r in subset),regional_seconds=sum(r['timing']['regional_seconds'] for r in subset),
            detections={c.name:sum(r['detections'].get(c.name,0) for r in subset) for c in configs},
            images_with_detections=sum(any(r['detections'].values()) for r in subset),
            gt_no_humans=sum(r['gt_no_humans'] for r in subset),
            unknown_reference_count=sum(len(r['unknown_reference']) for r in subset),reference_count=sum(len(r['truth']) for r in subset),
            top_incorrect_additions=Counter(tag for r in subset for tag in r['changes']['incorrect_added']).most_common(15),
            top_correct_additions=Counter(tag for r in subset for tag in r['changes']['correct_added']).most_common(15),
            top_harmful_removals=Counter(tag for r in subset for tag in r['changes']['harmful_removed']).most_common(15),
            stage_seconds={name:sum(r['timing']['stages'].get(name,0) for r in subset) for name in set(k for r in subset for k in r['timing']['stages'])})
    duplicates=defaultdict(list)
    for path in images:duplicates[before[path.relative_to(DATA).as_posix()]].append(path.relative_to(DATA).as_posix())
    integrity=all(digest(DATA/key)==value for key,value in before.items());assert integrity
    dump(OUT/'summary.json',dict(categories=summary,reference_unchanged=integrity,
         duplicate_images=[v for v in duplicates.values() if len(v)>1],
         unknown_reference_tags=Counter(tag for r in records for tag in r['unknown_reference']).most_common(),
         reference_images_with_textual_redundancy=sum(bool(r['textual_redundancy_in_reference']) for r in records),
         cache=dict(images=len(cache_rows),seconds=sum(r['read_seconds']+r['filter_seconds'] for r in cache_rows),
                    read_seconds=sum(r['read_seconds'] for r in cache_rows),filter_seconds=sum(r['filter_seconds'] for r in cache_rows),
                    parity_mismatches=[r['image'] for r in cache_rows if not r['parity']],weight_loads=0,inference_calls=0,
                    ingest_seconds=ingest_seconds,errors=cache_errors)))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
