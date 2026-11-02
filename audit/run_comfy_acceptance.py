"""Local ComfyUI acceptance. Uploads disposable test inputs; does not change originals."""
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/verification/comfyui';OUT.mkdir(parents=True,exist_ok=True)
BASE='http://127.0.0.1:8188'
catalog=requests.get(BASE+'/object_info',timeout=20).json()
stats=requests.get(BASE+'/system_stats',timeout=20).json()
(OUT/'versions.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'TestTagPic').iterdir() if p.is_file()}
results=[]


def run_graph(image_name, number):
    graph={}
    def add(key,kind,**values):
        data=catalog[kind]['input']['required'];inputs={}
        for name,spec in data.items():
            if name in values:continue
            options=spec[1] if len(spec)>1 else {}
            if 'default' in options:inputs[name]=options['default']
            elif isinstance(spec[0],list):inputs[name]=spec[0][0]
        graph[str(key)]={'class_type':kind,'inputs':{**inputs,**values}}
    add(1,'LoadImage',image=image_name)
    add(2,'WDV3ModelLoader',model='vit',device='cpu',download_missing=False)
    add(3,'WDV3RegionRules',only_from_source='yellow eyes',allowed_tags='yellow eyes, eyelashes, smile',remove_from_full='eyelashes')
    add(4,'WDV3YOLODetector',model='face_yolov9c.pt',source_name='face',classes='0',
        custom_model_path=str(ROOT/'models/yolo/face_yolov9c.pt'),rules=['3',0],device='cpu',download_missing=False)
    add(5,'WDV3YOLODetector',model='person_yolov8s-seg.pt',source_name='person',classes='0',
        custom_model_path=str(ROOT/'models/yolo/person_yolov8s-seg.pt'),previous_detectors=['4',0],device='cpu',download_missing=False)
    add(6,'WDV3CollectEvidence',images=['1',0],wd_model=['2',0],detectors=['5',0])
    add(7,'WDV3Options',general_threshold=.35,character_threshold=.75)
    add(8,'WDV3ApplyFilters',evidence=['6',0],options=['7',0],detectors=['5',0])
    add(9,'WDV3Inspect',results=['8',0])
    add(10,'WDV3ExportReport',report=['9',0],output_directory=str(ROOT/'temp/comfy-full-reports'),prefix=f'image-{number}')
    add(11,'WDV3SaveResult',results_json=['8',2],tags=['8',1],output_directory=str(ROOT/'temp/comfy-full-results'),
        filename=f'image-{number}-{time.time_ns()}',result_index=0)
    response=requests.post(BASE+'/prompt',json={'prompt':graph},timeout=20)
    response.raise_for_status();submitted=response.json();assert not submitted.get('node_errors'),submitted
    pid=submitted['prompt_id'];print('Submitted',number,pid,flush=True)
    (ROOT/f'temp/comfy-acceptance-api-{number}.json').write_text(json.dumps(graph,indent=2),encoding='utf8')
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        history=requests.get(BASE+'/history/'+pid,timeout=45).json()
        if pid in history:break
        time.sleep(1)
    else:raise RuntimeError('ComfyUI execution timed out')
    job=history[pid]
    if job['status']['status_str']!='success':
        errors=[message[1] for message in job['status']['messages'] if message[0]=='execution_error']
        raise RuntimeError([(e.get('node_id'),e.get('exception_message')) for e in errors])
    detail=job['outputs']['9']['wdv3_results'][0]['detail']
    assert len(detail['full_image']['raw_scores']['general'])>1000
    assert len(detail['decisions'])>1000
    assert 'settings' in detail and detail['schema_version']==1
    assert all(tag in {'yellow_eyes','eyelashes','smile'} for r in detail['detectors']['face'] for tag in r['general'])
    if 'yellow_eyes' in detail['merged_tags']['general']:
        assert detail['merged_tags']['provenance']['yellow_eyes']['source'].startswith('face:')
    rejection={d['reason'] for d in detail['decisions'] if d['status']=='rejected'}
    assert 'below_threshold' in rejection
    # Keep structured records, not all browser pixels or transient descriptors.
    (OUT/f'acceptance-image-{number}.json').write_text(json.dumps({
        'prompt_id':pid,'image':image_name,'status':'success','model':detail['model']['model_id'],
        'raw_general_count':len(detail['full_image']['raw_scores']['general']),
        'regions':{n:len(rs) for n,rs in detail['detectors'].items()},'rejection_reasons':sorted(rejection),
        'helmet':[d for d in detail['decisions'] if d['tag']=='helmet'],
        'eyelashes':[d for d in detail['decisions'] if d['tag']=='eyelashes'],
        'merged_tags':detail['merged_tags']},ensure_ascii=False,indent=2),encoding='utf8')
    return dict(prompt_id=pid,image=image_name,status='success')


for number, path in enumerate(sorted(p for p in (ROOT/'TestTagPic').iterdir() if p.suffix.lower() in ('.png','.jpg')),1):
    with path.open('rb') as stream:
        response=requests.post(BASE+'/upload/image',files={'image':(path.name,stream)},data={'subfolder':'wdv3-acceptance','overwrite':'true'},timeout=30)
    response.raise_for_status();uploaded=response.json()
    results.append(run_graph(uploaded.get('subfolder','')+'/'+uploaded['name'],number))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
(OUT/'acceptance.json').write_text(json.dumps({'runs':results,'originals_unchanged':True,'device':'cpu'},indent=2),encoding='utf8')
print('PASS four images, raw scores, policies, reports and source integrity',flush=True)
