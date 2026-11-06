"""Local WD+YOLO, cold refilter, neighboring TXT and selective extension."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'temp/persistent-yolo-config')
    from Scripts import workflow_api as api
    from Scripts.artifact_store import ArtifactStore
    from Scripts.caption_files import plan_caption_export,commit_caption_export
    from PIL import Image
    wd=ROOT/'models/taggers/wd-vit-tagger-v3'
    def model(path,kind):
        if kind=='wd':
            files={('weights' if p.name=='model.safetensors' else p.name):str(p) for p in wd.iterdir()
                   if p.name in ('model.safetensors','selected_tags.csv','config.json','preprocess.json')}
            return dict(model_id='vit',local_path=str(wd),files=files,compute={'device':'cpu'},adapter='wdv3-v1')
        return dict(local_path=str(path),files={'weights':str(path)},compute={'device':'cpu'},adapter='yolo-v1')
    work=ROOT/'temp/persistent-real';work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        dataset=Path(directory)
        source=next((ROOT/'TestTagPic').glob('*.png'))
        with Image.open(source) as original:
            image=original.convert('RGB');image.thumbnail((384,384));image.save(dataset/'one.png');image.close()
        inference=dict(schema_version=1,wd=model(wd,'wd'),sources=[dict(name='body',kind='yolo',
             confidence=.25,classes=[0],model=model(ROOT/'models/yolo/person_yolov8s-seg.pt','yolo'))])
        request=dict(schema_version=1,dataset_root=str(dataset),inference=inference,operation_id='first',mode='reuse')
        first=api.execute(request)
        assert first['complete'],first
        assert first['counters']['wd_load']==1 and first['counters']['yolo_inference']==1
        ref=first['entries'][0]['evidence_ref']
        # Fresh process explicitly blocks loaders and inference while applying changed rules.
        request_path=dataset/'request.json'
        request_path.write_text(json.dumps(dict(schema_version=1,dataset_root=str(dataset),artifact_id=ref,
                                               options={'general_threshold':.2,'remove_tags_regex':['watermark'],'tag_separator':' | '})))
        code='''import json,sys
from unittest.mock import patch
from Scripts import workflow_api as api
backend=api.runtime_module()
import importlib
mods=[(backend,'PersistentCalculator'),(importlib.import_module(backend.__package__+'.Scripts.tagger_resources'),'load_tagger'),
      (importlib.import_module(backend.__package__+'.Scripts.model'),'load_yolo_model')]
from contextlib import ExitStack
def forbidden(*a,**k):raise AssertionError('cold loader forbidden')
with ExitStack() as stack:
 for module,name in mods:stack.enter_context(patch.object(module,name,forbidden))
 result=api.refilter(json.load(open(sys.argv[1])))
print(json.dumps(result))'''
        cold=subprocess.run([sys.executable,'-c',code,str(request_path)],cwd=ROOT,capture_output=True,text=True,timeout=120)
        assert cold.returncode==0,cold.stderr
        candidate=json.loads(cold.stdout)
        captions=api.captions_from_results(candidate['results'])
        store=ArtifactStore(dataset)
        saved=commit_caption_export(store,plan_caption_export(store,captions,operation_id='final',config={'options':{'tag_separator':' | '}}))
        assert saved['files'][0]['status']=='written'
        assert (dataset/'one.txt').read_bytes()==candidate['tags'][0].encode()
        # A new file and source request computes only the new geometry and missing pixels.
        with Image.open(dataset/'one.png') as original:
            image=original.transpose(Image.Transpose.FLIP_LEFT_RIGHT);image.save(dataset/'two.png');image.close()
        inference['sources'].append(dict(name='face',kind='yolo',confidence=.35,classes=[0],
               model=model(ROOT/'models/yolo/face_yolov9c.pt','yolo')))
        second=api.execute(dict(request,inference=inference,operation_id='extend'))
        assert second['complete'],second
        body=[a for a in second['entries'][0]['actions'] if a.get('source')=='body']
        assert body[0]['status']=='reused'
        full=[a for a in second['entries'][0]['actions'] if a['stage']=='wd_full']
        assert full[0]['status']=='reused'
        assert len(list(store.scan()))==2
        summary=dict(schema_version=1,first=first,extended=second,cold_refilter_without_loaders=True,
                     final_neighbor_txt=True,raw_ref_unchanged=ref,local_weights=['wd-vit-tagger-v3','person_yolov8s-seg.pt','face_yolov9c.pt'],
                     sam_real_inference=False)
        destination=ROOT/'docs/verification/persistent-evidence';destination.mkdir(parents=True,exist_ok=True)
        (destination/'real-models.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS: local WD+YOLO, separate-process refilter without loaders, neighbor TXT, incremental image/source')


if __name__=='__main__':main()
