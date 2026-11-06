"""Verify the ordinary Tag Image graph using local WD weights and persistent evidence."""
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from Scripts.workflow_api import runtime_module,normalized_image
    backend=runtime_module();package=backend.__package__
    nodes=importlib.import_module(package+'.comfy_nodes')
    regions=importlib.import_module(package+'.Scripts.region_tagging')
    import numpy as np
    import torch
    directory=ROOT/'temp/automatic-real';directory.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as name:
        root=Path(name)
        wd,_=nodes.WDModelLoader().load('vit','cpu',download_missing=False,
                                      local_model_path=str(ROOT/'models/taggers/wd-vit-tagger-v3'))
        with normalized_image(ROOT/'TestTagPic/tag_managing1.png') as image:
            image.thumbnail((384,384))
            image.save(root/'frame.png')
            tensor=torch.from_numpy(np.array(image)).float().unsqueeze(0)/255
        options=dict(cache_mode='reuse',cache_dataset_root=str(root))
        first=nodes.WDTagger().tag(tensor,wd,options=options)
        fresh,_=nodes.WDModelLoader().load('vit','cpu',False,str(ROOT/'models/taggers/wd-vit-tagger-v3'))
        calls=[]
        def forbidden(*args,**kwargs):
            calls.append('inference');raise AssertionError('Ready evidence must skip WD inference')
        resources=importlib.import_module(package+'.Scripts.tagger_resources')
        with patch.object(regions.RegionTagger,'score_batch',forbidden),patch.object(resources,'load_network',forbidden):
            second=nodes.WDTagger().tag(tensor,fresh,options=dict(options,general_threshold=1,character_threshold=1,prefix_tags='edited'))
        assert not calls and second['result'][0]==['edited']
        assert fresh._network is None
        script='''import importlib,json,sys,numpy as np,torch
from PIL import Image
from unittest.mock import patch
from Scripts.workflow_api import runtime_module
backend=runtime_module();package=backend.__package__
nodes=importlib.import_module(package+'.comfy_nodes')
resources=importlib.import_module(package+'.Scripts.tagger_resources')
regions=importlib.import_module(package+'.Scripts.region_tagging')
def forbidden(*a,**k):raise AssertionError('cold cache must not allocate weights or infer')
with patch.object(resources,'load_network',forbidden),patch.object(regions.RegionTagger,'score_batch',forbidden):
 wd,_=nodes.WDModelLoader().load('vit','cpu',False,sys.argv[2])
 with Image.open(sys.argv[1]+'/frame.png') as image: tensor=torch.from_numpy(np.array(image)).float().unsqueeze(0)/255
 out=nodes.WDTagger().tag(tensor,wd,options={'cache_mode':'reuse_only','cache_dataset_root':sys.argv[1],'general_threshold':1,'character_threshold':1,'prefix_tags':'edited'})
 assert wd._network is None
print(json.dumps(out['result'][0]))'''
        child=subprocess.run([sys.executable,'-c',script,str(root),str(ROOT/'models/taggers/wd-vit-tagger-v3')],
                             cwd=ROOT,capture_output=True,text=True,timeout=120)
        assert child.returncode==0,child.stderr
        assert json.loads(child.stdout)==['edited']
        before=json.loads(first['result'][1][0])['full_image']['raw_scores']
        after=json.loads(second['result'][1][0])['full_image']['raw_scores']
        assert before==after
        assert (root/'.wdv3/index.sqlite3').exists()
        summary=dict(schema_version=1,local_model='wd-vit-tagger-v3',existing_node='WDV3Tagger',
                     changed_filters=True,fresh_runtime=True,second_inference_calls=len(calls),
                     raw_unchanged=True,second_weight_loads=0,cold_process_without_weights=True,
                     model_loading_remains_upstream=False)
    destination=ROOT/'docs/verification/persistent-evidence/automatic-real.json'
    destination.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: ordinary Tag Image graph, local WD, changed filters, fresh runtime, zero repeated inference')


if __name__=='__main__':main()
