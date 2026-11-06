"""Real sequential SAM-worker initialization and WD loading, without a ComfyUI server."""
import json
import os
from pathlib import Path
import sys
import time
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_CONFIG_DIR=str(ROOT/'temp/windows-load-check/yolo'))


def main():
    from Scripts.sam3_adapter import SamWorker
    from Scripts.sam_config import SamConfig,SamSourceConfig
    from Scripts.tagger_resources import load_network
    import torch
    from PIL import Image
    inference='--inference' in sys.argv
    torch.set_num_threads(4)
    config=SamConfig(checkpoint=str(ROOT/'models/sam3'),device='cpu',allow_download=False,
                     sources=(SamSourceConfig('person','person','person',max_regions=4),) if inference else (),
                     backend='transformers',model_id='unsloth/sam3.1')
    started=time.perf_counter()
    print('Loading SAM alone in its worker',flush=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'temp') as temporary, SamWorker(sys.executable,config,timeout=900) as worker:
        child=worker.process
        assert worker.provenance['backend']=='transformers'
        print('SAM initialized successfully',flush=True)
        if inference:
            path=Path(temporary)/'image.png'
            with Image.new('RGB',(128,128),'white') as image:image.save(path)
            records=worker.segment([dict(image_id='memory-check',path=str(path),size=[128,128])],temporary)
            assert len(records)==1 and records[0]['status']=='ok'
            print('Real SAM segmentation completed',flush=True)
    assert child.poll() is not None
    sam_seconds=time.perf_counter()-started
    print('SAM process exited; loading WD Huge',flush=True)
    started=time.perf_counter()
    network=load_network(ROOT/'models/taggers/animetimm/convnextv2_huge.dbv4-full')
    assert all(not p.is_meta for p in network.parameters())
    if inference:
        from Scripts.tagger_resources import load_tagger_metadata
        from Scripts.tagger_preprocessing import prepare_tensor
        labels,transform,_=load_tagger_metadata('animetimm/convnextv2_huge.dbv4-full',ROOT/'models/taggers',allow_download=False)
        with Image.new('RGB',(128,128),'white') as image,torch.inference_mode():
            scores=network(prepare_tensor(image,transform).unsqueeze(0))
        assert scores.shape==(1,len(labels.names)) and torch.isfinite(scores).all().item()
    report=dict(sam_loaded=True,sam_worker_closed_before_wd=True,sam_seconds=sam_seconds,
                wd_loaded=True,wd_seconds=time.perf_counter()-started,models_simultaneously_resident=False,
                real_sam_segmentation=inference,real_wd_forward=inference)
    (ROOT/'temp'/('sam-wd-inference.json' if inference else 'sam-wd-memory.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
