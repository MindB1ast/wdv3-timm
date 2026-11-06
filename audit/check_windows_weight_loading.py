"""Load a real WD checkpoint in an isolated process using the ComfyUI Python."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'temp/windows-load-check/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'temp/windows-load-check/matplotlib'))
faulthandler.enable()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import safetensors
    import psutil
    from Scripts.tagger_resources import load_network
    torch.set_num_threads(4)
    print('Loading real WD network without mmap:',args.model_folder,flush=True)
    start=time.perf_counter()
    network=load_network(args.model_folder)
    parameters=list(network.parameters())
    assert parameters and all(not p.is_meta and p.device.type=='cpu' for p in parameters)
    assert all(torch.isfinite(p.flatten()[:1]).all().item() for p in parameters)
    result=dict(model_folder=str(args.model_folder),python=sys.version,torch=torch.__version__,
                safetensors=safetensors.__version__,platform=sys.platform,
                reader='pread',seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in parameters),rss=psutil.Process().memory_info().rss,
                state_dict_strict=True,weights_loaded=True,finite_samples=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
