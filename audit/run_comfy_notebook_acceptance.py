"""Execute the existing notebook entry cells on a disposable image, without ComfyUI registration."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',MPLBACKEND='Agg',YOLO_CONFIG_DIR=str(ROOT/'temp/ultralytics-comfy-tests'),
                  MPLCONFIGDIR=str(ROOT/'temp/matplotlib-cache'),IPYTHONDIR=str(ROOT/'temp/comfy-ipython-profile'))
from IPython.core.interactiveshell import InteractiveShell
import torch

protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'TestTagPic').iterdir() if p.is_file()}
workspace=ROOT/'temp'/('comfy-notebook-'+uuid.uuid4().hex)
images=workspace/'images';images.mkdir(parents=True)
shutil.copy2(ROOT/'TestTagPic/tag_managing1.png',images/'sample.png')
notebook=json.loads((ROOT/'ManagingTags.ipynb').read_text(encoding='utf8'))
shell=InteractiveShell.instance()
reply=shell.run_cell(''.join(notebook['cells'][0]['source']))
assert reply.success
params=shell.user_ns['params']
params.ImageFolder=str(images);params.model='vit';params.gen_threshold=.35;params.char_threshold=.75
params.model_folder=str(ROOT/'models/taggers');params.yolo_model_dir=str(ROOT/'models/yolo')
params.detectors_config=str(ROOT/'detectors.json');params.output_file=str(workspace/'results.json')
params.save_txt=True;params.append_txt=False
reply=shell.run_cell(''.join(notebook['cells'][1]['source']))
assert reply.success
result=shell.user_ns['result']
assert isinstance(result,tuple) and [len(part) for part in result]==[1,1]
assert json.loads((workspace/'results.json').read_text(encoding='utf8'))
assert (images/'sample.txt').read_text(encoding='utf8')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in protected.items())
assert importlib.util.find_spec('comfy_api') is None
assert not any(name.endswith('.comfy_ui_nodes') for name in sys.modules)
summary=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,device='cpu',
             actual_notebook_cells=['ManagingTags:0','ManagingTags:1'],result_lengths=[len(p) for p in result],
             public_signature=str(inspect.signature(shell.user_ns['BatchTagging'])),
             txt_json_saved=True,originals_unchanged=True,without_comfy_api=True,custom_nodes_not_registered=True,
             workspace=str(workspace))
(ROOT/'docs/verification/comfyui/notebook.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('PASS existing notebook cells, public BatchTagging API, standalone exports and source integrity',flush=True)
