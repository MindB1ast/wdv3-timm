"""Execute saved notebook cells against disposable inputs in the actual kernel."""
import ast, json, os
from pathlib import Path
from jupyter_client import KernelManager

root=Path.cwd();out=root/'docs/verification/typed-contracts'
manager=KernelManager(kernel_name='wdv3-timm-venv')
manager.start_kernel(cwd=str(root),env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1', 'CUDA_VISIBLE_DEVICES':''})
client=manager.blocking_client();mimes=set()
def output(message):
    content=message['content']
    if message['msg_type']=='stream': print(content['text'],end='',flush=True)
    if message['msg_type'] in ('display_data','execute_result'):mimes.update(content.get('data',{}))
    if message['msg_type']=='error':print(content,flush=True)
def run(code):
    reply=client.execute_interactive(code,timeout=300,output_hook=output)
    assert reply['content']['status']=='ok',reply['content']
try:
    client.start_channels();client.wait_for_ready(timeout=60)
    managing=json.loads((root/'ManagingTags.ipynb').read_text(encoding='utf-8'))
    analytics=json.loads((root/'Analitycs.ipynb').read_text(encoding='utf-8'))
    run('''
from pathlib import Path
import json, os, shutil, torch
from PIL import Image
root = Path.cwd()
workspace = root/'temp/typed-notebook'
images = workspace/'images'
images.mkdir(parents=True, exist_ok=True)
labels = workspace/'labels'
labels.mkdir(parents=True, exist_ok=True)
sources = sorted(path for path in (root/'TestTagPic').iterdir() if path.suffix.lower() in ('.png','.jpg','.jpeg'))[:4]
for index, source in enumerate(sources):
    shutil.copy2(source, images/('sample.png' if index == 0 else source.name))
    shutil.copy2(source.with_suffix('.txt'), (labels/('sample.png' if index == 0 else source.name)).with_suffix('.txt'))
assert len(sources) == 4
assert not torch.cuda.is_available()
''')
    run(''.join(managing['cells'][0]['source']))
    run('''
params.ImageFolder = images
params.model = 'vit'
params.gen_threshold = .35
params.char_threshold = .75
params.detectors_config = str(root/'detectors.json')
params.save_txt = True
params.append_txt = False
params.output_file = str(workspace/'results.json')
''')
    run(''.join(managing['cells'][1]['source']))
    run('''
from Scripts.tag_utils import parse_tags
from Scripts.visualization import visualize_detection_results
assert result is not None and [len(items) for items in result] == [4, 4]
baseline = root/'temp/notebook-setup/integration'
old_detailed = json.loads((baseline/'results.json').read_text(encoding='utf-8'))[0]
old_merged = json.loads((baseline/'results_merged.json').read_text(encoding='utf-8'))[0]
new_detailed = next(item for item in result[0] if Path(item['image_path']).name == 'sample.png')
new_merged = next(item for item in result[1] if Path(item['image_path']).name == 'sample.png')
assert old_detailed['full_image'] == new_detailed['full_image']
assert old_detailed['detectors'] == new_detailed['detectors']
assert old_merged['merged_tags'] == new_merged['merged_tags']
for record in result[1]:
    assert set(parse_tags(Path(record['txt_path']).read_text(encoding='utf-8'))) == set(parse_tags(record['merged_tags']['taglist']))
for ext in ('png', 'jpg'):
    target = workspace/('regions.'+ext)
    figure = visualize_detection_results(new_detailed['image_path'],new_detailed,str(target),show=False)
    with Image.open(target) as saved:
        saved.load()
        assert saved.format == ('PNG' if ext=='png' else 'JPEG')
figure.show(renderer='plotly_mimetype')
''')
    for index in (2,3,4):run(''.join(analytics['cells'][index]['source']))
    # Override only dataset/model/presentation arguments on the saved comparison cell.
    tree=ast.parse(''.join(analytics['cells'][6]['source']))
    values={'image_folder':str(root/'temp/typed-notebook/images'), 'txt_folder':str(root/'temp/typed-notebook/labels'), 'model_labels_path':str(root/'models/taggers/wd-vit-tagger-v3/selected_tags.csv'), 'visualize':False,'interactive':False}
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg in values:node.value=ast.Constant(values[node.arg])
    run('os.chdir(workspace)\n'+ast.unparse(ast.fix_missing_locations(tree)))
    run('''
assert isinstance(data, tuple) and len(data) == 5
from Scripts.evaluation import METRIC_EPSILON, METRIC_POLICY_VERSION
assert METRIC_EPSILON == 1e-8
summary = {'kernel_python': __import__('sys').executable, 'device':'cpu',
    'actual_notebook_cells':['ManagingTags:0,1','Analitycs:2,3,4,6'],
    'overrides':'Disposable dataset, existing local WD ViT and detector configuration, presentation flags only',
    'result_lengths':[len(items) for items in result], 'comparison_tuple_length':len(data),
    'baseline_predictions_scores_provenance_geometry_unchanged':True,
    'txt_json_saved':True,'static_formats':['PNG','JPEG'],
    'metric_policy_version':METRIC_POLICY_VERSION,'epsilon':METRIC_EPSILON}
(root/'docs/verification/typed-contracts/notebook.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Notebook acceptance complete')
''')
    assert 'application/vnd.plotly.v1+json' in mimes,mimes
finally:
    client.stop_channels();manager.shutdown_kernel(now=True)
