"""Integration acceptance on disposable copies, using real local model weights."""
import ast
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', MPLBACKEND='Agg',
                  YOLO_CONFIG_DIR=str(ROOT / 'temp/yolo-settings'),
                  MPLCONFIGDIR=str(ROOT / 'temp/matplotlib-cache'))
from PIL import Image
from Scripts import BatchTagging, ScriptOptions
from Scripts.analitycs import compare_tagging_methods
from Scripts.savingProc import save_tags_to_txt
from Scripts.tag_utils import parse_tags
from Scripts.visualization import visualize_detection_results

OUT = ROOT / 'temp/apply-integration'
INPUT = OUT / 'images'
INPUT.mkdir(parents=True, exist_ok=True)
source = ROOT / 'TestTagPic/tag_managing1.png'
image = INPUT / source.name
shutil.copy2(source, image)
image.with_suffix('.txt').write_text('manual preserved', encoding='utf-8')
opts = ScriptOptions(ImageFolder=INPUT, model='vit', gen_threshold=0.35,
                     char_threshold=0.75, save_txt=True, append_txt=True,
                     output_file=str(OUT / 'results.json'))
results = BatchTagging(opts)
assert isinstance(results, tuple) and len(results) == 2
assert all(isinstance(items, list) and len(items) == 1 for items in results)
detailed, merged = results[0][0], results[1][0]
for record in [detailed['full_image'], merged['merged_tags']] + [
        region for regions in detailed['detectors'].values() for region in regions]:
    expected = set(record['general']) | set(record['character'])
    assert set(parse_tags(record['caption'])) == expected
    assert set(parse_tags(record['taglist'])) == expected
for filename, expected in [('results.json', results[0]), ('results_merged.json', results[1])]:
    assert json.loads((OUT / filename).read_text(encoding='utf-8')) == expected
saved = image.with_suffix('.txt')
assert set(parse_tags(saved.read_text(encoding='utf-8'))) == (
    set(parse_tags(merged['merged_tags']['taglist'])) | {'manual_preserved'})
first = saved.read_bytes()
save_tags_to_txt(image, merged['merged_tags'], append_tags=True)
assert first == saved.read_bytes(), 'Append must be idempotent'
with Image.open(image) as input_image:
    expected_size = input_image.size
for extension in ('png', 'jpg'):
    target = OUT / ('regions.' + extension)
    figure = visualize_detection_results(str(image), detailed, str(target), show=False)
    assert figure.__class__.__name__ == 'Figure'
    with Image.open(target) as preview:
        preview.load()
        assert preview.size == expected_size
    figure.write_html(str(OUT / ('interactive-' + extension + '.html')))

# Exercise the notebook comparison against existing dataset labels, without rewriting them.
before = json.loads((ROOT / 'temp/audit-20261029-smoke/results_merged.json').read_text(encoding='utf-8'))
after = json.loads((ROOT / 'temp/apply-fixes-smoke/results_merged.json').read_text(encoding='utf-8'))
old_by_name = {Path(item['image_path']).name: item for item in before}
changes = []
for item in after:
    old = old_by_name[Path(item['image_path']).name]['merged_tags']
    new = item['merged_tags']
    old_tags = set(old['general']) | set(old['character'])
    new_tags = set(new['general']) | set(new['character'])
    changes.append({'image': Path(item['image_path']).name, 'before': len(old_tags),
                    'after': len(new_tags), 'added': sorted(new_tags-old_tags),
                    'removed': sorted(old_tags-new_tags)})
OUTPUT = ROOT / 'docs/verification/fix-audited-tagging-defects'
(OUTPUT / 'smoke-before-after.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
smoke_root = ROOT / 'temp/apply-fixes-smoke/images'
label_root = ROOT / 'TestTagPic'
known = {path.stem for path in label_root.glob('*.txt')}
smoke_results = tuple([
    item for item in json.loads((ROOT / ('temp/apply-fixes-smoke/' + filename)).read_text(encoding='utf-8'))
    if Path(item['image_path']).stem in known] for filename in ('results.json', 'results_merged.json'))
original_cwd = Path.cwd()
try:
    os.chdir(OUT)
    with (OUT / 'notebook-comparison.log').open('w', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        comparison = compare_tagging_methods(str(smoke_root), str(label_root), smoke_results,
            str(ROOT / 'models/taggers/wd-vit-tagger-v3/selected_tags.csv'),
            visualize=False, interactive=False)
    assert isinstance(comparison, tuple) and len(comparison) == 5
finally:
    os.chdir(original_cwd)
old_tree = ast.parse((ROOT / 'temp/apply-original/Scripts/main.py').read_text(encoding='utf-8'))
old_signature = next(node.args for node in old_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'BatchTagging')
new_tree = ast.parse((ROOT / 'Scripts/main.py').read_text(encoding='utf-8'))
new_signature = next(node.args for node in new_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'BatchTagging')
assert ast.dump(old_signature) == ast.dump(new_signature)
hashes = json.loads((OUTPUT / 'before-hashes.json').read_text(encoding='utf-8-sig'))
protected = [entry for entry in hashes if 'TestTagPic' in entry['Path'] or
             entry['Path'].endswith('test_current_behavior.py')]
integrity = []
for entry in protected:
    actual = hashlib.sha256(Path(entry['Path']).read_bytes()).hexdigest().upper()
    integrity.append({'path': entry['Path'], 'unchanged': actual == entry['Hash']})
assert all(item['unchanged'] for item in integrity)
summary = {'signature': str(inspect.signature(BatchTagging)), 'result_lists': [len(x) for x in results],
           'json_round_trip': True, 'txt_append_idempotent': True, 'static_formats': ['PNG', 'JPEG'],
           'plotly_preserved': True, 'notebook_tuple_length': len(comparison),
           'source_integrity': integrity, 'model': 'wd-vit-tagger-v3', 'input_size': [448, 448],
           'device': 'cpu', 'source_images_unchanged': True}
(OUTPUT / 'integration.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
