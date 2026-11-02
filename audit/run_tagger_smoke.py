"""Real sequential CPU acceptance; saves no captions beside source images."""
import gc
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from Scripts.tagger_resources import load_tagger
from Scripts.tagging import tag_image
from Scripts.region_tagging import RegionTagger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', action='append', help='Run selected repositories, preserving other completed records')
    args = parser.parse_args()
    torch.set_num_threads(4)
    package_spec = importlib.util.spec_from_file_location('tagger_smoke_comfy', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_spec.name] = package
    package_spec.loader.exec_module(package)
    nodes = sys.modules['tagger_smoke_comfy.comfy_nodes']
    report = {'device': 'cpu', 'torch': torch.__version__, 'tolerance': {'rtol': 1e-5, 'atol': 1e-6}, 'models': []}
    directory = ROOT/'docs/verification/tagger-models'
    directory.mkdir(parents=True, exist_ok=True)
    if args.repo and (directory/'smoke.json').is_file():
        report = json.loads((directory/'smoke.json').read_text(encoding='utf-8'))
        report['models'] = [row for row in report['models'] if row['repo_id'] not in args.repo]
    image = Image.new('RGB', (173, 101), (180, 70, 30))
    for repo in (args.repo or ['SmilingWolf/wd-vit-tagger-v3', 'animetimm/caformer_b36.dbv4-full', 'animetimm/convnextv2_huge.dbv4-full']):
        print('Real smoke', repo, flush=True)
        start = time.monotonic()
        with patch('socket.socket', side_effect=AssertionError('offline network access')):
            model, labels, transform, target = load_tagger(repo, ROOT/'models/taggers', allow_download=False)
            seen = []
            hook = model.register_forward_pre_hook(lambda _, args: seen.append(list(args[0].shape)))
            full = tag_image(image, model, transform, labels, -1, -1, torch.device('cpu'))
            regions = RegionTagger(model, transform, labels, torch.device('cpu')).score_batch([image, image])
            hook.remove()
            for category in ('general', 'character', 'ratings'):
                for region in regions:
                    np.testing.assert_allclose(list(full[category].values()), list(region[category].values()), rtol=1e-5, atol=1e-6)
            del model
            gc.collect()
            # Exercise the actual ComfyUI loader offline, not a fixture runtime.
            runtime = nodes.WDModelLoader().load('vit', 'cpu', False, str(target), repo_id=repo)[0]
            tensor = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0)/255
            result = nodes.WDTagger().tag(tensor, runtime, general_threshold=0, character_threshold=0)
            decoded = json.loads(result['result'][1][0])
            actual = decoded['full_image']
            for category in ('general', 'character', 'ratings'):
                self_keys = sorted(full[category])
                np.testing.assert_allclose([full[category][key] for key in self_keys],
                                           [actual[category][key] for key in self_keys], rtol=1e-5, atol=1e-6)
            record = {'repo_id': repo, 'revision': transform.profile.revision,
                      'status': 'passed', 'seconds': time.monotonic()-start,
                      'inputs': seen, 'profile': transform.profile.metadata(), 'labels': len(labels.names),
                      'offline': True, 'notebook_comfy_parity': True}
            # Legacy manually supplied weights may have no discoverable Hub revision.
            weights = target/'model.safetensors'
            if not weights.is_file():
                weights = target/'pytorch_model.bin'
            with weights.open('rb') as stream:
                record['weights_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            report['models'].append(record)
            (directory/'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(repo, seen, 'passed', flush=True)
            del runtime, transform, labels, regions, full, result
            gc.collect()
    image.close()


if __name__ == '__main__':
    main()
