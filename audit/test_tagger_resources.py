"""Offline bundles, exact resource sets, access errors and metadata invalidation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests
import torch
from huggingface_hub.errors import HfHubHTTPError, EntryNotFoundError

from Scripts.tagger_resources import ensure_resources, load_tagger, load_weight_state
from Scripts.tagger_catalog import model_path


class ResourceTests(unittest.TestCase):
    def test_windows_weights_use_non_mmap_reader(self):
        calls=[]
        def reader(path,device='cpu',*,backend='mmap'):
            calls.append((path,device,backend))
            return {'weight':torch.tensor([1.])}
        with patch('Scripts.tagger_resources.sys.platform','win32'), \
             patch('Scripts.tagger_resources.safetensors.torch.load_file',reader):
            state=load_weight_state(Path('model.safetensors'))
        self.assertEqual(calls,[(Path('model.safetensors'),'cpu','pread')])
        self.assertEqual(state['weight'].item(),1.)

    def test_old_windows_safetensors_api_uses_bytes_without_mmap(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            path=Path(directory)/'model.safetensors';path.write_bytes(b'checkpoint')
            def old_reader(filename,device='cpu'):
                raise AssertionError('Legacy mmap reader must not be called on Windows')
            with patch('Scripts.tagger_resources.sys.platform','win32'), \
                 patch('Scripts.tagger_resources.safetensors.torch.load_file',old_reader), \
                 patch('Scripts.tagger_resources.safetensors.torch.load',return_value={'weight':1}) as reader:
                self.assertEqual(load_weight_state(path),{'weight':1})
            reader.assert_called_once_with(b'checkpoint')

    def test_non_windows_keeps_normal_reader(self):
        def reader(path):return {'weight':1}
        with patch('Scripts.tagger_resources.sys.platform','linux'), \
             patch('Scripts.tagger_resources.safetensors.torch.load_file',side_effect=reader) as load:
            self.assertEqual(load_weight_state(Path('model.safetensors')),{'weight':1})
        load.assert_called_once_with(Path('model.safetensors'))

    def test_complete_bundle_download_once_and_offline(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            fetched = []
            def fetch(repo, name, local_dir):
                fetched.append(name)
                path = Path(local_dir)/name
                contents = {
                    'config.json': json.dumps({'num_classes': 2, 'pretrained_cfg': {'input_size': [3,8,12]}, 'tags': ['tag', 'hero']}),
                    'selected_tags.csv': 'name,category\ntag,0\nhero,4',
                    'preprocess.json': json.dumps({'test': [{'type': 'resize', 'size': [8,12]}, {'type': 'maybe_to_tensor'}]}),
                }
                path.write_text(contents.get(name, '{}'))
                return str(path)
            with patch('Scripts.tagger_resources.hf_hub_download', side_effect=fetch), \
                 patch('Scripts.tagger_resources.load_network', return_value=torch.nn.Identity()):
                model, labels, transform, target = load_tagger('animetimm/example', root)
                self.assertEqual(set(fetched), {'config.json','selected_tags.csv','preprocess.json','model.safetensors'})
                self.assertEqual(transform.profile.input_size, (3,8,12))
                self.assertEqual(labels.names, ['tag', 'hero'])
                fetched.clear()
                load_tagger('animetimm/example', root, allow_download=False)
                self.assertEqual(fetched, [])
                config = json.loads((target/'config.json').read_text())
                config['tags'].reverse()
                (target/'config.json').write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, 'order'):
                    load_tagger('animetimm/example', root, allow_download=False)
                config['tags'].reverse()
                config['pretrained_cfg'].pop('input_size')
                (target/'config.json').write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, 'required pretrained_cfg'):
                    load_tagger('animetimm/example', root, allow_download=False)
            with patch('Scripts.tagger_resources.hf_hub_download') as download:
                with self.assertRaisesRegex(FileNotFoundError, 'preprocess.json'):
                    ensure_resources('animetimm/missing', root, allow_download=False)
                download.assert_not_called()

    def test_access_errors_and_only_missing_weights_fallback(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            for status in (401, 403):
                response = requests.Response(); response.status_code = status
                error = HfHubHTTPError('do not echo credential-bearing server text', response=response)
                with patch('Scripts.tagger_resources.hf_hub_download', side_effect=error):
                    with self.assertRaisesRegex(PermissionError, 'owner/model.*config.json') as caught:
                        ensure_resources('owner/model', root)
                    self.assertNotIn('credential-bearing', str(caught.exception))
            target = model_path(root, 'SmilingWolf/wd-vit-tagger-v3')
            target.mkdir(parents=True, exist_ok=True)
            (target/'config.json').write_text('{}'); (target/'selected_tags.csv').write_text('name,category\ntag,0')
            calls = []
            def fetch(repo, name, local_dir):
                calls.append(name)
                if name == 'model.safetensors':
                    response = requests.Response(); response.status_code = 404
                    raise EntryNotFoundError('missing')
                path = Path(local_dir)/name; path.write_bytes(b'weight'); return str(path)
            with patch('Scripts.tagger_resources.hf_hub_download', side_effect=fetch):
                self.assertEqual(ensure_resources('vit', root), target)
            self.assertEqual(calls, ['model.safetensors', 'pytorch_model.bin'])

    def test_repo_identity_mismatch(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            target = Path(directory)
            (target/'tagger-source.json').write_text(json.dumps({'repo_id': 'other/model'}))
            with self.assertRaisesRegex(ValueError, 'different repository'):
                ensure_resources('owner/model', target, local_path=target)
