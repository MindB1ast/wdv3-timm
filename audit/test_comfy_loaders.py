import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from audit.test_comfy_nodes import nodes,RecordingModel,transform

model_module=importlib.import_module('wdv3_comfy_test.Scripts.model')


class LoaderTests(unittest.TestCase):
    def test_wd_offline_missing_files_no_network_and_only_one_weight_format(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            target=Path(directory).resolve()
            with patch('wdv3_comfy_test.Scripts.tagger_resources.hf_hub_download') as download:
                with self.assertRaisesRegex(FileNotFoundError,'config.json'):
                    nodes.WDModelLoader().load('vit','cpu',False,str(target))
                download.assert_not_called()
            network=RecordingModel();network.pretrained_cfg={}
            fetched=[]
            def fetch(repo,filename,local_dir):
                fetched.append(filename);path=Path(local_dir)/filename
                path.write_text('name,category\nlight_tag,0\ndark_tag,0\nhero,4' if filename=='selected_tags.csv' else
                                json.dumps({'num_classes': 3, 'pretrained_cfg': {'input_size': [3,4,4],
                                            'mean': [0,0,0], 'std': [1,1,1], 'interpolation': 'bicubic'}}))
                return str(path)
            with patch('wdv3_comfy_test.Scripts.tagger_resources.hf_hub_download',side_effect=fetch), \
                 patch('timm.create_model',return_value=network), \
                 patch('safetensors.torch.load_file',return_value={}), \
                 patch('timm.data.config.resolve_data_config',return_value={'input_size': (3,4,4), 'mean': (0,0,0), 'std': (1,1,1), 'interpolation': 'bicubic'}), \
                 patch('timm.data.transforms_factory.create_transform',return_value=transform):
                loaded=nodes.WDModelLoader().load('vit','cpu',True,str(target))[0]
                self.assertEqual(set(fetched),{'model.safetensors','config.json','selected_tags.csv'})
                self.assertEqual(loaded.model_id,'SmilingWolf/wd-vit-tagger-v3')
                before=len(fetched);nodes.WDModelLoader().load('vit','cpu',False,str(target))
                self.assertEqual(len(fetched),before)
    def test_explicit_unavailable_cuda_and_yolo_offline(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'CUDA'):nodes.WDModelLoader().load('vit','cuda')
        with tempfile.TemporaryDirectory(dir='temp') as directory,patch.object(model_module,'_remote_yolo_model') as remote:
            with self.assertRaisesRegex(FileNotFoundError,'download_missing'):
                model_module.load_yolo_model('missing.pt',Path(directory),allow_download=False)
            remote.assert_not_called()
    def test_local_yolo_path_reused_offline(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            target=Path(directory)/'local.pt';target.write_bytes(b'fixture')
            with patch.object(model_module,'YOLO',return_value='loaded') as loader,patch.object(model_module,'_remote_yolo_model') as remote:
                self.assertEqual(model_module.load_yolo_model('local.pt',Path(directory),allow_download=False),'loaded')
                loader.assert_called_once_with(str(target));remote.assert_not_called()


if __name__=='__main__':unittest.main()
