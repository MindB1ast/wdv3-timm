"""Actual ComfyUI input contract, metadata outputs and evidence cache identity."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from audit.test_comfy_nodes import nodes, RecordingModel, LabelData
from audit.test_tagger_preprocessing import published

evidence = importlib.import_module('wdv3_comfy_test.comfy_evidence')
resources = importlib.import_module('wdv3_comfy_test.Scripts.tagger_resources')


class ComfyProfileTests(unittest.TestCase):
    def test_catalog_override_metadata_and_old_call(self):
        inputs = nodes.WDModelLoader.INPUT_TYPES()
        self.assertIn('animetimm/caformer_b36.dbv4-full', inputs['required']['model'][0])
        self.assertIn('vit', inputs['required']['model'][0])
        self.assertIn('repo_id', inputs['optional'])
        transform = published('animetimm/caformer_b36.dbv4-full')
        labels = LabelData(['a','b','c'], [], [0,1], [2])
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            folder = Path(directory).resolve()
            with patch.object(resources, 'load_tagger', return_value=(RecordingModel(), labels, transform, folder)) as load:
                saved = json.loads(json.dumps({'model': 'vit', 'device': 'cpu', 'download_missing': False,
                                             'local_model_path': str(folder), 'repo_id': transform.profile.repo_id}))
                runtime, info = nodes.WDModelLoader().load(**saved)
                self.assertEqual(load.call_args.args[0], transform.profile.repo_id)
                metadata = json.loads(info)
                self.assertEqual(metadata['input_size'], [3,384,384])
                self.assertEqual(metadata['color_order'], 'RGB')
                self.assertEqual(metadata['source'], 'preprocess.json:test')
                self.assertEqual(runtime.model_id, transform.profile.repo_id)
                nodes.WDModelLoader().load('vit', 'cpu', False, str(folder))
                self.assertEqual(load.call_args.args[0], 'SmilingWolf/wd-vit-tagger-v3')
            with patch.object(nodes, 'model_directory', return_value=folder):
                old = nodes.WDModelLoader.IS_CHANGED('vit','cpu',False,str(folder))
                (folder/'preprocess.json').write_text('{}')
                new = nodes.WDModelLoader.IS_CHANGED('vit','cpu',False,str(folder))
                self.assertNotEqual(old,new)

    def test_profile_and_dictionary_invalidate_identity(self):
        transform = published('animetimm/caformer_b36.dbv4-full')
        labels = LabelData(['a','b','c'], [], [0,1], [2])
        runtime = nodes.WDRuntime(RecordingModel(), transform, labels, torch.device('cpu'), transform.profile.repo_id)
        before = evidence.model_identity(runtime)
        transform.profile = replace(transform.profile, color_order='BGR')
        self.assertNotEqual(before, evidence.model_identity(runtime))
        before = evidence.model_identity(runtime)
        labels.names[0] = 'different'
        self.assertNotEqual(before, evidence.model_identity(runtime))
