"""A fresh ordinary loader must not allocate weights for ready persistent evidence."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from audit.test_comfy_nodes import nodes,RecordingModel,LabelData
from audit.test_tagger_preprocessing import published

resources=importlib.import_module('wdv3_comfy_test.Scripts.tagger_resources')
regions=importlib.import_module('wdv3_comfy_test.Scripts.region_tagging')


class DeferredRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='temp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.folder=self.root/'wd';self.folder.mkdir()
        (self.folder/'model.safetensors').write_bytes(b'fixture-weight-v1')
        self.transform=published('animetimm/caformer_b36.dbv4-full')
        self.labels=LabelData(['a','b','c'],[],[0,1],[2])
        self.metadata=patch.object(resources,'load_tagger_metadata',return_value=(self.labels,self.transform,self.folder))
        self.metadata.start();self.addCleanup(self.metadata.stop)
        self.images=torch.ones(1,6,8,3)
        self.options=dict(cache_mode='reuse',cache_dataset_root=str(self.root))

    def loader(self):return nodes.WDModelLoader().load('vit','cpu',False,str(self.folder))[0]

    def test_release_weights_preserves_identity_and_reloads_only_on_inference(self):
        with patch.object(resources,'load_network',side_effect=lambda folder:RecordingModel()) as load:
            wd=self.loader();identity=wd.persistent_identity
            first=wd.model
            wd.release_weights()
            self.assertIsNone(wd._network)
            self.assertEqual(wd.persistent_identity,identity)
            self.assertEqual(load.call_count,1)
            self.assertIsNot(wd.model,first)
            self.assertEqual(load.call_count,2)

    def test_loader_then_full_hit_with_weights_and_inference_forbidden(self):
        with patch.object(resources,'load_network',return_value=RecordingModel()) as network:
            first=self.loader();network.assert_not_called();self.assertIsNone(first._network)
            nodes.WDTagger().tag(self.images,first,options=self.options)
            network.assert_called_once()
        with patch.object(resources,'load_network',side_effect=AssertionError('weights loaded')) as network, \
             patch.object(regions.RegionTagger,'score_batch',side_effect=AssertionError('inferred')) as forward:
            second=self.loader()
            result=nodes.WDTagger().tag(self.images,second,options=dict(self.options,general_threshold=1,character_threshold=1,prefix_tags='edited'))
            self.assertEqual(result['result'][0],['edited'])
            network.assert_not_called();forward.assert_not_called();self.assertIsNone(second._network)

    def test_missing_dataset_root_fails_before_weights(self):
        with patch.object(resources,'load_network',side_effect=AssertionError('weights loaded')) as network:
            wd=self.loader()
            with self.assertRaisesRegex(ValueError,'cache_dataset_root'):
                nodes.WDTagger().tag(self.images,wd,options={'cache_mode':'reuse'})
            network.assert_not_called()

    def test_reuse_only_miss_does_not_load_and_refresh_does(self):
        with patch.object(resources,'load_network',return_value=RecordingModel()) as network:
            wd=self.loader()
            with self.assertRaises(ValueError):
                nodes.WDTagger().tag(self.images,wd,options=dict(self.options,cache_mode='reuse_only'))
            network.assert_not_called()
            nodes.WDTagger().tag(self.images,wd,options=dict(self.options,cache_mode='refresh'))
            network.assert_called_once()

    def test_changed_weights_identity_misses_without_loading(self):
        with patch.object(resources,'load_network',return_value=RecordingModel()):
            wd=self.loader()
            nodes.WDTagger().tag(self.images,wd,options=self.options)
        (self.folder/'model.safetensors').write_bytes(b'fixture-weight-v2')
        with patch.object(resources,'load_network',side_effect=AssertionError('weights loaded')) as network:
            with self.assertRaises(ValueError):
                nodes.WDTagger().tag(self.images,wd,options=dict(self.options,cache_mode='reuse_only'))
            network.assert_not_called()
            self.assertIsNone(wd._network)


if __name__=='__main__':unittest.main()
