import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from audit.test_conditional_tags import fixture, policy, pipeline, post, Bundle
from audit.test_comfy_nodes import runtime
processing=importlib.import_module('wdv3_comfy_test.comfy_processing')
cache=importlib.import_module('wdv3_comfy_test.comfy_automatic_cache')

class FolderPostprocessingTests(unittest.TestCase):
    def test_cached_batches_order_exports_and_validation(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root=Path(directory); (root/'nested').mkdir()
            paths=[root/'a.png',root/'nested/b.png',root/'c.png']
            for p in paths: Image.new('RGB',(8,6)).save(p)
            files=dict(root=root,paths=paths)
            original=fixture().data()
            def cached(frames,*args,**kw):
                data=dict(original);data['frames']=[original['frames'][0] for _ in frames]
                return Bundle.pack('evidence',data)
            chain=[policy(scope='full'),dict(explicit_enabled=True,scope='full',relations=[dict(specific='dress',general='skirt')])]
            for operations,has_skirt in [(chain,True),(list(reversed(chain)),False)]:
                out=root/str(has_skirt)
                with patch.object(cache,'cached_evidence',side_effect=cached) as hit, patch.object(pipeline,'collect',side_effect=AssertionError('inference')):
                    tags,details=processing.tag_folder(files,None,dict(image_batch_size=2,cache_mode='reuse_only'),postprocessing=operations,output_directory=out,save_txt=True,save_json=True)
                self.assertEqual(hit.call_count,2)
                self.assertEqual(len(tags),3)
                row=json.loads(details[1])
                self.assertEqual('skirt' in row['full_image']['general'],has_skirt)
                self.assertIn('dress',row['full_image']['raw_scores']['general'])
                self.assertEqual((out/'nested/b.png.txt').read_text(),tags[1])
                self.assertEqual(json.loads((out/'nested/b.png.json').read_text()),row)
                with self.assertRaises(FileExistsError): processing.tag_folder(files,None,{},output_directory=out,save_txt=True)
            with patch.object(cache,'cached_evidence',side_effect=AssertionError('must validate first')):
                with self.assertRaises(ValueError): processing.tag_folder(files,None,{},postprocessing=[dict(operation='unknown')])

    def test_uncached_matches_explicit_chain_and_empty_legacy(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root=Path(directory);path=root/'a.png';Image.new('RGB',(8,6)).save(path)
            files=dict(root=root,paths=[path]);chain=[policy([dict(id='remove',none_of=['missing'],remove=['1girl'])],scope='all')]
            with patch.object(cache,'cached_evidence',return_value=None):
                tags,details=processing.tag_folder(files,runtime(),{},postprocessing=chain)
                with Image.open(path) as image: raw=pipeline.collect([image],runtime(),options={},image_paths=[path])
                expected=pipeline.outputs(pipeline.merge_tag_sets(post.apply_chain(pipeline.filter_tag_sets(raw,{}),chain)))
                self.assertEqual(tags,expected[0])
                self.assertEqual(json.loads(details[0])['full_image'],json.loads(expected[1][0])['full_image'])
                legacy=processing.tag_folder(files,runtime(),{})
                self.assertEqual(legacy,processing.tag_folder(files,runtime(),{},postprocessing=[]))
