"""Existing graphs reuse persistent raw evidence with fresh loaded runtimes."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from audit.test_comfy_nodes import nodes,runtime

processing=importlib.import_module('wdv3_comfy_test.comfy_processing')
automatic=importlib.import_module('wdv3_comfy_test.comfy_automatic_cache')


class AutomaticCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='temp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.options=dict(cache_mode='reuse',cache_dataset_root=str(self.root))
        self.images=torch.ones(1,6,8,3)
        profile=patch.object(runtime().transform,'profile',SimpleNamespace(metadata=lambda:{'fixture':'rgb-resize4x4-v1'}),create=True)
        profile.start();self.addCleanup(profile.stop)

    def test_same_tagger_graph_new_runtime_new_filters_and_cold_process(self):
        first=runtime();nodes.WDTagger().tag(self.images,first,options=self.options)
        self.assertEqual(len(first.model.batches),1)
        second=runtime(True)
        result=nodes.WDTagger().tag(self.images,second,options=dict(self.options,general_threshold=1,character_threshold=1,prefix_tags='edited'))
        self.assertEqual(result['result'][0],['edited']);self.assertEqual(second.model.batches,[])
        script='''import json,sys,torch
from audit.test_comfy_nodes import nodes,runtime
from types import SimpleNamespace
wd=runtime(True)
wd.transform.profile=SimpleNamespace(metadata=lambda:{'fixture':'rgb-resize4x4-v1'})
out=nodes.WDTagger().tag(torch.ones(1,6,8,3),wd,options={'cache_mode':'reuse_only','cache_dataset_root':sys.argv[1],'general_threshold':1,'character_threshold':1})
assert wd.model.batches==[]
print(json.dumps(out['result'][0]))'''
        child=subprocess.run([sys.executable,'-c',script,str(self.root)],capture_output=True,text=True,timeout=60)
        self.assertEqual(child.returncode,0,child.stderr);self.assertEqual(json.loads(child.stdout),[''])

    def test_dataset_and_service_folder_both_reuse_without_inference(self):
        nodes.WDTagger().tag(self.images,runtime(),options=self.options)
        for folder in (self.root,self.root/'.wdv3',self.root/'.wdv3'/'.wdv3'):
            with self.subTest(folder=folder):
                wd=runtime(True)
                nodes.WDTagger().tag(self.images,wd,options=dict(self.options,cache_mode='reuse_only',cache_dataset_root=str(folder)))
                self.assertEqual(wd.model.batches,[])
        self.assertFalse((self.root/'.wdv3'/'.wdv3').exists())

    def test_missing_cache_explains_folder_and_does_not_infer(self):
        wd=runtime(True)
        with self.assertRaises(automatic.StoreError) as error:
            nodes.WDTagger().tag(self.images,wd,options=dict(self.options,cache_mode='reuse_only'))
        self.assertIn(str(self.root/'.wdv3'),str(error.exception))
        self.assertIn('No inference was performed',str(error.exception))
        self.assertEqual(wd.model.batches,[])

    def test_folder_repeated_save_uses_cache_and_explains_overwrite(self):
        with Image.new('RGB',(8,6),'white') as image:image.save(self.root/'one.png')
        files={'root':self.root,'paths':[self.root/'one.png']}
        kwargs=dict(output_directory=str(self.root/'output'),save_txt=True)
        processing.tag_folder(files,runtime(),{'cache_mode':'reuse'},**kwargs)
        wd=runtime(True)
        with self.assertRaisesRegex(FileExistsError,'enable overwrite'):
            processing.tag_folder(files,wd,{'cache_mode':'reuse_only'},**kwargs)
        self.assertEqual(wd.model.batches,[])
        tags,_=processing.tag_folder(files,wd,{'cache_mode':'reuse_only','prefix_tags':'updated'},overwrite=True,**kwargs)
        self.assertTrue(tags[0].startswith('updated'));self.assertEqual(wd.model.batches,[])
        self.assertEqual((self.root/'output/one.png.txt').read_text(),tags[0])

    def test_folder_root_without_store_or_inference_nodes(self):
        with Image.new('RGB',(8,6),'white') as image:image.save(self.root/'one.png')
        files={'root':self.root,'paths':[self.root/'one.png']}
        first=runtime();processing.tag_folder(files,first,{'cache_mode':'reuse'})
        second=runtime(True)
        tags,_=processing.tag_folder(files,second,{'cache_mode':'reuse','prefix_tags':'folder'})
        self.assertTrue(tags[0].startswith('folder'));self.assertEqual(second.model.batches,[])
        self.assertTrue((self.root/'.wdv3/index.sqlite3').exists())
        with Image.new('RGB',(8,6),'black') as image:image.save(self.root/'.wdv3/service.png')
        self.assertEqual(processing.scan_folder(self.root,True,0,0)['paths'],[self.root/'one.png'])

    def test_modes_and_missing_image_do_not_fake_hits(self):
        nodes.WDTagger().tag(self.images,runtime(),options=self.options)
        blocked=runtime(True)
        with self.assertRaisesRegex(ValueError,'Calculations unavailable'):
            nodes.WDTagger().tag(torch.zeros_like(self.images),blocked,options=dict(self.options,cache_mode='reuse_only'))
        self.assertEqual(blocked.model.batches,[])
        refreshed=runtime()
        nodes.WDTagger().tag(self.images,refreshed,options=dict(self.options,cache_mode='refresh'))
        self.assertEqual(len(refreshed.model.batches),1)

    def test_yolo_geometry_not_rerun_for_filters(self):
        path=self.root/'detector.pt';path.write_bytes(b'fixture-weights')
        config=nodes.DetectorConfig('body',str(path))
        class Detector:
            def __init__(self,fail=False):self.calls=0;self.fail=fail
            def to(self,*_):return self
            def __call__(self,*_,**kw):
                self.calls+=1
                if self.fail:raise AssertionError('YOLO must not run')
                return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[0,0,3,3]]),cls=torch.tensor([0]),conf=torch.tensor([.9]))])]
        detector=Detector()
        nodes.WDTagger().tag(self.images,runtime(),options=self.options,detectors=([config],{str(path):detector}))
        self.assertEqual(detector.calls,1)
        forbidden=Detector(True);wd=runtime(True)
        nodes.WDTagger().tag(self.images,wd,options=dict(self.options,remove_tags='light_tag'),detectors=([config],{str(path):forbidden}))
        self.assertEqual(forbidden.calls,0);self.assertEqual(wd.model.batches,[])

    def test_precomputed_sam_masks_survive_and_wd_is_reused(self):
        source=dict(name='face',prompt='face',confidence=.5,mask_threshold=.5,min_mask_area_px=1,max_regions=64,
                    view_mode='bbox',padding_px=0,rules={})
        mask=np.zeros((6,8),dtype=bool);mask[1:4,1:4]=True
        with Image.new('RGB',(8,6),'white') as image:digest=processing.frame_digest(image)
        record=dict(digest=digest,size=[8,6],sources=[source],model_settings={},regions={'face':[dict(mask=mask,bbox=[1,1,4,4],segmentation_score=.9)]})
        first=nodes.WDTagger().tag(self.images,runtime(),options=self.options,sam_regions=[record])
        wd=runtime(True)
        again=nodes.WDTagger().tag(self.images,wd,options=self.options,sam_regions=[record])
        self.assertEqual(wd.model.batches,[])
        self.assertTrue(np.array_equal(again['result'][2][0]['regions']['face'][0]['mask'],mask))

    def test_direct_sam_does_not_segment_again_for_new_rules(self):
        folder=self.root/'sam';folder.mkdir();(folder/'model.safetensors').write_bytes(b'sam-fixture')
        source=dict(name='face',prompt='face',confidence=.5,mask_threshold=.5,min_mask_area_px=1,max_regions=64,
                    view_mode='bbox',padding_px=0,rules={})
        mask=np.zeros((6,8),dtype=bool);mask[1:4,1:4]=True
        row=dict(mask=mask,bbox=[1,1,4,4],segmentation_score=.9)
        settings=dict(checkpoint=str(folder),model_id='fixture-sam',backend='fixture',device='cpu',allow_download=False)
        with patch.object(processing,'segment_frames',return_value=[{'regions':{'face':[row]}}]) as segment:
            nodes.WDTagger().tag(self.images,runtime(),options=self.options,sam_model=settings,sources=[source])
            self.assertEqual(segment.call_count,1)
        wd=runtime(True)
        with patch.object(processing,'segment_frames',side_effect=AssertionError('SAM must not run')):
            result=nodes.WDTagger().tag(self.images,wd,options=dict(self.options,remove_tags='light_tag'),sam_model=settings,
                                      sources=[dict(source,rules={'remove_tags_from_region':['dark_tag']})])
        self.assertEqual(wd.model.batches,[])
        self.assertTrue(np.array_equal(result['result'][2][0]['regions']['face'][0]['mask'],mask))


if __name__=='__main__':unittest.main()
