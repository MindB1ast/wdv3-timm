from copy import deepcopy
from io import BytesIO
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from PIL import Image
from audit.test_comfy_evidence import pipeline

storage=importlib.import_module('wdv3_comfy_test.Scripts.artifact_store')
runtime=importlib.import_module('wdv3_comfy_test.comfy_persistent_runtime')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='temp');self.addCleanup(self.temp.cleanup)
        self.store=storage.ArtifactStore(Path(self.temp.name))
        self.image=Image.new('RGB',(8,8),'white');self.addCleanup(self.image.close)
        self.spec={'identity':{'weights_sha256':'wd-v1'}}
        self.config=dict(schema_version=1,wd=self.spec,sources=[dict(kind='yolo',name='body',confidence=.3,
                         classes=[0],model={'identity':{'weights_sha256':'yolo-v1'}})])
        self.hooks={'wd':lambda images:[dict(general={'skirt':.9},character={},ratings={'safe':.8}) for _ in images],
                    'yolo':lambda image,source:dict(regions=[dict(bbox=[0,0,4,4],region_id=0)])}

    def calculator(self,config=None,hooks=None):
        calc=runtime.PersistentCalculator(self.store,config or self.config,hooks=hooks or self.hooks)
        self.addCleanup(calc.close)
        return calc

    def test_incremental_cold_reuse_new_source_and_new_image(self):
        calc=self.calculator()
        raw,report=calc.compute_image(self.image)
        self.assertIsNotNone(raw);self.assertTrue(report['complete'])
        self.assertEqual(calc.counters['yolo_inference'],1)
        forbidden=Mock(side_effect=AssertionError('model called on hit'))
        cold=self.calculator(hooks={'wd':forbidden,'yolo':forbidden})
        restored,again=cold.compute_image(self.image,mode='reuse_only')
        self.assertIsNotNone(restored);forbidden.assert_not_called()
        self.assertEqual(restored.data()['frames'][0]['full_image'],raw.data()['frames'][0]['full_image'])
        updated=deepcopy(self.config)
        updated['sources'].append(dict(kind='yolo',name='face',confidence=.8,classes=[0],model={'identity':{'weights_sha256':'face-v1'}}))
        added=self.calculator(updated)
        _,report=added.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(added.counters['yolo_inference'],1)
        # Same actual crop pixels reuse WD across sources.
        self.assertEqual(added.counters['wd_inference'],0)
        with Image.new('RGB',(8,8),'black') as black:
            _,report=added.compute_image(black)
        self.assertEqual(added.counters['yolo_inference'],3)
        self.assertEqual(added.counters['wd_inference'],2)

    def test_sam_geometry_finishes_before_wd_and_does_not_reload_on_hits(self):
        config=deepcopy(self.config)
        config['sources']=[dict(kind='sam',name='sam_person',prompt='person',confidence=.5,
                               model={'identity':{'weights_sha256':'sam-v1'}})]
        events=[]
        def wd(images):
            events.append('wd')
            return self.hooks['wd'](images)
        def sam(image,source):
            events.append('sam')
            return {'regions':[]}
        calc=self.calculator(config,{'wd':wd,'sam':sam,'before_sam':lambda:events.append('release')})
        _,report=calc.compute_image(self.image)
        self.assertTrue(report['complete'])
        self.assertEqual(events,['release','sam','wd'])
        events.clear()
        _,report=calc.compute_image(self.image,mode='reuse_only')
        self.assertTrue(report['complete']);self.assertEqual(events,[])
        with Image.new('RGB',(8,8),'black') as image:
            _,report=calc.compute_image(image)
        self.assertTrue(report['complete']);self.assertEqual(events,['release','sam','wd'])

    def test_disabling_low_memory_keeps_wd_and_reuses_same_evidence(self):
        config=deepcopy(self.config)
        config['sources']=[dict(kind='sam',name='sam_person',prompt='person',confidence=.5,
                               model={'identity':{'weights_sha256':'sam-v1'}})]
        release=Mock(side_effect=AssertionError('WD must stay resident'))
        calc=self.calculator(config,dict(self.hooks,sam=lambda *_:{'regions':[]},before_sam=release))
        calc.options={'low_memory_mode':False}
        resident=Mock();calc.wd=resident
        _,report=calc.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertIs(calc.wd,resident)
        release.assert_not_called()
        calc.options={'low_memory_mode':True}
        _,report=calc.compute_image(self.image,mode='reuse_only')
        self.assertTrue(report['complete']);release.assert_not_called()

    def test_padding_and_wd_change_reuse_geometry(self):
        self.calculator().compute_image(self.image)
        updated=deepcopy(self.config);updated['sources'][0]['padding_px']=1
        padded=self.calculator(updated)
        _,report=padded.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(padded.counters['yolo_inference'],0)
        self.assertEqual(padded.counters['wd_inference'],1)
        updated['wd']={'identity':{'weights_sha256':'wd-v2'}}
        changed=self.calculator(updated)
        _,report=changed.compute_image(self.image)
        self.assertEqual(changed.counters['yolo_inference'],0)
        self.assertEqual(changed.counters['wd_inference'],2)

    def test_plan_does_not_load_models_and_reuse_only_missing(self):
        forbidden=Mock(side_effect=AssertionError('forbidden'))
        calc=self.calculator(hooks={'wd':forbidden,'yolo':forbidden})
        plan=calc.plan_image(self.image,mode='reuse_only')
        self.assertFalse(plan['ready']);self.assertTrue(plan['pending_regions'])
        evidence,report=calc.compute_image(self.image,mode='reuse_only')
        self.assertIsNone(evidence);forbidden.assert_not_called()

    def test_bounded_crop_batches_and_resource_only_changes(self):
        batches=[]
        def scores(images):
            batches.append(len(images));return self.hooks['wd'](images)
        hooks=dict(self.hooks,wd=scores,yolo=lambda *_:{'regions':[dict(bbox=[0,0,3,3]),dict(bbox=[0,0,4,4])]})
        calc=self.calculator(hooks=hooks);calc.options={'region_batch_size':2}
        _,report=calc.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(batches,[1,2])
        batches.clear();calc.options={'region_batch_size':1,'image_batch_size':1,'sam_batch_size':8}
        _,report=calc.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(batches,[])

    def test_sam_masks_recrop_and_prompt_invalidation(self):
        config=deepcopy(self.config)
        config['sources']=[dict(kind='sam',name='clothes',prompt='skirt',confidence=.5,
                               view_mode='bbox',padding_px=0,model={'identity':{'weights_sha256':'sam-v1'}})]
        with Image.new('L',(8,8),0) as mask:
            mask.paste(255,(1,1,5,5))
            buffer=BytesIO();mask.save(buffer,format='PNG')
        ref=self.store.put_blob(buffer.getvalue())
        hooks=dict(self.hooks,sam=lambda image,source:{'regions':[dict(bbox=[1,1,5,5],mask_blob=ref,segmentation_score=.9)]})
        first=self.calculator(config,hooks)
        raw,report=first.compute_image(self.image)
        self.assertEqual(first.counters['sam_inference'],1)
        self.assertIn('mask_png',raw.data()['frames'][0]['detectors']['clothes'][0])
        changed=deepcopy(config);changed['sources'][0].update(view_mode='mask',padding_px=1)
        recrop=self.calculator(changed,hooks)
        _,report=recrop.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(recrop.counters['sam_inference'],0)
        self.assertEqual(recrop.counters['wd_inference'],1)
        changed['sources'][0]['prompt']='dress'
        reprompt=self.calculator(changed,hooks)
        _,report=reprompt.compute_image(self.image)
        self.assertTrue(report['complete']);self.assertEqual(reprompt.counters['sam_inference'],1)


if __name__=='__main__':
    unittest.main()
