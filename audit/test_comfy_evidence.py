import copy
from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from audit.test_comfy_nodes import runtime,nodes

pipeline=importlib.import_module('wdv3_comfy_test.comfy_evidence')
Bundle=importlib.import_module('wdv3_comfy_test.Scripts.evidence').Bundle
text_evidence=importlib.import_module('wdv3_comfy_test.Scripts.evidence').text_evidence
reports=importlib.import_module('wdv3_comfy_test.comfy_reports')


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(8,6),'white');self.addCleanup(self.image.close)
        self.wd=runtime()
    def collect(self,configs=None):
        return pipeline.collect([self.image],self.wd,detectors=(configs or [],{}))
    def test_raw_scores_and_filtering_do_not_run_models_again(self):
        evidence=self.collect();count=len(self.wd.model.batches)
        high=pipeline.apply_filters(evidence,{'general_threshold':1,'character_threshold':1})
        low=pipeline.apply_filters(evidence,{'general_threshold':0,'character_threshold':0})
        self.assertEqual(high.data()['frames'][0]['merged_tags']['general'],{})
        self.assertEqual(len(high.data()['frames'][0]['full_image']['raw_scores']['general']),2)
        self.assertEqual(high.data()['frames'][0]['decisions'][0]['reason'],'below_threshold')
        self.assertTrue(low.data()['frames'][0]['merged_tags']['general'])
        again=self.collect();self.assertEqual(len(self.wd.model.batches),count)
        self.assertEqual(again.payload,evidence.payload)
        with self.assertRaises(FrozenInstanceError):evidence.payload='bad'
        changed=evidence.data();changed['frames'].clear();self.assertEqual(len(evidence.data()['frames']),1)
        self.assertEqual(Bundle.from_json(evidence.to_json()),evidence)
    def test_cache_invalidation_and_model_offload_on_error(self):
        self.collect();count=len(self.wd.model.batches)
        with Image.new('RGB',(8,6),'black') as black:pipeline.collect([black],self.wd)
        self.assertGreater(len(self.wd.model.batches),count)
        failed=runtime(True)
        with self.assertRaisesRegex(RuntimeError,'inference failed'):pipeline.collect([self.image],failed)
        self.assertEqual(failed.model.moves[-1],'cpu')
    def test_cancel_and_detector_error_always_offload_resources(self):
        with patch.object(pipeline,'interrupt',side_effect=[None,RuntimeError('cancelled')]):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):self.collect()
        self.assertEqual(self.wd.model.moves[-1],'cpu')
        class Detector:
            def __init__(self):self.moves=[]
            def to(self,device):self.moves.append(str(device));return self
            def __call__(self,*args,**kwargs):raise RuntimeError('detector failed')
        detector=Detector()
        with self.assertRaisesRegex(RuntimeError,'detector failed'):
            pipeline.collect([self.image],runtime(),detectors=([nodes.DetectorConfig('face','fake')],{'fake':detector}))
        self.assertEqual(detector.moves[-1],'cpu')
    def test_comfy_inference_tensor_fingerprint_and_weight_change(self):
        with torch.inference_mode():
            self.wd.model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(48,3))
            original=self.collect()
            self.wd.model[1].weight.add_(1)
            changed=self.collect()
        self.assertNotEqual(original.data()['model']['weights_sha256'],changed.data()['model']['weights_sha256'])
    def test_filter_changes_reuse_yolo_and_wd_but_detection_changes_invalidate(self):
        detector_calls=[]
        from types import SimpleNamespace
        def detector(image,**kwargs):
            detector_calls.append(kwargs)
            return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[0,0,4,4]]))])]
        config=nodes.DetectorConfig('face','fake')
        def run(rule):return pipeline.collect([self.image],self.wd,detectors=([rule],{'fake':detector}))
        first=run(config);n=len(self.wd.model.batches)
        config.allowed_tags=['light_tag'];config.only_from_source=['light_tag']
        second=run(config);result=pipeline.apply_filters(second)
        self.assertEqual(len(detector_calls),1);self.assertEqual(len(self.wd.model.batches),n)
        self.assertEqual(result.data()['frames'][0]['merged_tags']['provenance']['light_tag']['source'],'face:0')
        config.confidence=.8;run(config);self.assertEqual(len(detector_calls),2)
        config.confidence=.9
        with self.assertRaisesRegex(ValueError,'Detection parameters changed'):pipeline.apply_filters(first,configs=[config])
        config.classes=None
        all_classes=run(config)
        filtered=pipeline.apply_filters(all_classes,configs=[config])
        self.assertIsNone(filtered.data()['frames'][0]['settings']['sources'][0]['classes'])
    def test_merging_deduplicates_and_rejects_conflicts_and_other_images(self):
        evidence=self.collect();result=pipeline.apply_filters(evidence)
        self.assertEqual(pipeline.merge_results(result,result).data()['frames'],result.data()['frames'])
        other=evidence.data();other['frames'][0]['image_id']='other'
        with self.assertRaisesRegex(ValueError,'different images'):
            pipeline.merge_results(result,pipeline.apply_filters(Bundle.pack('evidence',other)))
        original=evidence.data();row=copy.deepcopy(original['frames'][0]['full_image']);row.update(region_id=0,bbox=[0,0,4,4])
        original['frames'][0]['detectors']={'face':[row]};original['frames'][0]['source_settings']=[__import__('dataclasses').asdict(nodes.DetectorConfig('face','fake'))]
        a=pipeline.apply_filters(Bundle.pack('evidence',original));original['frames'][0]['detectors']['face'][0]['general']['light_tag']=.1
        b=pipeline.apply_filters(Bundle.pack('evidence',original))
        with self.assertRaisesRegex(ValueError,'Conflicting'):pipeline.merge_results(a,b)
    def test_text_adapters_preserve_text_and_no_artificial_score(self):
        text='A girl, with blue eyes.\n  Spaces remain!'
        bundle=text_evidence(text,image_id='i',source_id='face',model_id='classifier',output_kind='class_label',
                            settings={'prompt':'Describe the face','temperature':.2},region_id=0)
        copy=Bundle.from_json(bundle.to_json())
        self.assertEqual(copy.data()['text'],text);self.assertNotIn('score',copy.data())
        self.assertEqual(copy.data()['settings']['prompt'],'Describe the face')
        second=text_evidence('  Blue eyes, not_tags. ',image_id='i',source_id='face',model_id='vlm',region_id=0)
        adapters=[Bundle.from_json(value.to_json()).data() for value in [bundle,second]]
        self.assertEqual([value['model_id'] for value in adapters],['classifier','vlm'])
        self.assertEqual(adapters[1]['text'],'  Blue eyes, not_tags. ')
    def test_all_snapshot_types_are_versioned_and_read_only(self):
        for kind in ('model','regions','evidence','rules','results','report','text'):
            bundle=Bundle.pack(kind,{'values':[1]})
            self.assertEqual(Bundle.from_json(bundle.to_json()),bundle)
            bundle['values'].append(2)
            self.assertEqual(bundle['values'],[1])
            with self.assertRaises(TypeError):bundle['values']=[2]
        with self.assertRaises(ValueError):Bundle.from_json('{"schema_version":true,"kind":"rules","payload":{}}')
    def test_independent_branch_merge_order_and_source_completeness(self):
        data=self.collect().data()
        def branch(name,score):
            snapshot=copy.deepcopy(data)
            snapshot['frames'][0]['detectors']={name:[dict(general={'light_tag':score},character={},ratings={},region_id=0,bbox=[0,0,4,4])]}
            snapshot['frames'][0]['source_settings']=[__import__('dataclasses').asdict(nodes.DetectorConfig(name,'fake',only_from_source=['light_tag']))]
            return pipeline.apply_filters(Bundle.pack('evidence',snapshot))
        a,b=branch('eyes',.9),branch('face',.7)
        ab,ba=pipeline.merge_results(a,b),pipeline.merge_results(b,a)
        self.assertEqual(ab.data()['frames'],ba.data()['frames'])
        frame=ab.data()['frames'][0]
        self.assertEqual(frame['merged_tags']['provenance']['light_tag']['source'],'eyes:0')
        self.assertEqual({r['source'] for r in frame['decisions'] if r['tag']=='light_tag'},{'full','eyes','face'})
    def test_full_reports_api_assets_html_safety_and_repeat_export(self):
        result=pipeline.apply_filters(self.collect(),{'prefix_tags':'<script>alert(1)</script>'})
        report=reports.make_report(result)
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            first=reports.export_report(report,directory,prefix='safe')
            second=reports.export_report(report,directory,prefix='safe')
            self.assertNotEqual(first['html'],second['html'])
            html=Path(first['html']).read_text(encoding='utf8')
            self.assertNotIn('<script>alert(1)</script>',html)
            self.assertIn('\\u003cscript',html)
            data=json.loads(Path(first['json']).read_text(encoding='utf8'))
            self.assertTrue((Path(first['json']).parent/data['frames'][0]['image_asset']).is_file())
            self.assertIn('below_threshold',{d['reason'] for d in data['frames'][0]['decisions']})
            rows,images,tags=reports.preview_report(report,directory)
            self.assertEqual(images[0].shape,(1,6,8,3));self.assertEqual(len(rows),1)
            raw_report=reports.make_report(self.collect())
            with self.assertRaisesRegex(ValueError,'Global TXT'):reports.export_report(raw_report,directory)


if __name__=='__main__':unittest.main()
