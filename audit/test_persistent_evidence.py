import base64
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image

from audit.test_comfy_evidence import pipeline, Bundle, runtime

storage = importlib.import_module('wdv3_comfy_test.Scripts.artifact_store')
persistent = importlib.import_module('wdv3_comfy_test.Scripts.persistent_evidence')


class EvidenceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='temp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = storage.ArtifactStore(self.root)
        with Image.new('RGB',(8,6),'white') as image:
            self.raw = pipeline.collect([image], runtime())

    def test_cold_process_full_bundle_and_refilter_without_models(self):
        data = self.raw.data()
        with Image.new('L',(8,6),255) as mask:
            encoded = pipeline.png(mask)
        frame = data['frames'][0]
        from dataclasses import asdict
        frame['source_settings'] = [asdict(pipeline.DetectorConfig('face','sam3'))]
        frame['detectors'] = {'face':[dict(frame['full_image'], region_id=0, bbox=[0,0,8,6], mask_png=encoded)]}
        frame['geometry']['sam'] = [{'name':'face'}]
        raw = Bundle.pack('evidence', data)
        ref = persistent.save_evidence(self.store, raw)
        self.assertEqual(persistent.load_evidence(self.store, ref).data(), raw.data())
        program = '''import json,sys
from audit.test_comfy_evidence import pipeline
import importlib
s=importlib.import_module('wdv3_comfy_test.Scripts.artifact_store')
p=importlib.import_module('wdv3_comfy_test.Scripts.persistent_evidence')
from unittest.mock import patch
store=s.ArtifactStore(sys.argv[1]); raw=p.load_evidence(store,sys.argv[2])
with patch.object(pipeline,'collect',side_effect=AssertionError('inference forbidden')):
 result=pipeline.apply_filters(raw,{'general_threshold':0})
assert raw.data()['frames'][0]['detectors']['face'][0]['mask_png']
print(json.dumps({'frames':len(result.data()['frames']),'caps':p.capabilities(raw.data())}))'''
        result = subprocess.run([sys.executable,'-c',program,str(self.root),ref], capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['caps']['recrop_masks'])

    def test_legacy_json_refilter_only_and_incomplete_rejected(self):
        results = pipeline.apply_filters(self.raw)
        _, texts = pipeline.outputs(results)
        value = json.loads(texts[0])
        imported = persistent.import_calculations(self.store, value)
        self.assertTrue(imported['capabilities']['refilter'])
        self.assertFalse(imported['capabilities']['inspect_pixels'])
        old = persistent.load_evidence(self.store, imported['artifact_id'])
        self.assertEqual(old.data()['frames'][0]['full_image']['general'], self.raw.data()['frames'][0]['full_image']['general'])
        for broken in ({'tags':['smile']}, 'smile', dict(value, full_image={'general':{'smile':1}})):
            with self.assertRaises(ValueError):
                persistent.import_calculations(self.store, broken)
        bad = deepcopy(value); bad['model']['labels_count'] += 1
        with self.assertRaises(ValueError):
            persistent.import_calculations(self.store, bad)

    def test_rebind_digest_and_missing_source(self):
        with Image.new('RGB',(8,6),'white') as image:
            image.save(self.root/'image.png')
        rebound = persistent.rebind_paths(self.store,self.raw,{'0':'image.png'})
        self.assertEqual(rebound.data()['frames'][0]['relative_path'],'image.png')
        with Image.new('RGBA',(8,6),(255,0,0,0)) as image:
            image.save(self.root/'image.png')
        self.assertEqual(persistent.rebind_paths(self.store,self.raw,{'0':'image.png'}).data()['frames'][0]['relative_path'],'image.png')
        with Image.new('RGB',(8,6),'black') as image:
            image.save(self.root/'image.png')
        with self.assertRaises(ValueError):
            persistent.rebind_paths(self.store,self.raw,{'0':'image.png'})
        with self.assertRaises(ValueError):
            persistent.rebind_paths(self.store,self.raw,{'0':'missing.png'})


if __name__ == '__main__':
    unittest.main()
