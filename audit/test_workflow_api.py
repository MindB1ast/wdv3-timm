import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
from audit.test_comfy_evidence import pipeline,runtime

api=importlib.import_module('wdv3_comfy_test.Scripts.workflow_api')
storage=importlib.import_module('wdv3_comfy_test.Scripts.artifact_store')
persistent=importlib.import_module('wdv3_comfy_test.Scripts.persistent_evidence')


class WorkflowApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='temp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.store=storage.ArtifactStore(self.root)
        self.request=dict(schema_version=1,dataset_root=str(self.root))

    def cli(self,command,request,code=0):
        path=self.root/'request.json';path.write_text(json.dumps(request),encoding='utf-8')
        result=subprocess.run([sys.executable,'-m','Scripts.workflow_cli',command,'--request',str(path)],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,code,result.stderr+result.stdout)
        return json.loads(result.stdout)

    def test_cli_caption_json_exit_codes_and_idempotency(self):
        from Scripts.caption_files import document_from_tags
        request=dict(self.request,operation_id='export',entries=[dict(relative_path='image.png',document=document_from_tags(['custom_tag']))])
        plan=self.cli('caption-plan',request)['result']
        committed=self.cli('caption-commit',dict(self.request,plan=plan))['result']
        self.assertEqual(committed['files'][0]['status'],'written')
        self.assertEqual(self.cli('caption-commit',dict(self.request,plan=plan))['result'],committed)
        (self.root/'image.txt').write_text('manual')
        self.assertEqual(self.cli('caption-commit',dict(self.request,plan=plan),2)['result']['files'][0]['status'],'conflict')
        plan['files'][0]['after']=['different']
        self.assertFalse(self.cli('caption-commit',dict(self.request,plan=plan),1)['ok'])
        self.assertFalse(self.cli('status',dict(self.request,schema_version=2,operation_id='export'),1)['ok'])

    def test_policy_format_and_cold_cli_refilter(self):
        with Image.new('RGB',(8,6),'white') as image:
            raw=pipeline.collect([image],runtime())
        ref=persistent.save_evidence(self.store,raw)
        request=dict(self.request,artifact_id=ref,options={'general_threshold':0})
        first=api.refilter(request)
        second=api.refilter(dict(request,options={'general_threshold':0,'tag_separator':' | '}))
        self.assertEqual(second['policy_report']['actions'][0]['status'],'reused')
        self.assertEqual(second['text_report']['actions'][0]['status'],'computed')
        self.assertEqual(first['policy_report']['actions'][0]['artifact_id'],second['policy_report']['actions'][0]['artifact_id'])
        cold=self.cli('refilter',request)
        self.assertEqual(cold['result']['tags'],first['tags'])
        loaded=self.cli('load',request)['result']
        imported=self.cli('import',dict(self.request,calculations=loaded))['result']
        self.assertTrue(imported['capabilities']['refilter'])
        with Image.new('RGB',(8,6),'black') as image:image.save(self.root/'new.png')
        missing=dict(self.request,inference={'wd':{'identity':{'weights_sha256':'absent-pin'}},'sources':[]},
                     mode='reuse_only',files=['new.png'],operation_id='missing')
        self.assertFalse(self.cli('plan',missing,2)['result']['ready'])
        self.assertFalse(self.cli('execute',missing,2)['result']['complete'])
        self.assertEqual(self.cli('status',dict(self.request,operation_id='missing'))['result']['state'],'partial')

    def test_skip_only_prunes_inference_independent_of_evidence(self):
        with Image.new('RGB',(4,4),'white') as image:image.save(self.root/'image.png')
        (self.root/'image.txt').write_bytes(b'existing')
        request=dict(self.request,inference={'wd':{'identity':{'weights_sha256':'pin'}},'sources':[]},
                     outputs=['captions'],caption_export={'write_mode':'skip'},operation_id='skip')
        from unittest.mock import Mock
        forbidden=Mock(side_effect=AssertionError('inference'))
        report=api.execute(request,hooks={'wd':forbidden})
        self.assertTrue(report['complete']);forbidden.assert_not_called()
        self.assertEqual(report['entries'][0]['reason'],'existing_caption_only')
        self.assertEqual(api.execute(request),report)
        request=dict(request,mode='reuse_only',outputs=['captions','evidence'],operation_id='evidence')
        report=api.execute(request,hooks={'wd':forbidden})
        self.assertFalse(report['complete']);forbidden.assert_not_called()


if __name__=='__main__':unittest.main()
