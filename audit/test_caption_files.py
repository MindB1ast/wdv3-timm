import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from Scripts.artifact_store import ArtifactStore, StoreError
from Scripts.caption_files import (caption_document, combine_captions, commit_caption_export,
    document_from_tags, edit_caption, export_config, format_caption, parse_caption, plan_caption_export)


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='temp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = ArtifactStore(self.root)
        self.path = self.root / 'sub' / 'image.txt'
        self.path.parent.mkdir()
        self.entry = dict(relative_path='sub/image.png', document=document_from_tags(['blue_eyes','long_hair']))

    def plan(self, mode='error', operation='one', **config):
        return plan_caption_export(self.store, [self.entry], operation_id=operation,
                                   config=dict(write_mode=mode, **config))

    def test_bytes_bom_order_unknown_tags_and_pure_import(self):
        raw = b'\xef\xbb\xbfblue eyes,blue_eyes\r\nmy_custom_tag'
        doc = caption_document(raw)
        self.assertEqual(doc['tags'], ['blue_eyes','my_custom_tag'])
        self.assertEqual(base64.b64decode(doc['original_base64']), raw)
        self.assertNotIn('score', json.dumps(doc))
        code = 'import sys; from Scripts import caption_files; assert not {"torch","numpy","comfy_api","ultralytics","transformers"} & sys.modules.keys()'
        subprocess.run([sys.executable, '-c', code], check=True, timeout=15)

    def test_separators_parentheses_and_ambiguous_spaces(self):
        self.assertEqual(parse_caption(r'blue eyes | hero_\(series\)', input_format='literal_separator', input_separator=' | ', escaped_parentheses=True), ['blue_eyes','hero_(series)'])
        self.assertEqual(parse_caption('blue eyes, smile'), ['blue_eyes','smile'])
        for separator in ('', ' '):
            with self.assertRaises(StoreError):
                parse_caption('blue eyes smile', input_format='literal_separator', input_separator=separator)
        self.assertEqual(format_caption(['blue_eyes','smile'], {'tag_separator':' | '}), 'blue eyes | smile')

    def test_ordered_edits_removal_priority_and_literal_regex(self):
        doc = caption_document(b'blue eyes, yellow_eyes, watermark, custom_tag, .*')
        result = edit_caption(doc, [dict(replace={'blue eyes':'yellow_eyes'}, remove=['watermark'],
                                          add_before=['watermark','start'], remove_regex=['custom_.*']),
                                    dict(add_after=['end']), dict(sort='priority', priority=['end','start'])])
        self.assertEqual(result['tags'], ['end','start','yellow_eyes','.*'])
        self.assertEqual(doc['tags'][0], 'blue_eyes')
        self.assertNotIn('score', json.dumps(result))
        merged = combine_captions(doc, document_from_tags(['yellow_eyes','new_tag']))
        self.assertEqual(merged['tags'], doc['tags'] + ['new_tag'])

    def test_four_modes_existing_and_absent(self):
        original = b'\xef\xbb\xbfblue eyes, smile\r\n'
        for mode in ('error','skip','append','overwrite'):
            with self.subTest(mode=mode):
                self.path.write_bytes(original)
                plan = self.plan(mode, mode)
                self.assertEqual(self.path.read_bytes(), original)
                result = commit_caption_export(self.store, plan)
                status = result['files'][0]['status']
                if mode in ('error','skip'):
                    self.assertEqual(status, 'conflict' if mode=='error' else 'skipped')
                    self.assertEqual(self.path.read_bytes(), original)
                else:
                    self.assertEqual(status, 'written')
                    self.assertEqual(self.path.read_text(), 'blue eyes, smile, long hair' if mode=='append' else 'blue eyes, long hair')
                self.path.unlink()
                result = commit_caption_export(self.store, self.plan(mode, mode+'-absent'))
                self.assertEqual(result['files'][0]['status'], 'written')

    def test_append_unchanged_preserves_bytes(self):
        original = b'blue_eyes,  long hair\r\n'
        self.path.write_bytes(original)
        plan = self.plan('append')
        self.assertEqual(plan['files'][0]['status'], 'unchanged')
        commit_caption_export(self.store, plan)
        self.assertEqual(self.path.read_bytes(), original)

    def test_conflict_idempotency_and_manual_edit_after_commit(self):
        self.path.write_bytes(b'old')
        plan = self.plan('overwrite')
        self.path.write_bytes(b'edited')
        self.assertEqual(commit_caption_export(self.store, plan)['files'][0]['status'], 'conflict')
        self.assertEqual(self.path.read_bytes(), b'edited')
        fresh = self.plan('overwrite', 'two')
        self.assertEqual(commit_caption_export(self.store, fresh)['files'][0]['status'], 'written')
        self.assertEqual(commit_caption_export(self.store, fresh)['files'][0]['status'], 'written')
        self.path.write_bytes(b'new external edit')
        self.assertEqual(commit_caption_export(self.store, fresh)['files'][0]['status'], 'conflict')
        with self.assertRaises(StoreError):
            commit_caption_export(self.store, self.plan('append', 'two'))

    def test_faults_and_resume(self):
        self.path.write_bytes(b'original')
        for moment in ('before_replace','after_replace'):
            self.path.write_bytes(b'original')
            plan = self.plan('overwrite', moment)
            def fail(point, path):
                if point == moment:
                    raise OSError('injected interruption')
            result = commit_caption_export(self.store, plan, fault=fail)
            self.assertEqual(result['files'][0]['status'], 'failed')
            if moment == 'before_replace':
                self.assertEqual(self.path.read_bytes(), b'original')
            resumed = commit_caption_export(self.store, plan)
            self.assertIn(resumed['files'][0]['status'], ('written','unchanged'))
            self.assertEqual(self.path.read_text(), 'blue eyes, long hair')

    def test_mapping_collision_case_subdirectories_boundaries(self):
        entries = [self.entry, dict(self.entry, relative_path='sub/image.jpg')]
        with self.assertRaisesRegex(StoreError, 'collision'):
            plan_caption_export(self.store, entries, operation_id='one')
        plan = plan_caption_export(self.store, entries, operation_id='one', config={'naming':'image_extension_txt'})
        self.assertTrue(plan['files'][0]['path'].endswith('image.png.txt'))
        entries[1] = dict(self.entry, relative_path='other/image.png')
        self.assertEqual(len(plan_caption_export(self.store, entries, operation_id='one')['files']), 2)
        entries[1] = dict(self.entry, relative_path='SUB/IMAGE.PNG')
        with self.assertRaises(StoreError):
            plan_caption_export(self.store, entries, operation_id='one')
        with self.assertRaises(StoreError):
            self.plan(naming='explicit', mapping={'sub/image.png':'../escape.txt'})
        with self.assertRaises(StoreError):
            self.plan(naming='explicit', mapping={'sub/image.png':'.wdv3/evil.txt'})

    def test_concurrent_creation_rechecked(self):
        plan = self.plan()
        def create(point, path):
            if point == 'before_replace':
                path.write_bytes(b'external creator')
        result = commit_caption_export(self.store, plan, fault=create)
        self.assertEqual(result['files'][0]['status'], 'conflict')
        self.assertEqual(self.path.read_bytes(), b'external creator')

    def test_two_processes_one_destination(self):
        plan=self.plan()
        other=json.loads(json.dumps(plan));other['operation_id']='other'
        inputs=[]
        for index,value in enumerate((plan,other)):
            path=self.root/f'plan-{index}.json';path.write_text(json.dumps(value));inputs.append(path)
        code='''import json,sys
from Scripts.artifact_store import ArtifactStore
from Scripts.caption_files import commit_caption_export
print(json.dumps(commit_caption_export(ArtifactStore(sys.argv[1]),json.load(open(sys.argv[2])))))'''
        children=[subprocess.Popen([sys.executable,'-c',code,str(self.root),str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for p in inputs]
        statuses=[]
        for child in children:
            out,err=child.communicate(timeout=30)
            self.assertEqual(child.returncode,0,err)
            statuses.append(json.loads(out)['files'][0]['status'])
        self.assertIn('written',statuses)
        self.assertTrue(all(s in ('written','unchanged','conflict') for s in statuses))
        self.assertEqual(self.path.read_text(),'blue eyes, long hair')

    def test_partial_folder_resume_keeps_completed_file(self):
        entries=[self.entry,dict(self.entry,relative_path='sub/second.png')]
        plan=plan_caption_export(self.store,entries,operation_id='folder')
        def fail(point,path):
            if point=='before_replace' and path.name=='second.txt':raise OSError('second interrupted')
        first=commit_caption_export(self.store,plan,fault=fail)
        self.assertEqual([r['status'] for r in first['files']],['written','failed'])
        timestamp=self.path.stat().st_mtime_ns
        again=commit_caption_export(self.store,plan)
        self.assertEqual([r['status'] for r in again['files']],['written','written'])
        self.assertEqual(self.path.stat().st_mtime_ns,timestamp)

    def test_symlink_destination_boundary(self):
        inside=self.root/'export';inside.mkdir()
        outside=self.root/'outside';outside.mkdir()
        link=inside/'sub'
        if os.name=='nt':
            result=subprocess.run(['cmd','/c','mklink','/J',str(link),str(outside)],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
        else:
            link.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(StoreError,'boundary'):
            self.plan(in_place=False,target_root=str(inside))


if __name__ == '__main__':
    unittest.main()
