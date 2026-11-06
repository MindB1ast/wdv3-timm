import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from Scripts.artifact_store import ArtifactStore
from Scripts.inference_plan import (Stage, execute, file_digest, format_snapshot,
                                    model_identity, pixel_identity, plan, policy_snapshot)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='temp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ArtifactStore(self.root)
        self.spec = {'identity': {'weights_sha256': 'pinned', 'adapter': 'v1'}}

    def test_identity_content_not_stat_or_path(self):
        path = self.root / 'weight'
        path.write_bytes(b'aaa')
        stat = path.stat()
        first = model_identity({'files': {'weights': str(path)}})
        path.write_bytes(b'bbb')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(first, model_identity({'files': {'weights': str(path)}}))
        moved = path.rename(self.root / 'other')
        self.assertEqual(model_identity({'files': {'weights': str(moved)}}),
                         model_identity({'files': {'weights': str(moved)}, 'batch_size': 64}))
        self.assertNotEqual(model_identity({'files': {'weights': str(moved)}, 'adapter': 'v2'}),
                            model_identity({'files': {'weights': str(moved)}}))
        self.assertEqual(file_digest(moved), file_digest(moved))
        from PIL import Image
        image_path=self.root/'image.bmp'
        with Image.new('RGB',(8,8),'white') as image:
            image.save(image_path);first_pixels=pixel_identity(image)
        stat=image_path.stat()
        with Image.new('RGB',(8,8),'black') as image:image.save(image_path)
        os.utime(image_path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with Image.open(image_path) as image:changed_pixels=pixel_identity(image)
        self.assertNotEqual(first_pixels,changed_pixels)
        image_path.rename(self.root/'moved.bmp')
        with Image.open(self.root/'moved.bmp') as image:self.assertEqual(pixel_identity(image),changed_pixels)

    def test_policy_and_formatter_independent(self):
        a = {'general_threshold': .5, 'tag_separator': ', ', 'region_batch_size': 1}
        b = dict(a, tag_separator=' | ', region_batch_size=64)
        self.assertEqual(policy_snapshot(a), policy_snapshot(b))
        self.assertNotEqual(format_snapshot(a), format_snapshot(b))
        self.assertNotEqual(policy_snapshot(a), policy_snapshot(dict(a, general_threshold=.1)))

    def test_plan_modes_no_executor_on_hits_and_missing(self):
        full = Stage('wd_full', {'image': 'one'}, model=self.spec)
        source = Stage('yolo', {'image': 'one'}, source='person', model=self.spec)
        other = Stage('wd_full', {'image': 'two'}, model=self.spec)
        calls = []
        def run(action, deps):
            calls.append(action['computation_key'])
            return {'scores': [1, 2]}
        first = execute(self.store, plan(self.store, [full, source]), {'wd_full': run, 'yolo': run})
        self.assertTrue(first['complete'])
        forbidden = Mock(side_effect=AssertionError('loader called'))
        hit = execute(self.store, plan(self.store, [full, source], mode='reuse_only'), {'wd_full': forbidden, 'yolo': forbidden})
        self.assertTrue(hit['complete']); forbidden.assert_not_called()
        partial = execute(self.store, plan(self.store, [full, source, other]), {'wd_full': run, 'yolo': run})
        self.assertTrue(partial['complete']); self.assertEqual(len(calls), 3)
        unknown = Stage('sam', {}, model={'files': {'weights': 'missing'}})
        missing = execute(self.store, plan(self.store, [unknown], mode='reuse_only'), {'sam': forbidden})
        self.assertEqual(missing['actions'][0]['reason'], 'unresolved_identity'); forbidden.assert_not_called()
        blob = self.store.blob_path(first['actions'][0]['artifact_id'])
        blob.write_bytes(b'corrupt')
        corrupt = plan(self.store, [full], mode='reuse_only')
        self.assertEqual(corrupt['actions'][0]['reason'], 'corrupt_artifact')
        execute(self.store, corrupt, {'wd_full': forbidden}); forbidden.assert_not_called()

    def test_scoped_refresh_and_partial_failure(self):
        a = Stage('yolo', {}, source='body', model=self.spec)
        b = Stage('yolo', {}, source='face', model={**self.spec, 'identity': {'weights_sha256': 'face'}})
        c = Stage('evidence', {}, (a.key, b.key))
        executors = {'yolo': lambda *_: {'ok': 1}, 'evidence': lambda *_: {'ok': 1}}
        old = execute(self.store, plan(self.store, [a,b,c]), executors)
        new = execute(self.store, plan(self.store, [a,b,c], mode='refresh', refresh=['body']), executors)
        self.assertEqual([r['status'] for r in new['actions']], ['computed','reused','computed'])
        self.assertEqual(old['actions'][1]['artifact_id'], new['actions'][1]['artifact_id'])
        self.assertEqual(self.store.load(old['actions'][0]['artifact_id'])['payload'], {'ok': 1})
        d = Stage('wd_full', {'image': 'new'}, model=self.spec)
        failed = execute(self.store, plan(self.store, [a,d]), {'wd_full': Mock(side_effect=RuntimeError('cancel'))})
        self.assertEqual([r['status'] for r in failed['actions']], ['reused','failed'])
        self.assertEqual(self.store.lookup(a.key)['status'], 'reused')


if __name__ == '__main__':
    unittest.main()
