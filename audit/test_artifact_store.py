"""Storage acceptance in fresh processes, including fencing after owner death."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from Scripts.artifact_store import ArtifactStore, StoreError, computation_key


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='temp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = ArtifactStore(self.root)
        self.key = computation_key('wd', {'pixels': 'abc', 'model': 'def'})

    def child(self, code, *args):
        result = subprocess.run([sys.executable, '-c', code, str(self.root), *map(str, args)],
                                capture_output=True, text=True, timeout=20,
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def publish(self):
        claim = self.store.claim(self.key)
        # Lossless PNG used as both image and mask; no model/Pillow imports needed.
        data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jBv0AAAAASUVORK5CYII=')
        raw = {'general': {'skirt': .01, 'black_skirt': .99}, 'character': {}, 'ratings': {'safe': .9}}
        return self.store.publish(claim, 'wd', raw, blobs={'image.png': data, 'mask.png': data})

    def test_keys_and_versions(self):
        self.assertEqual(computation_key('wd', {'a': 1, 'b': 2}), computation_key('wd', {'b': 2, 'a': 1}))
        self.assertNotEqual(computation_key('wd', {'a': 1}), computation_key('wd', {'a': 2}))
        with self.assertRaises(ValueError):
            computation_key('wd', {'bad': float('nan')})
        metadata = self.store.root / 'store.json'
        data = json.loads(metadata.read_bytes()); data['schema_version'] = 2
        metadata.write_text(json.dumps(data))
        with self.assertRaises(StoreError):
            ArtifactStore(self.root)

    def test_roundtrip_in_other_process_and_corruption(self):
        ref = self.publish()
        record = self.child('''import sys,json
from Scripts.artifact_store import ArtifactStore
s=ArtifactStore(sys.argv[1]); r=s.load(sys.argv[2])
print(json.dumps(dict(record=r, png=s.get_blob(r['blobs']['image.png']).hex())))''', ref)
        self.assertEqual(record['record']['payload']['general']['skirt'], .01)
        self.assertTrue(record['png'].startswith('89504e47'))
        blob = record['record']['blobs']['mask.png']
        self.store.blob_path(blob).write_bytes(b'truncated')
        self.assertEqual(self.store.lookup(self.key)['status'], 'invalidated')
        with self.assertRaises(StoreError):
            self.store.load(ref)
        # Unindexed, complete-looking bytes must never be a hit.
        other = computation_key('wd', {'other': 1})
        self.store.put_blob(b'{"complete":true}')
        self.assertEqual(self.store.lookup(other)['status'], 'missing')

    def test_claim_wait_heartbeat_recovery_and_fencing(self):
        first = self.child('''import sys,json
from Scripts.artifact_store import ArtifactStore
print(json.dumps(ArtifactStore(sys.argv[1]).claim(sys.argv[2],lease_seconds=120)))''', self.key)
        second = self.child('''import sys,json
from Scripts.artifact_store import ArtifactStore
print(json.dumps(ArtifactStore(sys.argv[1]).claim(sys.argv[2])))''', self.key)
        self.assertEqual(first['status'], 'claimed')
        self.assertEqual(second['status'], 'waiting')
        self.store.heartbeat(first)
        # Simulate expired owner after process exit without wall-clock sleeps.
        with self.store.transaction() as db:
            db.execute('UPDATE claims SET expires=0')
        takeover = self.child('''import sys,json
from Scripts.artifact_store import ArtifactStore
s=ArtifactStore(sys.argv[1]); c=s.claim(sys.argv[2]); ref=s.publish(c,'wd',{'ok':True})
print(json.dumps(dict(claim=c,ref=ref)))''', self.key)
        self.assertGreater(takeover['claim']['generation'], first['generation'])
        with self.assertRaises(StoreError):
            self.store.publish(first, 'wd', {'stale': True})
        self.assertEqual(self.store.claim(self.key)['artifact_id'], takeover['ref'])

    def test_simultaneous_owners(self):
        code = '''import sys,json
from Scripts.artifact_store import ArtifactStore
print(json.dumps(ArtifactStore(sys.argv[1]).claim(sys.argv[2])))'''
        children = [subprocess.Popen([sys.executable, '-c', code, str(self.root), self.key],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
        statuses = []
        for child in children:
            out, err = child.communicate(timeout=20)
            self.assertEqual(child.returncode, 0, err)
            statuses.append(json.loads(out)['status'])
        self.assertCountEqual(statuses, ['claimed', 'waiting'])

    def test_versions_dependencies_and_operations(self):
        first = self.publish()
        second = self.store.publish(self.store.claim(self.key, refresh=True), 'wd', {'new': True}, dependencies=[first])
        self.assertNotEqual(first, second)
        self.assertIn('general', self.store.load(first)['payload'])
        self.assertEqual(self.store.lookup(self.key)['artifact_id'], second)
        self.store.begin_operation('a', {'foo': 1})
        self.store.finish_operation('a', {'artifact_id': second})
        self.assertEqual(self.store.begin_operation('a', {'foo': 1})['state'], 'complete')
        with self.assertRaises(StoreError):
            self.store.begin_operation('a', {'foo': 2})
        self.store.blob_path(first).unlink()
        self.assertEqual(self.store.lookup(self.key)['status'], 'invalidated')

    def test_paged_manifest_and_moving_dataset(self):
        (self.root / 'a.png').write_bytes(b'image')
        (self.store.root / 'mask.png').write_bytes(b'mask')
        (self.root / 'nested').mkdir()
        (self.root / 'nested' / 'b.png').write_bytes(b'image')
        ref = self.publish()
        old_record = self.store.load(ref)
        manifest = self.store.write_manifest((dict(relative_path=f'{i}.png', artifact_id=ref) for i in range(130)))
        self.assertEqual(len(list(self.store.iter_entries(manifest['run_id'], page_size=7))), 130)
        self.assertEqual(len(list(self.store.scan())), 2)
        moved = self.root / 'moved'
        moved.mkdir()
        shutil.move(str(self.store.root), str(moved / '.wdv3'))
        (self.root / 'a.png').rename(moved / 'a.png')
        new = ArtifactStore(moved)
        self.assertEqual(new.load(ref), old_record)
        self.assertEqual(len(list(new.iter_entries(manifest['run_id'], page_size=1))), 130)
        self.assertEqual(new.resolve_dataset_path('a.png').read_bytes(), b'image')
        with self.assertRaises(StoreError):
            new.resolve_dataset_path('../outside.png')

    def test_custom_store_excluded_for_images_and_txt(self):
        custom = ArtifactStore(self.root, self.root / 'cache')
        (custom.root / 'mask.png').write_bytes(b'mask')
        (custom.root / 'service.txt').write_bytes(b'caption')
        (self.root / '.wdv3' / 'service.txt').write_bytes(b'caption')
        (self.root / 'a.txt').write_bytes(b'caption')
        self.assertEqual([p.name for p in custom.scan(suffixes=['.txt'])], ['a.txt'])
        self.assertEqual(list(custom.scan()), [])


if __name__ == '__main__':
    unittest.main()
