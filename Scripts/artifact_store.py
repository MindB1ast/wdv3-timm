"""Versioned local artifact store. No model or UI imports; local filesystems only."""
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
import uuid
from typing import Any


class StoreError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def computation_key(stage, inputs, *, version=1):
    if not isinstance(stage, str) or not stage or type(version) is not int or version < 1:
        raise StoreError('Invalid stage/version')
    return sha256(canonical_bytes(dict(stage=stage, version=version, inputs=inputs)))


def check_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise StoreError('Invalid content identifier')
    return value


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class ArtifactStore:
    """SQLite metadata with content-addressed blobs and fenced expiring claims."""

    def __init__(self, dataset_root, store_path=None):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise StoreError('dataset_root must be an existing directory')
        self.root = Path(store_path).expanduser().resolve() if store_path else self.dataset_root / '.wdv3'
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'index.sqlite3'
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, artifact_id TEXT UNIQUE NOT NULL,
                    computation_key TEXT NOT NULL, stage TEXT NOT NULL, run_id TEXT NOT NULL,
                    created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS artifacts_key ON artifacts(computation_key, seq);
                CREATE TABLE IF NOT EXISTS claims (
                    computation_key TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    generation INTEGER NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                    state TEXT NOT NULL, result TEXT);
                CREATE TABLE IF NOT EXISTS entries (
                    run_id TEXT NOT NULL, position INTEGER NOT NULL, entry TEXT NOT NULL,
                    PRIMARY KEY(run_id, position));
                CREATE TABLE IF NOT EXISTS manifests (run_id TEXT PRIMARY KEY, manifest TEXT NOT NULL);
            ''')
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute("SELECT value FROM metadata WHERE name='store'").fetchone()
            if existing:
                metadata = json.loads(existing[0])
                if type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1:
                    raise StoreError('Unsupported store schema_version')
            else:
                metadata = dict(schema_version=1, store_id=uuid.uuid4().hex)
                db.execute("INSERT INTO metadata VALUES ('store', ?)", (canonical_bytes(metadata).decode(),))
            marker = self.root / 'store.json'
            if marker.exists() and json.loads(marker.read_bytes()) != metadata:
                raise StoreError('Store marker/index mismatch or unsupported version')
            atomic_write(marker, canonical_bytes(metadata))
        self.store_id = metadata['store_id']

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.db, timeout=30, isolation_level=None)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def blob_path(self, digest):
        return self.root / 'objects' / check_id(digest) / 'data'

    def put_blob(self, data):
        digest = sha256(data)
        path = self.blob_path(digest)
        if not path.exists() or sha256(path.read_bytes()) != digest:
            # Restore damaged storage to the very same content identity; never accept corrupt bytes.
            atomic_write(path, data)
        if sha256(path.read_bytes()) != digest:
            raise StoreError('Corrupt immutable blob: ' + digest)
        return digest

    def get_blob(self, digest):
        try:
            data = self.blob_path(digest).read_bytes()
        except OSError as exc:
            raise StoreError('Missing blob: ' + digest) from exc
        if sha256(data) != digest:
            raise StoreError('Checksum mismatch: ' + digest)
        return data

    def claim(self, key, *, lease_seconds=120, refresh=False):
        check_id(key)
        if lease_seconds <= 0:
            raise StoreError('lease_seconds must be positive')
        # Integrity is verified before considering a completed computation reusable.
        if not refresh:
            hit = self.lookup(key)
            if hit['status'] == 'reused':
                return hit
        with self.transaction() as db:
            if not refresh:
                row = db.execute('SELECT artifact_id FROM artifacts WHERE computation_key=? ORDER BY seq DESC LIMIT 1', (key,)).fetchone()
                if row:
                    try:
                        self.load(row[0])
                        return dict(status='reused', artifact_id=row[0])
                    except StoreError:
                        pass
            row = db.execute('SELECT owner,generation,expires FROM claims WHERE computation_key=?', (key,)).fetchone()
            now = time.time()
            if row and row[2] > now:
                return dict(status='waiting', computation_key=key, expires=row[2])
            token = dict(computation_key=key, owner=uuid.uuid4().hex,
                         generation=row[1] + 1 if row else 1)
            db.execute('INSERT OR REPLACE INTO claims VALUES (?,?,?,?)',
                       (key, token['owner'], token['generation'], now + lease_seconds))
            return dict(status='claimed', **token)

    @staticmethod
    def _assert_owner(db, claim):
        row = db.execute('SELECT owner,generation,expires FROM claims WHERE computation_key=?', (claim['computation_key'],)).fetchone()
        if not row or row[0] != claim['owner'] or row[1] != claim['generation'] or row[2] <= time.time():
            raise StoreError('Stale or expired owner (fencing)')

    def heartbeat(self, claim, *, lease_seconds=120):
        if lease_seconds <= 0:
            raise StoreError('lease_seconds must be positive')
        with self.transaction() as db:
            self._assert_owner(db, claim)
            db.execute('UPDATE claims SET expires=? WHERE computation_key=?',
                       (time.time() + lease_seconds, claim['computation_key']))

    def release(self, claim):
        with self.transaction() as db:
            self._assert_owner(db, claim)
            db.execute('UPDATE claims SET expires=0 WHERE computation_key=?', (claim['computation_key'],))

    def publish(self, claim, stage, payload, *, dependencies=(), blobs=None, run_id=None):
        refs = {name: self.put_blob(data) for name, data in (blobs or {}).items()}
        for dependency in dependencies:
            self.load(dependency)
        record = dict(schema_version=1, computation_key=claim['computation_key'], stage=stage,
                      run_id=run_id or uuid.uuid4().hex, generation=claim['generation'],
                      publication_id=uuid.uuid4().hex, dependencies=list(dependencies),
                      blobs=refs, payload=payload)
        artifact_id = self.put_blob(canonical_bytes(record))
        # All bytes must be durable and valid BEFORE making them visible in the index.
        self._verify(record)
        with self.transaction() as db:
            self._assert_owner(db, claim)
            db.execute('INSERT INTO artifacts(artifact_id,computation_key,stage,run_id,created) VALUES (?,?,?,?,?)',
                       (artifact_id, claim['computation_key'], stage, record['run_id'], time.time()))
            db.execute('UPDATE claims SET expires=0 WHERE computation_key=?', (claim['computation_key'],))
        return artifact_id

    def _verify(self, record, seen=None):
        if type(record.get('schema_version')) is not int or record['schema_version'] != 1:
            raise StoreError('Unsupported artifact schema_version')
        check_id(record['computation_key'])
        for ref in record['blobs'].values():
            self.get_blob(ref)
        seen = set() if seen is None else seen
        for ref in record['dependencies']:
            if ref in seen:
                continue
            seen.add(ref)
            self._verify(json.loads(self.get_blob(ref)), seen)

    def load(self, artifact_id):
        try:
            record = json.loads(self.get_blob(artifact_id))
            self._verify(record)
            return record
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise StoreError('Invalid artifact structure: ' + str(artifact_id)) from exc

    def lookup(self, key):
        with closing(sqlite3.connect(self.db)) as db:
            row = db.execute('SELECT artifact_id FROM artifacts WHERE computation_key=? ORDER BY seq DESC LIMIT 1', (check_id(key),)).fetchone()
        if not row:
            return dict(status='missing', reason='not_computed', computation_key=key)
        try:
            self.load(row[0])
            return dict(status='reused', reason='compatible', computation_key=key, artifact_id=row[0])
        except StoreError as exc:
            return dict(status='invalidated', reason='corrupt_artifact', computation_key=key, detail=str(exc))

    def begin_operation(self, operation_id, request) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id:
            raise StoreError('operation_id is required')
        request_hash = sha256(canonical_bytes(request))
        with self.transaction() as db:
            row = db.execute('SELECT request_hash,state,result FROM operations WHERE operation_id=?', (operation_id,)).fetchone()
            if row:
                if row[0] != request_hash:
                    raise StoreError('operation_id conflict: different request')
                return dict(state=row[1], result=json.loads(row[2]) if row[2] else None)
            db.execute('INSERT INTO operations VALUES (?,?,?,NULL)', (operation_id, request_hash, 'pending'))
        return dict(state='pending', result=None)

    def finish_operation(self, operation_id, result, *, state='complete'):
        with self.transaction() as db:
            changed = db.execute('UPDATE operations SET state=?,result=? WHERE operation_id=?',
                                 (state, canonical_bytes(result).decode(), operation_id)).rowcount
            if changed != 1:
                raise StoreError('Unknown operation_id')

    def write_manifest(self, entries, *, request=None, run_id=None):
        run_id = run_id or uuid.uuid4().hex
        if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
            raise StoreError('Invalid run_id')
        count = 0
        target = self.root / 'runs' / run_id / 'manifest.json'
        # Index streaming entries rather than retaining all images/arrays in a JSON list.
        with self.transaction() as db:
            if target.exists() or db.execute('SELECT 1 FROM manifests WHERE run_id=?', (run_id,)).fetchone():
                raise StoreError('Manifest run already exists')
            for count, entry in enumerate(entries, 1):
                relative = entry.get('relative_path')
                if relative is not None:
                    self.resolve_dataset_path(relative)
                db.execute('INSERT INTO entries VALUES (?,?,?)', (run_id, count-1, canonical_bytes(entry).decode()))
            manifest = dict(schema_version=1, run_id=run_id, store_id=self.store_id,
                            count=count, request=request or {}, entries='index.sqlite3:entries')
            encoded=canonical_bytes(manifest)
            atomic_write(target, encoded)
            db.execute('INSERT INTO manifests VALUES (?,?)',(run_id,encoded.decode()))
        return manifest

    def iter_entries(self, run_id, *, page_size=64):
        if type(page_size) is not int or page_size < 1:
            raise StoreError('page_size must be positive')
        if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
            raise StoreError('Invalid run_id')
        manifest = json.loads((self.root / 'runs' / run_id / 'manifest.json').read_bytes())
        if manifest.get('schema_version') != 1 or manifest.get('store_id') != self.store_id:
            raise StoreError('Unsupported/mismatched manifest')
        offset = 0
        with closing(sqlite3.connect(self.db)) as db:
            header=db.execute('SELECT manifest FROM manifests WHERE run_id=?',(run_id,)).fetchone()
            if not header or canonical_bytes(json.loads(header[0]))!=canonical_bytes(manifest):
                raise StoreError('Incomplete/corrupt manifest publication')
            while offset < manifest['count']:
                rows = db.execute('SELECT entry FROM entries WHERE run_id=? AND position>=? ORDER BY position LIMIT ?',
                                  (run_id, offset, page_size)).fetchall()
                if not rows:
                    raise StoreError('Incomplete manifest')
                for row in rows:
                    yield json.loads(row[0])
                offset += len(rows)

    def resolve_dataset_path(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise StoreError('Dataset path must be relative and inside dataset_root')
        resolved = (self.dataset_root / path).resolve()
        if not resolved.is_relative_to(self.dataset_root):
            raise StoreError('Dataset path escapes root through symlink')
        return resolved

    def scan(self, *, suffixes=('.png', '.jpg', '.jpeg', '.webp', '.bmp'), recursive=True):
        suffixes = {s.lower() for s in suffixes}
        for directory, names, files in os.walk(self.dataset_root, followlinks=False):
            names[:] = sorted(name for name in names if name != '.wdv3' and
                              not (Path(directory) / name).resolve().is_relative_to(self.root) and
                              not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() in suffixes and not path.resolve().is_relative_to(self.root):
                    self.resolve_dataset_path(path.relative_to(self.dataset_root))
                    yield path
            if not recursive:
                names.clear()
