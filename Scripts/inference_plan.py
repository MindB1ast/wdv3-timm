"""Content identities and a loader-free DAG planner for persistent calculations."""
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError,version as package_version
from pathlib import Path
import threading
import uuid
from typing import Any

from .artifact_store import StoreError, canonical_bytes, computation_key, sha256

FORMAT_FIELDS = ('tag_separator', 'replace_underscores', 'escape_parentheses')
RESOURCE_FIELDS = ('image_batch_size', 'region_batch_size', 'sam_batch_size',
                   'cache_mode','cache_dataset_root','cache_refresh')


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_identity(image):
    return computation_key('pixels', dict(size=list(image.size), mode=image.mode,
                           pixels=sha256(image.tobytes()), decode='exif-rgb-white-alpha-v1'))


def model_identity(spec):
    """Resolve only explicit local files or an explicit pin; never call a loader."""
    files = spec.get('files', {})
    local_ready=bool(files) and all(Path(path).is_file() for path in files.values())
    if spec.get('identity') and (spec.get('_resolved_identity') or not local_ready):
        identity = spec['identity']
        if not isinstance(identity, dict) or not identity.get('weights_sha256'):
            raise StoreError('Invalid pinned model identity')
        return {**identity,**{field:spec[field] for field in ('compute','adapter','profile') if field in spec}}
    if not files or not any('weight' in name for name in files):
        return None
    if not all(Path(path).is_file() for path in files.values()):
        return None
    hashes={name:file_digest(path) for name,path in sorted(files.items())}
    versions={}
    for package in ('torch','timm','ultralytics','transformers','Pillow'):
        try:versions[package]=package_version(package)
        except PackageNotFoundError:versions[package]=None
    return dict(files=hashes,
                weights_sha256=sha256(canonical_bytes({name:value for name,value in hashes.items() if 'weight' in name})),
                adapter=spec.get('adapter', 'wdv3-v1'), profile=spec.get('profile', {}),
                compute=spec.get('compute', {}),engines=versions)


def policy_snapshot(options=None, sources=None, postprocessing=None):
    options = options or {}
    return dict(version=1, options={k: v for k, v in options.items()
                                  if k not in FORMAT_FIELDS + RESOURCE_FIELDS},
                sources=sources or [], postprocessing=postprocessing or [])


def format_snapshot(options=None):
    options = options or {}
    return dict(version=1, tag_separator=options.get('tag_separator', ', '),
                replace_underscores=options.get('replace_underscores', True),
                escape_parentheses=options.get('escape_parentheses', True))


@dataclass(frozen=True)
class Stage:
    stage: str
    inputs: dict
    dependencies: tuple = ()
    source: str | None = None
    image_id: str | None = None
    model: dict | None = None
    version: int = 1
    key: str = field(init=False)

    def __post_init__(self):
        identity = model_identity(self.model) if self.model is not None else None
        object.__setattr__(self, 'key', computation_key(self.stage, dict(inputs=self.inputs,
                           dependencies=list(self.dependencies), model=identity), version=self.version))

    def data(self):
        return dict(stage=self.stage, inputs=self.inputs, dependencies=list(self.dependencies),
                    source=self.source, image_id=self.image_id, model=self.model,
                    version=self.version, computation_key=self.key)


def plan(store, stages, *, mode='reuse', refresh=()):
    if mode not in ('reuse', 'reuse_only', 'refresh'):
        raise StoreError('Unknown inference mode')
    stages = list(stages)
    keys = set()
    actions = []
    invalidated = set()
    for stage in stages:
        if stage.key in keys:
            continue
        if not set(stage.dependencies).issubset(keys):
            raise StoreError('Stages must be an ordered DAG; dependency missing')
        keys.add(stage.key)
        selected = mode == 'refresh' and (not refresh or stage.stage in refresh or stage.source in refresh or stage.key in refresh)
        dependent = bool(set(stage.dependencies) & invalidated)
        action = dict(stage.data(), **{k:v for k,v in store.lookup(stage.key).items() if k != 'computation_key'})
        if stage.model is not None and model_identity(stage.model) is None:
            action.update(status='missing', reason='unresolved_identity')
        elif selected or dependent:
            action.update(status='invalidated', reason='refresh_selected' if selected else 'dependency_refreshed')
            action.pop('artifact_id', None)
        if action['status'] != 'reused':
            invalidated.add(stage.key)
        actions.append(action)
    return dict(schema_version=1, mode=mode, refresh=list(refresh), actions=actions,
                ready=all(a['status'] == 'reused' for a in actions))


class LeaseHeartbeat:
    def __init__(self, store, claim, seconds=120):
        self.store, self.claim, self.seconds = store, claim, seconds
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(self.seconds / 3):
            try:
                self.store.heartbeat(self.claim, lease_seconds=self.seconds)
            except Exception as exc:
                self.error = exc
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def execute(store, calculation_plan, executors, *, run_id=None) -> dict[str, Any]:
    """Executors are lazy callables(action, dependency records), invoked only on misses."""
    if calculation_plan.get('schema_version') != 1:
        raise StoreError('Unsupported plan version')
    run_id = run_id or uuid.uuid4().hex
    report: dict[str, Any] = dict(schema_version=1, run_id=run_id, actions=[])
    refs = {}
    for action in calculation_plan['actions']:
        row = {k:v for k,v in action.items() if k not in ('inputs', 'model')}
        key = action['computation_key']
        missing_deps = [key for key in action['dependencies'] if key not in refs]
        if missing_deps:
            row.update(status='missing', reason='dependency_missing', missing=missing_deps)
        elif action['reason'] == 'unresolved_identity':
            row.update(status='missing', reason='unresolved_identity')
        else:
            hit = store.lookup(key)
            force = action['reason'] in ('refresh_selected', 'dependency_refreshed')
            if not force and hit['status'] == 'reused':
                row.update(hit)
                refs[key] = hit['artifact_id']
            elif calculation_plan['mode'] == 'reuse_only' and action['stage'] not in ('policy', 'text', 'evidence', 'crop'):
                row.update(hit)
            elif action['stage'] not in executors:
                row.update(status='missing', reason='executor_unavailable')
            else:
                claim = store.claim(key, refresh=force)
                if claim['status'] == 'reused':
                    row.update(claim)
                    refs[key] = claim['artifact_id']
                elif claim['status'] == 'waiting':
                    row.update(status='waiting', reason='active_owner')
                else:
                    try:
                        deps = [store.load(refs[d]) for d in action['dependencies']]
                        with LeaseHeartbeat(store, claim) as lease:
                            value = executors[action['stage']](action, deps)
                            if lease.error:
                                raise lease.error
                            ref = store.publish(claim, action['stage'], value.get('payload', value),
                                                dependencies=[refs[d] for d in action['dependencies']]+value.get('artifact_dependencies',[]),
                                                blobs=value.get('blobs'), run_id=run_id)
                        refs[key] = ref
                        row.update(status='computed', reason='refresh' if force else 'cache_miss', artifact_id=ref)
                    except Exception as exc:
                        try:
                            store.release(claim)
                        except StoreError:
                            pass
                        row.update(status='failed', reason=type(exc).__name__, detail=str(exc))
        row['parent_refs'] = [refs[d] for d in action['dependencies'] if d in refs]
        report['actions'].append(row)
    report['complete'] = len(refs) == len(calculation_plan['actions'])
    return report
