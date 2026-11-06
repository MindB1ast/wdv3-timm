"""Versioned JSON-facing API shared by CLI and ComfyUI; no frontend dependency."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types
import uuid
from typing import Any

from .artifact_store import ArtifactStore, StoreError, computation_key
from .caption_files import (caption_document, commit_caption_export, document_from_tags,
                            plan_caption_export, target_path, export_config)
from .inference_plan import Stage, execute as execute_stages, format_snapshot, plan as plan_stages, policy_snapshot
from .inference_plan import file_digest
from .persistent_evidence import import_calculations, load_evidence, save_evidence


def runtime_module():
    # Standalone CLI loads the backend namespace without running custom-node __init__.
    # When called from ComfyUI, reuse its existing package and exact Bundle classes.
    assert __package__ is not None
    package = __package__.rsplit('.Scripts',1)[0] if '.Scripts' in __package__ else '_wdv3_backend'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules[package] = module
        # Share pure contracts with this API's Scripts package.
        sys.modules[package+'.Scripts'] = sys.modules[__package__]
        for name,module in list(sys.modules.items()):
            if name.startswith(__package__+'.'):
                sys.modules[package+'.'+name] = module
    return importlib.import_module(package+'.comfy_persistent_runtime')


def open_store(request):
    if not isinstance(request,dict) or type(request.get('schema_version',1)) is not int or request.get('schema_version',1) != 1:
        raise StoreError('Unsupported request schema_version')
    return ArtifactStore(request['dataset_root'],request.get('store_path'))


@contextmanager
def normalized_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as original, ImageOps.exif_transpose(original) as oriented:
        if oriented.mode == 'RGBA' or 'transparency' in oriented.info:
            with oriented.convert('RGBA') as rgba, Image.new('RGBA',oriented.size,'white') as white:
                white.alpha_composite(rgba)
                image = white.convert('RGB')
        else:
            image = oriented.convert('RGB')
        try:
            yield image
        finally:
            image.close()


def selected_files(store,request):
    if 'files' in request:
        return [store.resolve_dataset_path(p) for p in request['files']
                if '.wdv3' not in Path(p).parts and not store.resolve_dataset_path(p).is_relative_to(store.root)]
    return list(store.scan(recursive=request.get('recursive',True)))


def _skip_only(store, relative, request):
    if request.get('outputs') != ['captions']:
        return False
    config = export_config(**request.get('caption_export',{}))
    return config['write_mode']=='skip' and target_path(store,relative,config)[0].is_file()


def plan(request):
    store = open_store(request)
    backend = runtime_module()
    calculator = backend.PersistentCalculator(store,request['inference'],options=request.get('options'))
    entries=[]
    try:
        for path in selected_files(store,request):
            relative = path.relative_to(store.dataset_root).as_posix()
            if _skip_only(store,relative,request):
                entries.append(dict(relative_path=relative,status='skipped',reason='existing_caption_only',actions=[]))
                continue
            with normalized_image(path) as image:
                entry = calculator.plan_image(image,mode=request.get('mode','reuse'),refresh=request.get('refresh',[]))
            entries.append(dict(entry,relative_path=relative))
    finally:
        calculator.close()
    return dict(schema_version=1,entries=entries,ready=all(e.get('ready',e.get('status')=='skipped') for e in entries))


def execute(request, *, hooks=None) -> dict[str, Any]:
    store = open_store(request)
    operation_id = request.get('operation_id')
    if not operation_id:
        raise StoreError('execute requires stable operation_id')
    files=selected_files(store,request)
    operation_request=dict(request,dataset_inputs=[dict(relative_path=p.relative_to(store.dataset_root).as_posix(),
                                                       file_digest=file_digest(p)) for p in files])
    operation = store.begin_operation(operation_id,operation_request)
    if operation['state']=='complete':
        return operation['result']
    backend = runtime_module()
    calculator = backend.PersistentCalculator(store,request['inference'],options=request.get('options'),hooks=hooks)
    run_id=(operation['result'] or {}).get('run_id') or uuid.uuid4().hex
    report: dict[str, Any]=dict(schema_version=1,operation_id=operation_id,run_id=run_id,entries=[])
    previous={e['relative_path']:e for e in (operation['result'] or {}).get('entries',[])}
    try:
        for path in files:
            relative=path.relative_to(store.dataset_root).as_posix()
            if previous.get(relative,{}).get('status')=='complete':
                prior=previous[relative]
                try:store.load(prior['evidence_ref'])
                except StoreError:pass
                else:
                    report['entries'].append(prior)
                    continue
            if _skip_only(store,relative,request):
                row=dict(relative_path=relative,status='skipped',reason='existing_caption_only')
            else:
                with normalized_image(path) as image:
                    evidence,output=calculator.compute_image(image,mode=request.get('mode','reuse'),
                        refresh=request.get('refresh',[]),relative_path=relative,run_id=run_id)
                row=dict(relative_path=relative,**output)
                if evidence is not None:
                    row['status']='complete'
                else:
                    row['status']='incomplete'
            report['entries'].append(row)
            store.finish_operation(operation_id,report,state='pending')
        report['complete']=all(row['status'] in ('skipped','complete') for row in report['entries'])
        report['counters']=deepcopy(calculator.counters)
        manifest=store.write_manifest((dict(relative_path=r['relative_path'],artifact_id=r['evidence_ref'])
                                      for r in report['entries'] if r.get('evidence_ref')),request=request)
        report['manifest_run_id']=manifest['run_id']
        store.finish_operation(operation_id,report,state='complete' if report['complete'] else 'partial')
        return report
    finally:
        calculator.close()


def status(request):
    store = open_store(request)
    from contextlib import closing
    import sqlite3
    with closing(sqlite3.connect(store.db)) as db:
        row=db.execute('SELECT state,result FROM operations WHERE operation_id=?',(request['operation_id'],)).fetchone()
    return dict(schema_version=1,state=row[0],result=json.loads(row[1]) if row[1] else None) if row else dict(schema_version=1,state='unknown')


def load(request):
    store=open_store(request)
    from itertools import islice
    refs=request.get('artifact_ids') or ([request['artifact_id']] if request.get('artifact_id') else
          [e['artifact_id'] for e in islice(store.iter_entries(request['run_id'],page_size=request.get('page_size',64)),
                                           request.get('start',0),request.get('start',0)+request.get('limit',64))])
    frames=[]; model=None
    for ref in refs:
        data=load_evidence(store,ref).data()
        if model is not None and data['model']!=model:
            raise StoreError('Cannot combine evidence from different models')
        model=data['model']
        for frame in data['frames']:
            frame['image_index']=len(frames)
            if frame.get('relative_path'):
                frame['image_path']=str(store.resolve_dataset_path(frame['relative_path']))
            frames.append(frame)
    from .evidence import Bundle
    if not frames:
        raise StoreError('No selected evidence frames')
    return Bundle.pack('evidence',dict(model=model,frames=frames))


def refilter(request):
    """Policy and formatter artifacts are independent; raw evidence remains immutable."""
    store=open_store(request)
    evidence=load(request)
    ref=save_evidence(store,evidence)
    options=request.get('options',{})
    from .inference_plan import RESOURCE_FIELDS, FORMAT_FIELDS
    policy=policy_snapshot(options,request.get('sources'),request.get('postprocessing'))
    stage=Stage('policy',dict(evidence_ref=ref,policy=policy))
    def filter_policy(action,deps):
        backend=runtime_module()
        assert backend.__package__ is not None
        module=importlib.import_module(backend.__package__+'.comfy_evidence')
        from .tag_postprocessing import apply_chain
        # Never use text formatting settings to compute the policy version.
        configs=[module.DetectorConfig.from_dict(s) for s in policy['sources']] if policy['sources'] else None
        results=module.merge_tag_sets(apply_chain(module.filter_tag_sets(evidence,policy['options'],configs),policy['postprocessing']))
        return {'payload':json.loads(results.to_json()),'artifact_dependencies':[ref]}
    policy_report=execute_stages(store,plan_stages(store,[stage],mode='reuse_only'),{'policy':filter_policy})
    if not policy_report['complete']:
        raise StoreError(json.dumps(policy_report,ensure_ascii=False))
    policy_ref=policy_report['actions'][0]['artifact_id']
    from .evidence import Bundle
    payload=deepcopy(store.load(policy_ref)['payload'])
    result=Bundle.from_json(json.dumps(payload))
    data=result.data()
    for frame in data['frames']:
        frame['settings']['options'].update(options)
    result=Bundle.pack('results',data)
    text_stage=Stage('text',dict(policy_ref=policy_ref,format=format_snapshot(options)))
    def render(action,deps):
        backend=runtime_module()
        assert backend.__package__ is not None
        module=importlib.import_module(backend.__package__+'.comfy_evidence')
        tags,_=module.outputs(result)
        return {'payload':{'tags':tags},'artifact_dependencies':[policy_ref]}
    text_report=execute_stages(store,plan_stages(store,[text_stage],mode='reuse_only'),{'text':render})
    if not text_report['complete']:
        raise StoreError(json.dumps(text_report,ensure_ascii=False))
    return dict(schema_version=1,results=json.loads(result.to_json()),
                tags=store.load(text_report['actions'][0]['artifact_id'])['payload']['tags'],
                policy_report=policy_report,text_report=text_report)


def captions_from_results(results):
    data=results.data() if hasattr(results,'data') else results['payload']
    entries=[]
    for frame in data['frames']:
        merged=frame['merged_tags']
        names=parse_result_names(merged)
        doc=document_from_tags(names,image_path=frame.get('image_path'))
        for tag in names:
            doc['provenance'][tag]=[dict(source='predictions',image_id=frame['image_id'],
                                       origin=merged.get('provenance',{}).get(tag))]
        relative=frame.get('relative_path')
        if relative is None:
            raise StoreError('Caption export requires explicit dataset-relative image mapping')
        entries.append(dict(relative_path=relative,document=doc))
    return entries


def parse_result_names(merged):
    from .tag_names import parse_tags
    names=set(merged.get('general',{})) | set(merged.get('character',{}))
    ordered=[n for n in parse_tags(merged.get('caption','')) if n in names]
    return ordered+sorted(names-set(ordered))


def dispatch(command,request) -> dict[str, Any]:
    if command=='plan': return plan(request)
    if command=='execute': return execute(request)
    if command=='status': return status(request)
    if command=='load': return json.loads(load(request).to_json())
    if command=='refilter': return refilter(request)
    if command=='import': return import_calculations(open_store(request),request['calculations'])
    if command=='caption-plan':
        return plan_caption_export(open_store(request),request['entries'],operation_id=request['operation_id'],config=request.get('caption_export'))
    if command=='caption-commit': return commit_caption_export(open_store(request),request['plan'])
    raise StoreError('Unknown workflow command')
