"""Lazy persistent WD/YOLO/SAM stages; shared by UI and the standalone API."""
import base64
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from .Scripts.artifact_store import StoreError
from .Scripts.evidence import Bundle
from .Scripts.inference_plan import Stage, execute, model_identity, pixel_identity, plan, LeaseHeartbeat
from .Scripts.persistent_evidence import save_evidence, load_evidence


def normalize_inference(config):
    if not isinstance(config,dict) or config.get('schema_version',1) != 1:
        raise StoreError('Unsupported inference config')
    data = deepcopy(config)
    if not data.get('wd'):
        raise StoreError('Inference Settings requires a WD model specification')
    sources = data.setdefault('sources', [])
    names = [s['name'] for s in sources]
    if len(set(names)) != len(names) or 'full' in names:
        raise StoreError('Source names must be unique and not full')
    for source in sources:
        if source.get('kind') not in ('yolo','sam') or not source.get('model'):
            raise StoreError('Each source needs kind yolo/sam and a lazy model specification')
        from .Scripts.sam_config import finite_number,RegionViewConfig
        finite_number(source.setdefault('confidence',.25 if source['kind']=='yolo' else .5),'confidence',maximum=1)
        RegionViewConfig(mode=source.setdefault('view_mode','bbox'),padding_value=source.setdefault('padding_px',0),
                         padding_unit=source.get('padding_unit','px'),background_rgb=tuple(source.get('background_rgb',[255,255,255])))
        if source['kind']=='sam':
            source.setdefault('mask_threshold',.5);source.setdefault('min_mask_area_px',1);source.setdefault('max_regions',64)
            source.setdefault('rules',{})
            from .comfy_config import validate_config
            validate_config(dict(schema_version=1,sam_sources=[source]))
        else:
            source.setdefault('classes',[0])
            if source['classes'] is not None and (not isinstance(source['classes'],list) or any(type(c) is not int or c<0 for c in source['classes'])):
                raise StoreError('YOLO classes must be nonnegative integer IDs or null')
    for spec in [data['wd']]+[s['model'] for s in sources]:
        spec.pop('_resolved_identity',None)
    return data


def geometry_inputs(source):
    excluded = {'name','model','rules','view_mode','padding_px','padding_unit','background_rgb'}
    return {k:v for k,v in source.items() if k not in excluded}


def _png(image):
    buffer = BytesIO(); image.save(buffer,format='PNG'); return buffer.getvalue()


class PersistentCalculator:
    def __init__(self, store, inference, *, options=None, hooks=None):
        self.store, self.inference = store, normalize_inference(inference)
        # Hash local resources once per request, retaining no model object during planning.
        for spec in [self.inference['wd']]+[s['model'] for s in self.inference['sources']]:
            identity=model_identity(spec)
            if identity is not None:
                spec['identity']=identity
                spec['_resolved_identity']=True
        self.options, self.hooks = options or {}, hooks or {}
        self.wd = None
        self.models = {}
        self.images = {}
        self.crops = {}
        self.counters = dict(wd_load=0, wd_inference=0, yolo_load=0, yolo_inference=0, sam_inference=0)

    def _wd_scores(self, images):
        from .comfy_evidence import interrupt
        interrupt()
        if 'wd' in self.hooks:
            self.counters['wd_inference'] += len(images)
            return self.hooks['wd'](images)
        from .Scripts.tagger_resources import load_tagger
        from .Scripts.region_tagging import RegionTagger
        import torch
        if self.wd is None:
            spec = self.inference['wd']
            path = spec.get('local_path')
            if not path:
                weight = next((v for k,v in spec.get('files', {}).items() if 'weight' in k), None)
                path = str(Path(weight).parent) if weight else None
            model, labels, transform, _ = load_tagger(spec.get('model_id','vit'), Path(path).parent if path else Path('models/taggers'),
                                                     allow_download=False, local_path=Path(path) if path else None)
            device = spec.get('compute', {}).get('device', 'cpu')
            self.wd = RegionTagger(model.eval(), transform, labels, torch.device(device))
            self.counters['wd_load'] += 1
        self.wd.model.to(self.wd.device)
        try:
            self.counters['wd_inference'] += len(images)
            results = self.wd.score_batch(images)
            interrupt()
            for row in results:
                row.pop('caption',None); row.pop('taglist',None)
                for category in ('general','character','ratings'):
                    row[category]={tag:float(score) for tag,score in row[category].items()}
            return results
        finally:
            self.wd.model.to('cpu')

    def _geometry(self, action, deps):
        from .comfy_evidence import interrupt
        interrupt()
        source = next(s for s in self.inference['sources'] if s['name'] == action['source'])
        image = self.images[action['image_id']]
        if source['kind']=='sam' and self.options.get('low_memory_mode',True):
            if self.wd is not None:
                self.wd=None
                import gc
                gc.collect()
            if 'before_sam' in self.hooks:
                self.hooks['before_sam']()
        if source['kind'] in self.hooks:
            self.counters[source['kind'] + '_inference'] += 1
            return self.hooks[source['kind']](image, source)
        if source['kind'] == 'sam':
            from .comfy_processing import segment_frames
            import numpy as np
            from PIL import Image
            model = source['model']
            settings = model['settings']
            sam_source = {k:v for k,v in source.items() if k not in ('kind','model')}
            sam_source.setdefault('rules', {})
            record = segment_frames([image],settings,[sam_source],1)[0]
            rows, blobs = [], {}
            for i, row in enumerate(record['regions'][source['name']]):
                row = {k:v for k,v in row.items() if k != 'mask'}
                with Image.fromarray(np.asarray(record['regions'][source['name']][i]['mask'],dtype=np.uint8)*255) as mask:
                    name = f'mask-{i}'
                    blobs[name] = _png(mask)
                rows.append(dict(row, mask_blob=self.store.put_blob(blobs[name])))
            self.counters['sam_inference'] += 1
            return dict(payload={'regions':rows}, blobs=blobs)
        from .Scripts.model import load_yolo_model
        from .Scripts.regions import rectangle_geometry
        spec = source['model']
        path = spec.get('local_path') or next(v for k,v in spec['files'].items() if 'weight' in k)
        if path not in self.models:
            self.models[path] = load_yolo_model(path,Path(path).parent,allow_download=False)
            self.counters['yolo_load'] += 1
        model = self.models[path]
        try:
            results = model(image,conf=source.get('confidence',.25),classes=source.get('classes'),
                            device=spec.get('compute',{}).get('device','cpu'),verbose=False)
            rows = []
            for result in results:
                if getattr(result,'boxes',None) is None:
                    continue
                for box in result.boxes:
                    geometry = rectangle_geometry(action['image_id'],source['name'],len(rows),image.size,box.xyxy[0].tolist())
                    rows.append(dict(region_id=len(rows),bbox=geometry['bbox'],
                                     class_id=int(box.cls[0]),confidence=float(box.conf[0])))
            self.counters['yolo_inference'] += 1
            return {'regions':rows}
        finally:
            model.to('cpu')

    def _score(self, action, deps):
        image = self.images[action['image_id']] if action['stage'] == 'wd_full' else self.crops[action['computation_key']]
        return self._wd_scores([image])[0]

    def _execute_crops(self,calculation_plan,run_id):
        report=dict(schema_version=1,run_id=run_id,actions=[])
        pending=[]
        batch_size=self.options.get('region_batch_size',8)
        if type(batch_size) is not int or batch_size<1:
            raise StoreError('region_batch_size must be a positive integer')
        def flush():
            if not pending:return
            try:
                with ExitStack() as stack:
                    leases=[stack.enter_context(LeaseHeartbeat(self.store,claim)) for _,claim in pending]
                    rows=self._wd_scores([self.crops[action['computation_key']] for action,_ in pending])
                    if len(rows)!=len(pending):raise StoreError('WD returned wrong batch length')
                    for (action,claim),raw,lease in zip(pending,rows,leases):
                        if lease.error:raise lease.error
                        ref=self.store.publish(claim,'wd_crop',raw,run_id=run_id)
                        report['actions'].append(dict(action,status='computed',reason='cache_miss',artifact_id=ref,parent_refs=[]))
            except Exception as exc:
                completed={a['computation_key'] for a in report['actions'] if a.get('artifact_id')}
                for action,claim in pending:
                    if action['computation_key'] in completed:continue
                    try:self.store.release(claim)
                    except StoreError:pass
                    report['actions'].append(dict(action,status='failed',reason=type(exc).__name__,detail=str(exc)))
            finally:pending.clear()
        for action in calculation_plan['actions']:
            key=action['computation_key']
            force=action['reason'] in ('refresh_selected','dependency_refreshed')
            hit=self.store.lookup(key)
            if not force and hit['status']=='reused':
                report['actions'].append(dict(action,**{k:v for k,v in hit.items() if k!='computation_key'}))
            elif calculation_plan['mode']=='reuse_only' or action['reason']=='unresolved_identity':
                report['actions'].append(dict(action,status=hit['status'],reason=action['reason']))
            else:
                claim=self.store.claim(key,refresh=force)
                if claim['status']=='claimed':
                    pending.append((action,claim))
                    if len(pending)>=batch_size:flush()
                else:
                    report['actions'].append(dict(action,**{k:v for k,v in claim.items() if k!='computation_key'}))
        flush()
        report['complete']=all(a['status'] in ('computed','reused') for a in report['actions'])
        return report

    def _stages(self, image):
        identity = pixel_identity(image)
        self.images[identity] = image
        stages = [Stage('wd_full',dict(pixels=identity),image_id=identity,model=self.inference['wd'])]
        metadata = {}
        pending = []
        from PIL import Image
        from .Scripts.regions import prepare_region_view
        from .Scripts.sam_config import RegionViewConfig
        for source in self.inference['sources']:
            stage = Stage(source['kind']+'_geometry',dict(pixels=identity,geometry=geometry_inputs(source)),
                          image_id=identity,source=source['name'],model=source['model'])
            stages.append(stage)
            hit = self.store.lookup(stage.key)
            if hit['status'] != 'reused':
                pending.append(dict(stage='wd_regions',source=source['name'],status='missing',reason='geometry_not_available',image_id=identity))
                continue
            record = self.store.load(hit['artifact_id'])
            local = []
            for index, row in enumerate(record['payload']['regions']):
                mask = None
                if row.get('mask_blob'):
                    import numpy as np
                    with Image.open(BytesIO(self.store.get_blob(row['mask_blob']))) as asset:
                        mask = np.array(asset) != 0
                view = RegionViewConfig(mode=source.get('view_mode','bbox'),padding_value=source.get('padding_px',0),
                                        padding_unit=source.get('padding_unit','px'),background_rgb=tuple(source.get('background_rgb',[255,255,255])))
                crop, crop_meta = prepare_region_view(image,dict(image_size=list(image.size),bbox=row['bbox']),view,mask)
                # Score identity intentionally excludes source/region/geometry: actual crop pixels are sufficient.
                crop_stage = Stage('wd_crop',dict(pixels=pixel_identity(crop),crop_policy=1), image_id=identity,
                                   model=self.inference['wd'])
                previous = self.crops.pop(crop_stage.key,None)
                if previous is not None:
                    previous.close()
                self.crops[crop_stage.key] = crop
                stages.append(crop_stage)
                local.append((index,row,crop_meta,crop_stage.key))
            metadata[source['name']] = (stage.key,local)
        return stages, metadata, pending

    def plan_image(self, image, *, mode='reuse', refresh=()):
        stages, _, pending = self._stages(image)
        result = plan(self.store,stages,mode=mode,refresh=refresh)
        result['pending_regions'] = pending
        result['ready'] = result['ready'] and not pending
        return result

    def compute_image(self, image, *, mode='reuse', refresh=(), relative_path=None, run_id=None):
        from .comfy_evidence import digest, geometry_config
        from .Scripts.configs import DetectorConfig
        executors = {'wd_full':self._score,'wd_crop':self._score,
                     'yolo_geometry':self._geometry,'sam_geometry':self._geometry}
        stages, _, _ = self._stages(image)
        # Publish geometry independently first; discover exact crop pixels without another model call.
        first_stages=[s for s in stages if s.stage != 'wd_crop']
        # Geometry has no WD dependency. Finish SAM (and release its process)
        # before allocating WD weights, including on later dataset images.
        first_stages.sort(key=lambda stage: stage.stage=='wd_full')
        first = execute(self.store,plan(self.store,first_stages,mode=mode,refresh=refresh),executors,run_id=run_id)
        if not first['complete']:
            return None, first
        stages, metadata, pending = self._stages(image)
        crop_refresh = []
        if mode=='refresh':
            if 'wd_crop' in refresh or 'wd_full' in refresh or not refresh:
                crop_refresh=[s.key for s in stages if s.stage=='wd_crop']
            else:
                crop_refresh=[key for source,(_,rows) in metadata.items() if source in refresh
                              for _,_,_,key in rows]
        second = self._execute_crops(plan(self.store,[s for s in stages if s.stage=='wd_crop'],
                         mode='refresh' if crop_refresh else ('reuse_only' if mode=='reuse_only' else 'reuse'),refresh=crop_refresh),run_id=first['run_id'])
        report = dict(first,actions=first['actions']+second['actions'],complete=second['complete'])
        if pending or not second['complete']:
            return None, report
        identity = pixel_identity(image)
        full_ref = self.store.lookup(stages[0].key)['artifact_id']
        full = self.store.load(full_ref)['payload']
        settings, detectors, sam_geometry, yolo_geometry = [], {}, [], []
        for source in self.inference['sources']:
            config = DetectorConfig(source['name'],'sam3' if source['kind']=='sam' else source['model'].get('local_path','persistent-yolo'),
                                    **source.get('rules',{}))
            if source['kind']=='yolo':
                config.confidence=source.get('confidence',.25); config.classes=source.get('classes',[0])
                yolo_geometry.append(dict(geometry_config(config),model_identity=model_identity(source['model'])))
            else:
                sam_geometry.append(dict({k:v for k,v in source.items() if k not in ('rules','model','kind')},
                                         model_identity=model_identity(source['model'])))
            settings.append(asdict(config))
            rows=[]
            for i,row,view,key in metadata[source['name']][1]:
                scores = self.store.load(self.store.lookup(key)['artifact_id'])['payload']
                out = dict(row,region_id=i,view=view,**scores)
                if row.get('mask_blob'):
                    out['mask_png']=base64.b64encode(self.store.get_blob(row['mask_blob'])).decode()
                rows.append(out)
            detectors[source['name']]=rows
        categories={c:sorted(full[c]) for c in ('general','character','ratings')}
        model=dict(model_id=self.inference['wd'].get('model_id','custom-wd'),identity=model_identity(self.inference['wd']),
                   categories=categories,labels_count=sum(map(len,categories.values())),labels_sha256=digest(categories))
        frame=dict(image_id=digest([__import__('hashlib').sha256(image.tobytes()).hexdigest(),list(image.size)]),
                   pixel_identity=identity,image_size=list(image.size),image_png=base64.b64encode(_png(image)).decode(),
                   full_image=full,detectors=detectors,source_settings=settings,
                   geometry=dict(yolo=yolo_geometry,sam=sam_geometry,
                                 sam_model={s['name']:dict(identity=model_identity(s['model']),settings=s['model'].get('settings'))
                                            for s in self.inference['sources'] if s['kind']=='sam'} or None),image_index=0)
        if relative_path:
            frame.update(relative_path=relative_path,image_path=str(self.store.resolve_dataset_path(relative_path)))
        evidence=Bundle.pack('evidence',dict(model=model,frames=[frame]))
        dependencies=list(dict.fromkeys(a['artifact_id'] for a in report['actions'] if a.get('artifact_id')))
        ref=save_evidence(self.store,evidence,run_id=first['run_id'],dependencies=dependencies)
        report['evidence_ref']=ref
        report['counters']=deepcopy(self.counters)
        return evidence,report

    def close(self):
        for crop in self.crops.values():
            crop.close()
        self.crops.clear()
        if self.wd:
            self.wd.model.to('cpu')
        for model in self.models.values():
            model.to('cpu')
