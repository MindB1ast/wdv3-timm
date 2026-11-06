"""Persistent evidence behind the existing tagging nodes; no additional graph stages."""
import base64
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from .Scripts.artifact_store import ArtifactStore, StoreError
from .Scripts.evidence import Bundle


CACHE_FIELDS=('cache_mode','cache_dataset_root','cache_refresh')


def cache_dataset_root(value):
    """Accept either the dataset directory or its service .wdv3 directory."""
    root=Path(value).expanduser().resolve()
    while root.name.lower()=='.wdv3':
        root=root.parent
    return root


def cached_evidence(frames,runtime,*,options=None,detectors=None,sam_model=None,sources=None,
                    sam_regions=None,image_paths=None,dataset_root=None,progress=None):
    options=dict(options or {})
    from .comfy_nested import has_nested, collect_nested
    nested_sources = sources or (sam_regions[0]['sources'] if sam_regions else None)
    if has_nested(detectors, nested_sources):
        return collect_nested(frames, runtime, detectors=detectors, sam_model=sam_model,
                              sources=nested_sources, sam_regions=sam_regions,
                              options=options, image_paths=image_paths, progress=progress)
    mode=options.get('cache_mode','off')
    if mode=='off':return None
    root=options.get('cache_dataset_root','').strip() or dataset_root
    if not root:
        raise StoreError('Укажите cache_dataset_root в Tagging Settings для постоянного кэша IMAGE или выберите cache_mode=off. Tag Folder берёт папку из Image Folder.')
    prepared=sam_regions
    if prepared is not None:
        if len(prepared)!=len(frames):raise StoreError('SAM and IMAGE counts differ')
        sources=prepared[0]['sources']
        sam_model=prepared[0].get('model_settings') or {}
    from .comfy_evidence import model_identity,detector_path,digest,png
    from .comfy_nodes import tag_regions
    from .comfy_persistent_runtime import PersistentCalculator
    from .Scripts.region_tagging import RegionTagger
    store=ArtifactStore(cache_dataset_root(root))
    configs,models=detectors or ([],{})
    by_name={c.name:c for c in configs}
    branches=[]
    for config in configs:
        path=detector_path(config.model_path).resolve()
        if not path.is_file() and config.download_missing and mode!='reuse_only':
            from .Scripts.model import load_yolo_model
            if config.model_path not in models:
                models[config.model_path]=load_yolo_model(config.model_path,path.parent,allow_download=True).to('cpu')
            path=detector_path(config.model_path).resolve()
        rules={k:v for k,v in asdict(config).items() if k not in ('name','model_path','confidence','classes')}
        branches.append(dict(kind='yolo',name=config.name,confidence=config.confidence,classes=config.classes,
                             rules=rules,model=dict(local_path=str(path),files={'weights':str(path)},
                                                   adapter='legacy-yolo-v1',compute={'device':config.device})))
    if sources:
        if sam_model is None:raise StoreError('Запросы SAM требуют sam_model')
        checkpoint=Path(sam_model.get('checkpoint',''))
        folder=checkpoint.parent if checkpoint.suffix else checkpoint
        if prepared is None and sam_model.get('allow_download') and mode!='reuse_only':
            from .comfy_processing import sam_folder
            folder=sam_folder(str(checkpoint),str(folder),True)
            sam_model={**dict(sam_model),'checkpoint':str(folder)}
        files={('weights' if p.name=='model.safetensors' else p.name):str(p.resolve()) for p in folder.iterdir()
               if p.is_file() and p.name in ('model.safetensors','config.json','processor_config.json','preprocessor_config.json',
                                             'tokenizer_config.json','tokenizer.json','vocab.json','merges.txt')} if folder.is_dir() else {}
        for source in sources:
            spec=dict(files=files,settings=dict(sam_model),adapter='legacy-sam-v1',compute={'device':sam_model.get('device','auto')})
            if prepared is not None:
                spec=dict(identity={'weights_sha256':'supplied-mask-pixels','adapter':'prepared-sam-v1'})
            branches.append(dict(source,kind='sam',model=spec))
    inference=dict(schema_version=1,wd=dict(model_id=getattr(runtime,'model_id','custom-wd'),identity=model_identity(runtime)),sources=branches)
    def score(images):
        from .comfy_evidence import interrupt
        interrupt();runtime.model.to(runtime.device)
        try:
            raw=RegionTagger(runtime.model,runtime.transform,runtime.labels,runtime.device).score_batch(images)
            return [{cat:{tag:float(value) for tag,value in row[cat].items()} for cat in ('general','character','ratings')} for row in raw]
        finally:runtime.model.to('cpu')
    class GeometryOnly:
        def tag_batch(self,images,rules):return [{} for _ in images]
    def detect(image,source):
        rows=tag_regions(image,GeometryOnly(),[by_name[source['name']]],models,options.get('region_batch_size',8))[source['name']]
        return {'regions':rows}
    current_index=0
    def saved_masks(image,source):
        from PIL import Image
        import numpy as np
        from .comfy_processing import frame_digest
        record=prepared[current_index]
        if record['digest']!=frame_digest(image) or record['size']!=list(image.size):
            raise StoreError('SAM regions belong to a different IMAGE')
        rows=[];blobs={}
        for index,row in enumerate(record['regions'][source['name']]):
            with Image.fromarray(np.asarray(row['mask'],dtype=np.uint8)*255) as mask:
                data=base64.b64decode(png(mask))
            blobs[str(index)]=data
            rows.append(dict(bbox=row['bbox'],segmentation_score=float(row['segmentation_score']),mask_blob=store.put_blob(data)))
        return dict(payload={'regions':rows},blobs=blobs)
    hooks={'wd':score,'yolo':detect}
    if prepared is None and options.get('low_memory_mode',True) and hasattr(runtime,'release_weights'):
        hooks['before_sam']=runtime.release_weights
    if prepared is not None:hooks['sam']=saved_masks
    calculator=None if prepared is not None else PersistentCalculator(store,inference,options=options,hooks=hooks)
    frames_out=[];model=None
    selection=[s.strip() for s in options.get('cache_refresh','').split(',') if s.strip()]
    try:
        for index,image in enumerate(frames):
            current_index=index
            if prepared is not None:
                if calculator is not None:calculator.close()
                local=deepcopy(inference)
                for branch in local['sources']:
                    if branch['kind']=='sam':
                        branch['prepared_geometry']=digest([dict(bbox=r['bbox'],score=float(r['segmentation_score']),
                             mask=__import__('hashlib').sha256(r['mask'].tobytes()).hexdigest())
                             for r in prepared[index]['regions'][branch['name']]])
                calculator=PersistentCalculator(store,local,options=options,hooks=hooks)
            if progress:progress(index,len(frames))
            relative=None
            if image_paths is not None:
                path=Path(image_paths[index]).resolve()
                if path.is_relative_to(store.dataset_root):relative=path.relative_to(store.dataset_root).as_posix()
            evidence,report=calculator.compute_image(image,mode=mode,refresh=selection,relative_path=relative)
            if evidence is None:
                missing=[a for a in report['actions'] if a.get('status') not in ('computed','reused')]
                stages=', '.join(dict.fromkeys(a.get('source') or a['stage'] for a in missing))
                name=relative or (str(image_paths[index]) if image_paths is not None else f'IMAGE {index+1}')
                reason=('No inference was performed (reuse_only).' if mode=='reuse_only' else
                        ' '.join(str(a.get('detail','')) for a in missing if a.get('detail')))
                raise StoreError(f'Calculations unavailable for {name}. Missing or failed: {stages}. '
                                 f'Cache folder: {store.root}. {reason} '
                                 'Select the dataset folder used for the original calculation; leave cache_dataset_root '
                                 'empty for Tag Folder. Use cache_mode=reuse only if you want to compute missing stages. '
                                 )
            data=evidence.data();model=data['model']
            frame=data['frames'][0];frame['image_index']=index
            if image_paths is not None:frame['image_path']=str(image_paths[index])
            frames_out.append(frame)
        if progress:progress(len(frames),len(frames))
    finally:
        if calculator is not None:calculator.close()
    return Bundle.pack('evidence',dict(model=model,frames=frames_out))


def sam_records(evidence,settings,sources):
    if not sources:return []
    from PIL import Image
    import numpy as np
    from .comfy_processing import frame_digest
    result=[]
    for frame in evidence.data()['frames']:
        with Image.open(BytesIO(base64.b64decode(frame['image_png']))) as image:
            digest=frame_digest(image)
        regions={}
        for source in sources:
            regions[source['name']]=[]
            for row in frame['detectors'][source['name']]:
                with Image.open(BytesIO(base64.b64decode(row['mask_png']))) as mask:
                    pixels=np.asarray(mask)!=0
                regions[source['name']].append(dict(bbox=row['bbox'],mask=pixels,segmentation_score=row['segmentation_score'],
                    **{key:row[key] for key in ('mask_sha256','model_bbox') if key in row}))
        result.append(dict(digest=digest,size=frame['image_size'],model_settings=settings,sources=sources,regions=regions))
    return result
