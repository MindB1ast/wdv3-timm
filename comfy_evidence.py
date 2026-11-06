"""Collect immutable raw evidence, then apply editable policy without inference."""
import base64
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle
from .Scripts.region_tagging import RegionTagger
from .Scripts.tag_utils import (apply_tag_rules, enforce_owners, format_result,
                               parse_tags, parse_additions, restrict_allowed_tags, source_owners,
                               rule_matcher, TagMatcher, REGEX_RULE_FIELDS)
from .Scripts.preprocess import composition_tags, merge_tags_from_regions


def png(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def model_identity(runtime):
    persistent_identity=getattr(runtime,'persistent_identity',None)
    if persistent_identity is not None:
        return persistent_identity
    from .Scripts.tagger_preprocessing import transform_metadata
    state = runtime.model.state_dict()
    def version(value):
        try: return value._version
        except RuntimeError: return None  # ComfyUI creates model tensors inside inference_mode.
    versions = tuple((name, value.data_ptr(), version(value)) for name, value in state.items())
    if any(v is None for _,_,v in versions) or getattr(runtime, '_model_versions', None) != versions or not hasattr(runtime, '_model_hash'):
        h = hashlib.sha256()
        for name, value in state.items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
        runtime._model_hash = h.hexdigest()
        runtime._model_versions = versions
    return dict(model_id=getattr(runtime, 'model_id', 'custom-wd'), weights_sha256=runtime._model_hash,
                device=str(runtime.device), transform=transform_metadata(runtime.transform), labels_count=len(runtime.labels.names),
                labels_sha256=digest([runtime.labels.names,runtime.labels.general,runtime.labels.character,runtime.labels.rating]))


def geometry_config(config):
    geometry={key: value for key, value in asdict(config).items() if key in
            ('name', 'model_path', 'confidence', 'classes', 'device', 'download_missing')}
    geometry['model_path']=str(detector_path(config.model_path))
    return geometry


def detector_path(value):
    path=Path(value)
    if path.is_absolute() or path.parent!=Path('.') or path.suffix not in ('.pt','.pth'):
        return path
    from .comfy_nodes import ROOT, model_directory
    candidates=[ROOT/'models'/'yolo'/path]
    try:candidates.insert(0,model_directory('yolo')/path)
    except ImportError:pass
    return next((p for p in candidates if p.is_file()),candidates[0])


def file_stamp(path):
    path = Path(path)
    if not path.is_file():
        return str(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def interrupt():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def cached_segment(frames, runtime, settings, sources, batch_size=1, progress=None, *, low_memory_mode=True):
    from .comfy_processing import frame_digest, segment_frames
    settings=dict(settings)
    checkpoint = Path(settings.get('checkpoint',''))
    weights = checkpoint / 'model.safetensors' if checkpoint.is_dir() else checkpoint
    stamps=[file_stamp(weights)]+[file_stamp(checkpoint/name) for name in ('config.json','processor_config.json','tokenizer_config.json')] if checkpoint.is_dir() else [file_stamp(weights)]
    key = digest([[frame_digest(f) for f in frames], settings, stamps,
                  [{k:v for k,v in source.items() if k != 'rules'} for source in sources]])
    cache = getattr(runtime, '_sam_cache', None)
    if cache is None: cache = runtime._sam_cache = OrderedDict()
    interrupt()
    if key not in cache:
        if low_memory_mode and hasattr(runtime,'release_weights'):
            runtime.release_weights()
        cache[key] = segment_frames(frames, settings, sources, batch_size, progress)
        while len(cache)>2: cache.popitem(last=False)
    result = deepcopy(cache[key])
    for record in result: record['sources'] = deepcopy(sources)
    return result


def collect(frames, runtime, *, detectors=None, sam_regions=None, image_paths=None, options=None, progress=None):
    from .comfy_nested import has_nested, collect_nested
    if has_nested(detectors, sam_regions[0]['sources'] if sam_regions else None):
        return collect_nested(frames, runtime, detectors=detectors,
            sam_model=sam_regions[0].get('model_settings') if sam_regions else None,
            sources=sam_regions[0]['sources'] if sam_regions else None,
            sam_regions=sam_regions, image_paths=image_paths, options=options, progress=progress)
    from .comfy_processing import frame_digest, sam_tag_regions
    from .comfy_nodes import tag_regions
    from .comfy_config import unique_names
    options = options or {}
    if not frames: raise ValueError('Collect Evidence requires at least one image')
    configs, models = detectors if detectors is not None else ([], {})
    from .comfy_config import validate_rules
    from .Scripts.sam_config import finite_number
    for config in configs:
        validate_rules(asdict(config))
        finite_number(config.confidence,'confidence',maximum=1)
    detector_stamps=getattr(runtime,'_detector_stamps',None)
    if detector_stamps is None:detector_stamps=runtime._detector_stamps={}
    for config in configs:
        path=str(detector_path(config.model_path));stamp=file_stamp(path)
        if path in detector_stamps and detector_stamps[path]!=stamp:
            models.pop(config.model_path,None)
        detector_stamps[path]=stamp
    metadata = model_identity(runtime)
    all_frames = []
    cache = getattr(runtime, '_evidence_cache', None)
    if cache is None:
        cache = runtime._evidence_cache = OrderedDict()
    if sam_regions is not None and len(sam_regions) != len(frames):
        raise ValueError('SAM and WD image counts differ')
    batch_size = options.get('region_batch_size', 8)
    if batch_size < 1 or options.get('image_batch_size', 4) < 1:
        raise ValueError('Batch sizes must be positive')
    class RawTagger:
        def __init__(self):
            self.tagger = RegionTagger(runtime.model, runtime.transform, runtime.labels, runtime.device)
        def tag_batch(self, images, rules):
            interrupt()
            result = self.tagger.score_batch(images)
            interrupt()
            return result
    raw = RawTagger()
    # Preserve true full-image batching for frames not already cached.
    raw_full = {}
    keys, geometries, policies = [], [], []
    for index, frame in enumerate(frames):
        sam = sam_regions[index] if sam_regions is not None else None
        sam_configs = [DetectorConfig(s['name'], 'sam3', **s.get('rules', {})) for s in sam['sources']] if sam else []
        current_configs = list(configs) + sam_configs
        unique_names([c.name for c in current_configs])
        policies.append(current_configs)
        sam_geometry = []
        if sam:
            if sam['digest'] != frame_digest(frame) or sam['size'] != list(frame.size):
                raise ValueError('SAM regions belong to a different image')
            sam_geometry = [{**{k:v for k,v in s.items() if k != 'rules'}, 'regions': [
                {'bbox': r['bbox'], 'score': r['segmentation_score'], 'mask': hashlib.sha256(r['mask'].tobytes()).hexdigest()}
                for r in sam['regions'][s['name']]]} for s in sam['sources']]
        geometry = {'yolo': [dict(geometry_config(c), weights=file_stamp(detector_path(c.model_path))) for c in configs], 'sam': sam_geometry,
                    'sam_model':sam.get('model_settings') if sam else None}
        geometries.append(geometry)
        keys.append(digest([frame_digest(frame), list(frame.size), metadata, geometry]))
    misses = list(dict.fromkeys(i for i,key in enumerate(keys) if key not in cache))
    image_batch = options.get('image_batch_size', 4)
    try:
        if misses:
            try:
                import comfy.model_management as management
                if str(runtime.device) != 'cpu':
                    management.free_memory(sum(p.numel()*p.element_size() for p in runtime.model.parameters())*1.3, runtime.device)
            except ImportError:
                pass
            runtime.model.to(runtime.device)
        for start in range(0, len(misses), image_batch):
            indices = misses[start:start+image_batch]
            predicted = raw.tag_batch([frames[i] for i in indices], None)
            raw_full.update(zip(indices, predicted))
        for index, (frame, key) in enumerate(zip(frames, keys)):
            interrupt()
            if progress: progress(index, len(frames))
            if key in cache:
                payload = deepcopy(cache[key]); cache.move_to_end(key)
            else:
                full = raw_full[index]
                detected = tag_regions(frame, raw, configs, models, batch_size)
                sam = sam_regions[index] if sam_regions is not None else None
                if sam:
                    detected.update(sam_tag_regions(frame, sam, raw, batch_size))
                    from PIL import Image
                    import numpy as np
                    for name, regions in detected.items():
                        if name not in sam['regions']: continue
                        for region, mask_row in zip(regions, sam['regions'][name]):
                            with Image.fromarray(np.asarray(mask_row['mask'], dtype=np.uint8) * 255) as mask:
                                region['mask_png'] = png(mask)
                for result in [full] + [r for rows in detected.values() for r in rows]:
                    result.pop('caption', None); result.pop('taglist', None)
                geometries[index]['yolo']=[dict(geometry_config(c),weights=file_stamp(detector_path(c.model_path))) for c in configs]
                payload = dict(image_id=digest([frame_digest(frame),list(frame.size)]), image_size=list(frame.size), image_png=png(frame),
                               full_image=full, detectors=detected, geometry=geometries[index])
                cache[key] = deepcopy(payload)
                final_key=digest([frame_digest(frame),list(frame.size),metadata,geometries[index]])
                if final_key!=key:cache[final_key]=cache.pop(key)
                for config in configs:
                    detector_stamps[str(detector_path(config.model_path))]=file_stamp(detector_path(config.model_path))
                # Bound retained raw evidence; active bundles own their immutable snapshots.
                while len(cache) > 4: cache.popitem(last=False)
            payload['image_index'] = index
            if image_paths is not None: payload['image_path'] = str(image_paths[index])
            payload['source_settings'] = [asdict(c) for c in policies[index]]
            all_frames.append(payload)
        if progress: progress(len(frames), len(frames))
    finally:
        runtime.model.to('cpu')
        for model in models.values():
            if hasattr(model, 'to'): model.to('cpu')
    return Bundle.pack('evidence', dict(model=metadata, frames=all_frames))


def filter_record(raw, rules, general, character, character_names):
    # Apply rules to the small accepted set before attaching the complete scores
    # and diagnostic records. Those large immutable values need no rule copies.
    result = {key: deepcopy(value) for key, value in raw.items()
              if key not in ('general', 'character', 'ratings')}
    result['ratings'] = dict(raw.get('ratings', {}))
    decisions = {}
    additions=parse_additions(rules.add_tags_to_region)
    character_names = set(character_names)
    remove = rule_matcher(rules, 'remove_tags_from_region')
    allowed = rule_matcher(rules, 'allowed_tags')
    for category, threshold in [('general', general), ('character', character)]:
        local = getattr(rules, 'region_gen_threshold' if category == 'general' else 'region_char_threshold')
        threshold = threshold if local is None else local
        result[category] = {tag: score for tag, score in raw.get(category, {}).items() if score > threshold}
        decisions[category] = {tag: dict(score=score, status='accepted' if score > threshold else 'rejected',
                                      reason='threshold_pass' if score > threshold else 'below_threshold')
                               for tag, score in raw.get(category, {}).items()}
    result = restrict_allowed_tags(apply_tag_rules(result, remove=rules.remove_tags_from_region,
                                  add=rules.add_tags_to_region, character_names=character_names,
                                  remove_regex=rules.remove_tags_from_region_regex), rules)
    for category in ('general', 'character'):
        for tag in set(decisions[category]) | set(result[category]) | set(additions):
            manual_category='character' if tag in character_names else 'general'
            if tag not in raw.get(category, {}) and tag not in result[category] and (tag not in additions or category!=manual_category): continue
            item = decisions[category].setdefault(tag, {'score': result[category].get(tag, additions.get(tag))})
            if tag in remove:
                item.update(status='rejected', reason='remove_from_regions')
            elif allowed and tag not in allowed:
                item.update(status='rejected', reason='not_in_allowed_tags')
            elif tag in result[category]:
                item.update(status='accepted', reason='manual_rule' if tag in result.get('provenance', {}) else 'threshold_pass')
                if tag in result.get('provenance', {}):
                    item.update(model_score=raw.get(category,{}).get(tag),score=result[category][tag],
                                assigned_score=result['provenance'][tag]['assigned_score'])
    result['raw_scores'] = {c: dict(raw.get(c, {})) for c in ('general', 'character', 'ratings')}
    result['decisions'] = decisions
    return format_result(result)


def filter_tag_sets(evidence, options=None, configs=None):
    if evidence.kind != 'evidence': raise ValueError('Apply Filters needs evidence')
    data = evidence.data(); options = deepcopy(dict(options or {}))
    from .comfy_config import validate_config, unique_names, validate_rules
    validate_config({'schema_version':1,'options':options})
    frames = []
    for original in data['frames']:
        if configs is not None and original.get('cascade', {}).get('nested_rules'):
            from .comfy_nested import filter_configs
            frame_configs = filter_configs(original, list(configs))
        else:
            frame_configs = configs
        settings = [DetectorConfig.from_dict(row) for row in original['source_settings']] if frame_configs is None else list(frame_configs)
        unique_names([c.name for c in settings])
        for config in settings:validate_rules(asdict(config))
        expected = {c.name for c in settings}
        if expected != set(original['detectors']):
            raise ValueError('Filter source settings must match collected sources')
        for current in settings:
            previous = next(row for row in original['source_settings'] if row['name'] == current.name)
            if geometry_config(current) != geometry_config(DetectorConfig.from_dict(previous)):
                raise ValueError('Detection parameters changed; collect new evidence before filtering')
        characters = set(original['full_image'].get('character', {}))
        full = filter_record(original['full_image'], DetectorConfig('full', ''),
                             options.get('general_threshold', .35), options.get('character_threshold', .75), characters)
        detected = {c.name: [filter_record(r,c,options.get('general_threshold',.35),options.get('character_threshold',.75),characters)
                            for r in original['detectors'][c.name]] for c in settings}
        for rows in detected.values():
            for row in rows:
                if row.get('tag_enabled') is False:
                    row.update(general={}, character={}, ratings={}, caption='', taglist='',
                               decisions={'general': {}, 'character': {}}, provenance={})
        detail = {**original, 'full_image': full, 'detectors': detected}
        detail.update(settings={'options': options, 'sources': [asdict(c) for c in settings],
                                'sam_model': original['geometry'].get('sam_model')},
                      schema_version=1, stage='tag_sets', model=data['model'])
        if detail.get('cascade'):
            detail['cascade']['routed'] = False
        frames.append(detail)
    result = Bundle.pack('tag_sets', dict(frames=frames, model=data['model'],
                       evidence=dict(schema_version=evidence.schema_version, kind=evidence.kind, payload=data),
                       postprocessing=[]))
    if any(frame.get('cascade') for frame in frames):
        from .comfy_cascade_tags import apply_local_operations
        result = apply_local_operations(result)
    return result


def apply_filters(evidence, options=None, configs=None):
    """Compatible shortcut: local filters followed by the unchanged global merge."""
    return merge_tag_sets(filter_tag_sets(evidence, options, configs))


def merge_tag_sets(tag_sets):
    if tag_sets.kind != 'tag_sets':
        raise ValueError('Merge Tag Sets needs tag_sets')
    data = tag_sets.data()
    if any(frame.get('cascade') for frame in data['frames']):
        from .comfy_cascade_tags import merge_cascade_tag_sets
        if all(f.get('cascade', {}).get('nested_rules') for f in data['frames']):
            from .comfy_cascade_tags import route_cascade_tags
            if not all(f['cascade'].get('routed') for f in data['frames']):
                tag_sets = route_cascade_tags(tag_sets)
        return merge_cascade_tag_sets(tag_sets)
    frames = []
    for original in data['frames']:
        # data() already returned an owned snapshot, separate from the input
        # Bundle. Mutating it cannot affect the preceding pipeline stage.
        detail = original
        options = detail['settings']['options']
        settings = [DetectorConfig.from_dict(row) for row in detail['settings']['sources']]
        matchers = {c.name: {field: rule_matcher(c, field,
                     composition_tags(c) if field == 'full_image_only_tags' else None)
                     for field in REGEX_RULE_FIELDS} for c in settings}
        characters = list(original['full_image'].get('raw_scores', {}).get('character', original['full_image']['character']))
        full, detected = detail['full_image'], detail['detectors']
        merged = enforce_owners(apply_tag_rules(merge_tags_from_regions(detail, settings),
            remove=options.get('remove_tags',''), before=options.get('prefix_tags',''), after=options.get('suffix_tags',''),
            character_names=characters, remove_regex=options.get('remove_tags_regex', [])), settings)
        candidates = set(full['general']) | set(full['character'])
        candidates.update(parse_tags(options.get('prefix_tags','')) + parse_tags(options.get('suffix_tags','')))
        for rows in detected.values():
            for row in rows:
                candidates.update(row['general']); candidates.update(row['character'])
        owners = source_owners(settings, candidates)
        excluded_full = {t for c in settings if detected[c.name] for t in candidates
                         if t in matchers[c.name]['remove_tags_from_full']}
        global_remove = TagMatcher(options.get('remove_tags',''), options.get('remove_tags_regex', []))
        decisions = []
        for name, region, result in [('full',None,full)] + [(name,r['region_id'],r) for name,rows in detected.items() for r in rows]:
            rule = next((c for c in settings if c.name == name), None)
            for category in ('general','character'):
                for tag, decision in result['decisions'][category].items():
                    item = dict(decision, tag=tag, category=category, source=name, region_id=region)
                    if item['status'] == 'accepted':
                        reason = None
                        if tag in owners and name not in owners[tag]: reason='owned_by_other_sources'
                        elif name=='full' and tag in excluded_full: reason='remove_from_full'
                        elif rule and tag in matchers[name]['exclude_from_region'] and tag not in full['general'] and tag not in full['character']: reason='requires_full_confirmation'
                        elif rule and (category=='general' or rule.full_image_only_tags is not None or rule.full_image_only_tags_regex) and tag in matchers[name]['full_image_only_tags']: reason='composition_full_only'
                        elif tag in global_remove: reason='global_remove'
                        if reason: item.update(status='rejected',reason=reason)
                        else: item['reason']='merged' if merged.get(category,{}).get(tag)==item['score'] else 'lower_score'
                    decisions.append(item)
        for tag in parse_tags(options.get('prefix_tags',''))+parse_tags(options.get('suffix_tags','')):
            category='character' if tag in characters else 'general'
            provenance=merged.get('provenance',{}).get(tag,{})
            if tag in merged[category] and (provenance.get('source')!='manual_rule' or provenance.get('region_source')):
                continue
            reason='global_remove' if tag in global_remove else 'owned_by_other_sources' if tag in owners and tag not in merged[category] else 'manual_rule'
            decisions.append(dict(tag=tag,category=category,source='manual_rule',region_id=None,
                score=provenance.get('assigned_score',1.0),status='accepted' if tag in merged[category] else 'rejected',reason=reason))
        # Literal rules already supplied the reasons above. Only actual regex
        # rules need the additional diagnostic matching pass, once compiled.
        regex_checks = {}
        for source in ('full', 'manual_rule', *detected):
            regex_checks[source] = [(c.name, field, matchers[c.name][field])
                for c in settings for field in REGEX_RULE_FIELDS
                if matchers[c.name][field].compiled and (
                    field == 'only_from_source'
                    or (field == 'remove_tags_from_full' and source == 'full' and detected[c.name])
                    or (field not in ('only_from_source', 'remove_tags_from_full') and source == c.name))]
        local_records = {('full', None): full}
        local_records.update({(name, r['region_id']): r for name, rows in detected.items() for r in rows})
        for item in decisions if global_remove.compiled or any(regex_checks.values()) else ():
            matches = []
            for name, field, matcher in regex_checks[item['source']]:
                for match in matcher.matches(item['tag']):
                    if match['kind'] == 'regex':
                        matches.append(dict(match, action=field, source=name))
            if global_remove.compiled:
                for match in global_remove.matches(item['tag']):
                    if match['kind'] == 'regex':
                        matches.append(dict(match, action='remove_tags', source='global'))
            if matches:
                item['matched_rules'] = matches
                local = local_records.get((item['source'], item['region_id']))
                if local is not None and item['tag'] in local['decisions'][item['category']]:
                    local['decisions'][item['category']][item['tag']]['matched_rules'] = deepcopy(matches)
        detail.update(merged_tags=merged, decisions=decisions, settings={'options':options,'sources':[asdict(c) for c in settings],
                      'sam_model':original['geometry'].get('sam_model')},
                      schema_version=1, stage='results', model=data['model'])
        if data.get('postprocessing'):
            detail['settings']['postprocessing'] = deepcopy(data['postprocessing'])
        frames.append(detail)
    payload = dict(frames=frames, model=data['model'], evidence=data['evidence'])
    if data.get('postprocessing'):
        payload['postprocessing'] = deepcopy(data['postprocessing'])
    return Bundle.pack('results', payload)


def formatted_text(frame, result=None):
    options=frame.get('settings',{}).get('options',{})
    result=result if result is not None else frame.get('merged_tags',frame.get('full_image',{}))
    names = set(result.get('general', {})) | set(result.get('character', {}))
    ordered = [tag for tag in parse_tags(result.get('caption', '')) if tag in names]
    ordered.extend(sorted(names - set(ordered)))
    def render(tag):
        if options.get('replace_underscores', True):
            tag = tag.replace('_', ' ')
        if options.get('escape_parentheses', True):
            tag = tag.replace('(', r'\(').replace(')', r'\)')
        return tag
    return options.get('tag_separator', ', ').join(render(tag) for tag in ordered)


def outputs(results):
    from .comfy_processing import json_result
    tags, details = [], []
    for frame in results.data()['frames']:
        text=formatted_text(frame)
        detail=frame  # results.data() already owns these dictionaries.
        detail.pop('image_png',None)
        for regions in detail['detectors'].values():
            for region in regions:region.pop('mask_png',None)
        tags.append(text); details.append(json_result(detail))
    return tags, details


def merge_results(left, right):
    if left.kind!='results' or right.kind!='results': raise ValueError('Merge Results needs result bundles')
    a,b=left.data(),right.data()
    if any(f.get('cascade') for f in a['frames'] + b['frames']):
        if not all(f.get('cascade') for f in a['frames'] + b['frames']):
            raise ValueError('Cannot merge hierarchical and flat results')
        from .comfy_cascade_tags import merge_cascade_results
        return merge_cascade_results(left, right)
    from .Scripts.tag_postprocessing import validate_chain, apply_chain
    def chain(value):
        fallback = value['frames'][0]['settings'].get('postprocessing', []) if value['frames'] else []
        return validate_chain(value.get('postprocessing', fallback))
    operations = chain(a)
    if operations != chain(b):
        raise ValueError('Cannot merge different postprocessing chains')
    x,y=a['evidence']['payload'],b['evidence']['payload']
    if x['model']!=y['model'] or len(x['frames'])!=len(y['frames']): raise ValueError('Incompatible evidence models or image counts')
    merged=deepcopy(x)
    for index,(p,q) in enumerate(zip(merged['frames'],y['frames'])):
        if (p['image_id'],p['image_size'],p['full_image']) != (q['image_id'],q['image_size'],q['full_image']):
            raise ValueError('Cannot merge results of different images or full-image evidence')
        if ({'tag_separator': ', ', **a['frames'][index]['settings']['options']}
                != {'tag_separator': ', ', **b['frames'][index]['settings']['options']}):
            raise ValueError('Cannot merge different global filter settings')
        sources={c['name']:c for c in a['frames'][index]['settings']['sources']}
        for row in b['frames'][index]['settings']['sources']:
            name=row['name']
            if name in sources and (sources[name]!=row or p['detectors'][name]!=q['detectors'][name]):
                raise ValueError('Conflicting evidence or rules for source '+name)
            sources[name]=row; p['detectors'][name]=deepcopy(q['detectors'][name])
        p['source_settings']=[sources[n] for n in sorted(sources)]
    evidence=Bundle.pack('evidence',merged)
    # Apply per-frame source snapshots; global options are shared and were checked above.
    local = filter_tag_sets(evidence, a['frames'][0]['settings']['options'] if a['frames'] else {})
    return merge_tag_sets(apply_chain(local, operations))
