"""Explicit bottom-up routing. Raw evidence and origin policy remain separate."""
from copy import deepcopy
from dataclasses import asdict

from .Scripts.cascade import compile_plan, combine_plans
from .Scripts.cascade_geometry import validate_tree
from .Scripts.configs import DetectorConfig
from .Scripts.evidence import Bundle
from .Scripts.tag_postprocessing import apply_chain
from .Scripts.tag_utils import format_result, rule_matcher


def _operate(record, operations, source, frame):
    if not operations:
        return record
    value = Bundle.pack('tag_sets', dict(frames=[dict(image_id=frame['image_id'],
        full_image=deepcopy(frame['full_image']), detectors={source: record and [record] or []})],
        model={}, postprocessing=[]))
    # Scope still denotes one selected aggregate; other sources remain invalid.
    return apply_chain(value, list(operations))['frames'][0]['detectors'][source][0]


def apply_local_operations(tag_sets):
    data = tag_sets.data()
    for frame in data['frames']:
        if not frame.get('cascade'):
            continue
        for stage in compile_plan(frame['cascade']['plan']).stages:
            frame['detectors'][stage.stage_id] = [_operate(row, stage.local_operations, stage.stage_id, frame)
                                                  for row in frame['detectors'][stage.stage_id]]
    return Bundle.pack('tag_sets', data)


def route_cascade_tags(tag_sets, plan=None):
    if tag_sets.kind != 'tag_sets':
        raise ValueError('Route Cascade Tags needs tag_sets')
    data = tag_sets.data()
    for frame in data['frames']:
        if not frame.get('cascade'):
            raise ValueError('Route Cascade Tags requires hierarchical evidence')
        validate_tree(frame)
        original_plan = compile_plan(frame['cascade']['plan'])
        selected = compile_plan(plan) if plan is not None else original_plan
        before = original_plan.data()
        after = selected.data()
        # A routing pass may change only routes and aggregate operations.
        for value in (before, after):
            for stage in value['stages']:
                stage.pop('tag_route')
                stage.pop('aggregate_operations')
        if before != after:
            raise ValueError('Routing plan changes inference or local policy; refilter/recollect first')
        stages = {s.stage_id: s for s in selected.stages}
        rules = {c['name']: DetectorConfig.from_dict(c) for c in frame['settings']['sources']}
        by_uid = {row['region_uid']: row for rows in frame['detectors'].values() for row in rows}
        # Repeat routing always starts from the saved own accepted records.
        for row in by_uid.values():
            row.pop('aggregate', None)
            row.pop('route_history', None)
            row['tag_route'] = stages[row['stage_id']].tag_route
            aggregate = deepcopy(row)
            aggregate['origin_candidates'] = {category: {tag: [dict(score=score,
                origin_source=row['stage_id'], origin_region_uid=row['region_uid'],
                path=[row['region_uid']], provenance=deepcopy(row.get('provenance', {}).get(tag)))]
                for tag, score in row[category].items()} for category in ('general', 'character')}
            row['aggregate'] = aggregate
            row['route_history'] = []
        for row in sorted(by_uid.values(), key=lambda r: (-r['depth'], r['region_uid'])):
            name = row['stage_id']
            aggregate = _operate(row['aggregate'], stages[name].aggregate_operations, name, frame)
            for category in ('general', 'character'):
                aggregate['origin_candidates'][category] = {tag: candidates for tag, candidates in
                    aggregate['origin_candidates'][category].items() if tag in aggregate[category]}
            row['aggregate'] = aggregate
            parent_uid = row['parent_region_uid']
            if not parent_uid or row['tag_route'] == 'separate':
                continue
            parent = by_uid[parent_uid]
            parent_rule = rules[parent['stage_id']]
            remove = rule_matcher(parent_rule, 'remove_tags_from_region')
            allowed = rule_matcher(parent_rule, 'allowed_tags')
            target = parent['aggregate']
            for category in ('general', 'character'):
                for tag, candidates in aggregate['origin_candidates'][category].items():
                    reason = 'remove_from_parent' if tag in remove else 'not_allowed_by_parent' if allowed and tag not in allowed else None
                    for candidate in candidates:
                        transferred = deepcopy(candidate)
                        transferred['path'].append(parent_uid)
                        event = dict(transferred, tag=tag, category=category, receiver=parent_uid,
                                     status='rejected' if reason else 'accepted', reason=reason or 'merge_to_parent')
                        parent['route_history'].append(event)
                        if reason:
                            continue
                        group = target['origin_candidates'][category].setdefault(tag, [])
                        if transferred not in group:
                            group.append(transferred)
                        group.sort(key=lambda c: (-c['score'], c['origin_source'], c['origin_region_uid'], c['path']))
                        target[category][tag] = group[0]['score']
                        target['decisions'][category][tag] = dict(score=group[0]['score'], status='accepted', reason='received_from_child')
            format_result(target)
        frame['cascade'].update(plan=selected.data(), routed=True)
    return Bundle.pack('tag_sets', data)


def merge_cascade_tag_sets(tag_sets):
    from .comfy_evidence import merge_tag_sets
    data = tag_sets.data()
    rendered = []
    for frame in data['frames']:
        validate_tree(frame)
        if not frame['cascade'].get('routed'):
            raise ValueError('Hierarchical tag_sets require Route Cascade Tags before Merge Tag Sets')
        virtual = deepcopy(frame)
        virtual.pop('cascade')
        # Keep source presence for remove_tags_from_full even when all tags route away.
        virtual['detectors'] = {name: [dict(general={}, character={}, ratings={},
            region_id=-1, decisions={'general': {}, 'character': {}})] if rows else []
            for name, rows in frame['detectors'].items()}
        for rows in frame['detectors'].values():
            for row in rows:
                if row['parent_region_uid'] and row['tag_route'] != 'separate':
                    continue
                aggregate = row['aggregate']
                for category in ('general', 'character'):
                    for tag, candidates in aggregate['origin_candidates'][category].items():
                        for candidate in candidates:
                            source = candidate['origin_source']
                            record = dict(general={}, character={}, ratings={}, region_id=len(virtual['detectors'][source]),
                                          decisions={'general': {}, 'character': {}}, provenance={})
                            record[category][tag] = candidate['score']
                            record['decisions'][category][tag] = dict(score=candidate['score'], status='accepted',
                                reason='routed', origin_region_uid=candidate['origin_region_uid'], path=candidate['path'])
                            record['provenance'][tag] = dict(source=source+':'+candidate['origin_region_uid'], origin_source=source,
                                origin_region_uid=candidate['origin_region_uid'], path=candidate['path'],
                                model_score=candidate['score'])
                            if candidate.get('provenance'):
                                record['provenance'][tag].update(candidate['provenance'])
                                record['provenance'][tag]['region_source'] = source+':'+candidate['origin_region_uid']
                            virtual['detectors'][source].append(record)
        flat = Bundle.pack('tag_sets', dict(data, frames=[virtual]))
        merged = merge_tag_sets(flat)['frames'][0]
        by_uid = {row['region_uid']: row for rows in frame['detectors'].values() for row in rows}
        for decision in merged['decisions']:
            origin = by_uid.get(decision.get('origin_region_uid'))
            if origin is not None:
                decision['region_id'] = origin['region_id']
        result = deepcopy(frame)
        result.update(merged_tags=merged['merged_tags'], decisions=merged['decisions'], stage='results')
        if data.get('postprocessing'):
            result['settings']['postprocessing'] = deepcopy(data['postprocessing'])
        rendered.append(result)
    return Bundle.pack('results', dict(data, frames=rendered))


def merge_cascade_results(left, right):
    from .comfy_evidence import filter_tag_sets, merge_tag_sets
    a, b = left.data(), right.data()
    if a.get('postprocessing', []) != b.get('postprocessing', []):
        raise ValueError('Conflicting cascade postprocessing')
    x, y = deepcopy(a['evidence']['payload']), b['evidence']['payload']
    if x['model'] != y['model'] or len(x['frames']) != len(y['frames']):
        raise ValueError('Incompatible cascade models/images')
    frames = []
    for index, (p, q) in enumerate(zip(x['frames'], y['frames'])):
        if (p['image_id'], p['image_size'], p['full_image']) != (q['image_id'], q['image_size'], q['full_image']):
            raise ValueError('Conflicting cascade frame or full evidence')
        af, bf = a['frames'][index], b['frames'][index]
        if af['settings']['options'] != bf['settings']['options']:
            raise ValueError('Conflicting global cascade settings')
        selected = combine_plans(af['cascade']['plan'], bf['cascade']['plan'])
        raw_plan = combine_plans(p['cascade']['plan'], q['cascade']['plan'])
        rules = {s['name']: s for s in af['settings']['sources']}
        for rule in bf['settings']['sources']:
            name = rule['name']
            if name in rules and (rules[name] != rule or p['detectors'][name] != q['detectors'][name]):
                raise ValueError('Conflicting cascade geometry/evidence/policy: ' + name)
            rules[name] = rule
            p['detectors'][name] = deepcopy(q['detectors'][name])
        p['source_settings'] = [rules[name] for name in sorted(rules)]
        p['cascade']['plan'] = raw_plan.data()
        validate_tree(p)
        local = filter_tag_sets(Bundle.pack('evidence', dict(model=x['model'], frames=[p])), af['settings']['options'])
        local = apply_chain(local, a.get('postprocessing', []))
        local = route_cascade_tags(local, selected)
        frames.extend(merge_tag_sets(local)['frames'])
    return Bundle.pack('results', dict(model=x['model'], frames=frames, evidence=dict(
        schema_version=1, kind='evidence', payload=x), postprocessing=a.get('postprocessing', [])))
