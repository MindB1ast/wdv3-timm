"""Deterministic operations on accepted local tags. Standard library only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING, NotRequired, TypedDict

from .tag_names import canonical_tag, parse_tags

if TYPE_CHECKING:
    from .evidence import Bundle


class LocalTagRecord(TypedDict):
    general: dict[str, float]
    character: dict[str, float]
    ratings: dict[str, float]
    caption: str
    taglist: str
    raw_scores: dict[str, dict[str, float]]
    decisions: dict[str, dict[str, dict]]
    region_id: NotRequired[int]
    bbox: NotRequired[list[float]]
    postprocessing_history: NotRequired[list[dict]]
    region_uid: NotRequired[str]
    parent_region_uid: NotRequired[str | None]
    stage_id: NotRequired[str]
    depth: NotRequired[int]
    aggregate: NotRequired[dict]
    route_history: NotRequired[list[dict]]


class LocalTagFrame(TypedDict):
    image_id: str
    image_size: list[int]
    full_image: LocalTagRecord
    detectors: dict[str, list[LocalTagRecord]]
    source_settings: list[dict]
    geometry: dict
    settings: dict
    stage: str


class TagSetPayload(TypedDict):
    """Bundle('tag_sets'): frames retain their original geometry/pixels/metadata."""
    frames: list[LocalTagFrame]
    model: dict
    evidence: dict
    postprocessing: list[dict]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _names(value, field):
    if isinstance(value, str):
        return sorted(parse_tags(value))
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f'{field}: expected a list of nonempty literal names')
    return sorted(set(canonical_tag(x) for x in value))


@lru_cache(maxsize=512)
def _regex(pattern):
    return re.compile(pattern)


def _check_cycles(pairs):
    edges, indegree = {}, {}
    for specific, general in set(pairs):
        edges.setdefault(specific, []).append(general)
        indegree.setdefault(specific, 0)
        indegree[general] = indegree.get(general, 0) + 1
    pending = [name for name, count in indegree.items() if not count]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        for child in edges.get(current, []):
            indegree[child] -= 1
            if not indegree[child]:
                pending.append(child)
    if visited != len(indegree):
        raise ValueError('relations: cycle involving ' + ', '.join(sorted(n for n, count in indegree.items() if count)))


def validate_operation(value):
    """Normalize a versioned redundancy operation, without inference or I/O."""
    defaults = dict(schema_version=1, operation='remove_redundant', explicit_enabled=False,
                    textual_enabled=False, relations=[], protected_tags=[], scope='all', sources=[],
                    regex_enabled=False, regex_relations=[])
    if not isinstance(value, dict) or set(value) - defaults.keys():
        raise ValueError('Unknown postprocessing operation fields')
    operation = {**defaults, **deepcopy(value)}
    if type(operation['schema_version']) is not int or operation['schema_version'] != 1:
        raise ValueError('Unsupported postprocessing schema_version')
    if operation['operation'] != 'remove_redundant':
        raise ValueError('Unsupported postprocessing operation')
    for field in ('explicit_enabled', 'textual_enabled', 'regex_enabled'):
        if type(operation[field]) is not bool:
            raise ValueError(f'{field}: expected boolean')
    if operation['scope'] not in ('all', 'full', 'regions'):
        raise ValueError('scope: expected all, full or regions')
    operation['protected_tags'] = _names(operation['protected_tags'], 'protected_tags')
    sources = operation['sources']
    if not isinstance(sources, list) or any(not isinstance(x, str) or not x.strip() for x in sources):
        raise ValueError('sources: expected a list of nonempty source names')
    operation['sources'] = sorted(set(x.strip() for x in sources))
    relations = operation['relations']
    if not isinstance(relations, list):
        raise ValueError('relations: expected an array of specific/general pairs')
    pairs = set()
    for index, row in enumerate(relations, 1):
        if not isinstance(row, dict) or set(row) != {'specific', 'general'}:
            raise ValueError(f'relations pair {index}: expected specific and general')
        for field in ('specific', 'general'):
            if not isinstance(row[field], str) or not row[field].strip() or any(c in row[field] for c in ',\r\n'):
                raise ValueError(f'relations pair {index}.{field}: expected one literal name')
        pair = canonical_tag(row['specific']), canonical_tag(row['general'])
        if pair[0] == pair[1]:
            raise ValueError(f'relations pair {index}: self-reference {pair[0]}')
        pairs.add(pair)
    _check_cycles(pairs)
    operation['relations'] = [dict(specific=a, general=b) for a, b in sorted(pairs)]
    rows = operation['regex_relations']
    if not isinstance(rows, list):
        raise ValueError('regex_relations: expected an array')
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {'id', 'specific', 'general'}:
            raise ValueError(f'regex_relations rule {index}: expected id, specific pattern and general template')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError(f'regex_relations rule {index} ({row["id"]}): fields must be nonempty text')
        if row['id'] in seen:
            raise ValueError(f'regex_relations: duplicate id {row["id"]}')
        seen.add(row['id'])
        if any(c in row['general'] for c in ',\r\n'):
            raise ValueError(f'regex_relations {row["id"]}: general must expand to one tag')
        try:
            # sub parses replacement references even when the input has no match.
            _regex(row['specific']).sub(row['general'], '')
        except (re.error, IndexError, OverflowError) as error:
            raise ValueError(f'regex_relations {row["id"]}: {error}') from error
    operation['regex_relations'] = sorted(rows, key=lambda row: row['id'])
    return operation


def operation_from_fields(relations_json='[]', protected_tags='', scope='all', sources='',
                          explicit_enabled=False, textual_enabled=False, regex_enabled=False,
                          regex_relations_json='[]'):
    """ComfyUI's literal fields; source names are not tag-canonicalized."""
    if not isinstance(sources, str):
        raise ValueError('sources: expected comma/newline separated names')
    return validate_operation(dict(relations=json.loads(relations_json), protected_tags=protected_tags,
        scope=scope, sources=[x.strip() for x in sources.replace('\n', ',').split(',') if x.strip()],
        explicit_enabled=explicit_enabled, textual_enabled=textual_enabled,
        regex_enabled=regex_enabled, regex_relations=json.loads(regex_relations_json)))


def validate_chain(value):
    if not isinstance(value, list):
        raise ValueError('postprocessing must be an ordered array')
    return [validate_operation(operation) for operation in value]


def _format(record):
    names = set(record['general']) | set(record['character'])
    ordered = [name for name in parse_tags(record.get('caption', '')) if name in names]
    ordered.extend(sorted(names - set(ordered)))
    record['caption'] = ', '.join(ordered)
    record['taglist'] = ', '.join(name.replace('_', ' ') for name in ordered)


def remove_redundant(tag_sets: Bundle, operation: dict | None = None) -> Bundle:
    """Return a new immutable bundle; compare only accepted general in one record."""
    if tag_sets.kind != 'tag_sets':
        raise ValueError('Remove Redundant Tags needs tag_sets')
    operation = validate_operation({} if operation is None else operation)
    data = tag_sets.data()
    # Check all sources before changing any frame, including empty source groups.
    for frame in data['frames']:
        unknown = set(operation['sources']) - set(frame['detectors'])
        if unknown:
            raise ValueError('Unknown postprocessing sources: ' + ', '.join(sorted(unknown)))
    operation_index = len(data.get('postprocessing', []))
    for frame in data['frames']:
        if frame.get('cascade'):
            frame['cascade']['routed'] = False
        records = []
        if operation['scope'] in ('all', 'full'):
            records.append(('full', None, frame['full_image']))
        if operation['scope'] in ('all', 'regions'):
            for source, rows in sorted(frame['detectors'].items()):
                if not operation['sources'] or source in operation['sources']:
                    records.extend((source, row['region_id'], row) for row in rows)
        for source, region_id, record in records:
            before = deepcopy(record['general'])
            reasons = {}
            def match(tag, witness, method, rule):
                if tag != witness and tag in before and witness in before and tag not in operation['protected_tags']:
                    cause = dict(method=method, rule=rule, witness=witness)
                    if method == 'regex':
                        cause['rule_digest'] = fingerprint(rule)
                    reasons.setdefault(tag, []).append(cause)
            if operation['explicit_enabled']:
                for pair in operation['relations']:
                    match(pair['general'], pair['specific'], 'explicit', pair)
            if operation['regex_enabled']:
                for rule in operation['regex_relations']:
                    for witness in sorted(before):
                        found = _regex(rule['specific']).fullmatch(witness)
                        if found is None:
                            continue
                        target = canonical_tag(found.expand(rule['general']))
                        if not target or any(c in target for c in ',\r\n'):
                            raise ValueError(f'regex_relations {rule["id"]}: invalid expanded tag for {witness}')
                        match(target, witness, 'regex', rule)
            if operation['textual_enabled']:
                for witness in sorted(before):
                    components = witness.split('_')
                    # Contiguous whole components, never arbitrary substrings.
                    for start in range(len(components)):
                        for end in range(start + 1, len(components) + 1):
                            tag = '_'.join(components[start:end])
                            if tag and tag != witness:
                                match(tag, witness, 'textual', 'whole_components')
            _check_cycles((cause['witness'], tag) for tag, causes in reasons.items() for cause in causes)
            removed = []
            for tag in sorted(reasons):
                score = record['general'].pop(tag)
                record.get('provenance', {}).pop(tag, None)
                causes = sorted(reasons[tag], key=lambda r: json.dumps(r, sort_keys=True))
                item = dict(tag=tag, category='general', score=score, source=source, region_id=region_id,
                            image_id=frame['image_id'], operation_index=operation_index, causes=causes)
                removed.append(item)
                record['decisions']['general'][tag].update(status='rejected', reason='redundant_tag',
                    postprocessing=deepcopy(item))
            # Disabled operations are true record-level no-ops.
            if operation['explicit_enabled'] or operation['textual_enabled'] or operation['regex_enabled']:
                record.setdefault('postprocessing_history', []).append(dict(operation_index=operation_index,
                    operation=deepcopy(operation), input_digest=fingerprint(before),
                    output_digest=fingerprint(record['general']), before=before,
                    after=deepcopy(record['general']), removed=removed))
                _format(record)
    data.setdefault('postprocessing', []).append(operation)
    return type(tag_sets).pack('tag_sets', data)


def local_decisions(frame):
    """Flatten local decisions without applying any cross-source merge policies."""
    result = []
    records = [('full', None, frame['full_image'])]
    records += [(name, row['region_id'], row) for name, rows in frame['detectors'].items() for row in rows]
    for source, region_id, record in records:
        for category, tags in record['decisions'].items():
            result.extend(dict(deepcopy(decision), tag=tag, category=category,
                               source=source, region_id=region_id) for tag, decision in tags.items())
    return result


def apply_chain(tag_sets: Bundle, operations: list) -> Bundle:
    operations = validate_chain(operations)
    for operation in operations:
        tag_sets = remove_redundant(tag_sets, operation)
    return tag_sets
