"""Model-free ordered captions, export previews and atomic resumable writes."""
import base64
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, NotRequired, TypedDict, cast

from .artifact_store import StoreError, canonical_bytes, sha256
from .tag_names import canonical_tag, parse_tags


class CaptionDocument(TypedDict):
    schema_version: int
    path: str | None
    original_base64: str | None
    original_digest: str | None
    tags: list[str]
    provenance: dict[str, list[dict[str, Any]]]
    history: NotRequired[list[dict[str, Any]]]


def parse_caption(text, *, input_format='comma_newline', input_separator=', ', escaped_parentheses=False):
    if input_format == 'comma_newline':
        tokens = re.split('[,\r\n]+', text)
    elif input_format == 'literal_separator':
        if not isinstance(input_separator, str) or not input_separator:
            raise StoreError('literal_separator requires a non-empty input_separator')
        if input_separator.isspace():
            raise StoreError('Whitespace-delimited captions are ambiguous when tag names contain spaces; provide comma_newline or structured tags')
        tokens = text.split(input_separator)
    else:
        raise StoreError('Unknown caption input_format')
    if escaped_parentheses:
        tokens = [t.replace(r'\(', '(').replace(r'\)', ')') for t in tokens]
    return list(dict.fromkeys(t for raw in tokens if (t := canonical_tag(raw))))


def caption_document(data, *, path=None, read_config=None, provenance='existing_txt') -> CaptionDocument:
    if not isinstance(data, bytes):
        raise StoreError('CaptionDocument input must be UTF-8 bytes')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise StoreError('Caption must be UTF-8 (BOM allowed)') from exc
    tags = parse_caption(text, **(read_config or {}))
    return CaptionDocument(schema_version=1, path=str(path) if path else None, original_base64=base64.b64encode(data).decode(),
                original_digest=sha256(data), tags=tags,
                provenance={tag:[dict(source=provenance, path=str(path) if path else None)] for tag in tags})


def validate_document(document) -> CaptionDocument:
    if not isinstance(document, dict) or type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise StoreError('Unsupported CaptionDocument version')
    if not isinstance(document.get('tags'), list) or not all(isinstance(t,str) and t and canonical_tag(t)==t for t in document['tags']):
        raise StoreError('CaptionDocument needs canonical structured tags')
    if len(set(document['tags'])) != len(document['tags']):
        raise StoreError('Duplicate caption tags')
    return cast(CaptionDocument,document)


def document_from_tags(tags, *, source='predictions', image_path=None) -> CaptionDocument:
    names = parse_tags(tags)
    return CaptionDocument(schema_version=1, path=image_path, original_base64=None, original_digest=None,
                tags=names, provenance={t:[dict(source=source)] for t in names})


def edit_caption(document: CaptionDocument, operations) -> CaptionDocument:
    validate_document(document)
    if not isinstance(operations, list):
        raise StoreError('caption operations must be an ordered list')
    result = deepcopy(document)
    history = result.setdefault('history', [])
    allowed = {'add_before','add_after','remove','remove_regex','replace','sort','priority'}
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) - allowed:
            raise StoreError('Unknown caption edit field')
        patterns = operation.get('remove_regex', [])
        if not isinstance(patterns,list) or not all(isinstance(p,str) for p in patterns):
            raise StoreError('remove_regex requires a list of patterns')
        compiled = [re.compile(p) for p in patterns]
        removed = set(parse_tags(operation.get('remove')))
        def blocked(tag):
            return tag in removed or any(p.fullmatch(tag) for p in compiled)
        replacements = operation.get('replace', {})
        if not isinstance(replacements,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in replacements.items()):
            raise StoreError('replace requires literal tag -> literal tag mapping')
        replacements = {canonical_tag(k):canonical_tag(v) for k,v in replacements.items()}
        names = []
        provenance = {}
        candidates = [(t, [dict(source='manual_edit')]) for t in parse_tags(operation.get('add_before'))]
        for tag in result['tags']:
            new = replacements.get(tag, tag)
            if new and not blocked(tag):
                candidates.append((new, result['provenance'].get(tag, [])))
        candidates += [(t, [dict(source='manual_edit')]) for t in parse_tags(operation.get('add_after'))]
        for tag, origins in candidates:
            if blocked(tag):
                continue
            if tag not in provenance:
                names.append(tag); provenance[tag] = []
            for origin in origins:
                if origin not in provenance[tag]:
                    provenance[tag].append(origin)
        order = operation.get('sort', 'preserve')
        if order == 'alphabetical':
            names.sort()
        elif order == 'priority':
            priority = parse_tags(operation.get('priority'))
            names.sort(key=lambda tag: priority.index(tag) if tag in priority else len(priority))
        elif order != 'preserve':
            raise StoreError('Unknown caption sort mode')
        result.update(tags=names, provenance=provenance)
        history.append(deepcopy(operation))
    return result


def combine_captions(existing: CaptionDocument, new: CaptionDocument) -> CaptionDocument:
    validate_document(existing); validate_document(new)
    result = deepcopy(existing)
    result['tags'] = list(dict.fromkeys(existing['tags'] + new['tags']))
    for tag in new['tags']:
        origins = result['provenance'].setdefault(tag, [])
        for origin in new['provenance'].get(tag, []):
            if origin not in origins:
                origins.append(deepcopy(origin))
    return result


def format_caption(tags, options=None):
    options = options or {}
    def render(tag):
        if options.get('replace_underscores', True):
            tag = tag.replace('_', ' ')
        if options.get('escape_parentheses', True):
            tag = tag.replace('(', r'\(').replace(')', r'\)')
        return tag
    separator = options.get('tag_separator', ', ')
    if not isinstance(separator,str):
        raise StoreError('tag_separator must be a string')
    return separator.join(render(tag) for tag in tags)


def export_config(**values):
    default = dict(schema_version=1, in_place=True, target_root=None, naming='stem_txt',
                   write_mode='error', mapping={}, read_config={}, options={})
    if set(values) - set(default):
        raise StoreError('Unknown caption export setting')
    config = {**default, **values}
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise StoreError('Unsupported export config version')
    if type(config['in_place']) is not bool or config['write_mode'] not in ('error','skip','append','overwrite') or config['naming'] not in ('stem_txt','image_extension_txt','explicit'):
        raise StoreError('Invalid caption export settings')
    if not config['in_place'] and not config['target_root']:
        raise StoreError('Alternative export requires target_root')
    return config


def target_path(store, image_relative, config):
    source = store.resolve_dataset_path(image_relative)
    relative = Path(image_relative)
    root = store.dataset_root if config['in_place'] else Path(config['target_root']).expanduser().resolve()
    if config['naming'] == 'explicit':
        if image_relative not in config['mapping']:
            raise StoreError('Explicit caption mapping missing: ' + image_relative)
        dest = Path(config['mapping'][image_relative])
        if dest.is_absolute() or '..' in dest.parts:
            raise StoreError('Explicit destination must be relative to export root')
    else:
        dest = relative.with_suffix('.txt') if config['naming'] == 'stem_txt' else Path(str(relative) + '.txt')
    path = (root / dest).resolve()
    if not path.is_relative_to(root) or (path == source and source.suffix.lower() != '.txt') or path.is_relative_to(store.root):
        raise StoreError('Caption destination escapes export boundary or targets evidence/source')
    return path, root


def plan_caption_export(store, entries, *, operation_id, config=None):
    config = export_config(**(config or {}))
    rows = []
    seen = set()
    for entry in entries:
        document = validate_document(entry['document'])
        relative = entry['relative_path']
        path, boundary = target_path(store, relative, config)
        case_key = str(path).casefold()
        if case_key in seen:
            raise StoreError('Caption destination collision; use image_extension_txt or explicit mapping: ' + str(path))
        seen.add(case_key)
        original = path.read_bytes() if path.exists() else None
        existing = caption_document(original, path=path, read_config=config['read_config']) if original is not None else document_from_tags([])
        mode = config['write_mode']
        output = combine_captions(existing, document) if mode == 'append' else deepcopy(document)
        status = 'ready'
        if original is not None and mode == 'error':
            status = 'conflict'
        elif original is not None and mode == 'skip':
            status = 'skipped'; output = existing
        elif original is not None and mode == 'append' and output['tags'] == existing['tags']:
            status = 'unchanged'
        data = original if status in ('skipped','unchanged') else format_caption(output['tags'], config['options']).encode('utf-8')
        assert data is not None
        if data == original and status == 'ready':
            status = 'unchanged'
        rows.append(dict(path=str(path), boundary=str(boundary), relative_path=relative,
                         mode=mode, status=status, expected_digest=sha256(original) if original is not None else None,
                         output_digest=sha256(data), output_base64=base64.b64encode(data).decode(),
                         before=existing['tags'], after=output['tags'],
                         added=[t for t in output['tags'] if t not in existing['tags']],
                         removed=[t for t in existing['tags'] if t not in output['tags']]))
    plan = dict(schema_version=1, operation_id=operation_id, config=config, files=rows)
    # Preview is deliberately pure: no operation row until explicit commit.
    return plan


@contextmanager
def file_lock(store, path, *, timeout=10):
    """OS locks release on process death; the persistent lock file is not ownership."""
    lock_root = store.root / 'locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / sha256(str(path).casefold().encode())
    with open(lock_path, 'a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise StoreError('Caption destination lock timeout')
                time.sleep(.025)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def commit_caption_export(store, plan, *, fault=None):
    if type(plan.get('schema_version')) is not int or plan['schema_version'] != 1:
        raise StoreError('Unsupported caption plan version')
    operation = store.begin_operation(plan['operation_id'], plan)
    previous = {row['path']:row for row in (operation['result'] or {}).get('files', [])}
    outcomes = []
    result = dict(schema_version=1, operation_id=plan['operation_id'], files=outcomes)
    for row in plan['files']:
        path = Path(row['path'])
        outcome = dict(path=str(path), status='failed')
        try:
            if not path.resolve().is_relative_to(Path(row['boundary']).resolve()):
                raise StoreError('Destination escaped boundary after preview')
            with file_lock(store, path):
                current = sha256(path.read_bytes()) if path.exists() else None
                if row['path'] in previous and previous[row['path']]['status'] in ('written','unchanged','skipped'):
                    expected = row['expected_digest'] if previous[row['path']]['status'] == 'skipped' else row['output_digest']
                    outcome['status'] = previous[row['path']]['status'] if current == expected else 'conflict'
                elif current == row['expected_digest'] and row['status'] in ('skipped','unchanged'):
                    outcome['status'] = row['status']
                elif current == row['output_digest'] and row['status'] != 'conflict':
                    outcome['status'] = 'unchanged'
                elif current != row['expected_digest'] or row['status'] == 'conflict':
                    outcome['status'] = 'conflict'
                elif row['status'] in ('skipped','unchanged'):
                    outcome['status'] = row['status']
                else:
                    data = base64.b64decode(row['output_base64'], validate=True)
                    if sha256(data) != row['output_digest']:
                        raise StoreError('Caption plan output checksum mismatch')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, temp_path = tempfile.mkstemp(prefix='.caption-', dir=path.parent)
                    try:
                        with os.fdopen(fd, 'wb') as stream:
                            stream.write(data); stream.flush(); os.fsync(stream.fileno())
                        if fault:
                            fault('before_replace', path)
                        actual = sha256(path.read_bytes()) if path.exists() else None
                        if actual != row['expected_digest']:
                            outcome['status'] = 'conflict'
                        else:
                            if row['expected_digest'] is None:
                                # link/create is atomic and cannot overwrite a concurrent creator.
                                try:
                                    if os.name == 'nt':
                                        os.rename(temp_path, path)  # Windows rename fails if destination exists.
                                    else:
                                        os.link(temp_path, path)
                                except FileExistsError:
                                    outcome['status'] = 'conflict'
                                else:
                                    outcome['status'] = 'written'
                            else:
                                os.replace(temp_path, path)
                                outcome['status'] = 'written'
                            if fault:
                                fault('after_replace', path)
                    finally:
                        Path(temp_path).unlink(missing_ok=True)
        except Exception as exc:
            outcome.update(status='failed', reason=type(exc).__name__, detail=str(exc))
        outcomes.append(outcome)
        store.finish_operation(plan['operation_id'], result, state='pending')
    store.finish_operation(plan['operation_id'], result,
                           state='complete' if all(r['status'] in ('written','skipped','unchanged') for r in outcomes) else 'partial')
    return result
