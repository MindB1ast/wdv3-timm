"""Replay the user's Region Tag Rules examples on pinned WD scores."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'audit/fixtures/region_filters'


def checked_bytes(path, expected_checksum):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_checksum:
        raise ValueError(f'{path.name}: pinned checksum differs')
    return data


def load_suite():
    settings = ROOT / 'temp/region-filter-tests'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(settings / 'yolo'))
    os.environ.setdefault('MPLCONFIGDIR', str(settings / 'mpl'))
    sys.path.insert(0, str(ROOT))
    from audit.test_comfy_nodes import PACKAGE
    backend = importlib.import_module(f'{PACKAGE.__name__}.comfy_evidence')
    exporter = importlib.import_module(f'{PACKAGE.__name__}.comfy_export')
    model = importlib.import_module(f'{PACKAGE.__name__}.Scripts.model')
    manifest = json.loads((FIXTURES / 'manifest.json').read_bytes())
    checked_bytes(FIXTURES / 'source_workflow.json', manifest['source_workflow_sha256'])
    graph = json.loads(checked_bytes(FIXTURES / 'workflow.json', manifest['workflow_sha256']))
    nodes = {node['id']: node for node in graph['nodes']}
    for case in manifest['cases']:
        saved_rules = nodes[case['rule_node']]['widgets_values_named']
        if saved_rules != dict(manifest['base_rules'], **case['overrides']):
            raise ValueError(f'{case["name"]}: workflow and replay rules differ')
    encoded = checked_bytes(FIXTURES / 'evidence.json.gz', manifest['evidence_sha256'])
    evidence = backend.Bundle.from_json(gzip.decompress(encoded).decode('utf-8'))
    return manifest, evidence, backend, exporter, model


def changes(baseline, result):
    def regions(image):
        return {(row['source'], row['region_id'], tuple(row['bbox'])): row
                for row in image['regions']}

    original, current = regions(baseline), regions(result)
    if original.keys() != current.keys():
        raise ValueError('Region identities differ from baseline')
    region_changes = []
    for key, before in original.items():
        after = current[key]
        change = {'source': key[0], 'region_id': key[1], 'bbox': list(key[2])}
        for category in ('general', 'character'):
            change[f'removed_{category}'] = sorted(set(before[category]) - set(after[category]))
            change[f'added_{category}'] = sorted(set(after[category]) - set(before[category]))
        region_changes.append(change)
    before_caption = set(baseline['caption'].split(', '))
    after_caption = set(result['caption'].split(', '))
    return {'regions': region_changes,
            'caption_removed': sorted(before_caption - after_caption),
            'caption_added': sorted(after_caption - before_caption)}


def run_case(case, manifest, evidence, backend, exporter, directory):
    expected = json.loads(checked_bytes(FIXTURES / case['expected'], case['expected_sha256']))
    frame = evidence['frames'][0]
    config = backend.DetectorConfig.from_dict(frame['source_settings'][0])
    rules = dict(manifest['base_rules'], **case['overrides'])
    field_names = {'general_threshold': 'region_gen_threshold',
                   'character_threshold': 'region_char_threshold',
                   'remove_from_full': 'remove_tags_from_full',
                   'remove_from_regions': 'remove_tags_from_region',
                   'add_to_regions': 'add_tags_to_region',
                   'require_full_confirmation': 'exclude_from_region'}
    for name, value in rules.items():
        if name in ('child_tag_route', 'tag_enabled'):
            continue
        if name.endswith('_regex'):
            value = backend.parse_regex(value)
        elif name == 'add_to_regions':
            value = backend.parse_additions(value)
        elif name not in ('general_threshold', 'character_threshold'):
            value = backend.parse_tags(value)
        setattr(config, field_names.get(name, name), value)
    original_fingerprint = evidence.fingerprint
    options = dict(manifest['options'])
    options['remove_tags_regex'] = backend.parse_regex(options.get('remove_tags_regex', ''))
    filtered = backend.apply_filters(evidence, options, [config])
    if evidence.fingerprint != original_fingerprint:
        raise ValueError('Filter changed the pinned input evidence')
    _, encoded = backend.outputs(filtered)
    output, _ = exporter.save_region_results(
        encoded, directory, case['name'], include_full=True, include_regions=True,
        include_caption=True, write_mode='error')
    actual = json.loads(Path(output).read_bytes())
    if actual != expected:
        for section in ('full', 'regions', 'caption'):
            if actual['images'][0].get(section) != expected['images'][0].get(section):
                raise ValueError(f'{section}: differs from saved user result')
        raise ValueError('Image metadata differs from saved user result')
    baseline = json.loads((FIXTURES / 'filter_test_default.regions.json').read_bytes())['images'][0]
    if changes(baseline, actual['images'][0]) != case['expected_changes']:
        raise ValueError('Region/caption changes differ from pinned expectations')


def forbid_inference(stack, backend, model):
    for module, name in ((backend, 'collect'), (model, 'load_yolo_model'),
                         (model, 'load_model_local_or_remote')):
        stack.enter_context(patch.object(module, name,
            side_effect=AssertionError('Filter regression attempted model inference/loading')))
    stack.enter_context(patch.object(backend.RegionTagger, 'score_batch',
        side_effect=AssertionError('Filter regression attempted WD inference')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', help='Run only the named saved examples')
    args = parser.parse_args()
    manifest, evidence, backend, exporter, model = load_suite()
    cases = manifest['cases']
    if args.case:
        unknown = set(args.case) - {case['name'] for case in cases}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = [case for case in cases if case['name'] in args.case]
    failures = 0
    with tempfile.TemporaryDirectory(prefix='wdv3-filter-regressions-') as directory, ExitStack() as stack:
        forbid_inference(stack, backend, model)
        for case in cases:
            try:
                run_case(case, manifest, evidence, backend, exporter, directory)
            except Exception as error:
                failures += 1
                print(f'FAIL {case["name"]}: {error}', flush=True)
            else:
                print(f'PASS {case["name"]}', flush=True)
    print(f'{len(cases) - failures}/{len(cases)} passed', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
