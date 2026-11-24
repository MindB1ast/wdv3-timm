"""Replay explicit filter outcomes on fixed scores, with optional JSON output."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

if __package__:
    from . import run_region_filter_regressions as historical_suite
else:
    import run_region_filter_regressions as historical_suite

HISTORICAL = historical_suite.FIXTURES
FIXTURES = HISTORICAL.with_name('region_filter_outcomes')
OUTPUT_OPTIONS = dict(include_full=True, include_regions=True, include_caption=True)
RULE_FIELDS = {
    'general_threshold': 'region_gen_threshold',
    'character_threshold': 'region_char_threshold',
    'remove_from_full': 'remove_tags_from_full',
    'remove_from_regions': 'remove_tags_from_region',
    'add_to_regions': 'add_tags_to_region',
    'require_full_confirmation': 'exclude_from_region',
}


def scenarios():
    manifest = json.loads((FIXTURES / 'manifest.json').read_bytes())
    return [(suite, case) for suite, cases in manifest.items() for case in cases]


def configured(backend, rows, defaults, overrides):
    configs = []
    for row in rows:
        config = backend.DetectorConfig.from_dict(row)
        for field, value in (defaults | overrides.get(config.name, {})).items():
            if field in ('child_tag_route', 'tag_enabled'):
                continue
            if field.endswith('_regex'):
                value = backend.parse_regex(value)
            elif field == 'add_to_regions':
                value = backend.parse_additions(value)
            elif not field.endswith('_threshold'):
                value = backend.parse_tags(value)
            setattr(config, RULE_FIELDS.get(field, field), value)
        configs.append(config)
    return configs


def parallel_evidence(evidence, backend):
    data = evidence.data()
    for frame in data['frames']:
        rows = frame['detectors']['person']
        config, = frame['source_settings']
        geometry, = frame['geometry']['yolo']
        sources = ('person_A', 'person_B')
        frame['detectors'] = {name: deepcopy(rows) for name in sources}
        frame['source_settings'] = [dict(config, name=name) for name in sources]
        frame['geometry']['yolo'] = [dict(geometry, name=name) for name in sources]
    return backend.Bundle.pack('evidence', data)


def boundary_evidence(evidence, backend):
    data = evidence.data()
    for frame in data['frames']:
        row = frame['detectors']['person'][0]
        scores = (.49, .50, .51)
        row['general'] = dict(zip(('boundary_below', 'boundary_equal', 'boundary_above'), scores))
        row['character'] = dict(zip(('ichihara_nina', 'tachibana_arisu', 'sajo_yukimi'), scores))
        frame['detectors']['person'] = [row]
    return backend.Bundle.pack('evidence', data)


def select(image, selector):
    location, _, category = selector.partition('.')
    category = category or 'general'
    if location == 'caption':
        return [set(image['caption'].split(', '))]
    if location == 'full':
        return [set(image['full'][category])]
    source, _, index = location.partition(':')
    matches = [set(row[category]) for row in image['regions'] if row['source'] == source
               and (not index or index == '*' or str(row['region_id']) == index)]
    if not matches:
        raise AssertionError(f'{selector}: no matching region')
    return matches if index else [set().union(*matches)]


def assert_outcomes(image, frame, case, expected_full):
    if image['full'] != expected_full:
        raise AssertionError('accepted full-frame tags changed')
    for selector, (present, absent) in case['checks'].items():
        for tags in select(image, selector):
            missing, extra = set(present) - tags, set(absent) & tags
            if missing or extra:
                raise AssertionError(f'{selector}: missing {sorted(missing)}, unexpected {sorted(extra)}')
    for selector, only in case.get('only', {}).items():
        for tags in select(image, selector):
            if tags - set(only):
                raise AssertionError(f'{selector}: outside allowlist: {sorted(tags - set(only))}')
    for tag, source in case.get('provenance', {}).items():
        actual = frame['merged_tags']['provenance'][tag]['source']
        if actual != source and not (source.endswith(':') and actual.startswith(source)):
            raise AssertionError(f'{tag}: expected contribution from {source}, got {actual}')
    for tag, score in case.get('manual', {}).items():
        for row in frame['detectors']['person']:
            if row['general'].get(tag) != score or row['provenance'][tag]['source'] != 'manual_rule':
                raise AssertionError(f'{tag}: missing manual score {score}')


def run_outcome(case, historical, evidence, backend, exporter, directory, expected_full):
    options = historical['options'].copy()
    options['remove_tags_regex'] = backend.parse_regex(options['remove_tags_regex'])
    configs = configured(backend, evidence['frames'][0]['source_settings'],
                         historical['base_rules'], case.get('sources', {'person': case.get('rules', {})}))
    filtered = backend.apply_filters(evidence, options, configs)
    _, encoded = backend.outputs(filtered)
    output, _ = exporter.save_region_results(
        encoded, directory, case['name'], write_mode='error', **OUTPUT_OPTIONS)
    image, = json.loads(Path(output).read_bytes())['images']
    assert_outcomes(image, filtered['frames'][0], case, expected_full)


def load_fixtures():
    historical, evidence, backend, exporter, model = historical_suite.load_suite()
    entry = historical['cases'][0]
    baseline = json.loads(historical_suite.checked_bytes(
        HISTORICAL / entry['expected'], entry['expected_sha256']))['images'][0]['full']
    fixtures = {'single': evidence, 'dual': parallel_evidence(evidence, backend),
                'boundary': boundary_evidence(evidence, backend)}
    return historical, fixtures, backend, exporter, model, baseline


def run(cases, directory):
    historical, fixtures, backend, exporter, model, baseline = load_fixtures()
    failures = 0
    with ExitStack() as stack:
        historical_suite.forbid_inference(stack, backend, model)
        for suite, case in cases:
            try:
                run_outcome(case, historical, fixtures[suite], backend, exporter, directory, baseline)
            except Exception as error:
                failures += 1
                print(f'FAIL {case["name"]}: {error}', flush=True)
            else:
                print(f'PASS {case["name"]}', flush=True)
    print(f'{len(cases) - failures}/{len(cases)} passed', flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', help='Run only the named scenarios')
    parser.add_argument('--output-dir', type=Path, help='Keep actual JSON in a new directory')
    args = parser.parse_args()
    cases = scenarios()
    if args.case:
        unknown = set(args.case) - {case['name'] for _, case in cases}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = [(suite, case) for suite, case in cases if case['name'] in args.case]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        return run(cases, args.output_dir)
    with tempfile.TemporaryDirectory(prefix='wdv3-filter-outcomes-') as directory:
        return run(cases, directory)


if __name__ == '__main__':
    sys.exit(main())
