"""Local accepted tags, snapshot semantics and compatibility without model calls."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from audit.test_comfy_evidence import pipeline, Bundle, reports, nodes

post = importlib.import_module('wdv3_comfy_test.Scripts.tag_postprocessing')
config = importlib.import_module('wdv3_comfy_test.comfy_config')
ROOT = Path(__file__).resolve().parents[1]


def evidence_fixture():
    def raw(tags, region_id=None):
        record = dict(general=tags, character={'hero_(series)': .91}, ratings={'general': .8},
                      caption='', taglist='')
        if region_id is not None:
            record.update(region_id=region_id, bbox=[0, 0, 4, 4])
        return record
    with Image.new('RGB', (8, 6), 'white') as image:
        encoded = pipeline.png(image)
    with Image.new('L', (8, 6), 255) as mask:
        mask_png = pipeline.png(mask)
    rules = [nodes.DetectorConfig('clothes', 'fake', full_image_only_tags=[]),
             nodes.DetectorConfig('face', 'sam3', allowed_tags=['1girl', 'skirt', 'miniskirt', 'blue_eyes'],
                                  full_image_only_tags=[], add_tags_to_region={'manual': .8})]
    first = dict(image_id='first', image_size=[8, 6], image_png=encoded,
        full_image=raw({'skirt': .95, 'solo': .99, 'below': .1}),
        detectors={'clothes': [raw({'skirt': .8, 'black_skirt': .9, '1girl': .9}, 0),
                               raw({'skirt': .7, 'miniskirt': .3, '1girl': .9}, 1)],
                   'face': [dict(raw({'skirt': .7, 'miniskirt': .9, 'blue_eyes': .8}, 0), mask_png=mask_png)]},
        source_settings=[asdict(rule) for rule in rules], geometry={'sam_model': {'model_id': 'fake-sam'}})
    second = deepcopy(first)
    second['image_id'] = 'second'
    second['detectors'] = {'clothes': [], 'face': []}
    return Bundle.pack('evidence', dict(model={'model_id': 'test'}, frames=[first, second]))


def operation(**overrides):
    return dict(explicit_enabled=True,
                relations=[{'specific': 'miniskirt', 'general': 'skirt'},
                           {'specific': 'black skirt', 'general': 'skirt'}], **overrides)


class TagSetTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_fixture()
        self.local = pipeline.filter_tag_sets(self.evidence)

    def test_roundtrip_identity_geometry_and_no_aliasing(self):
        self.assertEqual(Bundle.from_json(self.local.to_json()), self.local)
        data = self.local.data()
        self.assertEqual([f['image_id'] for f in data['frames']], ['first', 'second'])
        self.assertEqual(data['frames'][1]['detectors'], {'clothes': [], 'face': []})
        for source in ('clothes', 'face'):
            self.assertEqual(data['frames'][0]['detectors'][source][0]['region_id'], 0)
        self.assertEqual(data['evidence'], json.loads(self.evidence.to_json()))
        self.assertEqual(data['frames'][0]['geometry'], self.evidence.data()['frames'][0]['geometry'])
        data['frames'][0]['full_image']['general'].clear()
        self.assertIn('skirt', self.local.data()['frames'][0]['full_image']['general'])

    def test_legacy_golden_and_shortcut(self):
        golden = json.loads((ROOT / 'audit/fixtures/tag-sets-legacy.json').read_text(encoding='utf-8'))
        for options in ({}, {'remove_tags_regex': ['solo'], 'prefix_tags': 'manual', 'tag_separator': ' | '}):
            key = json.dumps(options, sort_keys=True)
            result = pipeline.apply_filters(self.evidence, options)
            self.assertEqual(result.data()['frames'], golden[key])
            self.assertEqual(result, pipeline.merge_tag_sets(pipeline.filter_tag_sets(self.evidence, options)))

    def test_strict_schema_and_pure_imports(self):
        bad = [[], {'oops': 1}, {'explicit_enabled': 1}, {'schema_version': True}, {'scope': 'other'},
               {'relations': {}}, {'relations': [{'specific': 'a'}]},
               {'relations': [{'specific': '', 'general': 'a'}]},
               {'relations': [{'specific': 'a, b', 'general': 'a'}]},
               {'relations': [{'specific': 'a b', 'general': 'a_b'}]},
               {'relations': [{'specific': 'a', 'general': 'b'}, {'specific': 'b', 'general': 'a'}]},
               {'sources': 'face'}, {'protected_tags': [4]}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                post.validate_operation(value)
        program = "import sys; from Scripts import tag_postprocessing; assert not {'torch','numpy','comfy_api','transformers','ultralytics'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT, check=True)
        before = self.local.to_json()
        with self.assertRaisesRegex(ValueError, 'Unknown.*missing'):
            post.remove_redundant(self.local, operation(sources=['missing']))
        self.assertEqual(self.local.to_json(), before)
        same = post.validate_operation(operation())
        duplicate = operation()
        duplicate['relations'] += [{'specific': 'black_skirt', 'general': 'skirt'}]
        self.assertEqual(post.validate_operation(duplicate), same)
        with self.assertRaises(ValueError):
            post.remove_redundant(self.local, [])

    def test_locality_protection_rejected_witness_and_raw_preservation(self):
        before = self.local.to_json()
        cleaned = post.remove_redundant(self.local, operation(scope='regions'))
        frame = cleaned.data()['frames'][0]
        self.assertIn('skirt', frame['full_image']['general'])
        self.assertNotIn('skirt', frame['detectors']['clothes'][0]['general'])
        self.assertIn('skirt', frame['detectors']['clothes'][1]['general'])  # miniskirt below threshold
        self.assertNotIn('skirt', frame['detectors']['face'][0]['general'])
        self.assertTrue(all('1girl' in r['general'] for r in frame['detectors']['clothes']))
        self.assertEqual(self.local.to_json(), before)
        for name, rows in frame['detectors'].items():
            for index, record in enumerate(rows):
                original = self.local.data()['frames'][0]['detectors'][name][index]
                for field in ('character', 'ratings', 'raw_scores', 'bbox'):
                    self.assertEqual(record[field], original[field])
        protected = post.remove_redundant(self.local, operation(protected_tags=['skirt']))
        self.assertIn('skirt', protected.data()['frames'][0]['detectors']['face'][0]['general'])
        selected = post.remove_redundant(self.local, operation(scope='regions', sources=['face']))
        self.assertEqual(selected.data()['frames'][0]['detectors']['clothes'], self.local.data()['frames'][0]['detectors']['clothes'])

    def test_exact_vs_textual_and_snapshot_all_reasons(self):
        data = self.local.data()
        record = data['frames'][0]['full_image']
        names = ['a', 'b', 'c', 'skirt', 'skirted', 'black_skirt', 'miniskirt', 'skirt_unrelated']
        record['general'] = {name: .9 for name in names}
        record['caption'] = ', '.join(names)
        record['decisions']['general'] = {name: dict(score=.9, status='accepted', reason='threshold_pass') for name in names}
        base = Bundle.pack('tag_sets', data)
        config = operation(scope='full')
        config['relations'] += [{'specific': 'a', 'general': 'b'}, {'specific': 'b', 'general': 'c'}]
        first = post.remove_redundant(base, config)
        config['relations'].reverse()
        self.assertEqual(first, post.remove_redundant(base, config))
        result = first.data()['frames'][0]['full_image']
        self.assertEqual(set(result['general']), set(names) - {'b', 'c', 'skirt'})
        removal = result['decisions']['general']['skirt']['postprocessing']
        self.assertEqual({r['witness'] for r in removal['causes']}, {'black_skirt', 'miniskirt'})
        exact = post.remove_redundant(base, dict(explicit_enabled=True, relations=[{'specific': 'miniskirt', 'general': 'skirt'}]))
        self.assertIn('skirt_unrelated', exact.data()['frames'][0]['full_image']['general'])
        text = post.remove_redundant(base, dict(textual_enabled=True, scope='full'))
        self.assertNotIn('skirt', text.data()['frames'][0]['full_image']['general'])
        self.assertIn('skirted', text.data()['frames'][0]['full_image']['general'])
        self.assertEqual(result['postprocessing_history'][0]['before'], record['general'])
        self.assertEqual(result['postprocessing_history'][0]['after'], result['general'])
        self.assertNotIn('skirt', result['caption'].split(', '))

    def test_disabled_and_empty_operations_preserve_records(self):
        disabled = post.remove_redundant(self.local)
        self.assertEqual(disabled.data()['frames'], self.local.data()['frames'])
        self.assertEqual(post.apply_chain(self.local, []), self.local)

    def test_regex_families_fullmatch_protection_trace_and_legacy(self):
        rules = [dict(id='colors', specific=r'(?:black|white|red|blue)_(?P<base>.+)', general=r'\g<base>'),
                 dict(id='lengths', specific=r'(?:mini|micro)(?P<base>skirt)', general=r'\g<base>')]
        policy = dict(regex_enabled=True, regex_relations=rules, scope='regions')
        original = self.local.to_json()
        cleaned = post.remove_redundant(self.local, policy)
        frame = cleaned.data()['frames'][0]
        self.assertIn('skirt', frame['full_image']['general'])
        self.assertNotIn('skirt', frame['detectors']['clothes'][0]['general'])
        self.assertNotIn('skirt', frame['detectors']['face'][0]['general'])
        self.assertIn('skirt', frame['detectors']['clothes'][1]['general'])
        cause = frame['detectors']['face'][0]['decisions']['general']['skirt']['postprocessing']['causes'][0]
        self.assertEqual(cause['method'], 'regex')
        self.assertEqual(cause['rule']['id'], 'lengths')
        self.assertEqual(cause['rule_digest'], post.fingerprint(rules[1]))
        self.assertEqual(original, self.local.to_json())
        self.assertEqual(cleaned, post.remove_redundant(self.local, {**policy, 'regex_relations': list(reversed(rules))}))
        protected = post.remove_redundant(self.local, {**policy, 'protected_tags': ['skirt']})
        self.assertIn('skirt', protected.data()['frames'][0]['detectors']['face'][0]['general'])
        disabled = post.remove_redundant(self.local, {'regex_relations': rules})
        self.assertEqual(disabled.data()['frames'], self.local.data()['frames'])
        # Existing literal patterns retain their literal meaning.
        self.assertIn('skirt', post.remove_redundant(self.local, {
            'explicit_enabled': True, 'relations': [{'specific': '.*_skirt', 'general': 'skirt'}]
        }).data()['frames'][0]['detectors']['clothes'][0]['general'])
        canonical = post.validate_operation(operation())
        old = {k: v for k,v in canonical.items() if k not in ('regex_enabled', 'regex_relations')}
        self.assertEqual(canonical, post.validate_operation(old))

    def test_regex_validation_expansion_cycles_and_no_implicit_add(self):
        for rule in [dict(id='bad', specific='(', general='skirt'),
                     dict(id='bad', specific='.*', general=r'\g<missing>'),
                     dict(id='bad', specific='.*', general=r'\2'),
                     dict(id='bad', specific='.*', general=''),
                     dict(id='bad', specific='.*', general='a,b')]:
            with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, 'regex_relations.*bad'):
                post.validate_operation(dict(regex_relations=[rule]))
        row = dict(id='same', specific='a', general='b')
        with self.assertRaisesRegex(ValueError, 'duplicate id'):
            post.validate_operation(dict(regex_relations=[row, row]))
        data = self.local.data()
        record = data['frames'][0]['full_image']
        names = ['skirt', 'unrelated_skirt_name', 'black_skirted', 'a', 'b']
        record['general'] = {name: .9 for name in names}
        record['caption'] = ', '.join(names)
        record['decisions']['general'] = {name: dict(score=.9, status='accepted', reason='threshold_pass') for name in names}
        value = Bundle.pack('tag_sets', data)
        policy = dict(regex_enabled=True, scope='full', regex_relations=[
            dict(id='exact_family', specific=r'(?:black|white)_(?P<base>skirt)', general=r'\g<base>')])
        self.assertEqual(post.remove_redundant(value, policy).data()['frames'][0]['full_image']['general'], record['general'])
        cyclic = dict(regex_enabled=True, scope='full', regex_relations=[
            dict(id='ab', specific='a', general='b'), dict(id='ba', specific='b', general='a')])
        with self.assertRaisesRegex(ValueError, 'cycle'):
            post.remove_redundant(value, cyclic)
        self.assertEqual(value.data(), data)
        # A regex may match the general tag itself: that is a no-op, not a deletion.
        self_rule = dict(regex_enabled=True, regex_relations=[dict(id='identity', specific=r'(?P<base>.+)', general=r'\g<base>')])
        self.assertEqual(post.remove_redundant(value, self_rule).data()['frames'][0]['full_image']['general'], record['general'])

    def test_confirmation_uses_processed_full_not_raw(self):
        evidence = self.evidence.data()
        frame = evidence['frames'][0]
        frame['full_image']['general'].update(skirt=.8, miniskirt=.9)
        frame['source_settings'][0]['exclude_from_region'] = ['skirt']
        local = pipeline.filter_tag_sets(Bundle.pack('evidence', evidence))
        cleaned = post.remove_redundant(local, operation(scope='full'))
        result = pipeline.merge_tag_sets(cleaned).data()['frames'][0]
        self.assertNotIn('skirt', result['full_image']['general'])
        self.assertIn('skirt', result['full_image']['raw_scores']['general'])
        decision = next(d for d in result['decisions'] if d['tag']=='skirt' and d['source']=='clothes')
        self.assertEqual(decision['reason'], 'requires_full_confirmation')

    def test_merge_replays_policies_and_rejects_conflicts(self):
        cleaned = post.remove_redundant(self.local, operation(scope='regions'))
        result = pipeline.merge_tag_sets(cleaned)
        self.assertEqual(pipeline.merge_results(result, result).data()['frames'], result.data()['frames'])
        with self.assertRaisesRegex(ValueError, 'postprocessing'):
            pipeline.merge_results(result, pipeline.apply_filters(self.evidence))
        other = post.remove_redundant(self.local, operation(scope='regions', protected_tags=['skirt']))
        with self.assertRaisesRegex(ValueError, 'postprocessing'):
            pipeline.merge_results(result, pipeline.merge_tag_sets(other))
        # Independent branches, equal policies, deterministic union of raw sources.
        data = self.evidence.data()
        results = []
        for source in ('clothes', 'face'):
            branch = deepcopy(data)
            for frame in branch['frames']:
                frame['detectors'] = {source: frame['detectors'][source]}
                frame['source_settings'] = [row for row in frame['source_settings'] if row['name']==source]
            local = pipeline.filter_tag_sets(Bundle.pack('evidence', branch))
            results.append(pipeline.merge_tag_sets(post.remove_redundant(local, operation(scope='regions'))))
        ab = pipeline.merge_results(*results)
        ba = pipeline.merge_results(*reversed(results))
        self.assertEqual(ab.data()['frames'], ba.data()['frames'])
        self.assertNotIn('skirt', ab.data()['frames'][0]['detectors']['face'][0]['general'])

    def test_presets_and_reports_preserve_policy_and_explain_regions(self):
        chain = [post.validate_operation(operation(scope='regions'))]
        preset = config.config_data(postprocessing=chain)
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as directory:
            path = config.save_config(directory, 'postprocessing', preset)
            restored = config.validate_config(json.loads(path.read_text(encoding='utf-8')))
            self.assertEqual(restored['postprocessing'], chain)
            self.assertNotIn('postprocessing', config.validate_config({'schema_version': 1}))
            cleaned = post.apply_chain(self.local, restored['postprocessing'])
            report = reports.make_report(cleaned)
            rows, images, tags = reports.preview_report(report, directory)
            self.assertEqual(report.data()['stage'], 'tag_sets')
            item = next(d for d in rows[0]['detail']['decisions'] if d['source']=='face' and d['tag']=='skirt')
            self.assertEqual(item['reason'], 'redundant_tag')
            self.assertEqual(item['region_id'], 0)
            self.assertEqual(item['postprocessing']['causes'][0]['witness'], 'miniskirt')
            self.assertIn('before', rows[0]['detail']['detectors']['face'][0]['postprocessing_history'][0])
            self.assertEqual(images[0].shape, (1, 6, 8, 3))
            paths = reports.export_report(report, directory, save_txt=False)
            document = json.loads(Path(paths['json']).read_text(encoding='utf-8'))
            self.assertEqual(document['postprocessing'], chain)
            self.assertIn('Before:', Path(paths['html']).read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError, 'Global TXT'):
                reports.export_report(report, directory)
        self.assertNotIn('postprocessing', preset['options'])


if __name__ == '__main__':
    unittest.main()
