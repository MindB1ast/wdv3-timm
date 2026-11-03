"""Explicit regex syntax remains independent of literal tag syntax."""
import unittest
from audit.test_tag_policies import utils
from audit.test_tag_policies import config, nodes
from audit.test_tag_policies import merge, record


class TagRegexTests(unittest.TestCase):
    def test_tagger_has_connections_only_and_sam_batch_comes_from_options(self):
        import torch
        from unittest.mock import patch
        from audit.test_comfy_nodes import runtime
        schema = nodes.WDTagger.INPUT_TYPES()
        for group in schema.values():
            self.assertFalse(any(spec[0] in ('FLOAT', 'INT', 'STRING', 'BOOLEAN') for spec in group.values()))
        images = torch.ones(1, 4, 4, 3)
        with patch('wdv3_comfy_test.comfy_evidence.cached_segment', return_value=[]) as segment, \
             patch('wdv3_comfy_test.comfy_processing.tag_frames', return_value=([''], [{}])):
            nodes.WDTagger().tag(images, runtime(), options={'sam_batch_size': 3},
                                 sam_model={}, sources=[{'name': 'sam'}])
        self.assertEqual(segment.call_args.args[-1], 3)
        saved = config.config_data(options={'sam_batch_size': 3})
        self.assertEqual(config.config_values(saved)[0]['sam_batch_size'], 3)

    def test_output_separators_order_escaping_and_merge(self):
        from audit.test_comfy_evidence import pipeline
        result = utils.format_result(record({'blue_eyes': .9, 'long_hair': .8},
                                           {'hero_(series)': .9}), ['long_hair', 'hero_(series)', 'blue_eyes'])
        frame = {'merged_tags': result, 'settings': {'options': {'replace_underscores': False}}}
        for separator in (', ', ',', ' ', ' | ', ''):
            frame['settings']['options']['tag_separator'] = separator
            self.assertEqual(pipeline.formatted_text(frame), separator.join(['long_hair', r'hero_\(series\)', 'blue_eyes']))
        frame['settings']['options'].update(replace_underscores=True, escape_parentheses=False, tag_separator='; ')
        self.assertEqual(pipeline.formatted_text(frame), 'long hair; hero (series); blue eyes')
        self.assertEqual(result['general'], {'blue_eyes': .9, 'long_hair': .8})
        self.assertEqual(pipeline.formatted_text({'merged_tags': record()}), '')
        self.assertEqual(pipeline.formatted_text({'merged_tags': record({'solo': 1})}), 'solo')

    def test_merge_separator_defaults_and_option_precedence(self):
        import torch
        from PIL import Image
        from audit.test_comfy_evidence import pipeline, runtime
        wd = runtime()
        with Image.new('RGB', (8, 6), 'white') as image:
            evidence = pipeline.collect([image], wd)
        old = pipeline.apply_filters(evidence)
        explicit = pipeline.apply_filters(evidence, {'tag_separator': ', '})
        self.assertEqual(pipeline.outputs(pipeline.merge_results(old, explicit))[0], pipeline.outputs(old)[0])
        with self.assertRaisesRegex(ValueError, 'global filter settings'):
            pipeline.merge_results(old, pipeline.apply_filters(evidence, {'tag_separator': ' '}))
        text = nodes.WDTagger().tag(torch.ones(1, 4, 4, 3), wd, tag_separator=';',
                                    options={'tag_separator': ' | '})['result'][0][0]
        self.assertIn(' | ', text)
        self.assertNotIn(';', text)

    def test_regex_policy_intersections(self):
        rule = nodes.DetectorConfig('eyes', '', allowed_tags_regex=['.*_eyes'],
                                    remove_tags_from_region_regex=['red_.*'],
                                    exclude_from_region_regex=['blue_.*'])
        detail = dict(full_image=record({'solo': .8}), detectors={'eyes': [
            record({'blue_eyes': .9, 'red_eyes': .8, 'hair': .9}, region_id=0)]})
        self.assertEqual(merge(detail, [rule])['general'], {'solo': .8})
        detail['full_image']['general']['blue_eyes'] = .5
        self.assertEqual(merge(detail, [rule])['general']['blue_eyes'], .9)
        rule.allowed_tags_regex = ['no_match']
        self.assertEqual(merge(detail, [rule])['general']['blue_eyes'], .5)
        self.assertEqual(utils.restrict_allowed_tags(record({'hair': 1}), rule)['general'], {})
        rule.allowed_tags = ['hair']
        self.assertIn('hair', utils.restrict_allowed_tags(record({'hair': 1}), rule)['general'])
        rule.full_image_only_tags = []
        rule.allowed_tags_regex = ['.*']
        rule.full_image_only_tags_regex = ['.*hair']
        self.assertNotIn('hair', merge(detail, [rule])['general'])
        rule.remove_tags_from_full_regex = ['solo']
        self.assertNotIn('solo', merge(detail, [rule])['general'])

    def test_regex_owners_and_manual_removal(self):
        rules = [nodes.DetectorConfig('eyes', '', only_from_source_regex=['.*_eyes']),
                 nodes.DetectorConfig('face', '', only_from_source=['blue eyes'])]
        detail = dict(full_image=record({'blue_eyes': .99}), detectors={
            'eyes': [record({'blue_eyes': .8}, region_id=0)],
            'face': [record({'blue_eyes': .7}, region_id=0)]})
        result = merge(detail, rules)
        self.assertEqual(result['general']['blue_eyes'], .8)
        self.assertEqual(result['ratings'], {'general': .8})
        detail['detectors'] = {'eyes': [], 'face': []}
        result = utils.enforce_owners(utils.apply_tag_rules(merge(detail, rules),
                                     before='blue eyes, red eyes'), rules)
        self.assertEqual(result['general'], {})
        spaced = utils.apply_tag_rules(record({'blue eyes': .9}), remove_regex=['.*_eyes'])
        self.assertEqual(spaced['general'], {})
        self.assertEqual(utils.source_owners(rules, ['red eyes'])['red_eyes'], {'eyes'})

    def test_evidence_regex_decisions_and_reuse(self):
        import copy
        from dataclasses import asdict
        from PIL import Image
        from unittest.mock import patch
        from audit.test_comfy_evidence import pipeline, Bundle, runtime
        wd = runtime()
        with Image.new('RGB', (8, 6), 'white') as image:
            evidence = pipeline.collect([image], wd)
        data = evidence.data()
        frame = data['frames'][0]
        region = copy.deepcopy(frame['full_image'])
        region.update(region_id=0, bbox=[0, 0, 4, 4])
        frame['detectors'] = {'eyes': [region]}
        rule = nodes.DetectorConfig('eyes', '', allowed_tags_regex=['.*_tag'])
        frame['source_settings'] = [asdict(rule)]
        evidence = Bundle.pack('evidence', data)
        original = evidence.to_json()
        calls = len(wd.model.batches)
        with patch.object(pipeline, 'collect', side_effect=AssertionError('new inference')):
            accepted = pipeline.apply_filters(evidence, configs=[rule])
            rule.remove_tags_from_region_regex = ['light_.*']
            rejected = pipeline.apply_filters(evidence, configs=[rule])
            with self.assertRaisesRegex(ValueError, 'pattern 1'):
                pipeline.apply_filters(evidence, {'remove_tags_regex': ['(']})
        self.assertEqual(len(wd.model.batches), calls)
        self.assertEqual(evidence.to_json(), original)
        decisions = rejected.data()['frames'][0]['decisions']
        item = next(d for d in decisions if d['tag'] == 'light_tag' and d['source'] == 'eyes')
        self.assertEqual(item['reason'], 'remove_from_regions')
        self.assertIn({'kind': 'regex', 'pattern': 'light_.*', 'action': 'remove_tags_from_region',
                       'source': 'eyes'}, item['matched_rules'])
        self.assertTrue(any(d.get('matched_rules') for d in accepted.data()['frames'][0]['decisions']))
        result = utils.apply_tag_rules(record(), before='blue eyes', after='red eyes',
                                       add={'green_eyes': 1}, remove_regex=['.*_eyes'])
        self.assertEqual(result['general'], {})

    def test_literal_and_pattern_names(self):
        literal = utils.TagMatcher('blue eyes, .*_eyes, character_(series)')
        self.assertIn('blue_eyes', literal)
        self.assertNotIn('red_eyes', literal)
        self.assertIn('character_(series)', literal)
        matcher = utils.TagMatcher(patterns='.*_eyes\na{1,3}\n\n(?i)smile')
        for tag in ('blue_eyes', 'red_eyes', 'aaa', 'SMILE'):
            self.assertIn(tag, matcher)
        self.assertNotIn('blue_eyes_extra', matcher)
        self.assertNotIn('blue_eyes', utils.TagMatcher(patterns=['blue eyes']))
        self.assertNotIn('BLUE_EYES', utils.TagMatcher(patterns=['.*_eyes']))

    def test_preserve_patterns_and_error_context(self):
        self.assertEqual(utils.parse_regex(' a{1,3} \n\n\\w+'), [' a{1,3} ', '\\w+'])
        with self.assertRaisesRegex(ValueError, r'eyes.allowed_tags_regex pattern 2'):
            utils.TagMatcher(patterns=['ok', '('], field='allowed_tags_regex', source='eyes')

    def test_config_patterns_and_separator_roundtrip(self):
        import json
        rule = nodes.DetectorConfig('eyes', 'eyes.pt', allowed_tags_regex=[r'\w+_eyes'])
        data = config.config_data(options={'tag_separator': ' ', 'remove_tags_regex': ['bad.*']},
                                  detectors=([rule], {}))
        options, detectors, _, _ = config.config_values(json.loads(json.dumps(data)))
        self.assertEqual(options['tag_separator'], ' ')
        self.assertEqual(detectors[0][0].allowed_tags_regex, [r'\w+_eyes'])
        self.assertEqual(nodes.DetectorConfig('old', '').allowed_tags_regex, [])
        for value in ('', ' ', ' | '):
            config.validate_config({'schema_version': 1, 'options': {'tag_separator': value}})
        for value in (12, None, []):
            with self.assertRaises(ValueError):
                config.validate_config({'schema_version': 1, 'options': {'tag_separator': value}})
        with self.assertRaisesRegex(ValueError, 'eyes.allowed_tags_regex pattern 1'):
            nodes.DetectorConfig('eyes', '', allowed_tags_regex=['('])


if __name__ == '__main__':
    unittest.main()
