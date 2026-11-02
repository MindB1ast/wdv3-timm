import importlib
import itertools
import json
import unittest
from audit.test_comfy_nodes import nodes, runtime
from audit.test_comfy_processing import processing, source
import torch

merge = importlib.import_module('wdv3_comfy_test.Scripts.preprocess').merge_tags_from_regions
utils = importlib.import_module('wdv3_comfy_test.Scripts.tag_utils')
config = importlib.import_module('wdv3_comfy_test.comfy_config')


def record(general=None, character=None, **extra):
    return dict(general=general or {}, character=character or {}, ratings={'general': .8},
                caption='', taglist='', **extra)


class TagPolicyTests(unittest.TestCase):
    def test_manual_assigned_scores_roundtrip(self):
        self.assertEqual(utils.parse_additions('blue eyes:0.8, smile'),{'blue_eyes':.8,'smile':1})
        for value in ('tag:2','tag:nan','tag:bad'):
            with self.assertRaises(ValueError):utils.parse_additions(value)
    def test_multiple_owners_and_tie_breaking(self):
        configs = [nodes.DetectorConfig(name, '', only_from_source=['blue eyes']) for name in ['eyes', 'face']]
        detail = dict(full_image=record({'blue_eyes': .99}), detectors={
            'eyes': [record({'blue_eyes': .7}, region_id=1), record({'blue_eyes': .7}, region_id=0)],
            'face': [record({'blue_eyes': .7}, region_id=0)],
            'other': [record({'blue_eyes': 1}, region_id=0)]})
        for order in itertools.permutations(configs):
            result = merge(detail, list(order) + [nodes.DetectorConfig('other', '')])
            self.assertEqual(result['general']['blue_eyes'], .7)
            self.assertEqual(result['provenance']['blue_eyes']['source'], 'eyes:0')
        detail['detectors']['eyes'] = []
        self.assertEqual(merge(detail, configs)['provenance']['blue_eyes']['source'], 'face:0')
        detail['detectors']['face'] = []
        self.assertNotIn('blue_eyes', merge(detail, configs)['general'])

    def test_whitelist_manual_tags_ratings_and_confirmation(self):
        rule = nodes.DetectorConfig('eyes', '', allowed_tags=['blue eyes'], only_from_source=['blue eyes'],
                                    exclude_from_region=['blue_eyes'])
        detail = dict(full_image=record({'solo': .9}), detectors={'eyes': [record({'blue_eyes': .8, 'hair': .9}, region_id=0)]})
        self.assertNotIn('blue_eyes', merge(detail, [rule])['general'])
        detail['full_image']['general']['blue_eyes'] = .4
        result = merge(detail, [rule])
        self.assertEqual(result['general'], {'solo': .9, 'blue_eyes': .8})
        filtered = utils.restrict_allowed_tags(utils.apply_tag_rules(record({'blue_eyes': .8}), add={'hair': 1}), rule)
        self.assertNotIn('hair', filtered['general'])
        self.assertEqual(filtered['ratings'], {'general': .8})
        rule.allowed_tags = []
        self.assertIn('hair', utils.restrict_allowed_tags(record({'hair': .8}), rule)['general'])

    def test_global_manual_add_cannot_bypass_absent_owner(self):
        rule = nodes.DetectorConfig('eyes', '', only_from_source=['blue_eyes'])
        merged = merge(dict(full_image=record({'blue_eyes': .9}), detectors={'eyes': []}), [rule])
        final = utils.enforce_owners(utils.apply_tag_rules(merged, before='blue eyes'), [rule])
        self.assertNotIn('blue_eyes', final['general'])
    def test_editable_composition_policy(self):
        detail=dict(full_image=record({'solo':.8}),detectors={'face':[record({'close-up':.9},region_id=0)]})
        rule=nodes.DetectorConfig('face','')
        self.assertNotIn('close-up',merge(detail,[rule])['general'])
        rule.full_image_only_tags=[]
        self.assertIn('close-up',merge(detail,[rule])['general'])

    def test_config_roundtrip_and_old_defaults(self):
        rules = {'only_from_source': ['blue eyes', 'blue_eyes'], 'allowed_tags': ['blue eyes']}
        sam = source(); sam['rules'].update(rules)
        data = config.config_data(detectors=([nodes.DetectorConfig('face', 'face.pt', **rules)], {}), sources=[sam])
        restored = config.config_values(json.loads(json.dumps(data)))
        self.assertEqual(restored[1][0][0].only_from_source, ['blue_eyes'])
        self.assertEqual(restored[2][0]['rules']['allowed_tags'], ['blue_eyes'])
        old = config.config_values({'schema_version': 1, 'detectors': [{'name':'face', 'model_path':'face.pt'}]})
        self.assertEqual(old[1][0][0].allowed_tags, [])


if __name__ == '__main__':
    unittest.main()
