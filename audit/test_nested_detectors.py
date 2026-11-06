"""Ordinary detector stacks attached to Region Tag Rules, without stage nodes."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from audit.test_comfy_evidence import pipeline, Bundle
from audit.test_comfy_nodes import runtime, nodes
from audit.test_cascade_runtime import cascade

nested = importlib.import_module('wdv3_comfy_test.comfy_nested')
config = importlib.import_module('wdv3_comfy_test.comfy_config')


def stack(name, children=None):
    return ([nodes.DetectorConfig(name, name+'.pt', children=children)], {})


def three_levels(route='separate'):
    eyes = stack('eyes')
    face = stack('face', nested.child_settings(eyes, tag_route=route))
    return stack('body', nested.child_settings(face, tag_route=route))


def detect(images, stage):
    w, h = images[0].size
    boxes = [[0, 0, w//2, h], [w//2, 0, w, h]] if stage.parent_stage is None else [[0, 0, w, h]]
    if stage.kind == 'sam':
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :w//2] = True
        return [[dict(bbox=[0, 0, w//2, h], confidence=.9, mask=mask, segmentation_score=.9)]]
    return [[dict(bbox=box, confidence=.9) for box in boxes]]


class NestedTests(unittest.TestCase):
    def test_flat_config_byte_shape_and_nested_roundtrip(self):
        ordinary = nodes.DetectorConfig('body', 'body.pt')
        self.assertEqual(ordinary.to_dict(), asdict(ordinary))
        roots = three_levels('merge_to_parent')
        saved = config.config_data(detectors=roots)
        loaded = config.config_values(json.loads(json.dumps(saved)))[1]
        self.assertEqual(nested.build_plan(roots), nested.build_plan(loaded))
        self.assertNotIn('cascade_plan', saved)
        self.assertNotIn('children', roots[0][0]._children['detectors'][0]['children']['detectors'][0])

    def test_standard_tagger_collect_and_automatic_merge(self):
        roots = three_levels('merge_to_parent')
        with patch.object(cascade, 'DetectorAdapters', return_value=detect):
            result = nodes.WDTagger().tag(torch.ones(1, 12, 20, 3), runtime(), detectors=roots)
            record = json.loads(result['result'][1][0])
            self.assertEqual(set(record['detectors']), {'body', 'body/face', 'body/face/eyes'})
            self.assertEqual(len(record['detectors']['body/face/eyes']), 2)
            self.assertTrue(record['cascade']['routed'])
            with Image.new('RGB', (20, 12), 'white') as image:
                evidence = pipeline.collect([image], runtime(), detectors=roots)
        local = pipeline.filter_tag_sets(evidence)
        merged = pipeline.merge_tag_sets(local)
        self.assertTrue(merged['frames'][0]['cascade']['routed'])
        changed = three_levels('separate')
        changed[0][0].region_gen_threshold = .8
        alternate = pipeline.merge_tag_sets(pipeline.filter_tag_sets(evidence, configs=changed[0]))
        self.assertEqual(alternate['frames'][0]['detectors']['body/face'][0]['tag_route'], 'separate')
        self.assertEqual(Bundle.from_json(merged.to_json()), merged)
        self.assertEqual(pipeline.merge_results(merged, merged)['frames'][0]['merged_tags'], merged['frames'][0]['merged_tags'])

    def test_same_child_names_under_different_parents_and_duplicates(self):
        children = nested.child_settings(stack('face'))
        roots = ([nodes.DetectorConfig('a', 'a.pt', children=children),
                  nodes.DetectorConfig('b', 'b.pt', children=children)], {})
        value, _ = nested.build_plan(roots)
        self.assertEqual({s['stage_id'] for s in value['stages']}, {'a', 'b', 'a/face', 'b/face'})
        with self.assertRaisesRegex(ValueError, 'unique'):
            nested.child_settings(([nodes.DetectorConfig('face', 'a.pt'), nodes.DetectorConfig('face', 'b.pt')], {}))
        with self.assertRaisesRegex(ValueError, 'SAM'):
            nested.child_settings(sources=[{'name': 'x'}])

    def test_sam_parent_and_mixed_child_stack(self):
        from audit.test_comfy_processing import source
        sampler = dict(checkpoint='fake', device='cpu')
        child_source = source()
        child_source['name'] = 'detail'
        children = nested.child_settings(stack('face'), sampler, [child_source], tag_route='merge_to_parent')
        parent = source()
        parent['rules']['children'] = children
        with patch.object(cascade, 'DetectorAdapters', return_value=detect), Image.new('RGB', (20, 12), 'red') as image:
            evidence = nested.collect_nested([image], runtime(), sam_model=sampler, sources=[parent])
        result = pipeline.apply_filters(evidence)
        self.assertEqual(set(result['frames'][0]['detectors']), {'sam_person', 'sam_person/face', 'sam_person/detail'})
        self.assertTrue(result['frames'][0]['detectors']['sam_person/detail'][0]['mask_png'])

    def test_depth_cycle_and_reuse_only(self):
        cyclic = dict(schema_version=1, detectors=[])
        cyclic['detectors'].append(dict(name='loop', model_path='x.pt', children=cyclic))
        with self.assertRaisesRegex(ValueError, 'cycle'):
            nested.validate_children(cyclic)
        value = nested.child_settings(stack('leaf'))
        for index in range(9):
            # Check the compiler directly so malformed external JSON cannot bypass limits.
            value = dict(schema_version=1, detectors=[dict(name=f'n{index}', model_path='x.pt', children=value)],
                         sam_model=None, sam_sources=[], tag_route='separate')
        with self.assertRaisesRegex(ValueError, 'max_depth'):
            nested.validate_children(value)
        with patch.object(cascade, 'DetectorAdapters', return_value=detect), Image.new('RGB', (20, 12), 'red') as image:
            with self.assertRaisesRegex(cascade.CascadeExecutionError, 'reuse_only'):
                nested.collect_nested([image], runtime(), detectors=three_levels(), options={'cache_mode': 'reuse_only'})


if __name__ == '__main__':
    unittest.main()
