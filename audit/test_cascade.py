"""Cascade graph contracts, independent of actual model files."""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest

from Scripts.cascade import compile_plan, combine_plans, region_uid


def stage(name, parent=None, **kw):
    return dict(stage_id=name, parent_stage=parent, kind='yolo',
                detector=dict(name=name, model_path='fake.pt'), **kw)


def plan(*rows, **kw):
    return dict(schema_version=1, stages=list(rows), **kw)


class PlanTests(unittest.TestCase):
    def test_documented_json(self):
        document = (Path(__file__).resolve().parents[1]/'docs/cascaded-regions.md').read_text(encoding='utf-8')
        value = json.loads(re.search(r'```json\n(.*?)\n```', document, re.S).group(1))
        from audit.test_nested_detectors import config, nested
        options, detectors, sources, model = config.config_values(value)
        self.assertEqual(len(nested.build_plan(detectors, model, sources)[0]['stages']), 3)

    def test_topology_branching_and_roundtrip(self):
        value = plan(stage('eyes', 'face'), stage('face', 'body'), stage('body'), stage('hands', 'body'))
        compiled = compile_plan(value)
        self.assertEqual([s.stage_id for s in compiled.stages], ['body', 'face', 'hands', 'eyes'])
        self.assertEqual(compiled, compile_plan(compiled.data()))
        self.assertEqual(combine_plans(value, value), compiled)
        self.assertIsNone(compiled.stages[1].detector['region_gen_threshold'])

    def test_invalid_before_loaders(self):
        bad = [plan(stage('body'), stage('body')), plan(stage('face', 'missing')),
               plan(stage('body', 'face'), stage('face', 'body')),
               plan(stage('body'), stage('face', 'body'), max_depth=1),
               plan(stage('body'), max_regions_per_image=True),
               plan(stage('body', tag_enabled=1)), plan(stage('body', tag_route='unknown')),
               plan(stage('body', tag_route='merge_to_parent')),
               plan(dict(stage_id='sam', kind='sam', sam_source={'prompt': 'person'})),
               dict(schema_version=True, stages=[stage('body')])]
        for value in bad:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                compile_plan(value)
        changed = deepcopy(plan(stage('body')))
        changed['stages'][0]['detector']['confidence'] = .8
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            combine_plans(plan(stage('body')), changed)

    def test_parent_and_frame_identity(self):
        self.assertEqual(len({region_uid(image, 'eyes', parent, 0)
                              for image in ('a', 'b') for parent in ('x', 'y')}), 4)

    def test_yolo_all_classes_and_json_view_types(self):
        value = plan(stage('body'))
        value['stages'][0]['detector']['classes'] = None
        compiled = compile_plan(value)
        self.assertIsNone(compiled.stages[0].detector['classes'])
        self.assertEqual(compiled.data(), json.loads(json.dumps(compiled.data())))


if __name__ == '__main__':
    unittest.main()
