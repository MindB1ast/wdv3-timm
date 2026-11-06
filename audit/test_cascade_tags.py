"""Origin-aware routing, global policies and repeatable merges."""
from copy import deepcopy
import importlib
import unittest

from PIL import Image
from audit.test_comfy_evidence import pipeline, Bundle
from audit.test_comfy_nodes import runtime
from audit.test_cascade import plan, stage
from audit.test_cascade_runtime import cascade, scores

tags = importlib.import_module('wdv3_comfy_test.comfy_cascade_tags')


def fixture():
    value = plan(stage('body'), stage('face', 'body', tag_route='merge_to_parent'),
                 stage('eyes', 'face', tag_route='merge_to_parent'))
    value['stages'][0]['detector'].update(full_image_only_tags=[], remove_tags_from_region=['blocked'])
    value['stages'][1]['detector'].update(full_image_only_tags=[])
    value['stages'][2]['detector'].update(full_image_only_tags=None, only_from_source=['blue_eyes'])
    def detect(images, s):
        return [[dict(bbox=[0, 0, max(1, images[0].width-1), max(1, images[0].height-1)], confidence=.9)]]
    with Image.new('RGB', (8, 8), 'white') as image:
        evidence = cascade.collect_cascade([image], runtime(), value, detect=detect, score=scores)
    data = evidence.data()
    frame = data['frames'][0]
    frame['full_image']['general'] = {'scene': .9, 'skirt': .9}
    for name, general in [('body', {'skirt': .8}), ('face', {'black_skirt': .9}),
                          ('eyes', {'blue_eyes': .95, '1girl': .99, 'blocked': .9})]:
        frame['detectors'][name][0]['general'] = general
    return Bundle.pack('evidence', data)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.evidence = fixture()
        self.local = pipeline.filter_tag_sets(self.evidence)

    def test_explicit_route_preserves_raw_ratings_and_ancestors(self):
        with self.assertRaisesRegex(ValueError, 'Route Cascade Tags'):
            pipeline.merge_tag_sets(self.local)
        routed = tags.route_cascade_tags(self.local)
        frame = routed['frames'][0]
        body = frame['detectors']['body'][0]
        self.assertIn('blue_eyes', body['aggregate']['general'])
        self.assertNotIn('blocked', body['aggregate']['general'])
        self.assertEqual(body['general'], {'skirt': .8})
        self.assertEqual(len(body['aggregate']['origin_candidates']['general']['blue_eyes'][0]['path']), 3)
        self.assertEqual(routed, tags.route_cascade_tags(routed))
        result = pipeline.merge_tag_sets(routed)
        merged = result['frames'][0]['merged_tags']
        self.assertIn('blue_eyes', merged['general'])
        self.assertNotIn('1girl', merged['general'])
        self.assertEqual(merged['provenance']['blue_eyes']['origin_source'], 'eyes')
        self.assertEqual(merged['ratings'], self.local['frames'][0]['full_image']['ratings'])
        self.assertEqual(result.data()['evidence'], self.local.data()['evidence'])
        self.assertEqual(self.evidence['frames'][0]['detectors']['body'][0]['general'], {'skirt': .8})
        self.assertEqual(Bundle.from_json(result.to_json()), result)

    def test_aggregate_cleanup_and_separate_descendant(self):
        value = deepcopy(self.local['frames'][0]['cascade']['plan'])
        value['stages'][0]['aggregate_operations'] = [dict(textual_enabled=True)]
        next(s for s in value['stages'] if s['stage_id'] == 'eyes')['tag_route'] = 'separate'
        routed = tags.route_cascade_tags(self.local, value)
        body = routed['frames'][0]['detectors']['body'][0]
        self.assertNotIn('skirt', body['aggregate']['general'])
        self.assertIn('skirt', body['general'])
        self.assertNotIn('blue_eyes', body['aggregate']['general'])
        self.assertIn('blue_eyes', pipeline.merge_tag_sets(routed)['frames'][0]['merged_tags']['general'])
        merged = pipeline.merge_tag_sets(routed)
        again = pipeline.merge_results(merged, merged)
        self.assertEqual(merged['frames'][0]['merged_tags'], again['frames'][0]['merged_tags'])
        self.assertEqual(merged['frames'][0]['detectors'], again['frames'][0]['detectors'])

    def test_merge_rejects_route_geometry_and_policy_conflicts(self):
        result = pipeline.merge_tag_sets(tags.route_cascade_tags(self.local))
        other = result.data()
        other['frames'][0]['cascade']['plan']['stages'][-1]['tag_route'] = 'separate'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            pipeline.merge_results(result, Bundle.pack('results', other))
        other = result.data()
        other['evidence']['payload']['frames'][0]['detectors']['eyes'][0]['bbox'][0] = 1
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            pipeline.merge_results(result, Bundle.pack('results', other))

    def test_full_confirmation_and_global_remove(self):
        data = self.evidence.data()
        settings = data['frames'][0]['source_settings'][-1]
        settings.update(exclude_from_region=['blue_eyes'], remove_tags_from_full=['scene'])
        local = pipeline.filter_tag_sets(Bundle.pack('evidence', data), {'remove_tags': 'black_skirt'})
        merged = pipeline.merge_tag_sets(tags.route_cascade_tags(local))['frames'][0]['merged_tags']['general']
        self.assertNotIn('blue_eyes', merged)
        self.assertNotIn('scene', merged)
        self.assertNotIn('black_skirt', merged)

    def test_disabled_parent_still_receives_and_siblings_do_not_mix(self):
        data = self.evidence.data()
        body = data['frames'][0]['detectors']['body'][0]
        body.update(tag_enabled=False, general={}, character={}, ratings={})
        data['frames'][0]['source_settings'][0]['add_tags_to_region'] = {'manual': 1.}
        local = pipeline.filter_tag_sets(Bundle.pack('evidence', data))
        routed = tags.route_cascade_tags(local)
        body = routed['frames'][0]['detectors']['body'][0]
        self.assertEqual(body['general'], {})
        self.assertIn('blue_eyes', body['aggregate']['general'])
        self.assertNotIn('manual', body['aggregate']['general'])
        # A postprocessing operation after Route invalidates the derived aggregate.
        post = importlib.import_module('wdv3_comfy_test.Scripts.tag_postprocessing')
        with self.assertRaisesRegex(ValueError, 'Route Cascade Tags'):
            pipeline.merge_tag_sets(post.remove_redundant(routed, {}))


if __name__ == '__main__':
    unittest.main()
