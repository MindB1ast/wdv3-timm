"""Pixel, identity and selective execution acceptance without external weights."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from audit.test_comfy_evidence import pipeline, Bundle
from audit.test_comfy_nodes import runtime
from audit.test_cascade import stage, plan

cascade = importlib.import_module('wdv3_comfy_test.comfy_cascade')
geometry = importlib.import_module('wdv3_comfy_test.Scripts.cascade_geometry')


def scores(images):
    return [dict(general={'detail': .9}, character={}, ratings={'general': .8}, caption='', taglist='') for _ in images]


def sam_stage(name='body', parent=None, **kw):
    return dict(stage_id=name, parent_stage=parent, kind='sam',
        sam_model=dict(checkpoint='fake', device='cpu'),
        sam_source=dict(name=name, prompt='person', view_mode='mask', padding_px=0, rules={}), **kw)


class GeometryTests(unittest.TestCase):
    def test_exif_and_clipped_half_open_bounds(self):
        regions = importlib.import_module('wdv3_comfy_test.Scripts.regions')
        directory = Path(__file__).resolve().parents[1]/'temp/cascade-tests'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary)/'oriented.png'
            with Image.new('RGB', (10, 6), 'red') as original:
                exif = original.getexif()
                exif[274] = 6
                original.save(path, exif=exif)
            image, meta = regions.canonical_image(path)
            with image, geometry.detector_crop(image, [-.2, 2.1, 8.1, 20])[0] as crop:
                self.assertEqual(meta['exif_orientation'], 6)
                self.assertEqual(image.size, (6, 10))
                self.assertEqual(crop.size, (6, 8))

    def test_three_levels_source_pixels_and_masks(self):
        pixels = np.arange(20*16*3, dtype=np.uint8).reshape(16, 20, 3)
        with Image.fromarray(pixels) as image:
            body = geometry.map_detection('a', 'body', None, 0, image.size, [0, 0, 20, 16],
                                          dict(bbox=[2.2, 3.1, 18, 15]))
            with geometry.detector_crop(image, body['bbox'])[0] as crop:
                np.testing.assert_array_equal(np.array(crop), pixels[3:15, 2:18])
            face = geometry.map_detection('a', 'face', body, 0, image.size, body['bbox'], dict(bbox=[1, 2, 9, 10]))
            mask = np.zeros((8, 8), dtype=bool)
            mask[1:3, 2:5] = True
            eye = geometry.map_detection('a', 'eyes', face, 0, image.size, face['bbox'], dict(bbox=[2, 1, 5, 3], mask=mask))
            self.assertEqual(eye['bbox'], [5, 6, 8, 8])
            np.testing.assert_array_equal(eye['mask'][5:13, 3:11], mask)
            self.assertEqual(eye['depth'], 3)
            self.assertNotIn('part_of', eye)
            with self.assertRaisesRegex(ValueError, 'Empty'):
                geometry.map_detection('a', 'eyes', face, 1, image.size, face['bbox'], dict(bbox=[0, 0, 0, 0]))

    def test_legacy_adaptation_and_bad_links(self):
        value = dict(image_id='a', detectors={'x': [dict(region_id=0)], 'y': [dict(region_id=0)]})
        adapted = geometry.adapt_flat_frame(value)
        self.assertNotEqual(adapted['detectors']['x'][0]['region_uid'], adapted['detectors']['y'][0]['region_uid'])
        self.assertEqual(adapted, json.loads(json.dumps(adapted)))
        adapted['detectors']['x'][0]['parent_region_uid'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'Missing parent'):
            geometry.validate_tree(adapted)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.arange(20*16*3, dtype=np.uint8).reshape(16, 20, 3))
        self.addCleanup(self.image.close)
        self.wd = runtime()

    def test_two_parents_skips_disabled_and_roundtrip(self):
        calls = []
        def detect(images, s):
            calls.append((s.stage_id, images[0].size))
            boxes = [[1, 1, 9, 13], [10, 1, 18, 13]] if s.stage_id == 'body' else [[1, 1, 5, 5]]
            return [[dict(bbox=box, confidence=.8) for box in boxes]]
        score = Mock(side_effect=scores)
        value = plan(stage('body', tag_enabled=False), stage('face', 'body'), stage('eyes', 'face'))
        evidence = cascade.collect_cascade([self.image], self.wd, value, detect=detect, score=score)
        frame = evidence['frames'][0]
        self.assertEqual(len(frame['detectors']['eyes']), 2)
        self.assertNotEqual(*[r['parent_region_uid'] for r in frame['detectors']['eyes']])
        self.assertEqual(frame['detectors']['body'][0]['general'], {})
        self.assertEqual(sum(len(call.args[0]) for call in score.call_args_list), 5)
        self.assertEqual(Bundle.from_json(evidence.to_json()), evidence)
        self.assertEqual(calls[0], ('body', (20, 16)))
        self.assertIn(('face', (8, 12)), calls)
        damaged = deepcopy(frame)
        damaged['detectors']['eyes'][0]['parent_region_uid'] = 'missing'
        with self.assertRaises(ValueError):
            geometry.validate_tree(damaged)

    def test_sam_parent_fill_does_not_touch_child_and_cache(self):
        calls = []
        wd_pixels = []
        def detect(images, s):
            image = images[0]
            calls.append(s.stage_id)
            if s.stage_id == 'body':
                mask = np.zeros((16, 20), dtype=bool)
                mask[2:10, 3:11] = True
                mask[4:7, 5:8] = False
                return [[dict(bbox=[3, 2, 11, 10], mask=mask, confidence=.9)]]
            np.testing.assert_array_equal(np.array(image), np.array(self.image)[2:10, 3:11])
            return [[dict(bbox=[1, 1, 4, 4], confidence=.8)]]
        def score(images):
            wd_pixels.extend(np.array(image) for image in images)
            return scores(images)
        value = plan(sam_stage(), stage('face', 'body'))
        first = cascade.collect_cascade([self.image], self.wd, value, detect=detect, score=score)
        changed = deepcopy(value)
        changed['stages'][0]['wd_view'] = dict(mode='bbox', padding_value=1)
        cascade.collect_cascade([self.image], self.wd, changed, detect=detect, score=score)
        self.assertEqual(calls, ['body', 'face'])
        self.assertTrue(np.all(wd_pixels[0][2:5, 2:5] == 255))
        self.assertEqual(len(wd_pixels), 4)  # first body/face/full; changed body only
        self.assertIn('mask_png', first['frames'][0]['detectors']['body'][0])

    def test_empty_failed_limits_and_cancellation(self):
        detect = Mock(return_value=[[]])
        result = cascade.collect_cascade([self.image], self.wd,
            plan(stage('body'), stage('face', 'body')), detect=detect, score=scores)
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(result['frames'][0]['cascade']['report'][1]['status'], 'skipped')
        bad = Mock(side_effect=RuntimeError('detector broke'))
        with self.assertRaises(cascade.CascadeExecutionError) as error:
            cascade.collect_cascade([self.image], self.wd, plan(stage('body')), detect=bad, score=scores, cache={})
        self.assertEqual(error.exception.partial['failed_frame']['cascade']['report'][-1]['status'], 'failed')
        boxes = [dict(bbox=[0, 0, 4, 4], confidence=.5), dict(bbox=[5, 0, 9, 4], confidence=.9)]
        result = cascade.collect_cascade([self.image], self.wd, plan(stage('body'), max_regions_per_parent=1),
            detect=lambda *_: [boxes], score=scores, cache={})
        self.assertEqual(result['frames'][0]['detectors']['body'][0]['bbox'], [5, 0, 9, 4])
        self.assertEqual(result['frames'][0]['cascade']['report'][0]['truncated'], 1)
        with patch.object(cascade, 'interrupt', side_effect=RuntimeError('cancelled')), self.assertRaisesRegex(RuntimeError, 'cancelled'):
            cascade.collect_cascade([self.image], self.wd, plan(stage('body')), detect=detect, score=scores)

    def test_actual_sam_adapter_and_model_reuse(self):
        from audit.test_comfy_processing import FakeSAM, fake_worker
        backend = FakeSAM()
        with patch('wdv3_comfy_test.Scripts.sam3_adapter.SamWorker', side_effect=fake_worker(backend)):
            result = cascade.collect_cascade([self.image], self.wd, plan(sam_stage()), score=scores)
        self.assertTrue(result['frames'][0]['detectors']['body'][0]['mask_png'])
        backend = FakeSAM()
        with patch('wdv3_comfy_test.Scripts.sam3_adapter.SamWorker', side_effect=fake_worker(backend)) as factory:
            cascade.collect_cascade([self.image], runtime(),
                plan(sam_stage(), sam_stage('face', 'body')), score=scores)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(backend.offloaded)
        detector = Mock()
        detector.to.return_value = detector
        detector.return_value = []
        adapters = cascade.DetectorAdapters(self.wd, {})
        value = cascade.compile_plan(plan(stage('a'), stage('b')))
        with patch('wdv3_comfy_test.Scripts.model.load_yolo_model', return_value=detector) as loader:
            adapters([self.image], value.stages[0])
            adapters([self.image], value.stages[1])
        self.assertEqual(loader.call_count, 1)

    def test_selective_reuse_added_branch_new_frame_and_batch_limits(self):
        called = []
        def detect(images, s):
            called.append(s.stage_id)
            return [[dict(bbox=[0, 0, images[0].width-1, images[0].height-1], confidence=.9)]]
        score = Mock(side_effect=scores)
        base = plan(stage('body'))
        cascade.collect_cascade([self.image], self.wd, base, detect=detect, score=score)
        value = plan(stage('body'), stage('face', 'body'))
        result = cascade.collect_cascade([self.image, self.image], self.wd, value,
            detect=detect, score=score, options={'region_batch_size': 1})
        self.assertEqual(called, ['body', 'face'])
        self.assertTrue(all(len(call.args[0]) <= 1 for call in score.call_args_list))
        self.assertNotEqual(result['frames'][0]['detectors']['body'][0]['region_uid'],
                            result['frames'][1]['detectors']['body'][0]['region_uid'])
        route = deepcopy(value)
        route['stages'][1]['tag_route'] = 'merge_to_parent'
        count = score.call_count
        cascade.collect_cascade([self.image], self.wd, route, detect=detect, score=score)
        self.assertEqual(score.call_count, count)

    def test_changed_bbox_invalidates_only_dependent_branch(self):
        calls = []
        def detect(images, s):
            calls.append(s.stage_id)
            w, h = images[0].size
            shrink = 2 if s.stage_id == 'body' and s.detector['confidence'] > .5 else 1
            return [[dict(bbox=[0, 0, w-shrink, h-shrink], confidence=.9)]]
        value = plan(stage('body'), stage('face', 'body'), stage('other'))
        value['stages'][2]['detector']['model_path'] = 'other.pt'
        score = Mock(side_effect=scores)
        cascade.collect_cascade([self.image], self.wd, value, detect=detect, score=score)
        changed = deepcopy(value)
        changed['stages'][0]['detector']['confidence'] = .8
        cascade.collect_cascade([self.image], self.wd, changed, detect=detect, score=score)
        self.assertEqual(calls.count('other'), 1)
        self.assertEqual(calls.count('face'), 2)
        self.assertEqual(calls.count('body'), 2)

    def test_child_sam_own_fill_toggle(self):
        def detect(images, s):
            w, h = images[0].size
            if s.kind == 'yolo':
                return [[dict(bbox=[1, 1, w-1, h-1], confidence=.9)]]
            mask = np.zeros((h, w), dtype=bool)
            mask[1:h-1, 1:w-1] = True
            mask[3:5, 3:5] = False
            return [[dict(bbox=[1, 1, w-1, h-1], mask=mask, confidence=.9)]]
        value = plan(stage('body', tag_enabled=False), sam_stage('face', 'body'))
        seen = []
        def score(images):
            seen.extend(np.array(image) for image in images)
            return scores(images)
        cascade.collect_cascade([self.image], self.wd, value, detect=detect, score=score)
        self.assertTrue(np.all(seen[0][2:4, 2:4] == 255))
        value['stages'][1]['wd_view'] = dict(mode='bbox')
        cascade.collect_cascade([self.image], self.wd, value, detect=detect, score=score)
        np.testing.assert_array_equal(seen[-1], np.array(self.image)[2:14, 2:18])


if __name__ == '__main__':
    unittest.main()
