"""Audit the current public behavior without changing production modules.

HealthyBehavior checks working contracts. DefectContracts deliberately checks
desired behavior against current defects, so the full audit is expected to fail
until those defects are fixed. No expectedFailure/skip decorators are used.
All file writes are confined to temporary directories under temp/audit-scratch.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'temp/matplotlib-cache'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'temp/yolo-settings'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
SCRATCH = ROOT / 'temp/audit-scratch'
SCRATCH.mkdir(parents=True, exist_ok=True)

import torch
from PIL import Image
from Scripts.configs import DetectorConfig, ScriptOptions
from Scripts.preprocess import merge_tags_from_regions, process_tags, save_results_to_json
from Scripts.savingProc import save_tags_to_txt
from Scripts.tag_manager import TagManager
from Scripts.wdv3_timm import LabelData, get_tags, pil_ensure_rgb, pil_pad_square
from Scripts.detection import extract_regions_with_detector, load_detectors_config
from Scripts.analitycs import calculate_tag_metrics, load_model_tags
import Scripts.main as batch_module
import Scripts.tagging as tagging_module
import Scripts.visualization as visualization_module


def tags(general=None, character=None):
    general, character = general or {}, character or {}
    caption = ', '.join([*general, *character])
    return dict(general=general, character=character, ratings={'general': 0.8},
                caption=caption, taglist=caption.replace('_', ' '))


def region(general=None, character=None):
    return dict(region_id=0, bbox=[0, 0, 10, 10], **tags(general, character))


def result(general=None, character=None, detectors=None):
    return dict(image_path='example.png', full_image=tags(general, character),
                detectors=detectors or {})


def config(name='face', **kwargs):
    return DetectorConfig(name=name, model_path='unused.pt', **kwargs)


class AuditFixture(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)
        self.image = self.directory / 'example.png'
        Image.new('RGB', (20, 30), 'white').save(self.image)

    def run_batch(self, **options):
        """Use real BatchTagging; replace expensive upstream model boundaries."""
        image_result = result({'blue_eyes': 0.8})
        image_result['image_path'] = str(self.image)
        fake_model = SimpleNamespace(pretrained_cfg={}, to=lambda _: fake_model)
        replacements = dict(
            ensure_model_folder=lambda path: path,
            download_model_files=lambda *args: None,
            load_model_local_or_remote=lambda *args: fake_model,
            load_labels_local_or_remote=lambda *args: None,
            create_transform=lambda **kwargs: None,
            resolve_data_config=lambda *args, **kwargs: {},
            load_detectors_config=lambda *args: [],
            process_image_with_multiple_detectors=lambda *args: copy.deepcopy(image_result),
        )
        opts = ScriptOptions(ImageFolder=self.directory, model='vit',
                             save_txt=False, append_txt=False, output_file='', **options)
        with patch.multiple(batch_module, **replacements):
            return batch_module.BatchTagging(opts)


class HealthyBehavior(AuditFixture):
    def test_config_defaults_do_not_share_mutable_lists(self):
        first, second = config(), config()
        first.remove_tags_from_full.append('smile')
        self.assertEqual(second.remove_tags_from_full, [])

    def test_process_tags_preserves_unremoved_scores(self):
        self.assertEqual(process_tags({'smile': 0.8, 'helmet': 0.4}, ['helmet']), {'smile': 0.8})

    def test_merge_uses_maximum_confidence(self):
        merged = merge_tags_from_regions(result({'smile': 0.4}, detectors={
            'face': [region({'smile': 0.9})]}), [config()])
        self.assertEqual(merged['general']['smile'], 0.9)
        self.assertEqual(merged['caption'], 'smile')

    def test_composition_tags_from_region_are_excluded(self):
        merged = merge_tags_from_regions(result({'3girls': 0.9}, detectors={
            'face': [region({'1girl': 0.99, 'close-up': 0.9})]}), [config()])
        self.assertEqual(set(merged['general']), {'3girls'})

    def test_region_confirmation_requires_full_image_tag(self):
        merged = merge_tags_from_regions(result({'smile': 0.8}, detectors={
            'face': [region({'smile': 0.7, 'fang': 0.9})]}),
            [config(exclude_from_region=['smile', 'fang'])])
        self.assertEqual(set(merged['general']), {'smile'})

    def test_merge_does_not_mutate_input(self):
        data = result({'smile': 0.8}, detectors={'face': [region({'fang': 0.9})]})
        before = copy.deepcopy(data)
        merge_tags_from_regions(data, [config()])
        self.assertEqual(data, before)

    def test_ratings_are_kept_separate(self):
        merged = merge_tags_from_regions(result({'smile': 0.8}), [])
        self.assertEqual(merged['ratings'], {'general': 0.8})
        self.assertNotIn('general', merged['caption'])

    def test_category_threshold_is_strict(self):
        labels = LabelData(['general', 'smile', 'example_character'], [0], [1], [2])
        caption, _, ratings, character, general = get_tags(
            torch.tensor([0.8, 0.5, 0.75]), labels, 0.5, 0.75)
        self.assertEqual((caption, character, general), ('', {}, {}))
        self.assertIn('general', ratings)

    def test_transparent_image_composites_on_white(self):
        image = pil_ensure_rgb(Image.new('RGBA', (2, 3), (255, 0, 0, 0)))
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_square_padding_preserves_original_pixels(self):
        image = pil_pad_square(Image.new('RGB', (2, 4), 'red'))
        self.assertEqual(image.size, (4, 4))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((1, 0)), (255, 0, 0))

    def test_detection_crops_in_original_coordinates(self):
        calls = []
        def detector(path, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[2, 3, 12, 23]]))])]
        regions = extract_regions_with_detector(str(self.image), config(confidence=0.4), detector)
        self.assertEqual(regions[0][0].size, (10, 20))
        self.assertEqual(regions[0][1], [2, 3, 12, 23])
        self.assertEqual(calls, [{'conf': 0.4, 'classes': [0]}])

    def test_empty_detection_returns_empty_list(self):
        self.assertEqual(extract_regions_with_detector(str(self.image), config(), lambda *a, **k: []), [])

    def test_existing_detector_configuration_loads(self):
        detectors = load_detectors_config(str(ROOT / 'detectors.json'))
        self.assertEqual([d.name for d in detectors], ['person_detector', 'face_detector'])

    def test_json_encoder_handles_float32(self):
        import numpy as np
        destination = self.directory / 'results.json'
        save_results_to_json([{'score': np.float32(0.5)}], str(destination))
        self.assertEqual(json.loads(destination.read_text()), [{'score': 0.5}])

    def test_append_txt_is_idempotent(self):
        payload = {'taglist': 'smile, blue eyes'}
        path = Path(save_tags_to_txt(str(self.image), payload))
        first = path.read_bytes()
        save_tags_to_txt(str(self.image), payload)
        self.assertEqual(path.read_bytes(), first)

    def test_tag_manager_edits_only_on_explicit_save(self):
        txt = self.image.with_suffix('.txt')
        txt.write_text('skirt, black skirt', encoding='utf-8')
        manager = TagManager(str(self.directory))
        manager.remove_redundant_subtags()
        self.assertEqual(txt.read_text(), 'skirt, black skirt')
        self.assertEqual(manager.save_changes(), 1)
        self.assertEqual(txt.read_text(), 'black skirt')

    def test_batch_entry_without_export_returns_two_lists(self):
        detailed, merged = self.run_batch()
        self.assertEqual((len(detailed), len(merged)), (1, 1))
        self.assertEqual(merged[0]['merged_tags']['caption'], 'blue_eyes')
        self.assertFalse(self.image.with_suffix('.txt').exists())

    def test_metrics_for_complete_correct_predictions(self):
        truth = {'a': {'smile'}, 'b': {'blue eyes'}}
        self.assertEqual(calculate_tag_metrics(truth, truth),
                         {'Precision': 1.0, 'Recall': 1.0, 'F1-score': 1.0})


class DefectContracts(AuditFixture):
    def test_empty_detector_must_not_remove_full_tags(self):
        merged = merge_tags_from_regions(result({'eyelashes': 0.8}, detectors={'face': []}),
                                         [config(remove_tags_from_full=['eyelashes'])])
        self.assertIn('eyelashes', merged['general'])

    def test_merge_must_not_depend_on_detector_order(self):
        full = {'eyelashes': 0.8, 'smile': 0.8}
        configs = [config('face', remove_tags_from_full=['eyelashes']),
                   config('person', remove_tags_from_full=['smile'])]
        first = merge_tags_from_regions(result(full, detectors={
            'face': [region()], 'person': [region()]}), configs)
        second = merge_tags_from_regions(result(full, detectors={
            'person': [region()], 'face': [region()]}), configs)
        self.assertEqual(first, second)

    def test_full_character_must_be_in_caption(self):
        merged = merge_tags_from_regions(result(character={'example_character': 0.9}), [])
        self.assertIn('example_character', merged['character'])
        self.assertIn('example_character', merged['caption'].split(', '))

    def test_spaced_filter_must_match_canonical_tag(self):
        merged = merge_tags_from_regions(result(detectors={'face': [region({'blue_eyes': 0.9})]}),
                                         [config(remove_tags_from_region=['blue eyes'])])
        self.assertNotIn('blue_eyes', merged['general'])

    def test_region_caption_must_match_filtered_dictionary(self):
        detector_config = config(remove_tags_from_region=['helmet'])
        predictions = [tags({'smile': 0.8}), tags({'helmet': 0.9, 'smile': 0.8})]
        with Image.open(self.image) as opened, \
             patch.object(tagging_module.Image, 'open', return_value=opened), \
             patch.object(tagging_module, 'tag_image', side_effect=predictions), \
             patch.object(tagging_module, 'extract_regions_with_detector',
                          return_value=[(Image.new('RGB', (10, 10)), [0, 0, 10, 10], 'face')]):
            data = tagging_module.process_image_with_multiple_detectors(
                str(self.image), None, None, SimpleNamespace(names=[], character=[]),
                [detector_config], {'unused.pt': object()}, 0.35, 0.75, torch.device('cpu'))
        face = data['detectors']['face'][0]
        self.assertNotIn('helmet', face['general'])
        self.assertNotIn('helmet', face['caption'].split(', '))

    def test_global_removal_must_update_dictionary(self):
        _, merged = self.run_batch(remove_tags='blue eyes')
        self.assertNotIn('blue_eyes', merged[0]['merged_tags']['caption'])
        self.assertNotIn('blue_eyes', merged[0]['merged_tags']['general'])

    def test_global_addition_must_update_dictionary(self):
        _, merged = self.run_batch(add_tags_before='smile')
        self.assertIn('smile', merged[0]['merged_tags']['caption'])
        self.assertIn('smile', merged[0]['merged_tags']['general'])

    def test_append_read_error_must_preserve_existing_file(self):
        txt = self.image.with_suffix('.txt')
        original = b'\xff\xfe\x00existing-annotation'
        txt.write_bytes(original)
        try:
            save_tags_to_txt(str(self.image), {'taglist': 'smile'}, append_tags=True)
        except (OSError, UnicodeError):
            pass
        self.assertEqual(txt.read_bytes(), original)

    def test_new_export_must_not_duplicate_before_after_tags(self):
        path = Path(save_tags_to_txt(str(self.image), {'taglist': 'prefix, smile, suffix'},
                                    append_tags=False, add_tags_before='prefix', add_tags_after='suffix'))
        self.assertEqual(path.read_text(), 'prefix, smile, suffix')

    def test_missing_predictions_must_not_report_perfect_recall(self):
        metrics = calculate_tag_metrics({'a': {'smile'}, 'b': {'blue eyes'}}, {'a': {'smile'}})
        self.assertLess(metrics['Recall'], 1.0)

    def test_same_basename_must_not_overwrite_another_image(self):
        results = ([], [
            {'image_path': 'first/example.png', 'merged_tags': {'taglist': 'smile'}},
            {'image_path': 'second/example.png', 'merged_tags': {'taglist': 'blue eyes'}},
        ])
        self.assertEqual(len(load_model_tags(results)), 2)

    def test_visualization_save_branch_must_not_raise_name_error(self):
        data = result({'smile': 0.8})
        data['image_path'] = str(self.image)
        with patch.object(visualization_module, 'visualize_detection_results'):
            visualization_module.view_image_results(
                ([data], [{'merged_tags': tags({'smile': 0.8})}]), 0, save_visualization=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
