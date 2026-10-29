"""Additional regression contracts for audited defects."""
import os
from pathlib import Path
from unittest.mock import patch

from test_current_behavior import AuditFixture
from Scripts.savingProc import save_tags_to_txt
from Scripts.tag_manager import TagManager


class SafeStorage(AuditFixture):
    def test_replace_failure_preserves_original_and_cleans_temp(self):
        target = self.image.with_suffix('.txt')
        target.write_text('original', encoding='utf-8')
        before = set(self.directory.iterdir())
        with patch('os.replace', side_effect=OSError('replace denied')):
            with self.assertRaises(OSError):
                save_tags_to_txt(str(self.image), {'taglist': 'new'}, append_tags=True)
        self.assertEqual(target.read_text(), 'original')
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_write_failure_preserves_original(self):
        target = self.image.with_suffix('.txt')
        target.write_text('original', encoding='utf-8')
        with patch('os.fsync', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                save_tags_to_txt(str(self.image), {'taglist': 'new'}, append_tags=False)
        self.assertEqual(target.read_text(), 'original')
        self.assertEqual(len(list(self.directory.glob('*.tmp'))), 0)

    def test_mixed_separators_and_canonical_duplicates(self):
        target = self.image.with_suffix('.txt')
        target.write_text('blue_eyes, smile,fang\nblonde hair', encoding='utf-8')
        save_tags_to_txt(str(self.image), {'taglist': 'blue eyes, fang'})
        self.assertEqual(target.read_text(), 'blue eyes, smile, fang, blonde hair')

    def test_tag_manager_counts_only_successful_replacements(self):
        good = self.image.with_suffix('.txt')
        bad = self.directory / 'bad.txt'
        good.write_text('old', encoding='utf-8')
        bad.write_text('old', encoding='utf-8')
        manager = TagManager(str(self.directory))
        manager.add_tags(add_tags_after='new')
        original_replace = os.replace
        def guarded_replace(source, target):
            if Path(target) == bad:
                raise OSError('locked')
            return original_replace(source, target)
        with patch('os.replace', side_effect=guarded_replace):
            self.assertEqual(manager.save_changes(), 1)
        self.assertEqual(bad.read_text(), 'old')
        self.assertEqual(good.read_text(), 'old, new')


class ConsistentTags(AuditFixture):
    def test_manual_character_and_original_model_score(self):
        from Scripts.tag_utils import apply_tag_rules
        from test_current_behavior import tags
        original = tags({'smile': 0.4})
        output = apply_tag_rules(original, add={'smile': 1.0, 'known character': 1.0},
                                 character_names=['known_character'])
        self.assertEqual(output['general']['smile'], 0.4)
        self.assertEqual(output['character'], {'known_character': 1.0})
        self.assertEqual(output['provenance']['known_character']['source'], 'manual_rule')
        self.assertNotIn('known_character', original['character'])

    def test_remove_wins_and_punctuation_survives(self):
        from Scripts.tag_utils import apply_tag_rules, canonical_tag
        from test_current_behavior import tags
        output = apply_tag_rules(tags({'blue_eyes': 0.8}), remove=['blue eyes'],
                                 add={'blue_eyes': 1.0})
        self.assertEqual(output['caption'], '')
        self.assertEqual(canonical_tag('name (series)'), 'name_(series)')

    def test_global_order_and_input_are_preserved(self):
        from Scripts.tag_utils import apply_tag_rules
        from test_current_behavior import tags
        original = tags({'middle': 0.5, 'prefix': 0.8})
        output = apply_tag_rules(original, before='prefix', after='suffix')
        self.assertEqual(output['caption'], 'prefix, middle, suffix')
        self.assertNotIn('suffix', original['general'])


class EvaluationContracts(AuditFixture):
    def test_missing_predictions_report_coverage_and_counts(self):
        from Scripts.analitycs import calculate_tag_metrics
        metrics, diagnostics = calculate_tag_metrics(
            {'a': {'smile'}, 'b': {'blue eyes'}}, {'a': {'smile'}},
            return_diagnostics=True)
        self.assertEqual(metrics['Recall'], 0.5)
        self.assertEqual(diagnostics['coverage'], 0.5)
        self.assertEqual((diagnostics['TP'], diagnostics['FN']), (1, 1))
        self.assertEqual(diagnostics['missing_ids'], ['b'])

    def test_missing_predictions_are_present_in_both_tables(self):
        from Scripts.analitycs import analyze_tagging_errors, analyze_tag_performance
        truth, predicted = {'a': {'smile'}, 'b': {'blue eyes'}}, {'a': {'smile'}}
        self.assertEqual(len(analyze_tagging_errors(truth, predicted)), 2)
        self.assertEqual(int(analyze_tag_performance(truth, predicted)['false_negatives'].sum()), 1)

    def test_empty_labels_are_zero_and_empty_dataset_is_error(self):
        from Scripts.analitycs import calculate_tag_metrics
        self.assertEqual(calculate_tag_metrics({'a': set()}, {'a': set()}),
                         {'Precision': 0.0, 'Recall': 0.0, 'F1-score': 0.0})
        with self.assertRaises(ValueError):
            calculate_tag_metrics({}, {})

    def test_unknown_prediction_id_is_error(self):
        from Scripts.analitycs import calculate_tag_metrics
        with self.assertRaises(ValueError):
            calculate_tag_metrics({'a': {'smile'}}, {'unlabelled': {'smile'}})

    def test_relative_roots_and_recursive_labels_match(self):
        from Scripts.analitycs import load_ground_truth_tags, load_model_tags
        nested = self.directory / 'nested'
        nested.mkdir()
        nested_image = nested / 'example.png'
        nested_image.write_bytes(self.image.read_bytes())
        nested_image.with_suffix('.txt').write_text('blue eyes', encoding='utf-8')
        truth = load_ground_truth_tags(str(self.directory), recursive=True,
                                      image_root=str(self.directory))
        predicted = load_model_tags(([], [{'image_path': str(nested_image),
            'merged_tags': {'taglist': 'blue_eyes'}}]), image_root=str(self.directory))
        self.assertEqual(truth, predicted)
        self.assertEqual(set(truth), {'nested/example'})

    def test_repeated_result_and_outside_root_are_errors(self):
        from Scripts.analitycs import load_model_tags
        sample = {'image_path': str(self.image), 'merged_tags': {'taglist': 'smile'}}
        with self.assertRaises(ValueError):
            load_model_tags(([], [sample, sample]))
        with self.assertRaises(ValueError):
            load_model_tags(([], [sample]), image_root=str(self.directory / 'other'))

    def test_ambiguous_extensions_are_rejected(self):
        from Scripts.analitycs import load_ground_truth_tags
        self.image.with_suffix('.jpg').write_bytes(self.image.read_bytes())
        self.image.with_suffix('.txt').write_text('smile', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_ground_truth_tags(str(self.directory), image_root=str(self.directory))

    def test_historical_f1_is_explicit_and_unchanged(self):
        from Scripts.analitycs import calculate_tag_metrics
        truth = {'a': {'x', 'y'}, 'b': {'z'}}
        predicted = {'a': {'x'}, 'b': {'z'} | {f'noise{i}' for i in range(9)}}
        metrics, diagnostics = calculate_tag_metrics(truth, predicted, return_diagnostics=True)
        self.assertAlmostEqual(metrics['F1-score'], 2 * .55 * .75 / (.55 + .75))
        self.assertIn('harmonic', diagnostics['F1_definition'])


class ImageResources(AuditFixture):
    def test_actual_png_export_without_plotly_renderer(self):
        from PIL import Image
        from Scripts.visualization import visualize_detection_results
        from test_current_behavior import result, region
        data = result(detectors={'face': [region({'smile': 0.8})]})
        destination = self.directory / 'preview.png'
        with patch('plotly.graph_objects.Figure.write_image', side_effect=AssertionError('no renderer')):
            figure = visualize_detection_results(str(self.image), data, str(destination), show=False)
        self.assertIsNotNone(figure)
        with Image.open(destination) as saved:
            self.assertEqual(saved.size, (20, 30))
            self.assertNotEqual(saved.convert('RGB').getpixel((0, 0)), (255, 255, 255))

    def test_empty_detection_closes_file(self):
        from PIL import Image
        import Scripts.detection as detection
        from test_current_behavior import config
        opened, real_open = [], Image.open
        def track(*args, **kwargs):
            image = real_open(*args, **kwargs)
            opened.append(image)
            self.addCleanup(image.close)
            return image
        with patch.object(detection.Image, 'open', side_effect=track):
            self.assertEqual(detection.extract_regions_with_detector(
                str(self.image), config(), lambda *a, **k: []), [])
        self.assertIsNone(opened[0].fp)

    def test_full_inference_exception_closes_file(self):
        from PIL import Image
        import Scripts.tagging as tagging
        opened, real_open = [], Image.open
        def track(*args, **kwargs):
            image = real_open(*args, **kwargs)
            opened.append(image)
            self.addCleanup(image.close)
            return image
        with patch.object(tagging.Image, 'open', side_effect=track), \
             patch.object(tagging, 'tag_image', side_effect=RuntimeError('inference failed')):
            with self.assertRaises(RuntimeError):
                tagging.process_image_with_multiple_detectors(
                    str(self.image), None, None, None, [], {}, .35, .75, None)
        self.assertIsNone(opened[0].fp)

    def test_crops_work_after_source_is_closed(self):
        from types import SimpleNamespace
        import torch
        from Scripts.detection import extract_regions_with_detector
        from test_current_behavior import config
        boxes = [SimpleNamespace(xyxy=torch.tensor([[2, 3, 12, 23]]))]
        regions = extract_regions_with_detector(str(self.image), config(),
            lambda *a, **k: [SimpleNamespace(boxes=boxes)])
        self.image.rename(self.directory / 'renamed.png')
        self.assertEqual(regions[0][0].getpixel((0, 0)), (255, 255, 255))
        regions[0][0].close()



class NotebookCompatibility(AuditFixture):
    def test_numeric_tables_render_with_supported_pandas(self):
        from Scripts import analitycs
        truth = {name: {'skirt', 'jacket', 'eyes'} for name in ('a', 'b', 'c')}
        predictions = {name: {'skirt', 'jacket'} for name in truth}
        errors = analitycs.analyze_tagging_errors(truth, predictions)
        tags = analitycs.analyze_tag_performance(truth, predictions)
        displayed = []
        with patch.object(analitycs, 'display', side_effect=lambda item: displayed.append(item.data)):
            analitycs.display_worst_performing_images(errors, 'method')
            analitycs.display_worst_performing_tags(tags, 'method')
        html = '\n'.join(displayed)
        self.assertIn('<td>0.67</td>', html)
        self.assertIn('<td>0.00</td>', html)
        self.assertIn('eyes', html)
