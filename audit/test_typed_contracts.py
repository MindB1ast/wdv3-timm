"""Independent numerical and runtime compatibility acceptance for typed results."""
import copy
from decimal import Decimal, localcontext
import json
from pathlib import Path
import runpy
import unittest

import numpy as np
from Scripts.analitycs import analyze_tagging_errors, analyze_tag_performance, calculate_tag_metrics
from Scripts.configs import DetectorConfig, ScriptOptions
from Scripts.preprocess import NumpyEncoder
from Scripts.tag_utils import apply_tag_rules


def expected_metrics(tp, fp, fn):
    # High precision oracle independent of production helpers/constants.
    with localcontext() as context:
        context.prec = 50
        epsilon = Decimal('0.00000001')
        precision = Decimal(tp) / (Decimal(tp + fp) + epsilon)
        recall = Decimal(tp) / (Decimal(tp + fn) + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return tuple(float(value) for value in (precision, recall, f1))


class TypedRuntimeContracts(unittest.TestCase):
    def test_plain_dictionary_round_trip_and_extra_region_fields(self):
        values = runpy.run_path(str(Path(__file__).parent / 'typecheck/positive.py'))
        region = values['region']
        original = copy.deepcopy(region)
        output = apply_tag_rules(region, add={'manual': .9})
        self.assertIs(type(output), dict)
        self.assertEqual(region, original)
        self.assertEqual(output['bbox'], [0., 0., 10., 10.])
        self.assertIsInstance(output['general']['blue_eyes'], np.float32)
        encoded = json.loads(json.dumps(output, cls=NumpyEncoder))
        self.assertEqual(encoded['region_id'], 0)
        self.assertEqual(encoded['provenance']['manual'], {'source': 'manual_rule', 'assigned_score': .9})

    def test_existing_path_and_none_configuration(self):
        options = ScriptOptions(ImageFolder='images', model_folder=Path('models'), output_file=None)
        self.assertIsNone(options.output_file)
        self.assertEqual(options.ImageFolder, 'images')
        detector = DetectorConfig('face', 'face.pt', classes=None, remove_tags_from_full=None)
        self.assertEqual(detector.classes, [0])
        self.assertEqual(detector.remove_tags_from_full, [])


class StableMetricContracts(unittest.TestCase):
    def assert_metrics(self, actual, expected):
        for key, value in zip(('Precision', 'Recall', 'F1-score'), expected):
            self.assertAlmostEqual(actual[key], value, delta=1e-12)

    def test_one_correct_tag_uses_epsilon_in_all_denominators(self):
        self.assert_metrics(calculate_tag_metrics({'a': {'x'}}, {'a': {'x'}}), expected_metrics(1, 0, 0))

    def test_tp_fp_fn_and_empty_agree_across_image_tables(self):
        cases = [(set(), set(), (0, 0, 0)), (set(), {'x'}, (0, 1, 0)),
                 ({'x'}, set(), (0, 0, 1)), ({'x'}, {'x'}, (1, 0, 0)),
                 ({'x', 'y'}, {'x', 'z'}, (1, 1, 1))]
        for truth, predicted, counts in cases:
            with self.subTest(counts=counts):
                metrics, evidence = calculate_tag_metrics({'a': truth}, {'a': predicted}, return_diagnostics=True)
                self.assertEqual(tuple(evidence[key] for key in ('TP', 'FP', 'FN')), counts)
                expected = expected_metrics(*counts)
                self.assert_metrics(metrics, expected)
                row = analyze_tagging_errors({'a': truth}, {'a': predicted}).iloc[0]
                for key, value in zip(('precision', 'recall', 'f1_score'), expected):
                    self.assertAlmostEqual(row[key], value, delta=1e-12)
                self.assertEqual((row['num_false_positives'], row['num_false_negatives']), counts[1:])

    def test_per_tag_counts_and_stabilization(self):
        truth = {'a': {'x'}, 'b': {'x'}, 'c': set()}
        pred = {'a': {'x'}, 'b': set(), 'c': {'x'}}
        row = analyze_tag_performance(truth, pred).iloc[0]
        self.assertEqual(tuple(row[k] for k in ('true_positives', 'false_positives', 'false_negatives')), (1, 1, 1))
        for key, value in zip(('precision', 'recall', 'f1_score'), expected_metrics(1, 1, 1)):
            self.assertAlmostEqual(row[key], value, delta=1e-12)

    def test_summary_keeps_historical_order_of_aggregation(self):
        truth = {'a': {'x', 'y'}, 'b': {'z'}}
        predicted = {'a': {'x'}, 'b': {'z'} | {f'noise{i}' for i in range(9)}}
        first, second = expected_metrics(1, 0, 1), expected_metrics(1, 9, 0)
        p, r = (first[0] + second[0]) / 2, (first[1] + second[1]) / 2
        expected = p, r, 2 * p * r / (p + r + 1e-8)
        metrics = calculate_tag_metrics(truth, predicted)
        self.assert_metrics(metrics, expected)
        self.assertGreater(abs(metrics['F1-score'] - (first[2] + second[2]) / 2), .1)

    def test_missing_predictions_and_policy_diagnostics(self):
        metrics, evidence = calculate_tag_metrics({'a': {'x'}, 'b': {'y'}}, {'a': {'x'}}, return_diagnostics=True)
        self.assertEqual(evidence['coverage'], .5)
        self.assertEqual((evidence['expected'], evidence['processed']), (2, 1))
        self.assertEqual(evidence['epsilon'], 1e-8)
        self.assertEqual(evidence['metric_policy_version'], 'complete-labels-epsilon-v1')
        self.assertEqual(evidence['missing_ids'], ['b'])
        self.assertEqual(set(metrics), {'Precision', 'Recall', 'F1-score'})
        p = expected_metrics(1, 0, 0)[0] / 2
        self.assert_metrics(metrics, (p, p, 2 * p * p / (2 * p + 1e-8)))

    def test_invalid_dataset_and_unknown_ids_still_fail(self):
        for truth, pred in [({}, {}), ({'a': set()}, {'unknown': {'x'}})]:
            with self.assertRaises(ValueError):
                calculate_tag_metrics(truth, pred)
