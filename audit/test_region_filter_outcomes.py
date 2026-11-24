"""Regression coverage for the separately named Region Tag Rules outcomes."""
from contextlib import ExitStack
import json
import tempfile
import unittest

from audit.build_region_filter_outcomes_workflow import build_workflow
from audit.run_region_filter_outcomes import (
    FIXTURES, historical_suite, load_fixtures, run_outcome, scenarios,
)


class RegionFilterOutcomeTests(unittest.TestCase):
    def test_named_workflow_matches_manifest(self):
        graph = json.loads((FIXTURES / 'workflow.json').read_bytes())
        self.assertEqual(graph, build_workflow())
        names = [case['name'] for _, case in scenarios()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([node['widgets_values_named']['filename'] for node in graph['nodes']
                          if node['type'] == 'WDV3SaveRegionResults'],
                         [case['name'] for suite, case in scenarios() if suite != 'boundary'])

    def test_explicit_outcomes_without_models(self):
        historical, fixtures, backend, exporter, model, baseline = load_fixtures()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            historical_suite.forbid_inference(stack, backend, model)
            for suite, case in scenarios():
                with self.subTest(case=case['name']):
                    run_outcome(case, historical, fixtures[suite], backend, exporter, directory, baseline)
