"""Integration examples captured from the user's filters_test workflow."""
from contextlib import ExitStack
import tempfile
import unittest

from audit.run_region_filter_regressions import load_suite, forbid_inference, run_case


class SavedRegionFilterTests(unittest.TestCase):
    def test_saved_region_filters(self):
        manifest, evidence, backend, exporter, model = load_suite()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            forbid_inference(stack, backend, model)
            for case in manifest['cases']:
                with self.subTest(filter=case['name']):
                    run_case(case, manifest, evidence, backend, exporter, directory)


if __name__ == '__main__':
    unittest.main()
