import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from audit.comfy_cpu_cases import CASES, build_prompt, check_report


class ComfyCPUCaseTests(unittest.TestCase):
    def test_graphs_wire_cascade_without_a_top_level_sam(self):
        kinds = ('LoadImage', 'WDV3ModelLoader', 'WDV3SAMLoader', 'WDV3SAMSource',
                 'WDV3RegionRules', 'WDV3YOLODetector', 'WDV3CollectEvidence',
                 'WDV3Inspect', 'WDV3ExportReport')
        catalog = {kind: {'input': {'required': {}}} for kind in kinds}
        for scenario in CASES:
            with self.subTest(scenario=scenario):
                graph, output = build_prompt(scenario, catalog, Path('models'), Path('out'))
                nodes = {node['class_type']: node['inputs'] for node in graph.values()}
                evidence = nodes['WDV3CollectEvidence']
                self.assertEqual('detectors' in evidence, CASES[scenario]['yolo'])
                self.assertEqual('sam_model' in evidence, scenario == 'sam-cpu')
                self.assertEqual('WDV3RegionRules' in nodes, scenario == 'yolo-sam-cpu')
                self.assertEqual(graph[output]['class_type'], 'WDV3ExportReport')
                if scenario == 'yolo-sam-cpu':
                    self.assertIn('child_sources', nodes['WDV3RegionRules'])

    def test_report_invariants(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            child = {'mask_asset': 'mask.png', 'bbox': [3, 3, 6, 6], 'local_bbox': [1, 1, 4, 4],
                     'parent_region_uid': 'cat', 'detector_input_bbox': [2, 2, 8, 8]}
            parent = {'class_id': 15, 'bbox': [2, 2, 8, 8], 'region_uid': 'cat'}
            detectors = {'yolo': [parent], 'sam': [child], 'yolo/sam': [child]}

            def write_report():
                path.write_text(json.dumps({'frames': [{'image_size': [10, 10],
                                                        'detectors': detectors}]}))

            def write_mask(size=(10, 10), filled=True):
                mask = Image.new('L', size)
                if filled:
                    mask.paste(255, (3, 3, 6, 6))
                mask.save(path.with_name('mask.png'))

            write_report()
            write_mask()
            for scenario in CASES:
                check_report(scenario, path, (10, 10))
            write_mask(filled=False)
            with self.assertRaisesRegex(ValueError, 'Empty SAM'):
                check_report('sam-cpu', path, (10, 10))
            write_mask((9, 9))
            with self.assertRaisesRegex(ValueError, 'full-frame PNG'):
                check_report('sam-cpu', path, (10, 10))
            write_mask()
            child.pop('parent_region_uid')
            write_report()
            with self.assertRaisesRegex(ValueError, 'no YOLO parent'):
                check_report('yolo-sam-cpu', path, (10, 10))
