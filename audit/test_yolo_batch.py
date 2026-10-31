"""YOLO-only orchestration must send real bounded WD tensor batches."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from Scripts.configs import DetectorConfig, ScriptOptions
from Scripts.main import BatchTagging, _BatchResources
from Scripts.tagging import process_image_with_multiple_detectors
from Scripts.wdv3_timm import LabelData


class RecordingWD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, batch):
        self.batches.append(batch.detach().cpu().clone())
        values = batch.mean(dim=(1, 2, 3)) * 8 - 4
        return torch.stack([values, -values], dim=1)


def transform(image):
    return torch.tensor(np.asarray(image.resize((4, 4))).copy(), dtype=torch.float32).permute(2, 0, 1) / 255


class YoloBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "input.png"
        with Image.new("RGB", (12, 4)) as image:
            for index, color in enumerate(["white", "black", "white"]):
                image.paste(color, (index * 4, 0, (index + 1) * 4, 4))
            image.save(self.path)
        self.labels = LabelData(["bright", "dark"], [], [0, 1], [])
        self.model = RecordingWD()
        self.detector = DetectorConfig("objects", "fixture.pt")
        self.boxes = [[0., 0., 4., 4.], [4., 0., 8., 4.], [8., 0., 12., 4.]]
        self.options = ScriptOptions(self.root, region_batch_size=2, save_txt=False, output_file=None)

    def yolo(self, *_, **__):
        return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([box])) for box in self.boxes])]

    def run_batch(self, batch_size=2):
        resources = _BatchResources(self.model, transform, self.labels, [self.detector],
                                    {"fixture.pt": self.yolo})
        with patch("Scripts.main._load_batch_resources", return_value=resources):
            return BatchTagging(replace(self.options, region_batch_size=batch_size))

    def test_batch_tagging_without_sam_routes_rows_and_partial_batch(self):
        detailed, merged = self.run_batch()
        self.assertEqual([len(batch) for batch in self.model.batches], [1, 2, 1])
        regions = detailed[0]["detectors"]["objects"]
        self.assertEqual([region["region_id"] for region in regions], [0, 1, 2])
        self.assertEqual([region["bbox"] for region in regions], self.boxes)
        self.assertEqual([list(region["general"])[0] for region in regions], ["bright", "dark", "bright"])
        self.assertTrue(all(set(region) == {"region_id", "bbox", "caption", "taglist", "general",
                                           "character", "ratings", "provenance"} for region in regions))
        self.assertEqual(len(merged), 1)
        self.model.batches.clear()
        serial, _ = self.run_batch(1)
        self.assertEqual([len(batch) for batch in self.model.batches], [1, 1, 1, 1])
        self.assertEqual(detailed, serial)

    def test_no_detections_only_tags_full_image(self):
        self.boxes = []
        detailed, _ = self.run_batch()
        self.assertEqual(detailed[0]["detectors"]["objects"], [])
        self.assertEqual([len(batch) for batch in self.model.batches], [1])

    def test_regional_thresholds_and_rules_apply_to_each_row(self):
        self.detector = DetectorConfig("objects", "fixture.pt", region_gen_threshold=.99,
                                       add_tags_to_region={"manual": .8})
        detailed, _ = self.run_batch()
        self.assertIn("bright", detailed[0]["full_image"]["general"])
        for region in detailed[0]["detectors"]["objects"]:
            self.assertEqual(region["general"], {"manual": .8})
            self.assertEqual(region["caption"], "manual")

    def test_detector_batches_keep_independent_rules(self):
        strict = DetectorConfig("strict", "fixture.pt", region_gen_threshold=.99)
        result = process_image_with_multiple_detectors(self.path, self.model, transform, self.labels,
            [self.detector, strict], {"fixture.pt": self.yolo}, .35, .75, torch.device("cpu"),
            region_batch_size=2)
        self.assertEqual([len(batch) for batch in self.model.batches], [1, 2, 1, 2, 1])
        self.assertTrue(all(region["general"] for region in result["detectors"]["objects"]))
        self.assertTrue(all(not region["general"] for region in result["detectors"]["strict"]))

    def test_failed_later_chunk_closes_all_crops(self):
        crops = [Image.new("RGB", (4, 4)) for _ in range(3)]
        regions = [(crop, box, "objects") for crop, box in zip(crops, self.boxes)]
        calls = []
        def fail_later(batch):
            calls.append(len(batch))
            if len(calls) == 3:
                raise RuntimeError("WD failed")
            return torch.zeros((len(batch), 2))
        with patch("Scripts.tagging.extract_regions_with_detector", return_value=regions):
            with self.assertRaisesRegex(RuntimeError, "WD failed"):
                process_image_with_multiple_detectors(self.path, fail_later, transform, self.labels,
                    [self.detector], {"fixture.pt": self.yolo}, .35, .75, torch.device("cpu"),
                    region_batch_size=2)
        self.assertEqual(calls, [1, 2, 1])
        for crop in crops:
            with self.assertRaises(ValueError):
                crop.getpixel((0, 0))

    def test_invalid_batch_rejected_before_loading_models(self):
        for value in [0, -1, True, 1.5]:
            with self.subTest(value=value), patch("Scripts.main._load_batch_resources") as load:
                with self.assertRaises(ValueError):
                    BatchTagging(replace(self.options, region_batch_size=value))
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
