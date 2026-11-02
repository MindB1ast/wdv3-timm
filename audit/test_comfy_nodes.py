"""Exercise the ComfyUI package loader and real tensor adapter without downloads."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wdv3_comfy_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules["wdv3_comfy_test.comfy_nodes"]
LabelData = sys.modules["wdv3_comfy_test.Scripts.wdv3_timm"].LabelData


class RecordingModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.batches = []
        self.fail = fail
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return super().to(device)

    def forward(self, batch):
        self.batches.append(batch.clone())
        if self.fail:
            raise RuntimeError("inference failed")
        light = batch.mean((1, 2, 3)) * 8 - 4
        return torch.stack([light, -light, torch.ones_like(light) * 3], dim=1)


def transform(image):
    return torch.tensor(np.array(image.resize((4, 4))), dtype=torch.float32).permute(2, 0, 1) / 255


def runtime(fail=False):
    return nodes.WDRuntime(RecordingModel(fail), transform,
                           LabelData(["light_tag", "dark_tag", "hero_(series)"], [], [0, 1], [2]),
                           torch.device("cpu"))


class ComfyTests(unittest.TestCase):
    def test_registration_and_lazy_imports(self):
        self.assertTrue({"WDV3Tagger", "WDV3ModelLoader", "WDV3DetectorLoader"} <= PACKAGE.NODE_CLASS_MAPPINGS.keys())
        for cls in PACKAGE.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))
            self.assertIn("required", cls.INPUT_TYPES())
        self.assertNotIn("wdv3_comfy_test.Scripts.visualization", sys.modules)
        self.assertNotIn("wdv3_comfy_test.Scripts.analitycs", sys.modules)

    def test_batch_outputs_thresholds_rules_and_escaping(self):
        model = runtime()
        images = torch.stack([torch.ones(4, 4, 3), torch.zeros(4, 4, 3)])
        result = nodes.WDTagger().tag(images, model, remove_tags="light tag",
                                     prefix_tags="first", suffix_tags="last")
        texts, details, _ = result["result"]
        self.assertEqual(len(texts), 2)
        self.assertNotIn("light tag", texts[0])
        self.assertIn(r"hero \(series\)", texts[0])
        self.assertTrue(all(t.startswith("first, ") and t.endswith(", last") for t in texts))
        self.assertIn("dark_tag", json.loads(details[1])["merged_tags"]["general"])
        self.assertEqual(model.model.moves[-1], "cpu")
        high = nodes.WDTagger().tag(images[:1], model, general_threshold=1, character_threshold=1)
        self.assertEqual(high["result"][0], [""])

    def test_rgba_white_compositing_and_bgr_channels(self):
        model = runtime()
        nodes.WDTagger().tag(torch.zeros(1, 4, 4, 4), model)
        self.assertTrue(torch.all(model.model.batches[-1] == 1))
        red = torch.zeros(1, 4, 4, 3)
        red[..., 0] = 1
        nodes.WDTagger().tag(red, model)
        batch = model.model.batches[-1]
        self.assertTrue(torch.all(batch[:, 2] == 1))
        self.assertTrue(torch.all(batch[:, :2] == 0))

    def test_regional_bounded_batches_and_merge(self):
        model = runtime()
        cfg = nodes.DetectorConfig("objects", "fake", add_tags_to_region={"regional": 1.0})
        boxes = [[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4]]
        def detector(image, **kwargs):
            self.assertEqual(image.mode, "RGB")
            return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([b])) for b in boxes])]
        result = nodes.WDTagger().tag(torch.ones(1, 4, 4, 3), model,
                                     detectors=([cfg], {"fake": detector}), region_batch_size=2)
        detail = json.loads(result["result"][1][0])
        self.assertEqual([len(b) for b in model.model.batches], [1, 2, 1])
        self.assertEqual(len(detail["detectors"]["objects"]), 3)
        self.assertIn("regional", detail["merged_tags"]["general"])
        self.assertEqual(detail["merged_tags"]["provenance"]["regional"]["region_source"], "objects:0")

    def test_cleanup_on_error_and_input_validation(self):
        model = runtime(fail=True)
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            nodes.WDTagger().tag(torch.zeros(1, 4, 4, 3), model)
        self.assertEqual(model.model.moves[-1], "cpu")
        for images in [torch.zeros(4, 4, 3), torch.zeros(1, 4, 4, 1), torch.zeros(0, 4, 4, 3)]:
            with self.assertRaises(ValueError):
                nodes.WDTagger().tag(images, runtime())
        with self.assertRaises(ValueError):
            nodes.WDDetectorLoader().load('{"sources": []}')
        with self.assertRaisesRegex(ValueError, "unique"):
            nodes.WDDetectorLoader().load('[{"name":"a","model_path":"b"}, {"name":"a","model_path":"c"}]')


if __name__ == "__main__":
    unittest.main()
