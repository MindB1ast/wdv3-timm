"""Multiple sources and portable configs retain independent settings without model objects."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit.test_comfy_nodes import runtime
from audit.test_comfy_processing import processing, FakeSAM, source, fake_worker

config = import_module("wdv3_comfy_test.comfy_config")
nodes = import_module("wdv3_comfy_test.comfy_nodes")


class ConfigTests(unittest.TestCase):
    def test_low_memory_setting_round_trips_and_rejects_non_boolean(self):
        for enabled in (False,True):
            data={'schema_version':1,'options':{'low_memory_mode':enabled}}
            self.assertIs(config.validate_config(data)['options']['low_memory_mode'],enabled)
        with self.assertRaisesRegex(ValueError,'boolean'):
            config.validate_config({'schema_version':1,'options':{'low_memory_mode':'false'}})

    def test_multi_sam_uses_one_backend_and_one_encoding_per_image_batch(self):
        preset = json.loads(Path("configs/multi_sam.json").read_text())
        sources = preset["sam_sources"]
        self.assertEqual(sources[0]["prompt"], sources[1]["prompt"])
        from PIL import Image
        with Image.new("RGB", (8, 4), "white") as image:
            backend = FakeSAM()
            calls = []
            original = backend._predict_source
            def predict(state, row, count):
                calls.append((row["name"], row["prompt"], row["confidence"]))
                return original(state, row, count)
            backend._predict_source = predict
            with patch("wdv3_comfy_test.Scripts.sam3_adapter.SamWorker", side_effect=fake_worker(backend)) as factory:
                regions = processing.segment_frames([image], {}, sources)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(backend.batches, [1])
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(calls[0][2], calls[1][2])
            # Fixture masks are 16 pixels, so lower the area threshold for this tiny frame.
            for row in sources:
                row["min_mask_area_px"] = 1
            with patch("wdv3_comfy_test.Scripts.sam3_adapter.SamWorker", side_effect=fake_worker(FakeSAM())):
                regions = processing.segment_frames([image], {}, sources)
            _, details = processing.tag_frames([image], runtime(), sam_regions=regions)
        views = details[0]["detectors"]
        self.assertEqual(set(views), {"person_detail", "person_context", "cup"})
        self.assertEqual(views["person_detail"][0]["view"]["mode"], "mask")
        self.assertEqual(views["person_context"][0]["view"]["padding_value"], 32)

    def test_combine_keeps_distinct_rules_and_rejects_duplicate_names(self):
        first = source()
        second = {**source(), "name": "cup", "prompt": "cup", "confidence": .8}
        result = config.combine_sources([[first], [second]])
        result[0]["rules"]["add_tags_to_region"]["new"] = 1
        self.assertNotIn("new", first["rules"]["add_tags_to_region"])
        self.assertEqual(result[1]["confidence"], .8)
        with self.assertRaisesRegex(ValueError, "unique"):
            config.combine_sources([[first], [deepcopy(first)]])
        a, b = nodes.DetectorConfig("person", "shared.pt"), nodes.DetectorConfig("face", "shared.pt")
        handle = object()
        configs, models = config.combine_detectors([([a], {"shared.pt": handle}), ([b], {})])
        self.assertEqual([row.name for row in configs], ["person", "face"])
        self.assertIs(models["shared.pt"], handle)

    def test_roundtrip_omits_model_objects_and_preserves_all_classes(self):
        detector = nodes.DetectorConfig("all", "my.pt")
        detector.classes = None
        payload = config.config_data({"image_batch_size": 2}, ([detector], {"my.pt": object()}), [source()])
        with tempfile.TemporaryDirectory(dir="temp") as directory:
            path = config.save_config(directory, "Мой конфиг", payload)
            saved = json.loads(path.read_text(encoding="utf-8"))
            options, detectors, sources, model = config.config_values(saved)
            self.assertIsNone(detectors[0][0].classes)
            self.assertEqual(detectors[1], {})
            self.assertEqual(options["image_batch_size"], 2)
            self.assertEqual(sources[0]["prompt"], "person")
            self.assertEqual(model["model"], "unsloth/sam3.1")
            with self.assertRaises(FileExistsError):
                config.save_config(directory, "Мой конфиг", payload)
            for name in ("../escape", "x/y", "CON"):
                with self.assertRaises(ValueError):
                    config.save_config(directory, name, payload)

    def test_sam_loader_settings_roundtrip_preserves_requested_device_and_download(self):
        settings = dict(model="unsloth/sam3.1", device="auto", download_missing=False,
                        local_model_path="D:/models/sam3")
        runtime = {"checkpoint": "D:/models/sam3", "device": "cpu", "loader_options": settings}
        saved = config.config_data(sources=[source()], sam_model=runtime)
        self.assertEqual(config.config_values(saved)[3], settings)
        self.assertNotIn("loader_options", json.dumps(saved))
        with self.assertRaises(ValueError):
            config.validate_config({**saved, "sam_model": {"model": "unknown/model"}})

    def test_all_bundled_presets_validate_and_cross_backend_names_cannot_collide(self):
        for path in Path("configs").glob("*.json"):
            config.config_values(json.loads(path.read_text()))
        data = json.loads(Path("configs/multi_all.json").read_text())
        data["sam_sources"][0]["name"] = data["detectors"][0]["name"]
        with self.assertRaisesRegex(ValueError, "unique"):
            config.validate_config(data)

    def test_two_yolo_sources_share_one_lazy_model_load(self):
        from PIL import Image
        a, b = nodes.DetectorConfig("a", "same.pt"), nodes.DetectorConfig("b", "same.pt")
        class Detector:
            def to(self, device):
                return self
            def __call__(self, *args, **kwargs):
                return []
        with tempfile.TemporaryDirectory(dir="temp") as directory, Image.new("RGB", (4, 4)) as image, \
             patch("wdv3_comfy_test.Scripts.model.load_yolo_model", return_value=Detector()) as load, \
             patch.object(nodes, "model_directory", return_value=Path(directory)):
            models = {}
            self.assertEqual(nodes.tag_regions(image, None, [a, b], models, 2), {"a": [], "b": []})
            self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
