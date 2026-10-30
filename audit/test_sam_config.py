"""Opt-in configuration acceptance without SAM or model weights."""
import json
from pathlib import Path
import tempfile
import unittest

from Scripts.configs import ScriptOptions
from Scripts.sam_config import load_sam_config, validate_batch_options


class SamConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sources.json"
        self.config = {"schema_version": 1, "model": {"checkpoint": "sam3.pt"},
                       "sources": [{"source_id": "eyes", "name": "sam_eyes", "prompt": "eye"}]}

    def load(self, **kwargs):
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        return load_sam_config(self.path, **kwargs)

    def test_defaults_and_relative_checkpoint(self):
        result = self.load()
        self.assertFalse(result.allow_download)
        self.assertEqual(Path(result.checkpoint), self.path.parent.resolve() / "sam3.pt")
        self.assertEqual(result.sources[0].view.mode, "bbox")
        self.assertEqual(result.sources[0].rules.name, "sam_eyes")
        self.assertIsNone(ScriptOptions("images").sam_config)

    def test_bad_inputs_fail_before_models(self):
        for key, bad in [("prompt", ""), ("confidence", 2), ("mask_threshold", float("nan")),
                         ("max_regions", 0), ("max_regions", True), ("min_mask_area_px", -1),
                         ("background_rgb", [0, 0, 256]), ("view_mode", "wrong"),
                         ("padding", {"value": -1}), ("padding", {"unit": "percent"}),
                         ("region_gen_threshold", 1.1), ("add_tags_to_region", {"eye": -1})]:
            with self.subTest(key=key, bad=bad):
                self.config["sources"][0][key] = bad
                with self.assertRaises(ValueError):
                    self.load()
                self.config["sources"][0].pop(key)
                self.config["sources"][0]["prompt"] = "eye"

    def test_conflicting_sources_and_batch_limits(self):
        with self.assertRaises(ValueError):
            self.load(detector_names=["sam_eyes"])
        self.config["sources"] *= 2
        with self.assertRaises(ValueError):
            self.load()
        for key, value in [("sam_batch_size", 0), ("region_batch_size", True),
                           ("sam_timeout", float("inf"))]:
            with self.assertRaises(ValueError):
                validate_batch_options(ScriptOptions("images", sam_python="python", **{key: value}))

    def test_shipped_example(self):
        result = load_sam_config("sam_sources.example.json")
        self.assertEqual(len(result.sources), 2)


if __name__ == "__main__":
    unittest.main()
