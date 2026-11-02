"""Batch identity, SAM masks, exports and opt-in model download without large weights."""
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from audit.test_comfy_nodes import runtime, nodes

processing = import_module("wdv3_comfy_test.comfy_processing")


def source():
    return dict(name="sam_person", prompt="person", confidence=.5, mask_threshold=.5,
                min_mask_area_px=1, max_regions=4, view_mode="mask", padding_px=0,
                rules={"add_tags_to_region": {"from_sam": 1.}})


class FakeSAM:
    def __init__(self):
        self.batches = []
        self.offloaded = False

    def load(self):
        pass

    def offload(self):
        self.offloaded = True

    def _encode_images(self, images):
        self.batches.append(len(images))
        return images

    def _predict_source(self, state, config, count):
        return state

    def _candidates(self, predictions, index, size, confidence):
        mask = np.zeros((size[1], size[0]))
        mask[:, :max(1, size[0] // 2)] = 1
        yield mask, .9, [0, 0, size[0] / 2, size[1]]


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.frames = [Image.new("RGB", size, color) for size, color in
                       [((6, 4), "white"), ((4, 8), "black"), ((8, 4), "white")]]
        for frame in self.frames:
            self.addCleanup(frame.close)

    def test_full_images_use_real_bounded_batches_with_different_sizes(self):
        wd = runtime()
        tags, details = processing.tag_frames(self.frames, wd, options={"image_batch_size": 2})
        self.assertEqual([len(batch) for batch in wd.model.batches], [2, 1])
        self.assertIn("light tag", tags[0])
        self.assertIn("dark tag", tags[1])
        self.assertEqual([d["image_index"] for d in details], [0, 1, 2])

    def test_tag_image_runs_sam_internally_and_exposes_masks_for_viewer(self):
        wd = runtime()
        backend = FakeSAM()
        with patch("wdv3_comfy_test.Scripts.sam_runtime.create_backend", return_value=backend) as factory:
            result = nodes.WDTagger().tag(torch.ones(2, 4, 4, 3), wd,
                sam_model={"backend": "test"}, sources=[source()], sam_batch_size=1)
        tags, details, regions = result["result"]
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(backend.batches, [1, 1])
        self.assertTrue(backend.offloaded)
        self.assertEqual(len(regions), 2)
        self.assertIn("from sam", tags[0])
        self.assertIn("sam_person", json.loads(details[0])["detectors"])
        with self.assertRaisesRegex(ValueError, "either"):
            nodes.WDTagger().tag(torch.ones(2, 4, 4, 3), wd, sam_model={},
                                sources=[source()], sam_regions=regions)

    def test_folder_range_nested_exports_and_no_source_changes(self):
        images = self.root / "input"
        images.mkdir()
        (images / "nested").mkdir()
        for index, frame in enumerate(self.frames):
            frame.save(images / ("nested/2.PNG" if index == 2 else f"{index}.png"))
        files = processing.scan_folder(str(images), True, 0, 0)
        original = [p.read_bytes() for p in files["paths"]]
        out = self.root / "output"
        wd = runtime()
        tags, details = processing.tag_folder(files, wd, {"image_batch_size": 2},
            output_directory=out, save_txt=True, save_json=True)
        self.assertEqual([len(b) for b in wd.model.batches], [2, 1])
        self.assertEqual(len(tags), 3)
        self.assertEqual([p.read_bytes() for p in files["paths"]], original)
        self.assertEqual((out / "nested/2.PNG.txt").read_text(encoding="utf-8"), tags[2])
        self.assertEqual(json.loads(details[2])["image_index"], 2)
        self.assertEqual(len(processing.scan_folder(str(images), True, 1, 1)["paths"]), 1)
        wd = runtime()
        with self.assertRaises(FileExistsError):
            processing.tag_folder(files, wd, {}, output_directory=out, save_txt=True)
        self.assertEqual(wd.model.batches, [])

    def test_sam_regions_are_bound_to_pixels_and_merged_with_rules(self):
        backend = FakeSAM()
        with patch("wdv3_comfy_test.Scripts.sam_runtime.create_backend", return_value=backend):
            records = processing.segment_frames(self.frames, {}, [source()], 2)
        self.assertEqual(backend.batches, [2, 1])
        self.assertTrue(backend.offloaded)
        tags, details = processing.tag_frames(self.frames, runtime(), sam_regions=records)
        self.assertIn("from sam", tags[0])
        self.assertEqual(details[0]["detectors"]["sam_person"][0]["bbox"], [0, 0, 3, 4])
        with Image.new("RGB", self.frames[0].size, "red") as wrong:
            with self.assertRaisesRegex(ValueError, "different image"):
                processing.tag_frames([wrong], runtime(), sam_regions=records[:1])

    def test_sam_offloads_on_prediction_error(self):
        backend = FakeSAM()
        with patch("wdv3_comfy_test.Scripts.sam_runtime.create_backend", return_value=backend), \
             patch.object(backend, "_predict_source", side_effect=RuntimeError("prediction failed")):
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                processing.segment_frames(self.frames, {}, [source()])
        self.assertTrue(backend.offloaded)

    def test_folder_sam_processes_every_group_and_preserves_file_identity(self):
        images = self.root / "input"
        images.mkdir()
        for index, frame in enumerate(self.frames):
            frame.save(images / f"{index}.png")
        files = processing.scan_folder(str(images), False, 0, 0)
        backend = FakeSAM()
        with patch("wdv3_comfy_test.Scripts.sam_runtime.create_backend", return_value=backend):
            tags, details = processing.tag_folder(files, runtime(), {"image_batch_size": 2},
                sam_settings={"backend": "test"}, sources=[source()])
        self.assertEqual(backend.batches, [1, 1, 1])
        self.assertEqual(len(tags), 3)
        for index, encoded in enumerate(details):
            detail = json.loads(encoded)
            self.assertEqual(detail["image_path"], str(files["paths"][index]))
            self.assertIn("sam_person", detail["detectors"])

    def test_download_patterns_exclude_quantized_variants_and_reuse_local(self):
        folder = self.root / "sam"
        def download(**kwargs):
            for name in ("model.safetensors", "config.json", "tokenizer_config.json",
                         "processor_config.json", "tokenizer.json"):
                (Path(kwargs["local_dir"]) / name).write_text("{}", encoding="utf-8")
            self.assertEqual(kwargs["repo_id"], "unsloth/sam3.1")
            self.assertIn("model.safetensors", kwargs["allow_patterns"])
            self.assertNotIn("*.safetensors", kwargs["allow_patterns"])
        with patch("huggingface_hub.snapshot_download", side_effect=download) as fetch:
            with self.assertRaises(FileNotFoundError):
                processing.sam_folder("", folder, False)
            self.assertEqual(fetch.call_count, 0)
            self.assertEqual(processing.sam_folder("", folder, True), folder)
            self.assertEqual(processing.sam_folder(str(folder), folder, False), folder)
            self.assertEqual(fetch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
