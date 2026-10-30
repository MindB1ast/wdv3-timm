"""Exercise official-backend batching and decoding with CPU contract doubles."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from PIL import Image

from Scripts.sam3_backend import OfficialSam3Backend


class BackendTests(unittest.TestCase):
    def test_native_batch_independent_prompts_and_probability_decoding(self):
        prompts, moves, batch_sizes, dummy_sizes = [], [], [], []
        class Model:
            def __init__(self):
                self.backbone = SimpleNamespace(forward_text=lambda texts, device: {"language": texts[0]})
            def to(self, device):
                moves.append(device)
                return self
            def _get_dummy_prompt(self, num_prompts):
                dummy_sizes.append(num_prompts)
                return num_prompts
            def forward_grounding(self, backbone_out, find_input, **_):
                torch.testing.assert_close(find_input.img_ids, torch.tensor([0, 1]))
                torch.testing.assert_close(find_input.text_ids, torch.tensor([0, 0]))
                self_prompt = backbone_out["language"]
                prompts.append(self_prompt)
                # Two visual frames and two candidates, different logits per frame.
                logits = torch.full((2, 2, 1), 5.0 if self_prompt == "eye" else -5.0)
                masks = torch.full((2, 2, 3, 3), -4.0)
                masks[:, 0, 1:, 1:] = 4
                masks[:, 1, 0, 0] = 4
                return {"pred_logits": logits, "pred_boxes": torch.full((2, 2, 4), .5),
                        "pred_masks": masks, "presence_logit_dec": torch.full((2, 1), 5.0)}
        class Processor:
            def set_image_batch(self, frames):
                batch_sizes.append(len(frames))
                return {"backbone_out": {"visual": "immutable shared features"}}
        backend = OfficialSam3Backend.__new__(OfficialSam3Backend)
        backend.torch = torch
        backend.device = "cpu"
        backend.model = Model()
        backend.processor = Processor()
        backend.find_stage_type = lambda **kwargs: SimpleNamespace(**kwargs)
        source = {"source_id": "eyes", "prompt": "eye", "confidence": .5,
                  "mask_threshold": .75, "min_mask_area_px": 1, "max_regions": 8}
        with tempfile.TemporaryDirectory(dir="temp") as folder:
            directory = Path(folder)
            images = []
            for index, size in enumerate([(6, 6), (9, 12)]):
                path = directory / f"input-{index}.png"
                with Image.new("RGB", size) as image:
                    image.save(path)
                images.append({"image_id": str(index), "size": list(size), "path": str(path)})
            records = backend.segment(images, [source, {**source, "source_id": "none", "prompt": "nothing"}], directory)
            self.assertEqual([len(r["regions"]) for r in records], [2, 2, 0, 0])
            for record in records[:2]:
                for region in record["regions"]:
                    with Image.open(directory / region["mask_path"]) as mask:
                        self.assertEqual(list(mask.size), images[int(record["image_id"])]["size"])
                        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertEqual(batch_sizes, [2])
        self.assertEqual(prompts, ["eye", "nothing"])
        self.assertEqual(dummy_sizes, [2, 2])
        self.assertEqual(moves, ["cpu", "cpu"])

    def test_failed_batch_releases_features_before_offload_and_can_retry(self):
        backend = OfficialSam3Backend.__new__(OfficialSam3Backend)
        backend.torch = torch
        backend.device = "cpu"
        moves, features = [], []
        backend.model = SimpleNamespace(to=lambda device: moves.append(device))
        def encode(_):
            tensor = torch.ones(2)
            features.append(weakref.ref(tensor))
            return {"features": tensor}
        backend._encode_images = encode
        backend._predict_source = lambda *_: {"prediction": torch.ones(1)}
        def candidates(predictions, *_):
            yield np.ones((3, 4)), .9, []
        backend._candidates = candidates
        def offload():
            self.assertTrue(all(reference() is None for reference in features))
            backend.model.to("cpu")
        backend.offload = offload
        source = {"source_id": "eyes", "prompt": "eye", "confidence": .5,
                  "mask_threshold": .5, "min_mask_area_px": 1, "max_regions": 8}
        with tempfile.TemporaryDirectory(dir="temp") as folder:
            path = Path(folder) / "input.png"
            with Image.new("RGB", (4, 3)) as image:
                image.save(path)
            images = [{"image_id": "frame", "size": [4, 3], "path": str(path)}]
            with patch("Scripts.sam3_backend.select_masks", side_effect=RuntimeError("decode failed")):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    backend.segment(images, [source], folder)
            self.assertEqual(len(backend.segment(images, [source], folder)), 1)
        self.assertEqual(moves, ["cpu"] * 4)


if __name__ == "__main__":
    unittest.main()
