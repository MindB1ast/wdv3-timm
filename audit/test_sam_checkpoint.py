"""Checkpoint-format and wrong-config regressions without full SAM weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from safetensors.torch import save_file
import torch

from Scripts.configs import ScriptOptions
from Scripts.detection import load_detectors_config
from Scripts.main import BatchTagging
from Scripts.sam_checkpoint import build_sam_from_checkpoint, inspect_sam_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "model.safetensors"

    def test_sam_config_in_yolo_field_fails_before_loading_weights(self):
        config = self.directory / "sam.json"
        config.write_text(json.dumps({"schema_version": 1, "model": {}, "sources": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sam_config"):
            load_detectors_config(config)
        with patch("Scripts.main.download_model_files") as download, self.assertRaisesRegex(ValueError, "sam_config"):
            BatchTagging(ScriptOptions(self.directory, detectors_config=config, save_txt=False))
        download.assert_not_called()

    def test_native_safetensors_loads_exact_values_without_downloading(self):
        expected = {"weight": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.tensor([5., 6.])}
        save_file(expected, self.path)
        builder = Mock(side_effect=lambda **_: torch.nn.Linear(2, 2))
        model = build_sam_from_checkpoint(builder, self.path)
        for name, tensor in expected.items():
            torch.testing.assert_close(model.state_dict()[name], tensor)
        self.assertEqual(builder.call_args.kwargs["load_from_HF"], False)
        self.assertIsNone(builder.call_args.kwargs["checkpoint_path"])

    def test_native_detector_prefix_and_tracker_are_handled(self):
        save_file({"detector.weight": torch.ones((2, 2)), "detector.bias": torch.zeros(2),
                   "tracker.unused": torch.zeros(1)}, self.path)
        model = build_sam_from_checkpoint(lambda **_: torch.nn.Linear(2, 2), self.path)
        torch.testing.assert_close(model.weight, torch.ones((2, 2)))

    def test_transformers_file_is_rejected_before_build_or_tensor_loading(self):
        save_file({"detr_decoder.box_head.layer1.weight": torch.zeros((2, 2))}, self.path)
        builder = Mock()
        with patch("safetensors.torch.load_file") as load, self.assertRaisesRegex(ValueError, "Transformers"):
            build_sam_from_checkpoint(builder, self.path)
        builder.assert_not_called()
        load.assert_not_called()

    def test_incomplete_and_wrong_shape_weights_are_not_silently_accepted(self):
        for values in [{"weight": torch.ones((2, 2))},
                       {"weight": torch.ones((3, 2)), "bias": torch.zeros(2)}]:
            save_file(values, self.path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                build_sam_from_checkpoint(lambda **_: torch.nn.Linear(2, 2), self.path)

    def test_pt_keeps_official_loading_path_and_invalid_header_fails(self):
        builder = Mock()
        build_sam_from_checkpoint(builder, self.directory / "sam3.pt")
        self.assertTrue(builder.call_args.kwargs["checkpoint_path"].endswith("sam3.pt"))
        self.assertFalse(builder.call_args.kwargs["load_from_HF"])
        self.path.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "header"):
            inspect_sam_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
