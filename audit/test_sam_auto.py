"""Automatic device/backend routing and offline loading contracts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from safetensors.torch import save_file
import torch

from Scripts.configs import ScriptOptions
from Scripts.sam_config import load_sam_config, sam_python_executable, validate_batch_options
from Scripts.sam_runtime import create_backend, resolve_backend, resolve_device
from Scripts.sam_transformers import TransformersSam3Backend, resolve_transformers_folder


class AutoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_auto_cpu_cuda_probe_and_explicit_device(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0), empty=Mock())
        self.assertEqual(resolve_device("auto", fake), "cpu")
        self.assertEqual(resolve_device("cpu", fake), "cpu")
        with self.assertRaisesRegex(RuntimeError, "Requested"):
            resolve_device("cuda:1", fake)
        fake.cuda.is_available = lambda: True
        fake.cuda.device_count = lambda: 2
        self.assertEqual(resolve_device("auto", fake), "cuda")
        self.assertEqual(resolve_device("cuda:1", fake), "cuda:1")
        fake.empty.side_effect = RuntimeError("driver failed")
        self.assertEqual(resolve_device("auto", fake), "cpu")
        with self.assertRaisesRegex(RuntimeError, "driver failed"):
            resolve_device("cuda", fake)

    def test_layout_detection_and_backend_override(self):
        path = self.root / "model.safetensors"
        save_file({"detr_decoder.test": torch.zeros(1)}, path)
        settings = {"checkpoint": str(path)}
        self.assertEqual(resolve_backend(settings), "transformers")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_backend({**settings, "backend": "native"})
        self.assertEqual(resolve_backend({"checkpoint": str(self.root/"sam3.pt")}), "native")
        self.assertEqual(resolve_backend({"model_id": "unsloth/sam3.1"}), "transformers")
        save_file({"backbone.visual.test": torch.zeros(1)}, path)
        self.assertEqual(resolve_backend(settings), "native")

    def test_current_interpreter_and_config_defaults(self):
        options = ScriptOptions("images")
        validate_batch_options(options)
        self.assertEqual(sam_python_executable(options), sys.executable)
        path = self.root / "config.json"
        path.write_text(json.dumps({"schema_version": 1, "model": {"model_id": "unsloth/sam3.1"},
            "sources": [{"source_id": "eyes", "name": "eyes", "prompt": "eye"}]}))
        config = load_sam_config(path)
        self.assertEqual((config.device, config.backend, config.checkpoint), ("auto", "auto", ""))
        for device in ("auto", "cpu", "cuda", "cuda:1"):
            data = json.loads(path.read_text()); data["model"]["device"] = device
            path.write_text(json.dumps(data))
            self.assertEqual(load_sam_config(path).device, device)

    def test_missing_metadata_requires_source_repo_without_network(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"unchanged weights")
        with patch("huggingface_hub.snapshot_download") as download:
            with self.assertRaisesRegex(ValueError, "model.model_id"):
                resolve_transformers_folder({"checkpoint": str(path), "allow_download": True})
            download.assert_not_called()

    def test_download_metadata_does_not_replace_existing_weights_or_config(self):
        folder, snapshot = self.root/"local", self.root/"snapshot"
        folder.mkdir(); snapshot.mkdir()
        (folder/"model.safetensors").write_bytes(b"my existing weights")
        (folder/"config.json").write_text('{"model_type":"sam3", "keep":true}')
        for name, data in [("config.json", '{"model_type":"sam3"}'),
                           ("processor_config.json", '{}'), ("tokenizer.json", '{}')]:
            (snapshot/name).write_text(data)
        with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
            resolved = resolve_transformers_folder({"checkpoint": str(folder/"model.safetensors"),
                "model_id": "unsloth/sam3.1", "allow_download": False})
            self.assertTrue(download.call_args.kwargs["local_files_only"])
            self.assertNotIn("model.safetensors", download.call_args.kwargs["allow_patterns"])
        self.assertEqual((folder/"model.safetensors").read_bytes(), b"my existing weights")
        self.assertTrue(json.loads((folder/"config.json").read_text())["keep"])
        with patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(resolve_transformers_folder({"checkpoint": str(resolved)}), resolved)
            download.assert_not_called()

    def test_auto_factory_uses_cpu_transformers_for_hf_weights(self):
        path = self.root/"model.safetensors"
        save_file({"detr_decoder.test": torch.zeros(1)}, path)
        with patch("Scripts.sam_runtime.resolve_device", return_value="cpu"), \
             patch("Scripts.sam_transformers.TransformersSam3Backend") as backend:
            create_backend({"checkpoint": str(path), "device": "auto"})
            self.assertEqual(backend.call_args.args[0]["device"], "cpu")

    def test_initial_repository_download_and_offline_failure(self):
        snapshot = self.root/"snapshot"
        snapshot.mkdir()
        for name, data in [("config.json", '{"model_type":"sam3"}'),
                           ("processor_config.json", '{}'), ("tokenizer.json", '{}')]:
            (snapshot/name).write_text(data)
        (snapshot/"model.safetensors").write_bytes(b"downloaded weights")
        target = self.root/"target"/"model.safetensors"
        with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
            folder = resolve_transformers_folder({"checkpoint": str(target), "model_id": "unsloth/sam3.1",
                "allow_download": True, "revision": "pinned-revision"})
            self.assertFalse(download.call_args.kwargs["local_files_only"])
            self.assertIn("model.safetensors", download.call_args.kwargs["allow_patterns"])
            self.assertEqual(download.call_args.kwargs["revision"], "pinned-revision")
            self.assertEqual((folder/"model.safetensors").read_bytes(), b"downloaded weights")
        with patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("not cached")) as download:
            with self.assertRaisesRegex(FileNotFoundError, "not cached"):
                resolve_transformers_folder({"model_id": "uncached/model", "allow_download": False})
            self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_hf_candidates_are_xyxy_and_use_presence_score(self):
        backend = TransformersSam3Backend.__new__(TransformersSam3Backend)
        backend.torch = torch
        predictions = SimpleNamespace(pred_logits=torch.tensor([[4., -4.]]),
            presence_logits=torch.tensor([[2.]]),
            pred_boxes=torch.tensor([[[.1, .2, .6, .8], [0, 0, 1, 1]]]),
            pred_masks=torch.tensor([[[[-4., 4.], [4., -4.]], [[0., 0.], [0., 0.]]]]))
        rows = list(backend._candidates(predictions, 0, (100, 40), .5))
        self.assertAlmostEqual(rows[0][1], float(torch.tensor(4.).sigmoid()*torch.tensor(2.).sigmoid()), places=6)
        torch.testing.assert_close(torch.tensor(rows[0][2]), torch.tensor([10., 8., 60., 32.]))
        self.assertEqual(rows[0][0].shape, (40, 100))


if __name__ == "__main__":
    unittest.main()
