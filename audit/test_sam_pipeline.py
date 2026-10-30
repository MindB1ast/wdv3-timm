"""End-to-end CPU acceptance through a real subprocess and portable exports."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from Scripts.configs import DetectorConfig, ScriptOptions
from Scripts.main import _BatchResources, BatchTagging
from Scripts.region_export import load_region_record, saved_region_input
from Scripts.region_tagging import RegionTagger, retag_saved_region
from Scripts.sam3_adapter import SamWorker
from Scripts.sam_config import RegionViewConfig
from Scripts.wdv3_timm import LabelData


class TinyWD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1))
        self.batch_sizes = []
        self.devices = []

    def to(self, device):
        self.devices.append(str(device))
        return super().to(device)

    def forward(self, batch):
        self.batch_sizes.append(len(batch))
        # Different rows retain their own RGB evidence, with actual B>1.
        values = batch.mean(dim=(1, 2, 3))*8-4
        return torch.stack([values, -values], dim=1)


def transform(image):
    return torch.tensor(np.array(image.resize((4, 4))), dtype=torch.float32).permute(2, 0, 1)/255


class SamPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.images = self.root / "inputs"
        self.images.mkdir()
        for directory, color, size in [("a", "white", (12, 8)), ("b", "black", (7, 11))]:
            folder = self.images / directory
            folder.mkdir()
            with Image.new("RGB", size, color) as image:
                image.save(folder / "same.png")
            (folder / "same.txt").write_text("reference", encoding="utf-8")
        self.config = self.root / "sam.json"
        self.config.write_text(json.dumps({"schema_version": 1,
            "model": {"checkpoint": "ok.pt"}, "sources": [
                {"source_id": "eyes", "name": "sam_eyes", "prompt": "eye", "view_mode": "mask"},
                {"source_id": "none", "name": "empty", "prompt": "nothing",
                 "remove_tags_from_full": ["bright"]},
            ]}), encoding="utf-8")
        self.options = ScriptOptions(self.images, sam_config=self.config, sam_python=sys.executable,
            sam_batch_size=2, region_batch_size=2, regions_output_dir=self.root / "runs",
            detectors_config=self.root / "missing.json", recursive=True, save_txt=False, output_file=None)
        self.model = TinyWD()
        self.detectors = []
        self.yolo_models = {}
        self.labels = LabelData(names=["bright", "dark"], rating=[], general=[0, 1], character=[])

    def resources(self, *_):
        return _BatchResources(self.model, transform, self.labels, list(self.detectors), self.yolo_models)

    def worker(self, python, config, timeout, **kwargs):
        return SamWorker(python, config, timeout, worker_path=Path(__file__).with_name("fake_sam_worker.py"), **kwargs)

    def run_batch(self, **kwargs):
        with patch("Scripts.main._load_batch_resources", self.resources), patch("Scripts.sam_pipeline.SamWorker", self.worker):
            return BatchTagging(replace(self.options, **kwargs))

    def test_portable_export_masks_views_retag_and_legacy_txt(self):
        detailed, merged = self.run_batch()
        self.assertEqual(len(detailed), 2)
        self.assertNotEqual(detailed[0]["image_id"], detailed[1]["image_id"])
        self.assertIn("bright", merged[0]["merged_tags"]["general"])
        self.assertIn("dark", merged[1]["merged_tags"]["general"])
        export = Path(detailed[0]["export_root"])
        copied = self.root / "moved"
        shutil.copytree(export, copied)
        for detail in detailed:
            image_id = detail["image_id"]
            record = load_region_record(copied, image_id)
            region = record["detailed"]["detectors"]["sam_eyes"][0]
            mask_path = copied / region["geometry"]["mask_path"]
            original_mask = mask_path.read_bytes()
            crop, meta = saved_region_input(copied, image_id, "sam_eyes", 0, RegionViewConfig("mask"))
            with crop:
                self.assertEqual(meta["fingerprint"], region["view"]["fingerprint"])
            retagged = retag_saved_region(copied, image_id, "sam_eyes", 0,
                RegionViewConfig("bbox", "px", 2), RegionTagger(self.model, transform, self.labels, torch.device("cpu")))
            self.assertEqual(retagged["geometry"], region["geometry"])
            self.assertNotEqual(retagged["view"]["fingerprint"], region["view"]["fingerprint"])
            self.assertEqual(mask_path.read_bytes(), original_mask)
        for path in self.images.rglob("*.txt"):
            self.assertEqual(path.read_text(), "reference")
        self.assertEqual(json.loads((export/"manifest.json").read_text())["status"], "complete")
        self.assertFalse(list((export/".staging").iterdir()))

    def test_wd_batch_rows_and_last_partial(self):
        tagger = RegionTagger(self.model, transform, self.labels, torch.device("cpu"))
        from Scripts.regions import prepare_region_view, rectangle_geometry
        def inputs():
            for index, color in enumerate(["white", "black", "white"]):
                with Image.new("RGB", (4, 4), color) as frame:
                    geometry = rectangle_geometry("image", "source", index, frame.size, [0, 0, 4, 4])
                    crop, view = prepare_region_view(frame, geometry, RegionViewConfig())
                yield crop, geometry, view
        results = tagger.tag_regions(inputs(), DetectorConfig("source", "wd"), 2)
        self.assertEqual(self.model.batch_sizes, [2, 1])
        self.assertEqual([list(row["general"])[0] for row in results], ["bright", "dark", "bright"])
        self.assertEqual([row["region_id"] for row in results], [0, 1, 2])

    def test_sam_wd_phase_order_across_portions(self):
        events = []
        phase = ["idle"]
        actual_worker = self.worker
        actual_to, actual_forward = self.model.to, self.model.forward
        def move(device):
            events.append(("wd_to", str(device)))
            return actual_to(device)
        def forward(batch):
            self.assertEqual(phase[0], "wd")
            return actual_forward(batch)
        def worker(*args, **kwargs):
            client = actual_worker(*args, **kwargs)
            segment, offload = client.segment, client.offload
            def segment_checked(*args):
                self.assertEqual(events[-1], ("wd_to", "cpu"))
                phase[0] = "sam"
                events.append(("sam_segment", "start"))
                return segment(*args)
            def offload_checked():
                result = offload()
                phase[0] = "wd"
                events.append(("sam_offload", "done"))
                return result
            client.segment, client.offload = segment_checked, offload_checked
            return client
        with patch.object(self.model, "to", move), patch.object(self.model, "forward", forward), \
             patch("Scripts.main._load_batch_resources", self.resources), \
             patch("Scripts.sam_pipeline.SamWorker", worker):
            detailed, _ = BatchTagging(replace(self.options, sam_batch_size=1))
        self.assertEqual(len(detailed), 2)
        self.assertEqual(events.count(("sam_segment", "start")), 2)
        self.assertEqual(events[-1], ("wd_to", "cpu"))

    def test_inspector_and_overview_render_both_modes(self):
        from Scripts.visualization import inspect_saved_region, visualize_detection_results
        detailed, _ = self.run_batch()
        detail = detailed[0]
        overview = self.root / "overview.png"
        visualize_detection_results(detail["image_path"], detail, output_path=overview, show=False)
        self.assertTrue(overview.is_file())
        for mode in ["mask", "bbox"]:
            output = self.root / f"{mode}.png"
            figure = inspect_saved_region(detail["export_root"], detail["image_id"], "sam_eyes", 0,
                RegionViewConfig(mode, "px", 1), show=False, output_path=output)
            self.assertEqual(sum(trace.type == "image" for trace in figure.data), 2)
            self.assertFalse(figure.layout.images[0].visible)
            import base64
            from io import BytesIO
            from Scripts.region_export import saved_region_input
            images = [trace for trace in figure.data if trace.type == "image"]
            with Image.open(BytesIO(base64.b64decode(images[0].source.split(",", 1)[1]))) as embedded, \
                 Image.open(Path(detail["export_root"])/detail["canonical_path"]) as original:
                np.testing.assert_array_equal(np.asarray(embedded), np.asarray(original))
            expected, _ = saved_region_input(detail["export_root"], detail["image_id"], "sam_eyes", 0,
                                            RegionViewConfig(mode, "px", 1))
            with expected, Image.open(BytesIO(base64.b64decode(images[1].source.split(",", 1)[1]))) as embedded:
                np.testing.assert_array_equal(np.asarray(embedded), np.asarray(expected))
            with Image.open(output) as panel:
                self.assertGreater(panel.width, 12)

    def test_corrupt_file_and_commit_failure_preserve_previous(self):
        (self.images / "z.png").write_bytes(b"broken")
        detailed, _ = self.run_batch(sam_batch_size=1)
        manifest = json.loads((Path(detailed[0]["export_root"])/"manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete_with_errors")
        self.assertEqual(sum(r["status"] == "complete" for r in manifest["images"].values()), 2)
        self.assertEqual(sum(r["status"] == "error" for r in manifest["images"].values()), 1)
        from Scripts.region_export import RegionRunExport
        original = RegionRunExport.commit
        commits = []
        def faulty(run, image_id, *args):
            if commits:
                raise OSError("disk full")
            commits.append(image_id)
            return original(run, image_id, *args)
        with patch.object(RegionRunExport, "commit", faulty), self.assertRaisesRegex(OSError, "disk full"):
            self.run_batch(sam_batch_size=1)
        manifests = [json.loads(path.read_text()) for path in (self.root/"runs").glob("*/manifest.json")]
        failed = next(m for m in manifests if m["status"] == "error")
        self.assertEqual(sum(r["status"] == "complete" for r in failed["images"].values()), 1)

    def test_mixed_yolo_sources_share_canonical_frame_and_filters(self):
        detector = DetectorConfig("face", "fake.pt", add_tags_to_region={"manual_face": .8})
        seen = []
        class FakeYOLO:
            def to(self, _):
                return self
            def __call__(self, image, **kwargs):
                seen.append((image.mode, image.size, kwargs["device"]))
                return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[0., 0., 4., 4.]]))])]
        self.detectors = [detector]
        self.yolo_models = {"fake.pt": FakeYOLO()}
        detailed, merged = self.run_batch()
        self.assertEqual([item[:2] for item in seen], [("RGB", (12, 8)), ("RGB", (7, 11))])
        for detail, combined in zip(detailed, merged):
            self.assertEqual(set(detail["detectors"]), {"sam_eyes", "empty", "face"})
            face = detail["detectors"]["face"][0]
            self.assertNotIn("mask_path", face["geometry"])
            self.assertIn("manual_face", combined["merged_tags"]["general"])
            self.assertNotIn("eye", combined["merged_tags"]["general"])

    def test_mask_copy_failure_cannot_commit_dangling_links(self):
        from Scripts import region_export
        real_copy = region_export.shutil.copyfile
        count = [0]
        def fail_second_mask(source, target):
            if Path(source).name.startswith("mask-"):
                count[0] += 1
                if count[0] == 2:
                    raise OSError("mask write failed")
            return real_copy(source, target)
        with patch.object(region_export.shutil, "copyfile", fail_second_mask), self.assertRaisesRegex(OSError, "mask write"):
            self.run_batch(sam_batch_size=1)
        root = next((self.root/"runs").iterdir())
        manifest = json.loads((root/"manifest.json").read_text())
        complete = [row for row in manifest["images"].values() if row["status"] == "complete"]
        self.assertEqual(len(complete), 1)
        self.assertTrue((root/complete[0]["record"]).is_file())
        self.assertFalse(list((root/".staging").iterdir()))

    def test_cancel_keeps_committed_image(self):
        from Scripts.sam_pipeline import run_sam_batch
        from Scripts.region_export import RegionRunExport
        cancelled = [False]
        original = RegionRunExport.commit
        def cancel_after_commit(run, *args):
            result = original(run, *args)
            cancelled[0] = True
            return result
        with patch("Scripts.main._load_batch_resources", self.resources), patch("Scripts.sam_pipeline.SamWorker", self.worker), \
             patch.object(RegionRunExport, "commit", cancel_after_commit), self.assertRaises(KeyboardInterrupt):
            run_sam_batch(self.options, cancelled=lambda: cancelled[0])
        manifest_path = next((self.root/"runs").glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "cancelled")
        self.assertEqual(sum(r["status"] == "complete" for r in manifest["images"].values()), 1)

    def test_example_runs_without_vlm_or_comfyui(self):
        import runpy
        arguments = ["sam3_batch.py", str(self.images), "--sam-python", sys.executable,
                     "--sam-config", str(self.config), "--output", str(self.root/"runs"),
                     "--sam-batch", "2", "--region-batch", "2"]
        with patch("Scripts.main._load_batch_resources", self.resources), \
             patch("Scripts.sam_pipeline.SamWorker", self.worker), patch.object(sys, "argv", arguments):
            runpy.run_path("examples/sam3_batch.py", run_name="__main__")
        manifest = next((self.root/"runs").glob("*/manifest.json"))
        self.assertEqual(json.loads(manifest.read_text())["status"], "complete")

    def test_recursive_scan_does_not_reingest_prior_exports(self):
        output = self.images / "exports"
        first = self.run_batch(regions_output_dir=output, sam_batch_size=1)
        second = self.run_batch(regions_output_dir=output)
        self.assertEqual(len(first[0]), 2)
        self.assertEqual(len(second[0]), 2)
        self.assertEqual([r["image_id"] for r in first[0]], [r["image_id"] for r in second[0]])
        self.assertEqual([r["tagging_fingerprint"] for r in first[0]],
                         [r["tagging_fingerprint"] for r in second[0]])

    def test_manifest_retries_only_transient_windows_replace_failure(self):
        import os
        from Scripts import region_export
        original = region_export.atomic_write_text
        failed = [False]
        def once(path, text):
            if os.name == "nt" and not failed[0]:
                failed[0] = True
                error = PermissionError("temporary sharing violation")
                error.winerror = 32
                raise error
            return original(path, text)
        with patch.object(region_export, "atomic_write_text", once):
            detailed, _ = self.run_batch()
        self.assertEqual(len(detailed), 2)

    def test_output_parent_does_not_exclude_the_input_folder(self):
        detailed, _ = self.run_batch(regions_output_dir=self.root)
        self.assertEqual(len(detailed), 2)


if __name__ == "__main__":
    unittest.main()
