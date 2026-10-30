"""Isolated-process and candidate acceptance without SAM weights or CUDA."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from Scripts.sam3_adapter import SamWorker, SamWorkerError
from Scripts.sam_config import SamConfig, SamSourceConfig
from Scripts.sam_masks import select_masks


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.config = SamConfig("ok.pt", "cuda", False, (
            SamSourceConfig("eyes", "eyes", "eye"), SamSourceConfig("empty", "empty", "nothing")))
        self.images = []
        for index, size in enumerate([(8, 5), (11, 9)]):
            path = self.directory / f"{index}.png"
            with Image.new("RGB", size) as image:
                image.save(path)
            self.images.append({"image_id": str(index), "path": str(path), "size": list(size)})

    def worker(self, mode="ok", **kwargs):
        return SamWorker(sys.executable, replace(self.config, checkpoint=mode+".pt"),
                         worker_path=Path(__file__).with_name("fake_sam_worker.py"), **kwargs)

    def test_actual_process_batch_empty_branches_and_cleanup(self):
        with self.worker() as worker:
            process = worker.process
            records = worker.segment(self.images, self.directory)
            self.assertEqual([(r["image_id"], r["source_id"]) for r in records],
                             [("0", "eyes"), ("0", "empty"), ("1", "eyes"), ("1", "empty")])
            self.assertEqual(records[1]["regions"], [])
            self.assertEqual(records[2]["regions"][0]["image_size"], [11, 9])
            self.assertTrue(any("stderr" in line for line in worker.logs))
            worker.offload()
            worker.load()
            repeated = worker.segment(self.images, self.directory)
            self.assertEqual(records, repeated)
            self.assertIs(worker.process, process)
        self.assertIsNotNone(process.poll())

    def test_bad_replies_and_mask_validation(self):
        for mode in ["malformed", "mismatch", "version", "bad_size", "bad_prompt", "escape", "incomplete", "exit"]:
            with self.subTest(mode=mode), self.worker(mode) as worker:
                with self.assertRaises((SamWorkerError, ValueError)):
                    worker.segment(self.images, self.directory)

    def test_oom_is_actionable_and_not_empty(self):
        with self.worker("oom") as worker:
            with self.assertRaisesRegex(SamWorkerError, "reduce sam_batch_size"):
                worker.segment(self.images, self.directory)

    def test_timeout_and_cancel_terminate_process(self):
        with self.worker("timeout") as worker:
            process = worker.process
            worker.timeout = .2
            with self.assertRaisesRegex(SamWorkerError, "timed out"):
                worker.segment(self.images, self.directory)
            self.assertIsNotNone(process.poll())
        with self.worker() as worker:
            process = worker.process
            worker.cancelled = lambda: True
            with self.assertRaises(KeyboardInterrupt):
                worker.segment(self.images, self.directory)
            self.assertIsNotNone(process.poll())

    def test_real_worker_preflight_without_network(self):
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(SamWorkerError, "CUDA device unavailable"):
                SamWorker(sys.executable, self.config)
        elif sys.version_info < (3, 12):
            with self.assertRaisesRegex(SamWorkerError, "Python >=3.12"):
                SamWorker(sys.executable, self.config)
        else:
            version = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
            if version < (2, 7):
                with self.assertRaisesRegex(SamWorkerError, "torch >=2.7"):
                    SamWorker(sys.executable, self.config)


class MaskSelectionTests(unittest.TestCase):
    def test_selection_order_duplicates_thresholds_and_limit(self):
        one = np.zeros((5, 6)); one[4, 5] = .8
        two = np.zeros((5, 6)); two[0:2, 0:2] = .9
        three = np.zeros((5, 6)); three[3:5, 0:2] = .9
        candidates = [(one, .1, []), (np.zeros((5, 6)), .9, [0]*4),
                      (one, .9, [0]*4), (two, .7, [0]*4), (two, .8, [0]*4),
                      (three, .75, [0]*4)]
        source = {"confidence": .5, "mask_threshold": .5, "min_mask_area_px": 2, "max_regions": 1}
        selected, counts = select_masks(candidates, source, (6, 5))
        self.assertEqual(selected[0]["bbox"], [0, 0, 2, 2])
        self.assertEqual(selected[0]["segmentation_score"], .8)
        self.assertEqual(counts, {"total": 6, "confidence": 1, "empty": 1, "area": 1,
                                  "duplicate": 1, "limited": 1, "selected": 1})


if __name__ == "__main__":
    unittest.main()
