"""Selectable previews keep clean pixels and correctly scaled mask assets."""
from importlib import import_module
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from audit.test_comfy_processing import processing

preview = import_module("wdv3_comfy_test.comfy_preview")


class PreviewTests(unittest.TestCase):
    def test_clean_image_and_separate_masks_keep_original_coordinates(self):
        with tempfile.TemporaryDirectory(dir="temp") as directory, Image.new("RGB", (1600, 800), "red") as frame:
            first = np.zeros((800, 1600), dtype=bool)
            first[:, :800] = True
            second = ~first
            record = {"digest": processing.frame_digest(frame), "size": [1600, 800],
                      "regions": {"left": [{"mask": first}], "right": [{"mask": second}]}}
            assets = preview.preview_assets(frame, directory, record)
            self.assertEqual(assets["image_size"], [1600, 800])
            self.assertEqual(assets["preview_size"], [1200, 600])
            with Image.open(Path(directory) / assets["image"]["filename"]) as image:
                self.assertEqual(image.getpixel((10, 10)), (255, 0, 0))
            for key, values in [("left:0", (255, 0)), ("right:0", (0, 255))]:
                with Image.open(Path(directory) / assets["region_masks"][key]["filename"]) as mask:
                    self.assertEqual((mask.getpixel((0, 0)), mask.getpixel((1199, 0))), values)
            self.assertEqual(frame.size, (1600, 800))

    def test_mask_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="temp") as directory, Image.new("RGB", (4, 4)) as frame:
            with self.assertRaisesRegex(ValueError, "another image"):
                preview.preview_assets(frame, directory, {"digest": "wrong", "size": [4, 4]})


if __name__ == "__main__":
    unittest.main()
