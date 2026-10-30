"""Pixel-level geometry acceptance independent of a segmentation model."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from Scripts.regions import canonical_image, mask_bbox, prepare_region_view, rectangle_geometry
from Scripts.sam_config import RegionViewConfig


class RegionTests(unittest.TestCase):
    def test_minimum_bbox_and_empty(self):
        mask = np.zeros((80, 100), dtype=bool)
        self.assertIsNone(mask_bbox(mask))
        mask[79, 99] = True
        self.assertEqual(mask_bbox(mask), [99, 79, 100, 80])

    def test_mask_holes_components_and_context(self):
        mask = np.zeros((9, 12), dtype=bool)
        mask[1:4, 1:4] = True
        mask[2, 2] = False
        mask[7, 10] = True
        before = mask.copy()
        pixels = np.arange(9*12*3, dtype=np.uint8).reshape(9, 12, 3)
        with Image.fromarray(pixels) as frame:
            geometry = rectangle_geometry("image", "eye", 0, frame.size, mask_bbox(mask))
            with self.subTest("no context"):
                crop, view = prepare_region_view(frame, geometry, RegionViewConfig("mask"), mask)
                with crop:
                    self.assertEqual(crop.getpixel((1, 1)), (255, 255, 255))
                    self.assertEqual(crop.getpixel((9, 6)), tuple(pixels[7, 10]))
            crop, context = prepare_region_view(frame, geometry, RegionViewConfig("mask", "px", 1), mask)
            with crop:
                self.assertEqual(crop.getpixel((2, 2)), tuple(pixels[2, 2]))
                self.assertEqual(crop.getpixel((6, 5)), (255, 255, 255))
            self.assertNotEqual(view["fingerprint"], context["fingerprint"])
            self.assertEqual(context["crop_bbox"], [0, 0, 12, 9])
        np.testing.assert_array_equal(mask, before)

    def test_fraction_clipping_and_bbox(self):
        with Image.new("RGB", (120, 60), (12, 34, 56)) as frame:
            geometry = rectangle_geometry("i", "s", 0, frame.size, [0, 10, 100, 50])
            crop, meta = prepare_region_view(frame, geometry, RegionViewConfig("bbox", "fraction", .1))
            with crop:
                self.assertEqual(crop.size, (110, 48))
                self.assertEqual(crop.getpixel((0, 0)), (12, 34, 56))
            self.assertEqual(meta["padding_px"], [10, 4])
            self.assertEqual(meta["crop_bbox"], [0, 6, 110, 54])
            self.assertEqual(meta["offset"], [0, -6])
            self.assertEqual(geometry["bbox"], [0, 10, 100, 50])
            with self.assertRaisesRegex(ValueError, "requires"):
                prepare_region_view(frame, geometry, RegionViewConfig("mask"))

    def test_exif_rotation_has_matching_pixels(self):
        with tempfile.TemporaryDirectory(dir="temp") as directory:
            path = Path(directory) / "rotated.png"
            with Image.new("RGB", (3, 2)) as image:
                image.putpixel((0, 0), (255, 0, 0))
                exif = Image.Exif()
                exif[274] = 6
                image.save(path, exif=exif)
            rgb, metadata = canonical_image(path)
            with rgb:
                self.assertEqual(rgb.size, (2, 3))
                self.assertEqual(rgb.getpixel((1, 0)), (255, 0, 0))
            np.testing.assert_array_equal(np.array(metadata["original_to_canonical"]) @ [0, 0, 1], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
