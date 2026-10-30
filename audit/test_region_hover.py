"""SAM mask overlays must not obscure source pixels before a hover."""
import base64
from io import BytesIO
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from Scripts.region_figure import HoverMaskFigure
from Scripts.visualization import visualize_detection_results


def hover_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "frame.png"
    pixels = np.zeros((300, 400, 3), dtype=np.uint8)
    pixels[:] = [210, 150, 95]
    pixels[:, 200:] = [100, 160, 220]
    with Image.fromarray(pixels) as frame:
        frame.save(path)
    def region(index, bbox, filename=None):
        row = {"region_id": index, "bbox": bbox, "caption": "test", "taglist": "test",
               "general": {"test": .9}, "character": {}, "ratings": {}}
        if filename:
            mask = np.zeros((300, 400), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
            mask[y1+15:y1+30, x1+15:x1+30] = 0
            with Image.fromarray(mask) as image:
                image.save(directory / filename)
            row["geometry"] = {"image_id": "fixture", "source_id": "sam", "region_id": index,
                "image_size": [400, 300], "bbox": bbox, "mask_path": filename,
                "prompt": "object", "segmentation_score": .9}
        return row
    detail = {"image_path": str(path), "full_image": {"general": {}, "character": {},
        "ratings": {}, "caption": "", "taglist": ""}, "export_root": str(directory),
        "detectors": {"sam": [region(0, [40, 40, 160, 160], "mask0.png"),
                               region(1, [230, 40, 350, 160], "mask1.png")],
                      "yolo": [region(0, [230, 210, 350, 280])]}}
    return path, detail, pixels


class HoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="temp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path, self.detail, self.pixels = hover_fixture(self.root)

    def test_source_is_only_image_trace_and_masks_start_hidden_with_png_alpha(self):
        figure = visualize_detection_results(self.path, self.detail, show=False)
        self.assertIsInstance(figure, HoverMaskFigure)
        self.assertEqual(sum(trace.type == "image" for trace in figure.data), 1)
        self.assertEqual(len(figure.layout.images), 2)
        for overlay in figure.layout.images:
            self.assertIs(overlay.visible, False)
            self.assertAlmostEqual(overlay.opacity, .3)
            with Image.open(BytesIO(base64.b64decode(overlay.source.split(",", 1)[1]))) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0))[3], 0)
                self.assertEqual(set(np.unique(np.asarray(image)[:, :, 3])), {0, 255})
        self.assertEqual([trace.meta["sam_mask_index"] for trace in figure.data if trace.type == "scatter"], [0, 1])
        self.assertEqual([trace.meta["sam_bbox"] for trace in figure.data if trace.type == "scatter"],
                         [[40, 40, 160, 160], [230, 40, 350, 160]])

    def test_static_overview_leaves_mask_interiors_untinted(self):
        output = self.root / "overview.png"
        visualize_detection_results(self.path, self.detail, output_path=output, show=False)
        with Image.open(output) as image:
            for point in [(100, 120), (300, 120), (300, 250), (190, 190)]:
                self.assertEqual(image.getpixel(point), tuple(self.pixels[point[1], point[0]]))

    def test_yolo_only_does_not_have_mask_layers_or_custom_renderer(self):
        self.detail["detectors"].pop("sam")
        figure = visualize_detection_results(self.path, self.detail, show=False)
        self.assertNotIsInstance(figure, HoverMaskFigure)
        self.assertEqual(len(figure.layout.images), 0)
        self.assertEqual(len(figure.data), 1)

    def test_notebook_and_html_keep_hover_and_unhover_callbacks(self):
        figure = visualize_detection_results(self.path, self.detail, show=False)
        html = self.root / "hover.html"
        figure.write_html(html)
        self.assertIn("graph.addEventListener('mousemove'", html.read_text(encoding="utf-8"))
        with patch("IPython.core.getipython.get_ipython", return_value=SimpleNamespace(kernel=True)), \
             patch("IPython.display.display") as display, patch("webbrowser.open_new_tab") as browser:
            figure.show()
            rendered = display.call_args.args[0].data
            self.assertIn("graph.addEventListener('mouseleave'", unescape(rendered))
            self.assertIn("sam_mask_index", rendered)
            browser.assert_not_called()
            display.reset_mock()
            figure._ipython_display_()
            self.assertIn("graph.addEventListener('mousemove'", unescape(display.call_args.args[0].data))
        self.assertIn("sam_mask_index", figure._repr_mimebundle_()["text/html"])

    def test_notebook_has_visible_original_png_without_executing_parent_scripts(self):
        class NotebookMarkup(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags = []

            def handle_starttag(self, tag, attrs):
                self.tags.append((tag, dict(attrs)))

        figure = visualize_detection_results(self.path, self.detail, show=False)
        markup = NotebookMarkup()
        markup.feed(figure._repr_mimebundle_()["text/html"])
        self.assertNotIn("script", [tag for tag, _ in markup.tags])
        fallback = next(attrs for tag, attrs in markup.tags if tag == "img")
        self.assertEqual(fallback["src"], figure.data[0].source)
        with Image.open(BytesIO(base64.b64decode(fallback["src"].split(",", 1)[1]))) as source:
            np.testing.assert_array_equal(np.asarray(source), self.pixels)
        frame = next(attrs for tag, attrs in markup.tags if tag == "iframe")
        self.assertIn("allow-scripts", frame["sandbox"])
        self.assertNotIn("allow-same-origin", frame["sandbox"])
        self.assertIn("plotly.js", frame["srcdoc"])
        self.assertIn("Plotly.newPlot", frame["srcdoc"])
        self.assertIn("graph.addEventListener('mousemove'", frame["srcdoc"])
        self.assertIn("graph.addEventListener('mouseleave'", frame["srcdoc"])
        self.assertNotIn("http", frame.get("src", ""))


if __name__ == "__main__":
    unittest.main()
