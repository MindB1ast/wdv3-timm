import importlib
import json
from pathlib import Path
import tempfile
import unittest

from audit.test_comfy_nodes import PACKAGE

save_result = importlib.import_module(f"{PACKAGE.__name__}.comfy_export").save_result


class SaveResultTests(unittest.TestCase):
    def test_saved_output_settings_without_tags(self):
        result = {'general': {'blue_eyes': .9}, 'character': {'hero_(series)': .8},
                  'caption': 'hero_(series), blue_eyes', 'taglist': 'hero (series), blue eyes'}
        detail = {'merged_tags': result, 'settings': {'options': {
            'tag_separator': ' | ', 'replace_underscores': False, 'escape_parentheses': True}}}
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            txt, _ = save_result([json.dumps(detail)], folder, 'formatted')
            self.assertEqual(Path(txt).read_text(encoding='utf-8'), r'hero_\(series\) | blue_eyes')
            txt, _ = save_result([json.dumps(detail)], folder, 'literal', tags=['literal text'])
            self.assertEqual(Path(txt).read_text(encoding='utf-8'), 'literal text')

    def test_selected_result_and_exact_tag_text(self):
        records = [json.dumps({"image_index": i, "merged_tags": {"taglist": f"тег {i}"},
                              "detectors": {"person": []}}) for i in range(2)]
        with tempfile.TemporaryDirectory(dir="temp") as folder:
            txt, detail = save_result(records, folder, "Результат", result_index=1, tags=["a", r"hero_\(series\)"])
            self.assertEqual(Path(txt).read_text(encoding="utf-8"), r"hero_\(series\)")
            self.assertEqual(json.loads(Path(detail).read_text(encoding="utf-8")), json.loads(records[1]))
            self.assertEqual(len(list(Path(folder).iterdir())), 2)

    def test_existing_file_preflight_no_partial_write_and_overwrite(self):
        record = ['{"merged_tags":{"taglist":"new"}}']
        with tempfile.TemporaryDirectory(dir="temp") as folder:
            detail = Path(folder) / "result.json"
            detail.write_text("old")
            with self.assertRaises(FileExistsError):
                save_result(record, folder, "result")
            self.assertFalse((Path(folder) / "result.txt").exists())
            self.assertEqual(detail.read_text(), "old")
            txt, disabled = save_result(record, folder, "result", save_json=False)
            self.assertEqual(disabled, "")
            self.assertEqual(Path(txt).read_text(), "new")
            save_result(record, folder, "result", overwrite=True)
            self.assertEqual(json.loads(detail.read_text())["merged_tags"]["taglist"], "new")

    def test_invalid_inputs_write_nothing(self):
        record = ['{"merged_tags":{"taglist":"tag"}}']
        with tempfile.TemporaryDirectory(dir="temp") as folder:
            for name in ("../escape", "x/y", "CON", "", "result.txt"):
                with self.assertRaises(ValueError):
                    save_result(record, folder, name)
            for kwargs in ({"result_index": 1}, {"result_index": -1}, {"tags": []}, {"save_txt": False, "save_json": False}):
                with self.assertRaises(ValueError):
                    save_result(record, folder, "result", **kwargs)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
