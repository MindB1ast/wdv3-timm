import importlib
import json
from pathlib import Path
import tempfile
import unittest

from audit.test_comfy_nodes import PACKAGE

save_result = importlib.import_module(f"{PACKAGE.__name__}.comfy_export").save_result

save_region_results = importlib.import_module(f"{PACKAGE.__name__}.comfy_export").save_region_results


def region_export_fixture():
    person = dict(region_id=0, region_uid='person-0', bbox=[0, 0, 8, 6],
                  general={'shirt': .8}, character={}, ratings={'general': .9},
                  raw_scores={'general': {'shirt': .8, 'rejected': .1}},
                  aggregate={'general': {'shirt': .8, 'blue_eyes': .95}, 'character': {}},
                  route_history=[{'tag': 'blue_eyes', 'reason': 'merge_to_parent'}])
    face = dict(region_id=0, region_uid='face-0', parent_region_uid='person-0', bbox=[1, 1, 4, 4],
                general={'blue_eyes': .95}, character={})
    return dict(image_id='image-a', full_image={'general': {'1girl': .95}, 'character': {}},
                merged_tags={'general': {'1girl': .95, 'blue_eyes': .95}, 'caption': '1girl, blue_eyes'},
                detectors={'person': [person], 'face': [face]}, decisions=[{'tag': 'blue_eyes'}])


class RegionExportTests(unittest.TestCase):
    def test_compact_identity_batch_and_optional_details(self):
        frame = region_export_fixture()
        encoded = [json.dumps(frame), json.dumps(dict(frame, image_id='image-b'))]
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            path, status = save_region_results(encoded, folder, 'compact', result_index=-1)
            self.assertEqual(status, 'written')
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual([row['image_id'] for row in payload['images']], ['image-a', 'image-b'])
            image = payload['images'][0]
            self.assertEqual(image['full']['general'], ['1girl'])
            parent, child = image['regions']
            self.assertEqual((parent['source'], parent['region_id']), ('person', 0))
            self.assertEqual((child['source'], child['region_id']), ('face', 0))
            self.assertEqual(child['parent_region_uid'], parent['region_uid'])
            self.assertEqual(parent['general'], ['shirt'])
            self.assertEqual(child['general'], ['blue eyes'])
            self.assertTrue({'raw_scores', 'aggregate', 'ratings', 'route_history'}.isdisjoint(parent))
            self.assertNotIn('decisions', image)
            path, _ = save_region_results(encoded, folder, 'details', result_index=1, include_full=False,
                include_scores=True, include_ratings=True, include_caption=True, include_aggregate=True, include_history=True)
            [image] = json.loads(Path(path).read_text(encoding='utf-8'))['images']
            self.assertEqual(image['image_id'], 'image-b')
            self.assertEqual(image['caption'], '1girl, blue eyes')
            self.assertNotIn('full', image)
            parent = image['regions'][0]
            self.assertEqual(parent['general'], {'shirt': .8})
            self.assertEqual(parent['ratings'], {'general': .9})
            self.assertEqual(parent['aggregate']['general']['blue eyes'], .95)
            self.assertEqual(parent['route_history'], frame['detectors']['person'][0]['route_history'])
            self.assertEqual(image['decisions'], frame['decisions'])

    def test_region_names_follow_replace_underscores(self):
        frame = region_export_fixture()
        frame['settings'] = {'options': {'replace_underscores': True}}
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            for scores in (False, True):
                path, _ = save_region_results([json.dumps(frame)], folder, 'spaces',
                    include_scores=scores, include_aggregate=True, write_mode='overwrite')
                person, face = json.loads(Path(path).read_bytes())['images'][0]['regions']
                self.assertEqual(face['general'], {'blue eyes': .95} if scores else ['blue eyes'])
                self.assertIn('blue eyes', person['aggregate']['general'])
            frame['settings']['options']['replace_underscores'] = False
            path, _ = save_region_results([json.dumps(frame)], folder, 'canonical')
            self.assertEqual(json.loads(Path(path).read_bytes())['images'][0]['regions'][1]['general'],
                             ['blue_eyes'])

    def test_depth_filter_preserves_cascade_links_and_aggregates(self):
        frame = region_export_fixture()
        face = frame['detectors']['face'][0]
        face['depth'] = 2
        frame['detectors']['eyes'] = [dict(face, depth=3, region_uid='eyes-0',
                                            parent_region_uid=face['region_uid'])]
        encoded = [json.dumps(frame)]
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            for mode, level, has_full, depths in (
                ('all', 0, True, [1, 2, 3]),
                ('exact', 0, True, []),
                ('exact', 2, False, [2]),
                ('up_to', 2, True, [1, 2]),
            ):
                with self.subTest(mode=mode, level=level):
                    path, _ = save_region_results(encoded, folder, 'levels', write_mode='overwrite',
                        depth_mode=mode, depth_level=level, include_caption=True, include_aggregate=True)
                    [image] = json.loads(Path(path).read_bytes())['images']
                    self.assertEqual('full' in image, has_full)
                    self.assertEqual([r['depth'] for r in image['regions']], depths)
                    self.assertEqual(image['caption'], '1girl, blue eyes')
                    if 1 in depths:
                        self.assertIn('blue eyes', image['regions'][0]['aggregate']['general'])
                    if mode == 'exact' and level == 2:
                        self.assertEqual(image['regions'][0]['parent_region_uid'], 'person-0')

    def test_export_full_only_and_select_categories(self):
        frame = region_export_fixture()
        frame['full_image']['character'] = {'hero': .9}
        frame['detectors']['person'][0]['character'] = {'hero': .8}
        frame['merged_tags']['character'] = {'hero': .9}
        frame['merged_tags']['caption'] += ', hero'
        encoded = [json.dumps(frame)]
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            path, _ = save_region_results(encoded, folder, 'full', include_regions=False,
                include_character=False, include_caption=True)
            [image] = json.loads(Path(path).read_bytes())['images']
            self.assertEqual(image['full'], {'depth': 0, 'general': ['1girl']})
            self.assertNotIn('regions', image)
            self.assertEqual(image['caption'], '1girl, blue eyes')
            path, _ = save_region_results(encoded, folder, 'regions', include_full=False,
                include_general=False, include_aggregate=True, include_caption=True)
            [image] = json.loads(Path(path).read_bytes())['images']
            self.assertNotIn('full', image)
            self.assertEqual(image['regions'][0]['character'], ['hero'])
            self.assertNotIn('general', image['regions'][0])
            self.assertNotIn('general', image['regions'][0]['aggregate'])
            self.assertEqual(image['caption'], 'hero')

    def test_existing_file_modes_and_invalid_batch(self):
        encoded = [json.dumps(region_export_fixture())]
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            path, _ = save_region_results(encoded, folder, 'result')
            original = Path(path).read_bytes()
            with self.assertRaises(FileExistsError):
                save_region_results(encoded, folder, 'result')
            _, status = save_region_results(encoded, folder, 'result', write_mode='skip', include_caption=True)
            self.assertEqual(status, 'skipped')
            self.assertEqual(Path(path).read_bytes(), original)
            save_region_results(encoded, folder, 'result', write_mode='overwrite', include_caption=True)
            self.assertIn('caption', json.loads(Path(path).read_bytes())['images'][0])
            with self.assertRaises(ValueError):
                save_region_results(encoded + ['invalid'], folder, 'invalid', result_index=-1)
            self.assertFalse((Path(folder) / 'invalid.regions.json').exists())


class SaveResultTests(unittest.TestCase):
    def test_txt_only_tags_without_json(self):
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            txt, js = save_result(None, folder, 'tags', tags=['first', 'blue eyes, dress'],
                                  result_index=1, save_json=False)
            self.assertEqual(Path(txt).read_text(), 'blue eyes, dress')
            self.assertEqual(js, '')
            self.assertFalse((Path(folder)/'tags.json').exists())
            txt, _ = save_result(['invalid ignored JSON'], folder, 'independent',
                                 tags=['text'], save_json=False)
            self.assertEqual(Path(txt).read_text(), 'text')

    def test_conditional_inputs(self):
        with tempfile.TemporaryDirectory(dir='temp') as folder:
            for kw in [dict(save_json=True, tags=['x']), dict(save_json=False),
                       dict(save_json=False, tags=['x'], result_index=1)]:
                with self.assertRaises(ValueError):
                    save_result(None, folder, 'missing', **kw)
            self.assertEqual(list(Path(folder).iterdir()), [])
            txt, js=save_result(['{"merged_tags":{}}'], folder, 'json', save_txt=False)
            self.assertEqual(txt, '')
            self.assertTrue(Path(js).exists())

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
