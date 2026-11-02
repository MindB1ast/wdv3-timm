"""Identifier compatibility does not depend on network access or model weights."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

from Scripts.tagger_catalog import WD_ALIASES, model_choices, model_path, resolve_model_id


class CatalogTests(unittest.TestCase):
    def test_leaderboard_order_and_unranked_publication_dates(self):
        root = Path(__file__).resolve().parents[1]
        with (root/'audit/fixtures/tagger-catalog/ranklist.csv').open(encoding='utf-8', newline='') as stream:
            ranked = sorted(csv.DictReader(stream), key=lambda row: (-float(row['macro_f1']), row['hf_repo']))
        choices = model_choices()
        self.assertEqual(choices[:21], [row['hf_repo'] for row in ranked])
        self.assertEqual(choices[0], 'animetimm/convnextv2_huge.dbv4-full')
        self.assertEqual(choices[2], 'animetimm/caformer_b36.dbv4-full')
        sources = {}
        for author in ('SmilingWolf','animetimm'):
            sources.update({row['id']: row.get('createdAt') or '' for row in
                            json.loads((root/f'audit/fixtures/tagger-catalog/{author}.json').read_text())})
        dates = [sources[resolve_model_id(value)] for value in choices[21:]]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_complete_source_union(self):
        root = Path(__file__).resolve().parents[1]
        fixtures = root / 'audit/fixtures/tagger-catalog'
        manifest = json.loads((root / 'Scripts/tagger_models.json').read_text(encoding='utf-8'))
        indexed = {entry['repo_id']: entry for entry in manifest}
        sources = set()
        for author in ('SmilingWolf', 'animetimm'):
            sources.update(entry['id'] for entry in json.loads(
                (fixtures / f'{author}.json').read_text(encoding='utf-8')))
        self.assertEqual(set(indexed), sources)
        with (fixtures / 'ranklist.csv').open(encoding='utf-8', newline='') as stream:
            ranked = {entry['hf_repo'] for entry in csv.DictReader(stream)}
        self.assertEqual(len(ranked), 21)
        self.assertTrue(ranked <= set(indexed))
        expected = {repo for repo in sources if repo.endswith('.dbv4-full') or repo.endswith('.dbv4-full.r224')}
        compatible = {entry['repo_id'] for entry in manifest if entry['status'] == 'metadata-compatible'}
        self.assertEqual(compatible, expected | set(WD_ALIASES.values()))
        with patch('socket.socket', side_effect=AssertionError('network')):
            self.assertEqual(set(model_choices()), expected | set(WD_ALIASES))
        for entry in manifest:
            self.assertTrue(entry['revision'] and entry['source'] and entry['checked_at'])
            if entry['status'] == 'excluded':
                self.assertTrue(entry['reason'])

    def test_aliases_and_direct_ids_offline(self):
        with patch("socket.socket", side_effect=AssertionError("network")):
            for alias, repo in WD_ALIASES.items():
                self.assertEqual(resolve_model_id(alias), repo)
                self.assertEqual(resolve_model_id(repo), repo)
            self.assertEqual(resolve_model_id("animetimm/caformer_b36.dbv4-full"),
                             "animetimm/caformer_b36.dbv4-full")
            for value in ("missing", "../model", "a/b/c", "a/../b", "https://hf.co/a/b"):
                with self.assertRaises(ValueError):
                    resolve_model_id(value)

    def test_namespace_and_legacy(self):
        with tempfile.TemporaryDirectory(dir="temp") as directory:
            root = Path(directory)
            self.assertNotEqual(model_path(root, "one/model"), model_path(root, "two/model"))
            old = root / "wd-vit-tagger-v3"
            old.mkdir()
            self.assertEqual(model_path(root, "vit"), old)
            self.assertNotEqual(model_path(root, "other/wd-vit-tagger-v3"), old)
