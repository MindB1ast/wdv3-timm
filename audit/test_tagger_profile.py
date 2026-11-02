"""Input profile identity preserves shape, order, normalization and immutability."""
from dataclasses import replace
import json
import unittest

from Scripts.tagger_profile import InputProfile


class ProfileTests(unittest.TestCase):
    def profile(self, operations=None):
        return InputProfile("owner/model", (3, 384, 512), "RGB", "preprocess.json:test",
                            json.dumps(operations or [
                                {"type": "resize", "size": [384, 512]},
                                {"type": "normalize", "mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]}]))

    def test_round_trip_and_rectangular_shape(self):
        profile = self.profile()
        self.assertEqual(InputProfile.from_json(profile.to_json()), profile)
        self.assertEqual(profile.input_size, (3, 384, 512))
        self.assertEqual(profile.metadata()["input_size"], [3, 384, 512])

    def test_fingerprint_tracks_effective_contract(self):
        profile = self.profile()
        self.assertEqual(profile.fingerprint, self.profile().fingerprint)
        for changed in (replace(profile, input_size=(3, 512, 384)),
                        replace(profile, color_order="BGR"),
                        self.profile(list(reversed(profile.operations)))):
            self.assertNotEqual(changed.fingerprint, profile.fingerprint)
        operations = profile.operations
        operations[-1]["mean"][0] = 0.9
        self.assertNotEqual(self.profile(operations).fingerprint, profile.fingerprint)
        self.assertEqual(profile.operations[-1]["mean"][0], 0.1)

    def test_invalid_contracts(self):
        profile = self.profile()
        for kwargs in ({"input_size": (3, 0, 512)}, {"input_size": (3, 2.5, 512)},
                       {"input_size": (1, 384, 512)}, {"dtype": "uint8"},
                       {"operations_json": "[]"}, {"operations_json": '[{"size": 2}]'},
                       {"pixel_range": (1, 0)}, {"schema_version": 2}):
            with self.assertRaises(ValueError):
                replace(profile, **kwargs)
