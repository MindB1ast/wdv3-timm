"""Check production postprocessing with manually accepted tags and no images."""
import json
import unittest

from audit.test_comfy_nodes import PACKAGE


NODES = PACKAGE.NODE_CLASS_MAPPINGS


def execute(name, *args, **kwargs):
    return NODES['WDV3' + name].execute(*args, **kwargs).result[0]


class TextPostprocessingTests(unittest.TestCase):
    def text(self, value):
        return execute('TagSetsToText', value)

    def test_input_literals_canonicalization_and_empty(self):
        value = execute('TagSetsFromText', ' jacket , purple jacket, purple_jacket')
        frame, = value.data()['frames']
        self.assertTrue({'image_png', 'bbox', 'raw_scores'}.isdisjoint(frame))
        self.assertNotIn('raw_scores', frame['full_image'])
        self.assertEqual(self.text(value), 'jacket, purple jacket')
        for target, source in [('full', ''), ('region', 'clothes')]:
            empty = execute('TagSetsFromText', '', target=target, source_name=source)
            self.assertEqual(self.text(execute('RemoveRedundantTags', empty, textual_enabled=True)), '')
        with self.assertRaisesRegex(ValueError, 'source_name'):
            execute('TagSetsFromText', 'apron', target='region')

    def test_redundancy_scope_protection_and_disabled(self):
        value = execute('TagSetsFromText', 'jacket, purple jacket')
        for fields, expected in [
            ({}, 'jacket, purple jacket'),
            (dict(textual_enabled=True, scope='full'), 'purple jacket'),
            (dict(textual_enabled=True, scope='regions'), 'jacket, purple jacket'),
            (dict(textual_enabled=True, protected_tags='jacket'), 'jacket, purple jacket'),
        ]:
            with self.subTest(fields=fields):
                self.assertEqual(self.text(execute('RemoveRedundantTags', value, **fields)), expected)

    def test_conditional_region_and_unknown_source(self):
        value = execute('TagSetsFromText', 'apron, dress', target='region', source_name='clothes')
        rules = json.dumps([{'id': 'apron', 'all_of': ['apron'], 'remove': ['dress']}])
        for scope, expected in [('full', 'apron, dress'), ('regions', 'apron')]:
            self.assertEqual(self.text(execute('ConditionalTagRules', value, rules, scope, 'clothes')),
                             expected)
        self.assertEqual(self.text(value), 'apron, dress')
        with self.assertRaisesRegex(ValueError, 'Unknown postprocessing sources'):
            execute('ConditionalTagRules', value, rules, 'regions', 'face')

    def test_connected_operations_consume_the_previous_result(self):
        original = 'jacket, purple jacket, apron'
        value = execute('TagSetsFromText', original, target='region', source_name='clothes')
        scope = dict(scope='regions', sources='clothes')
        rules = json.dumps([{'id': 'apron', 'all_of': ['apron'], 'remove': ['purple jacket']}])

        redundant_first = execute('RemoveRedundantTags', value, textual_enabled=True, **scope)
        conditional_first = execute('ConditionalTagRules', value, rules, **scope)

        self.assertEqual(self.text(redundant_first), 'purple jacket, apron')
        self.assertEqual(self.text(conditional_first), 'jacket, apron')
        self.assertEqual(self.text(execute('ConditionalTagRules', redundant_first, rules, **scope)),
                         'apron')
        self.assertEqual(self.text(execute('RemoveRedundantTags', conditional_first,
                                           textual_enabled=True, **scope)), 'jacket, apron')
        self.assertEqual(self.text(value), original)

    def test_merge_rejects_text_sets_with_output_node_hint(self):
        value = execute('TagSetsFromText', 'jacket, purple jacket')
        cleaned = execute('RemoveRedundantTags', value, textual_enabled=True)
        self.assertEqual(self.text(cleaned), 'purple jacket')
        for tag_sets in (value, cleaned):
            with self.assertRaisesRegex(ValueError, 'Tag Sets to Text'):
                execute('MergeTagSets', tag_sets)
