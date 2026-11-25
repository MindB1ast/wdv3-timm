"""Settings use the same operation as explicit postprocessing nodes."""
import json
import unittest

from audit.test_comfy_nodes import PACKAGE, ROOT
from audit.test_conditional_tags import fixture


class RedundancySettingsTests(unittest.TestCase):
    def test_settings_chain_and_port_types(self):
        nodes = PACKAGE.NODE_CLASS_MAPPINGS
        settings = nodes['WDV3RedundancySettings']
        filter_node = nodes['WDV3FilterTagSets']
        remove = nodes['WDV3RemoveRedundantTags']
        self.assertEqual(settings.RETURN_TYPES[0], 'WDV3_POSTPROCESSING')
        for kind in ('WDV3FilterTagSets', 'WDV3TagFolder'):
            self.assertEqual(nodes[kind].INPUT_TYPES()['optional']['postprocessing'][0],
                             settings.RETURN_TYPES[0])
        raw = fixture()
        before = filter_node.execute(raw).result[0]
        disabled, = settings.execute().result
        self.assertEqual(disabled, [])
        self.assertEqual(filter_node.execute(raw, postprocessing=disabled).result[0], before)
        fields = dict(textual_enabled=True, scope='regions', sources='clothes', protected_tags='black skirt')
        chain, = settings.execute(**json.loads(json.dumps(fields))).result
        self.assertEqual(chain[0]['sources'], ['clothes'])
        self.assertEqual(chain[0]['protected_tags'], ['black_skirt'])
        self.assertEqual(filter_node.execute(raw, postprocessing=chain).result[0],
                         remove.execute(before, **fields).result[0])

    def test_saved_workflow_widget_values_and_connection(self):
        path = ROOT / 'examples/comfyui_wdv3_folder_redundancy.json'
        graph = json.loads(path.read_text(encoding='utf8'))
        config = next(n for n in graph['nodes'] if n['type'] == 'WDV3RedundancySettings')
        folder = next(n for n in graph['nodes'] if n['type'] == 'WDV3TagFolder')
        index = next(i for i, field in enumerate(folder['inputs']) if field['name'] == 'postprocessing')
        link_id = folder['inputs'][index]['link']
        self.assertIn([link_id, config['id'], 0, folder['id'], index, 'WDV3_POSTPROCESSING'], graph['links'])
        schema = PACKAGE.NODE_CLASS_MAPPINGS[config['type']].INPUT_TYPES()['required']
        saved = dict(zip(schema, config['widgets_values'], strict=True))
        self.assertEqual(saved, dict(textual_enabled=True, scope='all', sources='', protected_tags=''))
        chain, = PACKAGE.NODE_CLASS_MAPPINGS[config['type']].execute(**saved).result
        self.assertTrue(chain[0]['textual_enabled'])
