"""Real installed ComfyUI schemas, Python node API and validator; no models/UI."""
import argparse
from contextlib import ExitStack
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from unittest.mock import patch


def complete(coroutine):
    # No loopback socket is needed for synchronous prompt validation.
    try:
        yielded = coroutine.send(None)
    except StopIteration as result:
        return result.value
    coroutine.close()
    raise AssertionError(f'Unexpected asynchronous operation: {yielded!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / 'temp/conditional-acceptance'
    work.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(work/'yolo')
    os.environ['MPLCONFIGDIR'] = str(work/'matplotlib')
    sys.path[:0] = [str(args.comfy_root), str(root)]
    sys.argv = ['redundancy-acceptance', '--cpu', '--base-directory', str(work)]
    import comfy.options
    comfy.options.enable_args_parsing()
    from audit.test_tag_postprocessing import evidence_fixture, pipeline, post, reports
    from audit.test_comfy_nodes import PACKAGE
    ui = importlib.import_module('wdv3_comfy_test.comfy_ui_nodes')
    operations = importlib.import_module('wdv3_comfy_test.comfy_postprocessing_nodes')
    from examples.build_comfy_workflows import build
    import execution
    import nodes
    nodes.NODE_CLASS_MAPPINGS.update(PACKAGE.NODE_CLASS_MAPPINGS)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Models and inference are forbidden during postprocessing')

    from audit.test_conditional_tags import fixture, policy
    raw = fixture()
    before = raw.to_json()
    params = dict(rules_json=json.dumps(policy()['rules']), scope='regions', sources='clothes')
    with ExitStack() as stack:
        for module, name in [('wdv3_comfy_test.Scripts.model', 'load_yolo_model'),
                             ('wdv3_comfy_test.Scripts.tagger_resources', 'load_tagger'),
                             ('wdv3_comfy_test.comfy_processing', 'segment_frames'),
                             ('wdv3_comfy_test.comfy_evidence', 'collect'),
                             ('wdv3_comfy_test.comfy_evidence', 'cached_segment')]:
            stack.enter_context(patch.object(importlib.import_module(module), name, forbidden))
        cls = operations.WDConditionalTagRules
        schema = cls.INPUT_TYPES()['required']
        assert schema['rules_json'][1]['default'] == '[]'
        assert schema['scope'][1]['default'] == 'regions'
        local = operations.WDFilterTagSets.execute(raw).result[0]
        assert cls.execute(local).result[0].data()['frames'] == local.data()['frames']
        cleaned = cls.execute(local, **params).result[0]
        result, tags, details = operations.WDMergeTagSets.execute(cleaned).result
        assert 'dress' not in cleaned.data()['frames'][0]['detectors']['clothes'][0]['general']
        assert 'dress' in result.data()['frames'][0]['merged_tags']['general']
        chain = cleaned.data()['postprocessing']
        setting = operations.WDPostprocessingConfig.execute(json.dumps(chain)).result[0]
        with patch.object(ui, 'preset_directory', return_value=work/'presets'):
            ui.WDSaveConfig.execute('conditional', True, postprocessing=setting)
            restored = ui.WDLoadConfig.execute('Saved: conditional').result[4]
        assert operations.WDFilterTagSets.execute(raw, postprocessing=restored).result[0] == cleaned
        redundancy = dict(explicit_enabled=True, relations_json='[{"specific":"dress","general":"skirt"}]', scope='all')
        conditional_first = operations.WDRemoveRedundantTags.execute(cls.execute(local, scope='all', rules_json=params['rules_json']).result[0], **redundancy).result[0]
        redundant_first = cls.execute(operations.WDRemoveRedundantTags.execute(local, **redundancy).result[0], scope='all', rules_json=params['rules_json']).result[0]
        assert 'skirt' in conditional_first.data()['frames'][0]['full_image']['general']
        assert 'skirt' not in redundant_first.data()['frames'][0]['full_image']['general']
        inspect_cls = PACKAGE.NODE_CLASS_MAPPINGS['WDV3Inspect']
        for value in (cleaned, conditional_first, redundant_first):
            inspected = inspect_cls.execute(tag_sets=value)
            paths = reports.export_report(inspected.result[0], work, save_txt=False)
            text = Path(paths['json']).read_text(encoding='utf-8')
            assert 'apron' in text and 'operation_digest' in text
            merged = operations.WDMergeTagSets.execute(value).result[0]
            final_paths = reports.export_report(inspect_cls.execute(results=merged).result[0], work)
        assert pipeline.merge_results(result, result).data()['frames'] == result.data()['frames']
        assert raw.to_json() == before and calls == []

        class EvidenceSource:
            INPUT_TYPES = classmethod(lambda cls: {'required': {}})
            RETURN_TYPES = ('WDV3_EVIDENCE',)
            FUNCTION = 'run'
            run = forbidden
        nodes.NODE_CLASS_MAPPINGS['RedundancyEvidenceSource'] = EvidenceSource
        prompt = {
            'source': {'class_type': 'RedundancyEvidenceSource', 'inputs': {}},
            'filter': {'class_type': 'WDV3FilterTagSets', 'inputs': {'evidence': ['source', 0]}},
            'op': {'class_type': 'WDV3ConditionalTagRules', 'inputs': {
                'tag_sets': ['filter', 0], **params}},
            'merge': {'class_type': 'WDV3MergeTagSets', 'inputs': {'tag_sets': ['op', 0]}},
            'inspect': {'class_type': 'WDV3Inspect', 'inputs': {'results': ['merge', 0]}},
        }
        assert complete(execution.validate_prompt('redundancy-test', prompt, None))[0] is True
        for invalid_rules in ('[', '[{"id":"bad","all_of":["apron"],"none_of":["apron"],"remove":["dress"]}]'):
            prompt['op']['inputs']['rules_json'] = invalid_rules
            previous = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                invalid = complete(execution.validate_prompt('conditional-invalid', prompt, None))
            finally:
                logging.disable(previous)
            assert invalid[0] is False
        assert calls == []

    # Folder node schema and real validator accept a shared multi-operation config.
    folder_cls = PACKAGE.NODE_CLASS_MAPPINGS['WDV3TagFolder']
    assert folder_cls.INPUT_TYPES()['optional']['postprocessing'][0] == 'WDV3_POSTPROCESSING'
    for kind in ('WDV3_FILES','WDV3_MODEL','WDV3_OPTIONS'):
        nodes.NODE_CLASS_MAPPINGS['FolderTest_'+kind] = type('FolderTest_'+kind, (), {
            'INPUT_TYPES': classmethod(lambda cls: {'required': {}}),
            'RETURN_TYPES': (kind,), 'FUNCTION':'run', 'run':forbidden})
    prompt = {name: {'class_type':'FolderTest_'+kind,'inputs':{}} for name,kind in
              [('files','WDV3_FILES'),('model','WDV3_MODEL'),('options','WDV3_OPTIONS')]}
    folder_inputs = {'files':['files',0], 'wd_model':['model',0], 'options':['options',0],
        'save_txt':False,'save_json':False,'output_directory':str(work),'overwrite':False,'sam_batch_size':1,
        'postprocessing':['config',0]}
    prompt['config']={'class_type':'WDV3PostprocessingConfig','inputs':{'postprocessing_json':json.dumps([
        policy(scope='all'), {'operation':'remove_redundant','textual_enabled':True}])}}
    prompt['folder']={'class_type':'WDV3TagFolder','inputs':folder_inputs}
    assert complete(execution.validate_prompt('folder-chain',prompt,None))[0]
    from PIL import Image
    cache_module=importlib.import_module('wdv3_comfy_test.comfy_automatic_cache')
    image_path=work/'folder-api.png'
    Image.new('RGB',(8,6)).save(image_path)
    with patch.object(cache_module,'cached_evidence',return_value=raw), patch.object(pipeline,'collect',forbidden):
        folder_result=folder_cls.execute(dict(root=work,paths=[image_path]),None,{},False,False,str(work),False,1,
            postprocessing=[policy(scope='all')]).result
    assert len(folder_result[0]) == 1
    assert 'conditional_remove' in folder_result[1][0]
    import unittest
    from audit.test_folder_postprocessing import FolderPostprocessingTests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FolderPostprocessingTests)
    assert unittest.TextTestRunner().run(suite).wasSuccessful()

    # Generate UI examples against real schemas without opening a browser.
    build(root)
    for path in (root/'examples').glob('comfyui_wdv3*.json'):
        graph = json.loads(path.read_text(encoding='utf-8'))
        entries = {entry['id']: entry for entry in graph['nodes']}
        assert len(entries) == len(graph['nodes']), path
        for link_id, source, output, target, index, kind in graph['links']:
            assert entries[source]['outputs'][output]['type'] == kind
            assert entries[target]['inputs'][index]['type'] in (kind, '*')
            assert entries[target]['inputs'][index]['link'] == link_id
    destination = root/'docs/verification/conditional-tags'
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(nodes=len(PACKAGE.NODE_CLASS_MAPPINGS), model_calls=len(calls),
        mixed_sources=['full', 'clothes:YOLO', 'face:SAM'], immutable_evidence=True,
        valid_prompt=True, invalid_rules_rejected=True, preset_roundtrip=True,
        global_caption_preserved=True, snapshot_and_order=True, folder_chain=True, conditional_preset_roundtrip=True,
        local_report=paths, merged_report=final_paths,
        python=sys.version, browser_tested=False)
    (destination/'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: real node schemas/API, presets, before/after Inspect, exports, prompt validation; zero model calls.')


if __name__ == '__main__':
    main()
