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
    work = root / 'temp/tag-redundancy-acceptance'
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

    raw = evidence_fixture()
    before = raw.to_json()
    params = dict(explicit_enabled=True, relations_json=json.dumps([
        {'specific': 'black_skirt', 'general': 'skirt'}, {'specific': 'miniskirt', 'general': 'skirt'}]),
        scope='regions')
    with ExitStack() as stack:
        for name in ('collect', 'cached_segment'):
            stack.enter_context(patch.object(pipeline, name, forbidden))
        for module, name in [('wdv3_comfy_test.Scripts.model', 'load_yolo_model'),
                             ('wdv3_comfy_test.Scripts.tagger_resources', 'load_tagger'),
                             ('wdv3_comfy_test.comfy_processing', 'segment_frames')]:
            stack.enter_context(patch.object(importlib.import_module(module), name, forbidden))
        local = operations.WDFilterTagSets.execute(raw).result[0]
        cleaned = operations.WDRemoveRedundantTags.execute(local, **params).result[0]
        output = operations.WDMergeTagSets.execute(cleaned).result
        result, tags, details = output
        region = result.data()['frames'][0]['detectors']['clothes'][0]
        assert 'skirt' not in region['general'] and 'black_skirt' in region['general']
        assert 'skirt' in result.data()['frames'][0]['full_image']['general']
        protected = operations.WDRemoveRedundantTags.execute(local, protected_tags='skirt', **params).result[0]
        assert 'skirt' in protected.data()['frames'][0]['detectors']['clothes'][0]['general']
        # Preset settings are independent of evidence and models.
        chain = cleaned.data()['postprocessing']
        setting = operations.WDPostprocessingConfig.execute(json.dumps(chain)).result[0]
        with patch.object(ui, 'preset_directory', return_value=work/'presets'):
            ui.WDSaveConfig.execute('redundancy', True, postprocessing=setting)
            restored = ui.WDLoadConfig.execute('Saved: redundancy').result
        assert len(restored) == 10 and restored[4] == chain
        replay = operations.WDFilterTagSets.execute(raw, postprocessing=restored[4]).result[0]
        assert replay == cleaned
        inspect_cls = PACKAGE.NODE_CLASS_MAPPINGS['WDV3Inspect']
        inspected = inspect_cls.execute(tag_sets=cleaned)
        assert inspected.result[0].data()['stage'] == 'tag_sets'
        assert inspected.ui['wdv3_results'][0]['detail']['detectors']['clothes'][0]['postprocessing_history']
        paths = reports.export_report(inspected.result[0], work, save_txt=False)
        final_report = inspect_cls.execute(results=result).result[0]
        final_paths = reports.export_report(final_report, work)
        assert Path(final_paths['txt']).read_text(encoding='utf-8').splitlines() == tags
        saved = ui.WDSaveResult.execute(details, [str(work)], ['merged'], [0], [True], [True], [True])
        assert Path(saved.result[0]).read_text(encoding='utf-8') == tags[0]
        assert pipeline.merge_results(result, result).data()['frames'] == result.data()['frames']
        assert raw.to_json() == before and calls == []
        regex_rules = [{'id': 'colors', 'specific': r'(?:black|white)_(?P<base>.+)', 'general': r'\g<base>'},
                       {'id': 'lengths', 'specific': r'(?:mini|micro)(?P<base>skirt)', 'general': r'\g<base>'}]
        regex_local = operations.WDRemoveRedundantTags.execute(local, regex_enabled=True,
            regex_relations_json=json.dumps(regex_rules), scope='regions').result[0]
        regex_result = operations.WDMergeTagSets.execute(regex_local).result[0]
        assert 'skirt' not in regex_local.data()['frames'][0]['detectors']['face'][0]['general']
        assert pipeline.merge_results(regex_result, regex_result).data()['frames'] == regex_result.data()['frames']
        regex_chain = operations.WDPostprocessingConfig.execute(json.dumps(regex_local.data()['postprocessing'])).result[0]
        with patch.object(ui, 'preset_directory', return_value=work/'presets'):
            ui.WDSaveConfig.execute('regex_redundancy', True, postprocessing=regex_chain)
            loaded_regex = ui.WDLoadConfig.execute('Saved: regex_redundancy').result[4]
        assert operations.WDFilterTagSets.execute(raw, postprocessing=loaded_regex).result[0] == regex_local

        class EvidenceSource:
            INPUT_TYPES = classmethod(lambda cls: {'required': {}})
            RETURN_TYPES = ('WDV3_EVIDENCE',)
            FUNCTION = 'run'
            run = forbidden
        nodes.NODE_CLASS_MAPPINGS['RedundancyEvidenceSource'] = EvidenceSource
        prompt = {
            'source': {'class_type': 'RedundancyEvidenceSource', 'inputs': {}},
            'filter': {'class_type': 'WDV3FilterTagSets', 'inputs': {'evidence': ['source', 0]}},
            'op': {'class_type': 'WDV3RemoveRedundantTags', 'inputs': {
                'tag_sets': ['filter', 0], 'explicit_enabled': True, 'textual_enabled': False,
                'relations_json': params['relations_json'], 'protected_tags': '', 'scope': 'regions', 'sources': ''}},
            'merge': {'class_type': 'WDV3MergeTagSets', 'inputs': {'tag_sets': ['op', 0]}},
            'inspect': {'class_type': 'WDV3Inspect', 'inputs': {'results': ['merge', 0]}},
        }
        assert complete(execution.validate_prompt('redundancy-test', prompt, None))[0] is True
        prompt['op']['inputs']['relations_json'] = '[{"specific":"a","general":"b"},{"specific":"b","general":"a"}]'
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            invalid = complete(execution.validate_prompt('redundancy-invalid', prompt, None))
        finally:
            logging.disable(previous)
        assert invalid[0] is False and 'cycle' in str(invalid)
        assert calls == []

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
    destination = root/'docs/verification/tag-redundancy'
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(nodes=len(PACKAGE.NODE_CLASS_MAPPINGS), model_calls=len(calls),
        mixed_sources=['full', 'clothes:YOLO', 'face:SAM'], immutable_evidence=True,
        valid_prompt=True, invalid_cycle_rejected=True, preset_roundtrip=True,
        global_caption_preserved=True, regex_families=True, regex_preset_roundtrip=True,
        local_report=paths, merged_report=final_paths,
        python=sys.version, browser_tested=False)
    (destination/'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: real node schemas/API, presets, before/after Inspect, exports, prompt validation; zero model calls.')


if __name__ == '__main__':
    main()
