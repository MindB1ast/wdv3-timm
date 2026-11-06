"""Exercise preflight through the installed ComfyUI validator, without inference."""
import argparse
import copy
import json
import importlib.util
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / 'temp/preflight-acceptance'
    work.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(work / 'yolo')
    os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
    sys.path.insert(0, str(args.comfy_root))
    sys.argv = ['preflight-acceptance', '--cpu', '--base-directory', str(work)]
    import comfy.options
    comfy.options.enable_args_parsing()
    spec = importlib.util.spec_from_file_location('preflight_acceptance', root / '__init__.py',
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    import nodes
    import execution
    nodes.NODE_CLASS_MAPPINGS.update(package.NODE_CLASS_MAPPINGS)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Validation must never execute a node')

    types = {'IMAGE', 'STRING'}
    for cls in package.NODE_CLASS_MAPPINGS.values():
        types.update(t for t in cls.RETURN_TYPES if isinstance(t, str))
        setattr(cls, cls.FUNCTION, classmethod(fail))
    for kind in types:
        nodes.NODE_CLASS_MAPPINGS['Test_' + kind] = type('Test_' + kind, (), {
            'INPUT_TYPES': classmethod(lambda cls: {'required': {}}),
            'RETURN_TYPES': (kind,), 'FUNCTION': 'run', 'run': fail})

    def graph(node_id, overrides=None):
        cls = package.NODE_CLASS_MAPPINGS[node_id]
        schema = cls.INPUT_TYPES()
        inputs, result = {}, {}

        def link(name, kind):
            key = 'up_' + name
            result[key] = {'class_type': 'Test_' + kind, 'inputs': {}}
            inputs[name] = [key, 0]

        for group in ('required', 'optional'):
            for name, entry in schema.get(group, {}).items():
                kind, opts = entry[0], entry[1] if len(entry) > 1 else {}
                if kind == 'COMFY_AUTOGROW_V3':
                    template = opts['template']
                    nested = next(iter(template['input']['required'].values()))
                    link(name + '.' + template['prefix'] + '0', nested[0])
                elif 'default' in opts:
                    inputs[name] = opts['default']
                elif isinstance(kind, list):
                    inputs[name] = kind[0]
                elif kind == 'COMBO':
                    inputs[name] = opts['options'][0]
                elif group == 'required':
                    link(name, kind)
        if node_id == 'WDV3Inspect':
            link('results', 'WDV3_RESULTS')
        if node_id == 'WDV3SaveConfig':
            link('options', 'WDV3_OPTIONS')
        if node_id == 'WDV3PlanCalculations':
            link('images', 'IMAGE')
        if node_id == 'WDV3LoadFolder':
            inputs['directory'] = str(work)
        if 'dataset_root' in inputs:
            inputs['dataset_root'] = str(work)
        inputs.update(overrides or {})
        result['target'] = {'class_type': node_id, 'inputs': inputs}
        return result

    async def check(node_id, overrides=None, expected=True, message=None):
        prompt = graph(node_id, overrides)
        outcome = await execution.validate_inputs('preflight-test', prompt, 'target', {})
        assert outcome[0] is expected, (node_id, overrides, outcome)
        if message:
            assert message in str(outcome), outcome
        return prompt

    async def tests():
        for node_id, cls in package.NODE_CLASS_MAPPINGS.items():
            assert cls.__dict__.get('_wdv3_preflight') == node_id
            await check(node_id)
        cases = [
            ('WDV3RemoveRedundantTags', {'regex_relations_json': '[{"id":"broken","specific":"(","general":"skirt"}]'}),
            ('WDV3RemoveRedundantTags', {'regex_relations_json': json.dumps([{'id': 'broken', 'specific': '.*', 'general': r'\g<missing>'}])}),
            ('WDV3RemoveRedundantTags', {'relations_json': '['}),
            ('WDV3RemoveRedundantTags', {'relations_json': '[{"specific":"a","general":"b"},{"specific":"b","general":"a"}]'}),
            ('WDV3PostprocessingConfig', {'postprocessing_json': '[{"unexpected":true}]'}),
            ('WDV3Options', {'sam_batch_size': 0}),
            ('WDV3Options', {'general_threshold': 1.5}),
            ('WDV3Options', {'remove_tags_regex': '['}),
            ('WDV3RegionRules', {'allowed_tags_regex': '['}),
            ('WDV3RegionRules', {'add_to_regions': 'tag:bad'}),
            ('WDV3YOLODetector', {'classes': '-1'}),
            ('WDV3YOLODetector', {'source_name': 'full'}),
            ('WDV3YOLODetector', {'device': 'invalid'}),
            ('WDV3SAMSource', {'prompt': ' '}),
            ('WDV3SAMSource', {'max_regions': 0}),
            ('WDV3LoadFolder', {'directory': str(work / 'absent')}),
            ('WDV3DetectorLoader', {'detectors_json': '{}'}),
            ('WDV3DetectorLoader', {'detectors_json': '[{"name":"x"}]'}),
            ('WDV3ModelLoader', {'download_missing': False, 'local_model_path': str(work / 'absent')}),
            ('WDV3SAMLoader', {'download_missing': False, 'local_model_path': str(work / 'absent')}),
            ('WDV3YOLODetector', {'download_missing': False, 'custom_model_path': str(work / 'absent.pt')}),
            ('WDV3SaveResult', {'save_txt': False, 'save_json': False}),
            ('WDV3SaveResult', {'filename': '../bad'}),
            ('WDV3SaveResult', {'filename': 'CON'}),
            ('WDV3SaveResult', {'output_directory': str(Path(__file__))}),
            ('WDV3SaveConfig', {'preset_name': '../bad'}),
            ('WDV3ExportReport', {'prefix': '../bad'}),
            ('WDV3ExportReport', {'save_txt': False, 'save_html': False, 'save_json': False}),
        ]
        for node_id, overrides in cases:
            await check(node_id, overrides, False)
        for restored in ('', '  ', None):
            await check('WDV3RegionRules', {'child_tag_route': restored, 'tag_enabled': restored})
        await check('WDV3RegionRules', {'child_tag_route': 'merge_to_parent', 'tag_enabled': False})
        await check('WDV3RegionRules', {'child_tag_route': 'invalid'}, False)
        # ComfyUI coerces BOOLEAN before calling custom validators. Exercise
        # rejection of a raw malformed boolean at our own validation boundary.
        preflight = sys.modules[package.__name__ + '.comfy_preflight']
        assert preflight.validate_preflight('WDV3RegionRules',
            package.NODE_CLASS_MAPPINGS['WDV3RegionRules'].INPUT_TYPES(),
            values={'tag_enabled': 'invalid'}) is not True
        prompt = graph('WDV3RegionRules')
        prompt['child'] = {'class_type': 'Test_WDV3_DETECTORS', 'inputs': {}}
        prompt['target']['inputs']['child_detectors'] = ['child', 0]
        assert (await execution.validate_inputs('nested-yolo', prompt, 'target', {}))[0] is True
        prompt['sam_queries'] = {'class_type': 'Test_WDV3_SAM_SOURCES', 'inputs': {}}
        prompt['target']['inputs']['child_sources'] = ['sam_queries', 0]
        assert (await execution.validate_inputs('nested-missing-model', prompt, 'target', {}))[0] is False
        prompt['sam'] = {'class_type': 'Test_WDV3_SAM_MODEL', 'inputs': {}}
        prompt['target']['inputs']['child_sam_model'] = ['sam', 0]
        assert (await execution.validate_inputs('nested-mixed', prompt, 'target', {}))[0] is True
        prompt['child']['class_type'] = 'Test_IMAGE'
        assert (await execution.validate_inputs('nested-wrong-type', prompt, 'target', {}))[0] is False
        # Prompt validation, the actual server pre-queue entry point, must reject
        # inconsistent optional links and wrong types before invoking any loader.
        for node_id in ('WDV3Tagger', 'WDV3TagFolder', 'WDV3CollectEvidence'):
            prompt = graph(node_id)
            prompt['source'] = {'class_type': 'Test_WDV3_SAM_SOURCES', 'inputs': {}}
            prompt['target']['inputs']['sources'] = ['source', 0]
            outcome = await execution.validate_inputs('test', prompt, 'target', {})
            assert outcome[0] is False and 'sam_model' in str(outcome), outcome
            if node_id == 'WDV3Tagger':
                queued = await execution.validate_prompt('test', copy.deepcopy(prompt), None)
                assert queued[0] is False and 'sam_model' in str(queued), queued
            prompt['source']['class_type'] = 'Test_IMAGE'
            outcome = await execution.validate_inputs('test', prompt, 'target', {})
            assert outcome[0] is False and 'несовместимый' in str(outcome), outcome
            prompt['source']['class_type'] = 'Test_WDV3_SAM_SOURCES'
            prompt['sam'] = {'class_type': 'Test_WDV3_SAM_MODEL', 'inputs': {}}
            prompt['target']['inputs']['sam_model'] = ['sam', 0]
            outcome = await execution.validate_inputs('test', prompt, 'target', {})
            assert outcome[0] is True, outcome
            del prompt['target']['inputs']['sources']
            outcome = await execution.validate_inputs('test', prompt, 'target', {})
            assert outcome[0] is False and 'sources' in str(outcome), outcome
            if node_id == 'WDV3Tagger':
                prompt['regions'] = {'class_type': 'Test_WDV3_SAM_REGIONS', 'inputs': {}}
                prompt['target']['inputs']['sam_regions'] = ['regions', 0]
                outcome = await execution.validate_inputs('test', prompt, 'target', {})
                assert outcome[0] is False and 'sam_regions' in str(outcome), outcome
                del prompt['target']['inputs']['sam_model']
                outcome = await execution.validate_inputs('test', prompt, 'target', {})
                assert outcome[0] is True, outcome
        for node_id in ('WDV3Inspect', 'WDV3SaveConfig', 'WDV3CombineDetectors', 'WDV3CombineSAMSources'):
            prompt = graph(node_id)
            prompt['target']['inputs'] = {k: v for k, v in prompt['target']['inputs'].items()
                                           if not isinstance(v, list)}
            outcome = await execution.validate_inputs('test', prompt, 'target', {})
            assert outcome[0] is False, (node_id, outcome)
        prompt = graph('WDV3Inspect')
        prompt['evidence'] = {'class_type': 'Test_WDV3_EVIDENCE', 'inputs': {}}
        prompt['target']['inputs']['evidence'] = ['evidence', 0]
        outcome = await execution.validate_inputs('test', prompt, 'target', {})
        assert outcome[0] is False, outcome
        prompt = graph('WDV3Tagger')
        del prompt['target']['inputs']['images']
        outcome = await execution.validate_inputs('test', prompt, 'target', {})
        assert outcome[0] is False and 'required_input_missing' in str(outcome), outcome
        # Scalar values supplied by another node are unknown until execution.
        prompt = graph('WDV3Options')
        nodes.NODE_CLASS_MAPPINGS['Test_INT'] = type('Test_INT', (), {
            'INPUT_TYPES': classmethod(lambda cls: {'required': {}}),
            'RETURN_TYPES': ('INT',), 'FUNCTION': 'run', 'run': fail})
        prompt['integer'] = {'class_type': 'Test_INT', 'inputs': {}}
        prompt['target']['inputs']['sam_batch_size'] = ['integer', 0]
        outcome = await execution.validate_inputs('test', prompt, 'target', {})
        assert outcome[0] is True, outcome
        assert calls == [], calls
        print(f'PASS: {len(package.NODE_CLASS_MAPPINGS)} node schemas, {len(cases)} invalid configurations, '
              'optional links, linked types, deferred values; zero node executions.')

    # Validation of synchronous nodes does not suspend. Drive the coroutine
    # directly: Windows asyncio creates a loopback socket even without network I/O,
    # which is unavailable in restricted test environments.
    coroutine = tests()
    try:
        yielded = coroutine.send(None)
    except StopIteration:
        pass
    else:
        coroutine.close()
        raise AssertionError(f'Unexpected asynchronous work during validation: {yielded!r}')


if __name__ == '__main__':
    main()
