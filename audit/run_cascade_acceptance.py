"""Installed ordinary ComfyUI nodes with nested Region Tag Rules; no real weights/UI."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root/'temp/cascade-acceptance'
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(work/'yolo'), MPLCONFIGDIR=str(work/'matplotlib'), TEMP=str(work), TMP=str(work))
    sys.path[:0] = [str(args.comfy_root), str(root)]
    sys.argv = ['nested-acceptance', '--cpu', '--base-directory', str(work)]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    from PIL import Image
    from audit.test_comfy_nodes import PACKAGE, runtime
    from audit.test_nested_detectors import detect, nested
    from audit.run_tag_redundancy_acceptance import complete
    ui = importlib.import_module('wdv3_comfy_test.comfy_ui_nodes')
    pipe = importlib.import_module('wdv3_comfy_test.comfy_pipeline_nodes')
    post = importlib.import_module('wdv3_comfy_test.comfy_postprocessing_nodes')
    core = importlib.import_module('wdv3_comfy_test.comfy_cascade')
    reports = importlib.import_module('wdv3_comfy_test.comfy_reports')
    processing = importlib.import_module('wdv3_comfy_test.comfy_processing')
    removed = {'WDV3CascadeStage', 'WDV3CombineCascadeStages', 'WDV3CascadePlan', 'WDV3CollectCascadeEvidence', 'WDV3RouteCascadeTags'}
    assert not removed & PACKAGE.NODE_CLASS_MAPPINGS.keys()
    assert all('WDV3_CASCADE_PLAN' not in node.RETURN_TYPES for node in PACKAGE.NODE_CLASS_MAPPINGS.values())
    assert 'child_detectors' in ui.WDRegionRules.INPUT_TYPES()['optional']
    def rules(**kw):
        return ui.WDRegionRules.execute(.35, .75, '', '', '', '', full_image_only_tags='', **kw).result[0]
    def yolo(name, policy=None):
        return ui.WDYOLODetector.execute('yolov8n.pt', name, .25, '0', 'fake-'+name+'.pt', rules=policy).result[0]
    for restored in ('', '  ', None):
        assert rules(child_tag_route=restored, tag_enabled=restored) == rules()
    explicit = rules(child_tag_route='merge_to_parent', tag_enabled=False)['children']
    assert explicit['tag_route'] == 'merge_to_parent' and explicit['tag_enabled'] is False
    eyes = yolo('eyes')
    assert rules(child_detectors=eyes, child_tag_route='')['children']['tag_route'] == 'separate'
    face = yolo('face', rules(child_detectors=eyes, child_tag_route='merge_to_parent'))
    body = yolo('body', rules(child_detectors=face, child_tag_route='merge_to_parent'))
    with patch.object(ui, 'preset_directory', return_value=work/'presets'):
        ui.WDSaveConfig.execute('nested', True, detectors=body)
        restored = ui.WDLoadConfig.execute('Saved: nested').result[1]
    assert nested.build_plan(body) == nested.build_plan(restored)
    sampler = ui.WDSAMLoader.execute('unsloth/sam3.1', 'cpu', False, 'fake').result[0]
    query = ui.WDSAMSource.execute('detail', 'face', .5, .5, 1, 16, 'mask', 0).result[0]
    mixed = yolo('person', rules(child_detectors=face, child_sam_model=sampler, child_sources=query))
    assert len(nested.build_plan(mixed)[0]['stages']) == 4
    calls = []
    def adapter(images, stage):
        calls.append(stage.stage_id)
        return detect(images, stage)
    with patch.object(core, 'DetectorAdapters', return_value=adapter):
        wd = runtime()
        image = torch.ones(1, 12, 20, 3)
        direct = PACKAGE.NODE_CLASS_MAPPINGS['WDV3Tagger']().tag(image, wd, detectors=body)
        record = json.loads(direct['result'][1][0])
        assert len(record['detectors']['body/face/eyes']) == 2
        evidence, regions = pipe.WDCollectEvidence.execute(image, wd, detectors=body).result
        local = post.WDFilterTagSets.execute(evidence).result[0]
        result = post.WDMergeTagSets.execute(local).result[0]
        assert result['frames'][0]['cascade']['routed']
        assert result['frames'][0]['detectors']['body/face/eyes'][1]['stage_path'] == ['body', 'face', 'eyes']
        source = work/'dataset'
        source.mkdir(exist_ok=True)
        path = source/'example.png'
        with Image.new('RGB', (20, 12), 'white') as pixels:
            pixels.save(path)
        folder = processing.tag_folder({'root': source, 'paths': [path]}, runtime(), {}, detectors=body)
        assert 'body/face/eyes' in json.loads(folder[1][0])['detectors']
    report = reports.make_report(result)
    preview, _, _ = reports.preview_report(report, work/'preview')
    assert preview[0]['detail']['cascade']['routed']
    paths = reports.export_report(report, work/'reports')
    assert 'Aggregate:' in Path(paths['html']).read_text(encoding='utf-8')
    import execution
    import nodes
    nodes.NODE_CLASS_MAPPINGS.update(PACKAGE.NODE_CLASS_MAPPINGS)
    class Source:
        INPUT_TYPES = classmethod(lambda cls: {'required': {}})
        RETURN_TYPES = ('WDV3_EVIDENCE',)
        FUNCTION = 'run'
        run = staticmethod(lambda: (_ for _ in ()).throw(AssertionError('Validation executed node')))
    nodes.NODE_CLASS_MAPPINGS['NestedTestSource'] = Source
    prompt = {'raw': {'class_type': 'NestedTestSource', 'inputs': {}},
        'filter': {'class_type': 'WDV3FilterTagSets', 'inputs': {'evidence': ['raw', 0]}},
        'merge': {'class_type': 'WDV3MergeTagSets', 'inputs': {'tag_sets': ['filter', 0]}},
        'inspect': {'class_type': 'WDV3Inspect', 'inputs': {'results': ['merge', 0]}}}
    assert complete(execution.validate_prompt('nested', prompt, None))[0]
    from examples.build_comfy_workflows import build
    build(root)
    for suffix in ('merge', 'separate'):
        graph = json.loads((root/f'examples/comfyui_wdv3_cascade_{suffix}.json').read_text(encoding='utf-8'))
        by_id = {n['id']: n for n in graph['nodes']}
        assert not removed & {n['type'] for n in graph['nodes']}
        for link_id, src, out, dst, inp, kind in graph['links']:
            assert by_id[src]['outputs'][out]['type'] == kind == by_id[dst]['inputs'][inp]['type']
            assert by_id[dst]['inputs'][inp]['link'] == link_id
    destination = root/'docs/verification/cascaded-regions'
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(interface='Region Tag Rules child stacks', removed_public_stage_nodes=True,
        schema_api=True, preset_roundtrip=True, standard_tagger=True, standard_collect=True, standard_folder=True,
        automatic_merge=True, two_parents=True, mixed_yolo_sam=True, tree_export=True, prompt_validation=True,
        generated_examples=True, legacy_widget_defaults=True, actual_weight_inference=False, browser_tested=False, report=paths,
        fake_detector_calls=calls, python=sys.version)
    (destination/'acceptance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: Region Tag Rules child stacks, ordinary Tagger/Folder/Collect/Merge, presets, reports and schemas.')


if __name__ == '__main__':
    main()
