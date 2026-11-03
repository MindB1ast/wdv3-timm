"""Run real ComfyUI node schemas/execution with deterministic WD/YOLO test doubles.

Use the installed ComfyUI Python interpreter and pass --comfy-root to its source.
All outputs are isolated in the project; no service or model download is needed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / 'temp/tag-regex-acceptance'
    work.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(work / 'yolo')
    sys.path.insert(0, str(args.comfy_root))
    sys.argv = ['regex-acceptance', '--cpu', '--base-directory', str(work)]
    import comfy.options
    comfy.options.enable_args_parsing()
    spec = importlib.util.spec_from_file_location('regex_acceptance', root / '__init__.py',
                                                 submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    import numpy as np
    import torch
    from regex_acceptance.comfy_nodes import WDRuntime, DetectorConfig, WDTagger
    from regex_acceptance.Scripts.wdv3_timm import LabelData
    from regex_acceptance.comfy_ui_nodes import WDRegionRules, WDTaggingOptions, WDPreview, WDSaveResult, WDSAMSource
    from regex_acceptance.comfy_pipeline_nodes import WDCollectEvidence, WDApplyFilters, WDInspect, WDExportReport
    from regex_acceptance.comfy_processing import tag_folder

    # These are actual installed ComfyUI schema objects, not a mock API.
    for cls in (WDRegionRules, WDTaggingOptions):
        inputs = {i.id: i for i in cls.GET_SCHEMA().inputs}
        assert inputs['remove_tags_regex' if cls == WDTaggingOptions else 'allowed_tags_regex'].tooltip
        literal = inputs['remove_tags' if cls == WDTaggingOptions else 'allowed_tags'].tooltip
        assert 'blue eyes = blue_eyes' in literal and 'Regex' in literal
    assert {i.id: i for i in WDTaggingOptions.GET_SCHEMA().inputs}['tag_separator'].default == ', '
    rules = WDRegionRules.execute(.35, .75, '', '', '', '',
                                  allowed_tags='blue eyes', allowed_tags_regex='.*_hair').result[0]
    assert rules['allowed_tags'] == ['blue_eyes']
    assert rules['allowed_tags_regex'] == ['.*_hair']
    sam_sources = WDSAMSource.execute('sam-eyes', 'eyes', .5, .5, 1, 4, 'mask', 0, rules=rules).result[0]
    assert sam_sources[0]['rules']['allowed_tags_regex'] == ['.*_hair']
    opts = WDTaggingOptions.execute(tag_separator=' | ', replace_underscores=False,
                                    remove_tags_regex='bad.*').result[0]
    assert opts['tag_separator'] == ' | ' and opts['remove_tags_regex'] == ['bad.*']

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return torch.tensor([[4., 3., 3.]]).repeat(len(batch), 1)

    class Detector:
        def __init__(self):
            self.calls = 0

        def __call__(self, image, **kwargs):
            self.calls += 1
            return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[0, 0, 4, 4]]))])]

    model = Model()
    runtime = WDRuntime(model, lambda image: torch.from_numpy(np.array(image.resize((4, 4)))).permute(2, 0, 1).float()/255,
                        LabelData(['blue_eyes', 'long_hair', 'hero_(series)'], [], [0, 1], [2]), torch.device('cpu'))
    a, b = Detector(), Detector()
    configs = [DetectorConfig('eyes', 'fake-eyes', **dict(rules)), DetectorConfig('face', 'fake-face')]
    detectors = (configs, {'fake-eyes': a, 'fake-face': b})
    image = torch.ones(1, 8, 8, 3)
    evidence = WDCollectEvidence.execute(image, runtime, detectors=detectors).result[0]
    snapshot = evidence.to_json()
    calls = (model.calls, a.calls, b.calls)
    first = WDApplyFilters.execute(evidence, options=opts, detectors=detectors)
    configs[0].remove_tags_from_region_regex = ['.*_eyes']
    second = WDApplyFilters.execute(evidence, options=opts, detectors=detectors)
    assert (model.calls, a.calls, b.calls) == calls and evidence.to_json() == snapshot
    results, tags, encoded = second.result
    assert tags[0] == r'blue_eyes | hero_\(series\) | long_hair'
    assert any(d.get('matched_rules') for d in json.loads(encoded[0])['decisions'])
    inspect = WDInspect.execute(results=results)
    report, _, inspected_tags = inspect.result
    assert inspected_tags == tags
    viewer = WDPreview.execute(encoded, [16], images=[image])
    assert viewer.ui['wdv3_results'][0]['detail']['settings']['options']['tag_separator'] == ' | '
    exported = WDExportReport.execute(report, str(work), 'regex', True, True, True)
    assert Path(exported.result[2]).read_text(encoding='utf-8') == tags[0]
    saved = WDSaveResult.execute(encoded, [str(work)], ['result'], [0], [True], [True], [True])
    assert Path(saved.result[0]).read_text(encoding='utf-8') == tags[0]
    from PIL import Image
    sample = work / 'sample.png'
    Image.new('RGB', (8, 8), 'white').save(sample)
    folder_tags, _ = tag_folder({'paths': [sample], 'root': str(work)}, runtime, opts, output_directory=work / 'folder',
                                save_txt=True, save_json=True, overwrite=True)
    assert folder_tags == tags
    assert (work / 'folder/sample.png.txt').read_text(encoding='utf-8') == tags[0]
    # The tagger exposes connections only; missing options use standard defaults.
    old_workflow = json.loads((root / 'examples/comfyui_wdv3.json').read_text(encoding='utf-8'))
    old_values = next(n['widgets_values'] for n in old_workflow['nodes'] if n['type'] == 'WDV3Tagger')
    inputs = WDTagger.INPUT_TYPES()
    widget_names = [name for group in ('required', 'optional') for name, spec in inputs[group].items()
                    if spec[0] in ('FLOAT', 'INT', 'BOOLEAN', 'STRING')]
    assert widget_names == []
    restored = dict(zip(widget_names, old_values))
    legacy = WDTagger().tag(image, runtime, **restored)['result'][0][0]
    assert legacy == r'blue eyes, hero \(series\), long hair'
    summary = dict(python=sys.version, schemas='actual installed ComfyUI', inference='deterministic test doubles',
                   sources=['eyes', 'face'], calls_before_filter=calls,
                   calls_after_two_filters=calls, text=tags[0], immutable_evidence=True,
                   inspect_viewer_export_folder_save_result=True, legacy_workflow_defaults=True,
                   report_html=exported.result[1])
    output = root / 'docs/verification/tag-regex-output'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'acceptance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
