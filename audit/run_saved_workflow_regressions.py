"""Compare saved ComfyUI graphs with pinned outputs."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'audit/fixtures/saved_workflows'
SAVERS = {'WDV3SaveResult', 'WDV3SaveRegionResults', 'WDV3TagFolder'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compare(case, directory):
    expected = case['expected']
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} if directory.is_dir() else set()
    if actual != expected.keys():
        raise ValueError(f'missing {sorted(expected.keys() - actual)}, extra {sorted(actual - expected.keys())}')
    for name, reference in expected.items():
        data = (directory / name).read_bytes()
        if 'fixture' in reference:
            expected_data = (FIXTURES / case['name'] / reference['fixture']).read_bytes()
            if name.endswith('.regions.json'):
                matches = json.loads(data) == json.loads(expected_data)
            else:
                matches = data == expected_data
            if not matches:
                raise ValueError(f'{name}: differs from pinned output')
        else:
            detail = json.loads(data)
            # The runner disables downloads. Compare every other field.
            for settings in (detail['settings']['sources'], detail['geometry']['yolo'], detail['source_settings']):
                del settings[0]['download_missing']
            fingerprint = digest(json.dumps(detail, sort_keys=True, separators=(',', ':')).encode())
            if fingerprint != reference['canonical_sha256']:
                raise ValueError(f'{name}: JSON checksum differs ({fingerprint})')


def load_nodes(comfy_root, input_dir, models_dir, output_dir):
    os.environ.update(YOLO_CONFIG_DIR=str(output_dir / 'yolo'),
                      MPLCONFIGDIR=str(output_dir / 'mpl'), HF_HUB_OFFLINE='1')
    sys.path[:0] = [str(comfy_root), str(ROOT)]
    sys.argv = ['saved-workflow-regression', '--cpu', '--base-directory', str(output_dir)]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(output_dir))
    folder_paths.models_dir = str(models_dir)
    spec = importlib.util.spec_from_file_location(
        'wdv3_workflow_regression', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    import nodes
    return {**package.NODE_CLASS_MAPPINGS, 'LoadImage': nodes.LoadImage}


def run_case(case, directory, classes, folder_input=None):
    graph = json.loads((FIXTURES / case['name'] / 'workflow.json').read_bytes())
    links = {row[0]: row for row in graph['links']}
    by_id = {n['id']: n for n in graph['nodes']}
    cache = {}

    def execute(node_id):
        if node_id in cache:
            return cache[node_id]
        node = by_id[node_id]
        kind = node['type']
        cls = classes[kind]
        schema = cls.INPUT_TYPES()
        fields = schema.get('required', {}) | schema.get('optional', {})
        saved = node.get('widgets_values_named') or {}
        if not saved and node.get('widgets_values'):
            raise ValueError(f'{kind} {node_id}: saved widget names missing')
        list_input = getattr(cls, 'INPUT_IS_LIST', False)
        args = {name: [value] if list_input else value for name, value in saved.items() if name in fields}
        for slot in node.get('inputs', []):
            if slot.get('link') is None:
                continue
            _, source_id, index, _, _, _ = links[slot['link']]
            value = execute(source_id)[index]
            source = classes[by_id[source_id]['type']]
            flags = getattr(source, 'OUTPUT_IS_LIST', ())
            list_output = index < len(flags) and flags[index]
            if list_input and not list_output:
                value = [value]
            elif list_output and not list_input:
                if len(value) != 1:
                    raise ValueError(f'{kind}: expected one linked result, got {len(value)}')
                value = value[0]
            group, separator, field = slot['name'].partition('.')
            if separator:
                args.setdefault(group, {})[field] = value
            else:
                args[group] = value
        if 'download_missing' in args:
            args['download_missing'] = False
        if kind == 'WDV3LoadFolder' and folder_input is not None:
            args.update(directory=str(folder_input), recursive=case['recursive'])
        if kind in SAVERS:
            args['output_directory'] = [str(directory)] if list_input else str(directory)
        result = getattr(cls(), cls.FUNCTION)(**args)
        if hasattr(result, 'result'):
            result = result.result
        if isinstance(result, dict):
            result = result['result']
        cache[node_id] = result
        return result

    directory.mkdir()
    for node in graph['nodes']:
        if node['type'] in SAVERS:
            execute(node['id'])
    return cache


def main():
    cases = json.loads((FIXTURES / 'manifest.json').read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--comfy-root', type=Path)
    mode.add_argument('--compare-dir', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--folder-input', type=Path, help='Directory with saved Tag Folder inputs')
    parser.add_argument('--case', action='append', choices=[c['name'] for c in cases])
    args = parser.parse_args()
    if args.case:
        cases = [c for c in cases if c['name'] in args.case]
    output = args.compare_dir
    if args.comfy_root:
        if not args.input_dir or not args.input_dir.is_dir():
            parser.error('--comfy-root requires an existing --input-dir')
        temp = ROOT / 'temp'
        temp.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='saved-workflows-', dir=temp))
        print(f'Outputs: {output}', flush=True)
        classes = load_nodes(args.comfy_root.resolve(), args.input_dir.resolve(),
                             (args.models_dir or args.comfy_root / 'models').resolve(), output)
    failures = 0
    for case in cases:
        name = case['name']
        print(f'CHECK {name}', flush=True)
        try:
            if args.comfy_root:
                folder_input = None
                if 'input_files' in case:
                    source = args.folder_input or args.input_dir / case['folder_name']
                    folder_input = output / 'folder-inputs' / name
                    for relative, expected_digest in case['input_files'].items():
                        image = source / relative
                        if digest(image.read_bytes()) != expected_digest:
                            raise ValueError(f'input checksum differs: {relative}')
                        target = folder_input / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(image, target)
                else:
                    image = args.input_dir / case['image']
                    if digest(image.read_bytes()) != case['image_sha256']:
                        raise ValueError(f'input checksum differs: {image.name}')
                run_case(case, output / name, classes, folder_input)
            compare(case, output / name)
        except Exception as error:
            failures += 1
            print(f'FAIL {name}: {error}', flush=True)
        else:
            print(f'PASS {name}', flush=True)
        gc.collect()
    print(f'{len(cases) - failures}/{len(cases)} passed', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
