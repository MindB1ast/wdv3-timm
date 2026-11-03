"""Export real WDV3 port definitions using the installed ComfyUI Python.

No inference or model download is performed. Pass --comfy-root to ComfyUI sources.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('temp/connection-node-defs.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.argv = ['connection-catalog', '--cpu', '--base-directory', str(root / 'temp/connection-comfy')]
    import comfy.options
    comfy.options.enable_args_parsing()
    spec = importlib.util.spec_from_file_location('wdv3_connection_catalog', root / '__init__.py',
                                                 submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    definitions = {name: dict(name=name, display_name=package.NODE_DISPLAY_NAME_MAPPINGS[name],
                              input=cls.INPUT_TYPES(), output=cls.RETURN_TYPES, output_name=cls.RETURN_NAMES)
                   for name, cls in package.NODE_CLASS_MAPPINGS.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(definitions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(definitions)} real node definitions to {args.output}')


if __name__ == '__main__':
    main()
