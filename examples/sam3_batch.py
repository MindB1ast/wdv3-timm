"""Run from the repository root using the WD interpreter; SAM has its own Python."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scripts import BatchTagging, ScriptOptions
from Scripts.visualization import inspect_saved_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--sam-python", type=Path, help="Defaults to the current Python interpreter")
    parser.add_argument("--sam-config", type=Path, default=Path("sam_sources.example.json"))
    parser.add_argument("--output", type=Path, default=Path("region_runs"))
    parser.add_argument("--sam-batch", type=int, default=1)
    parser.add_argument("--region-batch", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    results = BatchTagging(ScriptOptions(
        ImageFolder=args.images, sam_config=args.sam_config, sam_python=args.sam_python,
        regions_output_dir=args.output, sam_batch_size=args.sam_batch, region_batch_size=args.region_batch,
        recursive=True, save_txt=False, output_file=None,
    ))
    if results and args.show:
        for detail in results[0]:
            for source, regions in detail["detectors"].items():
                if regions:
                    inspect_saved_region(detail["export_root"], detail["image_id"], source, regions[0]["region_id"])
                    return


if __name__ == "__main__":
    main()
