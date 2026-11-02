"""Use in a notebook or run from the repository root; downloads only chosen weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scripts import BatchTagging, ScriptOptions


def tag_folder(folder, model='animetimm/caformer_b36.dbv4-full'):
    """Other examples: animetimm/convnextv2_huge.dbv4-full, vit (legacy WD)."""
    return BatchTagging(ScriptOptions(
        ImageFolder=folder, model=model, gen_threshold=0.35, char_threshold=0.75,
        region_batch_size=2, save_txt=False, output_file=None,
    ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--model', default='animetimm/caformer_b36.dbv4-full')
    args = parser.parse_args()
    tag_folder(args.folder, args.model)
