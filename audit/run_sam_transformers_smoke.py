"""Real Transformers SAM CPU/worker smoke with tiny random weights, no network."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (PreTrainedTokenizerFast, Sam3Config, Sam3ImageProcessor,
                          Sam3Model, Sam3Processor)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scripts.sam3_adapter import SamWorker, SamWorkerError
from Scripts.sam_config import SamConfig, SamSourceConfig


def tiny_model_folder(folder):
    """Small architecture to verify APIs, not segmentation quality of real SAM."""
    dimension = 48
    layer = {"hidden_size": dimension, "intermediate_size": 96,
             "num_attention_heads": 4, "num_layers": 1}
    config = Sam3Config(
        vision_config={"backbone_config": {"model_type": "sam3_vit_model", "hidden_size": dimension, "intermediate_size": 96,
            "num_hidden_layers": 1, "num_attention_heads": 4, "image_size": 64,
            "pretrain_image_size": 64, "patch_size": 8, "window_size": 4,
            "global_attn_indexes": [0]}, "fpn_hidden_size": dimension,
            "scale_factors": [2., 1.], "backbone_feature_sizes": [[16, 16], [8, 8]]},
        text_config={"hidden_size": dimension, "intermediate_size": 96,
            "projection_dim": dimension, "num_hidden_layers": 1, "num_attention_heads": 4,
            "vocab_size": 16, "max_position_embeddings": 32, "bos_token_id": 1,
            "eos_token_id": 2, "pad_token_id": 3},
        geometry_encoder_config=layer, detr_encoder_config=layer,
        detr_decoder_config={**layer, "num_queries": 4},
        mask_decoder_config={"hidden_size": dimension, "num_attention_heads": 4,
                             "num_upsampling_stages": 1},
    )
    model = Sam3Model(config).eval()
    model.save_pretrained(folder)
    del model
    vocabulary = {"[UNK]": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "eye": 4, "skirt": 5}
    raw_tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, unk_token="[UNK]",
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", model_max_length=32)
    processor = Sam3Processor(image_processor=Sam3ImageProcessor(size={"height": 64, "width": 64}),
                              tokenizer=tokenizer)
    processor.save_pretrained(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("temp/sam-transformers-smoke.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    report = {"status": "running", "scope": "tiny random SAM; real library and CPU subprocess; not user weights"}
    try:
        with tempfile.TemporaryDirectory(dir="temp", prefix="sam-hf-smoke-") as scratch:
            root = Path(scratch).resolve()
            folder = root / "model"
            tiny_model_folder(folder)
            images = []
            for index, size in enumerate([(31, 23), (47, 35)]):
                path = root / f"image-{index}.png"
                with Image.new("RGB", size, "white" if index == 0 else "blue") as image:
                    image.save(path)
                images.append({"image_id": str(index), "path": str(path), "size": list(size)})
            sources = tuple(SamSourceConfig(prompt, prompt, prompt, confidence=0,
                            mask_threshold=0, max_regions=4) for prompt in ("eye", "skirt"))
            config = SamConfig(str(folder), "auto", False, sources)
            batch_dir, serial_dir = root/"batch", root/"serial"
            batch_dir.mkdir(); serial_dir.mkdir()
            with SamWorker(sys.executable, config, timeout=180) as worker:
                assert worker.provenance["backend"] == "transformers"
                assert worker.provenance["device"] == "cpu", "Run this smoke with CPU PyTorch"
                process = worker.process
                worker.load()
                batch = worker.segment(images, batch_dir)
                worker.offload()
                serial = []
                for image in images:
                    destination = serial_dir / image["image_id"]
                    destination.mkdir()
                    worker.load()
                    serial.extend(worker.segment([image], destination))
                    worker.offload()
                    assert worker.process is process
                keyed = {(r["image_id"], r["source_id"]): r for r in serial}
                assert len(batch) == 4
                for record in batch:
                    other = keyed[record["image_id"], record["source_id"]]
                    assert len(record["regions"]) == len(other["regions"]) > 0
                    for left, right in zip(record["regions"], other["regions"]):
                        assert left["bbox"] == right["bbox"]
                        assert left["mask_sha256"] == right["mask_sha256"]
                        np.testing.assert_allclose(left["segmentation_score"], right["segmentation_score"],
                                                   rtol=1e-4, atol=1e-5)
                report.update(provenance=worker.provenance, metrics=worker.last_metrics,
                              images=2, prompts=2, batches=[1, 2], load_offload_roundtrip=True)
            weights = load_file(folder / "model.safetensors")
            missing_key = next(key for key in weights if key.startswith("vision_encoder.") and key.endswith("weight"))
            del weights[missing_key]
            save_file(weights, folder / "model.safetensors", metadata={"format": "pt"})
            try:
                with SamWorker(sys.executable, config, timeout=180):
                    raise AssertionError("Incomplete weights were silently accepted")
            except SamWorkerError as error:
                assert "Incomplete/incompatible" in str(error), str(error)
            report["missing_weights_rejected"] = True
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
