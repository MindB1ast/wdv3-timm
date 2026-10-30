"""Transformers SAM 3/3.1 with CPU support and shared region output semantics."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

import numpy as np

if __package__:
    from .sam3_backend import OfficialSam3Backend, file_hash
else:
    from sam3_backend import OfficialSam3Backend, file_hash

METADATA_FILES = (
    "config.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json",
    "tokenizer.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
)
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "model-*.safetensors")


def resolve_transformers_folder(settings):
    """Fill missing companion files without overwriting existing local weights/configs."""
    text = settings.get("checkpoint", "")
    checkpoint = Path(text) if text else None
    if checkpoint is not None and checkpoint.suffix.lower() == ".safetensors":
        if checkpoint.name != "model.safetensors":
            raise ValueError("Transformers loader expects the unquantized model.safetensors; FP8/INT8 variants need a quantized loader")
        folder = checkpoint.parent
    elif checkpoint is not None:
        folder = checkpoint
    else:
        folder = None
    has_weights = folder is not None and ((folder / "model.safetensors").is_file()
                                          or (folder / "model.safetensors.index.json").is_file())
    has_config = folder is not None and (folder / "config.json").is_file()
    has_processor = folder is not None and ((folder / "processor_config.json").is_file()
                                            or (folder / "preprocessor_config.json").is_file())
    has_tokenizer = folder is not None and ((folder / "tokenizer.json").is_file() or (
        (folder / "vocab.json").is_file() and (folder / "merges.txt").is_file()))
    if has_weights and has_config and has_processor and has_tokenizer:
        return folder.resolve()
    repo_id = settings.get("model_id")
    if not repo_id:
        raise ValueError(
            "Transformers SAM needs config, processor and tokenizer beside the weights. "
            "Set model.model_id to the source repository (e.g. unsloth/sam3.1) and "
            "allow_download=true to fetch missing files, or provide a complete local model folder."
        )
    from huggingface_hub import snapshot_download
    patterns = list(METADATA_FILES)
    if not has_weights:
        patterns.extend(WEIGHT_FILES)
    snapshot = Path(snapshot_download(repo_id=repo_id, revision=settings.get("revision"),
        allow_patterns=patterns, local_files_only=not settings.get("allow_download", False)))
    if folder is None:
        return snapshot.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        source, target = snapshot / name, folder / name
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
    if not has_weights:
        for source in snapshot.glob("model*.safetensors*"):
            if not (folder / source.name).exists():
                shutil.copyfile(source, folder / source.name)
    if checkpoint is not None and checkpoint.suffix == ".safetensors" and not checkpoint.is_file():
        raise FileNotFoundError(f"Model repository did not provide {checkpoint.name}")
    return folder.resolve()


class TransformersSam3Backend(OfficialSam3Backend):
    """Use the same batch/candidate/mask pipeline with a different model adapter."""
    def __init__(self, settings):
        import torch
        try:
            import transformers
            from transformers import Sam3Model, Sam3Processor
        except ImportError as error:
            raise RuntimeError(
                "Transformers SAM backend is unavailable in sam_python. Install "
                "requirements-sam-transformers.txt in that interpreter or set sam_python "
                "to an environment with transformers>=5 and SAM3 support."
            ) from error
        self.torch = torch
        self.device = settings["device"]
        folder = resolve_transformers_folder(settings)
        config_data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config_data.get("model_type") != "sam3":
            raise ValueError("Transformers model config must have model_type=sam3")
        self.processor = Sam3Processor.from_pretrained(str(folder), local_files_only=True,
                                                      trust_remote_code=False)
        self.model, loading_info = Sam3Model.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, use_safetensors=True, output_loading_info=True,
        )
        failures = {key: loading_info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        ) if loading_info.get(key)}
        if failures:
            raise ValueError(f"Incomplete/incompatible Transformers SAM weights: {failures}")
        self.model.eval()
        if not callable(getattr(self.model, "get_vision_features", None)):
            raise RuntimeError("Installed Transformers SAM lacks the batched vision-feature API")
        weight_files = sorted(folder.glob("model*.safetensors"))
        self.provenance = {
            "backend": "transformers", "device": self.device, "precision": "float32",
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers_version": transformers.__version__, "cuda": torch.version.cuda,
            "model_id": settings.get("model_id"), "revision": settings.get("revision"),
            "checkpoint_format": "transformers_safetensors",
            "checkpoint_sha256": {p.name: file_hash(p) for p in weight_files},
            "metadata_sha256": {name: file_hash(folder / name) for name in METADATA_FILES
                                if (folder / name).is_file()},
            "validation": "requires_full_model_smoke",
        }

    def _encode_images(self, frames):
        inputs = self.processor(images=frames, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=self.device, dtype=self.torch.float32)
        return self.model.get_vision_features(pixel_values=pixels)

    def _predict_source(self, state, source, batch_size):
        inputs = self.processor(text=[source["prompt"]] * batch_size, padding=True, return_tensors="pt")
        return self.model(vision_embeds=state,
                          input_ids=inputs["input_ids"].to(self.device),
                          attention_mask=inputs["attention_mask"].to(self.device))

    def _candidates(self, predictions, index, size, confidence):
        torch = self.torch
        scores = predictions.pred_logits[index].sigmoid()
        if predictions.presence_logits is not None:
            scores = scores * predictions.presence_logits[index].sigmoid().reshape(())
        width, height = size
        for candidate, score in enumerate(scores):
            value = float(score.item())
            if value <= confidence:
                yield np.empty((0, 0)), value, []
                continue
            x1, y1, x2, y2 = predictions.pred_boxes[index, candidate].tolist()
            bbox = [x1*width, y1*height, x2*width, y2*height]
            raw = predictions.pred_masks[index, candidate][None, None]
            probabilities = torch.nn.functional.interpolate(raw.float(), size=(height, width),
                mode="bilinear", align_corners=False).sigmoid()[0, 0].cpu().numpy()
            yield probabilities, value, bbox
