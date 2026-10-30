"""Inspect checkpoint layouts before allocating a SAM model or loading tensors."""
import json
from pathlib import Path
import struct


def inspect_sam_checkpoint(path, *, allow_transformers=False):
    """Identify weights without loading tensors; native callers reject HF layouts."""
    checkpoint = Path(path)
    if checkpoint.suffix.lower() != ".safetensors":
        return "pytorch"
    with checkpoint.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Invalid safetensors checkpoint: missing header")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= header_length <= min(64 * 1024 * 1024, checkpoint.stat().st_size-8):
            raise ValueError("Invalid safetensors checkpoint: invalid header length")
        try:
            header = json.loads(stream.read(header_length))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Invalid safetensors checkpoint header") from error
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors checkpoint: header must be an object")
    names = [key for key in header if key != "__metadata__"]
    if not names:
        raise ValueError("Empty safetensors checkpoint")
    if any(key.startswith(("detr_decoder.", "vision_encoder.", "model.detr_decoder.",
                           "model.vision_encoder.")) for key in names):
        if allow_transformers:
            return "transformers"
        raise ValueError(
            f"{checkpoint}: это SAM checkpoint в формате Hugging Face Transformers "
            "(например detr_decoder.*), несовместимом с native backend facebookresearch/sam3. "
            "Для текущего backend укажите официальный sam3.pt в model.checkpoint "
            "или native safetensors с исходными ключами SAM. "
            "Переименование model.safetensors в sam3.pt не конвертирует веса."
        )
    return "native_safetensors"


def build_sam_from_checkpoint(builder, checkpoint):
    """Load native safetensors strictly; retain the official loader for .pt files."""
    layout = inspect_sam_checkpoint(checkpoint)
    if layout == "pytorch":
        return builder(checkpoint_path=str(checkpoint), device="cpu", load_from_HF=False,
                       enable_inst_interactivity=False)
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("Install safetensors in the separate sam_python environment to load native .safetensors") from error
    weights = load_file(str(checkpoint), device="cpu")
    # Official complete SAM checkpoints can contain detector and tracker weights;
    # only the detector is used for the image text-segmentation backend.
    prefixed = any(key.startswith("detector.") for key in weights)
    if prefixed:
        weights = {key.removeprefix("detector."): value for key, value in weights.items()
                   if key.startswith("detector.")}
    model = builder(checkpoint_path=None, device="cpu", load_from_HF=False,
                    enable_inst_interactivity=False)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    wrong_shapes = sorted(key for key in set(expected) & set(weights)
                          if expected[key].shape != weights[key].shape)
    if missing or unexpected or wrong_shapes:
        raise ValueError(
            "Native SAM safetensors are incompatible with this SAM architecture: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, wrong_shapes={wrong_shapes[:5]}. "
            "Use weights matching the installed facebookresearch/sam3 revision."
        )
    model.load_state_dict(weights, strict=True)
    return model
