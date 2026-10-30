"""Resolve SAM device and weight backend before loading a model."""
import json
from pathlib import Path

if __package__:
    from .sam_checkpoint import inspect_sam_checkpoint
else:
    from sam_checkpoint import inspect_sam_checkpoint


def resolve_device(requested, torch):
    """Auto probes a usable CUDA device; explicit selections never silently change."""
    requested = requested or "auto"
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and requested != "cuda" and not (
        requested.startswith("cuda:") and requested[5:].isdigit()
    ):
        raise ValueError("device must be auto, cpu, cuda or cuda:N")
    candidate = "cuda" if requested == "auto" else requested
    try:
        index = int(candidate.split(":", 1)[1]) if ":" in candidate else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {candidate}")
        probe = torch.empty(1, device=candidate)
        del probe
        return candidate
    except (RuntimeError, AssertionError) as error:
        if requested != "auto":
            raise RuntimeError(f"Requested {candidate} cannot be used: {error}") from error
        return "cpu"


def resolve_backend(settings):
    """Inspect the local layout first; a repo id without local weights selects HF."""
    requested = settings.get("backend", "auto")
    if requested not in ("auto", "native", "transformers"):
        raise ValueError("backend must be auto, native or transformers")
    checkpoint_text = settings.get("checkpoint", "")
    checkpoint = Path(checkpoint_text) if checkpoint_text else None
    detected = None
    if checkpoint is not None and checkpoint.is_file():
        layout = inspect_sam_checkpoint(checkpoint, allow_transformers=True)
        detected = "transformers" if layout == "transformers" else "native"
    elif checkpoint is not None and checkpoint.is_dir() and (checkpoint / "config.json").is_file():
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "sam3":
            detected = "transformers"
    elif checkpoint is not None and checkpoint.suffix.lower() in (".pt", ".pth"):
        detected = "native"
    if requested != "auto":
        if detected is not None and detected != requested:
            raise ValueError(f"backend={requested} conflicts with the detected {detected} checkpoint")
        return requested
    return detected or ("transformers" if settings.get("model_id") else "native")


def create_backend(settings):
    import torch
    resolved = dict(settings)
    backend = resolve_backend(resolved)
    resolved["device"] = resolve_device(resolved.get("device", "auto"), torch)
    if backend == "native":
        if resolved["device"] == "cpu":
            raise RuntimeError(
                "Native SAM weights require CUDA in this backend. For CPU use a Transformers "
                "model such as model_id=unsloth/sam3.1 with backend=auto and device=auto."
            )
        if __package__:
            from .sam3_backend import OfficialSam3Backend
        else:
            from sam3_backend import OfficialSam3Backend
        return OfficialSam3Backend(resolved)
    if __package__:
        from .sam_transformers import TransformersSam3Backend
    else:
        from sam_transformers import TransformersSam3Backend
    return TransformersSam3Backend(resolved)
