"""Official SAM 3 image backend, imported only by its isolated interpreter.

The batch API is tied to the recorded hashes of upstream source files. Run the
real GPU acceptance before declaring a particular upstream revision supported.
"""
from contextlib import ExitStack
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image

if __package__:
    from .sam_masks import select_masks
    from .sam_checkpoint import build_sam_from_checkpoint, inspect_sam_checkpoint
else:
    from sam_masks import select_masks
    from sam_checkpoint import build_sam_from_checkpoint, inspect_sam_checkpoint


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialSam3Backend:
    def __init__(self, settings):
        checkpoint = Path(settings["checkpoint"])
        if checkpoint.is_file():
            inspect_sam_checkpoint(checkpoint)
        if sys.version_info < (3, 12):
            raise RuntimeError("SAM 3 requires Python >=3.12; set sam_python to a separate environment")
        import torch
        self.torch = torch
        version = torch.__version__.split("+")[0].split(".")
        if tuple(map(int, version[:2])) < (2, 7):
            raise RuntimeError("SAM 3 requires torch >=2.7 in its separate environment")
        self.device = settings["device"]
        device = torch.device(self.device)
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError(f"SAM CUDA device unavailable: {self.device}")
        if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 6):
            raise RuntimeError("SAM 3 requires a PyTorch build with CUDA >=12.6")
        torch.cuda.set_device(device)
        if not checkpoint.is_file():
            if not settings["allow_download"]:
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}; allow_download is false")
            from huggingface_hub import hf_hub_download
            checkpoint = Path(hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt"))
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.data_misc import FindStage
        self.find_stage_type = FindStage
        self.model = build_sam_from_checkpoint(build_sam3_image_model, checkpoint)
        self.model.eval()
        self.processor = Sam3Processor(self.model, device="cpu", confidence_threshold=0)
        if not callable(getattr(self.processor, "set_image_batch", None)):
            raise RuntimeError("Installed SAM 3 revision lacks the image batch API")
        package_root = Path(sam3.__file__).resolve().parent
        source_files = ("model_builder.py", "model/sam3_image_processor.py", "model/sam3_image.py")
        self.provenance = {
            "backend": "sam3", "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": self.device,
            "sam3_version": importlib.metadata.version("sam3"),
            "source_sha256": {name: file_hash(package_root / name) for name in source_files},
            "checkpoint_sha256": file_hash(checkpoint), "checkpoint_name": checkpoint.name,
            "checkpoint_format": inspect_sam_checkpoint(checkpoint),
            "precision": "float32", "validation": "requires_gpu_smoke",
        }

    def offload(self):
        self.model.to("cpu")
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def load(self):
        """Restore the existing model without reading checkpoint files again."""
        self.model.to(self.device)

    def _encode_images(self, frames):
        self.processor.device = self.device
        return self.processor.set_image_batch(frames)

    def _predict_source(self, state, source, batch_size):
        torch = self.torch
        source_backbone = dict(state["backbone_out"])
        source_backbone.update(self.model.backbone.forward_text([source["prompt"]], device=self.device))
        stage = self.find_stage_type(
            img_ids=torch.arange(batch_size, device=self.device),
            text_ids=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        return self.model.forward_grounding(
            backbone_out=source_backbone, find_input=stage, find_target=None,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=batch_size),
        )

    def segment(self, images, sources, directory):
        torch = self.torch
        started = time.monotonic()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        directory = Path(directory)
        results = []
        state = predictions = candidates = None
        try:
            self.load()
            with ExitStack() as stack, torch.inference_mode():
                frames = [stack.enter_context(Image.open(image["path"])) for image in images]
                if any(frame.mode != "RGB" or list(frame.size) != image["size"]
                       for frame, image in zip(frames, images)):
                    raise ValueError("Worker expects canonical RGB image sizes")
                # One actual visual-backbone batch, with IDs restored after grounding.
                state = self._encode_images(frames)
                for source_index, source in enumerate(sources):
                    predictions = self._predict_source(state, source, len(frames))
                    for image_index, (image, frame) in enumerate(zip(images, frames)):
                        candidates = self._candidates(predictions, image_index, frame.size, source["confidence"])
                        try:
                            selected, counts = select_masks(candidates, source, frame.size)
                        finally:
                            candidates.close()
                            candidates = None
                        geometries = []
                        for region_index, row in enumerate(selected):
                            filename = f"mask-{image_index}-{source_index}-{region_index}.png"
                            mask = row.pop("mask")
                            with Image.fromarray(mask.astype(np.uint8) * 255) as mask_image:
                                mask_image.save(directory / filename)
                            geometries.append({**row, "image_id": image["image_id"],
                                "source_id": source["source_id"], "region_id": region_index,
                                "image_size": list(frame.size), "mask_path": filename,
                                "prompt": source["prompt"]})
                        results.append({"image_id": image["image_id"], "source_id": source["source_id"],
                                        "status": "ok", "regions": geometries, "diagnostics": counts})
                    predictions = None
                state = None
            self.last_metrics = {"seconds": time.monotonic()-started, "images": len(images),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.startswith("cuda") else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.startswith("cuda") else 0}
            return results
        finally:
            # An exception traceback keeps this frame alive until the worker replies.
            # Release its GPU tensors before returning control to the WD process.
            state = predictions = candidates = None
            self.offload()

    def _candidates(self, predictions, index, size, confidence):
        torch = self.torch
        logits = predictions["pred_logits"][index].squeeze(-1)
        presence = predictions["presence_logit_dec"][index].sigmoid().reshape(())
        scores = logits.sigmoid() * presence
        width, height = size
        for candidate, score in enumerate(scores):
            value = float(score.item())
            if value <= confidence:
                yield np.empty((0, 0)), value, []
                continue
            cx, cy, w, h = predictions["pred_boxes"][index, candidate].tolist()
            bbox = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
            raw = predictions["pred_masks"][index, candidate][None, None]
            probabilities = torch.nn.functional.interpolate(
                raw.float(), size=(height, width), mode="bilinear", align_corners=False,
            ).sigmoid()[0, 0].cpu().numpy()
            yield probabilities, value, bbox
