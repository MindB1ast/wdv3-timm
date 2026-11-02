"""ComfyUI adapter: in-memory images, reusable WD weights, optional YOLO crops."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch

from .Scripts.configs import DetectorConfig
from .Scripts.tagger_catalog import model_choices, model_path, resolve_model_id

ROOT = Path(__file__).resolve().parent


def model_directory(kind):
    """Use ComfyUI's shared models directory, including --base-directory."""
    import folder_paths
    return Path(folder_paths.models_dir) / "wdv3" / kind


@dataclass
class WDRuntime:
    model: object = field(repr=False)
    transform: object = field(repr=False)
    labels: object = field(repr=False)
    device: object
    model_id: str = "custom-wd"
    metadata: object = field(default=None, repr=False)


class WDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (model_choices(), {"default": "vit", "tooltip": "Ranklist Macro F1@0.40 descending; unranked models follow by publication date."}),
            "device": (["auto", "cpu", "cuda"],),
        }, "optional": {
            "download_missing": ("BOOLEAN", {"default": True}),
            "local_model_path": ("STRING", {"default": ""}),
            "repo_id": ("STRING", {"default": "", "tooltip": "Optional Hugging Face owner/model; overrides model selection"}),
        }}

    RETURN_TYPES = ("WDV3_MODEL", "STRING")
    RETURN_NAMES = ("wd_model", "model_info")
    FUNCTION = "load"
    CATEGORY = "WDV3"

    @classmethod
    def IS_CHANGED(cls, model, device, download_missing=True, local_model_path="", repo_id=""):
        from .comfy_evidence import digest, file_stamp
        from .Scripts.tagger_resources import RESOURCE_FILES
        selected_id = resolve_model_id(repo_id.strip() or model)
        candidates = [Path(local_model_path)] if local_model_path.strip() else [
            model_path(ROOT / 'models' / 'taggers', selected_id),
            model_path(model_directory('taggers'), selected_id)]
        candidates = [p.parent if p.suffix in ('.safetensors', '.bin') else p for p in candidates]
        return digest([selected_id, device, download_missing,
                       [[file_stamp(p / name) for name in RESOURCE_FILES] for p in candidates]])

    def load(self, model, device, download_missing=True, local_model_path="", repo_id=""):
        from .Scripts.tagger_resources import load_tagger
        from .Scripts.tagger_preprocessing import transform_metadata
        selected_id = resolve_model_id(repo_id.strip() or model)
        if device == "auto":
            import comfy.model_management as management
            selected = management.get_torch_device()
        else:
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; select cpu or auto")
            selected = torch.device(device)
        local = ROOT / "models" / "taggers"
        local_target = model_path(local, selected_id)
        directory = local if any((local_target / name).is_file() for name in
                                 ('model.safetensors', 'pytorch_model.bin')) else model_directory('taggers')
        network, labels, transform, target = load_tagger(
            selected_id, directory, allow_download=download_missing,
            local_path=Path(local_model_path).expanduser().resolve() if local_model_path.strip() else None)
        from .Scripts.evidence import Bundle
        profile = transform_metadata(transform)
        metadata = Bundle.pack('model', dict(model_id=selected_id, directory=str(target), device=str(selected),
                    download_missing=download_missing, labels=len(labels.names), profile=profile))
        print(f"Tagger {selected_id}: {profile['input_size']} {profile['color_order']} ({profile['source']})")
        return (WDRuntime(network.cpu().eval(), transform, labels, selected, selected_id, metadata),
                json.dumps(profile, ensure_ascii=False, indent=2))


class WDDetectorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        example = [{"name": "person", "model_path": "person_yolov8m-seg.pt",
                    "confidence": 0.35, "classes": [0],
                    "region_gen_threshold": 0.35, "region_char_threshold": 0.75}]
        return {"required": {"detectors_json": ("STRING", {
            "default": json.dumps(example, indent=2), "multiline": True})}}

    RETURN_TYPES = ("WDV3_DETECTORS",)
    RETURN_NAMES = ("detectors",)
    FUNCTION = "load"
    CATEGORY = "WDV3"

    def load(self, detectors_json):
        from .Scripts.model import load_yolo_model

        entries = json.loads(detectors_json)
        if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
            raise ValueError("detectors_json must be a JSON array of detector objects")
        configs = [DetectorConfig(**entry) for entry in entries]
        if len({cfg.name for cfg in configs}) != len(configs):
            raise ValueError("Detector names must be unique")
        directory = model_directory("yolo")
        directory.mkdir(parents=True, exist_ok=True)
        models = {}
        for config in configs:
            if config.model_path not in models:
                local = ROOT / "models" / "yolo" / config.model_path
                path = str(local) if local.is_file() else config.model_path
                models[config.model_path] = load_yolo_model(path, directory, allow_download=config.download_missing)
                models[config.model_path].to("cpu")
        return ((configs, models),)


def tag_regions(image, tagger, configs, models, batch_size):
    """Consume bounded groups of crops; never write intermediate images."""
    output = {}
    for config in configs:
        if config.model_path not in models:
            from .Scripts.model import load_yolo_model
            directory = model_directory("yolo")
            directory.mkdir(parents=True, exist_ok=True)
            models[config.model_path] = load_yolo_model(config.model_path, directory, allow_download=config.download_missing).to("cpu")
        from .comfy_evidence import interrupt
        interrupt()
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable for YOLO; select cpu or auto')
        device = config.device
        if device == 'auto':
            try:
                import comfy.model_management as management
                device = str(management.get_torch_device())
            except ImportError:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = models[config.model_path](image, conf=config.confidence,
                                            classes=config.classes, device=device, verbose=False)
        objects = [box for result in results
                 if getattr(result, "boxes", None) is not None for box in result.boxes]
        import math
        boxes = []
        for box in objects:
            coordinates = box.xyxy[0].tolist()
            if len(coordinates) != 4 or not all(math.isfinite(c) for c in coordinates):
                raise ValueError('Detector returned an invalid bbox')
            x1,y1,x2,y2=coordinates
            bounded=[max(0,min(image.width,x1)),max(0,min(image.height,y1)),
                     max(0,min(image.width,x2)),max(0,min(image.height,y2))]
            if bounded[2]<=bounded[0] or bounded[3]<=bounded[1]:
                raise ValueError('Detector returned an empty bbox')
            boxes.append(bounded)
        regions = []
        for start in range(0, len(boxes), batch_size):
            crops = []
            try:
                for bbox in boxes[start:start + batch_size]:
                    crops.append(image.crop(tuple(bbox)))
                tagged = tagger.tag_batch(crops, config)
                for offset, raw in enumerate(tagged):
                    regions.append({"region_id": start + offset,
                                    "bbox": boxes[start + offset],
                                    "class_id": int(objects[start+offset].cls[0]) if getattr(objects[start+offset], 'cls', None) is not None else None,
                                    "confidence": float(objects[start+offset].conf[0]) if getattr(objects[start+offset], 'conf', None) is not None else None, **raw})
            finally:
                for crop in crops:
                    crop.close()
        output[config.name] = regions
        if hasattr(models[config.model_path], 'to'): models[config.model_path].to('cpu')
    return output


class WDTagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "wd_model": ("WDV3_MODEL",),
            "general_threshold": ("FLOAT", {"default": 0.35, "min": 0., "max": 1., "step": 0.01}),
            "character_threshold": ("FLOAT", {"default": 0.75, "min": 0., "max": 1., "step": 0.01}),
            "region_batch_size": ("INT", {"default": 8, "min": 1, "max": 128}),
            "replace_underscores": ("BOOLEAN", {"default": True}),
            "escape_parentheses": ("BOOLEAN", {"default": True}),
            "remove_tags": ("STRING", {"default": "", "multiline": True}),
            "prefix_tags": ("STRING", {"default": ""}),
            "suffix_tags": ("STRING", {"default": ""}),
        }, "optional": {"detectors": ("WDV3_DETECTORS",),
                          "sam_regions": ("WDV3_SAM_REGIONS",),
                          "options": ("WDV3_OPTIONS",),
                          "sam_model": ("WDV3_SAM_MODEL",),
                          "sources": ("WDV3_SAM_SOURCES",),
                          "sam_batch_size": ("INT", {"default": 1, "min": 1, "max": 16})}}

    RETURN_TYPES = ("STRING", "STRING", "WDV3_SAM_REGIONS")
    RETURN_NAMES = ("tags", "results_json", "sam_regions")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_NODE = True
    FUNCTION = "tag"
    CATEGORY = "WDV3"

    def tag(self, images, wd_model, general_threshold=0.35, character_threshold=0.75,
            region_batch_size=8, replace_underscores=True, escape_parentheses=True,
            remove_tags="", prefix_tags="", suffix_tags="", detectors=None,
            sam_regions=None, options=None, sam_model=None, sources=None, sam_batch_size=1):
        from contextlib import ExitStack
        from .comfy_processing import tensor_frame, tag_frames, json_result
        from .comfy_evidence import cached_segment, interrupt
        if (images.ndim != 4 or images.shape[-1] not in (3, 4)
                or any(size < 1 for size in images.shape)):
            raise ValueError("Expected a nonempty IMAGE tensor [batch, height, width, 3 or 4]")
        if sam_regions is not None and (sam_model is not None or sources is not None):
            raise ValueError("Use either precomputed sam_regions or sam_model + sources")
        if sources and sam_model is None:
            raise ValueError("SAM sources need sam_model")
        if sam_model is not None and sources is None:
            raise ValueError("SAM model needs sources")
        settings = dict(general_threshold=general_threshold, character_threshold=character_threshold,
                        region_batch_size=region_batch_size, replace_underscores=replace_underscores,
                        escape_parentheses=escape_parentheses, remove_tags=remove_tags,
                        prefix_tags=prefix_tags, suffix_tags=suffix_tags)
        settings.update(options or {})
        from .comfy_config import unique_names
        unique_names([c.name for c in (detectors[0] if detectors else [])]+[s['name'] for s in (sources or [])])
        with ExitStack() as stack:
            interrupt()
            frames = [stack.enter_context(tensor_frame(tensor)) for tensor in images]
            if sources:
                sam_regions = cached_segment(frames, wd_model, sam_model, sources, sam_batch_size)
            tags, records = tag_frames(frames, wd_model, detectors=detectors, sam_regions=sam_regions,
                                       options=settings)
        return {"ui": {"text": tags}, "result": (tags, [json_result(row) for row in records], sam_regions or [])}


NODE_CLASS_MAPPINGS = {
    "WDV3ModelLoader": WDModelLoader,
    "WDV3DetectorLoader": WDDetectorLoader,
    "WDV3Tagger": WDTagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WDV3ModelLoader": "WDV3 Load Tagger Model",
    "WDV3DetectorLoader": "WDV3 Load YOLO Detectors",
    "WDV3Tagger": "WDV3 Tag Image / Regions",
}
