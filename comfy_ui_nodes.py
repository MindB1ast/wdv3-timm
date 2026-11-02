"""ComfyUI V3 nodes for editable policies, SAM, folders and result inspection."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io
import folder_paths
from comfy.utils import ProgressBar

from .comfy_nodes import ROOT, model_directory
from .comfy_processing import (file_frame, scan_folder,
                               segment_frames, tag_folder, tensor_frame)
from .Scripts.tag_utils import parse_tags, parse_additions
from .Scripts.evidence import Bundle
from .Scripts.preprocess import GLOBAL_EXCLUDED_REGION_TAGS

RULES = io.Custom("WDV3_RULES")
OPTIONS = io.Custom("WDV3_OPTIONS")
DETECTORS = io.Custom("WDV3_DETECTORS")
SAM_MODEL = io.Custom("WDV3_SAM_MODEL")
SAM_SOURCES = io.Custom("WDV3_SAM_SOURCES")
SAM_REGIONS = io.Custom("WDV3_SAM_REGIONS")
FILES = io.Custom("WDV3_FILES")
MODEL = io.Custom("WDV3_MODEL")


def progress(done, total):
    import comfy.model_management as management
    management.throw_exception_if_processing_interrupted()
    ProgressBar(total).update_absolute(done, total)


class WDTaggingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3Options", display_name="WDV3 Tagging Settings", category="WDV3/Settings",
            description="Overrides tagging widgets when connected to options. Batch sizes bound WD tensors.",
            inputs=[io.Float.Input("general_threshold", default=.35, min=0, max=1, step=.01),
                    io.Float.Input("character_threshold", default=.75, min=0, max=1, step=.01),
                    io.Int.Input("image_batch_size", default=2, min=1, max=64),
                    io.Int.Input("region_batch_size", default=8, min=1, max=128),
                    io.Boolean.Input("replace_underscores", default=True),
                    io.Boolean.Input("escape_parentheses", default=True),
                    io.String.Input("remove_tags", default="", multiline=True),
                    io.String.Input("prefix_tags", default=""), io.String.Input("suffix_tags", default="")],
            outputs=[OPTIONS.Output("options")])

    @classmethod
    def execute(cls, **kwargs):
        from .comfy_config import validate_config
        validate_config({'schema_version':1,'options':kwargs})
        return io.NodeOutput(Bundle.pack('rules',kwargs))


class WDRegionRules(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3RegionRules", display_name="WDV3 Region Tag Rules", category="WDV3/Settings",
            description="Comma/newline separated tags. Removal wins. Manual additions have assigned score 1.0.",
            inputs=[io.Float.Input("general_threshold", default=.35, min=0, max=1, step=.01),
                    io.Float.Input("character_threshold", default=.75, min=0, max=1, step=.01),
                    io.String.Input("remove_from_full", default="", multiline=True),
                    io.String.Input("remove_from_regions", default="", multiline=True),
                    io.String.Input("add_to_regions", default="", multiline=True, tooltip="tag or tag:score (0..1), separated by commas/newlines"),
                    io.String.Input("require_full_confirmation", default="", multiline=True),
                    io.String.Input("only_from_source", default="", multiline=True,
                        tooltip="These tags may come only from this source and other declared owners. No full-image fallback."),
                    io.String.Input("allowed_tags", default="", multiline=True,
                        tooltip="Accept only these tags from this source. Empty means unrestricted."),
                    io.String.Input('full_image_only_tags', default=', '.join(sorted(GLOBAL_EXCLUDED_REGION_TAGS)),multiline=True,
                        tooltip='Composition tags that may come only from the full image. Clear to allow them from this source.')],
            outputs=[RULES.Output("rules")])

    @classmethod
    def execute(cls, general_threshold, character_threshold, remove_from_full,
                remove_from_regions, add_to_regions, require_full_confirmation,
                only_from_source="", allowed_tags="", full_image_only_tags=None):
        rules={"region_gen_threshold": general_threshold, "region_char_threshold": character_threshold,
            "remove_tags_from_full": parse_tags(remove_from_full),
            "remove_tags_from_region": parse_tags(remove_from_regions),
            "add_tags_to_region": parse_additions(add_to_regions),
            "exclude_from_region": parse_tags(require_full_confirmation),
            "only_from_source": parse_tags(only_from_source), "allowed_tags": parse_tags(allowed_tags),
            "full_image_only_tags": None if full_image_only_tags is None else parse_tags(full_image_only_tags)}
        from .comfy_config import validate_rules
        validate_rules(rules)
        return io.NodeOutput(Bundle.pack('rules',rules))


class WDYOLODetector(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, model, custom_model_path='', **kwargs):
        from .comfy_evidence import digest, file_stamp
        candidates = [Path(custom_model_path)] if custom_model_path.strip() else [
            directory / model for directory in (model_directory('yolo'), ROOT/'models'/'yolo',
                Path(folder_paths.models_dir)/'ultralytics'/'bbox',Path(folder_paths.models_dir)/'ultralytics'/'segm')]
        return digest([file_stamp(path) for path in candidates])

    @classmethod
    def define_schema(cls):
        known = ["person_yolov8m-seg.pt", "face_yolov9c.pt", "hand_yolov9c.pt", "yolov8n.pt"]
        for directory in (model_directory("yolo"), ROOT / "models" / "yolo",
                          Path(folder_paths.models_dir) / "ultralytics" / "bbox",
                          Path(folder_paths.models_dir) / "ultralytics" / "segm"):
            if directory.is_dir():
                known.extend(p.name for p in directory.glob("*.pt"))
        return io.Schema(node_id="WDV3YOLODetector", display_name="WDV3 YOLO Detector", category="WDV3/Regions",
            description="Configure a YOLO source. Weights are loaded on first tagging, not while saving presets.",
            inputs=[io.Combo.Input("model", options=list(dict.fromkeys(known))),
                    io.String.Input("source_name", default="person"),
                    io.Float.Input("confidence", default=.35, min=0, max=1, step=.01),
                    io.String.Input("classes", default="0", tooltip="Comma-separated class IDs; empty means all classes."),
                    io.String.Input("custom_model_path", default="", tooltip="Optional explicit weights path; overrides model."),
                    RULES.Input("rules", optional=True), DETECTORS.Input("previous_detectors", optional=True),
                    io.Combo.Input('device', options=['auto','cpu','cuda'], default='auto', optional=True),
                    io.Boolean.Input('download_missing', default=True, optional=True)],
            outputs=[DETECTORS.Output("detectors")])

    @classmethod
    def execute(cls, model, source_name, confidence, classes, custom_model_path, rules=None, previous_detectors=None,
                device='auto', download_missing=True):
        from .Scripts.configs import DetectorConfig
        choices = cls.define_schema().inputs[0].options
        if model not in choices:
            raise ValueError("Unknown YOLO selection")
        name = source_name.strip()
        if not name:
            raise ValueError("source_name cannot be empty")
        ids = [int(text.strip()) for text in classes.split(",") if text.strip()]
        if any(i < 0 for i in ids):
            raise ValueError("Class IDs must be nonnegative")
        path = custom_model_path.strip() or model
        if not custom_model_path.strip():
            for directory in (model_directory("yolo"), ROOT / "models" / "yolo",
                              Path(folder_paths.models_dir) / "ultralytics" / "bbox",
                              Path(folder_paths.models_dir) / "ultralytics" / "segm"):
                if (directory / model).is_file():
                    path = str(directory / model)
                    break
        if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable; select cpu or auto')
        configs, models = previous_detectors if previous_detectors is not None else ([], {})
        if name in {cfg.name for cfg in configs}:
            raise ValueError("Detector source names must be unique")
        config = DetectorConfig(name, path, confidence=confidence, classes=ids,
                                device=device, download_missing=download_missing, **(rules or {}))
        # The existing dataclass turns None into class 0; an empty selection explicitly means all.
        config.classes = ids or None
        models = dict(models)
        return io.NodeOutput(([*configs, config], models))


class WDSAMLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMLoader", display_name="WDV3 Load SAM 3.1", category="WDV3/SAM",
            description="Configure SAM 3.1. Missing files are downloaded when segmentation runs; weights are released afterwards.",
            inputs=[io.Combo.Input("model", options=["unsloth/sam3.1"]),
                    io.Combo.Input("device", options=["auto", "cpu", "cuda"]),
                    io.Boolean.Input("download_missing", default=True),
                    io.String.Input("local_model_path", default="", tooltip="Optional model folder or model.safetensors. Empty uses models/sam3/unsloth-sam3.1.")],
            outputs=[SAM_MODEL.Output("sam_model")])

    @classmethod
    def execute(cls, model, device, download_missing, local_model_path):
        return io.NodeOutput(sam_model_settings(model, device, download_missing, local_model_path))


def sam_model_settings(model, device, download_missing, local_model_path):
    from .Scripts.sam_runtime import resolve_device
    if model != "unsloth/sam3.1":
        raise ValueError("Unsupported SAM model")
    selected = resolve_device(device, torch)
    default = Path(folder_paths.models_dir) / "sam3" / "unsloth-sam3.1"
    if not local_model_path.strip() and (ROOT / "models" / "sam3" / "model.safetensors").is_file():
        default = ROOT / "models" / "sam3"
    return Bundle.pack('model', {"checkpoint": local_model_path.strip() or str(default), "model_id": model,
            "backend": "transformers", "device": selected, "allow_download": download_missing,
            "loader_options": dict(model=model, device=device, download_missing=download_missing,
                                   local_model_path=local_model_path)})


class WDSAMSource(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMSource", display_name="WDV3 SAM Text / Region Settings", category="WDV3/SAM",
            inputs=[io.String.Input("source_name", default="sam_person"), io.String.Input("prompt", default="person"),
                    io.Float.Input("confidence", default=.5, min=0, max=1, step=.01),
                    io.Float.Input("mask_threshold", default=.5, min=0, max=1, step=.01),
                    io.Int.Input("min_mask_area_px", default=32, min=1, max=100000000),
                    io.Int.Input("max_regions", default=32, min=1, max=256),
                    io.Combo.Input("view_mode", options=["bbox", "mask"]),
                    io.Int.Input("padding_px", default=0, min=0, max=4096),
                    RULES.Input("rules", optional=True), SAM_SOURCES.Input("previous_sources", optional=True)],
            outputs=[SAM_SOURCES.Output("sources")])

    @classmethod
    def execute(cls, source_name, prompt, confidence, mask_threshold, min_mask_area_px,
                max_regions, view_mode, padding_px, rules=None, previous_sources=None):
        sources = list(previous_sources or [])
        name = source_name.strip()
        if not name or not prompt.strip():
            raise ValueError("Source name and SAM prompt must be nonempty")
        if name in {s["name"] for s in sources}:
            raise ValueError("SAM source names must be unique")
        sources.append(dict(name=name, prompt=prompt.strip(), confidence=confidence, mask_threshold=mask_threshold,
            min_mask_area_px=min_mask_area_px, max_regions=max_regions, view_mode=view_mode,
            padding_px=padding_px, rules=dict(rules or {})))
        from .comfy_config import validate_config
        sources=validate_config({'schema_version':1,'sam_sources':sources})['sam_sources']
        return io.NodeOutput(sources)


class WDSAMSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SAMSegment", display_name="WDV3 SAM Segment Images", category="WDV3/SAM",
            inputs=[io.Image.Input("images"), SAM_MODEL.Input("sam_model"), SAM_SOURCES.Input("sources"),
                    io.Int.Input("sam_batch_size", default=1, min=1, max=16)],
            outputs=[SAM_REGIONS.Output("sam_regions"), io.Mask.Output("combined_mask")])

    @classmethod
    def execute(cls, images, sam_model, sources, sam_batch_size):
        if sam_model is None or not sources:
            raise ValueError("The selected config needs a SAM model and at least one SAM source")
        with ExitStack() as stack:
            frames = [stack.enter_context(tensor_frame(t)) for t in images]
            records = segment_frames(frames, sam_model, sources, sam_batch_size, progress)
        masks = []
        for record in records:
            mask = np.zeros((record["size"][1], record["size"][0]), dtype=np.float32)
            for rows in record["regions"].values():
                for row in rows:
                    mask[row["mask"]] = 1
            masks.append(torch.from_numpy(mask))
        return io.NodeOutput(records, torch.stack(masks))


class WDLoadFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3LoadFolder", display_name="WDV3 Image Folder", category="WDV3/Batch",
            description="Collects paths only. Images of different sizes are decoded in bounded groups by Tag Folder.",
            inputs=[io.String.Input("directory", default=folder_paths.get_input_directory()),
                    io.Boolean.Input("recursive", default=False),
                    io.Int.Input("start_index", default=0, min=0, max=10000000),
                    io.Int.Input("limit", default=0, min=0, max=10000000, tooltip="0 means all matching images")],
            outputs=[FILES.Output("files"), io.Int.Output("image_count")], not_idempotent=True)

    @classmethod
    def execute(cls, directory, recursive, start_index, limit):
        files = scan_folder(directory, recursive, start_index, limit)
        return io.NodeOutput(files, len(files["paths"]))


class WDTagFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3TagFolder", display_name="WDV3 Tag Folder", category="WDV3/Batch",
            is_output_node=True,
            inputs=[FILES.Input("files"), MODEL.Input("wd_model"), OPTIONS.Input("options"),
                    io.Boolean.Input("save_txt", default=True), io.Boolean.Input("save_json", default=True),
                    io.String.Input("output_directory", default=str(Path(folder_paths.get_output_directory()) / "wdv3")),
                    io.Boolean.Input("overwrite", default=False),
                    io.Int.Input("sam_batch_size", default=1, min=1, max=16),
                    DETECTORS.Input("detectors", optional=True), SAM_MODEL.Input("sam_model", optional=True),
                    SAM_SOURCES.Input("sources", optional=True)],
            outputs=[io.String.Output("tags", is_output_list=True), io.String.Output("results_json", is_output_list=True)])

    @classmethod
    def execute(cls, files, wd_model, options, save_txt, save_json, output_directory, overwrite,
                sam_batch_size, detectors=None, sam_model=None, sources=None):
        if sources and sam_model is None:
            raise ValueError("SAM sources need sam_model")
        tags, details = tag_folder(files, wd_model, options, detectors=detectors, sam_settings=sam_model if sources else None,
            sources=sources, sam_batch_size=sam_batch_size, output_directory=output_directory,
            save_txt=save_txt, save_json=save_json, overwrite=overwrite, progress=progress)
        return io.NodeOutput(tags, details)


class WDSaveResult(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SaveResult", display_name="WDV3 Save Result", category="WDV3/Output",
            description="Save one result as TXT and/or JSON. Connect tags too to preserve the tagger's exact text formatting. result_index selects one image from a batch (0 = first).",
            is_output_node=True, is_input_list=True, not_idempotent=True,
            inputs=[io.String.Input("results_json", force_input=True),
                    io.String.Input("output_directory", default=str(Path(folder_paths.get_output_directory()) / "wdv3")),
                    io.String.Input("filename", default="result"),
                    io.Int.Input("result_index", default=0, min=0, max=10000000),
                    io.Boolean.Input("save_txt", default=True), io.Boolean.Input("save_json", default=True),
                    io.Boolean.Input("overwrite", default=False),
                    io.String.Input("tags", force_input=True, optional=True)],
            outputs=[io.String.Output("txt_path"), io.String.Output("json_path")])

    @classmethod
    def execute(cls, results_json, output_directory, filename, result_index, save_txt, save_json, overwrite, tags=None):
        from .comfy_export import save_result
        txt_path, json_path = save_result(results_json, output_directory[0], filename[0],
            result_index=result_index[0], tags=tags, save_txt=save_txt[0], save_json=save_json[0], overwrite=overwrite[0])
        return io.NodeOutput(txt_path, json_path, ui={"text": [f"Saved: {path}" for path in (txt_path, json_path) if path]})


class WDPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3Preview", display_name="WDV3 Results Viewer", category="WDV3",
            is_output_node=True, is_input_list=True,
            inputs=[io.String.Input("results_json", force_input=True),
                    io.Int.Input("preview_limit", default=16, min=1, max=256),
                    io.Image.Input("images", optional=True), SAM_REGIONS.Input("sam_regions", optional=True)], outputs=[])

    @classmethod
    def execute(cls, results_json, preview_limit, images=None, sam_regions=None):
        summaries = []
        tensors = [tensor for batch in (images or []) for tensor in batch]
        records = [record for group in (sam_regions or []) for record in group]
        total = len(results_json)
        for index, encoded in enumerate(results_json[:preview_limit[0]]):
            detail = json.loads(encoded)
            image_index = detail.get("image_index", index)
            frame = (tensor_frame(tensors[image_index]) if image_index < len(tensors) else
                     file_frame(detail["image_path"]) if detail.get("image_path") else None)
            assets = {"image": None, "region_masks": {}}
            if frame is not None:
                from .comfy_preview import preview_assets
                try:
                    record = records[image_index] if image_index < len(records) else None
                    assets = preview_assets(frame, folder_paths.get_temp_directory(), record)
                finally:
                    frame.close()
            summaries.append({"detail": detail, **assets})
        return io.NodeOutput(ui={"wdv3_results": summaries, "wdv3_total": [total]})


class WDCombineDetectors(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3CombineDetectors", display_name="WDV3 Combine YOLO Detectors", category="WDV3/Regions",
            description="Connect independently configured YOLO sources. New sockets appear as you connect sources.",
            inputs=[io.Autogrow.Input("detectors", template=io.Autogrow.TemplatePrefix(
                DETECTORS.Input("detector"), prefix="detector", min=1, max=64))],
            outputs=[DETECTORS.Output("detectors")])

    @classmethod
    def execute(cls, detectors):
        from .comfy_config import combine_detectors
        return io.NodeOutput(combine_detectors(detectors.values()))


class WDCombineSAMSources(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3CombineSAMSources", display_name="WDV3 Combine SAM Sources", category="WDV3/SAM",
            description="One SAM model, several prompts/settings. Each source retains its own thresholds and tag rules.",
            inputs=[io.Autogrow.Input("sources", template=io.Autogrow.TemplatePrefix(
                SAM_SOURCES.Input("source"), prefix="source", min=1, max=64))],
            outputs=[SAM_SOURCES.Output("sources")])

    @classmethod
    def execute(cls, sources):
        from .comfy_config import combine_sources
        return io.NodeOutput(combine_sources(sources.values()))


def preset_directory():
    return Path(folder_paths.get_user_directory()) / "default" / "wdv3" / "configs"


def preset_files():
    paths = {}
    for label, directory in (("Example", ROOT / "configs"), ("Saved", preset_directory())):
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                paths[f"{label}: {path.stem}"] = path
    return paths


class WDSaveConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3SaveConfig", display_name="WDV3 Save Pipeline Config", category="WDV3/Settings",
            description="Save connected settings/sources as a portable JSON preset. No image inference or model downloads are needed.",
            is_output_node=True,
            inputs=[io.String.Input("preset_name", default="my_pipeline"), io.Boolean.Input("overwrite", default=False),
                    OPTIONS.Input("options", optional=True), DETECTORS.Input("detectors", optional=True),
                    SAM_SOURCES.Input("sources", optional=True), SAM_MODEL.Input("sam_model", optional=True)],
            outputs=[io.String.Output("config_path")])

    @classmethod
    def execute(cls, preset_name, overwrite, options=None, detectors=None, sources=None, sam_model=None):
        from .comfy_config import config_data, save_config
        if options is None and detectors is None and sources is None and sam_model is None:
            raise ValueError("Connect settings, YOLO detectors or SAM sources to save")
        path = save_config(preset_directory(), preset_name, config_data(options, detectors, sources, sam_model), overwrite)
        return io.NodeOutput(str(path), ui={"text": [f"Saved pipeline config: {path}"]})


class WDLoadConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="WDV3LoadConfig", display_name="WDV3 Load Pipeline Config", category="WDV3/Settings",
            description="Load tagging settings, detectors, SAM sources and SAM model settings from one config. No separate SAM Loader is needed.",
            inputs=[io.Combo.Input("preset", options=list(preset_files()) or ["No saved configs"])],
            outputs=[OPTIONS.Output("options"), DETECTORS.Output("detectors"), SAM_SOURCES.Output("sources"),
                     SAM_MODEL.Output("sam_model")])

    @classmethod
    def fingerprint_inputs(cls, preset):
        import hashlib
        path = preset_files().get(preset)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path else "missing"

    @classmethod
    def execute(cls, preset):
        from .comfy_config import config_values
        path = preset_files().get(preset)
        if path is None:
            raise ValueError("Unknown preset. Refresh ComfyUI to update the preset dropdown.")
        options, detectors, sources, model = config_values(json.loads(path.read_text(encoding="utf-8")))
        return io.NodeOutput(Bundle.pack('rules',options), detectors, sources, sam_model_settings(**model) if model else None)


NODE_CLASSES = [WDTaggingOptions, WDRegionRules, WDYOLODetector, WDSAMLoader, WDSAMSource,
                WDSAMSegment, WDLoadFolder, WDTagFolder, WDPreview, WDCombineDetectors, WDCombineSAMSources,
                WDSaveConfig, WDLoadConfig, WDSaveResult]

from .comfy_pipeline_nodes import NODE_CLASSES as PIPELINE_CLASSES
NODE_CLASSES += PIPELINE_CLASSES
