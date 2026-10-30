"""Bounded WD batches and retagging saved geometry independently of SAM."""
from dataclasses import dataclass
from itertools import islice
import hashlib

import torch

from .configs import DetectorConfig
from .region_export import load_region_record
from .regions import prepare_saved_region
from .sam_config import RegionViewConfig, positive_integer
from .tag_utils import apply_tag_rules, format_result
from .types import ImageTransform, RegionResult, WDTagResult
from .wdv3_timm import LabelData, get_tags, pil_ensure_rgb, pil_pad_square


@dataclass
class RegionTagger:
    model: torch.nn.Module
    transform: ImageTransform
    labels: LabelData
    device: torch.device
    gen_threshold: float = .35
    char_threshold: float = .75

    def tag_batch(self, images, rules: DetectorConfig) -> list[WDTagResult]:
        if not images:
            return []
        tensors = []
        for image in images:
            rgb = pil_ensure_rgb(image)
            try:
                with pil_pad_square(rgb) as square:
                    tensors.append(self.transform(square)[[2, 1, 0], :, :])
            finally:
                if rgb is not image:
                    rgb.close()
        with torch.inference_mode():
            batch = torch.stack(tensors).to(self.device)
            output = self.model(batch).sigmoid().cpu()
        if output.ndim != 2 or output.shape != (len(images), len(self.labels.names)):
            raise ValueError("WD batch output shape does not match regions/labels")
        gen = rules.region_gen_threshold if rules.region_gen_threshold is not None else self.gen_threshold
        char = rules.region_char_threshold if rules.region_char_threshold is not None else self.char_threshold
        names = [self.labels.names[index] for index in self.labels.character]
        results = []
        for row in output:
            caption, taglist, ratings, character, general = get_tags(row, self.labels, gen, char)
            raw: WDTagResult = {"caption": caption, "taglist": taglist, "ratings": ratings,
                                "character": character, "general": general}
            results.append(apply_tag_rules(raw, remove=rules.remove_tags_from_region,
                                           add=rules.add_tags_to_region, character_names=names))
        return results

    def tag_regions(self, inputs, rules: DetectorConfig, batch_size: int) -> list[RegionResult]:
        """Consume owned (RGB crop, geometry, view) lazily, closing every crop."""
        positive_integer(batch_size, "region_batch_size")
        iterator = iter(inputs)
        results: list[RegionResult] = []
        while True:
            chunk = []
            try:
                # Append incrementally so an input error also closes preceding crops.
                for item in islice(iterator, batch_size):
                    chunk.append(item)
                if not chunk:
                    break
                tagged = self.tag_batch([item[0] for item in chunk], rules)
                for (_, geometry, view), raw in zip(chunk, tagged):
                    results.append(format_result({**raw, "region_id": geometry["region_id"],
                        "bbox": list(map(float, geometry["bbox"])), "geometry": geometry, "view": view}))
            finally:
                for image, _, _ in chunk:
                    image.close()
        return results

    def identity(self) -> dict:
        """Hash actual loaded weights once per run, including arbitrary custom WD."""
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return {"weights_sha256": digest.hexdigest(), "labels": {
                    "names": self.labels.names, "rating": self.labels.rating,
                    "general": self.labels.general, "character": self.labels.character},
                "transform": repr(self.transform), "torch": torch.__version__,
                "gen_threshold": self.gen_threshold, "char_threshold": self.char_threshold}


def retag_saved_region(root, image_id: str, source_name: str, region_id: int,
                       view: RegionViewConfig, tagger: RegionTagger,
                       rules: DetectorConfig | None = None) -> RegionResult:
    """Return new evidence; never change a completed export in place."""
    record = load_region_record(root, image_id)
    detail = record["detailed"]
    region = next(r for r in detail["detectors"][source_name] if r["region_id"] == region_id)
    geometry = region["geometry"]
    crop, metadata = prepare_saved_region(root, detail["canonical_path"], geometry, view)
    result = tagger.tag_regions(iter([(crop, geometry, metadata)]),
                               rules or DetectorConfig(source_name, "saved"), 1)[0]
    return result
