"""Static consumers, also safe to import without model loading."""
from typing import Literal, assert_type
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from Scripts.configs import DetectorConfig, ScriptOptions
from Scripts.sam_config import RegionViewConfig, SamSourceConfig
from Scripts.main import BatchTagging
# Internal resource acceptance deliberately inspects the existing private container.
from Scripts.main import _BatchResources  # pyright: ignore[reportPrivateUsage]
from Scripts.tag_utils import apply_tag_rules, format_result
from Scripts.types import BatchResult, RegionResult, Score, WDTagResult
from Scripts.wdv3_timm import LabelData

sample: WDTagResult = {
    "caption": "blue_eyes", "taglist": "blue eyes", "general": {"blue_eyes": np.float32(.8)},
    "character": {}, "ratings": {},
}
region: RegionResult = {**sample, "region_id": 0, "bbox": [0., 0., 10., 10.]}
assert_type(format_result(region), RegionResult)
assert_type(apply_tag_rules(region), RegionResult)
assert_type(sample["general"]["blue_eyes"], Score)
if "provenance" in sample:
    assert_type(sample["provenance"]["blue_eyes"]["source"], str)


def resource_consumer(resources: _BatchResources, image: Image.Image) -> None:
    assert_type(resources.model, nn.Module)
    assert_type(resources.labels, LabelData)
    assert_type(resources.labels.names, list[str])
    assert_type(resources.transform(image), torch.Tensor)
    assert_type(resources.device, torch.device | None)


def result_consumer(result: BatchResult) -> None:
    region_result = result[0][0]["detectors"]["face"][0]
    assert_type(region_result["bbox"], list[float])
    assert_type(region_result["region_id"], int)
    assert_type(result[1][0]["merged_tags"]["caption"], str)
    if "txt_path" in result[1][0]:
        assert_type(result[1][0]["txt_path"], str)


def notebook_consumer() -> None:
    options = ScriptOptions(ImageFolder=Path("images"), output_file=None)
    assert_type(BatchTagging(options), BatchResult | None)
    DetectorConfig("face", "face.pt", classes=None, remove_tags_from_full=None)


def sam_consumer(region_result: RegionResult) -> None:
    source = SamSourceConfig("eyes", "sam_eyes", "eye", view=RegionViewConfig(mode="mask"))
    assert_type(source.prompt, str)
    assert_type(source.view.mode, Literal["mask", "bbox"])
    if "geometry" in region_result:
        assert_type(region_result["geometry"]["bbox"], list[int])
    if "view" in region_result:
        assert_type(region_result["view"]["fingerprint"], str)
