"""Exactly two intentional SAM consumer errors."""
from Scripts.sam_config import RegionViewConfig
from Scripts.types import RegionGeometry


def invalid_sam_consumer(geometry: RegionGeometry) -> None:
    geometry["bbox"] = "not coordinates"
    RegionViewConfig(mode="unknown")
