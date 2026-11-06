"""Literal tag names shared by inference and model-free postprocessing."""
from collections.abc import Iterable
import re


def canonical_tag(tag: str) -> str:
    return str(tag).strip().replace(' ', '_')


def parse_tags(value: str | Iterable[str] | None) -> list[str]:
    values = re.split(r'[,\r\n]+', value) if isinstance(value, str) else (value or [])
    return list(dict.fromkeys(tag for item in values if (tag := canonical_tag(item))))
