"""Canonical Danbooru names and consistent text representations."""

import copy
import re
from collections.abc import Mapping
from typing import TypeVar

from .types import CATEGORIES, Score, Scores, TagInput, WDTagResult

_Result = TypeVar("_Result", bound=WDTagResult)


def canonical_tag(tag: str) -> str:
    """Preserve punctuation/case, replacing spaces with underscores."""
    return str(tag).strip().replace(" ", "_")


def parse_tags(value: TagInput) -> list[str]:
    """Parse comma/newline separated input, deduplicating in order."""
    values = re.split(r"[,\r\n]+", value) if isinstance(value, str) else (value or [])
    return list(dict.fromkeys(tag for item in values if (tag := canonical_tag(item))))


def normalized_scores(scores: Mapping[str, Score]) -> Scores:
    """Merge equivalent names without losing the maximum score."""
    result: Scores = {}
    for name, score in scores.items():
        tag = canonical_tag(name)
        if tag:
            result[tag] = max(result.get(tag, score), score)
    return result


def format_result(result: _Result, order: TagInput = None) -> _Result:
    """Rebuild both strings from the authoritative score dictionaries."""
    names = set(result["general"]) | set(result["character"])
    ordered = [tag for tag in parse_tags(order or []) if tag in names]
    ordered.extend(sorted(names - set(ordered)))
    result["caption"] = ", ".join(ordered)
    result["taglist"] = ", ".join(tag.replace("_", " ") for tag in ordered)
    return result


# Preserve the existing notebook calling convention.
# pylint: disable-next=R0913,R0917
def apply_tag_rules(
    result: _Result, remove: TagInput = (), add: Mapping[str, Score] | None = None,
    before: TagInput = (), after: TagInput = (), character_names: TagInput = (),
) -> _Result:
    """Return a new result; explicit removal wins over manual additions."""
    output = copy.deepcopy(result)
    blocked = set(parse_tags(remove))
    for category in CATEGORIES:
        output[category] = {
            tag: score
            for tag, score in normalized_scores(output[category]).items()
            if tag not in blocked
        }
    provenance = output.setdefault("provenance", {})
    for tag in blocked:
        provenance.pop(tag, None)
    additions = {canonical_tag(tag): score for tag, score in (add or {}).items()}
    for tag in parse_tags(before) + parse_tags(after):
        additions.setdefault(tag, 1.0)
    known_characters = set(parse_tags(character_names))
    _add_manual_tags(output, additions, blocked, known_characters)
    suffix = [tag for tag in parse_tags(after) if tag not in parse_tags(before)]
    middle = sorted(
        (set(output["general"]) | set(output["character"])) - set(parse_tags(before)) - set(suffix)
    )
    return format_result(output, parse_tags(before) + middle + suffix)


def _add_manual_tags(
    output: WDTagResult, additions: Mapping[str, Score],
    blocked: set[str], known_characters: set[str],
) -> None:
    """Assign only new, unblocked manual tags to their declared category."""
    provenance = output.setdefault("provenance", {})
    for tag, score in additions.items():
        if not tag or tag in blocked or tag in output["general"] or (tag in output["character"]):
            continue
        category = "character" if tag in known_characters else "general"
        output[category][tag] = score
        provenance[tag] = {"source": "manual_rule", "assigned_score": score}
