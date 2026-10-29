# pylint: disable=invalid-name  # Preserve the legacy public import path.
"""Safe TXT export and consistent global tag additions."""

from pathlib import Path

from .io_utils import atomic_write_text
from .tag_utils import apply_tag_rules, parse_tags


def save_tags_to_txt(
    image_path, merged_tags, append_tags=True, add_tags_before="", add_tags_after=""
):
    """Export tags. Read/write errors propagate without replacing existing bytes."""
    txt_path = Path(image_path).with_suffix(".txt")
    existing = (
        parse_tags(txt_path.read_text(encoding="utf-8"))
        if (append_tags and txt_path.exists())
        else []
    )
    before, after = parse_tags(add_tags_before), parse_tags(add_tags_after)
    candidates = before + existing + parse_tags(merged_tags["taglist"]) + after
    # Preserve existing order in append mode; before/after position new tags.
    if existing:
        candidates = (
            [tag for tag in before if tag not in existing]
            + existing
            + [tag for tag in parse_tags(merged_tags["taglist"]) if tag not in existing]
            + after
        )
    final_tags = parse_tags(candidates)
    atomic_write_text(txt_path, ", ".join(tag.replace("_", " ") for tag in final_tags))
    print(f"Tags saved: {txt_path}")
    return str(txt_path)


def add_tag_to_merged(tags_for_image, add_tags_before="", add_tags_after=""):
    """Update merged dictionaries and strings together, retaining the public API."""
    if len(tags_for_image) != 2:
        raise ValueError("Expected detailed and merged result lists")
    for image_data in tags_for_image[1]:
        image_data["merged_tags"] = apply_tag_rules(
            image_data["merged_tags"], before=add_tags_before, after=add_tags_after
        )
    return tags_for_image
