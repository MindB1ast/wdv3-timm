"""Consistent regional tag filtering and JSON results."""

import copy
import json
from dataclasses import dataclass

import numpy as np

from .tag_utils import format_result, normalized_scores, parse_tags

GLOBAL_EXCLUDED_REGION_TAGS = {
    "1boy",
    "1girl",
    "1other",
    "2boys",
    "2girls",
    "2others",
    "3boys",
    "3girls",
    "3others",
    "4boys",
    "4girls",
    "4others",
    "5boys",
    "5girls",
    "5others",
    "6+boys",
    "6+girls",
    "6+others",
    "abstract_background",
    "animal_focus",
    "aqua_background",
    "argyle_background",
    "armpit_focus",
    "ass_focus",
    "back_focus",
    "black_background",
    "black_border",
    "blue_background",
    "blue_sky",
    "blurry",
    "blurry_background",
    "blurry_foreground",
    "border",
    "breast_focus",
    "bright_background",
    "brown_background",
    "checkered_background",
    "close-up",
    "colorful_background",
    "cowboy_shot",
    "creature_focus",
    "cross_background",
    "crotch_focus",
    "dark_background",
    "depth_of_field",
    "diagonal-striped_background",
    "dithered_background",
    "dotted_background",
    "drama_layer",
    "ear_focus",
    "eye_focus",
    "feet_out_of_frame",
    "film_grain",
    "flower_focus",
    "food_focus",
    "foot_focus",
    "foot_out_of_frame",
    "footwear_focus",
    "full_body",
    "gradient_background",
    "green_background",
    "grey_background",
    "grid_background",
    "hair_focus",
    "halftone_background",
    "hand_focus",
    "heaven_condition",
    "hip_focus",
    "honeycomb_background",
    "kanji_focus",
    "lace_background",
    "leg_focus",
    "looking_ahead",
    "looking_at_another",
    "looking_at_viewer",
    "looking_away",
    "looking_back",
    "looking_to_the_side",
    "male_focus",
    "marble_background",
    "mecha_focus",
    "meme",
    "monochrome_background",
    "monster_focus",
    "mosaic_background",
    "mouth_focus",
    "multiple_boys",
    "multiple_girls",
    "multiple_others",
    "multiple_views",
    "navel_focus",
    "neck_focus",
    "night",
    "night_sky",
    "object_focus",
    "oekaki",
    "orange_background",
    "other_focus",
    "out_of_frame",
    "parody",
    "patterned_background",
    "pectoral_focus",
    "penis_focus",
    "pink_background",
    "plaid_background",
    "plant_focus",
    "pokemon_focus",
    "polka_dot_background",
    "portrait",
    "pov",
    "pov_hands",
    "purple_background",
    "pussy_focus",
    "rainbow_background",
    "red_background",
    "reflection_focus",
    "simple_background",
    "sky",
    "sky_focus",
    "solo",
    "solo_focus",
    "spiral_background",
    "splatter_background",
    "striped_background",
    "sunburst_background",
    "text_focus",
    "thigh_focus",
    "triangle_background",
    "two-tone_background",
    "upper_body",
    "vehicle_focus",
    "weapon_focus",
    "white_background",
    "white_border",
}


def process_tags(tags_dict, tags_to_remove):
    """Remove tags using the same canonical names as configuration inputs."""
    blocked = set(parse_tags(tags_to_remove))
    return {tag: score for tag, score in normalized_scores(tags_dict).items() if tag not in blocked}


@dataclass(frozen=True)
class _MergePolicy:
    """Restrictions and provenance for one source of merged evidence."""

    removed: object = ()
    confirmation: object = ()
    full_tags: object = ()
    provenance: dict | None = None


def _merge_source(output, category, scores, source, policy=_MergePolicy()):
    """Merge one category using independent source policy."""
    blocked = set(parse_tags(policy.removed))
    conditional = set(parse_tags(policy.confirmation))
    for tag, score in normalized_scores(scores).items():
        if tag in blocked or (tag in conditional and tag not in policy.full_tags):
            continue
        if tag not in output[category] or score > output[category][tag]:
            output[category][tag] = score
            output["provenance"][tag] = copy.deepcopy(
                (policy.provenance or {}).get(tag, {"source": source, "model_score": score})
            )


def merge_tags_from_regions(image_result, detectors):
    """Union active exclusions; merge scores without mutating source evidence."""
    configs = {detector.name: detector for detector in detectors}
    full = image_result["full_image"]
    full_tags = set(normalized_scores(full["general"])) | set(normalized_scores(full["character"]))
    excluded = set()
    for name, regions in image_result["detectors"].items():
        if regions and name in configs:
            excluded.update(parse_tags(configs[name].remove_tags_from_full))
    output = {
        "general": {},
        "character": {},
        "ratings": copy.deepcopy(full["ratings"]),
        "provenance": {},
    }
    for category in ("general", "character"):
        _merge_source(
            output,
            category,
            full[category],
            "full",
            _MergePolicy(removed=excluded, provenance=full.get("provenance")),
        )
    for name, regions in sorted(image_result["detectors"].items()):
        if name not in configs:
            continue
        detector = configs[name]
        for region in regions:
            for category in ("general", "character"):
                blocked = set(parse_tags(detector.remove_tags_from_region))
                if category == "general":
                    blocked.update(GLOBAL_EXCLUDED_REGION_TAGS)
                _merge_source(
                    output,
                    category,
                    region[category],
                    f"{name}:{region.get('region_id', 0)}",
                    _MergePolicy(
                        removed=blocked,
                        confirmation=detector.exclude_from_region,
                        full_tags=full_tags,
                        provenance=region.get("provenance"),
                    ),
                )
    return format_result(output)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for the NumPy float32 scores returned by tagging."""

    def default(self, o):
        """Serialize NumPy float32 scores; delegate other values to JSON."""
        if isinstance(o, np.float32):
            return float(o)
        return json.JSONEncoder.default(self, o)


# Сохранение в Json результатов
def save_results_to_json(results, output_file):
    """Write detailed or merged results using the NumPy-aware JSON encoder."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Результаты сохранены в {output_file}")
