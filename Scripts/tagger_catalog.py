"""Offline model identifiers and collision-free resource locations."""
import json
from pathlib import Path

from huggingface_hub.utils import validate_repo_id

WD_ALIASES = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "big": "SmilingWolf/wd-eva02-large-tagger-v3",
    "bigvit": "SmilingWolf/wd-vit-large-tagger-v3",
}


def resolve_model_id(value: str) -> str:
    """Resolve legacy aliases or validate a fully qualified Hub identifier."""
    repo_id = WD_ALIASES.get(value, value)
    validate_repo_id(repo_id)
    if repo_id.count("/") != 1:
        raise ValueError(f"Unknown model: {value}. Use a WD alias or owner/model repo ID")
    return repo_id


def model_path(root: Path, repo_id: str) -> Path:
    """Read known legacy WD folders; namespace every other repository."""
    repo_id = resolve_model_id(repo_id)
    legacy = root / repo_id.split("/")[1]
    if repo_id in WD_ALIASES.values() and legacy.is_dir():
        return legacy
    return root.joinpath(*repo_id.split("/"))


def model_choices() -> list[str]:
    """Ranked Macro F1@0.40 first; unranked models follow by publication date."""
    manifest = Path(__file__).with_name("tagger_models.json")
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    eligible = [entry for entry in entries if entry['status'] == 'metadata-compatible']
    if not eligible:
        return list(WD_ALIASES)
    ranked = sorted((entry for entry in eligible if entry.get('leaderboard') is not None),
                    key=lambda entry: (-entry['leaderboard']['macro_f1'], entry['repo_id']))
    unranked = sorted((entry for entry in eligible if entry.get('leaderboard') is None),
                      key=lambda entry: (entry.get('published_at') or '', entry['repo_id']), reverse=True)
    aliases = {repo: alias for alias, repo in WD_ALIASES.items()}
    return [aliases.get(str(entry['repo_id']), str(entry['repo_id'])) for entry in ranked + unranked]
