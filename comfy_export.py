"""Save one selected tagging result without a frontend or model execution."""
import json
from pathlib import Path


def save_result(results_json, directory, filename, *, result_index=0, tags=None,
                save_txt=True, save_json=True, overwrite=False):
    if not save_txt and not save_json:
        raise ValueError("Enable save_txt or save_json")
    if not 0 <= result_index < len(results_json):
        raise ValueError(f"result_index {result_index} is outside the {len(results_json)} available results")
    filename = filename.strip()
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]}
    if (not filename or len(filename) > 120 or not filename[0].isalnum()
            or any(not (c.isalnum() or c in " _-") for c in filename)
            or filename.upper() in reserved):
        raise ValueError("Filename: letters, digits, spaces, _ and - only; no path or extension")
    if not str(directory).strip():
        raise ValueError("Output directory must not be empty")
    detail = json.loads(results_json[result_index])
    if not isinstance(detail, dict) or not isinstance(detail.get("merged_tags"), dict):
        raise ValueError("Connect results_json from Tag Image / Regions or Tag Folder")
    if tags is not None and len(tags) != len(results_json):
        raise ValueError("tags and results_json must contain the same number of results")
    text = tags[result_index] if tags is not None else detail["merged_tags"].get(
        "taglist", detail["merged_tags"].get("caption", ""))
    payloads = {}
    root = Path(directory).expanduser().resolve()
    if save_txt:
        payloads["txt"] = text
    if save_json:
        payloads["json"] = json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False)
    paths = {kind: root / f"{filename}.{kind}" for kind in payloads}
    # Check all destinations before creating either file.
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Change filename or enable overwrite.")
    root.mkdir(parents=True, exist_ok=True)
    for kind, path in paths.items():
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(payloads[kind])
    return str(paths.get("txt", "")), str(paths.get("json", ""))
