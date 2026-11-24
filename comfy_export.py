"""Save one selected tagging result without a frontend or model execution."""
import json
from pathlib import Path


def _result_destination(directory, filename):
    filename = filename.strip()
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]}
    if (not filename or len(filename) > 120 or not filename[0].isalnum()
            or any(not (c.isalnum() or c in " _-") for c in filename)
            or filename.upper() in reserved):
        raise ValueError("Filename: letters, digits, spaces, _ and - only; no path or extension")
    if not str(directory).strip():
        raise ValueError("Output directory must not be empty")
    return Path(directory).expanduser().resolve(), filename


def save_result(results_json, directory, filename, *, result_index=0, tags=None,
                save_txt=True, save_json=True, overwrite=False):
    if not save_txt and not save_json:
        raise ValueError("Enable save_txt or save_json")
    needs_json = save_json or (save_txt and tags is None)
    if needs_json and not results_json:
        raise ValueError("Connect results_json to save JSON or TXT without tags")
    selected = results_json if needs_json else tags
    if selected is None or not 0 <= result_index < len(selected):
        raise ValueError(f"result_index {result_index} is outside the available results")
    if save_txt and save_json and tags is not None and len(tags) != len(results_json):
        raise ValueError("tags and results_json must contain the same number of results")
    root, filename = _result_destination(directory, filename)
    detail = None
    if needs_json:
        detail = json.loads(results_json[result_index])
        if not isinstance(detail, dict) or not isinstance(detail.get("merged_tags"), dict):
            raise ValueError("Connect results_json from Tag Image / Regions or Tag Folder")
    text = None
    if save_txt:
        text = tags[result_index] if tags is not None else detail["merged_tags"].get(
            "taglist", detail["merged_tags"].get("caption", ""))
        if tags is None and 'options' in detail.get('settings', {}):
            from .comfy_evidence import formatted_text
            text = formatted_text(detail)
    payloads = {}
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



def _export_tags(record, categories, include_scores, include_ratings, replace_underscores):
    tags = {}
    for category in categories:
        values = record.get(category, {})
        if replace_underscores:
            values = {name.replace("_", " "): score for name, score in values.items()}
        tags[category] = values if include_scores else list(values)
    if include_ratings and 'ratings' in record:
        tags['ratings'] = record['ratings']
    return tags


def compact_region_result(detail, *, include_full=True, include_scores=False,
                          include_ratings=False, include_caption=False,
                          include_aggregate=False, include_history=False,
                          include_regions=True, include_general=True, include_character=True,
                          depth_mode="all", depth_level=0):
    if not isinstance(detail, dict) or not isinstance(detail.get('full_image'), dict) or not isinstance(detail.get('detectors'), dict):
        raise ValueError('Connect results_json containing full_image and detectors')
    if depth_mode not in ('all', 'exact', 'up_to'):
        raise ValueError('depth_mode: choose all, exact or up_to')
    if type(depth_level) is not int or depth_level < 0:
        raise ValueError('depth_level must be a nonnegative integer')

    def includes_depth(depth):
        if depth_mode == 'all':
            return True
        if depth_mode == 'exact':
            return depth == depth_level
        return depth <= depth_level

    include_full = include_full and includes_depth(0)
    result = {key: detail[key] for key in ('image_id', 'image_path', 'image_index', 'image_size') if key in detail}
    categories = [name for name, enabled in (('general', include_general), ('character', include_character)) if enabled]
    replace_underscores = detail.get('settings', {}).get('options', {}).get('replace_underscores', True)
    if include_full:
        result['full'] = {'depth': 0, **_export_tags(
            detail['full_image'], categories, include_scores, include_ratings, replace_underscores)}
    if include_caption:
        from .comfy_evidence import formatted_text
        merged = dict(detail.get('merged_tags', detail['full_image']))
        for category in ('general', 'character'):
            if category not in categories:
                merged.pop(category, None)
        result['caption'] = formatted_text(detail, merged)
    if include_regions:
        result['regions'] = []
    for source, regions in (detail['detectors'].items() if include_regions else ()):
        for region in regions:
            depth = region.get('depth', 1)
            if not includes_depth(depth):
                continue
            entry = {'source': source, 'region_id': region['region_id'], 'bbox': region['bbox'], 'depth': depth}
            for key in ('region_uid', 'parent_region_uid', 'stage_id'):
                if key in region:
                    entry[key] = region[key]
            entry.update(_export_tags(region, categories, include_scores, include_ratings, replace_underscores))
            if include_aggregate and isinstance(region.get('aggregate'), dict):
                entry['aggregate'] = _export_tags(region['aggregate'], categories, include_scores, include_ratings, replace_underscores)
            if include_history:
                for key in ('decisions', 'postprocessing_history', 'route_history'):
                    if key in region:
                        entry[key] = region[key]
            result['regions'].append(entry)
    if include_history:
        if 'decisions' in detail:
            result['decisions'] = detail['decisions']
        if include_full:
            for key in ('decisions', 'postprocessing_history'):
                if key in detail['full_image']:
                    result['full'][key] = detail['full_image'][key]
    return result


def save_region_results(results_json, directory, filename, *, result_index=0, write_mode='error', **options):
    if write_mode not in ('error', 'skip', 'overwrite'):
        raise ValueError('write_mode: choose error, skip or overwrite')
    if not results_json or not -1 <= result_index < len(results_json):
        raise ValueError(f'result_index {result_index} is outside the available results')
    root, name = _result_destination(directory, filename)
    target = root / f'{name}.regions.json'
    selected = results_json if result_index == -1 else [results_json[result_index]]
    images = [compact_region_result(json.loads(encoded), **options) for encoded in selected]
    payload = json.dumps(dict(schema_version=1, images=images), ensure_ascii=False,
                         indent=2, allow_nan=False)
    root.mkdir(parents=True, exist_ok=True)
    if write_mode == 'overwrite':
        from .Scripts.io_utils import atomic_write_text
        atomic_write_text(target, payload)
    else:
        try:
            with target.open('x', encoding='utf-8') as stream:
                stream.write(payload)
        except FileExistsError:
            if write_mode == 'skip' and target.is_file():
                return str(target), 'skipped'
            raise
    return str(target), 'written'
