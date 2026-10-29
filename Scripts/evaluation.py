"""Validated image identity and complete-sample evaluation helpers."""

import warnings
from pathlib import Path

from .tag_utils import parse_tags


def image_id(path, root=None, stem_only=False):
    """Identify a file relative to a known root, rejecting outside paths."""
    path = Path(path)
    if root is not None:
        try:
            path = path.resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ValueError(f"Image outside dataset root: {path}") from exc
    if stem_only:
        return path.stem
    return path.with_suffix("").as_posix()


def ground_truth_tags(txt_folder, recursive=False, image_root=None):
    """Load canonical labels with relative identities and optional image validation."""
    folder = Path(txt_folder)
    paths = sorted(folder.rglob("*.txt") if recursive else folder.glob("*.txt"))
    result = {}
    for path in paths:
        relative = path.relative_to(folder)
        key = relative.with_suffix("").as_posix()
        if image_root is not None:
            base = Path(image_root) / relative.with_suffix("")
            matches = [
                base.with_suffix(ext)
                for ext in (".png", ".jpg", ".jpeg")
                if base.with_suffix(ext).exists()
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one image for label {relative}; found {len(matches)}")
        if key in result:
            raise ValueError(f"Duplicate label identity: {key}")
        result[key] = {
            tag.replace("_", " ") for tag in parse_tags(path.read_text(encoding="utf-8"))
        }
    return result


def model_tags(results, use_merged=True, image_root=None):
    """Keep legacy single-folder stems, preserving paths for multiple folders."""
    records = results[1] if use_merged else results[0]
    parents = {Path(item["image_path"]).parent.as_posix() for item in records}
    stem_only = image_root is None and len(parents) <= 1
    output = {}
    for item in records:
        key = image_id(item["image_path"], root=image_root, stem_only=stem_only)
        if key in output:
            raise ValueError(f"Duplicate prediction identity: {key}: {item['image_path']}")
        payload = item["merged_tags"] if use_merged else item["full_image"]
        output[key] = {tag.replace("_", " ") for tag in parse_tags(payload["taglist"])}
    return output


def aligned_samples(ground_truth, predictions):
    """Every labelled sample participates; unknown predicted identities are errors."""
    if not ground_truth:
        raise ValueError("Ground truth dataset is empty")
    extra = set(predictions) - set(ground_truth)
    if extra:
        raise ValueError(f"Predictions without ground truth: {sorted(extra)}")
    truth = {
        key: {tag.replace("_", " ") for tag in parse_tags(values)}
        for key, values in ground_truth.items()
    }
    predicted = {
        key: {tag.replace("_", " ") for tag in parse_tags(predictions.get(key, []))}
        for key in truth
    }
    return truth, predicted


def _sample_metrics(truth, predicted):
    """Count one sample independently of aggregate dataset diagnostics."""
    correct = len(truth & predicted)
    return {
        "precision": correct / len(predicted) if predicted else 0.0,
        "recall": correct / len(truth) if truth else 0.0,
        "TP": correct,
        "FP": len(predicted - truth),
        "FN": len(truth - predicted),
    }


def tag_metrics(ground_truth, predictions, return_diagnostics=False):
    """Evaluate every labelled image, optionally reporting processing coverage."""
    truth, predicted = aligned_samples(ground_truth, predictions)
    samples = [_sample_metrics(values, predicted[key]) for key, values in truth.items()]
    precision = sum(sample["precision"] for sample in samples) / len(truth)
    recall = sum(sample["recall"] for sample in samples) / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"Precision": precision, "Recall": recall, "F1-score": f1}
    missing = sorted(set(truth) - set(predictions))
    diagnostics = {
        "expected": len(truth),
        "processed": len(predictions),
        "coverage": len(predictions) / len(truth),
        "missing_ids": missing,
        **{name: sum(sample[name] for sample in samples) for name in ("TP", "FP", "FN")},
        "F1_definition": "harmonic mean of samples-averaged Precision and Recall",
    }
    if return_diagnostics:
        return metrics, diagnostics
    if missing:
        warnings.warn(
            f"Missing predictions counted as empty: {missing}", RuntimeWarning, stacklevel=2
        )
    return metrics
