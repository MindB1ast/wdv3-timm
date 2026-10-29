"""Validated image identity and complete-sample evaluation helpers."""

import warnings
from pathlib import Path
from typing import Literal, overload

from .tag_utils import parse_tags
from .types import (BatchResult, MetricDiagnostics, Metrics, PathInput,
                    SampleMetrics, TagSamples, TagSets)

METRIC_EPSILON = 1e-8
METRIC_POLICY_VERSION = "complete-labels-epsilon-v1"
F1_DEFINITION = (
    "harmonic mean of samples-averaged Precision and Recall; "
    "P=mean(TP/(TP+FP+epsilon)), R=mean(TP/(TP+FN+epsilon)), "
    "F1=2*P*R/(P+R+epsilon)"
)



def image_id(
    path: PathInput, root: PathInput | None = None, stem_only: bool = False,
) -> str:
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


def ground_truth_tags(
    txt_folder: PathInput, recursive: bool = False, image_root: PathInput | None = None,
) -> TagSets:
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


def model_tags(
    results: BatchResult, use_merged: bool = True, image_root: PathInput | None = None,
) -> TagSets:
    """Keep legacy single-folder stems, preserving paths for multiple folders."""
    records = results[1] if use_merged else results[0]
    parents = {Path(item["image_path"]).parent.as_posix() for item in records}
    stem_only = image_root is None and len(parents) <= 1
    output = {}
    payloads = (
        ((item["image_path"], item["merged_tags"]) for item in results[1])
        if use_merged else
        ((item["image_path"], item["full_image"]) for item in results[0])
    )
    for path, payload in payloads:
        key = image_id(path, root=image_root, stem_only=stem_only)
        if key in output:
            raise ValueError(f"Duplicate prediction identity: {key}: {path}")
        output[key] = {tag.replace("_", " ") for tag in parse_tags(payload["taglist"])}
    return output


def aligned_samples(
    ground_truth: TagSamples, predictions: TagSamples,
) -> tuple[TagSets, TagSets]:
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


def harmonic_f1(precision: float, recall: float) -> float:
    """Stabilize the denominator without changing zero numerators."""
    return 2 * precision * recall / (precision + recall + METRIC_EPSILON)


def metrics_from_counts(tp: int, fp: int, fn: int) -> SampleMetrics:
    """One policy for summaries, image rows and accumulated tag evidence."""
    precision = tp / (tp + fp + METRIC_EPSILON)
    recall = tp / (tp + fn + METRIC_EPSILON)
    return {
        "precision": precision, "recall": recall,
        "f1_score": harmonic_f1(precision, recall), "TP": tp, "FP": fp, "FN": fn,
    }


def _sample_metrics(truth: set[str], predicted: set[str]) -> SampleMetrics:
    """Count one sample independently of aggregate dataset diagnostics."""
    return metrics_from_counts(
        len(truth & predicted), len(predicted - truth), len(truth - predicted),
    )


@overload
def tag_metrics(
    ground_truth: TagSamples, predictions: TagSamples, return_diagnostics: Literal[False] = False,
) -> Metrics: ...


@overload
def tag_metrics(
    ground_truth: TagSamples, predictions: TagSamples, return_diagnostics: Literal[True],
) -> tuple[Metrics, MetricDiagnostics]: ...


@overload
def tag_metrics(
    ground_truth: TagSamples, predictions: TagSamples, return_diagnostics: bool,
) -> Metrics | tuple[Metrics, MetricDiagnostics]: ...


def tag_metrics(
    ground_truth: TagSamples, predictions: TagSamples, return_diagnostics: bool = False,
) -> Metrics | tuple[Metrics, MetricDiagnostics]:
    """Evaluate every labelled image, optionally reporting processing coverage."""
    truth, predicted = aligned_samples(ground_truth, predictions)
    samples = [_sample_metrics(values, predicted[key]) for key, values in truth.items()]
    precision = sum(sample["precision"] for sample in samples) / len(truth)
    recall = sum(sample["recall"] for sample in samples) / len(truth)
    f1 = harmonic_f1(precision, recall)
    metrics: Metrics = {"Precision": precision, "Recall": recall, "F1-score": f1}
    missing = sorted(set(truth) - set(predictions))
    diagnostics: MetricDiagnostics = {
        "expected": len(truth),
        "processed": len(predictions),
        "coverage": len(predictions) / len(truth),
        "missing_ids": missing,
        "TP": sum(sample["TP"] for sample in samples),
        "FP": sum(sample["FP"] for sample in samples),
        "FN": sum(sample["FN"] for sample in samples),
        "F1_definition": F1_DEFINITION,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "epsilon": METRIC_EPSILON,
    }
    if return_diagnostics:
        return metrics, diagnostics
    if missing:
        warnings.warn(
            f"Missing predictions counted as empty: {missing}", RuntimeWarning, stacklevel=2
        )
    return metrics
