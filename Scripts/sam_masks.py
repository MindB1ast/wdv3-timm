"""SAM candidate selection, usable without loading the parent package."""
import numpy as np

if __package__:
    from .mask_ops import mask_bbox, mask_digest
else:
    from mask_ops import mask_bbox, mask_digest


def select_masks(candidates, source, size):
    """Candidates stream (probability map, score, model bbox); retain a bounded top-k."""
    counts = {key: 0 for key in ("total", "confidence", "empty", "area", "duplicate", "limited")}
    selected = []
    for probabilities, score, model_bbox in candidates:
        counts["total"] += 1
        score = float(score)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("SAM returned an invalid score")
        if score <= source["confidence"]:
            counts["confidence"] += 1
            continue
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (size[1], size[0]) or not np.isfinite(probabilities).all():
            raise ValueError("SAM returned invalid mask probabilities/size")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Mask must contain probabilities, not logits")
        mask = probabilities > source["mask_threshold"]
        bbox = mask_bbox(mask)
        if bbox is None:
            counts["empty"] += 1
            continue
        if int(mask.sum()) < source["min_mask_area_px"]:
            counts["area"] += 1
            continue
        digest = mask_digest(mask)
        if any(row["mask_sha256"] == digest for row in selected):
            counts["duplicate"] += 1
            existing = next(row for row in selected if row["mask_sha256"] == digest)
            if existing["segmentation_score"] < score:
                existing["segmentation_score"] = score
                existing["model_bbox"] = list(map(float, model_bbox))
        else:
            selected.append({"mask": mask, "bbox": bbox, "mask_sha256": digest,
                             "segmentation_score": score, "model_bbox": list(map(float, model_bbox))})
        selected.sort(key=lambda row: (-row["segmentation_score"], row["bbox"], row["mask_sha256"]))
        if len(selected) > source["max_regions"]:
            selected.pop()
            counts["limited"] += 1
    counts["selected"] = len(selected)
    return selected, counts
