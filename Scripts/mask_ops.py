"""Small mask geometry shared with the isolated SAM interpreter."""
import hashlib

import numpy as np
from numpy.typing import NDArray

Mask = NDArray[np.bool_]


def mask_bbox(mask: Mask) -> list[int] | None:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Expected a two-dimensional boolean mask")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_digest(mask: Mask) -> str:
    return hashlib.sha256(str(mask.shape).encode() + mask.tobytes()).hexdigest()
