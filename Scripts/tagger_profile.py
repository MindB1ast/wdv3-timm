"""Serializable model input contracts, independent of weights and Hub access."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class InputProfile:
    """A complete, ordered preprocessing contract represented by canonical JSON.

    Dimensions are C,H,W; pad operation sizes retain their publisher's semantics.
    Operations are stored as JSON to prevent mutation after fingerprinting.
    """

    repo_id: str
    input_size: tuple[int, int, int]
    color_order: str
    source: str
    operations_json: str
    dtype: str = "float32"
    layout: str = "CHW"
    batch_layout: str = "NCHW"
    pixel_range: tuple[float, float] = (0.0, 1.0)
    schema_version: int = 1
    revision: str | None = None

    def __post_init__(self):
        if len(self.input_size) != 3 or any(type(n) is not int or n <= 0 for n in self.input_size):
            raise ValueError("input_size must contain three positive integers C,H,W")
        if self.color_order not in ("RGB", "BGR") or self.input_size[0] != 3:
            raise ValueError("Only three-channel RGB/BGR input profiles are supported")
        if self.dtype != "float32" or self.layout != "CHW" or self.batch_layout != "NCHW":
            raise ValueError("Unsupported tensor dtype or layout")
        if (len(self.pixel_range) != 2 or not all(math.isfinite(n) for n in self.pixel_range)
                or self.pixel_range[0] >= self.pixel_range[1]):
            raise ValueError("pixel_range must contain increasing finite endpoints")
        if self.schema_version != 1 or not self.repo_id or not self.source:
            raise ValueError("Missing profile identity/source or unsupported schema version")
        operations = json.loads(self.operations_json)
        if not isinstance(operations, list) or not operations or any(
                not isinstance(op, dict) or not isinstance(op.get("type"), str) for op in operations):
            raise ValueError("operations must be a nonempty ordered list of typed objects")
        canonical = json.dumps(operations, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "operations_json", canonical)

    @property
    def operations(self) -> list[dict]:
        """Return a fresh copy so callers cannot change this profile."""
        return json.loads(self.operations_json)

    def to_json(self) -> str:
        """Canonical form includes every effective field, retaining operation order."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "InputProfile":
        data = json.loads(value)
        data["input_size"] = tuple(data["input_size"])
        data["pixel_range"] = tuple(data["pixel_range"])
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def metadata(self) -> dict:
        """Portable provenance; includes normalization and every ordered operation."""
        data = json.loads(self.to_json())
        data["operations"] = data.pop("operations_json")
        data["operations"] = json.loads(data["operations"])
        return {**data, "fingerprint": self.fingerprint}
