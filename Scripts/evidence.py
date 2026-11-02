"""Immutable, versioned, JSON-portable values shared by nodes and API clients."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
import json
from .region_export import json_text


@dataclass(frozen=True)
class Bundle(Mapping):
    kind: str
    payload: str = field(repr=False)
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.kind not in ('model', 'regions', 'evidence', 'rules', 'results', 'report', 'text'):
            raise ValueError('Unsupported WDV3 bundle kind/version')
        value = json.loads(self.payload)
        if not isinstance(value, dict):
            raise ValueError('Bundle payload must be an object')

    @classmethod
    def pack(cls, kind, value):
        return cls(kind, json_text(value))

    def data(self):
        return json.loads(self.payload)

    def __getitem__(self,key):
        return self.data()[key]

    def __iter__(self):
        return iter(self.data())

    def __len__(self):
        return len(self.data())

    @property
    def fingerprint(self):
        return hashlib.sha256((str(self.schema_version) + self.kind + self.payload).encode()).hexdigest()

    def to_json(self):
        return json.dumps(dict(schema_version=self.schema_version, kind=self.kind, payload=self.data()), ensure_ascii=False)

    @classmethod
    def from_json(cls, text):
        value = json.loads(text)
        if type(value.get('schema_version')) is not int or value.get('schema_version') != 1:
            raise ValueError('Unsupported bundle schema_version')
        return cls.pack(value['kind'], value['payload'])


def text_evidence(text, *, image_id, source_id, model_id, output_kind='text', region_id=None,
                  score=None, settings=None, model_version=None):
    if not isinstance(text, str) or not all(isinstance(x, str) and x for x in (image_id, source_id, model_id)):
        raise ValueError('Text evidence needs text, image, source and model identities')
    if output_kind not in ('text', 'class_label', 'tag_scores'):
        raise ValueError('Unsupported text output_kind')
    value = dict(text=text, image_id=image_id, source_id=source_id, model_id=model_id,
                 model_version=model_version, output_kind=output_kind, settings=settings or {})
    if region_id is not None:
        value['region_id'] = region_id
    if score is not None:
        value['score'] = score
    return Bundle.pack('text', value)
