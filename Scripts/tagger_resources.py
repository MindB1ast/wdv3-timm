"""One local/Hub resource bundle for notebook and ComfyUI taggers."""
import json
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import safetensors.torch
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from .tagger_catalog import model_path, resolve_model_id
from .tagger_preprocessing import build_transform
from .wdv3_timm import load_labels_csv

RESOURCE_FILES = ('config.json', 'selected_tags.csv', 'preprocess.json',
                  'model.safetensors', 'pytorch_model.bin', 'tagger-source.json')


def _fetch(repo_id, name, target, revision=None):
    try:
        kwargs = {'revision': revision} if revision else {}
        return Path(hf_hub_download(repo_id, name, local_dir=str(target), **kwargs))
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status in (401, 403):
            raise PermissionError(f'{repo_id}: access denied to {name} (HTTP {status}). '
                                  'Log in to Hugging Face and accept the repository conditions; '
                                  'author approval may still be pending.') from None
        raise


def ensure_resources(repo_id: str, root: Path, *, allow_download=True, local_path=None) -> Path:
    repo_id = resolve_model_id(repo_id)
    target = Path(local_path) if local_path is not None else model_path(root, repo_id)
    if target.suffix in ('.safetensors', '.bin'):
        target = target.parent
    source = target / 'tagger-source.json'
    if source.is_file() and json.loads(source.read_text(encoding='utf-8'))['repo_id'] != repo_id:
        raise ValueError(f'{repo_id}: local folder belongs to a different repository: {target}')
    required = ['config.json', 'selected_tags.csv']
    if repo_id.startswith('animetimm/') or (target / 'preprocess.json').is_file() or not repo_id.startswith('SmilingWolf/'):
        required.append('preprocess.json')
    weight = next((name for name in ('model.safetensors', 'pytorch_model.bin') if (target / name).is_file()), None)
    missing = [name for name in required if not (target / name).is_file()]
    if weight is None:
        missing.append('model.safetensors')
    if missing and not allow_download:
        raise FileNotFoundError(f'{repo_id}: incomplete model folder {target}. Missing {", ".join(missing)}. Enable download_missing.')
    if missing:
        target.mkdir(parents=True, exist_ok=True)
        cached_metadata = target / '.cache/huggingface/download/config.json.metadata'
        revision = json.loads(source.read_text(encoding='utf-8')).get('revision') if source.is_file() else None
        if not revision and cached_metadata.is_file():
            revision = cached_metadata.read_text(encoding='utf-8').splitlines()[0]
        for name in missing:
            try:
                _fetch(repo_id, name, target, revision)
                if name == 'config.json' and cached_metadata.is_file():
                    revision = cached_metadata.read_text(encoding='utf-8').splitlines()[0]
            except EntryNotFoundError:
                # Only a truly missing safetensors export permits a bin fallback.
                if name != 'model.safetensors':
                    raise FileNotFoundError(f'{repo_id}: required resource missing: {name}') from None
                _fetch(repo_id, 'pytorch_model.bin', target, revision)
        source.write_text(json.dumps({'repo_id': repo_id, 'revision': revision}, indent=2), encoding='utf-8')
    return target


def load_weight_state(path: Path):
    """Avoid the Windows torch storage-slicing crash in the mmap backend.

    A native access violation cannot be caught with try/except. Select a
    non-mmap reader before opening the checkpoint, retaining older APIs.
    """
    if sys.platform != 'win32':
        return safetensors.torch.load_file(path)
    if 'backend' in inspect.signature(safetensors.torch.load_file).parameters:
        return safetensors.torch.load_file(path, device='cpu', backend='pread')
    return safetensors.torch.load(path.read_bytes())


def load_network(target: Path):
    """Build from local timm config; no implicit pretrained download."""
    network = timm.create_model(f'local-dir:{target}', pretrained=False).eval()
    if (target / 'model.safetensors').is_file():
        state = load_weight_state(target / 'model.safetensors')
    else:
        state = torch.load(target / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    network.load_state_dict(state)
    return network


def load_tagger_metadata(repo_id: str, root: Path, *, allow_download=True, local_path=None):
    repo_id = resolve_model_id(repo_id)
    target = ensure_resources(repo_id, root, allow_download=allow_download, local_path=local_path)
    # Validate metadata and dictionary before allocating a large network.
    config = json.loads((target / 'config.json').read_text(encoding='utf-8'))
    labels = load_labels_csv(target / 'selected_tags.csv')
    if config.get('num_classes') != len(labels.names):
        raise ValueError(f'{repo_id}: config num_classes does not match selected_tags.csv')
    if 'tags' in config and config['tags'] != labels.names:
        raise ValueError(f'{repo_id}: config tags do not match selected_tags.csv order')
    pretrained = config.get('pretrained_cfg', {})
    required = ['input_size']
    if repo_id.startswith('SmilingWolf/'):
        required.extend(['mean', 'std', 'interpolation'])
    if any(key not in pretrained for key in required):
        raise ValueError(f'{repo_id}: config.json lacks required pretrained_cfg fields: {required}')
    from timm.data.config import resolve_data_config
    data = resolve_data_config(pretrained, use_test_size=True)
    transform = build_transform(repo_id, target, data)
    source = target / 'tagger-source.json'
    cached_metadata = target / '.cache/huggingface/download/config.json.metadata'
    revision = json.loads(source.read_text(encoding='utf-8')).get('revision') if source.is_file() else None
    if not revision and cached_metadata.is_file():
        revision = cached_metadata.read_text(encoding='utf-8').splitlines()[0]
    transform.profile = replace(transform.profile, revision=revision)
    return labels, transform, target


def load_tagger(repo_id: str, root: Path, *, allow_download=True, local_path=None):
    labels,transform,target=load_tagger_metadata(repo_id,root,allow_download=allow_download,local_path=local_path)
    return load_network(target),labels,transform,target
