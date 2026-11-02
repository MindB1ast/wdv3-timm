"""Published AnimeTimm operations and legacy WD preprocessing, owned in one place."""
import json
from pathlib import Path

import torch
from PIL import Image, ImageColor
from torchvision import transforms as T

from .tagger_profile import InputProfile
from .types import ImageTransform
from .wdv3_timm import pil_ensure_rgb, pil_pad_square


class PadToSize:
    """Match imgutils.data.pad_image_to_size (size is width,height)."""

    def __init__(self, size, background_color='white', interpolation='bilinear'):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2 or any(type(n) is not int or n <= 0 for n in self.size):
            raise ValueError('Invalid pad_to_size size')
        self.color = ImageColor.getrgb(background_color) if isinstance(background_color, str) else background_color
        self.interpolation = T.InterpolationMode(interpolation)

    def __call__(self, image):
        width, height = self.size
        ratio = min(width / image.width, height / image.height)
        new_size = (round(image.width * ratio), round(image.height * ratio))
        interpolation = self.interpolation.value.replace('-exact', '').upper()
        with image.resize(new_size, Image.Resampling[interpolation]) as resized:
            canvas = Image.new('RGB', self.size, tuple(self.color) if isinstance(self.color, list) else self.color)
            canvas.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
        return canvas


def compile_operations(operations):
    """Only explicitly supported declarative operations; no remote code."""
    result = []
    for operation in operations:
        args = dict(operation)
        kind = args.pop('type')
        if kind == 'pad_to_size':
            transform = PadToSize(**args)
        elif kind == 'resize':
            args['interpolation'] = T.InterpolationMode(args.get('interpolation', 'bilinear'))
            transform = T.Resize(**args)
        elif kind == 'center_crop':
            transform = T.CenterCrop(**args)
        elif kind == 'maybe_to_tensor':
            if args:
                raise ValueError('maybe_to_tensor takes no arguments')
            transform = T.ToTensor()
        elif kind == 'normalize':
            if len(args.get('mean', [])) != 3 or len(args.get('std', [])) != 3 or any(n <= 0 for n in args['std']):
                raise ValueError('normalize requires three means and positive standard deviations')
            transform = T.Normalize(**args)
        else:
            raise ValueError(f'Unsupported preprocessing operation: {kind}')
        result.append(transform)
    return result


class ProfileTransform:
    """Callable compatible with ImageTransform; profile is immutable provenance."""

    def __init__(self, profile: InputProfile, transforms, *, legacy_wd=False):
        self.profile = profile
        self.transforms = transforms
        self.legacy_wd = legacy_wd

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = pil_ensure_rgb(image)
        current = rgb
        try:
            if self.legacy_wd:
                current = pil_pad_square(rgb)
            for transform in self.transforms:
                updated = transform(current)
                if isinstance(current, Image.Image) and current is not rgb and current is not image and updated is not current:
                    current.close()
                current = updated
            if not isinstance(current, torch.Tensor):
                raise ValueError(f'{self.profile.repo_id}: preprocessing must produce a tensor')
            if self.legacy_wd:
                current = current[[2, 1, 0], :, :]
            if tuple(current.shape) != self.profile.input_size or current.dtype != torch.float32:
                raise ValueError(f'{self.profile.repo_id}: input does not match float32 {self.profile.input_size}')
            return current
        finally:
            if isinstance(current, Image.Image) and current is not rgb and current is not image:
                current.close()
            if rgb is not image:
                rgb.close()

    def __repr__(self):
        return f'ProfileTransform({self.profile.to_json()})'


def build_transform(repo_id: str, folder: Path, data_config: dict) -> ProfileTransform:
    """Build from local metadata; never download or infer shape from model names."""
    if (folder / 'preprocess.json').is_file() or repo_id.startswith('animetimm/'):
        source = folder / 'preprocess.json'
        try:
            operations = json.loads(source.read_text(encoding='utf-8'))['test']
            if not isinstance(operations, list) or not operations:
                raise ValueError('test must be a nonempty ordered list')
            profile = InputProfile(repo_id, tuple(data_config['input_size']), 'RGB',
                                   'preprocess.json:test', json.dumps(operations))
            transform = ProfileTransform(profile, compile_operations(operations))
            # Detect incompatible shapes/operation order before expensive inference.
            with Image.new('RGB', (73, 119), 'white') as probe:
                transform(probe)
            return transform
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f'{repo_id}: invalid {source.name}: {error}') from error
    if not repo_id.startswith('SmilingWolf/'):
        raise ValueError(f'{repo_id}: no supported preprocessing profile (preprocess.json required)')
    from timm.data.transforms_factory import create_transform
    required = ('input_size', 'mean', 'std', 'interpolation')
    if any(key not in data_config for key in required):
        raise ValueError(f'{repo_id}: incomplete config.json pretrained_cfg')
    transform = create_transform(**data_config)
    operations = [{'type': 'ensure_rgb', 'background_color': 'white'},
                  {'type': 'pad_square', 'background_color': 'white'},
                  {'type': 'timm_eval', 'config': data_config, 'transform': repr(transform)},
                  {'type': 'channel_order', 'order': 'BGR', 'position': 'after_normalize'}]
    profile = InputProfile(repo_id, tuple(data_config['input_size']), 'BGR',
                           'config.json:pretrained_cfg', json.dumps(operations))
    return ProfileTransform(profile, [transform], legacy_wd=True)


def prepare_tensor(image: Image.Image, transform: ImageTransform) -> torch.Tensor:
    """New transforms own the entire pipeline; raw legacy callables retain WD behavior."""
    if getattr(transform, 'profile', None) is not None:
        return transform(image)
    rgb = pil_ensure_rgb(image)
    try:
        with pil_pad_square(rgb) as square:
            tensor = transform(square)[[2, 1, 0], :, :]
        if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.dtype != torch.float32:
            raise ValueError('Legacy WD transform must produce float32 CHW with three channels')
        return tensor
    finally:
        if rgb is not image:
            rgb.close()


def transform_metadata(transform: ImageTransform) -> dict:
    profile = getattr(transform, 'profile', None)
    return profile.metadata() if profile is not None else {'legacy_wd': True, 'transform': repr(transform)}
