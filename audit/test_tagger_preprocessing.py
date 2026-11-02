"""Published profiles, independent pixel oracle, and full/region inference parity."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from timm.data import resolve_data_config

from Scripts.tagger_preprocessing import build_transform, prepare_tensor, ProfileTransform
from Scripts.tagging import tag_image
from Scripts.region_tagging import RegionTagger
from Scripts.wdv3_timm import LabelData, load_labels_csv, get_tags, pil_pad_square

FIXTURES = Path(__file__).parent / 'fixtures/tagger-profiles'


def published(repo):
    folder = FIXTURES / repo
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    return build_transform(repo, folder, resolve_data_config(cfg['pretrained_cfg'], use_test_size=True))


class Capture(torch.nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.inputs = []
        self.width = width

    def forward(self, batch):
        self.inputs.append(batch.clone())
        return batch.mean((1, 2, 3)).reshape(-1, 1).expand(-1, self.width)


class PreprocessingTests(unittest.TestCase):
    def test_every_accessible_published_profile(self):
        torch.set_num_threads(2)
        count = 0
        for config in FIXTURES.glob('*/*/config.json'):
            repo = '/'.join(config.parent.parts[-2:])
            with self.subTest(repo=repo):
                transform = published(repo)
                with Image.new('RGB', (193, 127), 'red') as image:
                    tensor = transform(image)
                self.assertEqual(tuple(tensor.shape), transform.profile.input_size)
                self.assertEqual(tensor.dtype, torch.float32)
                count += 1
        # Explicitly tracked gated profiles are never represented by invented fixtures.
        access = json.loads((FIXTURES / 'access.json').read_text())
        blocked = {entry['repo_id'] for entry in access}
        self.assertEqual(count + len(blocked), 31)

    def test_control_shapes_and_architecture_name(self):
        expected = {'convnextv2_huge.dbv4-full': 512, 'caformer_b36.dbv4-full': 384,
                    'vit_base_patch16_224.dbv4-full': 448}
        for name, size in expected.items():
            self.assertEqual(published('animetimm/' + name).profile.input_size, (3, size, size))

    def test_official_pixel_pipeline(self):
        # Independent equivalent of imgutils PadToSize + the published torchvision chain.
        rng = np.random.default_rng(4)
        pixels = rng.integers(0, 256, (31, 79, 4), dtype=np.uint8)
        for name in ('caformer_b36.dbv4-full', 'convnextv2_huge.dbv4-full'):
            transform = published('animetimm/' + name)
            ops = transform.profile.operations
            for mode in ('RGB', 'RGBA'):
                with Image.fromarray(pixels, 'RGBA') as rgba:
                    image = rgba.convert('RGB') if mode == 'RGB' else rgba.copy()
                try:
                    with Image.new('RGBA', image.size, 'white') as background:
                        background.alpha_composite(image.convert('RGBA'))
                        rgb = background.convert('RGB')
                    w, h = ops[0]['size']
                    ratio = min(w / rgb.width, h / rgb.height)
                    resized = rgb.resize((round(rgb.width * ratio), round(rgb.height * ratio)), Image.Resampling.BILINEAR)
                    canvas = Image.new('RGB', (w, h), 'white')
                    canvas.paste(resized, ((w-resized.width)//2, (h-resized.height)//2))
                    reference = T.Compose([
                        T.Resize(ops[1]['size'], interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                        T.CenterCrop(ops[2]['size']), T.ToTensor(),
                        T.Normalize(ops[4]['mean'], ops[4]['std'])])(canvas)
                    torch.testing.assert_close(transform(image), reference, rtol=0, atol=0)
                    rgb.close(); resized.close(); canvas.close()
                finally:
                    image.close()

    def test_bad_profile_and_full_shape(self):
        original = published('animetimm/caformer_b36.dbv4-full')
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            folder = Path(directory)
            cfg = {'input_size': (3, 224, 320)}
            with self.assertRaisesRegex(ValueError, 'preprocess.json'):
                build_transform('animetimm/example', folder, cfg)
            ops = [{'type': 'resize', 'size': [224, 320]}, {'type': 'maybe_to_tensor'}]
            (folder/'preprocess.json').write_text(json.dumps({'test': ops}))
            transform = build_transform('animetimm/example', folder, cfg)
            self.assertEqual(transform.profile.input_size, (3, 224, 320))
            square_ops = [{'type': 'resize', 'size': [224, 224]}, {'type': 'maybe_to_tensor'}]
            (folder/'preprocess.json').write_text(json.dumps({'test': square_ops}))
            square = build_transform('animetimm/example', folder, {'input_size': (3,224,224)})
            with Image.new('RGB', (31,19)) as image:
                self.assertEqual(tuple(square(image).shape), (3,224,224))
            (folder/'preprocess.json').write_text(json.dumps({'test': [{'type': 'run_python'}]}))
            with self.assertRaisesRegex(ValueError, 'Unsupported preprocessing'):
                build_transform('animetimm/example', folder, cfg)
            (folder/'preprocess.json').write_text(json.dumps({'test': original.profile.operations}))
            with self.assertRaisesRegex(ValueError, 'input does not match'):
                build_transform('animetimm/example', folder, cfg)

    def test_full_region_parity_and_legacy_bgr(self):
        labels = LabelData(['rating', 'general_tag', 'character_tag'], [0], [1], [2])
        image = Image.new('RGB', (31, 17), 'red')
        try:
            transform = published('animetimm/caformer_b36.dbv4-full')
            model = Capture()
            full = tag_image(image, model, transform, labels, -1, -1, torch.device('cpu'))
            tagger = RegionTagger(model, transform, labels, torch.device('cpu'))
            region = tagger.score_batch([image])[0]
            self.assertEqual(full, region)
            torch.testing.assert_close(model.inputs[0], model.inputs[1], rtol=0, atol=0)
            self.assertEqual(image.size, (31, 17))
            legacy = T.Compose([T.Resize((8, 8)), T.ToTensor()])
            with pil_pad_square(image) as square:
                expected = legacy(square)[[2, 1, 0]]
            torch.testing.assert_close(prepare_tensor(image, legacy), expected, rtol=0, atol=0)
            bad = Capture(width=2)
            with self.assertRaisesRegex(ValueError, 'output shape'):
                tag_image(image, bad, transform, labels, -1, -1, torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'output shape'):
                RegionTagger(bad, transform, labels, torch.device('cpu')).score_batch([image])
            rows = tagger.score_batch([image, image, image])
            for row in rows:
                self.assertEqual(row['caption'], region['caption'])
                for category in ('general', 'character', 'ratings'):
                    for tag, score in region[category].items():
                        self.assertAlmostEqual(float(row[category][tag]), float(score), delta=1e-6)
        finally:
            image.close()

    def test_dtype_guard_and_label_order(self):
        transform = published('animetimm/caformer_b36.dbv4-full')
        bad = ProfileTransform(transform.profile, [lambda _: torch.zeros(3, 384, 384, dtype=torch.float64)])
        with Image.new('RGB', (4, 4)) as image, self.assertRaisesRegex(ValueError, 'float32'):
            bad(image)
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            csv = Path(directory)/'selected_tags.csv'
            csv.write_text('name,category,best_threshold\ncharacter_tag,4,0.9\nrating,9,0.9\ngeneral_tag,0,0.9')
            labels = load_labels_csv(csv)
            self.assertEqual(labels.character, [0])
            self.assertEqual(labels.rating, [1])
            self.assertEqual(labels.general, [2])
            self.assertIn('general_tag', get_tags(torch.tensor([.5, .5, .5]), labels, .4, .4)[4])
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                get_tags(torch.zeros(2), labels, .4, .4)
