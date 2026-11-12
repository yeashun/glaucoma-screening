"""Synthetic-only regression checks: no restricted data, weights, network or GPU."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'competition/deployment/code'
sys.path.insert(0, str(DEPLOY))
from preprocessing import preprocess
from model import valid_bbox, fit_age, encode_age
from utils import fuse, compare_probabilities


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


cpu = module('portfolio_metrics', ROOT / 'competition/v3_experiment/cpu_utils.py')
ranking = module('portfolio_ranking', ROOT / 'competition/experiments/E_AUC01/auc_pairwise_loss.py')


class PublicPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preprocessing_matches_original_pixel_path(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as temporary:
            for height, width in [(19, 31), (121, 81), (161, 403), (511, 209), (384, 384)]:
                with self.subTest(size=(height, width)):
                    pixels = rng.integers(15, 220, (height, width, 3), dtype=np.uint8)
                    pixels[:2] = 0
                    path = Path(temporary) / 'synthetic.png'
                    Image.fromarray(pixels).save(path)
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert('RGB')
                    image = ImageOps.pad(image.crop(valid_bbox(image)), (384, 384),
                                         method=Image.Resampling.BILINEAR, color=(0, 0, 0))
                    expected = TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225])
                    self.assertTrue(torch.equal(preprocess(path), expected))

    def test_grayscale_and_rgba_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ['L', 'RGBA']:
                path = Path(temporary) / 'synthetic.png'
                Image.new(mode, (79, 57), 100).save(path)
                value = preprocess(path)
                self.assertEqual(tuple(value.shape), (3, 384, 384))
                self.assertTrue(torch.isfinite(value).all())

    def test_metric_perfect_order_and_ties(self):
        self.assertEqual(cpu.metrics([0, 1, 0, 1], [.1, .9, .2, .8])['auroc'], 1)
        self.assertEqual(cpu.metrics([0, 1], [.5, .5])['auroc'], .5)

    def test_ap_is_not_trapezoidal_pr_auc(self):
        value = cpu.metrics([0, 1], [.5, .5])
        self.assertEqual(value['ap'], .5)
        self.assertEqual(value['pr_auc_trapezoid'], .75)

    def test_single_class_auc_is_undefined(self):
        self.assertIsNone(cpu.metrics([1, 1], [.3, .4])['auroc'])

    def test_ranking_loss_direction_gradient_and_extremes(self):
        result = ranking.check_loss()
        self.assertTrue(result['direction_ok'])
        self.assertTrue(result['gradient_sign_ok'])
        self.assertTrue(result['extreme_logits_finite'])
        self.assertTrue(result['missing_class_rejected'])

    def test_fusion_uses_frozen_parameters(self):
        stack = {'logit_mean': [0., 0.], 'logit_std': [1., 1.], 'coefficients': [0., .5, .5]}
        self.assertAlmostEqual(float(fuse([.5, .5], stack)), .5)
        self.assertGreater(float(fuse([.8, .8], stack)), float(fuse([.2, .2], stack)))
        self.assertTrue(np.isfinite(fuse([0., 1.], stack)))

    def test_tiny_probability_delta_can_change_ranking(self):
        reference = {'synthetic_a': {'Probability': '.100000', 'Pred_Label': '0'},
                     'synthetic_b': {'Probability': '.100001', 'Pred_Label': '0'}}
        result = compare_probabilities(['synthetic_a', 'synthetic_b'], [.100001, .100000], reference)
        self.assertEqual(result['label_mismatches'], 0)
        self.assertEqual(result['ranking_mismatched_images'], 2)
        self.assertFalse(result['passed'])

    def test_breaking_a_tie_is_detected(self):
        reference = {'synthetic_a': {'Probability': '.2', 'Pred_Label': '0'},
                     'synthetic_b': {'Probability': '.2', 'Pred_Label': '0'}}
        self.assertFalse(compare_probabilities(['synthetic_a', 'synthetic_b'], [.2, .2000001], reference)['passed'])

    def test_age_statistics_only_fit_supplied_training_rows(self):
        stats = fit_age([{'Age': '20'}, {'Age': '40'}, {'Age': ''}], valid_only=True)
        self.assertEqual(stats['mean'], 30)
        self.assertEqual(float(encode_age({'Age': ''}, stats)[1]), 1)
        self.assertEqual(float(encode_age({'Age': '50'}, stats)[1]), 0)

    def test_public_backend_has_an_explicit_standard_pillow_fallback(self):
        backend = module('portfolio_backend', DEPLOY / 'image_backend.py')
        info = backend.activate()
        self.assertFalse(info['enabled'])


if __name__ == '__main__':
    unittest.main()
