"""Regressions for the observation contract and evaluation metrics."""

import unittest

import torch

from .image_eval import frechet_distance, polynomial_mmd
from .image_model import ImageUNet, observation_masks, sample_image


class ImageChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        torch.set_num_threads(2)

    def test_observed_pixels_exactly_preserved_in_both_solvers(self):
        class Oracle:
            def __call__(self, x, t, labels, known, mask):
                return torch.ones_like(x)
        noise = torch.randn(2, 1, 8, 8)
        known = torch.rand_like(noise) * 2 - 1
        mask = torch.zeros_like(noise)
        mask[:, :, :, :4] = 1
        for solver in ("euler", "heun"):
            result = sample_image(Oracle(), noise, torch.zeros(2).long(), steps=8,
                                  known=known, mask=mask, method=solver)
            torch.testing.assert_close(result.samples * mask, known * mask, atol=0, rtol=0)
            torch.testing.assert_close(result.samples * (1 - mask), (noise + 1) * (1 - mask))

    def test_masked_out_condition_is_invisible(self):
        model = ImageUNet(4).eval()
        x = torch.randn(2, 1, 28, 28)
        labels, t = torch.tensor([2, 8]), torch.tensor([0.2, 0.7])
        with torch.inference_mode():
            a = model(x, t, labels)
            b = model(x, t, labels, torch.randn_like(x) * 100, torch.zeros_like(x))
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_observation_changes_unknown_region(self):
        model = ImageUNet(4).eval()
        noise = torch.randn(1, 1, 28, 28)
        mask = torch.zeros_like(noise)
        mask[:, :, :, :14] = 1
        a = sample_image(model, noise, torch.tensor([3]), 4, known=-torch.ones_like(noise), mask=mask)
        b = sample_image(model, noise, torch.tensor([3]), 4, known=torch.ones_like(noise), mask=mask)
        self.assertGreater(float(((a.samples - b.samples).abs() * (1 - mask)).sum()), 0)

    def test_masks_include_generation_and_completion_without_empty_unknown_loss(self):
        masks = observation_masks(100, 28, torch.Generator().manual_seed(101))
        self.assertTrue(torch.all((masks == 0) | (masks == 1)))
        per_image = masks.flatten(1).sum(1)
        self.assertTrue(torch.any(per_image == 0))
        self.assertTrue(torch.any(per_image > 0))
        self.assertTrue(torch.all(per_image < 784))

    def test_nonbinary_masks_rejected(self):
        noise = torch.zeros(1, 1, 8, 8)
        with self.assertRaises(ValueError):
            sample_image(None, noise, torch.tensor([0]), mask=torch.full_like(noise, 0.5))

    def test_frechet_and_mmd_on_known_distributions(self):
        x = torch.randn(64, 8)
        self.assertLess(frechet_distance(x, x), 1e-8)
        self.assertAlmostEqual(frechet_distance(x, x + 1), 8, places=5)
        self.assertAlmostEqual(polynomial_mmd(torch.ones(64, 8), torch.ones(64, 8)), 0)
        self.assertGreater(polynomial_mmd(torch.ones(64, 8), torch.ones(64, 8) * 2), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
