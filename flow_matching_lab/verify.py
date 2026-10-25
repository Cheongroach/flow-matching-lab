"""Analytical correctness checks; optional parity with Meta's installed package."""

import argparse
import json
from pathlib import Path
import unittest

import torch

from .core import guided_velocity, integrate, linear_path


class CorrectnessChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        torch.set_num_threads(2)

    def test_path_endpoints_and_derivative(self):
        x0, x1 = torch.randn(8, 1, 4, 4), torch.randn(8, 1, 4, 4)
        torch.testing.assert_close(linear_path(x0, x1, torch.zeros(8))[0], x0)
        torch.testing.assert_close(linear_path(x0, x1, torch.ones(8))[0], x1)
        t = torch.rand(8, dtype=torch.float64)
        x0, x1 = x0.double(), x1.double()
        left, velocity = linear_path(x0, x1, t - 1e-5)
        right, _ = linear_path(x0, x1, t + 1e-5)
        torch.testing.assert_close((right - left) / 2e-5, velocity)

    def test_gaussian_marginal_oracle(self):
        # Independent X0~N(0,I), X1~N(mu,sigma^2 I) have an analytic marginal VF.
        # Check full ODE against its closed-form affine solution for fixed inputs.
        mean, sigma = 2.0, 0.7
        class Oracle:
            def __call__(self, x, t, labels):
                t = t[:, None]
                variance = (1 - t).square() + sigma ** 2 * t.square()
                coefficient = (-(1 - t) + sigma ** 2 * t) / variance
                return mean + coefficient * (x - t * mean)
        x0 = torch.tensor([[-1.5], [0.0], [0.7]], dtype=torch.float64)
        labels = torch.zeros(3, dtype=torch.long)
        target = mean + sigma * x0
        euler = integrate(Oracle(), x0, labels, steps=128, method="euler")
        heun = integrate(Oracle(), x0, labels, steps=64, method="heun")
        euler_error = (euler.samples - target).abs().max().item()
        heun_error = (heun.samples - target).abs().max().item()
        self.assertLess(heun_error, 0.002)
        self.assertLess(heun_error, euler_error)

    def test_noise_to_data_direction_and_nfe(self):
        class Constant:
            def __call__(self, x, t, labels):
                return torch.ones_like(x) * 3
        x0 = torch.zeros(4, 2)
        labels = torch.zeros(4, dtype=torch.long)
        for method, expected_nfe in (("euler", 10), ("heun", 20)):
            solved = integrate(Constant(), x0, labels, steps=10, method=method)
            torch.testing.assert_close(solved.samples, torch.ones_like(x0) * 3)
            torch.testing.assert_close(x0, torch.zeros_like(x0))
            self.assertEqual(solved.nfe, expected_nfe)

    def test_cfg_definition(self):
        class Conditional:
            def __call__(self, x, t, labels):
                return torch.where(labels[:, None] == 10, 1.0, 3.0).expand_as(x)
        x, t, labels = torch.zeros(4, 2), torch.zeros(4), torch.zeros(4, dtype=torch.long)
        for scale, expected in ((0.0, 1.0), (1.0, 3.0), (2.0, 5.0)):
            torch.testing.assert_close(guided_velocity(Conditional(), x, t, labels, scale),
                                       torch.full_like(x, expected))
        solved = integrate(Conditional(), x, labels, steps=8, guidance=2)
        self.assertEqual(solved.network_evaluations_per_sample, 16)

    def test_invalid_solver_options(self):
        for steps, method in ((0, "euler"), (2, "invalid")):
            with self.assertRaises(ValueError):
                integrate(None, torch.zeros(4, 2), torch.zeros(4).long(), steps, method)


def official_parity():
    from flow_matching.path import CondOTProbPath
    rng = torch.Generator().manual_seed(5678)
    checks = []
    for shape in ((16, 2), (16, 196), (16, 1, 14, 14)):
        x0, x1 = torch.randn(shape, generator=rng), torch.randn(shape, generator=rng)
        t = torch.rand(shape[0], generator=rng)
        native_xt, native_ut = linear_path(x0, x1, t)
        reference = CondOTProbPath().sample(x_0=x0, x_1=x1, t=t)
        torch.testing.assert_close(native_xt, reference.x_t, rtol=0, atol=0)
        torch.testing.assert_close(native_ut, reference.dx_t, rtol=0, atol=0)
        checks.append({"shape": list(shape), "path_max_error": 0, "velocity_max_error": 0})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CorrectnessChecks)
    )
    summary = {"analytical_checks": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors)}
    if args.official:
        summary["official_flow_matching_1_0_10_parity"] = official_parity()
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
