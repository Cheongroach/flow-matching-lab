"""An independent implementation of linear conditional Flow Matching.

Convention: t=0 is Gaussian noise, t=1 is data. No ODE in the training loss.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


def linear_path(x0, x1, t):
    if x0.shape != x1.shape or t.shape != (x0.shape[0],):
        raise ValueError("Expected equal endpoint shapes and t shaped [batch].")
    time = t.reshape(-1, *([1] * (x0.ndim - 1)))
    return (1 - time) * x0 + time * x1, x1 - x0


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return (x + self.layers(x)) / math.sqrt(2)


class VelocityMLP(nn.Module):
    """Pixel-space velocity with sinusoidal time and class embeddings."""

    def __init__(self, dim=196, width=256, depth=3):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 32)))
        self.label_embedding = nn.Embedding(11, 64)  # class 10 is the null condition
        self.input = nn.Linear(dim + 65 + 64, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(depth)))
        self.output = nn.Linear(width, dim)
        # Start near the zero-velocity baseline, without suppressing hidden gradients.
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, labels):
        phase = t[:, None] * self.frequencies[None] * (2 * math.pi)
        time = torch.cat((t[:, None], phase.sin(), phase.cos()), dim=1)
        hidden = torch.nn.functional.silu(
            self.input(torch.cat((x, time, self.label_embedding(labels)), dim=1))
        )
        return self.output(self.blocks(hidden))


def guided_velocity(model, x, t, labels, guidance=1.0):
    """CFG uses v_null + guidance * (v_cond - v_null); guidance=1 is conditional."""
    if guidance == 1:
        return model(x, t, labels)
    null = torch.full_like(labels, 10)
    unconditional, conditional = model(
        torch.cat((x, x)), torch.cat((t, t)), torch.cat((null, labels))
    ).chunk(2)
    return unconditional + guidance * (conditional - unconditional)


@dataclass
class Solution:
    samples: torch.Tensor
    trajectory: torch.Tensor
    nfe: int
    network_evaluations_per_sample: int


@torch.inference_mode()
def integrate(model, x0, labels, steps=32, method="euler", guidance=1.0, snapshots=8):
    """Fixed-step Euler or explicit trapezoidal Heun, without clipping the state."""
    if steps < 1 or method not in {"euler", "heun"}:
        raise ValueError("Positive steps and method euler/heun are required.")
    if guidance < 0 or not math.isfinite(guidance):
        raise ValueError("Guidance must be finite and non-negative.")
    x = x0.clone()
    dt = 1.0 / steps
    states = [x.clone()]
    record = {round(i * steps / snapshots) for i in range(1, snapshots + 1)}
    for index in range(steps):
        t = x.new_full((len(x),), index / steps)
        velocity = guided_velocity(model, x, t, labels, guidance)
        if method == "euler":
            x = x + dt * velocity
        else:
            proposal = x + dt * velocity
            endpoint_velocity = guided_velocity(model, proposal, t + dt, labels, guidance)
            x = x + dt / 2 * (velocity + endpoint_velocity)
        if index + 1 in record:
            states.append(x.clone())
    if not torch.isfinite(x).all():
        raise FloatingPointError("ODE sampling produced non-finite values.")
    nfe = steps * (1 if method == "euler" else 2)
    return Solution(x, torch.stack(states), nfe, nfe * (1 if guidance == 1 else 2))


class DigitClassifier(nn.Module):
    """Independent evaluator, trained exclusively on real MNIST training images."""

    def __init__(self, dim=196):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)
