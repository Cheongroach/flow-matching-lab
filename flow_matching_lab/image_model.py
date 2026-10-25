"""28x28 image-conditioned velocity models, independent of upstream libraries."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import ResidualBlock, Solution


class Conditions(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(100), 32)))
        self.labels = nn.Embedding(11, width)
        self.time = nn.Sequential(nn.Linear(65, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t, labels):
        phase = t[:, None] * self.frequencies * (2 * math.pi)
        return self.time(torch.cat((t[:, None], phase.sin(), phase.cos()), 1)) + self.labels(labels)


class ImageBlock(nn.Module):
    def __init__(self, incoming, outgoing, condition_width=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.condition = nn.Linear(condition_width, outgoing)
        self.norm2 = nn.GroupNorm(4, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition(F.silu(condition))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2)


class ImageUNet(nn.Module):
    """Inputs: current noisy image, observed pixels, observation mask, time, label."""

    def __init__(self, channels=16):
        super().__init__()
        self.conditions = Conditions()
        self.input = nn.Conv2d(3, channels, 3, padding=1)
        self.enc1 = ImageBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.enc2 = ImageBlock(channels * 2, channels * 2)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1)
        self.middle = ImageBlock(channels * 4, channels * 4)
        self.up2 = ImageBlock(channels * 6, channels * 2)
        self.up1 = ImageBlock(channels * 3, channels)
        self.output = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, labels, known=None, mask=None):
        known = torch.zeros_like(x) if known is None else known
        mask = torch.zeros_like(x) if mask is None else mask
        condition = self.conditions(t, labels)
        a = self.enc1(self.input(torch.cat((x, known * mask, mask), 1)), condition)
        b = self.enc2(self.down1(a), condition)
        h = self.middle(self.down2(b), condition)
        h = self.up2(torch.cat((F.interpolate(h, size=b.shape[-2:], mode="nearest"), b), 1), condition)
        h = self.up1(torch.cat((F.interpolate(h, size=a.shape[-2:], mode="nearest"), a), 1), condition)
        return self.output(F.silu(h))


class ImageMLP(nn.Module):
    """Control: same conditions and 28x28 data, with fully connected spatial layers."""

    def __init__(self, width=256):
        super().__init__()
        self.conditions = Conditions()
        self.input = nn.Linear(3 * 28 * 28 + 128, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(3)))
        self.output = nn.Linear(width, 28 * 28)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, labels, known=None, mask=None):
        known = torch.zeros_like(x) if known is None else known
        mask = torch.zeros_like(x) if mask is None else mask
        features = torch.cat((x.flatten(1), (known * mask).flatten(1), mask.flatten(1),
                              self.conditions(t, labels)), 1)
        return self.output(self.blocks(F.silu(self.input(features)))).reshape_as(x)


def make_image_model(config):
    if config["architecture"] == "unet":
        return ImageUNet(config.get("channels", 16))
    if config["architecture"] == "mlp":
        return ImageMLP(config.get("width", 256))
    raise ValueError("Unknown architecture")


def observation_masks(batch, side, generator, device="cpu", completion_probability=0.5):
    """Half generation; completion masks observe backgrounds as well as strokes."""
    mask = torch.zeros(batch, 1, side, side)
    kinds = torch.rand(batch, generator=generator)
    coordinates = torch.randint(side // 4, 3 * side // 4, (batch, 4), generator=generator)
    for i in range(batch):
        if kinds[i] >= completion_probability:
            continue
        kind = i % 5
        cut = int(coordinates[i, 0])
        if kind == 0:
            mask[i, :, :, :cut] = 1
        elif kind == 1:
            mask[i, :, :, cut:] = 1
        elif kind == 2:
            mask[i, :, :cut, :] = 1
        elif kind == 3:
            mask[i, :, cut:, :] = 1
        else:
            left, right = sorted(coordinates[i, :2].tolist())
            top, bottom = sorted(coordinates[i, 2:].tolist())
            mask[i] = 1
            mask[i, :, top:bottom + 1, left:right + 1] = 0
    return mask.to(device)


@torch.inference_mode()
def sample_image(model, noise, labels, steps=32, guidance=1, known=None, mask=None,
                 method="euler", snapshots=8):
    if steps < 1 or method not in {"euler", "heun"} or snapshots < 1:
        raise ValueError("Invalid ODE configuration")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Invalid guidance")
    known = torch.zeros_like(noise) if known is None else known
    mask = torch.zeros_like(noise) if mask is None else mask
    if known.shape != noise.shape or mask.shape != noise.shape:
        raise ValueError("Observation and noise shapes must match")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Observation mask must be binary")
    if not torch.isfinite(known).all() or not torch.isfinite(noise).all():
        raise ValueError("Image inputs must be finite")

    def project(x, time):
        return (1 - mask) * x + mask * ((1 - time) * noise + time * known)

    def velocity(x, time):
        t = x.new_full((len(x),), time)
        if guidance == 1:
            return model(x, t, labels, known, mask)
        null = torch.full_like(labels, 10)
        unconditional, conditional = model(
            torch.cat((x, x)), torch.cat((t, t)), torch.cat((null, labels)),
            torch.cat((known, known)), torch.cat((mask, mask)),
        ).chunk(2)
        # CFG here guides labels; both branches retain the same observed image.
        return unconditional + guidance * (conditional - unconditional)

    x = noise.clone()
    states = [x.clone()]
    record = {round(i * steps / snapshots) for i in range(1, snapshots + 1)}
    dt = 1 / steps
    for i in range(steps):
        v = velocity(x, i / steps)
        proposal = project(x + dt * v, (i + 1) / steps)
        if method == "heun":
            v2 = velocity(proposal, (i + 1) / steps)
            x = project(x + dt / 2 * (v + v2), (i + 1) / steps)
        else:
            x = proposal
        if i + 1 in record:
            states.append(x.clone())
    if not torch.isfinite(x).all():
        raise FloatingPointError("Sampling diverged")
    nfe = steps * (1 if method == "euler" else 2)
    return Solution(x, torch.stack(states), nfe, nfe * (1 if guidance == 1 else 2))
