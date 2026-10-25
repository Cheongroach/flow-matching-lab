"""Train 28x28 generation and masked completion with a paired MLP/U-Net control."""

import argparse
import copy
import json
import platform
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .core import linear_path
from .data import load_mnist
from .image_model import make_image_model, observation_masks
from .train import save_checkpoint


@torch.inference_mode()
def validate(model, images, labels):
    rng = torch.Generator().manual_seed(271828)
    noise = torch.randn(images.shape, generator=rng)
    times = torch.rand(len(images), generator=rng)
    masks = observation_masks(len(images), 28, rng)
    errors = []
    for offset in range(0, len(images), 128):
        sl = slice(offset, offset + 128)
        xt, target = linear_path(noise[sl], images[sl], times[sl])
        prediction = model(xt, times[sl], labels[sl], images[sl] * masks[sl], masks[sl])
        errors.append((prediction - target).square().flatten(1).mean(1))
    return float(torch.cat(errors).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("unet", "mlp"), default="unet")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.channels, args.width, args.steps, args.batch_size, args.threads) < 1
            or args.channels % 4 or args.lr <= 0):
        parser.error("Use positive settings and channels divisible by 4")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1000)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"image_size": 28, "completion_probability": 0.5, "label_dropout": 0.1,
                   "ema_decay": 0.995, "time_sampling": "uniform", "device": "cpu",
                   "data_split_seed": 314159})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "checkpoint.pt"
    if path.exists() and not args.resume:
        parser.error("Checkpoint exists; use --resume or a new output directory")
    data = load_mnist(image_size=28)
    images, labels = data["train"]
    images = images.reshape(-1, 1, 28, 28)
    model = make_image_model(config)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    history, start, prior_seconds = [], 0, 0
    initial = None
    if args.resume:
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        for key in ("architecture", "channels", "width", "batch_size", "lr", "seed", "threads"):
            if config[key] != checkpoint["config"][key]:
                parser.error(f"Resume mismatch for {key}")
        if args.steps < checkpoint["step"]:
            parser.error("Cannot resume to fewer steps")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start, prior_seconds = checkpoint["step"], checkpoint["training_seconds"]
        history, initial = checkpoint["history"], checkpoint["initial_validation_mse"]
    else:
        initial = validate(ema, data["validation"][0].reshape(-1, 1, 28, 28),
                           data["validation"][1])
    print(json.dumps({"config": config, "initial_validation_mse": initial}), flush=True)
    begun = time.perf_counter()
    accumulated = 0
    count = 0
    for step in range(start + 1, args.steps + 1):
        indices = torch.randint(len(images), (args.batch_size,), generator=rng)
        x1, y = images[indices], labels[indices].clone()
        noise = torch.randn(x1.shape, generator=rng)
        t = torch.rand(len(x1), generator=rng)
        masks = observation_masks(len(x1), 28, rng)
        y[torch.rand(len(x1), generator=rng) < 0.1] = 10
        xt, target = linear_path(noise, x1, t)
        loss = F.mse_loss(model(xt, t, y, x1 * masks, masks), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"Invalid update at step {step}")
        optimizer.step()
        with torch.no_grad():
            for shadow, live in zip(ema.parameters(), model.parameters()):
                shadow.lerp_(live, 0.005)
        accumulated += float(loss.detach())
        count += 1
        if step % 100 == 0 or step == args.steps:
            record = {"step": step, "train_mse": accumulated / count,
                      "training_seconds": prior_seconds + time.perf_counter() - begun}
            history.append(record)
            print(json.dumps(record), flush=True)
            accumulated, count = 0, 0
        if step % 500 == 0 or step == args.steps:
            save_checkpoint(path, {
                "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "config": config,
                "sampler_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                "training_seconds": prior_seconds + time.perf_counter() - begun,
                "history": history, "initial_validation_mse": initial,
                "data_manifest": data["manifest"],
            })
    seconds = prior_seconds + time.perf_counter() - begun
    final = validate(ema, data["validation"][0].reshape(-1, 1, 28, 28),
                     data["validation"][1])
    summary = {
        "config": config, "step": args.steps, "history": history,
        "initial_validation_mse": initial, "final_validation_mse": final,
        "training_seconds": seconds, "training_examples": args.steps * args.batch_size,
        "data_manifest": data["manifest"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    (args.output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "final_validation_mse": final}), flush=True)


if __name__ == "__main__":
    main()
