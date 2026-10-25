"""Run: python -m flow_matching_lab.train --output flow_matching_lab/results/seed2026"""

import argparse
import copy
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import VelocityMLP, linear_path
from .data import load_mnist


@torch.inference_mode()
def velocity_validation(model, data, device, seed=271828):
    images, labels = data
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(images.shape, generator=rng)
    times = torch.rand(len(images), generator=rng)
    totals = torch.zeros(5, dtype=torch.float64)
    counts = torch.zeros(5, dtype=torch.float64)
    for offset in range(0, len(images), 512):
        end = offset + 512
        x1, x0, t, y = (v.to(device) for v in (images[offset:end], noise[offset:end],
                                            times[offset:end], labels[offset:end]))
        xt, target = linear_path(x0, x1, t)
        errors = (model(xt, t, y) - target).square().mean(1).cpu().double()
        bins = (times[offset:end] * 5).long().clamp(max=4)
        totals.scatter_add_(0, bins, errors)
        counts.scatter_add_(0, bins, torch.ones_like(errors))
    return {"mse": (totals.sum() / counts.sum()).item(),
            "time_bin_mse": (totals / counts).tolist(), "seed": seed, "samples": len(images)}


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".partial")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("flow_matching_lab/results/seed2026"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=14)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.width, args.depth, args.threads) < 1 or args.lr <= 0:
        parser.error("Training sizes and learning rate must be positive.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    if checkpoint_path.exists() and not args.resume:
        parser.error(f"{checkpoint_path} exists; use --resume or a different output.")
    data = load_mnist(args.cache, args.image_size)
    images, labels = (v.to(device) for v in data["train"])
    model = VelocityMLP(args.image_size ** 2, args.width, args.depth).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"device": str(device), "condition_dropout": 0.1, "ema_decay": 0.995,
                   "path": "independent_linear_cfm", "time_sampling": "uniform_0_1",
                   "pixel_range": [-1, 1], "train_samples": 55000, "validation_samples": 5000,
                   "parameters": sum(p.numel() for p in model.parameters())})
    history, start_step, previous_seconds = [], 0, 0.0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for key in ("batch_size", "width", "depth", "image_size", "lr", "seed", "device", "threads"):
            if checkpoint["config"][key] != config[key]:
                parser.error(f"Resume configuration differs for {key}.")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_step, history = checkpoint["step"], checkpoint["history"]
        previous_seconds = checkpoint["training_seconds"]
        initial = checkpoint["initial_validation"]
    else:
        initial = velocity_validation(model, data["validation"], device)
    print(json.dumps({"event": "start", "config": config, "initial_validation": initial}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    last_log, window_loss, window_count = started, 0.0, 0
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(len(images), (args.batch_size,), device=device)
        x1, y = images[indices], labels[indices].clone()
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)
        y[torch.rand(args.batch_size, device=device) < 0.1] = 10
        xt, target = linear_path(x0, x1, t)
        loss = F.mse_loss(model(xt, t, y), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite training update at {step}")
        optimizer.step()
        with torch.no_grad():
            for shadow, live in zip(ema.parameters(), model.parameters()):
                shadow.lerp_(live, 1 - 0.995)
        window_loss += loss.item()
        window_count += 1
        if step % 250 == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            record = {"step": step, "train_mse": window_loss / window_count,
                      "elapsed_seconds": previous_seconds + now - started,
                      "steps_per_second": window_count / (now - last_log)}
            history.append(record)
            print(json.dumps(record), flush=True)
            last_log, window_loss, window_count = now, 0.0, 0
        if step % 1000 == 0 or step == args.steps:
            save_checkpoint(checkpoint_path, {
                "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "step": step, "config": config, "history": history,
                "training_seconds": previous_seconds + time.perf_counter() - started,
                "initial_validation": initial, "data_manifest": data["manifest"],
            })
    final = velocity_validation(ema, data["validation"], device)
    summary = {
        "config": config, "step": max(start_step, args.steps), "history": history,
        "training_seconds": previous_seconds + time.perf_counter() - started,
        "initial_validation": initial, "final_validation": final,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "data_manifest": data["manifest"],
    }
    (args.output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "final_validation": final}), flush=True)


if __name__ == "__main__":
    main()
