"""Evaluate paired ODE samples with an independent classifier and pixel distance."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.nn import functional as F

from .core import DigitClassifier, VelocityMLP, integrate
from .data import load_mnist


@torch.inference_mode()
def classify(model, images):
    return torch.cat([model(chunk).argmax(1) for chunk in images.split(512)])


def train_classifier(data, path, steps=1800):
    model = DigitClassifier(data["train"][0].shape[1])
    if path.exists():
        saved = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(saved["model"])
        return model.eval(), saved["summary"]
    torch.manual_seed(424242)
    model = DigitClassifier(data["train"][0].shape[1])
    images, labels = data["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(images), (256,))
        loss = F.cross_entropy(model(images[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 600 == 0:
            print(json.dumps({"classifier_step": step + 1, "loss": loss.item()}), flush=True)
    model.eval()
    correct = int((classify(model, data["test"][0]) == data["test"][1]).sum())
    summary = {"seed": 424242, "steps": steps, "train_samples": 55000,
               "test_samples": 10000, "test_correct": correct, "test_accuracy": correct / 10000,
               "training_seconds": time.perf_counter() - start}
    torch.save({"model": model.state_dict(), "summary": summary}, path)
    return model, summary


def wilson_interval(correct, count):
    z, p = 1.959963984540054, correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def balanced_subset(images, labels, count_per_class, seed):
    rng = torch.Generator().manual_seed(seed)
    indices = []
    for label in range(10):
        candidates = (labels == label).nonzero().flatten()
        if len(candidates) < count_per_class:
            raise ValueError("Too many requested samples for the held-out reference.")
        indices.append(candidates[torch.randperm(len(candidates), generator=rng)[:count_per_class]])
    selected = torch.cat(indices)
    return images[selected], labels[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("flow_matching_lab/results/seed2026"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--classifier", type=Path)
    args = parser.parse_args()
    if args.samples % 10 or not 80 <= args.samples <= 8000:
        parser.error("--samples must be a multiple of 10 in [80, 8000].")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.run / "checkpoint.pt", weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    side, dim = config["image_size"], config["image_size"] ** 2
    data = load_mnist(args.cache, side)
    model = VelocityMLP(dim, config["width"], config["depth"]).eval()
    model.load_state_dict(checkpoint["ema"])
    torch.manual_seed(config["seed"])
    untrained = VelocityMLP(dim, config["width"], config["depth"]).eval()
    classifier_path = args.classifier or args.run.parent / f"classifier_{side}.pt"
    classifier, classifier_summary = train_classifier(data, classifier_path)
    generator = torch.Generator().manual_seed(123456)
    noise = torch.randn(args.samples, dim, generator=generator)
    labels = torch.arange(10).repeat_interleave(args.samples // 10)
    reference, _ = balanced_subset(*data["test"], args.samples // 10, 10001)
    real_train, _ = balanced_subset(*data["train"], args.samples // 10, 10002)
    directions = F.normalize(torch.randn(dim, 128, generator=generator), dim=0)
    sorted_reference = (reference @ directions).sort(dim=0).values
    plot_indices = torch.cat([
        torch.arange(label * (args.samples // 10), label * (args.samples // 10) + 8)
        for label in range(10)
    ])
    rows, arrays = [], {"noise": noise.numpy(), "labels": labels.numpy(),
                        "reference": reference.numpy(), "directions": directions.numpy(),
                        "plot_indices": plot_indices.numpy()}

    def metrics(name, samples, nfe=0, network_evaluations=0, seconds=None):
        prediction = classify(classifier, samples.clamp(-1, 1))
        correct = int((prediction == labels).sum())
        row = {
            "name": name, "nfe": nfe, "network_evaluations_per_sample": network_evaluations,
            "class_accuracy": correct / len(labels),
            "class_accuracy_95ci": wilson_interval(correct, len(labels)),
            "per_class_accuracy": [
                (prediction[labels == label] == label).float().mean().item() for label in range(10)
            ],
            "sliced_w1_raw": (
                (samples @ directions).sort(dim=0).values - sorted_reference
            ).abs().mean().item(),
            "out_of_range_fraction": ((samples < -1) | (samples > 1)).float().mean().item(),
            "mean_clip_displacement": (samples - samples.clamp(-1, 1)).abs().mean().item(),
            "batch_sampling_seconds": seconds,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    metrics("gaussian_noise", noise)
    metrics("real_train_reference", real_train)
    baseline = integrate(untrained, noise, labels, steps=32)
    metrics("untrained_euler_32", baseline.samples, baseline.nfe, baseline.network_evaluations_per_sample)
    arrays["untrained"] = baseline.samples.numpy()
    for method in ("euler", "heun"):
        budgets = (1, 2, 4, 8, 16, 32, 64) if method == "euler" else (2, 4, 8, 16, 32, 64)
        for budget in budgets:
            steps = budget if method == "euler" else budget // 2
            integrate(model, noise[:16], labels[:16], steps=2, method=method)
            started = time.perf_counter()
            solution = integrate(model, noise, labels, steps=steps, method=method)
            seconds = time.perf_counter() - started
            name = f"{method}_nfe{budget}_cfg1"
            metrics(name, solution.samples, solution.nfe, solution.network_evaluations_per_sample, seconds)
            arrays[name] = solution.samples.numpy()
    for guidance in (0.0, 2.0):
        solution = integrate(model, noise, labels, steps=32, method="euler", guidance=guidance)
        name = f"euler_nfe32_cfg{int(guidance)}"
        metrics(name, solution.samples, solution.nfe, solution.network_evaluations_per_sample)
        arrays[name] = solution.samples.numpy()
    trajectory = integrate(model, noise[plot_indices], labels[plot_indices], steps=64, snapshots=32)
    arrays["trajectory"] = trajectory.trajectory.numpy()
    np.savez_compressed(args.run / "samples.npz", **arrays)
    summary = {
        "training_seed": config["seed"], "evaluation_seed": 123456, "samples": args.samples,
        "classifier": classifier_summary, "metrics": rows, "projection_directions": 128,
        "evaluation_device": "cpu", "evaluation_threads": args.threads,
        "metric_notes": {
            "class_accuracy": "Classifier agreement with requested label on clipped display pixels; not human image quality.",
            "sliced_w1_raw": "Mean empirical 1D W1 over 128 fixed unit directions in raw pixel coordinates; not FID.",
            "nfe": "Guided RHS evaluations; CFG 0 or 2 doubles network work per sample with concatenated batches.",
            "timing": "One CPU batch after a small warmup; diagnostic only, not a controlled throughput benchmark.",
        },
    }
    (args.run / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.run / "sampling_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from .plot import render
    render(args.run)


if __name__ == "__main__":
    main()
