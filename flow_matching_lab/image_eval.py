"""Paired full-resolution generation and held-out masked-completion evaluation."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_mnist
from .evaluate import balanced_subset
from .image_model import make_image_model, sample_image


class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.embedding = nn.Linear(32 * 7 * 7, 128)
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return F.relu(self.embedding(self.conv(x).flatten(1)))

    def forward(self, x):
        return self.head(self.features(x))


@torch.inference_mode()
def features(model, x):
    return torch.cat([model.features(c) for c in x.split(128)])


def evaluator(data, path):
    model = Evaluator()
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        return model.eval(), saved["summary"]
    torch.manual_seed(424242)
    model = Evaluator()
    rng = torch.Generator().manual_seed(55555)
    images, labels = data["train"]
    images = images.reshape(-1, 1, 28, 28)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(1500):
        indices = torch.randint(len(images), (128,), generator=rng)
        loss = F.cross_entropy(model(images[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(json.dumps({"classifier_step": step + 1, "loss": float(loss.detach())}), flush=True)
    model.eval()
    test_images, test_labels = data["test"]
    with torch.inference_mode():
        predicted = model.head(features(model, test_images.reshape(-1, 1, 28, 28))).argmax(1)
    summary = {"seed": 424242, "training_steps": 1500, "test_samples": 10000,
               "test_accuracy": float((predicted == test_labels).float().mean())}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "summary": summary}, path)
    return model, summary


def frechet_distance(real, generated):
    """Gaussian feature-moment distance in a local MNIST embedding, NOT Inception FID."""
    a, b = real.double().numpy(), generated.double().numpy()
    mean_error = np.square(a.mean(0) - b.mean(0)).sum()
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, q = np.linalg.eigh(c1)
    root = (q * np.sqrt(w.clip(min=0))) @ q.T
    cross = root @ c2 @ root
    trace_root = np.sqrt(np.linalg.eigvalsh((cross + cross.T) / 2).clip(min=0)).sum()
    return float(max(0, mean_error + np.trace(c1) + np.trace(c2) - 2 * trace_root))


def polynomial_mmd(real, generated):
    """Unbiased cubic-kernel MMD; feature scaling fixed from the real reference."""
    scale = real.square().mean().sqrt().clamp_min(1e-6)
    x, y = real.double() / scale, generated.double() / scale
    dim = x.shape[1]
    xx, yy, xy = (x @ x.T / dim + 1) ** 3, (y @ y.T / dim + 1) ** 3, (x @ y.T / dim + 1) ** 3
    return float((xx.sum() - xx.diagonal().sum()) / (len(x) * (len(x) - 1))
                 + (yy.sum() - yy.diagonal().sum()) / (len(y) * (len(y) - 1)) - 2 * xy.mean())


def grid(images, rows, columns, path):
    values = images.reshape(rows, columns, 28, 28).permute(0, 2, 1, 3)
    pixels = ((values.reshape(rows * 28, columns * 28).clamp(-1, 1) + 1) * 127.5).byte().numpy()
    Image.fromarray(pixels).resize((columns * 112, rows * 112), Image.Resampling.NEAREST).save(path)


@torch.inference_mode()
def observation_sensitivity(model, noise, real, labels, observed, outputs):
    """Change only observations, with ten paired samples from each digit."""
    per_class = len(real) // 10
    indices = torch.cat([torch.arange(c * per_class, c * per_class + min(10, per_class))
                         for c in range(10)])
    alternate = real.reshape(10, per_class, 1, 28, 28).roll(1, dims=1).reshape_as(real)
    changed = sample_image(model, noise[indices], labels[indices], 32, 1,
                           alternate[indices], observed[indices]).samples
    missing = 1 - observed[indices]
    return {
        "unknown_rms_after_changed_observation": float(
            (((changed - outputs[indices]).square() * missing).sum() / missing.sum()).sqrt()),
        "sensitivity_samples": len(indices),
        "sensitivity_classes": labels[indices].unique().tolist(),
    }


def completion_region_errors(outputs, baseline, real, observed):
    """Separate target strokes and backgrounds so empty pixels cannot hide errors."""
    missing = 1 - observed
    strokes = missing * (real > 0)
    background = missing * (real <= 0)
    result = {}
    for name, region in (("unknown", missing), ("foreground", strokes), ("background", background)):
        for prefix, values in (("", outputs), ("unconditioned_then_paste_", baseline)):
            result[prefix + name + "_mae"] = float(
                ((values - real).abs() * region).sum() / region.sum().clamp_min(1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("flow_matching_lab/improved"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 100 or args.samples % 10 or args.samples > 8000:
        parser.error("Use 100–8000 samples divisible by 10")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_mnist(image_size=28)
    classifier, classifier_summary = evaluator(data, args.output / "classifier_28.pt")
    classifier.requires_grad_(False)
    real, labels = balanced_subset(*data["test"], args.samples // 10, 10001)
    real = real.reshape(-1, 1, 28, 28)
    real_features = features(classifier, real)
    real_train, _ = balanced_subset(*data["train"], args.samples // 10, 10002)
    real_train = real_train.reshape(-1, 1, 28, 28)
    rng = torch.Generator().manual_seed(123456)
    noise = torch.randn(real.shape, generator=rng)
    directions = F.normalize(torch.randn(784, 128, generator=rng), dim=0)
    sorted_real = (real.flatten(1) @ directions).sort(0).values
    observed = torch.zeros_like(real)
    observed[:, :, :, :14] = 1
    completion_noise = torch.randn(real.shape, generator=rng)
    plot_indices = torch.cat([torch.arange(c * (args.samples // 10), c * (args.samples // 10) + 8)
                              for c in range(10)])
    results = []
    reference_row = {"name": "real_train_reference",
                     "feature_frechet": frechet_distance(real_features, features(classifier, real_train)),
                     "feature_mmd": polynomial_mmd(real_features, features(classifier, real_train)),
                     "sliced_w1_raw": float(((real_train.flatten(1) @ directions).sort(0).values - sorted_real).abs().mean())}
    for run in args.runs:
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = make_image_model(checkpoint["config"]).eval()
        model.load_state_dict(checkpoint["ema"])
        figure_root = run / "figures"
        figure_root.mkdir(exist_ok=True)
        arrays = {"noise": noise.numpy(), "labels": labels.numpy(), "reference": real.numpy(),
                  "plot_indices": plot_indices.numpy()}
        for budget, method, guidance in ((1, "euler", 1), (8, "euler", 1), (32, "euler", 1),
                                        (8, "heun", 1), (32, "euler", 2)):
            steps = budget if method == "euler" else budget // 2
            started = time.perf_counter()
            chunks = [sample_image(model, x, y, steps, guidance, method=method).samples
                      for x, y in zip(noise.split(100), labels.split(100))]
            seconds = time.perf_counter() - started
            samples = torch.cat(chunks)
            embedding = features(classifier, samples.clamp(-1, 1))
            predicted = classifier.head(embedding).argmax(1)
            row = {
                "run": run.name, "architecture": checkpoint["config"]["architecture"],
                "seed": checkpoint["config"]["seed"], "training_steps": checkpoint["step"],
                "name": f"{method}_nfe{budget}_cfg{guidance}", "nfe": budget,
                "network_evaluations_per_sample": budget * (1 if guidance == 1 else 2),
                "class_accuracy": float((predicted == labels).float().mean()),
                "per_class_accuracy": [float((predicted[labels == c] == c).float().mean()) for c in range(10)],
                "feature_frechet": frechet_distance(real_features, embedding),
                "feature_mmd": polynomial_mmd(real_features, embedding),
                "sliced_w1_raw": float(((samples.flatten(1) @ directions).sort(0).values - sorted_real).abs().mean()),
                "variance_ratio": float(torch.stack([samples[labels == c].var(0, unbiased=False).mean()
                                                     for c in range(10)]).mean()
                                       / torch.stack([real[labels == c].var(0, unbiased=False).mean()
                                                      for c in range(10)]).mean()),
                "seconds": seconds,
            }
            arrays[row["name"]] = samples.numpy()
            results.append(row)
            grid(samples[plot_indices], 10, 8, figure_root / (row["name"] + ".png"))
            print(json.dumps(row), flush=True)
        # The same held-out left halves, noise and labels for every architecture.
        outputs = torch.cat([sample_image(model, x, y, 32, 1, target, mask).samples
                             for x, y, target, mask in zip(completion_noise.split(100), labels.split(100),
                                                          real.split(100), observed.split(100))])
        wrong_condition = torch.cat([sample_image(model, x, y, 32, 1).samples
                                     for x, y in zip(completion_noise.split(100), labels.split(100))])
        # Compare to unconditional generation followed by an identical pixel paste.
        paste_baseline = observed * real + (1 - observed) * wrong_condition
        completion = {
            "run": run.name, "observed_fraction": 0.5,
            **completion_region_errors(outputs, paste_baseline, real, observed),
            "known_max_error": float(((outputs - real).abs() * observed).max()),
            "class_accuracy": float((classifier.head(features(classifier, outputs.clamp(-1, 1))).argmax(1)
                                      == labels).float().mean()),
        }
        completion.update(observation_sensitivity(model, completion_noise, real, labels, observed, outputs))
        arrays["completed"] = outputs.numpy()
        arrays["completion_noise"] = completion_noise.numpy()
        arrays["mask"] = observed.numpy()
        arrays["paste_baseline"] = paste_baseline.numpy()
        np.savez_compressed(run / "evaluation_samples.npz", **arrays)
        # Four columns per example: observed / paste baseline / completion / target.
        examples = torch.stack((real * observed - (1 - observed), paste_baseline, outputs, real), 1)
        selected = torch.arange(0, len(real), max(1, len(real) // 10))[:10]
        grid(examples[selected], 10, 4, figure_root / "completion.png")
        record = {"classifier": classifier_summary, "samples": args.samples,
                  "evaluation_seed": 123456, "reference": reference_row,
                  "generation": [r for r in results if r["run"] == run.name], "completion": completion}
        (run / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(completion), flush=True)
    summary = {"classifier": classifier_summary, "reference": reference_row,
               "samples_per_setting": args.samples, "results": results,
               "notes": ["Feature Fréchet/MMD use a local real-trained MNIST CNN, not Inception FID/KID.",
                         "Pixel W1 uses raw pixels; CNN metrics clip to [-1,1].",
                         "Paired 28x28 data, seeds, minibatches, updates and evaluation noise; not equal compute or parameter count.",
                         "Unknown-region MAE is a posterior fidelity diagnostic, not a unique-target generation score."]}
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "generation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
