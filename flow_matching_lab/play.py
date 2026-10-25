"""Generate your own digits, or use a persistent terminal prompt."""

import argparse
import json
from pathlib import Path
import shlex
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import VelocityMLP, integrate


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", default="0123456789", help="Digits to generate, e.g. 333 or 2026")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--copies", type=int, default=8)
    return parser


def generate(model, side, args, output):
    if not args.digits or any(c not in "0123456789" for c in args.digits):
        raise ValueError("--digits must contain digits 0–9.")
    if not 1 <= args.copies <= 32 or not 1 <= len(args.digits) <= 20:
        raise ValueError("Use 1–32 copies and 1–20 digits.")
    if not 1 <= args.steps <= 256:
        raise ValueError("Use 1–256 ODE steps.")
    labels = torch.tensor([int(c) for c in args.digits]).repeat_interleave(args.copies)
    noise = torch.randn(len(labels), side * side, generator=torch.Generator().manual_seed(args.seed))
    solution = integrate(model, noise, labels, steps=args.steps, method=args.solver,
                         guidance=args.cfg, snapshots=args.steps)
    folder = output / f"{time.time_ns()}_seed{args.seed}"
    folder.mkdir(parents=True)

    def frame(samples, caption):
        pixels = ((samples.reshape(len(args.digits), args.copies, side, side)
                   .permute(0, 2, 1, 3).reshape(len(args.digits) * side, args.copies * side)
                   .clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
        image = Image.fromarray(pixels).convert("RGB").resize(
            (args.copies * side * 8, len(args.digits) * side * 8), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (image.width + 32, image.height + 40), "white")
        canvas.paste(image, (32, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), caption, fill="black")
        for row, digit in enumerate(args.digits):
            draw.text((10, 40 + row * side * 8 + side * 4), digit, fill="black")
        return canvas

    caption = f"seed={args.seed} {args.solver} steps={args.steps} cfg={args.cfg}"
    frame(solution.samples, caption).save(folder / "digits.png")
    indices = np.unique(np.linspace(0, args.steps, min(args.steps + 1, 33)).round().astype(int))
    frames = [frame(solution.trajectory[i], f"{caption} t={i/args.steps:.2f}") for i in indices]
    frames[0].save(folder / "generation.gif", save_all=True, append_images=frames[1:],
                   duration=120, loop=0)
    np.savez_compressed(folder / "samples.npz", noise=noise.numpy(),
                        samples=solution.samples.numpy(), labels=labels.numpy())
    metadata = vars(args) | {"nfe": solution.nfe,
                            "network_evaluations_per_sample": solution.network_evaluations_per_sample}
    (folder / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Image: {folder / 'digits.png'}\nAnimation: {folder / 'generation.gif'}", flush=True)
    return folder


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--run", type=Path, default=Path("flow_matching_lab/results/seed2026"))
    bootstrap.add_argument("--output", type=Path, default=Path("flow_matching_lab/play_outputs"))
    bootstrap.add_argument("--interactive", action="store_true")
    bootstrap.add_argument("--threads", type=int, default=4)
    setup, remaining = bootstrap.parse_known_args()
    parser = options()
    args = parser.parse_args(remaining)
    if setup.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(setup.threads)
    saved = torch.load(setup.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    model = VelocityMLP(config["image_size"] ** 2, config["width"], config["depth"]).eval()
    model.load_state_dict(saved["ema"])
    generate(model, config["image_size"], args, setup.output)
    if setup.interactive:
        print("Change settings without reloading the model. Example:\n"
              "--digits 2026 --seed 7 --steps 8 --cfg 2\n"
              "Enter help for options, or quit to stop.")
        while True:
            try:
                line = input("fm> ").strip()
                if line in {"quit", "exit", "q"}:
                    break
                if line in {"help", "?"}:
                    parser.print_help()
                    continue
                if line:
                    updated = parser.parse_args(shlex.split(line), namespace=argparse.Namespace(**vars(args)))
                    generate(model, config["image_size"], updated, setup.output)
                    args = updated
            except (ValueError, SystemExit) as exc:
                print(f"Invalid settings: {exc}")
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
