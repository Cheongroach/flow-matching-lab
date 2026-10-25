"""Render measured MNIST results. Every image is a real sample or an ODE state."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def grid(images, side, rows=10, columns=8):
    images = images[:rows * columns].reshape(rows, columns, side, side)
    # Display clipping only; saved arrays and all raw-pixel metrics stay unclipped.
    return images.transpose(0, 2, 1, 3).reshape(rows * side, columns * side).clip(-1, 1)


def render(run):
    run = Path(run)
    arrays = np.load(run / "samples.npz")
    training = json.loads((run / "training.json").read_text())
    evaluation = json.loads((run / "evaluation.json").read_text())
    side, selected = training["config"]["image_size"], arrays["plot_indices"]
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    for key, name, title in (
        ("euler_nfe64_cfg1", "generated_digits.png", "Linear CFM / Euler / 64 NFE / guidance 1"),
        ("euler_nfe32_cfg2", "generated_digits_cfg2.png", "Linear CFM / Euler / 32 RHS NFE / guidance 2"),
        ("reference", "real_reference.png", f"Real held-out MNIST / {side} x {side}"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 8))
        ax.imshow(grid(arrays[key][selected], side), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks(np.arange(10) * side + side / 2, labels=range(10))
        ax.set_ylabel("Class")
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(11, 8))
    for column, budget in enumerate((4, 8, 16, 32, 64)):
        for row, method in enumerate(("euler", "heun")):
            axes[row, column].imshow(grid(arrays[f"{method}_nfe{budget}_cfg1"][selected], side),
                                    cmap="gray", vmin=-1, vmax=1)
            axes[row, column].set_title(f"{method.title()} / {budget} NFE")
            axes[row, column].axis("off")
    fig.suptitle("Paired samples: same initial noise, class and RHS evaluation budget")
    fig.tight_layout()
    fig.savefig(figures / "solver_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in ("euler", "heun"):
        subset = [row for row in evaluation["metrics"] if row["name"].startswith(method + "_nfe")
                  and row["name"].endswith("cfg1")]
        axes[0].plot([r["nfe"] for r in subset], [100 * r["class_accuracy"] for r in subset], "o-", label=method)
        axes[1].plot([r["nfe"] for r in subset], [r["sliced_w1_raw"] for r in subset], "o-", label=method)
    axes[0].set_ylabel("Requested-class agreement (%)")
    axes[1].set_ylabel("Sliced W1 to balanced test images (raw pixels)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("RHS evaluations (guidance 1)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "sampling_metrics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["step"] for r in training["history"]], [r["train_mse"] for r in training["history"]])
    axes[0].set_xlabel("Optimizer steps")
    axes[0].set_ylabel("Training velocity MSE (250-step means)")
    axes[1].plot(np.arange(5) * 0.2 + 0.1, training["initial_validation"]["time_bin_mse"], "o-", label="initial")
    axes[1].plot(np.arange(5) * 0.2 + 0.1, training["final_validation"]["time_bin_mse"], "o-", label="trained EMA")
    axes[1].set_xlabel("Interpolation time bin midpoint")
    axes[1].set_ylabel("Held-out velocity MSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "training_and_validation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 9, figsize=(15, 3.5))
    for i, ax in enumerate(axes):
        subset = arrays["trajectory"][i * 4][::8].reshape(10, side, side).clip(-1, 1)
        ax.imshow(subset.reshape(10 * side, side), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={i / 8:.3g}")
        ax.axis("off")
    fig.suptitle("Actual learned ODE states: noise at t=0 to digits at t=1", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "ode_trajectory.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 6))
    artist = ax.imshow(grid(arrays["trajectory"][0], side), cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")
    def animate(index):
        artist.set_data(grid(arrays["trajectory"][index], side))
        ax.set_title(f"Learned ODE / t={index / 32:.3f}")
        return artist,
    animation = FuncAnimation(fig, animate, frames=33, interval=100, blit=False)
    animation.save(figures / "generation.gif", writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
