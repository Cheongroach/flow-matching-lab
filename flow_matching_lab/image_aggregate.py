"""Aggregate completed paired image experiments; plot measured data only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("flow_matching_lab/improved"))
    args = parser.parse_args()
    root = args.root
    runs, rows, completions = [], [], []
    for path in sorted(root.glob("*/evaluation.json")):
        evaluation = json.loads(path.read_text())
        training = json.loads((path.parent / "training.json").read_text())
        config = training["config"]
        group = ("U-Net (16 channels)" if config["architecture"] == "unet"
                 else f"MLP ({config['width']})")
        run = {
            "run": path.parent.name, "group": group, "seed": config["seed"],
            "parameters": config["parameters"], "training_steps": training["step"],
            "validation_mse": training["final_validation_mse"],
            "training_seconds": training["training_seconds"],
        }
        runs.append(run)
        rows.extend(dict(row, group=group) for row in evaluation["generation"])
        completions.append(dict(evaluation["completion"], group=group))
    if not runs:
        parser.error("No completed evaluation.json files")
    settings = []
    keys = ("class_accuracy", "feature_frechet", "feature_mmd", "sliced_w1_raw", "variance_ratio")
    for group in dict.fromkeys(run["group"] for run in runs):
        for name in dict.fromkeys(row["name"] for row in rows):
            subset = [row for row in rows if row["group"] == group and row["name"] == name]
            if not subset:
                continue
            entry = {"group": group, "setting": name, "runs": len(subset)}
            for key in keys:
                values = np.array([row[key] for row in subset])
                entry[key + "_mean"] = float(values.mean())
                entry[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            settings.append(entry)
    summary = {
        "runs": runs, "settings": settings, "completion": completions,
        "samples_per_setting": evaluation["samples"], "classifier": evaluation["classifier"],
        "reference": evaluation["reference"],
        "notes": [
            "Sample SD across training seeds, not a confidence interval; wide MLP has one seed.",
            "Same resolution, data split, 3000 updates, batch64, training sample RNG and evaluation inputs.",
            "Models differ in parameters and compute; shared CPU timings are diagnostic only.",
            "Feature Fréchet/MMD use a local MNIST CNN, not Inception FID/KID.",
            "Known pixel retention is enforced by projection, not evidence of learned completion.",
            "Unknown MAE compares one posterior sample to one target; ambiguity remains.",
        ],
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, values in (("generation.csv", rows), ("completion.csv", completions)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# 28×28 实测对照", "",
        "每设置 1,000 个固定噪声；32 步 Euler、CFG 1。± 表示训练种子的样本标准差。",
        "加宽 MLP 只有一个种子。不同架构未匹配计算成本或分别调参。", "",
        "| 模型 | 种子数 | 参数 | 类别一致率 | MNIST 特征 Fréchet ↓ | 原始像素 W1 ↓ | 类内方差/真实 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def display(row, key, digits=4, multiplier=1):
        result = f"{row[key + '_mean'] * multiplier:.{digits}f}"
        if row[key + "_sd"] is not None:
            result += f" ± {row[key + '_sd'] * multiplier:.{digits}f}"
        return result
    for row in settings:
        if row["setting"] != "euler_nfe32_cfg1":
            continue
        parameters = next(r["parameters"] for r in runs if r["group"] == row["group"])
        lines.append(
            f"| {row['group']} | {row['runs']} | {parameters:,} "
            f"| {display(row, 'class_accuracy', 2, 100)}% | {display(row, 'feature_frechet', 2)} "
            f"| {display(row, 'sliced_w1_raw')} | {display(row, 'variance_ratio', 3)} |"
        )
    lines.extend(["", "| 补全 run | 未知区域 MAE ↓ | 生成后粘贴 MAE ↓ | 降幅 | 已知误差 |",
                  "|---|---:|---:|---:|---:|"])
    for row in completions:
        reduction = 1 - row["unknown_mae"] / row["unconditioned_then_paste_unknown_mae"]
        lines.append(f"| {row['run']} | {row['unknown_mae']:.4f} "
                     f"| {row['unconditioned_then_paste_unknown_mae']:.4f} "
                     f"| {100 * reduction:.1f}% | {row['known_max_error']:.1f} |")
    lines.extend(["", "辅助 CNN 的真实测试准确率为 98.75%。特征距离不可与论文 FID/KID 比较。",
                  "未知 MAE 是固定噪声下的补全诊断，不能衡量所有合理补全的质量。"])
    (root / "results.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for group in dict.fromkeys(run["group"] for run in runs):
        subset = [r for r in settings if r["group"] == group
                  and r["setting"] in ("euler_nfe1_cfg1", "euler_nfe8_cfg1", "euler_nfe32_cfg1")]
        nfe = [int(r["setting"].split("_")[1][3:]) for r in subset]
        for axis, key in zip(axes, ("class_accuracy", "feature_frechet", "variance_ratio")):
            means = [r[key + "_mean"] for r in subset]
            sd = [r[key + "_sd"] or 0 for r in subset]
            axis.errorbar(nfe, means, yerr=sd, marker="o", label=group, capsize=3)
            axis.set_xscale("log", base=2)
            axis.set_xticks([1, 8, 32], ["1", "8", "32"])
            axis.set_xlabel("Euler NFE, CFG = 1")
            axis.grid(alpha=0.2)
    axes[0].set_title("Requested digit agreement")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("MNIST feature Frechet (not FID)")
    axes[1].set_yscale("log")
    axes[2].set_title("Within-class pixel variance / real")
    axes[2].axhline(1, color="gray", linestyle="--")
    axes[0].legend(fontsize=8)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"runs": len(runs), "output": str(root / "results.md")}))


if __name__ == "__main__":
    main()
