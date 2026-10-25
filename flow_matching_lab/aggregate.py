"""Summarize three measured training runs, including raw pixel variance."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    root = Path("flow_matching_lab")
    runs, detail = [], []
    for seed in (2026, 2027, 2028):
        run = root / "results" / f"seed{seed}"
        training = json.loads((run / "training.json").read_text())
        evaluation = json.loads((run / "evaluation.json").read_text())
        arrays = np.load(run / "samples.npz")
        labels = arrays["labels"]
        def variance(images):
            return float(np.mean([images[labels == c].var(0).mean() for c in range(10)]))
        reference_variance = variance(arrays["reference"])
        for row in evaluation["metrics"]:
            key = {"gaussian_noise": "noise", "untrained_euler_32": "untrained"}.get(row["name"], row["name"])
            if key in arrays:
                v = variance(arrays[key])
                detail.append({
                    "seed": seed, "name": row["name"], "nfe": row["nfe"],
                    "network_evaluations_per_sample": row["network_evaluations_per_sample"],
                    "class_accuracy": row["class_accuracy"], "sliced_w1_raw": row["sliced_w1_raw"],
                    "variance_ratio_to_real_test": v / reference_variance,
                    "mean_clip_displacement": row["mean_clip_displacement"],
                })
        runs.append({
            "seed": seed, "steps": training["step"], "training_seconds": training["training_seconds"],
            "initial_validation_mse": training["initial_validation"]["mse"],
            "final_validation_mse": training["final_validation"]["mse"],
            "classifier_test_accuracy": evaluation["classifier"]["test_accuracy"],
        })
    summary = []
    for name in dict.fromkeys(r["name"] for r in detail):
        group = [r for r in detail if r["name"] == name]
        entry = {"name": name, "runs": len(group), "nfe": group[0]["nfe"],
                 "network_evaluations_per_sample": group[0]["network_evaluations_per_sample"]}
        for key in ("class_accuracy", "sliced_w1_raw", "variance_ratio_to_real_test", "mean_clip_displacement"):
            values = np.array([r[key] for r in group])
            entry[key+"_mean"] = float(values.mean())
            entry[key+"_sd"] = float(values.std(ddof=1))
        summary.append(entry)
    (root / "results-summary.json").write_text(json.dumps({
        "runs": runs, "settings": summary, "evaluation_seed": 123456,
        "notes": ["Means and sample SD across three training seeds, not confidence intervals.",
                  "Shared evaluation pairs; pixel variance is descriptive, not perceptual."]
    }, indent=2) + "\n")
    with (root / "all-sampling-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    lines = ["# Flow Matching 实测结果", "", "均值 ± 3 个训练种子的样本标准差；每设置 1,000 张。",
             "", "| 设置 | NFE | 类别一致率 | Sliced W1 | 类内方差/真实数据 |",
             "|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['name']} | {r['nfe']} "
                     f"| {100*r['class_accuracy_mean']:.2f}% ± {100*r['class_accuracy_sd']:.2f}% "
                     f"| {r['sliced_w1_raw_mean']:.5f} ± {r['sliced_w1_raw_sd']:.5f} "
                     f"| {r['variance_ratio_to_real_test_mean']:.3f} ± {r['variance_ratio_to_real_test_sd']:.3f} |")
    (root / "results-summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
