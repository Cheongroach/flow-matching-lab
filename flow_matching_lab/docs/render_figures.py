"""Rebuild documentation figures from saved experiment outputs, without training.

Run from the repository root:
python flow_matching_lab/docs/render_figures.py --font /path/to/CJK-font.ttf
The full release archive provides the evaluation arrays.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True, help="A font with Chinese glyphs")
    args = parser.parse_args()
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(args.font)).get_name(),
        "font.size": 12, "axes.unicode_minus": False, "savefig.facecolor": "white",
    })
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/images"
    output.mkdir(parents=True, exist_ok=True)

    # Relabel a previously generated controlled edit; do not alter its pixels.
    source = np.asarray(Image.open(root / "improved/edit-response.png"))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5))
    titles = ["输入 A：已知左半图", "输出 A", "输入 B：笔画下移两行", "输出 B"]
    for i, (axis, title) in enumerate(zip(axes, titles)):
        axis.imshow(source[36:260, i * 224:(i + 1) * 224])
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    fig.suptitle("固定初始噪声，修改输入笔画，右半边也随之改变", y=0.99, fontsize=16)
    fig.text(0.5, 0.035, "目标数字 3 · U-Net 训练种子 2026 · 32 步 Euler · CFG = 1",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.83, wspace=0.10)
    fig.savefig(output / "edit-response-friendly.png", dpi=160)
    plt.close(fig)

    arrays = np.load(root / "improved/unet_seed2026/evaluation_samples.npz")
    labels = arrays["labels"]
    indices = [np.flatnonzero(labels == c)[0] for c in (2, 3, 8)]
    fig, axes = plt.subplots(3, 4, figsize=(9, 7.1))
    titles = ["已知左半图", "先生成，再粘贴左半图", "根据左半图补全", "原始测试图"]
    for row, index in enumerate(indices):
        real, mask = arrays["reference"][index, 0], arrays["mask"][index, 0]
        known_display = real * mask + (-0.25) * (1 - mask)
        images = [known_display, arrays["paste_baseline"][index, 0],
                  arrays["completed"][index, 0], real]
        for column, pixels in enumerate(images):
            axis = axes[row, column]
            axis.imshow(pixels, vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(titles[column], fontsize=11)
    fig.suptitle("补全是否有效，要看模型画出的右半边", fontsize=16, y=0.98)
    fig.text(0.5, 0.028,
             "灰色是未知区域 · 每类固定取第一张测试样例 · U-Net 种子 2026\n"
             "两种方法使用相同噪声、类别与 32 步 Euler；CFG = 1",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.11, top=0.89,
                        wspace=0.12, hspace=0.10)
    fig.savefig(output / "completion-friendly.png", dpi=160)
    plt.close(fig)

    indices = np.flatnonzero(labels == 3)[:8]
    keys = ["euler_nfe1_cfg1", "euler_nfe32_cfg1", "reference"]
    row_labels = ["模型只走 1 步", "同样噪声，走 32 步", "真实测试图（独立参考）"]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for row, (key, row_label) in enumerate(zip(keys, row_labels)):
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.imshow(arrays[key][index, 0], vmin=-1, vmax=1, cmap="gray",
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(row_label, rotation=0, ha="right", va="center",
                              labelpad=10, fontsize=11)
    fig.suptitle("都容易认成「3」，不代表生成了足够多的写法", fontsize=16, y=0.98)
    fig.text(0.60, 0.027, "固定取类别 3 的前 8 个样本 · U-Net 种子 2026 · Euler · CFG = 1\n"
             "下排真实图仅供比较；上两排生成时没有看到这些完整图像。",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.15, top=0.88,
                        wspace=0.08, hspace=0.12)
    fig.savefig(output / "diversity-friendly.png", dpi=160)
    plt.close(fig)
    print(f"Rendered three figures to {output}")


if __name__ == "__main__":
    main()
