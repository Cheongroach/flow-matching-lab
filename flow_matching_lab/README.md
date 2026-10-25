# Flow Matching 最小复现

2026-10-25 增加了完整 28×28 的图像条件 U-Net、同分辨率 MLP 对照与笔画补全模式。下面的原始 14×14 实验仍保留，用于理解基础公式。项目使用主流 Flow Matching 方法，项目规模属于课程大作业；没有 SOTA 或生产级框架的证据。

## 28×28 升级版

```bash
python -m flow_matching_lab.image_train --architecture unet --steps 3000 --seed 2026 --output flow_matching_lab/improved/my_unet
python -m flow_matching_lab.image_train --architecture mlp --steps 3000 --seed 2026 --output flow_matching_lab/improved/my_mlp
python -m flow_matching_lab.image_eval --runs flow_matching_lab/improved/my_unet flow_matching_lab/improved/my_mlp
python -m flow_matching_lab.draw --run flow_matching_lab/improved/my_unet --host 0.0.0.0 --port 7866
```

训练同时使用 50% 的普通生成样本和 50% 的遮挡补全样本。输入包含当前 noisy image、已知像素、二值 observation mask、时间和数字类别。训练遮挡包含左右半边、上下半边与矩形缺口；上传任意草图或自由画笔属于更宽的输入分布。

补全模式在每个 ODE 步骤把已知像素投影到相应的线性路径，终点精确保留输入区域；灰色未知区域由网络预测。已知像素误差为 0 是采样约束的结果，不能作为模型学会补全的证据。评估另外比较未知区域误差、类别一致率、条件变化敏感度，以及“普通生成后粘贴已知区域”对照。

```bash
python -m unittest flow_matching_lab.verify flow_matching_lab.test_images -v
```

`image_eval` 使用独立的 MNIST CNN 特征计算 Fréchet 与 cubic-kernel MMD，作为同实验内的辅助距离；它们不是 Inception FID/KID，不能和外部论文数字比较。两个模型对照固定数据、批次与更新次数，参数量和计算量不同；不声称相同训练成本。

这是独立的 PyTorch 实现：在真实 MNIST 上训练线性 Conditional Flow Matching，生成数字图像并验证速度场与 ODE 求解。默认将 28×28 图像平均池化到 14×14，适合 CPU 运行；支持 28×28 和 CUDA。它复现方法、训练流程和采样行为，实验规模与模型结构不同于原论文的 ImageNet 实验。

完整原理、论文脉络与实验结论见 [调研报告](research-report.md)。已训练模型、原始生成数组、指标和图表位于 `results/seed2026`、`seed2027`、`seed2028`。

## 自己玩：生成数字和调整参数

### 画纹路，让输出随输入改变

```bash
python -m flow_matching_lab.draw --host 0.0.0.0 --port 7866
```

在浏览器打开 `http://localhost:7866`。如果运行在远程开发机，将端口 7866 转发到本机，或从 IDE 的端口面板打开预览。

左侧可以用亮笔、暗笔编辑初始噪声，右侧自动调用真实模型重新生成。可以固定目标数字、步数、CFG，然后改变纹路来观察输出笔画变化；支持调整笔刷大小与力度、恢复初始噪声、换噪声和保存输入/输出。保存结果在 `draw_outputs/`。

原始 MLP 支持噪声编辑；使用升级版 checkpoint 时，页面同时提供笔画补全、真实半图示例与上传图片。补全模式保留已知区域，生成未知区域；噪声模式仍允许观察初始噪声的变化。推荐使用 32 步、CFG 1。

在仓库根目录执行，无需重新训练，也无需下载数据：

```bash
python -m flow_matching_lab.play --interactive
```

首次会生成默认数字，然后出现 `fm>` 提示符。输入参数即可再次生成，未指定的参数保持上一次的值：

```text
--digits 2026 --seed 7 --steps 32 --cfg 1.5
--steps 1
--steps 64
--cfg 0
--cfg 2
--seed 123
quit
```

每次输出保存在独立的 `play_outputs/` 子目录，包含 `digits.png`、实际生成轨迹 `generation.gif`、原始数组与设置。`--digits 333` 会生成三行数字 3；`--copies 8` 是每行八种样本。固定 seed、digits 与 copies 后再改变步数、CFG 或 solver，初始噪声相同，可以直接比较结果。

也可以单次执行：

```bash
python -m flow_matching_lab.play --digits 2026 --seed 7 --steps 32 --cfg 2 --copies 8
python -m flow_matching_lab.play --digits 3 --seed 7 --steps 8 --solver heun
```

建议先比较一步与 64 步，再比较 CFG 0、1、2，最后固定参数更换随机种子。这个命令使用原始模型，只学过数字图像和类别条件。升级版补全使用 `draw` 和图像条件 checkpoint；生成其他图案仍需要相应数据与训练。

## 运行

在包含 `flow_matching_lab/` 的仓库根目录执行，依赖 Python ≥3.10、PyTorch、NumPy、Matplotlib 和 Pillow：

```bash
python -m pip install torch numpy matplotlib pillow
python -m flow_matching_lab.train --output flow_matching_lab/results/my_run --device cpu --threads 4 --steps 6000
python -m flow_matching_lab.evaluate --run flow_matching_lab/results/my_run --threads 4 --samples 1000
```

自动下载原始 MNIST IDX 文件到 `/tmp/flow_matching_mnist`，校验四个文件的 MD5，并在训练记录中保存 SHA256。可用 `--cache /path/to/cache` 或 `FLOW_MATCHING_CACHE` 指定持久缓存。首次下载需要联网，后续直接使用缓存。

训练使用官方 60,000 张训练图像中的 55,000 张，5,000 张用于固定速度回归验证；官方 10,000 张测试图像仅用于评估分类器和构建分布参考。所有训练种子使用相同的固定切分。

```bash
# 恢复训练：必须保持模型结构、batch、lr、seed、device 和线程数一致。
python -m flow_matching_lab.train --output flow_matching_lab/results/my_run --device cpu --threads 4 --steps 10000 --resume

# 在有 GPU 的机器上训练完整分辨率；这个配置未在本次 CPU 实验中实测。
python -m flow_matching_lab.train --output flow_matching_lab/results/mnist28_gpu --device cuda --image-size 28 --width 512 --steps 20000
python -m flow_matching_lab.evaluate --run flow_matching_lab/results/mnist28_gpu

# 用已有模型重画图，无需训练。
python -m flow_matching_lab.plot flow_matching_lab/results/seed2026
```

检查点在每 1,000 步和结束时保存模型、EMA、优化器、随机数状态和训练进度。运行相同输出目录时必须显式提供 `--resume`。评估器默认使用 CPU；分类器保存在 `results/classifier_14.pt`，可用 `--classifier` 指定已有分类器。

## 核心公式与实现

约定 \(t=0\) 是噪声，\(t=1\) 是数据，训练端点独立采样：

\[
x_0\sim\mathcal N(0,I),\quad x_1\sim p_{\rm data},\quad
x_t=(1-t)x_0+t x_1,\quad u_t=x_1-x_0.
\]

优化 \(v_\theta(x_t,t,c)\) 与 \(u_t\) 之间的逐像素 MSE。训练时随机采样一个时间点，不调用 ODE 求解器。生成从新的 \(x_0\) 出发，使用 Euler 或 Heun 求解 \(dx/dt=v_\theta(x,t,c)\)。

| 文件 | 阅读重点 |
|---|---|
| `core.py` | `linear_path`、`VelocityMLP`、`guided_velocity`、`integrate` |
| `train.py` | 随机端点与时间采样、条件丢弃、单步 MSE、EMA、恢复训练 |
| `data.py` | 原始数据校验、池化归一化、固定 train/validation/test 切分 |
| `evaluate.py` | 独立分类器、固定噪声、相同 NFE 对比、CFG、原始数组 |
| `plot.py` | 生成数字、采样轨迹动画、求解器对比、训练曲线 |
| `verify.py` | 路径导数、Gaussian 边缘场解析解、积分方向、CFG、官方库对照 |

CFG 定义为 \(v_{\rm null}+s(v_{\rm cond}-v_{\rm null})\)。`s=1` 是普通条件生成，`s=0` 是无条件生成，`s=2` 增强条件。训练以 10% 概率丢弃类别条件。不要直接套用其他项目的 CFG 数值：部分项目用 `1+rate` 作为这里的 `s`。

## 正确性验证

```bash
python -m flow_matching_lab.verify

# 可选：使用官方包校验路径和速度逐值一致。
python -m pip install flow-matching==1.0.10
python -m flow_matching_lab.verify --official --output /tmp/fm-verification.json
```

本次实际完成的验证结果在 `verification.json`：5 项解析检查通过；三种张量形状下与官方 `CondOTProbPath` 的路径及速度误差均为 0。断点恢复检查将连续 40 步与“20 步 + 恢复 20 步”逐值比较，结果记录在 `resume-verification.json`。

## 指标与解释边界

- `class_accuracy` 是独立 MNIST 分类器判断生成图像是否符合请求类别，图像在输入分类器时截到 [-1,1]。这不等于人类图像质量、多样性或 FID。
- `sliced_w1_raw` 是 128 个固定单位方向上的经验一维 Wasserstein-1 距离，使用未截断的原始像素和类别均衡测试参考。这里的数值不能跨分辨率或对比论文 FID。
- `NFE` 表示 ODE 右端速度函数调用次数。Euler 每步 1 次，Heun 每步 2 次。CFG 增强使用条件和无条件两条分支，因此网络对每个样本的计算量会翻倍。
- 采样图展示每类固定的前 8 个噪声，不按分类置信度筛选。动画来自实际积分状态。
- `batch_sampling_seconds` 是共享 CPU 上预热后的单次诊断耗时，不能作为严格性能基准。

## 上游资料

- [Flow Matching 原论文](https://arxiv.org/abs/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)
- [Meta 官方代码](https://github.com/facebookresearch/flow_matching)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [OT-CFM](https://arxiv.org/abs/2302.00482)
- [TorchCFM](https://github.com/atong01/conditional-flow-matching)
