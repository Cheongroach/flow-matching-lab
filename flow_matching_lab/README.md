# Flow Matching：先玩起来，再理解它

这个项目让你修改一部分数字笔画，再由模型补完剩下的部分。模型学过的内容是 MNIST——一个手写数字数据集，所以它目前只会尝试画 0 到 9，不会根据任意纹路生成照片或其他物体。

![固定噪声时，改变已知笔画会改变补全结果](docs/images/edit-response-friendly.png)

图中只把输入笔画向下移动了两行，目标数字和初始噪声都没变。模型画出的右半边也改变了。这是实际模型输出；保留已知左半边则是程序明确执行的约束。

## 打开画板

所有命令都在**包含 `flow_matching_lab/` 的目录**运行。需要 Python 3.10 或更新版本。

```bash
python -m pip install -r flow_matching_lab/requirements.txt

# 已有 checkpoint.pt 时可以跳过下载。
python -m flow_matching_lab.download

python -m flow_matching_lab.draw \
  --run flow_matching_lab/improved/unet_seed2026
```

打开 `http://localhost:7866`。如果程序在远程开发机上运行，需要在 IDE 中把 7866 端口转发到自己的电脑，再打开该地址。

这里下载的是**训练好的模型参数**，不是训练数据。只玩画板不需要重新训练；下载脚本会校验文件是否完整。

## 第一次可以这样玩

1. 目标数字选 `3`，保持“32 步”和“CFG 1”，点击“载入半个真实数字”。
2. 白色笔画和黑色背景是你指定的部分；灰色表示“这里请模型来画”。
3. 用白笔或黑笔修改左边。右边会重新生成，已知区域保持原样，灰色区域由模型补全。
4. 点击“交给模型”后涂抹，可以把指定区域重新变成未知区域。
5. 点击“撤回到初始输入”恢复载入时的内容；点击“换一种补全”尝试另一种结果。

“上传自己的图片”会把图片缩成 28×28，默认只保留左半边。它不是照着完整草图描画；自己的字迹也可能与训练数据差别很大。

另一个模式叫“噪声编辑”。噪声是生成开始前的一张随机像素图。在这个模式里，画笔修改的是随机起点，不是要求模型保留的笔画，因此局部涂改可能影响整个数字。

## 界面上的参数是什么意思

| 参数 | 改的是什么 | 第一次怎么选 |
|---|---|---|
| 目标数字 | 告诉模型希望生成哪个类别 | 选一个数字，比较时保持不变 |
| 种子（seed） | 决定初始随机噪声；同一种子让比较更容易 | 保持不变，只修改笔画 |
| 步数 | 生成一张图时，分多少小步修改图像 | 32；减少步数可能更快，也可能损失细节 |
| CFG | 加强“画出目标数字”的引导力度 | 1 是普通生成，2 会加强；不是越大越好 |

CFG 是 classifier-free guidance 的缩写。这里只需要把它当作“类别引导强度”。它与“保留已知像素”是两件事；补全模式即使把 CFG 设为 0，也仍然接收已知像素。

“保存输入与输出”会在 `flow_matching_lab/draw_outputs/` 下创建一个新目录，保存图片、已知区域标记和生成设置。修改过噪声时，保存的完整噪声数组比种子更准确地描述这次输入。

## 接下来读哪篇

| 你想弄清什么 | 阅读位置 |
|---|---|
| 为什么随机噪声能变成数字？训练时有没有答案？ | [一步步认识 Flow Matching](research-report-friendly.md) |
| 升级版到底有没有改善？89.35% 和 25% 是什么意思？ | [28×28 实验解读](improved/results-friendly.md) |
| 原始版本为什么一步分数很高，却不一定更好？ | [14×14 实验解读](results-summary-friendly.md) |
| 这个项目是什么水平？再往上做难在哪？ | [项目水平与下一步](upgrade-report.md) |

研究报告和两份结果表保留了技术细节，配套的 `-friendly` 是第一次阅读时的入口。运行说明和项目水平说明直接更新原文件，不需要比较两个文字版本。

## 想自己训练，再看这里

画板使用的升级版网络叫 U-Net。它在处理图像时保留像素的空间关系，并结合局部笔画和整体形状。用于对照的 MLP 把像素摊成一列数字，再用全连接层处理。两者都采用同一种 Flow Matching 训练方法；网络结构和训练方法不是同一个概念。

训练命令中的 `--steps 3000` 是**更新模型参数 3,000 次**，不是画板中的“生成走 32 步”。每次更新默认使用 64 张图，称为 batch size 64。

```bash
python -m flow_matching_lab.image_train \
  --architecture unet --seed 2026 --steps 3000 \
  --output flow_matching_lab/improved/my_unet

python -m flow_matching_lab.image_train \
  --architecture mlp --width 1024 --seed 2026 --steps 3000 \
  --output flow_matching_lab/improved/my_mlp

python -m flow_matching_lab.image_eval \
  --runs flow_matching_lab/improved/my_unet flow_matching_lab/improved/my_mlp

python -m flow_matching_lab.draw \
  --run flow_matching_lab/improved/my_unet
```

`image_eval` 为命令中指定的每个模型生成评估记录。`image_aggregate` 则汇总 `improved/` 下所有已经评估完成的目录；如果里面还保留着发布版实验，新实验会与它们一起列出。它会重建完整表格，配套的 `-friendly` 解读记录的是已发布实验，不会自动改写。

训练第一次会下载 MNIST，并检查原始文件的 MD5，记录 SHA256。默认缓存位于 `/tmp/flow_matching_mnist`；需要持久保存时，可以设置 `FLOW_MATCHING_CACHE` 环境变量。数据固定分成 55,000 张训练、5,000 张验证和 10,000 张测试。训练只使用训练部分。

升级版 `image_train` 当前使用 CPU。训练样本约一半用于普通生成，一半用于半图或矩形遮挡补全；自由画笔不完全符合这些训练条件。

每 500 次更新和训练结束时保存 `checkpoint.pt`，其中包含模型参数、优化器进度、随机数状态，以及供评估使用的参数滑动平均（EMA）。继续同一次训练时使用：

```bash
python -m flow_matching_lab.image_train \
  --architecture unet --seed 2026 --steps 5000 \
  --output flow_matching_lab/improved/my_unet --resume
```

这里的 5,000 是累计目标更新次数。恢复时必须保持架构、通道数、宽度、batch size、学习率、种子和线程数一致。

## 原始 14×14 版本有什么用

原始版本把图像缩小到 14×14，结构更简单，适合阅读基本训练代码。它可以生成数字和编辑初始噪声，但不支持新版的已知笔画补全。

```bash
# 下载原始实验的模型与评估数组。
python -m flow_matching_lab.download --asset original
python -m flow_matching_lab.play --interactive
```

看到 `fm>` 后，可输入 `--digits 333 --seed 7 --steps 32 --cfg 1`。它会生成三行数字 3；每行有多个不同样本。输入 `--steps 1` 只改生成步数，输入 `quit` 退出。图片和实际生成过程的动画保存在 `flow_matching_lab/play_outputs/`。

要重新训练原始版本：

```bash
python -m flow_matching_lab.train \
  --output flow_matching_lab/results/my_run --device cpu \
  --threads 4 --steps 6000
python -m flow_matching_lab.evaluate \
  --run flow_matching_lab/results/my_run --threads 4 --samples 1000
```

原始训练器每 1,000 次更新保存一次。恢复方式同样是加 `--resume` 并提高累计 `--steps`，保持原配置不变。它还提供 `--image-size 28 --device cuda`，但这仍是 MLP，不会变成新版 U-Net；本项目没有该 GPU 配置的实测结果。

评估器默认使用 CPU，原始数字识别器保存为 `flow_matching_lab/results/classifier_14.pt`。可用 `evaluate --classifier` 指定已有识别器，用 `python -m flow_matching_lab.plot flow_matching_lab/results/seed2026` 重画原始图表。

## 文件、验证与已知限制

| 要找的内容 | 文件 |
|---|---|
| 最简单的训练路径和生成循环 | `core.py`、`train.py` |
| 图像条件网络、补全和升级版训练 | `image_model.py`、`image_train.py` |
| 数据下载与划分 | `data.py` |
| 原始版 / 升级版评估 | `evaluate.py` / `image_eval.py` |
| 完整结果表的生成 | `aggregate.py`、`image_aggregate.py` |
| 画板与服务 | `editor.html`、`draw.py` |

```bash
python -m unittest flow_matching_lab.verify flow_matching_lab.test_images -v
python -m flow_matching_lab.http_check \
  --run flow_matching_lab/improved/unet_seed2026 \
  --output /tmp/fm-http-verification.json
```

发布版已通过 11 项 CPU 检查和真实模型的 HTTP 请求检查；后者验证输入改变、重复输入的确定性、已知像素保留、非法请求拒绝与保存回读。官方路径对照和恢复训练的结果保存在 [verification.json](verification.json)、[resume-verification.json](resume-verification.json) 和 [升级版恢复记录](improved/resume-verification.json)。可安装 `flow-matching==1.0.10` 后运行 `python -m flow_matching_lab.verify --official` 重新核对路径。

这些检查不等于完整浏览器验证。本环境的 Chromium 在打开页面时退出，画笔、图片上传和手机布局的自动化操作尚未通过。可选检查入口是 `browser_check.py`，需要另行安装 Playwright 和浏览器。

模型与原始生成数组存放在 [GitHub Release](https://github.com/Cheongroach/flow-matching-lab/releases/tag/v0.2.0)。默认下载只包含演示模型；`download --asset full` 下载全部 28×28 实验，`--asset original` 下载原始 14×14 实验。
