# 从零读懂：用黎曼几何约束 ViGeo 的相机轨迹

这个文件夹是一套**从零开始的教程**，讲清楚这个分支做了什么、为什么这么做、背后的黎曼几何是什么。
不需要微分几何背景，会线性代数（矩阵乘法、向量范数）和一点 PyTorch 就够了。

## 30 秒版本

> ViGeo 每帧预测一个相机位姿，原来的 loss **一帧一帧单独比**，而且把旋转当成普通向量来减。
> 我们把一段视频的所有相机位姿看成**一条曲线**，在一个"弯曲的空间"（旋转群）里，按**位置、速度、加速度**三个层次和真值曲线对齐。
> 加速度这一层专门惩罚轨迹"抖"，这是视频几何"不一致"的直接表现，也是我们想要的 novelty。

![pipeline](figures/concept_pipeline.svg)

## 阅读路线

按顺序读，每章 10–15 分钟，每章只引入一个新概念。

| 章 | 文件 | 你会搞懂 | 前置 |
|---|---|---|---|
| 0 | [00_overview.md](00_overview.md) | **不讲数学**：问题是什么、改了哪些文件、怎么开关 | 无 |
| 1 | [01_manifold_basics.md](01_manifold_basics.md) | 黎曼几何最小必需集：流形、切空间、度量、测地线、Log/Exp | 无 |
| 2 | [02_rotations_lie_groups.md](02_rotations_lie_groups.md) | 为什么旋转是"弯的"、四元数、SO(3)/SE(3)、李群/李代数 | 第 1 章 |
| 3 | [03_curves_on_manifolds.md](03_curves_on_manifolds.md) | 弯曲空间里的曲线：速度、加速度、为什么能"直接相减" | 第 1、2 章 |
| 4 | [04_vigeo_and_our_losses.md](04_vigeo_and_our_losses.md) | ViGeo 的位姿怎么来、原 loss 的问题、三个新 loss **逐行对照代码** | 第 1–3 章 |
| 5 | [05_figures_and_evidence.md](05_figures_and_evidence.md) | 三个小实验：原 loss 会跳变、各阶分工、为什么不用平滑先验 | 第 4 章 |
| 6 | [06_experiments_and_novelty.md](06_experiments_and_novelty.md) | novelty 边界、要跑的消融、后续方向、已知局限 | 第 4 章 |
| 7 | [07_novelty_catalog.md](07_novelty_catalog.md) | **黎曼几何 novelty 全清单**：不只相机，还有法向、尺度、形状空间等，每条都有实现方法、风险、优先级 | 第 1–3 章 |
| — | [glossary.md](glossary.md) | 中英术语对照 + 一句话解释，随时查 | — |

**只想快速了解**：读第 0 章和第 5 章，再打开 [`interactive.html`](interactive.html)（下载到本地后用浏览器打开，离线可用）拖一拖滑块。
**要跑实验**：读第 0、4、6 章。
**要定下一步做什么**：读第 7 章开头的总览表和路线图，以及 [`cpu_checks/RESULTS.md`](cpu_checks/RESULTS.md)（CPU 验证结果）。
**要写论文的方法部分**：第 1–4 章全部读完。

## 文件夹结构

```
docs/riemannian/
├── README.md                     ← 你在这里
├── 00_overview.md … 07_*.md      ← 教程正文（第 7 章 = novelty 全清单）
├── glossary.md                   ← 术语表
├── interactive.html              ← 交互演示（离线、无依赖）
├── smoke_test.py                 ← torch 版 loss 的冒烟测试（需要 CPU 版 torch）
├── cpu_checks/                   ← 纯 Python 验证 R0–R4，结果见 RESULTS.md：python docs/riemannian/cpu_checks/run_all.py
└── figures/
    ├── make_figures.py           ← 生成所有图，只用 Python 标准库：python docs/riemannian/figures/make_figures.py
    ├── concept_*.svg             ← 概念示意图（第 0–3 章用）
    └── fig1~3_*.svg              ← 小实验结果图（第 5 章用）
```

代码在仓库里的位置：`src/loss/riemannian.py`（实现）、`src/loss/video_depth_loss.py`（接入）、`configs/video_depth/vigeo_stage1_224x224_riemannian.py`（消融配置）。
