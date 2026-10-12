# 黎曼几何 / 黎曼曲线切入 ViGeo：分析与实验方案

**一句话**：把每个视频 clip 的相机位姿看成 SE(3) 上的一条曲线，在 0/1/2 阶（位置/速度/加速度）上和 GT 曲线对齐。novelty 在 2 阶及以上的曲线几何。

| 文件 | 内容 | 适合谁 |
|---|---|---|
| `README.md`（本文） | 现状分析、novelty 评估、改动清单、消融方案 | 所有人，先读这个 |
| [`01_theory.md`](01_theory.md) | 原理推导：测地距离、body velocity、协变加速度、为什么用匹配式而不是平滑先验 | 想弄懂数学的人 |
| [`02_figures.md`](02_figures.md) | 3 张图，以及每张图的结论 | 组会、快速说服别人 |
| [`interactive.html`](interactive.html) | 浏览器直接打开，拖滑块看各阶损失如何反应（离线，无依赖） | 现场讲解 |
| [`figures/make_figures.py`](figures/make_figures.py) | 生成上面的图，只用 Python 标准库 | 想改图或复现数值的人 |

代码入口：`src/loss/riemannian.py`（实现）、`src/loss/video_depth_loss.py`（接入）、`configs/video_depth/vigeo_stage1_224x224_riemannian.py`（消融配置）。

> 分支 `riemannian-curve`（基于 `origin/train`）。本分支只**新增**文件，外加 `video_depth_loss.py` 里一个开关式分支。
> 默认配置下训练行为与原版完全一致。代码未在 GPU 上跑过，仅做了代码与理论层面的检查。

## 1. 现状：ViGeo 里"几何"在哪，哪里是欧氏的

| 位置 | 现在的做法 | 几何上的问题 |
|---|---|---|
| `src/model/camera_head.py` | 每帧 camera token → MLP → `[t(3), q(4)]`，**逐帧独立** | 输出在 R^7，不在 SE(3)；帧与帧之间没有耦合 |
| `src/loss/loss_util.py::camera_loss` | `L1(t) + L1(q)`，q 经 `mat_to_quat` 标准化（w≥0） | SO(3) 被当成 R^4 里的点；L1 ≠ 测地距离；w≈0（~180°）附近有不连续 |
| `src/loss/video_depth_loss.py` | 所有 loss 都是逐帧/逐像素求和 | **没有任何"轨迹级"监督**。论文标题是 *Consistent Video Geometry*，但时间一致性只能隐式靠 attention 学到 |
| `normal_map_loss` | `angle_between(...)^2` | 已经是 S² 上的测地距离，**这里没什么可做的** |

结论：最自然、改动最小、和论文主张（consistency）最贴的切入点是**相机轨迹 = SE(3) 上的离散曲线**。

## 2. Novelty 评估（诚实版）

| 想法 | 是否已有 | 参考 |
|---|---|---|
| 0 阶：逐帧 SE(3)/SO(3) 测地 loss | **已有**，不算 novelty，只能作为 baseline 修正 | Hou et al. 2018, *Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry* |
| 1 阶：相对位姿 / RPE loss | **已有**（VO 里很常见；FFVO 2026 也用了 relative-pose 监督） | FFVO (arXiv 2609.13733) |
| 无监督"平滑"奖励 | **已有**（VGGRPO 把 camera smoothness 当 reward，用于视频生成） | VGGRPO (arXiv 2603.26599) |
| **2 阶：协变加速度匹配**（把 pred 曲线与 GT 曲线在 Lie 代数里按二阶 Sobolev / H² 方式对齐） | 我没搜到在 feed-forward video geometry（VGGT/DA3/ViGeo 系）里用过 | 理论来源：Lie 群上的曲线形状分析（Celledoni–Eslitzbichler–Schmeding, SRVF on Lie groups） |
| **弹性（重参数化不变）曲线度量**，处理随机步长采样 | 同上，更进一步 | 同上 |

→ 可宣称的 novelty 应放在**二阶 / 弹性曲线几何**上，而不是"用了测地距离"。

## 3. 关键实现约束（读数据代码时发现的）

`src/datasets/base_data.py::sample_sequence`：

- 帧间步长随机 `1..max_interval`（非均匀时间参数化）；
- 到序列末尾会**反向**（GT 轨迹本身有尖点）；
- 以 `shuffle_sequence_prob=0.3` 的概率**打乱帧序**；
- 帧号/时间戳**没有**传给 loss。

所以**不能**用"让预测轨迹变平滑"这种纯先验（会被 30% 的乱序样本和尖点误伤）。
本分支的做法：**只做 pred 与 GT 的匹配**。两条曲线共享同一个离散参数化，因此速度/加速度的差值比较对任意步长、乱序都成立，不需要时间戳。

## 4. 本分支的内容

数学设定：c2w 位姿 `T_i ∈ SE(3)`，度量取 body-frame 下 SO(3)×R³ 的左不变积度量。

```
0 阶  geodesic_camera_loss   d(T_i^pred, T_i^gt) = ||Log(R_pred^T R_gt)|| + λ|t_pred - t_gt|
1 阶  relative_motion_loss   ξ_i = (Log(R_i^T R_{i+1}),  R_i^T (t_{i+1}-t_i))   ← body velocity
2 阶  curve_accel_loss       a_i = ξ_{i+1} - ξ_i                                ← 协变加速度
```

SO(3) 部分取双不变度量，所以左平凡化后的二阶差分就是离散协变加速度（`ad_ξ ξ = 0`）。
1、2 阶项对整条轨迹的全局刚体变换不变，与 `normalize_gt` 的 gauge 选择无关。

| 文件 | 改动 |
|---|---|
| `src/loss/riemannian.py` | **新增**。`so3_log`、`relative_twist`、三个 loss、一个诊断函数 `trajectory_curvature_stats` |
| `src/loss/video_depth_loss.py` | import + `TRAJECTORY_LOSSES` 表 + 一个 `elif` 分支（尺度对齐方式与原 `camera_loss` 相同） |
| `configs/video_depth/vigeo_stage1_224x224_riemannian.py` | **新增**。继承 stage1，只替换 `loss_cfg` |
| `docs/riemannian/` | 本文件夹：概览（本文）、原理推导、图解、交互讲解页 |

启用方法：在任意 config 的 `loss_cfg.losses` 里加上 `geodesic_camera_loss` / `relative_motion_loss` / `curve_accel_loss` 及其权重即可。

## 5. 建议的实验（有 GPU 的合作者来跑）

建议在 stage1（224×224）上做短程 fine-tune 消融，成本最低：

| # | camera 相关 loss | 要验证的问题 |
|---|---|---|
| A | `camera_loss`（原版） | baseline |
| B | `geodesic_camera_loss` | 只换度量有没有用（预计提升很小；不是卖点） |
| C | B + `relative_motion_loss` | 1 阶项的贡献 |
| D | C + `curve_accel_loss` | **核心：2 阶项是否带来额外的轨迹一致性** |

指标：除 ATE 外，**一定要报 RPE（平移/旋转）和抖动指标**（可直接用 `trajectory_curvature_stats` 比较 pred 与 GT 的 `ang_accel`/`lin_accel`），以及 online/chunk 模式下 chunk 边界处的跳变——二阶项预期主要改善这些，ATE 不一定变。

## 6. 后续可以往深做的方向（按推荐顺序）

1. **弹性曲线度量（SRVF on SE(3)）**：把 `curve_accel_loss` 换成重参数化不变的形式，正面处理随机步长。需要数据侧把帧号传出来（改 `sample_sequence` 返回 `seq_ids`，所有 dataset 共用，改动集中）。
2. **流形上的输出头**：camera head 直接输出 se(3) 增量并沿曲线 Exp 累积（pose = 曲线积分），轨迹连续性从结构上保证；属于模型改动，风险更大。
3. **训练无关的验证**：在 benchmark 分支上对已有预测轨迹做 SE(3) 测地样条平滑（纯 CPU），先粗看"二阶正则"对 RPE 的上限收益，再决定要不要训练。

## 7. 已知局限

- 本地没有 torch，**代码没有执行过**，只做了 shape/约定的人工核对（`pose_pred_i_norm` 为 `[S,3,4]`、`pose_gt_i_norm` 为 `[S,4,4]`，函数里只用 `[..., :3, :3]` 和 `[..., :3, 3]`）。合作者第一次跑时建议先在 CPU 上用随机位姿跑一次 forward/backward。
- `so3_log` 在 θ→π 附近数值不稳（`mat_to_quat` 的 w≥0 标准化）；相邻帧相对旋转远小于 π，所以 1、2 阶项不受影响。0 阶项与原版 `camera_loss` 有相同的局限。
- 平移项沿用原版的 L1 + `clamp(max=100)`，保证量级与原 `camera_loss` 可比；权重未调。
