# 第 6 章 novelty 边界、实验方案、后续方向

[← 上一章](05_figures_and_evidence.md) · [目录](README.md) · [术语表 →](glossary.md)

> 本章目标：说清楚哪些是别人做过的、哪些可能是新的；有 GPU 的合作者该跑什么实验、看什么指标；以及接下来可以往哪里深挖。

---

## 6.1 novelty：哪些已有，哪些可能是新的

| 想法 | 是否已有 | 参考 |
|---|---|---|
| 0 阶：逐帧位姿用测地距离 | **已有**，只能算对 baseline 的修正 | Hou et al. 2018, *Computing CNN Loss and Gradients for Pose Estimation with Riemannian Geometry* |
| 1 阶：相对位姿 / RPE 监督 | **已有**，VO 里很常见 | FFVO (arXiv 2609.13733) 等 |
| 不看真值、直接要求轨迹平滑 | **已有**，作为视频生成的 reward | VGGRPO (arXiv 2603.26599) |
| **2 阶：协变加速度匹配**（预测与真值曲线在李代数里按二阶 Sobolev 方式对齐） | 在 feed-forward 视频几何模型（VGGT / DA3 / ViGeo 系）里**没搜到** | 理论来源：Lie 群上的曲线形状分析（Celledoni、Eslitzbichler、Schmeding） |
| **弹性曲线度量**（对重参数化不变，正面处理随机帧间隔） | 同上 | 同上 |

**可以宣称的 novelty 在 2 阶和弹性度量**，不在"用了测地距离"。

> ⚠️ 上表只基于两次网络检索，**投稿前必须做系统的文献调研**。重点查：VGGT 后续工作、视频深度/位姿联合估计、李群上的 Sobolev / SRVF 度量在视觉中的应用。

---

## 6.2 一个实现约束（读数据代码时发现的）

`src/datasets/base_data.py :: sample_sequence` 的行为：

- 帧间隔在 `1..max_interval` 之间随机，时间参数化不均匀；
- 走到视频末尾会**掉头**，真值轨迹本身带尖点；
- 以 `shuffle_sequence_prob=0.3` 的概率**打乱帧序**；
- 帧号 / 时间戳**没有**传给 loss。

所以本分支只用匹配式 loss（第 3.5 节、第 5 章图 3）。如果要做 6.4 节的弹性度量，需要把帧号传出来。

---

## 6.3 建议的消融实验（需要 GPU）

在 stage1（224×224）上做短程 fine-tune，成本最低：

| 组 | 相机相关 loss | 要回答的问题 |
|---|---|---|
| A | `camera_loss`（原版） | baseline |
| B | `geodesic_camera_loss` | 只换度量有没有用？（预计提升很小，不是卖点） |
| C | B + `relative_motion_loss` | 1 阶的贡献 |
| D | C + `curve_accel_loss` | **核心：2 阶是否带来额外的轨迹一致性** |

### 要看的指标

| 指标 | 衡量什么 | 预期 |
|---|---|---|
| ATE（绝对轨迹误差） | 0 阶 | 不一定变 |
| RPE 平移 / 旋转（相对位姿误差） | 1 阶 | C、D 应该更好 |
| **抖动指标**：预测与真值的 `ang_accel` / `lin_accel` 之差 | 2 阶 | **D 应该明显更好** |
| online / chunk 模式下 chunk 边界处的位姿跳变 | 2 阶 | D 应该更好 |
| 深度 / 点云指标 | 不应变差 | 持平即可 |

抖动指标可以直接用 `trajectory_curvature_stats(pose)` 算。

**最可能的结果**：ATE 持平，RPE 和抖动明显改善。写论文时故事要围绕"一致性"，不要围绕 ATE。

### 要调的超参数

| 参数 | 位置 | 初值 | 建议范围 |
|---|---|---|---|
| `relative_motion_loss` 权重 | config | 1.0 | 0.3–3 |
| `curve_accel_loss` 权重 | config | 0.5 | 0.1–2 |
| `trans_weight`（平移 vs 旋转） | 函数参数 | 1.0 | 0.3–3 |

---

## 6.4 后续方向（按推荐顺序）

### ① 不训练也能做的验证（纯 CPU，最先做）

在 benchmark 分支上拿到已有的预测轨迹，对它做 **SE(3) 测地样条平滑**（一种"在弯曲空间里拟合光滑曲线"的方法），看 RPE 能改善多少。
这能粗略估计"2 阶约束"收益的**上限**，再决定值不值得花 GPU 训练。

### ② 弹性曲线度量（novelty 的加强版）

现在的 2 阶 loss 隐含假设"预测和真值在同一组帧上比"。弹性度量（SRVF：先把曲线变换成 $q(s) = \xi(s) / \sqrt{\lVert \xi(s) \rVert}$，再用普通 L² 距离比较）对"曲线走得快慢"不敏感，理论上更优雅。
需要改数据侧：让 `sample_sequence` 把帧号一起返回。所有 dataset 共用这个函数，改动集中在一处。

### ③ 流形上的输出头（模型改动，风险最大）

让 camera head 不直接输出每帧位姿，而是输出相邻帧的运动 $\xi_i$，再用 $T_{i+1} = T_i \cdot \mathrm{Exp}(\xi_i)$ 一步步积分出位姿。
这样轨迹的连续性由**结构本身**保证。代价是误差会累积，需要配合 0 阶 loss。

### ④ 更完整的 SE(3) 度量

现在用的是 SO(3)×ℝ³ 积度量，旋转和平移分开算。SE(3) 上还有把两者耦合的左不变度量，此时 3.3 节的修正项 $\mathrm{ad}^\dagger_\xi \xi$ 不再为 0，需要显式计算。属于理论上更完整的版本。

---

## 6.5 已知局限

- **代码没在 torch 上跑过**（作者本地无 GPU、没装 torch）。shape 和约定是人工核对的。第一次跑请先执行 `python docs/riemannian/smoke_test.py`。
- `so3_log` 在转角接近 180° 时不稳定。1、2 阶只处理相邻帧的小转动，不受影响；0 阶和原版有同样的局限。
- 平移项沿用原版的 L1 + `clamp(max=100)`，保证新旧 loss 量级可比；权重未调。
- 平移部分的"加速度"是平坦近似（第 3.3 节）。

[← 上一章](05_figures_and_evidence.md) · [目录](README.md) · [术语表 →](glossary.md)
