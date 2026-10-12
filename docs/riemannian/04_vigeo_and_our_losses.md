# 第 4 章 ViGeo 的位姿从哪来，新 loss 怎么接进去（对照代码）

[← 上一章](03_curves_on_manifolds.md) · [目录](README.md) · [下一章：三个小实验 →](05_figures_and_evidence.md)

> 本章目标：沿着数据流走一遍，从网络输出位姿，到 loss 里怎么比较，再到新代码的每一行在做什么。
> 读完你应该能独立修改、调试这部分代码。

---

## 4.1 位姿是怎么预测出来的

```
视频 [B, S, 3, H, W]
  → DINOv2 编码器（帧间 attention）          src/model/vigeo_train.py
  → 每帧一个 camera token [B, S, 2C]
  → CameraHead：两层 MLP + 两个线性层        src/model/camera_head.py
  → pose_pred [B, S, 7] = [t(3), q(4)]        四元数顺序 xyzw
```

`CameraHead` 的核心只有三行：

```python
feat = self.backbone(feat)       # 两层 Linear + ReLU
out_t = self.fc_t(feat)          # 平移 (3)
out_qvec = self.fc_qvec(feat)    # 四元数 (4)，没有归一化
```

注意两点：
- **每帧独立**：MLP 对每个 token 单独作用。帧间信息只来自编码器里的 attention，输出端没有任何结构保证轨迹连贯。
- **四元数没归一化**：后面 `pose_encoding_to_extri → quat_to_mat` 会隐式除以 |q|²，所以模长不影响旋转矩阵。

---

## 4.2 loss 里位姿经历了什么

入口：`src/loss/video_depth_loss.py :: VideoDepthLoss._forward_single_sample`，对 batch 里每个样本单独计算。

| 步骤 | 代码 | 做了什么 | 为什么 |
|---|---|---|---|
| ① | `pose_encoding_to_extri(pose_pred)` | `[t, q]` → `[S, 3, 4]` 矩阵 | 统一成矩阵形式 |
| ② | `normalize_gt(...)` | 真值变到**第 0 帧相机坐标系**，再按点云平均距离缩放 | 消除世界坐标系和尺度的任意性 |
| ③ | `normalize_pred(...)` | 预测按自己的点云平均距离缩放 | 同上（不做坐标系变换：网络被训练成直接以第 0 帧为原点输出） |
| ④ | `compute_pointmap_scale` → `scale_i` | 求一个标量，让预测点云尺度对齐真值 | 单目深度只有相对尺度 |
| ⑤ | `pose_pred_i_norm[:, :3, 3] *= scale_i` | 预测的平移乘上这个尺度 | 平移和点云共享尺度 |
| ⑥ | 进入各个 loss 分支 | | |

得到的两个张量：
- `pose_pred_i_norm_scaled`：`[S, 3, 4]`，camera-to-world
- `pose_gt_i_norm`：`[S, 4, 4]`，camera-to-world

新 loss 只用 `[..., :3, :3]`（R）和 `[..., :3, 3]`（t），所以两种形状都能处理。

---

## 4.3 原来的 camera_loss

```python
# src/loss/loss_util.py
def camera_loss(pose_pred, pose_gt):       # 输入是 [t(3), q(4)] 编码
    loss_t = (pose_pred[..., :3] - pose_gt[..., :3]).abs()
    loss_r = (pose_pred[..., 3:7] - pose_gt[..., 3:7]).abs()   # ← 四元数直接相减
    return loss_t.clamp(max=100).mean() + loss_r.mean()
```

调用前会先 `extri_to_pose_encoding`（矩阵 → 四元数，并强制 w ≥ 0）。问题见第 2.5 节：
算的是"弦"不是"弧"，和转轴方向有关，在 w ≈ 0 附近会符号跳变，而且**逐帧独立**。

---

## 4.4 新代码逐段讲解：`src/loss/riemannian.py`

文件分三块：几何工具、三个 loss、诊断函数。

### (a) 几何工具

| 函数 | 数学 | 章节 |
|---|---|---|
| `so3_log(R)` | $\mathrm{Log}: SO(3) \to \mathbb{R}^3$，输出轴角 ω | 2.3、2.5 |
| `so3_geodesic_angle(R1, R2)` | $\lVert \mathrm{Log}(R_1^\top R_2) \rVert$，两个朝向相差的角度 | 2.4 |
| `relative_twist(pose)` | $\xi_i = (\mathrm{Log}(R_i^\top R_{i+1}),\ R_i^\top(t_{i+1}-t_i))$ | 2.6、3.2 |

`relative_twist` 输入 `[S, ·, 4]`，输出两个 `[S-1, 3]`，一次算出所有相邻帧对，没有 for 循环。

### (b) 0 阶：`geodesic_camera_loss`

```python
def geodesic_camera_loss(pose_pred, pose_gt, trans_weight=1.0):
    loss_r = so3_geodesic_angle(pose_pred[..., :3, :3], pose_gt[..., :3, :3])   # 每帧：差几度（弧度）
    loss_t = (pose_pred[..., :3, 3] - pose_gt[..., :3, 3]).abs().sum(-1)       # 每帧：位置差（L1）
    return loss_r.mean() + trans_weight * loss_t.clamp(max=100).mean()
```

和原版的唯一区别：旋转项从"四元数 L1"换成了"真实角度差"。平移项保持不变。**这一项本身不是 novelty**，只是把地基打正。

### (c) 1 阶：`relative_motion_loss`

```python
_, v_pred = relative_twist(pose_pred)            # 预测：每一步走了多远（body frame）
_, v_gt = relative_twist(pose_gt)                # 真值：同上
dR_pred = R_pred[:-1]ᵀ @ R_pred[1:]              # 预测：每一步转了多少（旋转矩阵）
dR_gt   = R_gt[:-1]ᵀ   @ R_gt[1:]                # 真值：同上
loss_r = so3_geodesic_angle(dR_pred, dR_gt)      # 两个"转动"差几度
loss_t = (v_pred - v_gt).abs().sum(-1)           # 两个"位移"差多少
```

**细节**：旋转部分比较的是 $\mathrm{Log}(\Delta R_{\text{pred}}^\top \Delta R_{\text{gt}})$，而不是 $\omega_{\text{pred}} - \omega_{\text{gt}}$。
两者在误差小时几乎一样，但前者在转角接近 180° 时也正确，后者会遇到轴角表示的不连续点（割迹）。

### (d) 2 阶：`curve_accel_loss`

```python
omega_pred, v_pred = relative_twist(pose_pred)   # 预测速度 ξ_i
omega_gt, v_gt = relative_twist(pose_gt)         # 真值速度 ξ_i
acc_r = (omega_pred.diff(0) - omega_gt.diff(0)).norm(dim=-1)   # 旋转加速度之差（L2）
acc_t = (v_pred.diff(0) - v_gt.diff(0)).abs().sum(-1)          # 平移加速度之差（L1）
```

这里直接用 $\omega$ 相减（`.diff`），原因见 3.3 节：body frame 下所有速度都在同一个空间里，而且对旋转部分这就是精确的协变加速度。
相邻帧的转角很小，离 180° 的不连续点很远，所以直接相减是安全的。

### (e) 边界情况

| 情况 | 处理 |
|---|---|
| S < 2（单帧） | 1 阶返回 `pose_pred.sum() * 0`（可以反传梯度的 0） |
| S < 3 | 2 阶同上 |
| 角度 → 0 | `atan2` + `eps`，梯度有限（2.5 节） |
| 混合精度 | 调用处用 `autocast(float32)` 包住，输入也 `.float()` |

### (f) 诊断：`trajectory_curvature_stats(pose)`

不参与训练，用来分析：输入一条轨迹，返回平均角速度、线速度、角加速度、线加速度。
可以在 benchmark 上比较"预测轨迹的抖动程度"和"真值轨迹的抖动程度"。

---

## 4.5 接入点：`src/loss/video_depth_loss.py`（+20 行）

```python
from .riemannian import geodesic_camera_loss, relative_motion_loss, curve_accel_loss

TRAJECTORY_LOSSES = {                      # 配置里的名字 → 函数
    'geodesic_camera_loss': geodesic_camera_loss,
    'relative_motion_loss': relative_motion_loss,
    'curve_accel_loss': curve_accel_loss,
}
...
            elif loss_name in TRAJECTORY_LOSSES:
                pose_pred_i_norm_scaled = pose_pred_i_norm.clone()
                pose_pred_i_norm_scaled[:, :3, 3] *= scale_i           # 与 camera_loss 相同的尺度对齐
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    sample_losses[loss_name] = TRAJECTORY_LOSSES[loss_name](
                        pose_pred_i_norm_scaled.float(), pose_gt_i_norm.float())
```

`for loss_name in self.losses` 只会遍历配置里写了的名字，所以**不写就不启用**，原版行为不受影响。

---

## 4.6 配置：`configs/video_depth/vigeo_stage1_224x224_riemannian.py`

```python
_base_ = ['./vigeo_stage1_224x224.py']     # 继承 stage1 的所有设置
loss_cfg = dict(
    _delete_=True,                         # mmengine 语法：整个替换 loss_cfg，而不是合并
    losses=dict(
        scale_invariant_pointmap_loss=1.0,
        geodesic_camera_loss=1.0,          # 替代 camera_loss
        relative_motion_loss=1.0,
        curve_accel_loss=0.5,
        normal_loss=1.0,
        ray_loss=1.0,
    )
)
```

权重是拍脑袋定的初值，没有调过。

---

## 4.7 第一次运行前：CPU 冒烟测试

```bash
pip install torch numpy        # CPU 版即可
python docs/riemannian/smoke_test.py
```

[`smoke_test.py`](smoke_test.py) 直接从文件加载 `riemannian.py`，不会触发 `src/__init__.py` 里的重依赖。它检查 6 件事：

1. `so3_log` 能还原已知的轴角
2. 预测 = 真值时，三个 loss 都是 0
3. 整体刚体变换后，1、2 阶仍然是 0（gauge 不变性）
4. 每帧抖动时，2 阶 loss 最大
5. 反向传播梯度有限，包括预测 = 真值这个最容易出 NaN 的情况
6. `[S,3,4]` 和 `[S,4,4]` 输入结果一致

> ⚠️ 作者本地没有 torch，这个脚本**也还没跑过**。如果失败，请把输出贴回来。

[← 上一章](03_curves_on_manifolds.md) · [目录](README.md) · [下一章：三个小实验 →](05_figures_and_evidence.md)
