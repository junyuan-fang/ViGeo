"""
Riemannian (Lie-group) camera-trajectory losses.

The baseline camera_loss compares [t, q_xyzw] with L1 in R^7, i.e. it treats
SO(3) as a subset of R^4 and each frame independently. Here we instead view
the predicted camera poses of one clip as a discrete curve on SE(3):

    gamma: {0, ..., S-1} -> SE(3),   gamma(i) = T_i  (camera-to-world)

and compare it with the GT curve at three orders:

    0th order  geodesic_camera_loss   d(T_i^pred, T_i^gt)            (per frame)
    1st order  relative_motion_loss   xi_i = Log(T_i^-1 T_{i+1})     (body velocity)
    2nd order  curve_accel_loss       a_i  = xi_{i+1} - xi_i         (acceleration)

Metric: left-invariant product metric on SO(3) x R^3 in the body frame,
    ||(omega, v)||^2 = ||omega||^2 + trans_weight^2 * ||v||^2.
For the rotation part this is the bi-invariant metric, so the left-trivialized
second difference a_i coincides with the discrete covariant acceleration.

Design notes:
    - Only GT-matched terms (no pure smoothness prior). Training clips are
      sampled with random strides (1..max_interval), reflect at sequence ends,
      and are shuffled with prob 0.3 (see BaseDataset.sample_sequence). Since
      pred and GT share the same indexing, xi / a are compared on the same
      discrete parametrization, so no frame timestamps are needed.
    - All terms are invariant to a global rigid transform of the trajectory
      except geodesic_camera_loss, matching the gauge of normalize_gt.
    - Inputs are [S, 3|4, 4] camera-to-world matrices whose translations are
      already in the same scale (see the camera_loss branch in VideoDepthLoss).
"""

import torch

from .utils import mat_to_quat


# -----------------------------------------------------------------------------
# SO(3) / SE(3) primitives
# -----------------------------------------------------------------------------


def so3_log(R: torch.Tensor, eps: float = 1e-12):
    """Logarithm map SO(3) -> so(3) ~ R^3 (axis-angle), gradient-safe near identity.

    Args:
        R: [..., 3, 3]

    Returns:
        omega: [..., 3], with ||omega|| = rotation angle in [0, pi]
    """
    quat = mat_to_quat(R)  # [..., 4], XYZW with w >= 0
    v, w = quat[..., :3], quat[..., 3]
    v_norm = torch.sqrt(v.square().sum(-1) + eps)
    theta = 2.0 * torch.atan2(v_norm, w)
    return v * (theta / v_norm)[..., None]


def so3_geodesic_angle(R1: torch.Tensor, R2: torch.Tensor):
    """Geodesic distance on SO(3): ||Log(R1^T R2)||, in radians. Shape [...]."""
    return so3_log(R1.transpose(-1, -2) @ R2).norm(dim=-1)


def relative_twist(pose: torch.Tensor):
    """Body-frame velocity of a discrete SE(3) curve.

    Args:
        pose: [S, 3|4, 4] camera-to-world

    Returns:
        omega: [S-1, 3]  Log(R_i^T R_{i+1})
        v:     [S-1, 3]  R_i^T (t_{i+1} - t_i)
    """
    R, t = pose[..., :3, :3], pose[..., :3, 3]
    R_prev_T = R[:-1].transpose(-1, -2)
    omega = so3_log(R_prev_T @ R[1:])
    v = torch.einsum('sij,sj->si', R_prev_T, t[1:] - t[:-1])
    return omega, v


# -----------------------------------------------------------------------------
# Losses (0th / 1st / 2nd order)
# -----------------------------------------------------------------------------


def geodesic_camera_loss(pose_pred: torch.Tensor, pose_gt: torch.Tensor, trans_weight: float = 1.0):
    """Drop-in replacement for camera_loss: geodesic rotation angle + L1 translation."""
    loss_r = so3_geodesic_angle(pose_pred[..., :3, :3], pose_gt[..., :3, :3])
    loss_t = (pose_pred[..., :3, 3] - pose_gt[..., :3, 3]).abs().sum(-1)
    return loss_r.mean() + trans_weight * loss_t.clamp(max=100).mean()


def relative_motion_loss(pose_pred: torch.Tensor, pose_gt: torch.Tensor, trans_weight: float = 1.0):
    """1st order: match body velocities xi_i = Log(T_i^-1 T_{i+1}) of pred and GT curves."""
    if pose_pred.shape[0] < 2:
        return pose_pred.sum() * 0
    _, v_pred = relative_twist(pose_pred)
    _, v_gt = relative_twist(pose_gt)
    # Compare relative rotations on SO(3) (not in so(3)) to avoid the cut locus at pi.
    dR_pred = pose_pred[:-1, :3, :3].transpose(-1, -2) @ pose_pred[1:, :3, :3]
    dR_gt = pose_gt[:-1, :3, :3].transpose(-1, -2) @ pose_gt[1:, :3, :3]
    loss_r = so3_geodesic_angle(dR_pred, dR_gt)
    loss_t = (v_pred - v_gt).abs().sum(-1)
    return loss_r.mean() + trans_weight * loss_t.clamp(max=100).mean()


def curve_accel_loss(pose_pred: torch.Tensor, pose_gt: torch.Tensor, trans_weight: float = 1.0):
    """2nd order: match discrete covariant accelerations a_i = xi_{i+1} - xi_i.

    Penalizes jitter / kinks in the predicted trajectory *relative to* the GT
    curve, rather than pushing towards constant velocity.
    """
    if pose_pred.shape[0] < 3:
        return pose_pred.sum() * 0
    omega_pred, v_pred = relative_twist(pose_pred)
    omega_gt, v_gt = relative_twist(pose_gt)
    acc_r = (omega_pred.diff(dim=0) - omega_gt.diff(dim=0)).norm(dim=-1)
    acc_t = (v_pred.diff(dim=0) - v_gt.diff(dim=0)).abs().sum(-1)
    return acc_r.mean() + trans_weight * acc_t.clamp(max=100).mean()


# -----------------------------------------------------------------------------
# Diagnostics (no gradient, for logging / analysis only)
# -----------------------------------------------------------------------------


@torch.no_grad()
def trajectory_curvature_stats(pose: torch.Tensor):
    """Mean angular / linear speed and acceleration magnitude of one trajectory.

    Useful to compare how "rough" pred vs GT curves are on a benchmark clip.
    """
    if pose.shape[0] < 3:
        return {}
    omega, v = relative_twist(pose.float())
    return {
        'ang_speed': omega.norm(dim=-1).mean().item(),
        'lin_speed': v.norm(dim=-1).mean().item(),
        'ang_accel': omega.diff(dim=0).norm(dim=-1).mean().item(),
        'lin_accel': v.diff(dim=0).norm(dim=-1).mean().item(),
    }
