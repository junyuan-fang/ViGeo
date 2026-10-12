# Ablation of vigeo_stage1_224x224 with Riemannian camera-trajectory losses.
# Only loss_cfg differs; see docs/riemannian/README.md for the ablation table.
_base_ = ['./vigeo_stage1_224x224.py']

loss_cfg = dict(
    _delete_=True,
    losses=dict(
        scale_invariant_pointmap_loss=1.0,
        geodesic_camera_loss=1.0,  # replaces camera_loss (R^7 L1 -> SO(3) geodesic)
        relative_motion_loss=1.0,  # 1st order: body velocity matching
        curve_accel_loss=0.5,      # 2nd order: covariant acceleration matching
        normal_loss=1.0,
        ray_loss=1.0,
    )
)
