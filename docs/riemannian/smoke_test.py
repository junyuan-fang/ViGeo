"""
CPU smoke test for src/loss/riemannian.py. Needs only torch + numpy (no GPU, no checkpoints).

    python docs/riemannian/smoke_test.py

Loads src/loss/{utils,riemannian}.py directly so the heavy src/__init__.py
(datasets, utils3d, ...) is not imported.

Checks:
    1. so3_log recovers known axis-angle rotations
    2. pred == GT  ->  all three losses are 0
    3. gauge invariance: a global rigid transform leaves 1st/2nd-order losses at 0
    4. per-frame jitter raises the 2nd-order loss the most (cf. 05_figures_and_evidence.md, fig 2)
    5. backward gives finite gradients, including the near-identity case
    6. pred as [S, 3, 4] and GT as [S, 4, 4] (the shapes used in VideoDepthLoss)
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LOSS_DIR = ROOT / 'src' / 'loss'

pkg = types.ModuleType('rloss')
pkg.__path__ = [str(LOSS_DIR)]
sys.modules['rloss'] = pkg
for name in ('utils', 'riemannian'):
    spec = importlib.util.spec_from_file_location(f'rloss.{name}', LOSS_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[f'rloss.{name}'] = module
    spec.loader.exec_module(module)
rm = sys.modules['rloss.riemannian']

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def exp_so3(omega):
    """Rodrigues: axis-angle [..., 3] -> rotation matrix [..., 3, 3]."""
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = omega / theta
    K = torch.zeros(omega.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    return torch.eye(3) + th.sin() * K + (1 - th.cos()) * K @ K


def make_pose(R, t):
    T = torch.eye(4).repeat(R.shape[0], 1, 1)
    T[:, :3, :3], T[:, :3, 3] = R, t
    return T


def smooth_trajectory(S=12):
    s = torch.linspace(0, 1, S)
    omega = torch.stack([0.3 * s, 0.8 * s, 0.1 * torch.sin(3 * s)], -1)
    t = torch.stack([2 * s, 0.5 * torch.sin(4 * s), 0.1 * s], -1)
    return make_pose(exp_so3(omega), t)


def losses(pred, gt):
    return [fn(pred, gt).item() for fn in (rm.geodesic_camera_loss, rm.relative_motion_loss, rm.curve_accel_loss)]


def check(name, ok):
    print(f'[{"PASS" if ok else "FAIL"}] {name}')
    if not ok:
        check.failed = True


check.failed = False

# 1. so3_log recovers axis-angle
omega = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.0, 2.5, 0.0], [1.0, 1.0, 1.0]])
rec = rm.so3_log(exp_so3(omega))
check('so3_log inverts Exp', torch.allclose(rec, omega, atol=1e-6))

# 2. pred == GT
gt = smooth_trajectory()
check('pred == GT -> all losses 0', max(losses(gt, gt)) < 1e-6)

# 3. gauge invariance
G = make_pose(exp_so3(torch.tensor([[0.4, -0.3, 0.9]])), torch.tensor([[1.0, -2.0, 0.5]]))[0]
moved = G @ gt
l0, l1, l2 = losses(moved, gt)
check(f'global transform: 0th={l0:.3f} > 0, 1st={l1:.1e} ~ 0, 2nd={l2:.1e} ~ 0', l0 > 0.1 and l1 < 1e-6 and l2 < 1e-6)

# 4. jitter hits 2nd order hardest
noise_R = exp_so3(0.02 * torch.randn(gt.shape[0], 3))
jitter = make_pose(gt[:, :3, :3] @ noise_R, gt[:, :3, 3] + 0.02 * torch.randn(gt.shape[0], 3))
l0, l1, l2 = losses(jitter, gt)
check(f'jitter: 0th={l0:.3f} < 1st={l1:.3f} < 2nd={l2:.3f}', l0 < l1 < l2)

# 5. finite gradients (incl. pred == GT, the near-identity case of so3_log)
for label, base in (('pred == GT', gt), ('jittered', jitter)):
    pred = base.clone().requires_grad_(True)
    total = sum(fn(pred, gt) for fn in (rm.geodesic_camera_loss, rm.relative_motion_loss, rm.curve_accel_loss))
    total.backward()
    check(f'finite gradients ({label})', torch.isfinite(pred.grad).all().item())

# 6. [S, 3, 4] pred vs [S, 4, 4] GT
check('[S,3,4] pred vs [S,4,4] GT', abs(losses(jitter[:, :3, :], gt)[2] - losses(jitter, gt)[2]) < 1e-9)

print('diagnostics (GT):', rm.trajectory_curvature_stats(gt))
sys.exit(1 if check.failed else 0)
