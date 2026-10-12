"""
R1 / R0 CPU check: the math behind src/loss/riemannian.py (pure-Python port in geom.py).

    python docs/riemannian/cpu_checks/check_r1_losses.py
"""

import math
import random

from geom import (apply_global, curve_accel_loss, exp_so3, geodesic_camera_loss, mat_mul, mat_to_quat,
                  mat_T, norm, perturb, relative_motion_loss, smooth_trajectory, so3_log, vsub)

FAILED = []


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def quat_to_mat(q):
    """Port of src/loss/utils.py::quat_to_mat."""
    i, j, k, r = q
    two_s = 2.0 / sum(x * x for x in q)
    return [[1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
            [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
            [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)]]


def max_abs_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(3) for j in range(3))


def losses(pred, gt):
    return geodesic_camera_loss(pred, gt), relative_motion_loss(pred, gt), curve_accel_loss(pred, gt)


rng = random.Random(0)

# 1. mat_to_quat / quat_to_mat round trip, and so3_log inverts Exp (incl. tiny and near-pi angles)
worst_rt, worst_log = 0.0, 0.0
for _ in range(500):
    axis = [rng.gauss(0, 1) for _ in range(3)]
    ang = rng.choice([1e-7, 1e-3, rng.uniform(0, math.pi * 0.98)])
    w = [a / norm(axis) * ang for a in axis]
    R = exp_so3(w)
    worst_rt = max(worst_rt, max_abs_diff(quat_to_mat(mat_to_quat(R)), R))
    worst_log = max(worst_log, norm(vsub(so3_log(R), w)))
check('mat_to_quat -> quat_to_mat round trip', worst_rt < 1e-9, f'max err {worst_rt:.1e}')
check('so3_log inverts Exp for angles in [1e-7, 0.98*pi]', worst_log < 1e-6, f'max err {worst_log:.1e}')

# 2. pred == GT -> all three losses are 0
for kind in ('arc', 'orbit', 'sharp_turn'):
    gt = smooth_trajectory(24, kind)
    vals = losses(gt, gt)
    check(f'pred == GT -> all losses 0 [{kind}]', max(vals) < 1e-6, ', '.join(f'{v:.1e}' for v in vals))

# 3. gauge invariance
gt = smooth_trajectory(24, 'arc')
G = (exp_so3([0.4, -0.3, 0.9]), [1.0, -2.0, 0.5])
l0, l1, l2 = losses(apply_global(gt, G), gt)
check('global rigid transform: 0th > 0, 1st = 2nd = 0', l0 > 0.1 and l1 < 1e-6 and l2 < 1e-6,
      f'0th={l0:.3f}, 1st={l1:.1e}, 2nd={l2:.1e}')

# 4. jitter is seen most by the 2nd order (statistically, over 200 seeds)
wins, ratios = 0, []
for seed in range(200):
    r = random.Random(seed)
    pred = perturb(gt, r, rot_sigma=0.02, trans_sigma=0.02)
    l0, l1, l2 = losses(pred, gt)
    wins += l0 < l1 < l2
    ratios.append(l2 / l0)
check('per-frame jitter: 0th < 1st < 2nd', wins / 200 > 0.95,
      f'{wins}/200 seeds, mean 2nd/0th = {sum(ratios) / len(ratios):.2f}')

# 4b. velocity drift is seen by the 1st order, barely by the 2nd
pred = perturb(gt, rng, drift_rot=0.01, drift_scale=1.05)
l0, l1, l2 = losses(pred, gt)
check('velocity drift: 1st >> 2nd', l1 > 5 * l2, f'0th={l0:.3f}, 1st={l1:.3f}, 2nd={l2:.4f}')

# 5. finite-difference directional derivatives stay bounded (no gradient blow-up)
#    including pred == GT, where the losses behave like |x| (kink, but finite slope)
def directional_slope(pred, gt, fn, eps=1e-6, trials=30):
    base = fn(pred, gt)
    worst = 0.0
    for _ in range(trials):
        d = [[rng.gauss(0, 1) for _ in range(6)] for _ in pred]
        moved = [(mat_mul(R, exp_so3([eps * x for x in di[:3]])), [t[k] + eps * di[3 + k] for k in range(3)])
                 for (R, t), di in zip(pred, d)]
        dn = math.sqrt(sum(x * x for di in d for x in di))
        worst = max(worst, abs(fn(moved, gt) - base) / (eps * dn))
    return worst


for label, pred in (('pred == GT', gt), ('jittered', perturb(gt, rng, 0.02, 0.02))):
    slopes = [directional_slope(pred, gt, fn) for fn in (geodesic_camera_loss, relative_motion_loss, curve_accel_loss)]
    check(f'bounded directional slopes [{label}]', all(math.isfinite(s) and s < 10 for s in slopes),
          ', '.join(f'{s:.2f}' for s in slopes))

# 6. continuity: baseline quaternion-L1 jumps where the geodesic loss does not
def quat_l1(R1, R2):
    return sum(abs(a - b) for a, b in zip(mat_to_quat(R1), mat_to_quat(R2)))


R_gt = exp_so3([0.0, 0.0, math.radians(150)])
axis = [0.0, 0.0, 1.0]  # same axis as GT: total rotation sweeps 150 -> 330 deg, crossing 180 (w = 0)
prev_g = prev_q = None
max_jump_g = max_jump_q = 0.0
for k in range(0, 721):
    th = math.radians(k * 0.25)
    R_pred = mat_mul(R_gt, exp_so3([a * th for a in axis]))
    g, q = norm(so3_log(mat_mul(mat_T(R_pred), R_gt))), quat_l1(R_pred, R_gt)
    if prev_g is not None:
        max_jump_g, max_jump_q = max(max_jump_g, abs(g - prev_g)), max(max_jump_q, abs(q - prev_q))
    prev_g, prev_q = g, q
step = math.radians(0.25)
check('geodesic loss is continuous (max jump ~ step size)', max_jump_g < 1.01 * step,
      f'max jump {max_jump_g:.4f} vs step {step:.4f}')
check('baseline quaternion L1 jumps (expected: this is the bug being fixed)', max_jump_q > 100 * step,
      f'max jump {max_jump_q:.3f} for a 0.25 deg change')

print('\nR1 summary:', 'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
