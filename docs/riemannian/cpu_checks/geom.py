"""
Pure-Python (standard library only) port of the geometry in src/loss/riemannian.py
and src/loss/utils.py, used by the CPU checks in this folder.

Each function mirrors its torch counterpart line by line so that the checks
validate the *math and design*; they do not exercise the torch code itself
(that is what docs/riemannian/smoke_test.py is for, once torch is available).

Conventions (same as the repo):
    - quaternions are XYZW, standardized to w >= 0
    - poses are camera-to-world, represented here as (R, t) with R a 3x3 list
"""

import math
import random

# -----------------------------------------------------------------------------
# Small linear algebra
# -----------------------------------------------------------------------------


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_T(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vscale(a, s):
    return [x * s for x in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(a):
    return math.sqrt(dot(a, a))


def eye():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def angle_between(a, b):
    """Stable angle between 3-vectors: atan2(|a x b|, a . b)."""
    return math.atan2(norm(cross(a, b)), dot(a, b))


# -----------------------------------------------------------------------------
# SO(3)
# -----------------------------------------------------------------------------


def exp_so3(w):
    """Rodrigues formula: axis-angle -> rotation matrix."""
    th = norm(w)
    if th < 1e-12:
        return eye()
    k = vscale(w, 1.0 / th)
    K = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    KK = mat_mul(K, K)
    s, c = math.sin(th), 1.0 - math.cos(th)
    return [[(1.0 if i == j else 0.0) + s * K[i][j] + c * KK[i][j] for j in range(3)] for i in range(3)]


def _sqrt_positive_part(x):
    return math.sqrt(x) if x > 0 else 0.0


def mat_to_quat(M):
    """Port of src/loss/utils.py::mat_to_quat (incl. branch choice and w >= 0)."""
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]
    q_abs = [_sqrt_positive_part(1.0 + m00 + m11 + m22), _sqrt_positive_part(1.0 + m00 - m11 - m22),
             _sqrt_positive_part(1.0 - m00 + m11 - m22), _sqrt_positive_part(1.0 - m00 - m11 + m22)]
    rows = [
        [q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2],
    ]
    k = max(range(4), key=lambda i: q_abs[i])
    denom = 2.0 * max(q_abs[k], 0.1)
    r, i, j, kk = [x / denom for x in rows[k]]
    quat = [i, j, kk, r]                         # XYZW
    return [-x for x in quat] if quat[3] < 0 else quat


def so3_log(R, eps=1e-12):
    """Port of riemannian.so3_log."""
    q = mat_to_quat(R)
    v, w = q[:3], q[3]
    v_norm = math.sqrt(dot(v, v) + eps)
    theta = 2.0 * math.atan2(v_norm, w)
    return vscale(v, theta / v_norm)


def so3_geodesic_angle(R1, R2):
    return norm(so3_log(mat_mul(mat_T(R1), R2)))


# -----------------------------------------------------------------------------
# SE(3) curves and the three losses (ports of riemannian.py)
# -----------------------------------------------------------------------------


def compose(T1, T2):
    """(R1, t1) * (R2, t2)."""
    (R1, t1), (R2, t2) = T1, T2
    return (mat_mul(R1, R2), vadd(mat_vec(R1, t2), t1))


def relative_twist(traj):
    """Body velocities xi_i = (Log(R_i^T R_{i+1}), R_i^T (t_{i+1} - t_i))."""
    omegas, vs = [], []
    for (R0, t0), (R1, t1) in zip(traj[:-1], traj[1:]):
        R0T = mat_T(R0)
        omegas.append(so3_log(mat_mul(R0T, R1)))
        vs.append(mat_vec(R0T, vsub(t1, t0)))
    return omegas, vs


def _mean(xs):
    return sum(xs) / max(1, len(xs))


def l1(v):
    return sum(abs(x) for x in v)


def geodesic_camera_loss(pred, gt, trans_weight=1.0):
    loss_r = [so3_geodesic_angle(p[0], g[0]) for p, g in zip(pred, gt)]
    loss_t = [min(l1(vsub(p[1], g[1])), 100.0) for p, g in zip(pred, gt)]
    return _mean(loss_r) + trans_weight * _mean(loss_t)


def relative_motion_loss(pred, gt, trans_weight=1.0):
    if len(pred) < 2:
        return 0.0
    _, v_pred = relative_twist(pred)
    _, v_gt = relative_twist(gt)
    loss_r = []
    for i in range(len(pred) - 1):
        dR_pred = mat_mul(mat_T(pred[i][0]), pred[i + 1][0])
        dR_gt = mat_mul(mat_T(gt[i][0]), gt[i + 1][0])
        loss_r.append(so3_geodesic_angle(dR_pred, dR_gt))
    loss_t = [min(l1(vsub(a, b)), 100.0) for a, b in zip(v_pred, v_gt)]
    return _mean(loss_r) + trans_weight * _mean(loss_t)


def curve_accel_loss(pred, gt, trans_weight=1.0):
    if len(pred) < 3:
        return 0.0
    w_p, v_p = relative_twist(pred)
    w_g, v_g = relative_twist(gt)
    acc_r, acc_t = [], []
    for i in range(len(w_p) - 1):
        acc_r.append(norm(vsub(vsub(w_p[i + 1], w_p[i]), vsub(w_g[i + 1], w_g[i]))))
        acc_t.append(min(l1(vsub(vsub(v_p[i + 1], v_p[i]), vsub(v_g[i + 1], v_g[i]))), 100.0))
    return _mean(acc_r) + trans_weight * _mean(acc_t)


# -----------------------------------------------------------------------------
# Trajectory metrics (for R2)
# -----------------------------------------------------------------------------


def ate(pred, gt):
    """Mean translation error (both trajectories already in the same frame)."""
    return _mean([norm(vsub(p[1], g[1])) for p, g in zip(pred, gt)])


def rpe(pred, gt):
    """Mean relative rotation error (deg) and relative translation error over adjacent frames."""
    w_p, v_p = relative_twist(pred)
    w_g, v_g = relative_twist(gt)
    rot = []
    for i in range(len(pred) - 1):
        dR_pred = mat_mul(mat_T(pred[i][0]), pred[i + 1][0])
        dR_gt = mat_mul(mat_T(gt[i][0]), gt[i + 1][0])
        rot.append(math.degrees(so3_geodesic_angle(dR_pred, dR_gt)))
    trans = [norm(vsub(a, b)) for a, b in zip(v_p, v_g)]
    return _mean(rot), _mean(trans)


def jitter(traj):
    """Mean acceleration magnitude (rotation in deg, translation) of a trajectory."""
    w, v = relative_twist(traj)
    ang = [math.degrees(norm(vsub(w[i + 1], w[i]))) for i in range(len(w) - 1)]
    lin = [norm(vsub(v[i + 1], v[i])) for i in range(len(v) - 1)]
    return _mean(ang), _mean(lin)


# -----------------------------------------------------------------------------
# Synthetic trajectories
# -----------------------------------------------------------------------------


def smooth_trajectory(S=24, kind='arc'):
    """A smooth GT camera path. kind in {'arc', 'orbit', 'sharp_turn'}."""
    traj = []
    for i in range(S):
        s = i / (S - 1)
        if kind == 'arc':
            w = [0.3 * s, 0.8 * s, 0.1 * math.sin(3 * s)]
            t = [2.0 * s, 0.5 * math.sin(4 * s), 0.1 * s]
        elif kind == 'orbit':                       # look-at-centre orbit, large total rotation
            a = 2.4 * s
            w = [0.0, -a, 0.0]
            t = [2.0 * math.sin(a), 0.2 * math.sin(3 * s), 2.0 - 2.0 * math.cos(a)]
        elif kind == 'sharp_turn':                  # straight, then a fast 60 deg yaw in a few frames
            yaw = math.radians(60) / (1 + math.exp(-40 * (s - 0.5)))
            w = [0.0, yaw, 0.0]
            t = [0.1 * i * math.sin(yaw), 0.0, 0.1 * i * math.cos(yaw)]
        else:
            raise ValueError(kind)
        traj.append((exp_so3(w), t))
    return traj


def perturb(traj, rng, rot_sigma=0.0, trans_sigma=0.0, drift_rot=0.0, drift_scale=1.0):
    """Per-frame jitter (right-multiplied) plus optional velocity drift."""
    out = [traj[0]]
    if drift_rot or drift_scale != 1.0:
        w, v = relative_twist(traj)
        cur = traj[0]
        for wi, vi in zip(w, v):
            step = (exp_so3(vadd(wi, [0.0, drift_rot, 0.0])), vscale(vi, drift_scale))
            cur = compose(cur, step)
            out.append(cur)
    else:
        out = list(traj)
    noisy = []
    for R, t in out:
        dR = exp_so3([rng.gauss(0, rot_sigma) for _ in range(3)])
        noisy.append((mat_mul(R, dR), vadd(t, [rng.gauss(0, trans_sigma) for _ in range(3)])))
    return noisy


def apply_global(traj, G):
    return [compose(G, T) for T in traj]
