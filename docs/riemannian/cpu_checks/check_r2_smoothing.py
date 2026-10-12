"""
R2 CPU check: training-free SE(3) trajectory smoothing as an *upper bound* for what
a 2nd-order constraint can buy, on synthetic trajectories.

    python docs/riemannian/cpu_checks/check_r2_smoothing.py

Smoother (Riemannian gradient descent with right perturbations T_i <- T_i Exp(delta_i)):

    E(T) = sum_i ||Log(R_i^pred^T R_i)||^2 + ||t_i - t_i^pred||^2           (stay close to the prediction)
         + lam * sum_i ||xi_{i+1}(T) - xi_i(T)||^2                          (small covariant acceleration)

The same `smooth_se3` can later be applied to real ViGeo predictions (saved poses)
once a GPU run has produced them; this script only uses synthetic data.
"""

import math
import random

from geom import (ate, curve_accel_loss, dot, exp_so3, jitter, mat_mul, mat_T, perturb, relative_twist, rpe,
                  smooth_trajectory, so3_log, vadd, vsub)


def _data_term(T, P):
    w = so3_log(mat_mul(mat_T(P[0]), T[0]))
    d = vsub(T[1], P[1])
    return dot(w, w) + dot(d, d)


def _accel_terms(traj, k_from, k_to):
    """sum_k ||xi_{k+1} - xi_k||^2 for k in [k_from, k_to], restricted to valid k."""
    lo, hi = max(0, k_from), min(len(traj) - 3, k_to)
    if lo > hi:
        return 0.0
    w, v = relative_twist(traj[lo:hi + 3])
    total = 0.0
    for i in range(len(w) - 1):
        a, b = vsub(w[i + 1], w[i]), vsub(v[i + 1], v[i])
        total += dot(a, a) + dot(b, b)
    return total


def energy(traj, pred, lam):
    return sum(_data_term(T, P) for T, P in zip(traj, pred)) + lam * _accel_terms(traj, 0, len(traj))


def _retract(T, d):
    R, t = T
    return (mat_mul(R, exp_so3(d[:3])), vadd(t, d[3:]))


def smooth_se3(pred, lam, iters=150, eps=1e-6):
    traj = list(pred)
    E = energy(traj, pred, lam)
    step = 0.5
    for _ in range(iters):
        grads = []
        for i in range(len(traj)):
            # local energy: only terms touching frame i (accel k in [i-2, i])
            def local(Ti):
                tr = traj[:i] + [Ti] + traj[i + 1:]
                return _data_term(Ti, pred[i]) + lam * _accel_terms(tr, i - 2, i)
            base = local(traj[i])
            g = []
            for k in range(6):
                d = [0.0] * 6
                d[k] = eps
                g.append((local(_retract(traj[i], d)) - base) / eps)
            grads.append(g)
        # Armijo backtracking along -grad
        gnorm2 = sum(x * x for g in grads for x in g)
        if gnorm2 < 1e-14:
            break
        while step > 1e-6:
            cand = [_retract(T, [-step * x for x in g]) for T, g in zip(traj, grads)]
            E_new = energy(cand, pred, lam)
            if E_new <= E - 1e-4 * step * gnorm2:
                traj, E = cand, E_new
                step *= 1.5
                break
            step *= 0.5
    return traj


def report(gt, pred):
    r_rot, r_t = rpe(pred, gt)
    j_ang, j_lin = jitter(pred)
    return {'ATE': ate(pred, gt), 'RPE_rot(deg)': r_rot, 'RPE_t': r_t,
            'accel_loss': curve_accel_loss(pred, gt), 'jitter_ang(deg)': j_ang}


def main():
    lams = [0.0, 1.0, 4.0, 16.0, 64.0]
    scenarios = [
        ('jitter', dict(rot_sigma=math.radians(0.6), trans_sigma=0.01)),
        ('jitter+drift', dict(rot_sigma=math.radians(0.6), trans_sigma=0.01, drift_rot=0.004, drift_scale=1.03)),
    ]
    keys = ['ATE', 'RPE_rot(deg)', 'RPE_t', 'accel_loss', 'jitter_ang(deg)']
    summary = []
    for kind in ('arc', 'orbit', 'sharp_turn'):
        gt = smooth_trajectory(24, kind)
        gt_jit = jitter(gt)[0]
        for name, kw in scenarios:
            rows = {lam: {k: 0.0 for k in keys} for lam in lams}
            n_seeds = 3
            for seed in range(n_seeds):
                pred = perturb(gt, random.Random(seed), **kw)
                for lam in lams:
                    out = pred if lam == 0 else smooth_se3(pred, lam)
                    for k, v in report(gt, out).items():
                        rows[lam][k] += v / n_seeds
            print(f'\n=== {kind:10s} | {name:13s} | GT jitter_ang = {gt_jit:.3f} deg ===')
            print(f'{"lambda":>7s} ' + ' '.join(f'{k:>16s}' for k in keys))
            for lam in lams:
                print(f'{lam:7.1f} ' + ' '.join(f'{rows[lam][k]:16.4f}' for k in keys))
            best = min(lams[1:], key=lambda l: rows[l]['RPE_rot(deg)'])
            summary.append((kind, name, rows[0.0], rows[best], best))

    print('\n=== Summary: best lambda (by RPE_rot) vs. no smoothing ===')
    print(f'{"trajectory":12s} {"noise":14s} {"lam":>5s} {"RPE_rot":>18s} {"RPE_t":>18s} {"accel_loss":>18s} {"ATE":>18s}')
    for kind, name, r0, rb, best in summary:
        def pct(k):
            return f'{r0[k]:.3f}->{rb[k]:.3f} ({100 * (rb[k] - r0[k]) / r0[k]:+.0f}%)'
        print(f'{kind:12s} {name:14s} {best:5.0f} {pct("RPE_rot(deg)"):>18s} {pct("RPE_t"):>18s} '
              f'{pct("accel_loss"):>18s} {pct("ATE"):>18s}')


if __name__ == '__main__':
    main()
