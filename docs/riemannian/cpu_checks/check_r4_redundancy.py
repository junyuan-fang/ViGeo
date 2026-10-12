"""
R4 CPU check: is the per-frame Sim(3) alignment curve redundant with the existing
clip-level scale_invariant_pointmap_loss? (the "must do first" pre-check in 07_novelty_catalog.md)

    python docs/riemannian/cpu_checks/check_r4_redundancy.py

Setup: GT point maps of S frames of the synthetic room (from check_r3_normal_pose.render).
Prediction = per-frame scale flicker  s_i = exp(N(0, sigma))  x GT, plus per-pixel relative
depth noise nu. We then ask two questions:

  Q1 (detection)  Does the existing loss change noticeably when flicker is added?
  Q2 (correction) Does the gradient of the existing loss w.r.t. each frame's log-scale point
                  in the direction that removes the flicker? Measured as the fraction of frames
                  where a gradient step moves log s_i towards the mean (sign agreement). Because the
                  existing loss is L1, its per-frame gradient is sign-like, so we also report the
                  (less meaningful) linear correlation. Sign agreement ~1 => the existing loss already
                  corrects flicker => R4's scale part is redundant as a *loss*.
  Q3 (measurement) Does the R4 estimator (per-frame scale alignment) recover the true flicker
                  under pixel noise? If yes, it is still useful as an *evaluation metric*.

Ports (pure Python, simplified but faithful in structure):
  - utils/alignment.py::align (truncated-L1 scale search over candidate ratios), weight 1/z, trunc 1.0
  - loss_util.normalize_pred / normalize_gt (divide by mean point norm)
  - loss_util.scale_invariant_pointmap_loss with ratio=1.0 (weights 1/z, clamp_max 10*avg, clamp(max=1))
"""

import math
import random

from check_r3_normal_pose import H, W, CX, CY, F, box_faces, render
from geom import exp_so3

STRIDE = 4


def point_maps(S):
    box = box_faces([0.3, 0.6, 3.0], [0.5, 0.4, 0.4], math.radians(25))
    maps = []
    for i in range(S):
        a = (i - S / 2) * 0.03
        pose = (exp_so3([0.0, a, 0.0]), [0.12 * (i - S / 2), 0.0, 0.05 * i])
        depth, _ = render(pose, box)
        pts = []
        for v in range(0, H, STRIDE):
            for u in range(0, W, STRIDE):
                D = depth[v][u]
                if D > 0:
                    pts.append([(u + 0.5 - CX) / F * D, (v + 0.5 - CY) / F * D, D])
        maps.append(pts)
    return maps


def normalize(maps):
    n = sum(len(m) for m in maps)
    f = sum(math.sqrt(sum(c * c for c in p)) for m in maps for p in m) / n
    return [[[c / f for c in p] for p in m] for m in maps]


def align_scale(src, tgt, max_pts=300, trunc=1.0):
    """Truncated-L1 weighted scale: argmin_s sum_k w_k * min(|s*src_k - tgt_k|, trunc), w = 1/z (per point)."""
    step = max(1, len(src) // max_pts)
    xs, ys, ws = [], [], []
    for p, q in zip(src[::step], tgt[::step]):
        w = 1.0 / max(q[2], 1e-2)
        for k in range(3):
            xs.append(p[k]); ys.append(q[k]); ws.append(w)
    cands = [y / x for x, y in zip(xs, ys) if abs(x) > 1e-6 and y / x > 0]
    best, best_s = 1e18, 1.0
    for s in cands[::3]:
        loss = sum(w * min(abs(s * x - y), trunc) for x, y, w in zip(xs, ys, ws))
        if loss < best:
            best, best_s = loss, s
    return best_s


def pointmap_loss(pred, gt, scale):
    """scale_invariant_pointmap_loss, ratio=1 (weights 1/z, clamp_max 10*avg, clamp(max=1))."""
    num, den = 0.0, 0
    for mp, mg in zip(pred, gt):
        ws = [1.0 / max(q[2], 1e-5) for q in mg]
        cap = 10.0 * sum(ws) / len(ws)
        for p, q, w in zip(mp, mg, ws):
            num += min(w, cap) * sum(abs(scale * a - b) for a, b in zip(p, q))
            den += 1
    return min(num / den, 1.0)


def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb + 1e-12)


def main():
    S = 8
    gt_raw = point_maps(S)
    gt = normalize(gt_raw)
    print(f'{S} frames, {sum(len(m) for m in gt)} points\n')
    print(f'{"flicker":>8s} {"noise":>6s} | {"L_exist":>8s} {"dL vs no-flicker":>17s} | '
          f'{"sign agree":>10s} {"corr":>9s} | {"R4: std log s_i (est / true)":>28s}')
    results = []
    for nu in (0.0, 0.02, 0.05, 0.10):
        base_loss = None
        for sigma in (0.0, 0.02, 0.05, 0.10):
            cs, agree, l_vals, r4_est, r4_true = [], [], [], [], []
            for seed in range(4):
                rng = random.Random(100 * seed + 7)
                log_s = [rng.gauss(0, sigma) for _ in range(S)]
                pred_raw = [[[math.exp(ls) * c * (1 + rng.gauss(0, nu)) for c in p] for p in m]
                            for ls, m in zip(log_s, gt_raw)]
                pred = normalize(pred_raw)
                scale = align_scale([p for m in pred for p in m], [q for m in gt for q in m])
                L = pointmap_loss(pred, gt, scale)
                l_vals.append(L)
                # Q2: numeric gradient of the existing loss w.r.t. each frame's log-scale (clip scale held fixed)
                eps = 1e-4
                grads = []
                for i in range(S):
                    bumped = [[[c * math.exp(eps) for c in p] for p in m] if j == i else m for j, m in enumerate(pred)]
                    grads.append((pointmap_loss(bumped, gt, scale) - L) / eps)
                centred = [x - sum(log_s) / S for x in log_s]
                if sigma > 0:
                    cs.append(corr([-g for g in grads], [-c for c in centred]))
                    agree += [(g > 0) == (c > 0) for g, c in zip(grads, centred) if abs(c) > 1e-3]
                # R4: per-frame scale estimates -> dispersion of log s_i
                est = [math.log(align_scale(mp, mg, max_pts=120)) for mp, mg in zip(pred, gt)]
                m_est = sum(est) / S
                r4_est.append(math.sqrt(sum((e - m_est) ** 2 for e in est) / S))
                r4_true.append(math.sqrt(sum(c * c for c in centred) / S))
            L = sum(l_vals) / len(l_vals)
            if sigma == 0.0:
                base_loss = L
            c = sum(cs) / len(cs) if cs else float('nan')
            ag = sum(agree) / len(agree) if agree else float('nan')
            est, tru = sum(r4_est) / len(r4_est), sum(r4_true) / len(r4_true)
            results.append((sigma, nu, L, L - base_loss, c, est, tru, ag))
            rel = f'{100 * (L - base_loss) / base_loss:16.0f}%' if base_loss > 1e-6 else f'{"(no base)":>17s}'
            print(f'{sigma:8.2f} {nu:6.2f} | {L:8.4f} {rel} | {ag:10.2f} {c:9.3f} | {est:13.4f} / {tru:.4f}')
        print()

    flick = [r for r in results if r[0] > 0]
    min_agree = min(r[7] for r in flick)
    worst_est = max(abs(r[5] - r[6]) / r[6] for r in flick)
    print(f'Q2 (correction): sign agreement min {min_agree:.2f}; linear corr min {min(r[4] for r in flick):.3f}')
    print(f'Q3 (measurement): R4 flicker estimate within {100 * worst_est:.0f}% of the true value in all cases')
    if min_agree > 0.9:
        print('Verdict on synthetic data: the scale part of R4 is REDUNDANT as a loss (the existing clip-level '
              'loss already pushes every frame back to the shared scale), but USEFUL as an evaluation metric.')
    else:
        print('Verdict on synthetic data: NOT redundant: the existing loss does not reliably correct per-frame scale.')

if __name__ == '__main__':
    main()
