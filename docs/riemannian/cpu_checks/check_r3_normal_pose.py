"""
R3 CPU check: cross-frame normal x pose consistency on a ray-cast synthetic room.

    python docs/riemannian/cpu_checks/check_r3_normal_pose.py

    L = mean angle( R_j^T R_i n_i(u),  n_j(u') ),   u' = projection of pixel u of frame i into frame j

Checks the "pre-check" from 07_novelty_catalog.md (loss ~ 0 on GT), whether the loss
responds to pose error (R3b signal), and the dynamic-object / occlusion risks.
Camera convention as in ViGeo: x right, y down, z forward; poses camera-to-world.
"""

import math
import random

from geom import (angle_between, dot, exp_so3, mat_mul, mat_T, mat_vec, norm, vadd, vscale, vsub)

W, H, F = 64, 48, 48.0
CX, CY = W / 2, H / 2
NEAR_REL_DEPTH = 0.05

# Room: floor (y = +1, camera height 1 m), ceiling y = -1.5, walls x = +-2.5, back wall z = 6, front z = -2
PLANES = [([0, -1, 0], 1.0), ([0, 1, 0], 1.5), ([1, 0, 0], 2.5), ([-1, 0, 0], 2.5), ([0, 0, -1], 6.0), ([0, 0, 1], 2.0)]
# plane: n . x = -d form expressed as dot(n_out, x) + d = 0 with n pointing into the room


def box_faces(center, half, yaw):
    """Oriented box as (R_box, center, half extents)."""
    return (exp_so3([0.0, yaw, 0.0]), center, half)


def intersect_box(o, d, box):
    Rb, c, hx = box
    RbT = mat_T(Rb)
    ol, dl = mat_vec(RbT, vsub(o, c)), mat_vec(RbT, d)
    tmin, tmax, n_local = -1e9, 1e9, None
    for k in range(3):
        if abs(dl[k]) < 1e-12:
            if abs(ol[k]) > hx[k]:
                return None
            continue
        t1, t2 = (-hx[k] - ol[k]) / dl[k], (hx[k] - ol[k]) / dl[k]
        sgn = -1.0
        if t1 > t2:
            t1, t2, sgn = t2, t1, 1.0
        if t1 > tmin:
            tmin = t1
            n_local = [0.0, 0.0, 0.0]
            n_local[k] = sgn
        tmax = min(tmax, t2)
    if tmin > tmax or tmin <= 1e-6:
        return None
    return tmin, mat_vec(Rb, n_local)


def render(pose, box):
    """Returns depth[v][u] (camera z) and normal[v][u] (camera frame, facing the camera)."""
    R, t = pose
    depth = [[0.0] * W for _ in range(H)]
    normal = [[[0.0, 0.0, 0.0] for _ in range(W)] for _ in range(H)]
    for v in range(H):
        for u in range(W):
            dc = [(u + 0.5 - CX) / F, (v + 0.5 - CY) / F, 1.0]
            dw = mat_vec(R, dc)
            best, n_best = 1e9, None
            for n, dpl in PLANES:
                den = dot(n, dw)
                if abs(den) < 1e-12:
                    continue
                s = -(dot(n, t) + dpl) / den
                if 1e-6 < s < best:
                    best, n_best = s, n
            if box is not None:
                hit = intersect_box(t, dw, box)
                if hit and hit[0] < best:
                    best, n_best = hit
            if n_best is None:
                continue
            n_w = n_best if dot(n_best, dw) < 0 else vscale(n_best, -1.0)
            depth[v][u] = best * dc[2]
            normal[v][u] = mat_vec(mat_T(R), n_w)
    return depth, normal


def sample(img, x, y, mode):
    """img[v][u] at continuous pixel coords (x, y) = (u + 0.5, v + 0.5) centres."""
    if mode == 'nearest':
        u, v = int(x), int(y)
        return img[v][u] if 0 <= u < W and 0 <= v < H else None
    x0, y0 = x - 0.5, y - 0.5
    u0, v0 = math.floor(x0), math.floor(y0)
    if u0 < 0 or v0 < 0 or u0 + 1 >= W or v0 + 1 >= H:
        return None
    a, b = x0 - u0, y0 - v0
    p = [img[v0][u0], img[v0][u0 + 1], img[v0 + 1][u0], img[v0 + 1][u0 + 1]]
    w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    if isinstance(p[0], list):
        out = [sum(wk * pk[c] for wk, pk in zip(w, p)) for c in range(3)]
        n = norm(out)
        return vscale(out, 1.0 / n) if n > 1e-9 else None
    return sum(wk * pk for wk, pk in zip(w, p))


def consistency(frame_i, frame_j, pose_i, pose_j, rel_R_used, mode='bilinear', filt=True, stride=2):
    """Mean / median angle (deg) and kept fraction for pixels of frame i that land in frame j."""
    (depth_i, normal_i), (depth_j, normal_j) = frame_i, frame_j
    (Ri, ti), (Rj, tj) = pose_i, pose_j
    angles, total = [], 0
    for v in range(0, H, stride):
        for u in range(0, W, stride):
            D = depth_i[v][u]
            if D <= 0:
                continue
            total += 1
            Xc = [(u + 0.5 - CX) / F * D, (v + 0.5 - CY) / F * D, D]
            Xj = mat_vec(mat_T(Rj), vsub(vadd(mat_vec(Ri, Xc), ti), tj))
            if Xj[2] <= 1e-3:
                continue
            x, y = F * Xj[0] / Xj[2] + CX, F * Xj[1] / Xj[2] + CY
            dj = sample(depth_j, x, y, 'nearest')
            nj = sample(normal_j, x, y, mode)
            if dj is None or nj is None or dj <= 0:
                continue
            if filt and abs(Xj[2] - dj) / Xj[2] > NEAR_REL_DEPTH:
                continue
            ni_in_j = mat_vec(rel_R_used, normal_i[v][u])
            angles.append(math.degrees(angle_between(ni_in_j, nj)))
    angles.sort()
    if not angles:
        return float('nan'), float('nan'), 0.0
    consistency.trimmed = trimmed_mean(angles)
    return sum(angles) / len(angles), angles[len(angles) // 2], len(angles) / total


def trimmed_mean(sorted_vals, ratio=0.9):
    """Mean of the lowest `ratio` fraction (same trick as normal_map_loss(ratio=0.9))."""
    k = max(1, int(len(sorted_vals) * ratio))
    return sum(sorted_vals[:k]) / k


FAILED = []


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def main():
    box = box_faces([0.3, 0.6, 3.0], [0.5, 0.4, 0.4], math.radians(25))
    pose_i = (exp_so3([0.0, math.radians(-4), 0.0]), [-0.15, 0.0, 0.0])
    pose_j = (exp_so3([math.radians(2), math.radians(6), 0.0]), [0.25, -0.05, 0.15])
    fi, fj = render(pose_i, box), render(pose_j, box)
    R_rel_gt = mat_mul(mat_T(pose_j[0]), pose_i[0])           # R_j^T R_i

    # (a) pre-check: GT normals + GT relative rotation -> ~0
    m_nn, med_nn, keep = consistency(fi, fj, pose_i, pose_j, R_rel_gt, mode='nearest')
    tr_nn = consistency.trimmed
    m_bl, med_bl, _ = consistency(fi, fj, pose_i, pose_j, R_rel_gt, mode='bilinear')
    tr_bl = consistency.trimmed
    check('GT pre-check (nearest sampling): loss ~ 0', med_nn < 1e-6 and m_nn < 2.0,
          f'mean {m_nn:.3f} deg, median {med_nn:.1e} deg, {100 * keep:.0f}% pixels kept')
    print(f'       bilinear sampling (as grid_sample would): mean {m_bl:.3f} deg, median {med_bl:.1e} deg '
          f'-> nonzero mean comes from blending normals across depth edges')
    check('90% trimmed mean removes the edge residual on GT', tr_nn < 1e-6 and tr_bl < 0.05,
          f'trimmed: nearest {tr_nn:.1e} deg, bilinear {tr_bl:.3f} deg')

    # (b) R3b signal: perturb the relative rotation used in the loss by delta about a random axis
    rng = random.Random(0)
    print('       pose-error response (R3b):  delta(deg) -> mean loss(deg)')
    resp, resp_tr = [], []
    for delta in (0.5, 1.0, 2.0, 5.0):
        vals, vals_tr = [], []
        for _ in range(8):
            ax = [rng.gauss(0, 1) for _ in range(3)]
            ax = vscale(ax, math.radians(delta) / norm(ax))
            vals.append(consistency(fi, fj, pose_i, pose_j, mat_mul(exp_so3(ax), R_rel_gt), mode='nearest')[0] - m_nn)
            vals_tr.append(consistency.trimmed)
        resp.append(sum(vals) / len(vals))
        resp_tr.append(sum(vals_tr) / len(vals_tr))
        print(f'         {delta:4.1f} -> mean {resp[-1]:.3f}, 90% trimmed {resp_tr[-1]:.3f}')
    check('loss grows ~linearly with relative-rotation error (slope 0.4-1.0)',
          all(0.4 * d < r < 1.0 * d + 0.05 for d, r in zip((0.5, 1.0, 2.0, 5.0), resp)),
          f'slopes {", ".join(f"{r / d:.2f}" for d, r in zip((0.5, 1.0, 2.0, 5.0), resp))}')
    check('trimming keeps the pose signal (trimmed slope > 0.4)',
          all(r > 0.4 * d for d, r in zip((0.5, 1.0, 2.0, 5.0), resp_tr)),
          f'trimmed slopes {", ".join(f"{r / d:.2f}" for d, r in zip((0.5, 1.0, 2.0, 5.0), resp_tr))}')

    # (c) occlusion: without the depth filter, mismatched pixels inflate the loss
    m_nf, _, keep_nf = consistency(fi, fj, pose_i, pose_j, R_rel_gt, mode='nearest', filt=False)
    check('depth filter removes occlusion errors', m_nf > m_nn,
          f'no filter: mean {m_nf:.3f} deg ({100 * keep_nf:.0f}% kept) vs filtered {m_nn:.3f} deg')

    # (d) dynamic object: the box rotates 20 deg between the frames (static-scene assumption broken)
    box_moved = box_faces([0.3, 0.6, 3.0], [0.5, 0.4, 0.4], math.radians(45))
    fj_dyn = render(pose_j, box_moved)
    m_dyn_f, _, keep_dyn = consistency(fi, fj_dyn, pose_i, pose_j, R_rel_gt, mode='nearest', filt=True)
    tr_dyn = consistency.trimmed
    m_dyn_nf, _, _ = consistency(fi, fj_dyn, pose_i, pose_j, R_rel_gt, mode='nearest', filt=False)
    print(f'       dynamic box: no filter mean {m_dyn_nf:.3f} deg, depth-filtered mean {m_dyn_f:.3f} deg '
          f'({100 * keep_dyn:.0f}% kept)')
    check('dynamic objects leak through the depth filter (expected risk)', m_dyn_f > m_nn + 0.3,
          f'{m_dyn_f:.3f} vs static {m_nn:.3f} deg')
    print(f'       dynamic box, 90% trimmed mean: {tr_dyn:.3f} deg '
          f'(trimming only helps if moving pixels are < 10% of the overlap)')

    print('\nR3 summary:', 'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')


if __name__ == '__main__':
    main()
