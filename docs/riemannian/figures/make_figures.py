"""
Generate the explanatory SVG figures for docs/riemannian/.

Pure Python standard library (no numpy / torch / matplotlib), so anyone can run:

    python docs/riemannian/figures/make_figures.py

Figures:
    fig1_l1_vs_geodesic.svg   baseline quaternion L1 vs true rotation error (SO(3))
    fig2_error_orders.svg     which error type each order (0/1/2) is sensitive to (SE(2) toy)
    fig3_sampling.svg         why we match pred vs GT instead of a smoothness prior

Concept figures for the tutorial chapters:
    concept_pipeline.svg      where camera poses come from in ViGeo, old vs new loss
    concept_circle.svg        chord vs geodesic, tangent space, Log / Exp on a circle
    concept_curve.svg         velocity / acceleration of a discrete curve on a circle
"""

import math
import os
import random

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

C_GT = '#6b7280'
C_SERIES = ['#2563eb', '#d97706', '#059669']
C_AXIS = '#9ca3af'
C_TEXT = '#111827'
FONT = 'font-family="-apple-system, \'PingFang SC\', \'Noto Sans CJK SC\', \'Microsoft YaHei\', Helvetica, Arial, sans-serif"'


# -----------------------------------------------------------------------------
# Minimal SVG helpers
# -----------------------------------------------------------------------------


class Svg:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.items = []

    def add(self, s):
        self.items.append(s)

    def text(self, x, y, s, size=12, anchor='start', color=C_TEXT, weight='normal'):
        s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{color}" font-weight="{weight}" {FONT}>{s}</text>')

    def line(self, x1, y1, x2, y2, color=C_AXIS, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{color}" stroke-width="{width}"{d}/>')

    def polyline(self, pts, color, width=2, dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-opacity="{opacity}" stroke-linejoin="round"{d}/>')

    def polygon(self, pts, color, opacity=0.2):
        p = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.add(f'<polygon points="{p}" fill="{color}" fill-opacity="{opacity}" stroke="none"/>')

    def rect(self, x, y, w, h, color, opacity=1.0):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'fill="{color}" fill-opacity="{opacity}" rx="2"/>')

    def save(self, name):
        body = '\n'.join(self.items)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">\n'
               f'<rect width="100%" height="100%" fill="#ffffff"/>\n{body}\n</svg>\n')
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            f.write(svg)
        print('wrote', name)


class Axes:
    """Linear data -> pixel mapping inside a rectangle of an Svg."""

    def __init__(self, svg, x0, y0, w, h, xlim, ylim):
        self.svg, self.x0, self.y0, self.w, self.h = svg, x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def px(self, x, y):
        (a, b), (c, d) = self.xlim, self.ylim
        return (self.x0 + (x - a) / (b - a) * self.w,
                self.y0 + self.h - (y - c) / (d - c) * self.h)

    def frame(self, xticks, yticks, xlabel='', ylabel='', fmt='{:g}'):
        s = self.svg
        s.line(self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h)
        s.line(self.x0, self.y0, self.x0, self.y0 + self.h)
        for t in xticks:
            x, y = self.px(t, self.ylim[0])
            s.line(x, y, x, y + 4)
            s.text(x, y + 16, fmt.format(t), 11, 'middle', C_AXIS)
        for t in yticks:
            x, y = self.px(self.xlim[0], t)
            s.line(x - 4, y, x, y)
            s.line(x, y, self.x0 + self.w, y, '#eef0f3')
            s.text(x - 7, y + 4, fmt.format(t), 11, 'end', C_AXIS)
        if xlabel:
            s.text(self.x0 + self.w / 2, self.y0 + self.h + 34, xlabel, 12, 'middle')
        if ylabel:
            s.add(f'<text transform="translate({self.x0 - 40:.1f},{self.y0 + self.h / 2:.1f}) rotate(-90)" '
                  f'font-size="12" text-anchor="middle" fill="{C_TEXT}" {FONT}>{ylabel}</text>')


def legend(svg, x, y, entries):
    for i, (label, color, dash) in enumerate(entries):
        yy = y + i * 18
        svg.line(x, yy, x + 22, yy, color, 2.5, dash)
        svg.text(x + 28, yy + 4, label, 12)


# -----------------------------------------------------------------------------
# SO(3) helpers (quaternions are XYZW, as in src/loss/utils.py)
# -----------------------------------------------------------------------------


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_axis_angle(axis, angle):
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(angle / 2) / n
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def standardize(q):
    return tuple(-c for c in q) if q[3] < 0 else q


def random_axis(rng):
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        if sum(a * a for a in v) > 1e-6:
            return v


# -----------------------------------------------------------------------------
# SE(2) toy trajectories (planar analogue of the SE(3) losses in riemannian.py)
# -----------------------------------------------------------------------------


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gt_trajectory(n, s_max=math.pi):
    """Smooth planar camera path: list of (x, y, heading)."""
    traj = []
    for i in range(n):
        s = s_max * i / (n - 1)
        x, y = 4.0 * s / s_max * 3, 1.2 * math.sin(2 * s)
        dx, dy = 12.0 / s_max, 2.4 * math.cos(2 * s)
        traj.append((x, y, math.atan2(dy, dx) + 0.2 * math.sin(3 * s)))
    return traj


def body_twists(traj):
    """xi_i = Log(T_i^-1 T_{i+1}) in SE(2): (omega, v_x, v_y) in the body frame of T_i."""
    out = []
    for (x0, y0, p0), (x1, y1, p1) in zip(traj[:-1], traj[1:]):
        c, s = math.cos(p0), math.sin(p0)
        dx, dy = x1 - x0, y1 - y0
        out.append((wrap(p1 - p0), c * dx + s * dy, -s * dx + c * dy))
    return out


def loss_order0(pred, gt):
    return sum(abs(wrap(p[2] - g[2])) + abs(p[0] - g[0]) + abs(p[1] - g[1])
               for p, g in zip(pred, gt)) / len(gt)


def loss_order1(pred, gt):
    xp, xg = body_twists(pred), body_twists(gt)
    return sum(abs(wrap(a[0] - b[0])) + abs(a[1] - b[1]) + abs(a[2] - b[2])
               for a, b in zip(xp, xg)) / len(xg)


def accel(traj):
    xi = body_twists(traj)
    return [tuple(b[k] - a[k] for k in range(3)) for a, b in zip(xi[:-1], xi[1:])]


def loss_order2(pred, gt):
    ap, ag = accel(pred), accel(gt)
    return sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
               for a, b in zip(ap, ag)) / len(ag)


def smoothness_prior(traj):
    """A GT-free prior: mean ||a_i||. Should be ~0 for a 'good' trajectory, but is it?"""
    a = accel(traj)
    return sum(abs(v[0]) + abs(v[1]) + abs(v[2]) for v in a) / len(a)


def apply_global(traj, dphi, tx, ty):
    c, s = math.cos(dphi), math.sin(dphi)
    return [(c * x - s * y + tx, s * x + c * y + ty, p + dphi) for x, y, p in traj]


def add_jitter(traj, rng, sig_t, sig_r):
    return [(x + rng.gauss(0, sig_t), y + rng.gauss(0, sig_t), p + rng.gauss(0, sig_r))
            for x, y, p in traj]


def add_drift(traj, scale, yaw_bias):
    """Re-integrate body twists with a constant multiplicative + yaw bias (1st-order error)."""
    xi = body_twists(traj)
    x, y, p = traj[0]
    out = [(x, y, p)]
    for w, vx, vy in xi:
        c, s = math.cos(p), math.sin(p)
        x, y = x + scale * (c * vx - s * vy), y + scale * (s * vx + c * vy)
        p = p + w + yaw_bias
        out.append((x, y, p))
    return out


# -----------------------------------------------------------------------------
# Figure 1: quaternion L1 (baseline camera_loss) vs geodesic angle
# -----------------------------------------------------------------------------


def fig1():
    rng = random.Random(0)
    q_gt = standardize(quat_axis_angle((0.3, -0.5, 0.8), math.radians(150)))
    thetas = [math.radians(d) for d in range(0, 181, 3)]
    lo, hi, med = [], [], []
    for th in thetas:
        vals = []
        for _ in range(300):
            q_pred = standardize(quat_mul(q_gt, quat_axis_angle(random_axis(rng), th)))
            vals.append(sum(abs(a - b) for a, b in zip(q_pred, q_gt)))
        vals.sort()
        lo.append(vals[0]); hi.append(vals[-1]); med.append(vals[len(vals) // 2])

    svg = Svg(640, 470)
    svg.text(320, 26, 'Baseline camera_loss (L1 on quaternion) vs. true rotation error', 15, 'middle', weight='bold')
    svg.text(320, 46, 'GT = 150° rotation; pred = GT perturbed by θ about 300 random axes', 12, 'middle', C_AXIS)
    ax = Axes(svg, 80, 70, 520, 280, (0, 180), (0, 1.05 * max(max(hi), math.pi)))
    ax.frame(range(0, 181, 30), [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 'true geodesic error θ (degrees)', 'loss value')
    deg = [math.degrees(t) for t in thetas]
    band = [ax.px(d, v) for d, v in zip(deg, hi)] + [ax.px(d, v) for d, v in zip(reversed(deg), reversed(lo))]
    svg.polygon(band, C_SERIES[1], 0.18)
    svg.polyline([ax.px(d, v) for d, v in zip(deg, med)], C_SERIES[1], 2)
    svg.polyline([ax.px(d, t) for d, t in zip(deg, thetas)], C_SERIES[0], 2.5)
    legend(svg, 90, 400, [
        ('geodesic angle θ (rad) — geodesic_camera_loss', C_SERIES[0], None),
        ('quaternion L1, median over axes — camera_loss', C_SERIES[1], None),
    ])
    svg.rect(92, 430, 20, 10, C_SERIES[1], 0.18)
    svg.text(118, 439, 'quaternion L1, min–max over axes', 12)
    svg.save('fig1_l1_vs_geodesic.svg')


# -----------------------------------------------------------------------------
# Figure 2: which order sees which error type
# -----------------------------------------------------------------------------


def fig2():
    rng = random.Random(1)
    gt = gt_trajectory(24)
    preds = {
        'global offset': apply_global(gt, math.radians(6), 0.4, -0.3),
        'per-frame jitter': add_jitter(gt, rng, 0.06, math.radians(2.0)),
        'velocity drift': add_drift(gt, 1.06, math.radians(0.6)),
    }

    svg = Svg(980, 380)
    svg.text(490, 26, 'Each order of the curve loss is sensitive to a different error type (SE(2) toy)',
             15, 'middle', weight='bold')

    # Left: trajectories
    ax = Axes(svg, 40, 60, 470, 260, (-0.5, 12.8), (-2.6, 2.6))
    svg.text(275, 52, 'GT (gray) and three corrupted predictions', 12, 'middle', C_AXIS)
    svg.polyline([ax.px(x, y) for x, y, _ in gt], C_GT, 3)
    for (name, tr), color in zip(preds.items(), C_SERIES):
        svg.polyline([ax.px(x, y) for x, y, _ in tr], color, 1.8)
        for x, y, p in tr[::3]:
            x0, y0 = ax.px(x, y)
            x1, y1 = ax.px(x + 0.35 * math.cos(p), y + 0.35 * math.sin(p))
            svg.line(x0, y0, x1, y1, color, 1.2)
    legend(svg, 60, 290, [('GT', C_GT, None)] + [(n, c, None) for n, c in zip(preds, C_SERIES)][:1])
    legend(svg, 200, 290, [(n, c, None) for n, c in zip(preds, C_SERIES)][1:])

    # Right: grouped bars, each loss normalized by its max over the three preds
    names = list(preds)
    losses = [('0th: per-frame', loss_order0), ('1st: velocity', loss_order1), ('2nd: acceleration', loss_order2)]
    table = [[fn(preds[n], gt) for n in names] for _, fn in losses]
    bx = Axes(svg, 580, 70, 370, 240, (0, 3), (0, 1.1))
    bx.frame([], [0, 0.5, 1.0], '', 'loss / max over preds')
    gw = 370 / 3
    for gi, ((lname, _), row) in enumerate(zip(losses, table)):
        m = max(row) or 1.0
        for k, v in enumerate(row):
            x = 580 + gi * gw + 14 + k * (gw - 28) / 3
            y = bx.px(0, v / m)[1]
            svg.rect(x, y, (gw - 28) / 3 - 4, 310 - y, C_SERIES[k])
            svg.text(x + (gw - 28) / 6 - 2, y - 4, f'{v:.3f}', 9, 'middle', C_AXIS)
        svg.text(580 + gi * gw + gw / 2, 328, lname, 12, 'middle')
    svg.text(765, 360, 'raw values printed on top of bars', 11, 'middle', C_AXIS)
    svg.save('fig2_error_orders.svg')
    return names, losses, table


# -----------------------------------------------------------------------------
# Figure 3: training-time sampling (random stride, reflection, shuffle)
# -----------------------------------------------------------------------------


def sample_sequence(n, seq_len, rng, max_interval, shuffle):
    """Re-implementation of BaseDataset.sample_sequence (src/datasets/base_data.py)."""
    eff = max(1, min(max_interval, n - 1))
    pos, direction, out = rng.randrange(n), 1, []
    for _ in range(seq_len):
        out.append(pos)
        nxt = pos + direction * rng.randint(1, eff)
        if nxt >= n:
            direction = -1
            nxt = max(0, pos + direction * rng.randint(1, eff))
        elif nxt < 0:
            direction = 1
            nxt = min(n - 1, pos + direction * rng.randint(1, eff))
        pos = nxt
    if shuffle:
        rng.shuffle(out)
    return out


def fig3():
    rng = random.Random(2)
    full = gt_trajectory(200, 2 * math.pi)
    modes = [('stride 1', 1, False), ('stride 1–8 + reflect', 8, False), ('shuffled (p=0.3 in training)', 8, True)]
    prior_gt, match_jit, match_perfect = [], [], []
    for _, mi, sh in modes:
        a, b, c = [], [], []
        for _ in range(300):
            idx = sample_sequence(len(full), 24, rng, mi, sh)
            g = [full[i] for i in idx]
            p = add_jitter(g, rng, 0.03, math.radians(1.0))
            a.append(smoothness_prior(g))
            b.append(loss_order2(p, g))
            c.append(loss_order2(g, g))
        prior_gt.append(sum(a) / len(a)); match_jit.append(sum(b) / len(b)); match_perfect.append(sum(c) / len(c))

    svg = Svg(820, 380)
    svg.text(410, 26, 'GT-free smoothness prior vs. GT-matched curve_accel_loss under training-time sampling',
             15, 'middle', weight='bold')
    ymax = max(prior_gt + match_jit) * 1.15
    ax = Axes(svg, 90, 70, 690, 240, (0, 3), (0, ymax))
    step = 10 ** math.floor(math.log10(ymax / 2))
    ticks = [k * step for k in range(int(ymax / step) + 1)][::max(1, int(ymax / step) // 5)]
    ax.frame([], ticks, '', 'mean loss over 300 sampled clips', fmt='{:.2g}')
    series = [('smoothness prior on the GT clip itself (should be 0)', prior_gt, C_SERIES[1]),
              ('curve_accel_loss: jittered pred vs GT', match_jit, C_SERIES[0]),
              ('curve_accel_loss: perfect pred vs GT', match_perfect, C_SERIES[2])]
    gw = 690 / 3
    for gi, (mname, _, _) in enumerate(modes):
        for k, (_, vals, color) in enumerate(series):
            v = vals[gi]
            x = 90 + gi * gw + 24 + k * (gw - 48) / 3
            y = ax.px(0, v)[1]
            svg.rect(x, y, (gw - 48) / 3 - 6, 310 - y, color)
            svg.text(x + (gw - 48) / 6 - 3, y - 4, f'{v:.3f}', 10, 'middle', C_AXIS)
        svg.text(90 + gi * gw + gw / 2, 328, mname, 12, 'middle')
    for k, (label, _, color) in enumerate(series):
        svg.rect(110, 78 + 18 * k, 14, 10, color)
        svg.text(130, 87 + 18 * k, label, 12)
    svg.save('fig3_sampling.svg')
    return modes, prior_gt, match_jit, match_perfect


# -----------------------------------------------------------------------------
# Concept figures (tutorial)
# -----------------------------------------------------------------------------


def box(svg, x, y, w, h, title, sub='', color='#eef2ff', stroke='#6366f1'):
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="{stroke}"/>')
    svg.text(x + w / 2, y + h / 2 + (-3 if sub else 5), title, 13, 'middle', weight='bold')
    if sub:
        svg.text(x + w / 2, y + h / 2 + 15, sub, 11, 'middle', C_AXIS)


def arrow(svg, x1, y1, x2, y2, color=C_AXIS, width=1.6):
    svg.line(x1, y1, x2, y2, color, width)
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + 2.6, ang - 2.6
    svg.add(f'<polygon points="{x2:.1f},{y2:.1f} {x2 + 8 * math.cos(a1):.1f},{y2 + 8 * math.sin(a1):.1f} '
            f'{x2 + 8 * math.cos(a2):.1f},{y2 + 8 * math.sin(a2):.1f}" fill="{color}"/>')


def concept_pipeline():
    svg = Svg(940, 330)
    svg.text(470, 26, 'ViGeo 里相机位姿从哪来，以及 loss 加在哪', 15, 'middle', weight='bold')
    xs = [20, 200, 380, 560, 740]
    items = [('视频帧 ×S', 'RGB [S,3,H,W]'), ('DINOv2 编码器', '帧间 attention'),
             ('camera token', '每帧 1 个向量'), ('camera head', '2 层 MLP'), ('位姿 T_0 … T_S-1', '每帧 [t(3), q(4)]')]
    for x, (t, sub) in zip(xs, items):
        box(svg, x, 50, 160, 60, t, sub)
    for x in xs[:-1]:
        arrow(svg, x + 160, 80, x + 180, 80)

    # old loss
    box(svg, 40, 160, 400, 140, '', '', '#fff7ed', '#d97706')
    svg.text(240, 182, '原版 camera_loss：每帧单独比较', 13, 'middle', weight='bold')
    for k in range(5):
        cx = 90 + k * 75
        svg.add(f'<circle cx="{cx}" cy="225" r="7" fill="{C_SERIES[0]}"/>')
        svg.add(f'<circle cx="{cx}" cy="262" r="7" fill="{C_GT}"/>')
        svg.line(cx, 232, cx, 255, C_SERIES[1], 2, '3,3')
    svg.text(60, 229, '预测', 11, 'end'); svg.text(60, 266, 'GT', 11, 'end')
    svg.text(240, 292, '帧和帧之间互不相关；四元数当成 R⁴ 里的点', 11, 'middle', C_AXIS)

    # new loss
    box(svg, 500, 160, 400, 140, '', '', '#ecfdf5', '#059669')
    svg.text(700, 182, '本分支：把整段位姿当成一条曲线比较', 13, 'middle', weight='bold')
    pts_p = [(550 + k * 75, 225 + 8 * math.sin(k * 1.3)) for k in range(5)]
    pts_g = [(550 + k * 75, 262 + 4 * math.sin(k * 0.9)) for k in range(5)]
    svg.polyline(pts_p, C_SERIES[0], 2.5); svg.polyline(pts_g, C_GT, 2.5)
    for x, y in pts_p:
        svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{C_SERIES[0]}"/>')
    for x, y in pts_g:
        svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{C_GT}"/>')
    svg.text(700, 292, '比位置（0 阶）、每一步怎么动（1 阶）、动法怎么变（2 阶）', 11, 'middle', C_AXIS)
    arrow(svg, 820, 110, 700, 158, '#059669')
    arrow(svg, 820, 110, 240, 158, '#d97706')
    svg.save('concept_pipeline.svg')


def concept_circle():
    svg = Svg(900, 400)
    svg.text(450, 26, '最简单的弯曲空间：圆。直线距离 vs 测地线，以及 Log / Exp', 15, 'middle', weight='bold')
    cx, cy, r = 230, 220, 140
    svg.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{C_GT}" stroke-width="2"/>')
    a_p, a_q = math.radians(90), math.radians(-10)

    def pt(a):
        return cx + r * math.cos(a), cy - r * math.sin(a)

    (px, py), (qx, qy) = pt(a_p), pt(a_q)
    svg.line(px, py, qx, qy, C_SERIES[1], 2.5, '6,4')
    arc = [pt(a_p + (a_q - a_p) * k / 40) for k in range(41)]
    svg.polyline(arc, C_SERIES[0], 4)
    # tangent line at p and Log_p(q)
    L = r * abs(a_q - a_p)
    svg.line(px - 60, py, px + L + 30, py, C_SERIES[2], 1.5)
    arrow(svg, px, py, px + L, py, C_SERIES[2], 3)
    for (x, y), lab, dx in [((px, py), 'p', -14), ((qx, qy), 'q', 14)]:
        svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{C_TEXT}"/>')
        svg.text(x + dx, y - 10, lab, 15, 'middle', weight='bold')
    svg.text(px + L / 2, py - 12, 'Log_p(q)：把弧"摊平"到切线上', 12, 'middle', C_SERIES[2])
    svg.text(cx + 40, cy + 30, '弦（直线距离）', 12, 'start', C_SERIES[1])
    svg.text(cx + 128, cy - 105, '弧 = 测地线', 12, 'start', C_SERIES[0])
    svg.text(px - 64, py - 4, '切空间 T_pM', 11, 'end', C_SERIES[2])

    # right: explanation
    x0 = 520
    lines = [
        ('流形 M', '局部看是平的、整体是弯的空间（这里是圆）'),
        ('切空间 T_pM', '在 p 点贴上的一条"平"的线 / 面，可以做加减'),
        ('测地线', '只能沿着 M 走时的最短路径（蓝色弧）'),
        ('测地距离', '弧长。圆上就是角度差 × 半径'),
        ('Log_p(q)', '从 p 看 q 在哪个方向、多远 → 切空间里的向量'),
        ('Exp_p(v)', 'Log 的反操作：沿方向 v 走 |v| 远，落回 M 上'),
    ]
    for k, (a, b) in enumerate(lines):
        svg.text(x0, 90 + k * 46, a, 13, 'start', weight='bold')
        svg.text(x0, 108 + k * 46, b, 12, 'start', C_AXIS)
    svg.text(450, 388, '弦长 < 弧长，而且两点越远差得越多。用弦长当 loss，衡量的就不是"真正走了多远"。', 12, 'middle')
    svg.save('concept_circle.svg')


def concept_curve():
    svg = Svg(900, 360)
    svg.text(450, 26, '曲线上的 0 / 1 / 2 阶：位置、速度、加速度（圆上的离散曲线）', 15, 'middle', weight='bold')

    def panel(x_off, angles, title, color):
        cx, cy, r = x_off + 200, 200, 105
        svg.text(cx, 62, title, 13, 'middle', weight='bold')
        svg.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="2"/>')
        pts = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in angles]
        svg.polyline(pts, color, 2)
        for k, ((x, y), a) in enumerate(zip(pts, angles)):
            svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            if k + 1 < len(angles):
                d = angles[k + 1] - a
                L = r * d  # arc length of the step = |Log| on a circle of radius r
                arrow(svg, x, y, x - L * math.sin(a), y - L * math.cos(a), C_SERIES[1], 1.8)
        steps = [b - a for a, b in zip(angles[:-1], angles[1:])]
        acc = [abs(b - a) for a, b in zip(steps[:-1], steps[1:])]
        svg.text(cx, 330, f'平均 |加速度| = {sum(acc) / len(acc):.3f} rad/步²', 12, 'middle', C_AXIS)

    smooth = [math.radians(200 - 22 * k) for k in range(8)]
    rng = random.Random(5)
    jitter = [math.radians(200 - 22 * k + (rng.gauss(0, 7) if 0 < k < 7 else 0)) for k in range(8)]
    panel(20, smooth, '匀速：每一步一样大 → 加速度 ≈ 0', C_SERIES[0])
    panel(470, jitter, '抖动：步长忽大忽小 → 加速度大', C_SERIES[2])
    svg.text(450, 352, '圆点 = 每帧位置（0 阶）；橙色箭头 = 到下一帧的 Log，即离散速度（1 阶）；相邻箭头之差 = 加速度（2 阶）',
             11, 'middle', C_TEXT)
    svg.save('concept_curve.svg')


if __name__ == '__main__':
    concept_pipeline()
    concept_circle()
    concept_curve()
    fig1()
    names, losses, table = fig2()
    for (lname, _), row in zip(losses, table):
        print(f'  fig2 {lname:20s}', '  '.join(f'{n}={v:.4f}' for n, v in zip(names, row)))
    modes, a, b, c = fig3()
    for (m, _, _), x, y, z in zip(modes, a, b, c):
        print(f'  fig3 {m:30s} prior(GT)={x:.4f}  match(jitter)={y:.4f}  match(perfect)={z:.4f}')
