"""
Freehand squiggle: a simulated hand holding the pen down and scribbling.

The pen circles around a wandering centre ("loops"); the line may cross
itself. Everything is driven by momentum plus random perturbation, applied to
*rates of change* so the line stays smooth and organic:

* heading: integrates a turn rate; the turn rate drifts randomly (mean-
  reverting noise) and is nudged towards the image flow (contours) and towards
  where darkness is still missing - so the centre swings in smooth arcs;
* contours: a soft repulsion that grows near drawn contour lines turns the
  centre away in a curve (no instant bounce); occasionally it is allowed through;
* loop radius, angular speed, ellipse squash and ellipse orientation each glide
  towards their (tone / flow dependent) target with inertia and a random drift;
* tone by feedback: every bit of line subtracts its ink (pen width x length,
  saturating where ink already is, spread at viewing scale) from a
  "still needed" map. Dark areas keep the hand busy and get crossed over and
  over; light areas get a quick pass;
* strokes: the hand scribbles for a while (random duration around
  `stroke_px`), erasing the needed darkness as it goes, then lifts, samples a
  new start (probability ~ darkness still missing) and starts elsewhere. A
  stroke also ends early when its neighbourhood is satisfied.

All lengths are in image pixels (the pipeline knows mm per pixel).
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage


@dataclass
class FreehandConfig:
    pen_px: float = 1.0
    loop_min: float = 1.5            # loop radius in the darkest areas (px) - never smaller
    loop_max: float = 6.0            # loop radius in light areas (px)
    advance: float = 0.25            # centre speed relative to pen speed (lower = denser loops)
    step: float = 0.5                # pen step along the curve (px)
    momentum: float = 0.9            # 0..1 inertia of the turn rate (higher = longer smooth arcs)
    flow_pull: float = 0.6           # pull along the flow direction
    tone_pull: float = 0.8           # pull towards missing darkness
    turn_noise: float = 0.35         # random drift of the turn rate
    wobble: float = 0.35             # randomness of loop size / speed / shape (0 = mechanical)
    squash: float = 0.35             # mean ellipse flattening (0 = circles)
    view_px: float = 2.0             # blur at which ink and target are compared
    done: float = 0.06               # an area counts as finished below this missing darkness
    stroke_px: float = 400.0         # typical stroke length before lifting (px); random around this
    stroke_spread: float = 0.6       # log-normal spread of stroke lengths
    cross_prob: float = 0.08         # how easily the centre slips through a contour (0 = never)
    wall_margin: float = 3.0         # contour repulsion range beyond the loop radius (px)
    max_ink_px: float = 0.0          # safety cap on total line length (0 = auto)
    seed: int = 0


def _kernel(sigma):
    r = max(1, int(math.ceil(2.5 * sigma)))
    x = np.arange(-r, r + 1)
    g = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum(), r


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def freehand(target, theta, cfg: FreehandConfig, walls=None, progress=None):
    """target: darkness still needed (0..1, H x W); theta: flow orientation
    (x=col, y=row, axial). walls: optional bool mask of contour lines.
    Returns strokes as lists of (row, col) floats, plus stats."""
    rng = np.random.default_rng(cfg.seed)
    H, W = target.shape
    tone = np.clip(target, 0, 1).astype(float)
    D = tone.copy()                                           # still-needed darkness (viewing scale)
    cov = np.zeros((H + 1, W + 1))                            # pixel ink coverage, for saturation
    K, kr = _kernel(cfg.view_px)
    th = theta

    if walls is None:
        walls = np.zeros((H, W), dtype=bool)
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    dist = ndimage.distance_transform_edt(~(walls | border))
    gy, gx = np.gradient(ndimage.gaussian_filter(dist, 1.0))  # points away from walls

    B = 4                                                     # block size for sampling new starts
    Hb, Wb = (H + B - 1) // B, (W + B - 1) // B

    def sample_start():
        """New start ~ (darkness still missing)^1.5 at block level, None when done."""
        Dp = np.zeros((Hb * B, Wb * B)); Dp[:H, :W] = D
        blk = Dp.reshape(Hb, B, Wb, B).mean(axis=(1, 3))
        w = np.clip(blk - cfg.done, 0, None) ** 1.5
        tot = w.sum()
        if tot <= 0:
            return None
        k = int(np.searchsorted(np.cumsum(w.ravel()), rng.random() * tot, side="right"))
        k = min(k, w.size - 1)
        by, bx = divmod(k, Wb)
        return (min(H - 2.0, by * B + rng.random() * B + 1), min(W - 2.0, bx * B + rng.random() * B + 1))

    def deposit(py, px, seg):
        """Ink of a pen segment: bilinear share over 4 pixels, saturating, spread at view scale."""
        y0, x0 = int(py), int(px)
        fy, fx = py - y0, px - x0
        ink = cfg.pen_px * seg
        for dy, dx, wgt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                            (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
            if wgt <= 0:
                continue
            qy, qx = y0 + dy, x0 + dx
            c = cov[qy, qx]
            add = ink * wgt
            new = 1.0 - (1.0 - c) * math.exp(-add)            # opaque ink stacks with saturation
            a = new - c
            cov[qy, qx] = new
            if a <= 1e-6 or qy >= H or qx >= W:
                continue
            ys0, ys1, xs0, xs1 = qy - kr, qy + kr + 1, qx - kr, qx + kr + 1
            ky0, kx0 = max(0, -ys0), max(0, -xs0)
            ky1 = K.shape[0] - max(0, ys1 - H); kx1 = K.shape[1] - max(0, xs1 - W)
            D[max(0, ys0):min(H, ys1), max(0, xs0):min(W, xs1)] -= a * K[ky0:ky1, kx0:kx1]

    total = D.sum() / max(cfg.pen_px, 1e-6)
    budget = cfg.max_ink_px or 3.0 * total + 1000
    strokes, cur = [], []
    drawn = 0.0
    sd = math.sqrt(1 - cfg.momentum ** 2)                     # OU noise scale for the given inertia

    def begin():
        st = sample_start()
        if st is None:
            return None
        y, x = st
        return dict(cy=y, cx=x, head=rng.uniform(-math.pi, math.pi), turn=0.0,
                    phase=rng.uniform(0, 2 * math.pi), spin=1.0 if rng.random() < 0.5 else -1.0,
                    r=None, rad_w=0.0, spd_w=0.0, sq=cfg.squash, ori=None,
                    left=cfg.stroke_px * math.exp(cfg.stroke_spread * rng.standard_normal()), idle=0)

    s = begin()
    if s is None:
        return [], {"strokes": 0, "length_px": 0.0, "left": float(np.clip(D, 0, None).mean())}
    step_i = 0
    while drawn < budget:
        cy, cx = s["cy"], s["cx"]
        iy, ix = int(cy), int(cx)
        need = D[iy, ix]

        if step_i % 3 == 0:
            # ---- desired direction: along the flow (sign closest to heading) + towards missing ink
            fa = th[iy, ix]
            if abs(_wrap(fa - s["head"])) > math.pi / 2:
                fa += math.pi
            R = 2 * cfg.loop_max + 3
            tx = ty = 0.0
            best = 0.0
            for k in range(8):
                a = k * math.pi / 4
                sy, sx = int(cy + R * math.sin(a)), int(cx + R * math.cos(a))
                if 0 <= sy < H and 0 <= sx < W:
                    v = max(0.0, D[sy, sx])
                    tx += v * math.cos(a); ty += v * math.sin(a)
                    best = max(best, v)
            want_x = cfg.flow_pull * math.cos(fa) + cfg.tone_pull * tx
            want_y = cfg.flow_pull * math.sin(fa) + cfg.tone_pull * ty
            # soft repulsion from contours: grows as the centre gets within margin of a wall
            reach = (s["r"] or cfg.loop_max) + cfg.wall_margin
            dwall = dist[iy, ix]
            if dwall < reach:
                g = math.hypot(gx[iy, ix], gy[iy, ix]) or 1.0
                push = (1 - dwall / reach) ** 2 * (1.0 - cfg.cross_prob) * 4.0
                want_x += push * gx[iy, ix] / g
                want_y += push * gy[iy, ix] / g
            err = _wrap(math.atan2(want_y, want_x) - s["head"]) if (want_x or want_y) else 0.0
            # ---- turn rate: inertia + gentle steering + random drift (OU process)
            m = cfg.momentum
            s["turn"] = m * s["turn"] + (1 - m) * 0.5 * err + cfg.turn_noise * sd * 0.3 * rng.standard_normal()
            s["turn"] = max(-0.35, min(0.35, s["turn"]))
            # ---- hand rhythm drifts (mean-reverting random processes)
            s["rad_w"] += 0.05 * (0.0 - s["rad_w"]) + cfg.wobble * 0.3 * rng.standard_normal()
            s["spd_w"] += 0.05 * (0.0 - s["spd_w"]) + cfg.wobble * 0.3 * rng.standard_normal()
            s["sq"] = min(0.85, max(0.0, s["sq"] + 0.05 * (cfg.squash - s["sq"])
                                    + cfg.wobble * 0.1 * rng.standard_normal()))
            if rng.random() < 0.003:
                s["spin"] = -s["spin"]
            # ---- stroke finished? lift, sample a new start elsewhere
            s["idle"] = s["idle"] + 1 if max(best, need) < cfg.done else 0
            if s["idle"] > 8 or s["left"] <= 0:
                if len(cur) > 1:
                    strokes.append(cur)
                cur = []
                s = begin()
                if s is None:
                    break
                step_i += 1
                continue

        # ---- heading integrates the turn rate (smooth arcs)
        s["head"] = _wrap(s["head"] + s["turn"] * cfg.step)
        hx, hy = math.cos(s["head"]), math.sin(s["head"])

        # ---- loop radius / speed glide towards tone-driven targets
        t = min(1.0, max(0.0, tone[iy, ix]))
        r_t = (cfg.loop_max - (cfg.loop_max - cfg.loop_min) * t ** 0.7) * math.exp(0.5 * s["rad_w"])
        r_t = max(cfg.loop_min, min(r_t, max(cfg.loop_min, dist[iy, ix] - 0.5)))
        s["r"] = r_t if s["r"] is None else s["r"] + 0.08 * (r_t - s["r"])
        r = s["r"]
        adv = cfg.advance * (0.4 + 1.2 * (1 - min(1.0, max(0.0, need)))) * math.exp(0.5 * s["spd_w"])

        # ---- move the centre (hard stop only at the paper edge)
        ny_, nx_ = cy + hy * adv * cfg.step, cx + hx * adv * cfg.step
        if not (1 <= ny_ < H - 2 and 1 <= nx_ < W - 2):
            s["head"] = _wrap(s["head"] + math.pi * 0.5 * (1 if s["turn"] >= 0 else -1))
            ny_, nx_ = cy, cx
        s["cy"], s["cx"] = ny_, nx_

        # ---- ellipse orientation glides towards the flow
        fo = th[int(ny_), int(nx_)]
        if s["ori"] is None:
            s["ori"] = fo
        else:
            d = _wrap(2 * (fo - s["ori"])) / 2                # axial difference
            s["ori"] += 0.05 * d

        # ---- pen on the rotating, squashed loop
        s["phase"] += s["spin"] * cfg.step / max(r, 0.6)
        ca, sa = math.cos(s["ori"]), math.sin(s["ori"])
        u, v = r * math.cos(s["phase"]), r * (1 - s["sq"]) * math.sin(s["phase"])
        py = min(max(ny_ + u * sa + v * ca, 0.0), H - 1.001)
        px = min(max(nx_ + u * ca - v * sa, 0.0), W - 1.001)
        seg = math.hypot(py - cur[-1][0], px - cur[-1][1]) if cur else 0.0
        cur.append((py, px))
        drawn += seg
        s["left"] -= seg
        if seg > 0:
            deposit(py, px, seg)
        step_i += 1
        if progress and step_i % 300000 == 0:
            progress(f"  freehand: {drawn:.0f} px drawn, still needed {np.clip(D, 0, None).mean():.3f}")
    if len(cur) > 1:
        strokes.append(cur)
    return strokes, {"strokes": len(strokes), "length_px": drawn,
                     "left": float(np.clip(D, 0, None).mean())}
