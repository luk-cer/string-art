"""
Flow hatching: curved, tone-dependent crosshatch.

Instead of families of straight parallel chords, strokes are streamlines of a
direction field derived from the image itself (structure tensor):

* layer 1 follows the contours (iso-brightness lines) - strokes wrap around
  the face, along the hair, around the jaw;
* layer 2 runs across layer 1 (follows the gradient) and only appears in
  darker tones;
* layers 3/4 are diagonal to those, for the darkest tones.

Within each layer, line spacing varies continuously with darkness (dark =
dense, light = sparse) and is never tighter than a multiple of the pen width,
so layers stay readable instead of merging into a black mess. Streamlines are
placed with an evenly-spaced-streamlines scheme (Jobard & Lefer 1997): seed
new lines beside finished ones, stop a line when it gets too close to another.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np
from scipy import ndimage

from . import stringart as sa


@dataclass
class HatchConfig:
    # layout (shared with string art)
    paper: str = "A4"
    portrait: bool = False
    padding_mm: float = 10.0
    diameter_mm: float | None = None
    shape: str = "circle"            # circle | rect (full square)
    pen_mm: float = 0.3
    # image prep (fields used by stringart.prepare_target)
    crop: tuple | None = None
    gamma: float = 1.0
    contrast_clip: float = 1.0
    clahe: float = 0.0
    vignette: float = 0.0
    max_darkness: float = 1.0
    edge_boost: float = 0.0
    # hatching
    cell_mm: float = 0.25             # field / occupancy raster resolution
    layer_angles: tuple = (0.0, 90.0, 45.0, -45.0)   # offsets from the flow direction
    layer_thresholds: tuple = (0.12, 0.40, 0.62, 0.80)  # darkness where each layer starts
    spacing_max_mm: float = 2.0       # spacing where a layer fades in
    spacing_min_mm: float | None = None  # darkest spacing (default 2.5 x pen)
    flow_smooth_mm: float = 4.0       # bigger = calmer, broader curves
    flow_bias_angle: float = 30.0     # direction used in flat areas
    flow_bias: float = 0.15           # how strongly flat areas fall back to the bias angle
    flow: str = "contour"             # contour | gradient | fixed
    step_mm: float = 0.5
    min_len_mm: float = 5.0
    max_len_mm: float = 120.0
    max_turn_deg: float = 35.0
    # anti-stub / anti-seam
    tone_smooth_mm: float = 1.0       # smooth the darkness map that switches layers on/off
    seed_margin: float = 0.05         # only start lines this much darker than the layer threshold
    overrun: float = 0.10             # a started line may continue until threshold - U(0, overrun)
    min_len_spacings: float = 4.0     # drop strokes shorter than this many local spacings        # stop at sharp turns / field singularities
    seed: int = 1

    def s_min(self):
        return self.spacing_min_mm or 2.5 * self.pen_mm


def direction_field(gray, cfg, px):
    """Unit stroke direction (axial) per cell, as an angle array (radians)."""
    if cfg.flow == "fixed":
        return np.full(gray.shape, math.radians(cfg.flow_bias_angle))
    g = ndimage.gaussian_filter(gray, 1.0)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    s = cfg.flow_smooth_mm / px
    j11 = ndimage.gaussian_filter(gx * gx, s)
    j22 = ndimage.gaussian_filter(gy * gy, s)
    j12 = ndimage.gaussian_filter(gx * gy, s)
    # doubled-angle vector of the gradient orientation, plus a constant bias
    vx, vy = j11 - j22, 2 * j12
    mag = np.hypot(vx, vy)
    b = cfg.flow_bias * np.percentile(mag, 90) + 1e-12
    grad_angle_bias = math.radians(cfg.flow_bias_angle) - (math.pi / 2 if cfg.flow == "contour" else 0)
    vx = vx + b * math.cos(2 * grad_angle_bias)
    vy = vy + b * math.sin(2 * grad_angle_bias)
    grad = 0.5 * np.arctan2(vy, vx)
    return grad + (math.pi / 2 if cfg.flow == "contour" else 0.0)


def _disk(r):
    r = max(1, int(math.ceil(r)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = xx * xx + yy * yy <= r * r
    return yy[keep], xx[keep]


class _Stamp:
    """Boolean raster with variable-radius disk stamping."""

    def __init__(self, shape):
        self.a = np.zeros(shape, dtype=bool)
        self.disks = {}

    def stamp(self, ys, xs, radii):
        h, w = self.a.shape
        for y, x, r in zip(ys, xs, radii):
            k = max(1, int(math.ceil(r)))
            if k not in self.disks:
                self.disks[k] = _disk(k)
            dy, dx = self.disks[k]
            yy = np.clip(dy + int(y), 0, h - 1)
            xx = np.clip(dx + int(x), 0, w - 1)
            self.a[yy, xx] = True

    def hit(self, y, x):
        return self.a[int(y), int(x)]


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = np.array(pts[0]), np.array(pts[-1])
    p = np.array(pts)
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-12:
        d = np.hypot(*(p - a).T)
    else:
        d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return _rdp(pts[:i + 1], eps)[:-1] + _rdp(pts[i:], eps)
    return [pts[0], pts[-1]]


def hatch_layer(darkness, mask, theta, threshold, cfg, px, rng):
    """Evenly spaced streamlines for one layer. Returns polylines in cell units."""
    h, w = darkness.shape
    smin, smax = cfg.s_min() / px, cfg.spacing_max_mm / px
    tone = ndimage.gaussian_filter(darkness, cfg.tone_smooth_mm / px) if cfg.tone_smooth_mm else darkness
    u = np.clip((tone - threshold) / max(1e-6, 1 - threshold), 0, 1)
    seedable = mask & (tone > threshold + cfg.seed_margin)
    spacing = smax - (smax - smin) * u            # cells
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    sep = _Stamp(darkness.shape)                  # radius = spacing: no new seeds inside
    stop = _Stamp(darkness.shape)                 # radius = spacing/2: tracing stops
    step = cfg.step_mm / px
    max_steps = int(cfg.max_len_mm / cfg.step_mm)
    cos_turn = math.cos(math.radians(cfg.max_turn_deg))

    end_thr = threshold

    def ok(y, x):
        return (0 <= y < h and 0 <= x < w and mask[int(y), int(x)]
                and tone[int(y), int(x)] > end_thr and not stop.hit(y, x))

    def trace(y, x, sign):
        pts = []
        py, px_ = y, x
        prev = None
        for _ in range(max_steps):
            iy, ix = int(py), int(px_)
            d = np.array([cos_t[iy, ix], sin_t[iy, ix]])
            if prev is None:
                d *= sign
            elif d @ prev < 0:
                d = -d
            # midpoint (RK2)
            my, mx = py + d[1] * step / 2, px_ + d[0] * step / 2
            if not (0 <= my < h and 0 <= mx < w):
                break
            d2 = np.array([cos_t[int(my), int(mx)], sin_t[int(my), int(mx)]])
            if d2 @ d < 0:
                d2 = -d2
            if prev is not None and d2 @ prev < cos_turn:
                break
            ny, nx = py + d2[1] * step, px_ + d2[0] * step
            if not ok(ny, nx):
                break
            pts.append((ny, nx))
            py, px_, prev = ny, nx, d2
        return pts

    lines = []
    queue = deque()
    gs = max(1, int(smin))
    grid = [(y + rng.random() * gs, x + rng.random() * gs)
            for y in range(0, h - gs, gs) for x in range(0, w - gs, gs)]
    rng.shuffle(grid)
    gi = 0
    while True:
        if queue:
            y, x = queue.popleft()
        elif gi < len(grid):
            y, x = grid[gi]; gi += 1
        else:
            break
        if not (0 <= y < h and 0 <= x < w) or not seedable[int(y), int(x)] or sep.hit(y, x):
            continue
        # staggered per-line end threshold: no stubs at ragged edges, no aligned seams
        end_thr = threshold - cfg.overrun * rng.random()
        fwd = trace(y, x, 1.0)
        bwd = trace(y, x, -1.0)
        pts = bwd[::-1] + [(y, x)] + fwd
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        sp = spacing[ys.astype(int), xs.astype(int)]
        length = (len(pts) - 1) * step               # cells
        if length * px < cfg.min_len_mm or length < cfg.min_len_spacings * sp.mean():
            continue
        stop.stamp(ys[::2], xs[::2], sp[::2] * 0.5)
        sep.stamp(ys[::2], xs[::2], sp[::2] * 0.95)
        lines.append(pts)
        # side seeds (Jobard-Lefer): one spacing away on both sides
        k = max(1, int(smin / step))
        for i in range(0, len(pts), k):
            iy, ix = int(ys[i]), int(xs[i])
            ny_, nx_ = cos_t[iy, ix], -sin_t[iy, ix]   # normal (y, x)
            s = sp[i]
            queue.append((ys[i] + ny_ * s, xs[i] + nx_ * s))
            queue.append((ys[i] - ny_ * s, xs[i] - nx_ * s))
    return lines


def flow_hatch(image_path, cfg, progress=print):
    pw, ph, (cx, cy), R = sa.layout(cfg)
    px = cfg.cell_mm
    size = int(round(2 * R / px))
    if cfg.shape == "rect":
        darkness, _, mask, box = _prepare_rect(image_path, cfg, size)
    else:
        darkness, _, mask, box = sa.prepare_target(image_path, cfg, size)
    gray = 1.0 - darkness
    base = direction_field(gray, cfg, px)
    rng = np.random.default_rng(cfg.seed)

    layers = []
    for k, (off, thr) in enumerate(zip(cfg.layer_angles, cfg.layer_thresholds)):
        theta = base + math.radians(off)
        cell_lines = hatch_layer(darkness, mask, theta, thr, cfg, px, rng)
        mm = []
        x0, y0 = cx - R, cy - R
        for pts in cell_lines:
            pl = [(x0 + (x + 0.5) * px, y0 + (y + 0.5) * px) for y, x in pts]
            mm.append(_rdp(pl, 0.03))
        layers.append(mm)
        if progress:
            progress(f"  layer {k+1}: offset {off:+.0f}°, from darkness {thr:.2f}: {len(mm)} strokes")
    return layers, darkness, box


def _prepare_rect(image_path, cfg, size):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("L")
    x, y, w, h = cfg.crop or sa._auto_crop_box(img)
    img = img.crop((x, y, x + w, y + h)).resize((size, size), Image.Resampling.LANCZOS)
    g = np.asarray(img, dtype=float) / 255.0
    lo, hi = np.percentile(g, [cfg.contrast_clip, 100 - cfg.contrast_clip])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** cfg.gamma
    if cfg.clahe > 0:
        from skimage.exposure import equalize_adapthist
        g = equalize_adapthist(g, kernel_size=max(8, size // 8), clip_limit=cfg.clahe)
    return (1 - g) * cfg.max_darkness, None, np.ones_like(g, dtype=bool), (x, y, w, h)
