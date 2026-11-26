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
    min_len_spacings: float = 4.0     # drop strokes shorter than this many local spacings
    # optional improvements (defaults keep the original look)
    tone_model: str = "linear"        # linear | coverage (spacing from real ink coverage of crossing layers)
    merge_gap_mm: float = 0.0         # join aligned stroke ends closer than this (fewer pen lifts); 0 = off
    merge_angle_deg: float = 50.0
    outline_levels: tuple = ()        # darkness iso-levels to draw as contour lines, e.g. (0.35, 0.65)
    outline_smooth_mm: float = 2.5
    outline_min_len_mm: float = 15.0        # stop at sharp turns / field singularities
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
    """Boolean raster with variable-radius disk stamping.

    Backed by a bytearray so the tracer can read single cells at Python
    speed, with a numpy view over the same memory for vectorised stamping.
    """

    def __init__(self, shape):
        self.h, self.w = shape
        self.buf = bytearray(self.h * self.w)
        self.a = np.frombuffer(self.buf, dtype=np.uint8).reshape(shape)
        self.disks = {}

    def stamp(self, ys, xs, radii):
        k = np.maximum(1, np.ceil(radii).astype(np.int64))
        iy = ys.astype(np.int64)
        ix = xs.astype(np.int64)
        for kk in np.unique(k):
            if kk not in self.disks:
                self.disks[kk] = _disk(kk)
            dy, dx = self.disks[kk]
            sel = k == kk
            yy = np.clip(iy[sel, None] + dy[None, :], 0, self.h - 1)
            xx = np.clip(ix[sel, None] + dx[None, :], 0, self.w - 1)
            self.a[yy, xx] = 1

    def hit(self, y, x):
        return self.buf[int(y) * self.w + int(x)]


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker, iterative (same result as the recursive form)."""
    n = len(pts)
    if n < 3:
        return pts
    p = np.asarray(pts, dtype=float)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        a, b = p[i0], p[i1]
        seg = p[i0:i1 + 1]
        ab = b - a
        nrm = np.hypot(*ab)
        if nrm < 1e-12:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / nrm
        i = int(np.argmax(d))
        if d[i] > eps:
            keep[i0 + i] = True
            stack.append((i0, i0 + i))
            stack.append((i0 + i, i1))
    return [pts[i] for i in np.flatnonzero(keep)]


class _Shared:
    """Per-image data reused by every layer."""

    def __init__(self, darkness, mask, cfg, px):
        self.tone = (ndimage.gaussian_filter(darkness, cfg.tone_smooth_mm / px)
                     if cfg.tone_smooth_mm else darkness)
        self.tone_l = self.tone.ravel().tolist()
        self.mask_l = mask.ravel().tolist()


def hatch_layer(darkness, mask, theta, threshold, cfg, px, rng, shared=None):
    """Evenly spaced streamlines for one layer. Returns polylines in cell units."""
    h, w = darkness.shape
    smin, smax = cfg.s_min() / px, cfg.spacing_max_mm / px
    shared = shared or _Shared(darkness, mask, cfg, px)
    tone = shared.tone
    u = np.clip((tone - threshold) / max(1e-6, 1 - threshold), 0, 1)
    seedable = mask & (tone > threshold + cfg.seed_margin)
    if cfg.tone_model == "coverage":
        # each active layer adds coverage c = pen / spacing; crossing layers combine
        # like random opaque ink: 1 - prod(1 - c). Solve for c so the total matches tone.
        thr_all = np.asarray(cfg.layer_thresholds, dtype=float)
        n_active = np.maximum(1, (tone[..., None] > thr_all).sum(-1))
        cov = 1.0 - (1.0 - np.clip(tone, 0.0, 0.999)) ** (1.0 / n_active)
        spacing = np.clip(cfg.pen_mm / np.maximum(cov, 1e-6) / px, smin, smax)
    elif cfg.tone_model == "linear":
        spacing = smax - (smax - smin) * u        # cells
    else:
        raise ValueError(f"unknown tone_model {cfg.tone_model!r}")
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # flat Python lists: scalar reads are ~5x faster than numpy indexing in the tracer
    cos_l = cos_t.ravel().tolist()
    sin_l = sin_t.ravel().tolist()
    tone_l = shared.tone_l
    mask_l = shared.mask_l

    sep = _Stamp(darkness.shape)                  # radius = spacing: no new seeds inside
    stop = _Stamp(darkness.shape)                 # radius = spacing/2: tracing stops
    stop_b, sep_b = stop.buf, sep.buf
    step = cfg.step_mm / px
    half = step / 2
    max_steps = int(cfg.max_len_mm / cfg.step_mm)
    cos_turn = math.cos(math.radians(cfg.max_turn_deg))

    def trace(y, x, sign, end_thr):
        pts = []
        py, px_ = y, x
        has_prev = False
        pdx = pdy = 0.0
        for _ in range(max_steps):
            c = int(py) * w + int(px_)
            dx, dy = cos_l[c], sin_l[c]
            if not has_prev:
                dx *= sign; dy *= sign
            elif dx * pdx + dy * pdy < 0:
                dx = -dx; dy = -dy
            # midpoint (RK2)
            my, mx = py + dy * step / 2, px_ + dx * step / 2
            if not (0 <= my < h and 0 <= mx < w):
                break
            c = int(my) * w + int(mx)
            d2x, d2y = cos_l[c], sin_l[c]
            if d2x * dx + d2y * dy < 0:
                d2x = -d2x; d2y = -d2y
            if has_prev and d2x * pdx + d2y * pdy < cos_turn:
                break
            ny, nx = py + d2y * step, px_ + d2x * step
            if not (0 <= ny < h and 0 <= nx < w):
                break
            c = int(ny) * w + int(nx)
            if not mask_l[c] or not tone_l[c] > end_thr or stop_b[c]:
                break
            pts.append((ny, nx))
            py, px_, pdx, pdy, has_prev = ny, nx, d2x, d2y, True
        return pts

    # jittered seed grid (same random stream as drawing it point by point),
    # shuffled, then pre-filtered to cells where this layer may start a line
    gs = max(1, int(smin))
    gy, gx = np.mgrid[0:h - gs:gs, 0:w - gs:gs]
    r = rng.random(2 * gy.size).reshape(-1, 2)
    seeds_y = gy.ravel() + r[:, 0] * gs
    seeds_x = gx.ravel() + r[:, 1] * gs
    order = list(range(gy.size))
    rng.shuffle(order)
    order = np.asarray(order, dtype=np.int64)
    seeds_y, seeds_x = seeds_y[order], seeds_x[order]
    ok_seed = seedable[seeds_y.astype(np.int64), seeds_x.astype(np.int64)]
    grid = list(zip(seeds_y[ok_seed].tolist(), seeds_x[ok_seed].tolist()))
    seedable_l = seedable.ravel().tolist()

    lines = []
    queue = deque()
    gi = 0
    side_k = max(1, int(smin / step))
    while True:
        if queue:
            y, x = queue.popleft()
            if not (0 <= y < h and 0 <= x < w):
                continue
        elif gi < len(grid):
            y, x = grid[gi]; gi += 1
        else:
            break
        c = int(y) * w + int(x)
        if not seedable_l[c] or sep_b[c]:
            continue
        # staggered per-line end threshold: no stubs at ragged edges, no aligned seams
        end_thr = threshold - cfg.overrun * rng.random()
        fwd = trace(y, x, 1.0, end_thr)
        bwd = trace(y, x, -1.0, end_thr)
        pts = bwd[::-1] + [(y, x)] + fwd
        arr = np.asarray(pts)
        ys, xs = arr[:, 0], arr[:, 1]
        sp = spacing[ys.astype(np.int64), xs.astype(np.int64)]
        length = (len(pts) - 1) * step               # cells
        if length * px < cfg.min_len_mm or length < cfg.min_len_spacings * sp.mean():
            continue
        stop.stamp(ys[::2], xs[::2], sp[::2] * 0.5)
        sep.stamp(ys[::2], xs[::2], sp[::2] * 0.95)
        lines.append(pts)
        # side seeds (Jobard-Lefer): one spacing away on both sides
        idx = np.arange(0, len(pts), side_k)
        cc = ys[idx].astype(np.int64) * w + xs[idx].astype(np.int64)
        ny_, nx_ = cos_t.ravel()[cc], -sin_t.ravel()[cc]    # normal (y, x)
        s_ = sp[idx]
        a_y, a_x = ys[idx] + ny_ * s_, xs[idx] + nx_ * s_
        b_y, b_x = ys[idx] - ny_ * s_, xs[idx] - nx_ * s_
        for t in zip(a_y.tolist(), a_x.tolist(), b_y.tolist(), b_x.tolist()):
            queue.append((t[0], t[1]))
            queue.append((t[2], t[3]))
    return lines


def merge_strokes(lines, max_gap, max_angle_deg):
    """Join stroke ends that are close and roughly continue each other.

    Each end links to at most one other end, no cycles; the short connector is
    drawn pen-down. Returns fewer, longer polylines (same points, same order
    within each stroke) -> fewer pen lifts.
    """
    from scipy.spatial import cKDTree
    n = len(lines)
    if n < 2 or max_gap <= 0:
        return lines
    P = np.array([[l[0], l[-1]] for l in lines], dtype=float)          # (n, 2 ends, xy)
    out = np.array([[np.subtract(l[0], l[1]), np.subtract(l[-1], l[-2])] for l in lines], dtype=float)
    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12
    pts = P.reshape(-1, 2)
    cos_max = math.cos(math.radians(max_angle_deg))
    pairs = cKDTree(pts).query_pairs(max_gap, output_type="ndarray")
    cand = []
    for a, b in pairs:
        i, j = a // 2, b // 2
        if i == j:
            continue
        oa, ob = out[i, a % 2], out[j, b % 2]
        v = pts[b] - pts[a]
        d = float(np.hypot(*v))
        if d > 1e-9:
            v /= d
            if v @ oa < cos_max or -(v @ ob) < cos_max:
                continue
        elif -(oa @ ob) < cos_max:
            continue
        cand.append((d, a, b))
    cand.sort()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    link = {}
    for d, a, b in cand:
        if a in link or b in link:
            continue
        ri, rj = find(a // 2), find(b // 2)
        if ri == rj:
            continue
        parent[ri] = rj
        link[a] = b
        link[b] = a

    used = [False] * n
    merged = []

    def walk(i, enter_end):
        chain = []
        while True:
            used[i] = True
            seg = lines[i] if enter_end == 0 else lines[i][::-1]
            chain.extend(seg)
            exit_end = 1 - enter_end
            nxt = link.get(2 * i + exit_end)
            if nxt is None or used[nxt // 2]:
                return chain
            i, enter_end = nxt // 2, nxt % 2

    for i in range(n):                       # start chains at free ends
        if used[i]:
            continue
        if (2 * i) not in link:
            merged.append(walk(i, 0))
        elif (2 * i + 1) not in link:
            merged.append(walk(i, 1))
    for i in range(n):                       # (cannot happen: no cycles) safety
        if not used[i]:
            merged.append(walk(i, 0))
    return merged


def outline_contours(darkness, mask, cfg, px):
    """Iso-darkness contour lines (cell units), clipped away from the mask edge."""
    from skimage.measure import find_contours
    if not cfg.outline_levels:
        return []
    d = ndimage.gaussian_filter(darkness, cfg.outline_smooth_mm / px)
    inner = ndimage.binary_erosion(mask, iterations=3)
    h, w = d.shape
    out = []
    for level in cfg.outline_levels:
        for c in find_contours(d, level):
            iy = np.clip(c[:, 0].astype(int), 0, h - 1)
            ix = np.clip(c[:, 1].astype(int), 0, w - 1)
            ok = inner[iy, ix]
            # split where the contour leaves the drawable area
            breaks = np.flatnonzero(np.diff(ok.astype(np.int8)) != 0) + 1
            for part, okp in zip(np.split(c, breaks), np.split(ok, breaks)):
                if not okp[0] or len(part) < 2:
                    continue
                seglen = np.hypot(*np.diff(part, axis=0).T).sum() * px
                if seglen >= cfg.outline_min_len_mm:
                    out.append([(float(y), float(x)) for y, x in part])
    return out


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
    shared = _Shared(darkness, mask, cfg, px)

    layers = []
    for k, (off, thr) in enumerate(zip(cfg.layer_angles, cfg.layer_thresholds)):
        theta = base + math.radians(off)
        cell_lines = hatch_layer(darkness, mask, theta, thr, cfg, px, rng, shared)
        mm = []
        x0, y0 = cx - R, cy - R
        for pts in cell_lines:
            pl = [(x0 + (x + 0.5) * px, y0 + (y + 0.5) * px) for y, x in pts]
            mm.append(_rdp(pl, 0.03))
        if cfg.merge_gap_mm > 0:
            before = len(mm)
            mm = merge_strokes(mm, cfg.merge_gap_mm, cfg.merge_angle_deg)
            if progress:
                progress(f"  layer {k+1}: merged {before} -> {len(mm)} strokes")
        layers.append(mm)
        if progress:
            progress(f"  layer {k+1}: offset {off:+.0f}°, from darkness {thr:.2f}: {len(mm)} strokes")
    if cfg.outline_levels:
        x0, y0 = cx - R, cy - R
        outl = [_rdp([(x0 + (x + 0.5) * px, y0 + (y + 0.5) * px) for y, x in c], 0.03)
                for c in outline_contours(darkness, mask, cfg, px)]
        layers.append(outl)                   # drawn last, on top
        if progress:
            progress(f"  outline: {len(outl)} contours at levels {list(cfg.outline_levels)}")
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
