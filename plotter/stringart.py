"""
Pen-plotter "string art": one continuous line hopping between nails placed
on a circle, chosen greedily to reproduce a portrait.

Unlike thread, ink has a fixed physical width and stacks opaquely, so the
simulation is done in physical units:

* the canvas is a grid of `pixel_mm` cells (think of it as how far away the
  viewer stands - each cell is what the eye averages);
* a line of length L and pen width w deposits w*L mm^2 of ink, spread over the
  cells it crosses, so each crossed cell gains alpha = w*L / (cells * pixel^2);
* ink overlaps like opaque dots: coverage' = 1 - (1 - coverage)(1 - alpha).
  Dark areas saturate, and once a region reaches its target darkness further
  lines through it *increase* the error, so the greedy search stops
  piling lines there. That is what prevents the "black mess" in the centre.

Each step picks the next nail that most reduces the weighted squared error
between target darkness and simulated coverage.
"""

from dataclasses import dataclass, asdict
import json
import math

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter

from radon_renderer.config import PAPER_SIZES


@dataclass
class StringArtConfig:
    nails: int = 240
    max_lines: int = 3000
    paper: str = "A4"
    portrait: bool = False          # paper orientation
    padding_mm: float = 10.0
    diameter_mm: float | None = None  # default: largest circle on the paper
    pen_mm: float = 0.3
    pixel_mm: float = 0.6           # simulation cell size (viewing resolution)
    min_skip: int = 15              # minimum nails between endpoints (no short edge chords)
    max_darkness: float = 0.9       # darkest target tone (1 = solid ink)
    gamma: float = 1.0              # >1 lightens midtones, <1 darkens
    clahe: float = 0.02             # local contrast boost (0 = off); brings out eyes/mouth
    vignette: float = 0.0           # 0..1 fade target to white towards the rim (busy backgrounds)
    contrast_clip: float = 1.0      # percentile clip for auto-contrast
    edge_boost: float = 1.0         # extra error weight on edges (features)
    stop_ratio: float = 0.0         # stop when best gain < ratio * first gain
    overshoot: float = 3.0          # penalty multiplier for ink over light areas (keeps highlights clean)
    continuous: bool = False        # True = single unbroken path like real thread
    nail_spread_mm: float = 0.0     # spread endpoints around each nail (0 = off)
    frame: bool = False             # draw the circle outline too
    crop: tuple | None = None       # (x, y, w, h) in source pixels; None = auto


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _auto_crop_box(gray):
    """Trim uniform near-white borders, then take the centred square."""
    a = np.asarray(gray)
    ys, xs = np.where(a < 245)
    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, a.shape[1], a.shape[0]
    else:
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    s = min(w, h)
    # portraits: keep the top part (faces are rarely at the bottom)
    return (x0 + (w - s) // 2, y0 + max(0, (h - s) // 4), s, s)


def prepare_target(image_path, cfg, size_px):
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("L")
    box = cfg.crop or _auto_crop_box(img)
    x, y, w, h = box
    img = img.crop((x, y, x + w, y + h)).resize((size_px, size_px), Image.Resampling.LANCZOS)

    g = np.asarray(img, dtype=float) / 255.0
    if cfg.clahe > 0:
        from skimage.exposure import equalize_adapthist
        g = equalize_adapthist(g, kernel_size=max(8, size_px // 8), clip_limit=cfg.clahe)
    lo, hi = np.percentile(g, [cfg.contrast_clip, 100 - cfg.contrast_clip])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** cfg.gamma
    darkness = (1.0 - g) * cfg.max_darkness

    # error weights: emphasise edges so eyes / mouth / jaw survive
    blurred = img.filter(ImageFilter.GaussianBlur(1.0))
    e = np.asarray(blurred.filter(ImageFilter.FIND_EDGES), dtype=float)
    e = e / (np.percentile(e, 99) + 1e-6)
    weight = 1.0 + cfg.edge_boost * np.clip(e, 0, 1)

    yy, xx = np.mgrid[:size_px, :size_px]
    c = (size_px - 1) / 2
    mask = (xx - c) ** 2 + (yy - c) ** 2 <= (size_px / 2) ** 2
    if cfg.vignette > 0:
        rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size_px / 2)
        darkness *= 1 - cfg.vignette * np.clip((rr - 0.6) / 0.4, 0, 1) ** 2
    darkness *= mask
    weight *= mask
    return darkness, weight, mask, box


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def nail_angles(n):
    return np.arange(n) * 2 * np.pi / n - np.pi / 2  # nail 0 at the top


def _line_cells(p, q, size):
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    m = max(2, int(math.ceil(length * 2)) + 1)
    xs = np.rint(np.linspace(p[0], q[0], m)).astype(np.int32)
    ys = np.rint(np.linspace(p[1], q[1], m)).astype(np.int32)
    ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    return np.unique(ys[ok] * size + xs[ok]), length


class LineTable:
    """Pre-rasterised cells for every nail pair, grouped per start nail so a
    whole fan of candidate lines is scored with one vectorised gather."""

    def __init__(self, n, size, pixel_mm, pen_mm):
        ang = nail_angles(n)
        c = (size - 1) / 2
        r = size / 2 - 0.5
        pts = np.stack([c + r * np.cos(ang), c + r * np.sin(ang)], axis=1)

        cells = {}
        alpha = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                idx, length_px = _line_cells(pts[i], pts[j], size)
                cells[(i, j)] = idx
                ink = pen_mm * length_px * pixel_mm          # mm^2 of ink
                a = min(1.0, ink / (max(len(idx), 1) * pixel_mm ** 2))
                alpha[i, j] = alpha[j, i] = a
        self.alpha = alpha
        self.n = n

        # per start nail: concatenated cells of lines to every other nail
        self.fan_cells, self.fan_starts, self.fan_alpha = [], [], []
        for i in range(n):
            chunks, starts, alph, pos = [], [], [], 0
            for j in range(n):
                idx = cells[(min(i, j), max(i, j))] if i != j else np.zeros(1, np.int32)
                starts.append(pos)
                chunks.append(idx)
                alph.append(np.full(len(idx), alpha[i, j] if i != j else 0.0))
                pos += len(idx)
            self.fan_cells.append(np.concatenate(chunks))
            self.fan_starts.append(np.array(starts))
            self.fan_alpha.append(np.concatenate(alph))
        self.cells = cells


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _fan_gain(table, cur, t, w, cov, overshoot):
    cells = table.fan_cells[cur]
    cg = cov[cells]
    d = table.fan_alpha[cur] * (1.0 - cg)   # coverage increase
    r = t[cells] - cg                       # residual before (>0: needs ink)
    r2 = r - d                              # residual after
    before = np.where(r > 0, r * r, overshoot * r * r)
    after = np.where(r2 > 0, r2 * r2, overshoot * r2 * r2)
    return np.add.reduceat(w[cells] * (before - after), table.fan_starts[cur])


def solve(target, weight, cfg, table=None, progress=None):
    """Greedy solver. Returns (strokes, coverage, error_log).

    strokes is a list of nail-index lists; each one is drawn pen-down.
    When no line from the current nail helps any more, the pen lifts and
    jumps to the globally best line (unless cfg.continuous, which mimics
    real thread and simply stops).
    """
    size = target.shape[0]
    n = cfg.nails
    table = table or LineTable(n, size, cfg.pixel_mm, cfg.pen_mm)

    t = target.ravel()
    w = weight.ravel()
    cov = np.zeros_like(t)

    ring = np.arange(n)
    dist = np.abs(ring[:, None] - ring[None, :])
    blocked = np.minimum(dist, n - dist) < cfg.min_skip
    used = np.zeros((n, n), dtype=bool)
    strokes = []
    cur = None
    first_gain = None
    errors = []

    def gains_from(i):
        g = _fan_gain(table, i, t, w, cov, cfg.overshoot)
        g[blocked[i] | used[i]] = -np.inf
        return g

    for step in range(cfg.max_lines):
        nxt, g = None, -np.inf
        if cur is not None:
            gain = gains_from(cur)
            nxt = int(np.argmax(gain))
            g = gain[nxt]
        if g <= 0:
            if cfg.continuous and cur is not None:
                break
            # pen up: best line anywhere
            best = (-np.inf, None, None)
            for i in range(n):
                gi = gains_from(i)
                j = int(np.argmax(gi))
                if gi[j] > best[0]:
                    best = (gi[j], i, j)
            g, start, nxt = best
            if not np.isfinite(g) or g <= 0:
                break
            strokes.append([start])
            cur = start
        if first_gain is None:
            first_gain = g
        elif g < cfg.stop_ratio * first_gain:
            break

        idx = table.cells[(min(cur, nxt), max(cur, nxt))]
        cov[idx] = 1.0 - (1.0 - cov[idx]) * (1.0 - table.alpha[cur, nxt])
        used[cur, nxt] = used[nxt, cur] = True
        strokes[-1].append(nxt)
        cur = nxt

        if progress and step % 250 == 0:
            err = float(np.sum(w * (t - cov) ** 2) / np.sum(w))
            errors.append((step, err))
            progress(step, err)

    return strokes, cov.reshape(target.shape), errors


# ---------------------------------------------------------------------------
# Output geometry (mm, y-down, paper coordinates)
# ---------------------------------------------------------------------------

def layout(cfg):
    pw, ph = PAPER_SIZES[cfg.paper]
    if cfg.portrait:
        pw, ph = min(pw, ph), max(pw, ph)
    dia = cfg.diameter_mm or (min(pw, ph) - 2 * cfg.padding_mm)
    return pw, ph, (pw / 2, ph / 2), dia / 2


def strokes_to_polylines(strokes, cfg):
    """Nail strokes -> pen-down polylines in paper mm.

    With nail_spread_mm > 0, each visit to a nail lands at a slightly
    different point (low-discrepancy offset along the rim) so the hundreds of
    lines meeting at a nail don't pool into a blob of ink.
    """
    pw, ph, (cx, cy), R = layout(cfg)
    ang = nail_angles(cfg.nails)
    visits = np.zeros(cfg.nails, dtype=int)
    golden = (math.sqrt(5) - 1) / 2
    out = []
    for seq in strokes:
        pts = []
        for k in seq:
            a = ang[k]
            if cfg.nail_spread_mm > 0:
                off = ((visits[k] * golden) % 1.0 - 0.5) * cfg.nail_spread_mm
                a += off / R
                visits[k] += 1
            pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
        out.append(pts)
    return out


def circle_polyline(cfg, segments=720):
    _, _, (cx, cy), R = layout(cfg)
    return [(cx + R * math.cos(t), cy + R * math.sin(t))
            for t in np.linspace(0, 2 * np.pi, segments + 1)]


def render_preview(polylines, cfg, path, px_per_mm=6, supersample=3):
    """Opaque-ink preview at true pen width (what the paper will look like)."""
    pw, ph, _, _ = layout(cfg)
    s = px_per_mm * supersample
    im = Image.new("L", (int(pw * s), int(ph * s)), 255)
    dr = ImageDraw.Draw(im)
    width = max(1, round(cfg.pen_mm * s))
    for pl in polylines:
        dr.line([(x * s, y * s) for x, y in pl], fill=0, width=width)
    im = im.resize((int(pw * px_per_mm), int(ph * px_per_mm)), Image.Resampling.LANCZOS)
    im.save(path)
    return im


def save_json(path, strokes, cfg, box, stats):
    with open(path, "w") as f:
        json.dump({
            "config": asdict(cfg),
            "crop_box": [int(v) for v in box],
            "strokes": [[int(k) for k in st] for st in strokes],
            "stats": stats,
        }, f, indent=1)
