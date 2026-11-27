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
    min_lines: int = 0              # keep adding the best line until this many, even past the target tone
    paper: str = "A4"
    portrait: bool = False          # paper orientation
    padding_mm: float = 10.0
    diameter_mm: float | None = None  # default: largest circle on the paper
    pen_mm: float = 0.3
    pixel_mm: float = 0.6           # simulation cell size (viewing resolution)
    min_skip: int = 15              # minimum nails between endpoints (no short edge chords)
    max_darkness: float = 0.9       # darkest target tone (1 = solid ink)
    tone: float = 1.0               # overall darkness: <1 = lighter picture, fewer lines
    gamma: float = 1.0              # >1 lightens midtones, <1 darkens
    clahe: float = 0.02             # local contrast boost (0 = off); brings out eyes/mouth
    vignette: float = 0.0           # 0..1 fade target to white towards the rim (busy backgrounds)
    contrast_clip: float = 1.0      # percentile clip for auto-contrast
    edge_boost: float = 1.0         # extra error weight on edges (features)
    # importance: extra error weight where likeness matters (detected faces/eyes/mouth and/or
    # a painted mask, white = important): weight *= 1 + importance * map. 0 = off.
    importance: float = 2.0
    face_detect: bool = True
    importance_mask: str | None = None
    stop_ratio: float = 0.0         # stop when best gain < ratio * first gain
    overshoot: float = 3.0          # penalty multiplier for ink over light areas (keeps highlights clean)
    continuous: bool = False        # True = single unbroken path like real thread
    nail_spread_mm: float = 0.0     # spread endpoints around each nail (0 = off)
    frame: bool = False             # draw the circle outline too
    crop: tuple | None = None       # (x, y, w, h) in source pixels; None = auto
    # line shape: straight chords, hyperbolic (Poincare) arcs, or let the solver bend each line
    chords: str = "straight"        # straight | poincare | warp
    bends: tuple = (0.0, 0.5, 1.0, -0.5, -1.0)   # warp: bend choices (1 = Poincare arc, - = towards the rim)
    # dashed lines: lift the pen where the photo is light
    dashed: bool = False
    gap_tone: float = 0.15          # no ink where the (smoothed) target is lighter than this
    taper: float = 0.3              # darkness range over which dashes thin out into solid line (0 = hard cut)
    dash_period_mm: float = 5.0     # one dash + one gap; dash share of it follows the tone
    dash_min_mm: float = 0.5        # drop dashes shorter than this
    gap_min_mm: float = 0.4         # don't lift the pen for gaps shorter than this
    # after the greedy pass: rounds of "remove lines that hurt, add lines that help" (0 = off)
    refine: int = 10


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
    darkness = (1.0 - g) * cfg.max_darkness * getattr(cfg, "tone", 1.0)

    # error weights: emphasise edges so eyes / mouth / jaw survive
    blurred = img.filter(ImageFilter.GaussianBlur(1.0))
    e = np.asarray(blurred.filter(ImageFilter.FIND_EDGES), dtype=float)
    e = e / (np.percentile(e, 99) + 1e-6)
    weight = 1.0 + cfg.edge_boost * np.clip(e, 0, 1)
    if getattr(cfg, "importance", 0) > 0:        # string art only (other styles share this function)
        from .importance import importance_map
        imp, _ = importance_map(image_path, box, size_px, cfg.face_detect, cfg.importance_mask)
        weight *= 1.0 + cfg.importance * imp

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


def chord_bends(cfg):
    if cfg.chords == "poincare":
        return (1.0,)
    if cfg.chords == "warp":
        return tuple(cfg.bends)
    return (0.0,)


def arc_points(p, q, c, R, bend, step):
    """Points from p to q (2D, same units as c/R) along a chord of the circle
    (centre c, radius R) bent by `bend`: 0 = straight, 1 = the Poincare
    geodesic (circular arc meeting the rim at right angles, bowing towards the
    centre), negative = bowing towards the rim (at most 90% of the way), always
    inside the circle. `step` = spacing of the returned points."""
    p = np.asarray(p, float); q = np.asarray(q, float); c = np.asarray(c, float)
    chord = q - p
    L = float(np.hypot(*chord))
    m = max(2, int(math.ceil(L / step)) + 1)
    if abs(bend) < 1e-9 or L < 1e-9:
        t = np.linspace(0, 1, m)[:, None]
        return p + t * chord
    mid = (p + q) / 2
    to_c = c - mid
    dc = float(np.hypot(*to_c))
    if dc < 1e-9 * R:                              # diameter: geodesic is straight
        t = np.linspace(0, 1, m)[:, None]
        return p + t * chord
    n_in = to_c / dc
    half = 0.5 * L / R                             # sin of the half angle
    cos_h = math.sqrt(max(0.0, 1 - half * half))
    if bend > 0:                                   # towards the centre, fraction of the geodesic sagitta
        geo_apex = (1 - half) / max(cos_h, 1e-9)   # distance of the geodesic apex from the centre (unit R)
        sag = bend * (cos_h - geo_apex) * R
    else:                                          # towards the rim, capped inside
        sag = bend * 0.9 * (1 - cos_h) * R
    if abs(sag) < 1e-9:
        t = np.linspace(0, 1, m)[:, None]
        return p + t * chord
    apex = mid + sag * n_in
    # circle through p, apex, q
    h = L / 2
    rr = (h * h + sag * sag) / (2 * abs(sag))
    cc = apex - math.copysign(rr, sag) * n_in          # centre on the far side of the bulge
    a0 = math.atan2(p[1] - cc[1], p[0] - cc[0])
    a1 = math.atan2(q[1] - cc[1], q[0] - cc[0])
    am = math.atan2(apex[1] - cc[1], apex[0] - cc[0])
    # go from a0 to a1 through am
    def wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi
    d1 = wrap(a1 - a0); dm = wrap(am - a0)
    if not (0 < dm < d1 or d1 < dm < 0):
        d1 = d1 - 2 * math.pi if d1 > 0 else d1 + 2 * math.pi
    arc_len = abs(d1) * rr
    m = max(2, int(math.ceil(arc_len / step)) + 1)
    ang = a0 + np.linspace(0, d1, m)
    pts = np.stack([cc[0] + rr * np.cos(ang), cc[1] + rr * np.sin(ang)], 1)
    pts[0], pts[-1] = p, q
    return pts


def _curve_cells(pts, size):
    seg = np.hypot(*np.diff(pts, axis=0).T).sum()
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
    return np.unique(ys[ok] * size + xs[ok]), seg


class LineTable:
    """Pre-rasterised cells for every nail pair (and every bend), grouped per
    start nail so a whole fan of candidate lines is scored with one gather.
    Candidate index within a fan: j * B + b (target nail j, bend b)."""

    def __init__(self, n, size, pixel_mm, pen_mm, bends=(0.0,)):
        ang = nail_angles(n)
        c = (size - 1) / 2
        r = size / 2 - 0.5
        pts = np.stack([c + r * np.cos(ang), c + r * np.sin(ang)], axis=1)
        self.bends = tuple(bends)
        B = len(self.bends)
        self.B = B

        cells = {}
        alpha = np.zeros((n, n, B))
        for i in range(n):
            for j in range(i + 1, n):
                for bi, bend in enumerate(self.bends):
                    curve = arc_points(pts[i], pts[j], (c, c), r, bend, 0.5)
                    idx, length_px = _curve_cells(curve, size)
                    cells[(i, j, bi)] = idx
                    ink = pen_mm * length_px * pixel_mm          # mm^2 of ink
                    a = min(1.0, ink / (max(len(idx), 1) * pixel_mm ** 2))
                    alpha[i, j, bi] = alpha[j, i, bi] = a
        self.alpha = alpha
        self.n = n
        empty = np.zeros(1, np.int32)
        self.fan_cells, self.fan_starts, self.fan_alpha = [], [], []
        for i in range(n):
            chunks, starts, alph, pos = [], [], [], 0
            for j in range(n):
                for bi in range(B):
                    idx = cells[(min(i, j), max(i, j), bi)] if i != j else empty
                    starts.append(pos)
                    chunks.append(idx)
                    alph.append(np.full(len(idx), alpha[i, j, bi] if i != j else 0.0, dtype=np.float32))
                    pos += len(idx)
            self.fan_cells.append(np.concatenate(chunks).astype(np.int32))
            self.fan_starts.append(np.array(starts))
            self.fan_alpha.append(np.concatenate(alph))
        self.cells = cells

    def line_cells(self, i, j, b=0):
        return self.cells[(min(i, j), max(i, j), b)]


class Stroke(list):
    """Nail sequence of one pen-down stroke; .bends[k] is the bend index of
    the line from nail k to nail k+1."""

    def __init__(self, *a):
        super().__init__(*a)
        self.bends = []


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _fan_gain(table, cur, t, w, cov, overshoot, drawable=None):
    cells = table.fan_cells[cur]
    cg = cov[cells]
    d = table.fan_alpha[cur] * (1.0 - cg)   # coverage increase
    if drawable is not None:
        d = d * drawable[cells]             # dashed: no ink where the pen is lifted
    r = t[cells] - cg                       # residual before (>0: needs ink)
    r2 = r - d                              # residual after
    before = np.where(r > 0, r * r, overshoot * r * r)
    after = np.where(r2 > 0, r2 * r2, overshoot * r2 * r2)
    return np.add.reduceat(w[cells] * (before - after), table.fan_starts[cur])


def drawable_mask(target, cfg):
    """Dash duty per cell for dashed lines (share of the line that is inked):
    0 below gap_tone, rising to 1 over `taper` darkness, so lines fade out in
    ever shorter, sparser dashes instead of stopping abruptly."""
    from scipy.ndimage import gaussian_filter
    t = gaussian_filter(target, 1.0)
    if cfg.taper <= 0:
        return (t > cfg.gap_tone).astype(np.float32)
    return np.clip((t - cfg.gap_tone) / cfg.taper, 0.0, 1.0).astype(np.float32)


def solve(target, weight, cfg, table=None, progress=None):
    """Greedy solver. Returns (strokes, coverage, error_log).

    strokes is a list of nail-index lists; each one is drawn pen-down.
    When no line from the current nail helps any more, the pen lifts and
    jumps to the globally best line (unless cfg.continuous, which mimics
    real thread and simply stops).
    """
    size = target.shape[0]
    n = cfg.nails
    table = table or LineTable(n, size, cfg.pixel_mm, cfg.pen_mm, chord_bends(cfg))
    B = table.B

    t = target.ravel()
    w = weight.ravel()
    cov = np.zeros_like(t)
    drawable = drawable_mask(target, cfg).ravel().astype(np.float32) if cfg.dashed else None

    ring = np.arange(n)
    dist = np.abs(ring[:, None] - ring[None, :])
    blocked = np.repeat(np.minimum(dist, n - dist) < cfg.min_skip, B, axis=1)   # (n, n*B)
    used = np.zeros((n, n * B), dtype=bool)
    strokes = []
    cur = None
    first_gain = None
    errors = []

    def gains_from(i):
        g = _fan_gain(table, i, t, w, cov, cfg.overshoot, drawable)
        g[blocked[i] | used[i]] = -np.inf
        return g

    n_lines = 0
    for step in range(cfg.max_lines):
        forced = n_lines < cfg.min_lines              # below the minimum: accept lines that don't help
        nxt, g = None, -np.inf
        if cur is not None:
            gain = gains_from(cur)
            nxt = int(np.argmax(gain))
            g = gain[nxt]
        if g <= 0 and not (forced and cur is not None and np.isfinite(g)):
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
            if not np.isfinite(g) or (g <= 0 and not forced):
                break
            strokes.append(Stroke([start]))
            cur = start
        if first_gain is None:
            first_gain = g
        elif g < cfg.stop_ratio * first_gain and not forced:
            break

        j, bi = divmod(nxt, B)
        idx = table.line_cells(cur, j, bi)
        a = table.alpha[cur, j, bi]
        if drawable is not None:
            a = a * drawable[idx]
        cov[idx] = 1.0 - (1.0 - cov[idx]) * (1.0 - a)
        used[cur, nxt] = used[j, cur * B + bi] = True
        strokes[-1].append(j)
        strokes[-1].bends.append(bi)
        cur = j
        n_lines += 1

        if progress and step % 250 == 0:
            err = float(np.sum(w * (t - cov) ** 2) / np.sum(w))
            errors.append((step, err))
            progress(step, err)

    if cfg.refine > 0 and not cfg.continuous and strokes:
        strokes, cov = refine(strokes, t, w, cov, cfg, table, drawable, blocked, progress=progress)
    return strokes, cov.reshape(target.shape), errors


def refine(strokes, t, w, cov, cfg, table, drawable, blocked, progress=None):
    """Local search after the greedy solve. The greedy pass never revisits a line:
    early lines were chosen when the picture was empty and many end up redundant
    once later lines cross the same area. Each round:

    * remove: drop every line whose removal lowers the error (its ink is
      divided back out: cov' = 1 - (1 - cov) / (1 - a));
    * add: add the best lines anywhere while they still lower the error
      (lazy greedy: a line's gain only shrinks as ink is added, so a stale
      gain is an upper bound and only the top of the queue is re-scored).

    The resulting set of lines is chained into pen strokes again (chain_lines).
    Returns (strokes, cov)."""
    import heapq
    n, B, os_ = cfg.nails, table.B, cfg.overshoot

    def cells_alpha(i, j, bi):
        idx = table.line_cells(i, j, bi)
        a = np.full(len(idx), table.alpha[i, j, bi], np.float64)
        if drawable is not None:
            a = a * drawable[idx]
        return idx, np.minimum(a, 0.999)

    def err(r, ww):
        return ww * np.where(r > 0, r * r, os_ * r * r)

    def add_gain(idx, a):
        c = cov[idx]
        r = t[idx] - c
        return float(np.sum(err(r, w[idx]) - err(r - a * (1 - c), w[idx])))

    lines = {}
    for s in strokes:
        for k in range(len(s) - 1):
            i, j, bi = s[k], s[k + 1], s.bends[k]
            lines[(min(i, j), max(i, j), bi)] = None
    for key in lines:
        lines[key] = cells_alpha(*key)
    n0 = len(lines)
    removed_total = added_total = 0

    for rnd in range(cfg.refine):
        # --- remove pass: worst lines first, gains re-checked as we go
        def rm_gain(key):
            idx, a = lines[key]
            c = cov[idx]
            cw = 1 - (1 - c) / (1 - a)
            r = t[idx] - c
            return float(np.sum(err(r, w[idx]) - err(t[idx] - cw, w[idx]))), cw

        order = sorted(lines, key=lambda k: -rm_gain(k)[0])
        removed = 0
        for key in order:
            if len(lines) <= cfg.min_lines:
                break
            g, cw = rm_gain(key)
            if g > 0:
                cov[lines[key][0]] = np.maximum(cw, 0.0)
                del lines[key]
                removed += 1

        # --- add pass: lazy greedy over every allowed line
        heap = []
        for i in range(n):
            g = _fan_gain(table, i, t, w, cov, os_, drawable)
            for c in np.flatnonzero(~blocked[i] & (g > 0)):
                j, bi = divmod(int(c), B)
                if j > i and (i, j, bi) not in lines:
                    heap.append((-float(g[c]), i, j, bi))
        heapq.heapify(heap)
        added = 0
        while heap and len(lines) < cfg.max_lines:
            _, i, j, bi = heapq.heappop(heap)
            if (i, j, bi) in lines:
                continue
            idx, a = cells_alpha(i, j, bi)
            g = add_gain(idx, a)
            if g <= 0:
                continue
            if heap and g < -heap[0][0]:
                heapq.heappush(heap, (-g, i, j, bi))      # stale: re-queue with its true gain
                continue
            cov[idx] = 1 - (1 - cov[idx]) * (1 - a)
            lines[(i, j, bi)] = (idx, a)
            added += 1
        removed_total += removed
        added_total += added
        if progress:
            progress(len(lines), float(np.sum(err(t - cov, w)) / np.sum(w)))
        if removed == 0 and added == 0:
            break
    refine.last = dict(lines_before=n0, lines_after=len(lines), removed=removed_total, added=added_total)
    return chain_lines(list(lines), n), cov


def chain_lines(lines, n):
    """Order a set of nail-to-nail lines (i, j, bend) into as few pen-down strokes as
    practical: walk from nail to nail along unused lines (Hierholzer-style: start at a
    nail with an odd number of lines left, prefer moves that don't strand lines), and
    lift the pen to the nearest nail with lines left when stuck."""
    adj = [[] for _ in range(n)]
    for e, (i, j, bi) in enumerate(lines):
        adj[i].append((j, bi, e))
        adj[j].append((i, bi, e))
    used = [False] * len(lines)
    deg = [len(a) for a in adj]
    strokes = []
    cur = None
    left = len(lines)
    while left:
        if cur is None or deg[cur] == 0:
            cands = [k for k in range(n) if deg[k] > 0]
            odd = [k for k in cands if deg[k] % 2] or cands
            if cur is None:
                cur = odd[0]
            else:
                ring = lambda k: min(abs(k - cur), n - abs(k - cur))
                cur = min(odd, key=ring)
            strokes.append(Stroke([cur]))
        best = None
        for j, bi, e in adj[cur]:
            if used[e]:
                continue
            # prefer a nail that still has other lines (deg after this move > 0), i.e. not a dead end
            score = (deg[j] - 1 > 0, deg[j] % 2 == 0)
            if best is None or score > best[0]:
                best = (score, j, bi, e)
        _, j, bi, e = best
        used[e] = True
        deg[cur] -= 1
        deg[j] -= 1
        left -= 1
        strokes[-1].append(j)
        strokes[-1].bends.append(bi)
        cur = j
    return strokes


# ---------------------------------------------------------------------------
# Output geometry (mm, y-down, paper coordinates)
# ---------------------------------------------------------------------------

def layout(cfg):
    pw, ph = PAPER_SIZES[cfg.paper]
    if cfg.portrait:
        pw, ph = min(pw, ph), max(pw, ph)
    dia = cfg.diameter_mm or (min(pw, ph) - 2 * cfg.padding_mm)
    return pw, ph, (pw / 2, ph / 2), dia / 2


def strokes_to_polylines(strokes, cfg, target=None):
    """Nail strokes -> pen-down polylines in paper mm.

    Lines follow their chord shape (straight / bent arcs). With nail_spread_mm
    > 0 each visit to a nail lands at a slightly different point on the rim so
    line ends don't pool into ink blobs. With cfg.dashed (and the target the
    solver used), the pen lifts where the target is lighter than gap_tone.
    """
    pw, ph, (cx, cy), R = layout(cfg)
    ang = nail_angles(cfg.nails)
    visits = np.zeros(cfg.nails, dtype=int)
    golden = (math.sqrt(5) - 1) / 2
    bends = chord_bends(cfg)
    dashed = cfg.dashed and target is not None
    if dashed:
        dmask = drawable_mask(target, cfg)
        size = dmask.shape[0]
        cell = 2 * R / size

    def nail_xy(k):
        a = ang[k]
        if cfg.nail_spread_mm > 0:
            a += ((visits[k] * golden) % 1.0 - 0.5) * cfg.nail_spread_mm / R
            visits[k] += 1
        return (cx + R * math.cos(a), cy + R * math.sin(a))

    out = []
    line_no = 0
    for seq in strokes:
        sb = getattr(seq, "bends", None) or [0] * (len(seq) - 1)
        cur = []
        prev = nail_xy(seq[0])
        for k in range(1, len(seq)):
            nxt = nail_xy(seq[k])
            pts = arc_points(prev, nxt, (cx, cy), R, bends[sb[k - 1]], 0.2 if dashed else 0.5)
            if not dashed:
                cur.extend(map(tuple, pts if not cur else pts[1:]))
            else:
                gx = np.clip(((pts[:, 0] - (cx - R)) / cell).astype(int), 0, size - 1)
                gy = np.clip(((pts[:, 1] - (cy - R)) / cell).astype(int), 0, size - 1)
                duty = dmask[gy, gx]
                seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
                # dash pattern: pen down for the first `duty` share of every period;
                # per-line phase (golden ratio) so neighbouring lines don't form moire bands
                line_no += 1
                phase = (line_no * golden) % 1.0
                on = ((seg / cfg.dash_period_mm + phase) % 1.0) < duty
                on |= duty >= 0.98
                # close short gaps, then drop short dashes
                runs = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
                starts, ends = runs[0::2], runs[1::2] - 1
                for i in range(len(starts) - 1):
                    if seg[starts[i + 1]] - seg[ends[i]] < cfg.gap_min_mm:
                        on[ends[i]:starts[i + 1] + 1] = True
                runs = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
                for s0, e0 in zip(runs[0::2], runs[1::2] - 1):
                    if seg[e0] - seg[s0] < cfg.dash_min_mm:
                        on[s0:e0 + 1] = False
                for i in range(len(pts)):
                    if on[i]:
                        cur.append(tuple(pts[i]))
                    elif cur:
                        if len(cur) > 1:
                            out.append(cur)
                        cur = []
            prev = nxt
        if len(cur) > 1:
            out.append(cur)
    from .hatch import _rdp
    return [_rdp(pl, 0.02) for pl in out]


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
