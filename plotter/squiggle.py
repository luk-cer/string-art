"""
Squiggle fill: one continuous, non-self-crossing meandering line (a
pseudo-random "Hilbert / maze curve") whose spacing follows tone and whose
corridors follow the image's flow.

Construction (maze-curve trick):
1. tone -> desired cell size: dark = small cells, light = big cells; a
   quadtree quantises this to power-of-two cells;
2. a random spanning tree over neighbouring cells (Kruskal), where edges that
   run along the flow direction (+ an optional angle) are preferred;
3. the squiggle is the outline of the tree drawn with corridors half a cell
   wide. Walking around a tree never crosses itself and closes on itself, so
   each connected part of the tree gives exactly one closed pen-down loop.
   Strands end up ~half a cell apart, which sets the tone.

Cells lighter than `white` are left out, so highlights stay paper.
"""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


@dataclass
class SquiggleConfig:
    pen_px: float = 1.0              # pen width in image pixels (from paper scale)
    min_cell: int = 0                # smallest cell (px): darkest tone; 0 = auto (~4 pen widths, strands 2 pens apart)
    max_cell: int = 32               # largest cell (px): lightest drawn tone
    white: float = 0.08              # darkness below this is left blank
    gain: float = 1.0                # >1 darker overall
    flow_bias: float = 1.0           # 0 = pure random maze; ~1 = maze with a tendency; 3+ = straight rows
    angle_deg: float = 0.0           # corridor direction relative to the flow (0 = along contours)
    dither: bool = True              # randomly round cell sizes between powers of two (no tone banding)
    patch_px: float = 0.0            # >0: stop growing a maze at ~this size (side, px) -> many local squiggles
    supersample: int = 3             # raster resolution for tracing the outline
    smooth: int = 5                  # moving-average window on the outline (rounder corners)
    seed: int = 0


def _cell_size_map(dark, cfg):
    """Desired cell size per pixel: strands ~cell/2 apart, coverage ~ pen/(cell/2)."""
    d = np.clip(dark * cfg.gain, 1e-3, 1.0)
    c = 2.0 * cfg.pen_px / d
    return np.clip(c, cfg.min_cell, cfg.max_cell)


def _quadtree(cmap, dark, cfg, rng, walls=None):
    """Leaves (r0, c0, size) of a quadtree whose cell size is the desired size
    (median inside the cell) rounded to a power of two - randomly up or down in
    proportion (dithering), so tones blend instead of banding."""
    H, W = cmap.shape
    top = 1 << int(math.ceil(math.log2(max(H, W, cfg.min_cell))))
    if walls is not None:                         # summed-area table: "any wall pixel in this cell?"
        sat = np.zeros((H + 1, W + 1), dtype=np.int64)
        sat[1:, 1:] = walls.astype(np.int64).cumsum(0).cumsum(1)

        def has_wall(r, c, s):
            r1, c1 = min(r + s, H), min(c + s, W)
            return sat[r1, c1] - sat[r, c1] - sat[r1, c] + sat[r, c] > 0
    leaves = []
    stack = [(0, 0, top)]
    while stack:
        r, c, s = stack.pop()
        if r >= H or c >= W:
            continue
        reg = cmap[r:r + s, c:c + s]
        want = float(np.median(reg))
        # split while s is above the desired size; between s/2 and s decide at random
        # with probability matching the log-distance (stochastic rounding)
        thr = want * (2.0 ** rng.random() if cfg.dither else math.sqrt(2.0))
        wall = walls is not None and has_wall(r, c, s)
        if s > cfg.min_cell and (s > thr or wall):     # cells on an outline shrink to the minimum ...
            h = s // 2
            stack += [(r, c, h), (r, c + h, h), (r + h, c, h), (r + h, c + h, h)]
        else:
            if wall:                                     # ... and are left out: squiggles can't cross outlines
                continue
            if dark[r:r + s, c:c + s].mean() >= cfg.white:
                leaves.append((r, c, s))
    return leaves


def _adjacency(leaves, shape):
    H, W = shape
    lab = np.full((H, W), -1, dtype=np.int64)
    for i, (r, c, s) in enumerate(leaves):
        lab[r:r + s, c:c + s] = i
    pairs = []
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        p = np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1)
        if len(p):
            pairs.append(p)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pairs), axis=0)


def _spanning_tree(leaves, edges, theta, cfg, rng):
    """Kruskal on random weights + flow misalignment penalty."""
    if len(edges) == 0:
        return edges
    L = np.asarray(leaves, dtype=float)
    cen = np.stack([L[:, 0] + L[:, 2] / 2, L[:, 1] + L[:, 2] / 2], 1)       # (row, col)
    a, b = edges[:, 0], edges[:, 1]
    v = cen[b] - cen[a]
    ang = np.arctan2(v[:, 0], v[:, 1])                                       # in (x=col, y=row)
    mid = ((cen[a] + cen[b]) / 2).astype(int)
    mid[:, 0] = np.clip(mid[:, 0], 0, theta.shape[0] - 1)
    mid[:, 1] = np.clip(mid[:, 1], 0, theta.shape[1] - 1)
    th = theta[mid[:, 0], mid[:, 1]] + math.radians(cfg.angle_deg)
    mis = np.abs(np.sin(ang - th))                                           # 0 = along the flow
    w = rng.random(len(edges)) + cfg.flow_bias * mis
    order = np.argsort(w, kind="stable")
    parent = np.arange(len(leaves))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    area = (L[:, 2] ** 2).copy()
    cap = cfg.patch_px ** 2 if cfg.patch_px else np.inf
    keep = []
    for k in order:
        ra, rb = find(a[k]), find(b[k])
        if ra != rb and area[ra] + area[rb] <= cap:
            parent[ra] = rb
            area[rb] += area[ra]
            keep.append(k)
    return edges[keep]


def _outline(leaves, tree, shape, cfg):
    """Outline of the tree drawn with corridors half a cell wide -> closed loops (row, col)."""
    from skimage.measure import find_contours
    H, W = shape
    f = cfg.supersample
    m = int(math.ceil(0.5 * max(s for _, _, s in leaves) + 2)) * f   # margin so outlines never touch the edge
    im = Image.new("L", (W * f + 2 * m, H * f + 2 * m), 0)
    d = ImageDraw.Draw(im)
    L = np.asarray(leaves, dtype=float)
    cen = np.stack([L[:, 0] + L[:, 2] / 2, L[:, 1] + L[:, 2] / 2], 1)
    off = m

    def P(rc):
        return (rc[1] * f + off, rc[0] * f + off)

    deg = np.zeros(len(leaves), dtype=int)
    for i, j in tree:
        deg[i] += 1; deg[j] += 1
        wdt = max(1.0, 0.5 * min(L[i, 2], L[j, 2]) * f)
        d.line([P(cen[i]), P(cen[j])], fill=255, width=int(round(wdt)))
    for i in range(len(leaves)):                                 # round joints / isolated cells
        r = 0.25 * L[i, 2] * f
        x, y = P(cen[i])
        d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    a = np.asarray(im, dtype=float) / 255.0
    loops = []
    for cnt in find_contours(a, 0.5):
        if len(cnt) < 4 or np.hypot(*(cnt[0] - cnt[-1])) > 1.5:      # closed loops only
            continue
        pts = (cnt - off) / f                                     # back to image px (row, col)
        if cfg.smooth and cfg.smooth > 1 and len(pts) > cfg.smooth:
            k = cfg.smooth
            ker = np.ones(k) / k
            ext = np.concatenate([pts[-k:], pts, pts[:k]])          # closed: wrap around
            sm = np.stack([np.convolve(ext[:, 0], ker, "same"), np.convolve(ext[:, 1], ker, "same")], 1)
            pts = sm[k:-k]
            pts = np.vstack([pts, pts[:1]])
        loops.append([tuple(p) for p in pts.tolist()])
    return loops


def squiggle(dark, theta, cfg: SquiggleConfig, walls=None):
    """dark: target darkness 0..1 (H x W); theta: flow orientation (x=col, y=row);
    walls: optional bool mask (e.g. the drawn contour lines) squiggles must not cross.
    Returns closed loops as lists of (row, col) floats, plus stats."""
    rng = np.random.default_rng(cfg.seed)
    if not cfg.min_cell:
        cfg = SquiggleConfig(**{**cfg.__dict__, "min_cell": max(2, int(round(4 * cfg.pen_px)))})
    dark = np.clip(dark, 0, 1)
    cmap = _cell_size_map(dark, cfg)
    leaves = _quadtree(cmap, dark, cfg, rng, walls)
    if not leaves:
        return [], {"cells": 0, "tree_edges": 0, "loops": 0}
    edges = _adjacency(leaves, dark.shape)
    tree = _spanning_tree(leaves, edges, theta, cfg, rng)
    loops = _outline(leaves, tree, dark.shape, cfg)
    return loops, {"cells": len(leaves), "tree_edges": int(len(tree)), "loops": len(loops)}
