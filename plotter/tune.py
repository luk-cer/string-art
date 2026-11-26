"""
Suggest string-art settings for a photo.

Each candidate is solved and its simulated ink (what the plot will look like)
is compared with the photo at viewing distance. The reference is fixed
(auto-contrast only), so changing gamma/CLAHE can't game the score:

    score = SSIM(sim, ref) - 0.5 * mean|sim - ref|      (inside the circle, blurred)

Search: coordinate descent over the settings that matter most, starting from
the current settings, two rounds. Typical cost: 20-40 solves.
"""

from dataclasses import asdict, replace
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

from . import stringart as sa

GRID = {
    "overshoot": [1.0, 1.5, 2.0, 3.0, 5.0],
    "pixel_mm": [0.8, 1.2, 1.6, 2.2],
    "gamma": [0.7, 0.85, 1.0, 1.2, 1.5],
    "clahe": [0.0, 0.01, 0.02, 0.04],
    "edge_boost": [0.0, 1.0, 2.5],
    "vignette": [0.0, 0.3, 0.6],
    "max_darkness": [0.8, 0.9, 1.0],
    "min_lines": [0, 1500, 3000],
}
ORDER = ["overshoot", "pixel_mm", "gamma", "min_lines", "clahe", "edge_boost", "max_darkness", "vignette"]
SCORE_PX = 220


def _reference(image_path, cfg):
    """Fixed comparison target: the photo with plain auto-contrast, as darkness."""
    ref_cfg = replace(cfg, gamma=1.0, clahe=0.0, vignette=0.0, max_darkness=1.0, edge_boost=0.0)
    dark, _, mask, _ = sa.prepare_target(image_path, ref_cfg, SCORE_PX)
    return dark, mask


def _score(cov, ref, mask, view_px=1.5):
    sim = np.asarray(Image.fromarray((cov * 255).astype(np.uint8)).resize((SCORE_PX, SCORE_PX),
                                                                          Image.Resampling.BOX), float) / 255
    a = ndimage.gaussian_filter(sim, view_px)
    b = ndimage.gaussian_filter(ref, view_px)
    a, b = a * mask, b * mask
    ssim = structural_similarity(a, b, data_range=1.0)
    mae = np.abs(a - b)[mask].mean()
    return ssim - 0.5 * mae, ssim, mae


def suggest_stringart(image_path, cfg: sa.StringArtConfig, rounds=2, time_limit=240, progress=None):
    """Returns (best settings dict, best score, history)."""
    ref, mask = _reference(image_path, cfg)
    tables = {}
    cache = {}
    t0 = time.time()

    def evaluate(c):
        key = tuple(getattr(c, k) for k in ORDER)
        if key in cache:
            return cache[key]
        pw, ph, _, R = sa.layout(c)
        size = int(round(2 * R / c.pixel_mm))
        tk = (c.nails, size, c.pixel_mm, c.pen_mm, sa.chord_bends(c))
        if tk not in tables:
            tables[tk] = sa.LineTable(c.nails, size, c.pixel_mm, c.pen_mm, sa.chord_bends(c))
        target, weight, _, _ = sa.prepare_target(image_path, c, size)
        strokes, cov, _ = sa.solve(target, weight, c, tables[tk])
        sc = _score(cov, ref, mask)
        lines = sum(len(s) - 1 for s in strokes)
        cache[key] = (sc, lines)
        return cache[key]

    best = cfg
    (best_s, ssim, mae), lines = evaluate(best)
    history = [dict(step="start", score=round(best_s, 4), ssim=round(ssim, 4), mae=round(mae, 4), lines=lines)]
    if progress:
        progress(f"  start: score {best_s:.4f} ({lines} lines)")
    for rnd in range(rounds):
        improved = False
        for k in ORDER:
            for v in GRID[k]:
                if time.time() - t0 > time_limit:
                    break
                if v == getattr(best, k):
                    continue
                cand = replace(best, **{k: v})
                if cand.min_lines > cand.max_lines:
                    cand = replace(cand, max_lines=cand.min_lines + 500)
                (s, ssim, mae), lines = evaluate(cand)
                if s > best_s + 1e-4:
                    best, best_s, improved = cand, s, True
                    history.append(dict(step=f"{k}={v}", score=round(s, 4), ssim=round(ssim, 4),
                                        mae=round(mae, 4), lines=lines))
                    if progress:
                        progress(f"  {k} = {v}: score {s:.4f} ({lines} lines)")
        if not improved:
            break
    out = {k: getattr(best, k) for k in ORDER}
    out["max_lines"] = best.max_lines
    return out, best_s, history
