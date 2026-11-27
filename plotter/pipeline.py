"""
Shared generation pipeline used by both the CLI and the preview UI.

Each build_* function takes an image path and a config and returns a Result:
ordered polylines (paper mm, y-down) ready for G-code, plus preview rasters.
"""

from dataclasses import dataclass, field, fields
import os
import re
import tempfile

import numpy as np
from PIL import Image

from . import stringart as sa
from . import hatch as hz
from .gcode import order_polylines, write_gcode, write_svg, PlotterProfile
from radon_renderer.config import PAPER_SIZES


@dataclass
class Result:
    polylines: list
    paper_w: float
    paper_h: float
    pen_mm: float
    info: dict = field(default_factory=dict)
    target: np.ndarray | None = None      # darkness 0..1 (what we aim for)
    simulation: np.ndarray | None = None  # darkness 0..1 (what the solver thinks it drew)


def config_from_dict(cls, d):
    """Build a dataclass from a loose dict (UI / JSON), coercing types and ignoring unknown keys."""
    out = {}
    for f in fields(cls):
        if f.name not in d or d[f.name] is None or d[f.name] == "":
            continue
        v = d[f.name]
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", None) or str(f.type)
        if t.startswith("Optional") or t.startswith("Union"):
            t = str(f.type)
        if f.name == "crop":
            v = tuple(int(round(x)) for x in v)
        elif t.startswith("tuple"):
            if isinstance(v, str):
                v = [x for x in re.split(r"[,\s]+", v) if x]
            v = tuple(tuple(float(y) for y in x) if isinstance(x, (list, tuple)) else float(x)
                      for x in v)
        elif t.startswith("bool"):
            v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes", "on")
        elif t.startswith("int"):
            v = int(round(float(v)))
        elif t.startswith("float"):
            v = float(v)
            if "None" in t and v <= 0:      # optional floats: 0 means "auto"
                continue
        elif t.startswith("list"):
            if isinstance(v, str):
                v = [line for line in v.splitlines() if line.strip()]
        out[f.name] = v
    return cls(**out)


def build_stringart(image_path, cfg: sa.StringArtConfig, progress=None):
    pw, ph, _, R = sa.layout(cfg)
    size = int(round(2 * R / cfg.pixel_mm))
    target, weight, _, box = sa.prepare_target(image_path, cfg, size)
    table = sa.LineTable(cfg.nails, size, cfg.pixel_mm, cfg.pen_mm, sa.chord_bends(cfg))
    strokes, cov, _ = sa.solve(target, weight, cfg, table, progress=progress)
    polylines = sa.strokes_to_polylines(strokes, cfg, target)
    if cfg.frame:
        polylines.append(sa.circle_polyline(cfg))
    n_lines = sum(len(s) - 1 for s in strokes)
    return Result(polylines, pw, ph, cfg.pen_mm,
                  info={"lines": n_lines, "strokes": len(strokes), "crop_box": list(map(int, box)),
                        "nail_strokes": strokes},
                  target=target, simulation=cov)


def build_hatch(image_path, cfg: hz.HatchConfig, progress=None):
    pw, ph, _, _ = sa.layout(cfg)
    layers, darkness, box = hz.flow_hatch(image_path, cfg, progress=progress)
    polylines, pos = [], (0.0, 0.0)
    for layer in layers:                    # light layer first, each path-optimised
        ordered = order_polylines(layer, start=pos)
        if ordered:
            pos = ordered[-1][-1]
        polylines += ordered
    return Result(polylines, pw, ph, cfg.pen_mm,
                  info={"strokes": len(polylines), "layers": [len(l) for l in layers],
                        "crop_box": list(map(int, box))},
                  target=darkness)


@dataclass
class RadonConfig:
    """The original radon renderer, unchanged look."""
    paper: str = "A4"
    pen_mm: float = 0.3
    padding_mm: float = 10.0
    lines: int = 500
    mode: str = "fixed_6"           # fixed_6 | 6_plus_refine | pure_optimize | hierarchical
    hierarchical_strategy: str = "strength_based"
    quality_threshold: float = 0.01
    edge_weight: float = 0.75
    intensity_weight: float = 0.25
    crop: tuple | None = None


def build_radon(image_path, cfg: RadonConfig, progress=None):
    from radon_renderer.renderer import AdvancedRadonRenderer
    img = Image.open(image_path)
    if cfg.crop:
        x, y, w, h = cfg.crop
        img = img.crop((x, y, x + w, y + h))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "crop.png")
        img.save(p)
        r = AdvancedRadonRenderer(p, num_lines=cfg.lines, paper_size=cfg.paper,
                                  pen_width=cfg.pen_mm, padding=cfg.padding_mm,
                                  edge_weight=cfg.edge_weight, intensity_weight=cfg.intensity_weight)
        r.load_and_preprocess()
        r.compute_edges()
        r.compute_target_image()
        r.analyze_with_radon()
        if cfg.mode == "fixed_6":
            angles, per = r.select_angles_fixed_6()
            r.render_angles(angles, per)
        elif cfg.mode == "6_plus_refine":
            r.select_angles_6_plus_refine(quality_threshold=cfg.quality_threshold)
        elif cfg.mode == "pure_optimize":
            r.select_angles_pure_optimize(quality_threshold=cfg.quality_threshold)
        elif cfg.mode == "hierarchical":
            r.select_angles_hierarchical(strategy=cfg.hierarchical_strategy, config={})
        else:
            raise ValueError(f"unknown radon mode {cfg.mode}")
    segs = [[(l["x1"], l["y1"]), (l["x2"], l["y2"])] for l in r.lines
            if abs(l["x1"] - l["x2"]) + abs(l["y1"] - l["y2"]) > 0.05]
    pw, ph = PAPER_SIZES[cfg.paper]
    tgt = r.target_image / (r.target_image.max() + 1e-9)
    return Result(order_polylines(segs), pw, ph, cfg.pen_mm,
                  info={"lines": len(segs)}, target=tgt)


def build_scribble(image_path, cfg, progress=None):
    """Gradient-tracing scribble (port of scribble-code/c_v25) -> paper mm polylines."""
    import math
    from . import scribble as scr
    from .hatch import _rdp, merge_strokes
    lines, (H, W), sc = scr.scribble(image_path, cfg, progress=progress)
    lines = [l for l in lines if len(l) >= cfg.min_points_out]

    s, rot, pw, ph = scr.paper_fit(H, W, cfg)
    cw, ch = (H, W) if rot else (W, H)            # content size in pixels after rotation
    ox = (pw - cw * s) / 2
    oy = (ph - ch * s) / 2

    def to_mm(r, c):
        # image (row, col) -> paper (x, y) y-down; rotation is 90 deg clockwise
        x, y = (H - 1 - r, c) if rot else (c, r)
        return (ox + (x + 0.5) * s, oy + (y + 0.5) * s)

    polylines = []
    tol = cfg.simplify_px * s
    for l in lines:
        pl = [to_mm(r, c) for r, c in l]
        polylines.append(_rdp(pl, tol) if tol > 0 else pl)
    if cfg.join_px > 0:
        polylines = merge_strokes(polylines, cfg.join_px * s, 180.0)
    ordered = order_polylines(polylines)
    return Result(ordered, pw, ph, cfg.pen_mm,
                  info={"traced_lines": len(lines), "image_px": [W, H], "rotated": bool(rot),
                        "mm_per_px": round(s, 4)},
                  target=sc.ipix)


STYLES = {
    "stringart": (sa.StringArtConfig, build_stringart),
    "hatch": (hz.HatchConfig, build_hatch),
}
# The original radon renderer (build_radon / RadonConfig) stays available for scripts and
# `python -m plotter lines2gcode`, but is no longer offered as a UI style: on photos it
# produces families of full-circle chords that don't resemble the image.


def _scribble_entry():
    from .scribble import ScribbleConfig
    return ScribbleConfig


STYLES["scribble"] = (_scribble_entry(), build_scribble)


def export(result: Result, out_base, profile: PlotterProfile, comment=None):
    """Write .gcode and .svg. The ink cap is applied first, at the final size and true pen
    width; result.polylines is replaced by what is actually plotted."""
    from .gcode import limit_ink
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    result.polylines, ink = limit_ink(result.polylines, result.pen_mm, profile.ink_cap, profile.ink_cell_mm)
    stats = write_gcode(result.polylines, out_base + ".gcode", profile,
                        paper_height=result.paper_h, comment=comment, paper_width=result.paper_w)
    write_svg(result.polylines, out_base + ".svg", result.paper_w, result.paper_h, result.pen_mm)
    stats["ink_removed_pct"] = round(ink["ink_removed_pct"], 1)
    return stats


def darkness_png(arr, path):
    Image.fromarray((255 * (1 - np.clip(arr, 0, 1))).astype(np.uint8)).save(path)
