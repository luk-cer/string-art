"""
Plotter command line.

    python -m plotter stringart photo.jpg -o out/portrait
    python -m plotter lines2gcode output.json -o out/radon
    python -m plotter profile --dump my_plotter.json
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from . import stringart as sa
from .gcode import PlotterProfile, write_gcode, write_svg, order_polylines
from radon_renderer.config import PAPER_SIZES


def _profile(args):
    return PlotterProfile.load(args.profile) if args.profile else PlotterProfile()


def _add_gcode_args(p):
    p.add_argument("--profile", help="plotter profile JSON (see `profile --dump`)")
    p.add_argument("--feed", type=float, help="override draw feed mm/min")


def cmd_stringart(args):
    cfg = sa.StringArtConfig(
        nails=args.nails, max_lines=args.lines, paper=args.paper,
        portrait=args.portrait, padding_mm=args.padding, diameter_mm=args.diameter,
        pen_mm=args.pen, pixel_mm=args.pixel, min_skip=args.min_skip,
        max_darkness=args.max_darkness, gamma=args.gamma, edge_boost=args.edge_boost,
        stop_ratio=args.stop_ratio, continuous=args.continuous, overshoot=args.overshoot, clahe=args.clahe, vignette=args.vignette, nail_spread_mm=args.nail_spread, frame=args.frame,
        crop=tuple(args.crop) if args.crop else None,
    )
    pw, ph, _, R = sa.layout(cfg)
    size = int(round(2 * R / cfg.pixel_mm))
    print(f"Circle {2*R:.0f} mm on {cfg.paper} -> sim grid {size}x{size} "
          f"({cfg.pixel_mm} mm/cell), {cfg.nails} nails, pen {cfg.pen_mm} mm")

    t0 = time.time()
    target, weight, mask, box = sa.prepare_target(args.image, cfg, size)
    table = sa.LineTable(cfg.nails, size, cfg.pixel_mm, cfg.pen_mm)
    print(f"Line table built in {time.time()-t0:.1f}s; solving...")
    strokes, cov, _ = sa.solve(target, weight, cfg, table,
                               progress=lambda s, e: print(f"  {s:5d} lines  err {e:.5f}"))
    n_lines = sum(len(st) - 1 for st in strokes)
    print(f"Done: {n_lines} lines in {len(strokes)} pen-down strokes, {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    polylines = sa.strokes_to_polylines(strokes, cfg)
    if cfg.frame:
        polylines.append(sa.circle_polyline(cfg))

    prof = _profile(args)
    if args.feed:
        prof.draw_feed = args.feed
    stats = write_gcode(polylines, args.output + ".gcode", prof, paper_height=ph,
                        comment=f"string art: {n_lines} lines, {cfg.nails} nails")
    write_svg(polylines, args.output + ".svg", pw, ph, cfg.pen_mm)
    sa.render_preview(polylines, cfg, args.output + "_preview.png")

    # target vs simulation side by side, for tuning
    tgt = Image.fromarray((255 * (1 - target)).astype(np.uint8))
    sim = Image.fromarray((255 * (1 - cov)).astype(np.uint8))
    both = Image.new("L", (size * 2 + 10, size), 255)
    both.paste(tgt, (0, 0)); both.paste(sim, (size + 10, 0))
    both.save(args.output + "_compare.png")

    stats.update(lines=n_lines, strokes=len(strokes))
    sa.save_json(args.output + ".json", strokes, cfg, box, stats)
    print(f"Ink length {stats['draw_mm']/1000:.1f} m, est. plot time "
          f"{stats['est_minutes']/60:.1f} h at F{prof.draw_feed:g}")
    print(f"Wrote {args.output}.gcode / .svg / _preview.png / _compare.png / .json")


def cmd_lines2gcode(args):
    """Convert the radon renderer JSON (unchanged look) into ordered G-code."""
    with open(args.json) as f:
        data = json.load(f)
    meta = data["metadata"]
    pw = meta["paper_dimensions_mm"]["width"]
    ph = meta["paper_dimensions_mm"]["height"]
    segs = [[(l["x1"], l["y1"]), (l["x2"], l["y2"])] for l in data["lines"]
            if abs(l["x1"] - l["x2"]) + abs(l["y1"] - l["y2"]) > 0.05]
    ordered = order_polylines(segs) if not args.no_order else segs
    prof = _profile(args)
    if args.feed:
        prof.draw_feed = args.feed
    stats = write_gcode(ordered, args.output + ".gcode", prof, paper_height=ph)
    write_svg(ordered, args.output + ".svg", pw, ph, meta.get("pen_width_mm", 0.3))
    print(f"{len(ordered)} segments ({len(data['lines'])-len(segs)} degenerate dropped); "
          f"draw {stats['draw_mm']/1000:.1f} m, travel {stats['travel_mm']/1000:.1f} m, "
          f"~{stats['est_minutes']:.0f} min")


def cmd_profile(args):
    PlotterProfile().dump(args.dump)
    print(f"Wrote default profile to {args.dump}; edit and pass with --profile")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="plotter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stringart", help="circular nail-to-nail portrait")
    s.add_argument("image")
    s.add_argument("-o", "--output", default="out/stringart")
    s.add_argument("--nails", type=int, default=240)
    s.add_argument("--lines", type=int, default=3000, help="max lines (stops earlier if no gain)")
    s.add_argument("--paper", default="A4", choices=list(PAPER_SIZES))
    s.add_argument("--portrait", action="store_true", help="portrait paper orientation")
    s.add_argument("--padding", type=float, default=10.0)
    s.add_argument("--diameter", type=float, help="circle diameter mm (default: fit paper)")
    s.add_argument("--pen", type=float, default=0.3, help="pen width mm")
    s.add_argument("--pixel", type=float, default=0.6,
                   help="simulation cell mm; larger = viewed from further away, smoother tones")
    s.add_argument("--min-skip", type=int, default=15)
    s.add_argument("--max-darkness", type=float, default=0.9)
    s.add_argument("--gamma", type=float, default=1.0)
    s.add_argument("--edge-boost", type=float, default=1.0)
    s.add_argument("--overshoot", type=float, default=3.0,
                   help="penalty for ink over lighter-than-target areas; higher = cleaner highlights")
    s.add_argument("--clahe", type=float, default=0.02, help="local contrast boost, 0 = off")
    s.add_argument("--vignette", type=float, default=0.0, help="0..1 fade background towards rim")
    s.add_argument("--stop-ratio", type=float, default=0.0)
    s.add_argument("--nail-spread", type=float, default=0.0,
                   help="mm to spread line endpoints around each nail (avoids ink blobs at rim)")
    s.add_argument("--continuous", action="store_true",
                   help="one unbroken path (thread-like); default lifts the pen when stuck")
    s.add_argument("--frame", action="store_true", help="also draw the circle outline")
    s.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    _add_gcode_args(s)
    s.set_defaults(fn=cmd_stringart)

    l = sub.add_parser("lines2gcode", help="radon renderer JSON -> G-code/SVG")
    l.add_argument("json")
    l.add_argument("-o", "--output", default="out/radon")
    l.add_argument("--no-order", action="store_true", help="keep original line order")
    _add_gcode_args(l)
    l.set_defaults(fn=cmd_lines2gcode)

    p = sub.add_parser("profile", help="write default plotter profile")
    p.add_argument("--dump", default="plotter_profile.json")
    p.set_defaults(fn=cmd_profile)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
