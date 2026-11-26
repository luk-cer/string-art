"""
Plotter command line.

    python -m plotter stringart photo.jpg -o out/portrait
    python -m plotter hatch photo.jpg -o out/hatch
    python -m plotter ui                       # browser preview UI
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
from . import hatch as hz
from . import pipeline
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
    print(f"Circle {2*R:.0f} mm on {cfg.paper}, {cfg.nails} nails, pen {cfg.pen_mm} mm")
    t0 = time.time()
    res = pipeline.build_stringart(args.image, cfg,
                                   progress=lambda s, e: print(f"  {s:5d} lines  err {e:.5f}"))
    n_lines = res.info["lines"]
    print(f"Done: {n_lines} lines in {res.info['strokes']} pen-down strokes, {time.time()-t0:.1f}s")

    prof = _profile(args)
    if args.feed:
        prof.draw_feed = args.feed
    stats = pipeline.export(res, args.output, prof,
                            comment=f"string art: {n_lines} lines, {cfg.nails} nails")
    sa.render_preview(res.polylines, cfg, args.output + "_preview.png")

    # target vs simulation side by side, for tuning
    size = res.target.shape[0]
    tgt = Image.fromarray((255 * (1 - res.target)).astype(np.uint8))
    sim = Image.fromarray((255 * (1 - res.simulation)).astype(np.uint8))
    both = Image.new("L", (size * 2 + 10, size), 255)
    both.paste(tgt, (0, 0)); both.paste(sim, (size + 10, 0))
    both.save(args.output + "_compare.png")

    stats.update(lines=n_lines, strokes=res.info["strokes"])
    sa.save_json(args.output + ".json", res.info["nail_strokes"], cfg, res.info["crop_box"], stats)
    print(f"Ink length {stats['draw_mm']/1000:.1f} m, est. plot time "
          f"{stats['est_minutes']/60:.1f} h at F{prof.draw_feed:g}")
    print(f"Wrote {args.output}.gcode / .svg / _preview.png / _compare.png / .json")


def cmd_hatch(args):
    cfg = hz.HatchConfig(
        paper=args.paper, portrait=args.portrait, padding_mm=args.padding,
        diameter_mm=args.diameter, shape=args.shape, pen_mm=args.pen,
        crop=tuple(args.crop) if args.crop else None, gamma=args.gamma, clahe=args.clahe,
        layer_angles=tuple(args.angles[:args.layers]),
        layer_thresholds=tuple(args.thresholds[:args.layers]),
        spacing_max_mm=args.spacing_max, spacing_min_mm=args.spacing_min,
        flow=args.flow, flow_smooth_mm=args.flow_smooth, flow_bias_angle=args.flow_angle,
        min_len_mm=args.min_len, max_len_mm=args.max_len, seed=args.seed,
        tone_smooth_mm=args.tone_smooth, seed_margin=args.seed_margin,
        overrun=args.overrun, min_len_spacings=args.min_len_spacings,
        tone_model=args.tone_model, merge_gap_mm=args.merge_gap, merge_angle_deg=args.merge_angle,
        outline_levels=tuple(args.outline or ()), outline_smooth_mm=args.outline_smooth,
        outline_min_len_mm=args.outline_min_len,
    )
    t0 = time.time()
    res = pipeline.build_hatch(args.image, cfg)
    prof = _profile(args)
    if args.feed:
        prof.draw_feed = args.feed
    stats = pipeline.export(res, args.output, prof,
                            comment=f"flow hatch ({cfg.flow}), {len(cfg.layer_angles)} layers, pen {cfg.pen_mm} mm")
    sa.render_preview(res.polylines, cfg, args.output + "_preview.png")
    print(f"{len(res.polylines)} strokes in {time.time()-t0:.1f}s; ink {stats['draw_mm']/1000:.1f} m, "
          f"travel {stats['travel_mm']/1000:.1f} m, {stats['pen_lifts']} pen lifts, "
          f"~{stats['est_minutes']:.0f} min (of which lifts {stats['pen_lift_minutes']:.0f})")
    print(f"Wrote {args.output}.gcode / .svg / _preview.png")


def cmd_scribble(args):
    from .scribble import preset
    # flags left unset fall back to the chosen preset (c_v25 / app5 / app7)
    m = {"padding": "padding_mm", "pen": "pen_mm", "kernel": "trace_kernel_size",
         "sigma": "trace_kernel_sigma", "hatch_interval": "hatch_sample_interval",
         "simplify": "simplify_px", "join": "join_px", "white_cut": "white_cut",
         "hatch_mode": "hatch_mode", "momentum": "main_momentum", "smooth": "smooth_main",
         "hatch_fill": "hatch_fill_px", "ink_passes": "hatch_ink_passes",
         "cleanup_mode": "cleanup_mode", "hatch_source": "hatch_source"}
    keys = ["paper", "rotate", "max_side", "gradient_threshold", "cleanup_radius", "max_lines",
            "light_percentile", "hatch_max_length", "sampler", "workers", "seed"] + list(m)
    over = {m.get(k, k): getattr(args, k) for k in keys if getattr(args, k) is not None}
    if args.portrait:
        over["portrait"] = True
    if args.crop:
        over["crop"] = tuple(args.crop)
    if args.denoise:
        over["remove_noise"] = True
    if args.no_hatch:
        over["hatch"] = False
    if args.hatch_angles:
        over["hatch_angles"] = tuple(tuple(float(v) for v in a.split(":")) for a in args.hatch_angles.split(","))
    if args.hatch_spacing:
        over["hatch_spacing_min_px"], over["hatch_spacing_max_px"] = args.hatch_spacing
    over.setdefault("workers", 0)
    cfg = preset(args.preset, **over)
    print(f"preset {args.preset}")
    t0 = time.time()
    res = pipeline.build_scribble(args.image, cfg, progress=print)
    prof = _profile(args)
    if args.feed:
        prof.draw_feed = args.feed
    stats = pipeline.export(res, args.output, prof, comment="scribble (gradient tracing)")
    from types import SimpleNamespace
    sa.render_preview(res.polylines, SimpleNamespace(paper=cfg.paper, portrait=cfg.portrait,
                      padding_mm=cfg.padding_mm, diameter_mm=None, pen_mm=cfg.pen_mm),
                      args.output + "_preview.png")
    print(f"{res.info['traced_lines']} traced lines -> {len(res.polylines)} pen-down paths in "
          f"{time.time()-t0:.1f}s; {res.info['mm_per_px']} mm/px, rotated={res.info['rotated']}")
    print(f"ink {stats['draw_mm']/1000:.1f} m, travel {stats['travel_mm']/1000:.1f} m, "
          f"{stats['pen_lifts']} pen lifts, ~{stats['est_minutes']:.0f} min")
    print(f"Wrote {args.output}.gcode / .svg / _preview.png")


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

    h = sub.add_parser("hatch", help="curved, tone-layered crosshatch following the image")
    h.add_argument("image")
    h.add_argument("-o", "--output", default="out/hatch")
    h.add_argument("--paper", default="A4", choices=list(PAPER_SIZES))
    h.add_argument("--portrait", action="store_true")
    h.add_argument("--padding", type=float, default=10.0)
    h.add_argument("--diameter", type=float)
    h.add_argument("--shape", default="circle", choices=["circle", "rect"])
    h.add_argument("--pen", type=float, default=0.3)
    h.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    h.add_argument("--gamma", type=float, default=1.0)
    h.add_argument("--clahe", type=float, default=0.0)
    h.add_argument("--flow", default="contour", choices=["contour", "gradient", "fixed"],
                   help="contour: strokes wrap around shapes; fixed: classic straight hatching")
    h.add_argument("--flow-smooth", type=float, default=4.0, help="mm; bigger = calmer curves")
    h.add_argument("--flow-angle", type=float, default=30.0, help="direction in flat areas / fixed mode")
    h.add_argument("--layers", type=int, default=4, choices=[1, 2, 3, 4])
    h.add_argument("--angles", type=float, nargs="+", default=[0, 90, 45, -45],
                   help="per-layer offset from the flow direction (deg)")
    h.add_argument("--thresholds", type=float, nargs="+", default=[0.12, 0.40, 0.62, 0.80],
                   help="darkness where each layer starts")
    h.add_argument("--spacing-max", type=float, default=2.0, help="mm, lightest spacing")
    h.add_argument("--spacing-min", type=float, help="mm, darkest spacing (default 2.5 x pen)")
    h.add_argument("--min-len", type=float, default=5.0)
    h.add_argument("--max-len", type=float, default=120.0)
    h.add_argument("--seed", type=int, default=1)
    h.add_argument("--tone-smooth", type=float, default=1.0,
                   help="mm; smooths layer boundaries (fewer stubs, softer tone edges)")
    h.add_argument("--seed-margin", type=float, default=0.05,
                   help="start lines only this much darker than the layer threshold")
    h.add_argument("--overrun", type=float, default=0.10,
                   help="lines may run past their layer boundary by up to this much darkness (staggers ends)")
    h.add_argument("--tone-model", default="linear", choices=["linear", "coverage"],
                   help="coverage: spacing from real ink coverage of crossing layers (richer tones, more ink)")
    h.add_argument("--merge-gap", type=float, default=0.0,
                   help="mm; join aligned stroke ends closer than this (fewer pen lifts). Try 4")
    h.add_argument("--merge-angle", type=float, default=50.0)
    h.add_argument("--outline", type=float, nargs="*", metavar="LEVEL",
                   help="draw contour lines at these darkness levels on top, e.g. --outline 0.5")
    h.add_argument("--outline-smooth", type=float, default=2.5)
    h.add_argument("--outline-min-len", type=float, default=15.0)
    h.add_argument("--min-len-spacings", type=float, default=4.0,
                   help="drop strokes shorter than N local line spacings")
    _add_gcode_args(h)
    h.set_defaults(fn=cmd_hatch)

    c = sub.add_parser("scribble", help="gradient-tracing scribble (fast port of scribble-code/c_v25)")
    c.add_argument("image")
    c.add_argument("-o", "--output", default="out/scribble")
    c.add_argument("--preset", default="c_v25", choices=["c_v25", "app5", "app7"],
                   help="c_v25: even texture; app5: high contrast, white highlights, least ink; "
                        "app7: like app5, denser hatching. Flags below override the preset.")
    c.add_argument("--paper", choices=list(PAPER_SIZES), help="default A4")
    c.add_argument("--portrait", action="store_true")
    c.add_argument("--padding", type=float, help="mm, default 20")
    c.add_argument("--pen", type=float, help="mm, default 0.3")
    c.add_argument("--rotate", choices=["auto", "none", "90"], help="default auto")
    c.add_argument("--max-side", type=int, help="downscale so the longer side is at most N px "
                   "(controls line density on paper: aim for >= pen width per pixel)")
    c.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    c.add_argument("--gradient-threshold", type=float)
    c.add_argument("--cleanup-radius", type=int, help="max gap between main lines in px (c_v25 4, app 20)")
    c.add_argument("--cleanup-mode", choices=["adaptive", "quadrant"])
    c.add_argument("--white-cut", type=float, help="0..1: keep areas lighter than this white (app 0.2)")
    c.add_argument("--max-lines", type=int)
    c.add_argument("--light-percentile", type=float)
    c.add_argument("--kernel", type=int, help="trace kernel size d")
    c.add_argument("--sigma", type=float, help="trace kernel sigma")
    c.add_argument("--denoise", action="store_true", help="non-local-means denoise first (slow)")
    c.add_argument("--no-hatch", action="store_true")
    c.add_argument("--hatch-source", choices=["all", "longest", "merged"])
    c.add_argument("--hatch-mode", choices=["legacy", "clean", "ink"],
                   help="legacy: original; clean: no U-turns/spirals/clumps, evenly spaced, fills flat dark "
                        "areas; ink: clean, but only adds the darkness the main lines haven't drawn")
    c.add_argument("--ink-passes", type=int, help="ink mode: 1 = lighter, 2 = calibrated to the photo's tone (default)")
    c.add_argument("--hatch-spacing", type=float, nargs=2, metavar=("DARK", "LIGHT"),
                   help="clean: hatch spacing in px (default 1.5 4)")
    c.add_argument("--hatch-fill", type=float, help="clean: px scale flat areas borrow direction from (0 = off)")
    c.add_argument("--momentum", type=float, help="0..0.95 main-line momentum (0.5 = calmer contours)")
    c.add_argument("--smooth", type=int, help="main-line smoothing window in points (e.g. 3)")
    c.add_argument("--hatch-interval", type=int)
    c.add_argument("--hatch-max-length", type=int)
    c.add_argument("--hatch-angles", help="angle:min_darkness pairs, e.g. 210:0.5,150:0.7")
    c.add_argument("--simplify", type=float, help="RDP tolerance in px (default 0.5)")
    c.add_argument("--join", type=float, help="join line ends closer than N px (default 1.5)")
    c.add_argument("--sampler", choices=["fast", "exact"])
    c.add_argument("--workers", type=int, help="processes for hatching (0 = all cores)")
    c.add_argument("--seed", type=int)
    _add_gcode_args(c)
    c.set_defaults(fn=cmd_scribble)

    l = sub.add_parser("lines2gcode", help="radon renderer JSON -> G-code/SVG")
    l.add_argument("json")
    l.add_argument("-o", "--output", default="out/radon")
    l.add_argument("--no-order", action="store_true", help="keep original line order")
    _add_gcode_args(l)
    l.set_defaults(fn=cmd_lines2gcode)

    u = sub.add_parser("ui", help="local preview UI in the browser")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--workdir", default="out/ui", help="where uploads and renders are kept")
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(fn=lambda a: __import__("plotter.ui", fromlist=["serve"]).serve(
        a.host, a.port, a.workdir, open_browser=not a.no_browser))

    p = sub.add_parser("profile", help="write default plotter profile")
    p.add_argument("--dump", default="plotter_profile.json")
    p.set_defaults(fn=cmd_profile)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
