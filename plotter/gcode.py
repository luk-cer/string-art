"""
G-code writer for pen plotters.

The default profile reproduces the dialect of the hand-edited files that
plotted correctly (svg2gcode style):

    G21
    G90
    G0 Z5          ; pen up
    G4 P0.05       ; settle
    G0 X.. Y..     ; travel
    G0 Z0          ; pen down
    G4 P0.05
    G1 X.. Y.. F1500

Everything is configurable via a JSON profile file so the next time the
header needs a tweak it is a config change, not a manual edit of the output.
"""

from dataclasses import dataclass, field, asdict, fields
import json
import math

import numpy as np


@dataclass
class PlotterProfile:
    pen_up: str = "G0 Z5"           # command that lifts the pen
    pen_down: str = "G0 Z0"         # command that lowers the pen
    dwell_s: float = 0.05           # G4 pause after each pen move (0 = none)
    draw_feed: float = 1500.0       # mm/min for G1 drawing moves
    travel_feed: float | None = None  # mm/min for G0 travel (None = machine rapid)
    decimals: int = 3               # coordinate precision (0.001 mm is plenty)
    header: list[str] = field(default_factory=lambda: ["G21", "G90"])
    footer: list[str] = field(default_factory=lambda: ["G0 X0 Y0"])
    # Coordinate transform: SVG / image coords are y-down, most plotters are
    # y-up with origin bottom-left. flip_y mirrors around the paper height.
    flip_y: bool = True
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Optional pause for checking / swapping the pen every N mm of drawing
    # (0 = never). Uses M0, which GRBL treats as "program pause".
    pause_every_mm: float = 0.0
    pause_command: str = "M0"

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown profile keys: {sorted(unknown)}")
        return cls(**data)

    def dump(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(pl):
    return sum(_dist(pl[i], pl[i + 1]) for i in range(len(pl) - 1))


def order_polylines(polylines, start=(0.0, 0.0)):
    """Greedy nearest-neighbour ordering with optional reversal.

    Cuts pen-up travel dramatically for unordered stroke soups. Vectorised:
    O(n^2) work but only n numpy calls.
    """
    pls = [list(p) for p in polylines if len(p) >= 2]
    if not pls:
        return []
    starts = np.array([p[0] for p in pls], dtype=float)
    ends = np.array([p[-1] for p in pls], dtype=float)
    left = np.ones(len(pls), dtype=bool)
    pos = np.array(start, dtype=float)
    ordered = []
    for _ in range(len(pls)):
        d0 = np.hypot(*(starts - pos).T)
        d1 = np.hypot(*(ends - pos).T)
        d0[~left] = np.inf
        d1[~left] = np.inf
        i0, i1 = int(np.argmin(d0)), int(np.argmin(d1))
        if d1[i1] < d0[i0]:
            pl = pls[i1][::-1]; left[i1] = False
        else:
            pl = pls[i0]; left[i0] = False
        ordered.append(pl)
        pos = np.array(pl[-1], dtype=float)
    return ordered


def write_gcode(polylines, path, profile=None, paper_height=None, comment=None):
    """Write polylines (lists of (x, y) in mm, y-down) as plotter G-code.

    Consecutive polylines whose end/start coincide are drawn without lifting
    the pen. Returns a stats dict (draw / travel distance, estimated time).
    """
    p = profile or PlotterProfile()
    if p.flip_y and paper_height is None:
        raise ValueError("flip_y requires paper_height")

    fmt = f"{{:.{p.decimals}f}}"

    def xy(pt):
        x = pt[0] + p.offset_x
        y = (paper_height - pt[1]) if p.flip_y else pt[1]
        y += p.offset_y
        return f"X{fmt.format(x)} Y{fmt.format(y)}"

    out = []
    if comment:
        out += [f"; {line}" for line in comment.splitlines()]
    out += list(p.header)

    def pen(cmd):
        out.append(cmd)
        if p.dwell_s > 0:
            out.append(f"G4 P{p.dwell_s:g}")

    pen(p.pen_up)
    pen_is_down = False
    pos = None
    draw_mm = travel_mm = 0.0
    since_pause = 0.0
    eps = 10 ** (-p.decimals)
    travel_f = f" F{p.travel_feed:g}" if p.travel_feed else ""

    for pl in polylines:
        if len(pl) < 2:
            continue
        if pos is None or _dist(pos, pl[0]) > eps:
            if pen_is_down:
                pen(p.pen_up)
                pen_is_down = False
            if pos is not None:
                travel_mm += _dist(pos, pl[0])
            out.append(f"G0 {xy(pl[0])}{travel_f}")
        if not pen_is_down:
            pen(p.pen_down)
            pen_is_down = True
        for a, b in zip(pl, pl[1:]):
            seg = _dist(a, b)
            if seg <= eps:
                continue
            out.append(f"G1 {xy(b)} F{p.draw_feed:g}")
            draw_mm += seg
            since_pause += seg
            if p.pause_every_mm and since_pause >= p.pause_every_mm:
                pen(p.pen_up)
                out.append(p.pause_command)
                pen(p.pen_down)
                since_pause = 0.0
        pos = pl[-1]

    if pen_is_down:
        pen(p.pen_up)
    out += list(p.footer)

    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")

    est_min = draw_mm / p.draw_feed + travel_mm / (p.travel_feed or 3000.0)
    return {
        "draw_mm": draw_mm,
        "travel_mm": travel_mm,
        "est_minutes": est_min,
        "gcode_lines": len(out),
    }


def write_svg(polylines, path, width_mm, height_mm, pen_width_mm=0.3, extra=""):
    """Plain SVG with mm units: open in Inkscape, or feed to svg2gcode."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
        f'height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">',
        f'<rect width="{width_mm}" height="{height_mm}" fill="white"/>',
        extra,
        f'<g fill="none" stroke="black" stroke-width="{pen_width_mm}" '
        f'stroke-linecap="round" stroke-linejoin="round">',
    ]
    for pl in polylines:
        pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in pl)
        parts.append(f'<polyline points="{pts}"/>')
    parts.append("</g></svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
