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
    # Short hop: for pen-up travel shorter than hop_max_mm, lift with hop_up instead of pen_up
    # (a lower Z, so the lift is quicker). Dash gaps and scribble restarts are mostly short.
    # Empty = always the full pen_up. The pen must still clear the paper at the hop height.
    hop_up: str = ""
    hop_max_mm: float = 10.0
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
    # Corner rounding: replace each joint of a stroke with a short curve that stays within
    # this distance (mm) of the original corner, so the controller can take it without
    # braking. 0 = off. 0.05 is invisible with a 0.1-0.3 mm pen.
    corner_tol_mm: float = 0.05
    # Machine, for the time estimate only (defaults = hardware/fluidnc/config_dlc32_extruh.yaml).
    # The estimate replays the firmware planner: trapezoid speed profiles, junction deviation,
    # limited look-ahead, full stop at every pen move and dwell.
    accel_mm_s2: float = 1500.0
    rapid_feed: float = 10000.0           # max_rate_mm_per_min of X/Y (G0 speed)
    z_accel_mm_s2: float = 300.0
    z_rapid_feed: float = 3000.0
    junction_deviation_mm: float = 0.01   # FluidNC default
    planner_blocks: int = 16              # FluidNC default
    # Bounds check: refuse G-code with any X/Y target outside 0..machine size (G-code coordinates,
    # i.e. after flip and offsets). Defaults = max_travel_mm in the FluidNC config. Assumes the work
    # origin is the homed corner (0, 0); 0 = no machine check.
    check_bounds: bool = True
    machine_x_mm: float = 320.0
    machine_y_mm: float = 380.0
    # fallback when the pen commands have no Z word (e.g. M3/M5 servo): seconds per pen move
    pen_move_s: float = 0.15
    # Ink limit: at the final size and true pen width, cut line pieces where the paper would get
    # more than this many full covers of ink per ink_cell_mm cell. Past ~1 cover the paper can't
    # get darker; the pen only scrubs it (soaked paper, tears). 0 = off.
    ink_cap: float = 3.0
    ink_cell_mm: float = 0.5

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
    if len(pls) > 2000:
        return _order_kdtree(pls, start)
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


def _order_kdtree(pls, start):
    """Same greedy nearest-endpoint ordering, using a KD-tree: O(n log n)
    instead of O(n^2), for tens of thousands of strokes (scribble style)."""
    from scipy.spatial import cKDTree
    n = len(pls)
    ends = np.empty((2 * n, 2))
    ends[0::2] = [p[0] for p in pls]
    ends[1::2] = [p[-1] for p in pls]
    alive = np.ones(n, dtype=bool)
    idx_map = np.arange(2 * n)               # tree row -> endpoint id
    tree = cKDTree(ends)
    n_alive = n
    pos = np.asarray(start, dtype=float)
    ordered = []
    while n_alive:
        k = 8
        found = None
        while found is None:
            k = min(k, len(idx_map))
            _, rows = tree.query(pos, k=k)
            rows = np.atleast_1d(rows)
            for r in rows:
                if r >= len(idx_map):
                    continue
                e = idx_map[r]
                if alive[e >> 1]:
                    found = e
                    break
            if found is None:
                if k >= len(idx_map):
                    break
                k *= 4
        i, is_end = found >> 1, found & 1
        pl = pls[i][::-1] if is_end else pls[i]
        alive[i] = False
        n_alive -= 1
        ordered.append(pl)
        pos = np.asarray(pl[-1], dtype=float)
        if n_alive and n_alive * 2 < len(idx_map) // 2:   # rebuild with live endpoints only
            keep = alive[idx_map >> 1]
            idx_map = idx_map[keep]
            tree = cKDTree(ends[idx_map])
    return ordered


def round_corners(pl, tol, step_deg=45.0, min_deg=8.0):
    """Replace each interior joint with a quadratic Bezier whose legs are long enough
    that the curve passes within `tol` of the corner (deviation = r*sin(phi/2)/2),
    capped at half of each adjacent segment. A turn of phi gets ~phi/step_deg + 1 segments
    (one extra point per corner = a chamfer already halves each bend; more points add
    file size for little speed). Bends under min_deg are left alone: the controller
    takes them at full speed anyway."""
    if tol <= 0 or len(pl) < 3:
        return pl
    out = [tuple(pl[0])]
    for i in range(1, len(pl) - 1):
        (ax, ay), (bx, by), (cx, cy) = pl[i - 1], pl[i], pl[i + 1]
        ux, uy, wx, wy = bx - ax, by - ay, cx - bx, cy - by
        lu, lw = math.hypot(ux, uy), math.hypot(wx, wy)
        if lu < 1e-9 or lw < 1e-9:
            continue
        cphi = (ux * wx + uy * wy) / (lu * lw)
        phi = math.acos(max(-1.0, min(1.0, cphi)))
        s2 = math.sin(phi / 2)
        r = min(0.5 * lu, 0.5 * lw, 2 * tol / s2) if s2 > 1e-6 else 0.0
        if phi < math.radians(min_deg) or r < 1e-3:
            out.append((bx, by))
            continue
        p0 = (bx - ux / lu * r, by - uy / lu * r)
        p2 = (bx + wx / lw * r, by + wy / lw * r)
        n = int(math.degrees(phi) / step_deg) + 1
        for k in range(n + 1):
            t = k / n
            a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
            out.append((a * p0[0] + b * bx + c * p2[0], a * p0[1] + b * by + c * p2[1]))
    out.append(tuple(pl[-1]))
    return out


def limit_ink(polylines, pen_mm, cap, cell_mm=0.5, min_piece_mm=1.0, fixed=()):
    """Cut the parts of strokes that would put more than `cap` full covers of ink on any
    cell_mm x cell_mm patch of paper (ink = pen width x length). Strokes are taken in
    drawing order, so what is drawn first (contours, string-art lines) is kept and
    later overdraw is dropped. Pieces shorter than min_piece_mm are dropped too (they'd
    cost a pen lift for a dot). `fixed` strokes count as ink already on the paper but are
    never cut and not returned. Units only need to agree (mm, or pixels with pen in pixels).
    Returns (polylines, stats)."""
    if cap <= 0 or not polylines:
        return polylines, {"ink_removed_mm": 0.0, "ink_removed_pct": 0.0}
    allp = np.concatenate([np.asarray(p, float) for p in list(polylines) + list(fixed) if len(p) > 1])
    x0, y0 = allp.min(0) - cell_mm
    nx, ny = (np.ceil((allp.max(0) - allp.min(0)) / cell_mm) + 3).astype(int)
    load = np.zeros(nx * ny)
    step = cell_mm / 4
    unit = step * pen_mm / cell_mm ** 2               # covers one sample adds to its cell
    limit = cap + 1e-9
    for pl in fixed:
        P = np.asarray(pl, float)
        if len(P) < 2:
            continue
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        n = max(1, int(math.ceil(cum[-1] / step)))
        s = (np.arange(n) + 0.5) * (cum[-1] / n)
        x, y = np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])
        np.add.at(load, ((y - y0) / cell_mm).astype(int) * nx + ((x - x0) / cell_mm).astype(int),
                  unit * (cum[-1] / n) / step)
    out, total, removed = [], 0.0, 0.0
    for pl in polylines:
        P = np.asarray(pl, float)
        if len(P) < 2:
            continue
        seg = np.hypot(*np.diff(P, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        L = cum[-1]
        total += L
        if L <= 0:
            continue
        n = max(1, int(math.ceil(L / step)))
        s = (np.arange(n) + 0.5) * (L / n)             # sample positions along the stroke
        k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
        t = (s - cum[k]) / np.maximum(seg[k], 1e-12)
        q = P[k] + t[:, None] * (P[k + 1] - P[k])
        cid = ((q[:, 1] - y0) / cell_mm).astype(int) * nx + ((q[:, 0] - x0) / cell_mm).astype(int)
        w = unit * (L / n) / step
        # running count of this stroke's own samples per cell (exact sequential check: a sample is
        # kept iff the cell plus every earlier kept sample of this stroke stays under the cap;
        # earlier samples in the same cell are kept whenever a later one is)
        order = np.argsort(cid, kind="stable")
        sc = cid[order]
        first = np.r_[True, sc[1:] != sc[:-1]]
        idx = np.arange(n)
        rank = idx - np.maximum.accumulate(np.where(first, idx, 0))
        occ = np.empty(n, int)
        occ[order] = rank
        keep = load[cid] + (occ + 1) * w <= limit
        if keep.all():
            np.add.at(load, cid, w)
            out.append(pl)
            continue
        np.add.at(load, cid[keep], w)
        # kept runs -> sub-polylines: run start/end points plus the original vertices in between
        edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
        for a, b in zip(edges[0::2], edges[1::2] - 1):
            sa_, sb_ = s[a] - 0.5 * L / n, s[b] + 0.5 * L / n
            if sb_ - sa_ < min_piece_mm:
                removed += sb_ - sa_
                continue
            pa = np.interp(sa_, cum, P[:, 0]), np.interp(sa_, cum, P[:, 1])
            pb = np.interp(sb_, cum, P[:, 0]), np.interp(sb_, cum, P[:, 1])
            mid = [tuple(v) for v in P[(cum > sa_) & (cum < sb_)]]
            out.append([pa, *mid, pb])
        removed += (~keep).sum() * (L / n)
    return out, {"ink_removed_mm": removed, "ink_removed_pct": 100.0 * removed / max(total, 1e-9)}


def ink_load(polylines, pen_mm, cell_mm=1.0):
    """Covers of ink per cell (1.0 = paper covered once), for checking a plot. Returns
    (load grid, origin (x0, y0))."""
    allp = np.concatenate([np.asarray(p, float) for p in polylines if len(p) > 1])
    x0, y0 = allp.min(0) - cell_mm
    nx, ny = (np.ceil((allp.max(0) - allp.min(0)) / cell_mm) + 3).astype(int)
    load = np.zeros(nx * ny)
    step = cell_mm / 4
    for pl in polylines:
        P = np.asarray(pl, float)
        if len(P) < 2:
            continue
        seg = np.hypot(*np.diff(P, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        n = max(1, int(math.ceil(cum[-1] / step)))
        s = (np.arange(n) + 0.5) * (cum[-1] / n)
        x, y = np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])
        np.add.at(load, ((y - y0) / cell_mm).astype(int) * nx + ((x - x0) / cell_mm).astype(int),
                  (cum[-1] / n) * pen_mm / cell_mm ** 2)
    return load.reshape(ny, nx), (x0, y0)


def _z_of(cmd):
    import re
    m = re.search(r"Z\s*(-?[\d.]+)", cmd or "")
    return float(m.group(1)) if m else None


def estimate_time(events, p):
    """Plot time (s) from motion events, replaying a GRBL/FluidNC-style planner.
    events: ("xy", dx, dy, feed_mm_min or None for rapid) | ("stop", seconds) | ("z", dz_mm).
    Returns seconds: total, draw, ideal_draw (all drawing at full feed), travel, pen."""
    res = dict(total=0.0, draw=0.0, ideal_draw=0.0, travel=0.0, pen=0.0)
    run = []
    a = p.accel_mm_s2
    jd = p.junction_deviation_mm
    k = max(1, int(p.planner_blocks) - 1)

    def flush():
        if not run:
            return
        n = len(run)
        L = [b[0] for b in run]
        V = [b[1] for b in run]
        vj = [0.0] * (n + 1)
        for i in range(1, n):
            # GRBL junction speed; c = cos of the angle between reversed previous and next direction
            c = -(run[i - 1][2] * run[i][2] + run[i - 1][3] * run[i][3])
            s2 = math.sqrt(max(0.0, 0.5 * (1 - c)))
            if s2 >= 0.999999:
                vj[i] = min(V[i - 1], V[i])
            elif s2 > 1e-6:
                vj[i] = min(V[i - 1], V[i], math.sqrt(a * jd * s2 / (1 - s2)))
        # limited look-ahead: must be able to stop within the next planner_blocks blocks
        cum = [0.0]
        for d in L:
            cum.append(cum[-1] + d)
        for i in range(1, n):
            vj[i] = min(vj[i], math.sqrt(2 * a * (cum[min(n, i + k)] - cum[i])))
        for i in range(n - 1, -1, -1):
            vj[i] = min(vj[i], math.sqrt(vj[i + 1] ** 2 + 2 * a * L[i]))
        for i in range(n):
            vj[i + 1] = min(vj[i + 1], math.sqrt(vj[i] ** 2 + 2 * a * L[i]))
        for i in range(n):
            v0, v1, vm, d = vj[i], vj[i + 1], V[i], L[i]
            vp = math.sqrt((2 * a * d + v0 * v0 + v1 * v1) / 2)
            if vp < vm:
                dt = (2 * vp - v0 - v1) / a
            else:
                da, dd = (vm * vm - v0 * v0) / (2 * a), (vm * vm - v1 * v1) / (2 * a)
                dt = (2 * vm - v0 - v1) / a + (d - da - dd) / vm
            res["travel" if run[i][4] else "draw"] += dt
            res["total"] += dt
        run.clear()

    for e in events:
        if e[0] == "xy":
            _, dx, dy, feed = e
            d = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            v = (feed if feed else p.rapid_feed) / 60.0
            if feed:
                res["ideal_draw"] += d / v
            run.append((d, v, dx / d, dy / d, feed is None))
        else:
            flush()
            if e[0] == "stop":
                dt = e[1]
            else:
                d, vm, az = abs(e[1]), p.z_rapid_feed / 60.0, p.z_accel_mm_s2
                dt = 2 * math.sqrt(d / az) if d < vm * vm / az else d / vm + vm / az
            res["pen"] += dt
            res["total"] += dt
    flush()
    return res


class OutOfBounds(ValueError):
    """The G-code would move outside the machine or draw off the paper."""


def check_bounds(p, ext, extra_lines, off_paper, paper_width, paper_height):
    """Raise OutOfBounds when any XY target (ext = [xmin, ymin, xmax, ymax] in G-code
    coordinates), or an X/Y word in the header/footer, lies outside 0..machine size,
    or a stroke point lies off the paper. Called before anything is written."""
    import re
    for line in extra_lines:
        line = (line or "").split(";")[0]
        if re.match(r"\s*G0*[01]\b", line, re.I):
            for ax, i in (("X", 0), ("Y", 1)):
                m = re.search(ax + r"\s*(-?[\d.]+)", line, re.I)
                if m:
                    v = float(m.group(1))
                    ext[i], ext[i + 2] = min(ext[i], v), max(ext[i + 2], v)
    problems = []
    if off_paper:
        pts = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in off_paper)
        problems.append(f"strokes leave the {paper_width:g} x {paper_height:g} mm paper, e.g. at {pts}")
    if p.check_bounds and p.machine_x_mm > 0 and p.machine_y_mm > 0 and ext[0] <= ext[2]:
        tol = 1e-6
        if ext[0] < -tol or ext[1] < -tol or ext[2] > p.machine_x_mm + tol or ext[3] > p.machine_y_mm + tol:
            problems.append(f"moves span X {ext[0]:.1f}..{ext[2]:.1f}, Y {ext[1]:.1f}..{ext[3]:.1f} mm, "
                            f"but the machine area is X 0..{p.machine_x_mm:g}, Y 0..{p.machine_y_mm:g} "
                            f"(profile machine_x_mm / machine_y_mm)")
    if problems:
        raise OutOfBounds("G-code not written: " + "; ".join(problems) +
                          ". Use a smaller paper size, more padding, or change the offsets.")


def write_gcode(polylines, path, profile=None, paper_height=None, comment=None, paper_width=None):
    """Write polylines (lists of (x, y) in mm, y-down) as plotter G-code.

    Consecutive polylines whose end/start coincide are drawn without lifting
    the pen. Returns a stats dict (draw / travel distance, estimated time).
    Raises OutOfBounds (and writes nothing) when a move would leave the machine
    area, or, when paper_width is given, a stroke would leave the paper.
    """
    p = profile or PlotterProfile()
    if p.flip_y and paper_height is None:
        raise ValueError("flip_y requires paper_height")

    fmt = f"{{:.{p.decimals}f}}"

    ext = [math.inf, math.inf, -math.inf, -math.inf]     # machine-coordinate extents of every XY target
    off_paper = []

    def xy(pt):
        if paper_width is not None and paper_height is not None and len(off_paper) < 3 and not (
                -0.01 <= pt[0] <= paper_width + 0.01 and -0.01 <= pt[1] <= paper_height + 0.01):
            off_paper.append(pt)
        x = pt[0] + p.offset_x
        y = (paper_height - pt[1]) if p.flip_y else pt[1]
        y += p.offset_y
        ext[0], ext[1], ext[2], ext[3] = min(ext[0], x), min(ext[1], y), max(ext[2], x), max(ext[3], y)
        return f"X{fmt.format(x)} Y{fmt.format(y)}"

    out = []
    if comment:
        out += [f"; {line}" for line in comment.splitlines()]
    out += list(p.header)

    events = []
    z_up, z_down = _z_of(p.pen_up), _z_of(p.pen_down)
    feed_now = None                          # F is modal: written only when it changes

    z_now = [None]

    def pen(cmd):
        out.append(cmd)
        z = _z_of(cmd)
        if z is not None and z_up is not None and z_down is not None:
            events.append(("z", (z - z_now[0]) if z_now[0] is not None else (z_up - z_down)))
            z_now[0] = z
        else:
            events.append(("stop", p.pen_move_s))
        if p.dwell_s > 0:
            out.append(f"G4 P{p.dwell_s:g}")
            events.append(("stop", p.dwell_s))

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
        if p.corner_tol_mm > 0:
            pl = round_corners(pl, p.corner_tol_mm)
        if pos is None or _dist(pos, pl[0]) > eps:
            if pen_is_down:
                hop = p.hop_up and _dist(pos, pl[0]) < p.hop_max_mm
                pen(p.hop_up if hop else p.pen_up)
                pen_is_down = False
            if pos is not None:
                travel_mm += _dist(pos, pl[0])
                events.append(("xy", pl[0][0] - pos[0], pl[0][1] - pos[1], None))
            out.append(f"G0 {xy(pl[0])}{travel_f}")
            if p.travel_feed:
                feed_now = p.travel_feed
        if not pen_is_down:
            pen(p.pen_down)
            pen_is_down = True
        for a, b in zip(pl, pl[1:]):
            seg = _dist(a, b)
            if seg <= eps:
                continue
            if feed_now != p.draw_feed:
                out.append(f"G1 {xy(b)} F{p.draw_feed:g}")
                feed_now = p.draw_feed
            else:
                out.append(f"G1 {xy(b)}")
            events.append(("xy", b[0] - a[0], b[1] - a[1], p.draw_feed))
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

    check_bounds(p, ext, list(p.header) + list(p.footer) + [p.pause_command], off_paper,
                 paper_width, paper_height)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")

    lifts = sum(1 for line in out if line == p.pen_up or (p.hop_up and line == p.hop_up))
    t = estimate_time(events, p)
    return {
        "draw_mm": draw_mm,
        "travel_mm": travel_mm,
        "pen_lifts": lifts,
        "pen_lift_minutes": t["pen"] / 60.0,   # pen moves + dwells
        "est_minutes": t["total"] / 60.0,
        # drawing time with the planner vs. if every segment ran at full feed:
        # the difference is time lost braking at joints
        "draw_minutes": t["draw"] / 60.0,
        "draw_ideal_minutes": t["ideal_draw"] / 60.0,
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
