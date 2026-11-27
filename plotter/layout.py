"""
Sheet layout: place several finished G-code drawings on one sheet of paper.

parse_gcode() reads plotter G-code back into pen-down polylines, whatever wrote it:
our own files, svg2gcode / scribble-code output (G2/G3 arcs, M3/M5 servo pens), or
hand-edited ones. Pen state comes from Z (the lowest Z used is "down") or, for files
without Z, from M3 (down) / M5 (up).

compose() places parsed drawings on a sheet (position, scale, 90° rotation) and
returns paper polylines (mm, y-down) ready for gcode.write_gcode, so the combined
file gets the same bounds check, time estimate and profile as any other plot.
"""

import math
import re

WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")


def parse_gcode(text, arc_step_mm=0.3):
    """Pen-down polylines in the file's own coordinates (mm, y-up as written).
    Returns (polylines, info)."""
    cmds = []                                   # (g, x, y, z, i, j, m, s)
    zs = set()
    has_m3 = False
    g_modal, inch = 0, False
    for raw in text.splitlines():
        line = raw.split(";")[0].split("(")[0].strip().upper()
        if not line:
            continue
        words = WORD.findall(line)
        if not words:
            continue
        d = {}
        gs, ms = [], []
        for k, v in words:
            if k == "G":
                gs.append(float(v))
            elif k == "M":
                ms.append(int(float(v)))
            else:
                d[k] = float(v)
        for g in gs:
            if g == 91:
                raise ValueError("relative coordinates (G91) are not supported")
            if g == 20:
                inch = True
            if g == 21:
                inch = False
            if g in (0, 1, 2, 3):
                g_modal = int(g)
        for m in ms:
            if m in (3, 4):
                has_m3 = True
            cmds.append(("M", m, d.get("S")))
        if any(k in d for k in "XYZ"):
            if "Z" in d:
                zs.add(round(d["Z"] * (25.4 if inch else 1), 4))
            f = 25.4 if inch else 1.0
            cmds.append(("G", g_modal, *(d.get(k, None) if d.get(k) is None else d[k] * f
                                          for k in ("X", "Y", "Z", "I", "J", "R"))))
    use_z = len(zs) >= 2
    z_down = min(zs) if use_z else None

    pls, cur = [], []
    x = y = 0.0
    z = None
    down = False

    def pen(state):
        nonlocal down, cur
        if down and not state and len(cur) > 1:
            pls.append(cur)
        if not state or not down:
            cur = []
        down = state

    for c in cmds:
        if c[0] == "M":
            if not use_z and has_m3:
                if c[1] in (3, 4):
                    pen(c[2] is None or c[2] > 0)
                elif c[1] == 5:
                    pen(False)
            continue
        _, g, nx, ny, nz, ci, cj, cr = c
        nx = x if nx is None else nx
        ny = y if ny is None else ny
        if nz is not None:
            z = nz
            if use_z:
                pen(z <= z_down + 1e-6)
        drawing = down and g != 0
        if drawing and not cur:
            cur = [(x, y)]
        if drawing and g in (2, 3) and (ci is not None or cj is not None or cr is not None):
            pts = _arc(x, y, nx, ny, ci or 0.0, cj or 0.0, cr, g == 2, arc_step_mm)
            cur.extend(pts)
        elif drawing:
            if (nx, ny) != (x, y):
                cur.append((nx, ny))
        elif down and g == 0 and (nx, ny) != (x, y):
            # rapid with the pen down: treat as a break (never draw rapids)
            if len(cur) > 1:
                pls.append(cur)
            cur = []
        x, y = nx, ny
    if down and len(cur) > 1:
        pls.append(cur)
    pls = [p for p in pls if len(p) > 1]
    info = {"pen": "Z" if use_z else ("M3/M5" if has_m3 else "none"), "z_down": z_down,
            "strokes": len(pls)}
    if not pls:
        raise ValueError("no pen-down moves found (pen state from Z or M3/M5)")
    return pls, info


def _arc(x0, y0, x1, y1, i, j, r, cw, step):
    if r is not None and i == 0 and j == 0:
        # radius form: centre on the perpendicular bisector
        dx, dy = x1 - x0, y1 - y0
        dd = math.hypot(dx, dy)
        if dd == 0 or abs(r) < dd / 2:
            return [(x1, y1)]
        h = math.sqrt(r * r - dd * dd / 4) * (1 if (r > 0) != cw else -1)
        cx, cy = x0 + dx / 2 - h * dy / dd, y0 + dy / 2 + h * dx / dd
    else:
        cx, cy = x0 + i, y0 + j
    rad = math.hypot(x0 - cx, y0 - cy)
    a0, a1 = math.atan2(y0 - cy, x0 - cx), math.atan2(y1 - cy, x1 - cx)
    sweep = a1 - a0
    if cw and sweep >= 0:
        sweep -= 2 * math.pi
    elif not cw and sweep <= 0:
        sweep += 2 * math.pi
    n = max(1, int(abs(sweep) * rad / step))
    pts = [(cx + rad * math.cos(a0 + sweep * k / n), cy + rad * math.sin(a0 + sweep * k / n)) for k in range(1, n)]
    pts.append((x1, y1))
    return pts


def normalise(pls):
    """File coordinates (y-up) -> local drawing coordinates: y-down, bounding box at (0, 0).
    Returns (polylines, width, height)."""
    xs = [p[0] for pl in pls for p in pl]
    ys = [p[1] for pl in pls for p in pl]
    x0, y1 = min(xs), max(ys)
    w, h = max(xs) - x0, y1 - min(ys)
    return [[(px - x0, y1 - py) for px, py in pl] for pl in pls], w, h


def place(pls, w, h, x, y, scale, rot):
    """Local drawing (w x h, y-down) -> paper: rotate clockwise by rot (0/90/180/270),
    scale, then put its top-left corner at (x, y)."""
    rot = int(rot) % 360
    s = float(scale)

    def f(u, v):
        if rot == 90:
            u, v = h - v, u
        elif rot == 180:
            u, v = w - u, h - v
        elif rot == 270:
            u, v = v, w - u
        return (x + u * s, y + v * s)

    return [[f(u, v) for u, v in pl] for pl in pls]


def compose(items, sources):
    """items: [{id, x, y, scale, rot}], sources: id -> (local polylines, w, h).
    Returns paper polylines, drawn item by item (each item keeps its own stroke order)."""
    out = []
    for it in items:
        if it["id"] not in sources:
            raise KeyError(f"drawing {it['id']} is no longer loaded on the server; add it again")
        pls, w, h = sources[it["id"]]
        out += place(pls, w, h, float(it.get("x", 0)), float(it.get("y", 0)),
                     float(it.get("scale", 1)), it.get("rot", 0))
    return out
