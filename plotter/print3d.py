"""
3D-printable line art: a thin plate with the drawing raised on it.

Hung in a window or over a light, the thin plate glows and the raised lines (thicker, or
printed in dark filament after a swap at the plate's top) show up dark - the pen plot,
backlit. In normal light it reads as a relief. Everything sits on the plate, so the print is
one connected piece whatever the style.

    plate  = outline (circle or rounded rectangle) + hanging lug, `base_mm` thick, hole cut
    raised = lines widened to `line_width_mm` + rim + ring round the hole, `line_height_mm`
             tall, starting at z = base_mm (the layer to change filament at)

Geometry: shapely widens the polylines and merges them into outlines; manifold3d extrudes
them and unions the solids (watertight, printable). Both are optional dependencies:
`pip install shapely manifold3d` (or `uv sync --extra print3d`).

Line spacing should be designed for the printed line width: render the drawing with
pen = line width (e.g. 0.7 mm) at the size you print, otherwise neighbouring lines merge.
"""

from dataclasses import dataclass
import io
import math
import struct
import zipfile

import numpy as np

PLA_G_PER_CM3 = 1.24


@dataclass
class PrintConfig:
    line_width_mm: float = 0.7       # line width as it should come out of the printer
    line_spread_mm: float = 0.15     # how much wider the printer lays a line than it's modelled (squish):
                                     # measure a printed line, subtract the modelled width. Lines are
                                     # modelled this much narrower so they print at line_width_mm.
    min_feature_mm: float = 0.42     # narrowest line the slicer still prints (0.4 nozzle)
    line_height_mm: float = 1.2      # how far the lines stand above the plate
    base_mm: float = 0.6             # plate thickness (thin = bright when backlit)
    margin_mm: float = 4.0           # plate beyond the drawing
    shape: str = "auto"              # auto | circle | rect
    rim_mm: float = 2.5              # raised border (0 = none)
    corner_mm: float = 4.0           # rect: corner radius
    hole_mm: float = 4.0             # hanging hole in a lug at the top (0 = no lug)
    max_size_mm: float = 215.0       # bed limit (Creality K1C: 220 x 220)
    simplify_mm: float = 0.02
    # mode "plate": the drawing raised on a solid plate (backlit piece).
    # mode "open": no plate - free-standing line art. The white base layers (base_mm) hold the lines,
    # rim, lug and the bridges that tie every piece together; the dark layers above the filament
    # swap hold only the lines, so on a white wall the thin white bridges all but disappear.
    mode: str = "plate"
    bridge_mm: float = 1.0           # open: bridge width
    web: bool = False                # open: also a sparse white triangle web under everything (stiffer)
    web_mm: float = 12.0             # open: web cell size
    web_width_mm: float = 0.6        # open: web line width
    anchor_mm: float = 20.0          # open: also bridge the rim to the drawing every this many mm (0 = off)
    fill: str = "lines"              # lines: print the lines | gaps: print the space between them, the
                                     # lines stay empty (grooves on a plate, slots when free-standing)
    light_lines: bool = False        # colours: False = dark lines on a white base, True = white lines on a
                                     # dark base (for drawings rendered with invert) - previews and 3MF colours


def _require():
    try:
        import shapely  # noqa: F401
        import manifold3d  # noqa: F401
    except ImportError as e:
        raise RuntimeError("3D export needs shapely and manifold3d: pip install shapely manifold3d "
                           "(or uv sync --extra print3d)") from e


def _contours(geom):
    """shapely (Multi)Polygon -> list of (n, 2) rings for an even-odd CrossSection."""
    polys = list(geom.geoms) if geom.geom_type in ("MultiPolygon", "GeometryCollection") else [geom]
    out = []
    for p in polys:
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        out.append(np.asarray(p.exterior.coords)[:-1])
        out += [np.asarray(r.coords)[:-1] for r in p.interiors]
    return out


def fit_circle(polylines):
    """Least-squares circle through the drawing's convex hull: (cx, cy, r, residual / r).
    String-art lines all end on the nail circle and circular scribbles are clipped to one,
    so their hull fits a circle closely even when the lines are sparse."""
    from shapely.geometry import MultiPoint
    p = np.array([q for pl in polylines for q in pl], float)
    hull = MultiPoint(p).convex_hull
    h = np.asarray(hull.exterior.coords)[:-1] if hull.geom_type == "Polygon" else p
    if len(h) < 5:
        return None
    A = np.c_[h, np.ones(len(h))]
    a, b, c = np.linalg.lstsq(A, (h ** 2).sum(1), rcond=None)[0]
    cx, cy = a / 2, b / 2
    r = math.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    res = np.abs(np.hypot(h[:, 0] - cx, h[:, 1] - cy) - r).max() / r
    return cx, cy, r, res


def detect_shape(polylines):
    """'circle' when the drawing's outline is a circle (within 2 %), else 'rect'."""
    f = fit_circle(polylines)
    return "circle" if f is not None and f[3] < 0.02 else "rect"


def connect(pieces, start_mm=2.0):
    """Shortest bridges that join every polygon of `pieces` into one: Kruskal over the gaps
    between pieces, searched in widening rounds (d, 2d, 4d ...) so only nearby pairs are
    measured. Returns (bridge LineStrings, number of separate pieces before)."""
    import shapely
    from shapely import STRtree
    polys = [g for g in (pieces.geoms if hasattr(pieces, "geoms") else [pieces])
             if g.geom_type == "Polygon" and not g.is_empty]
    n = len(polys)
    if n < 2:
        return [], n
    arr = np.array(polys, dtype=object)
    tree = STRtree(arr)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bridges, comps, d = [], n, start_mm
    while comps > 1 and d < 5000:
        a, b = tree.query(arr, predicate="dwithin", distance=d)
        keep = a < b
        a, b = a[keep], b[keep]
        if len(a):
            roots = np.array([find(i) for i in range(n)])
            keep = roots[a] != roots[b]
            a, b = a[keep], b[keep]
        if len(a):
            dist = shapely.distance(arr[a], arr[b])
            for k in np.argsort(dist, kind="stable"):
                ra, rb = find(int(a[k])), find(int(b[k]))
                if ra != rb:
                    parent[ra] = rb
                    comps -= 1
                    bridges.append(shapely.shortest_line(arr[a[k]], arr[b[k]]))
        d *= 2
    return bridges, n


def _web(inner, cell, width):
    """Triangle lattice (lines at 0/60/120 deg, `cell` apart) clipped to `inner`."""
    import shapely
    from shapely.geometry import MultiLineString
    x0, y0, x1, y1 = inner.bounds
    c, rr = ((x0 + x1) / 2, (y0 + y1) / 2), math.hypot(x1 - x0, y1 - y0) / 2 + cell
    segs = []
    for ang in (0.0, 60.0, 120.0):
        t = math.radians(ang)
        ux, uy, nx, ny = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
        for k in np.arange(-rr, rr + 1e-9, cell * math.sqrt(3) / 2):
            px, py = c[0] + nx * k, c[1] + ny * k
            segs.append([(px - ux * rr, py - uy * rr), (px + ux * rr, py + uy * rr)])
    return shapely.buffer(MultiLineString(segs), width / 2, quad_segs=2).intersection(inner)


def build(polylines, cfg: PrintConfig = PrintConfig()):
    """polylines: drawing in mm (y down, as on paper). Returns {'plate', 'raised', 'combined'}
    manifolds and an info dict."""
    _require()
    import shapely
    from shapely.geometry import MultiLineString, Point, box
    import manifold3d as m3

    pls = [pl for pl in polylines if len(pl) > 1]
    if not pls:
        raise ValueError("no lines to print")
    p = np.array([q for pl in pls for q in pl], float)
    lo, hi = p.min(0), p.max(0)
    W, H = hi - lo
    shape = detect_shape(pls) if cfg.shape == "auto" else cfg.shape
    # paper y-down -> printer y-up; origin at the drawing's centre (circle: the fitted circle's)
    cx, cy = (lo + hi) / 2
    if shape == "circle":
        f = fit_circle(pls)
        if f is not None and f[3] < 0.05:
            cx, cy = f[0], f[1]
    lines = MultiLineString([[(x - cx, cy - y) for x, y in pl] for pl in pls])
    gaps = cfg.fill == "gaps"
    if gaps:   # the filled gaps spread into the slots: model the slots wider so they print at line_width
        model_w = cfg.line_width_mm + cfg.line_spread_mm
        printed_w = cfg.line_width_mm
    else:      # the lines spread: model them narrower (never under what the slicer prints)
        model_w = max(cfg.line_width_mm - cfg.line_spread_mm, cfg.min_feature_mm)
        printed_w = model_w + cfg.line_spread_mm
    half_w = model_w / 2
    pad = cfg.margin_mm + half_w + cfg.rim_mm
    if shape == "circle":
        R = np.hypot(p[:, 0] - cx, p[:, 1] - cy).max() + pad
        outline = Point(0, 0).buffer(R, quad_segs=64)
        top = R
    else:
        rc = min(cfg.corner_mm, W / 2 + pad, H / 2 + pad)
        outline = box(-W / 2 - pad + rc, -H / 2 - pad + rc, W / 2 + pad - rc, H / 2 + pad - rc).buffer(rc, quad_segs=16)
        top = H / 2 + pad
    inner = outline.buffer(-cfg.rim_mm) if cfg.rim_mm > 0 else outline
    rim = outline.difference(inner) if cfg.rim_mm > 0 else None

    region = shapely.buffer(shapely.simplify(lines, cfg.simplify_mm), half_w, quad_segs=3)
    region = region.intersection(inner)
    if gaps:
        filled = inner.difference(region)
        speck = max(0.3, 2 * cfg.min_feature_mm ** 2)     # gaps too small to print are left out
        keep = [g for g in getattr(filled, "geoms", [filled]) if g.geom_type == "Polygon" and g.area >= speck]
        top2d = shapely.MultiPolygon(keep) if keep else filled
    else:
        top2d = region
    raised = top2d if rim is None else top2d.union(rim)
    hole = lug = ring = None
    if cfg.hole_mm > 0:
        r_lug = cfg.hole_mm / 2 + 3.0
        lug_c = (0.0, top + cfg.hole_mm / 2 + 0.5)
        hole = Point(lug_c).buffer(cfg.hole_mm / 2, quad_segs=32)
        lug = Point(lug_c).buffer(r_lug, quad_segs=32).union(
            box(-r_lug, top - 2.0, r_lug, lug_c[1]))              # neck into the plate / rim
        ring = Point(lug_c).buffer(r_lug, quad_segs=32).difference(
            Point(lug_c).buffer(r_lug - 1.6, quad_segs=32))       # stiff ring round the hole
        raised = raised.union(ring)
    extra = None
    info_open = {}
    if cfg.mode == "open":
        body = raised if lug is None else raised.union(lug)
        if cfg.web:
            extra = _web(inner, cfg.web_mm, cfg.web_width_mm)
            body = body.union(extra)
        bridges, n_pieces = connect(body)
        if not gaps and cfg.anchor_mm > 0 and rim is not None and not region.is_empty:   # (gaps: fill meets the rim)
            # one bridge would hold the whole drawing: tie it to the rim all the way round
            edge = inner.exterior
            reach = cfg.margin_mm + cfg.line_width_mm + 15.0
            for t in np.arange(0, edge.length, cfg.anchor_mm):
                ln = shapely.shortest_line(edge.interpolate(t), region)
                if 1e-6 < ln.length <= reach:
                    bridges.append(ln)
        if bridges:
            bz = shapely.buffer(shapely.MultiLineString([list(b.coords) for b in bridges]),
                                cfg.bridge_mm / 2, quad_segs=3)
            body = body.union(bz)
            extra = bz if extra is None else extra.union(bz)
        lens = [b.length for b in bridges]
        info_open = {"pieces": n_pieces, "bridges": len(bridges),
                     "bridge_m": round(sum(lens) / 1000, 2),
                     "longest_bridge_mm": round(max(lens), 1) if lens else 0.0}
    else:
        body = outline if lug is None else outline.union(lug)
    if hole is not None:
        body, raised = body.difference(hole), raised.difference(hole)
        if extra is not None:
            extra = extra.difference(hole)

    plate = m3.CrossSection(_contours(body), m3.FillRule.EvenOdd).extrude(cfg.base_mm)
    up = m3.CrossSection(_contours(raised), m3.FillRule.EvenOdd).extrude(cfg.line_height_mm)
    up = up.translate((0, 0, cfg.base_mm))
    combined = plate + up
    bx0, by0, bx1, by1 = body.bounds
    size = (bx1 - bx0, by1 - by0, cfg.base_mm + cfg.line_height_mm)
    # coverage as printed: each edge moves out by spread/2 (area + boundary length * spread/2)
    sgn = -1 if gaps else 1                        # gaps: the fill spreads into the slots
    cover = (min(1.0, max(0.0, region.area + sgn * region.boundary.length * cfg.line_spread_mm / 2) / inner.area)
             if inner.area else 0.0)
    info = {
        "mode": cfg.mode, "fill": cfg.fill, "shape": shape, "size_mm": [round(v, 1) for v in size],
        "swap_at_mm": round(cfg.base_mm, 2),
        "line_cover_pct": round(100 * cover, 1),
        "model_width_mm": round(model_w, 3), "printed_width_mm": round(printed_w, 3),
        "plate_g": round(plate.volume() / 1000 * PLA_G_PER_CM3, 1),
        "lines_g": round(up.volume() / 1000 * PLA_G_PER_CM3, 1),
        "triangles": combined.num_tri(),
        "fits_bed": max(size[0], size[1]) <= cfg.max_size_mm,
        "drawing_mm": [round(W, 1), round(H, 1)],
        **info_open,
    }
    return {"plate": plate, "raised": up, "combined": combined, "body2d": body, "raised2d": raised,
            "extra2d": extra}, info


# ---------------------------------------------------------------------------- files

def _mesh_arrays(man):
    mesh = man.to_mesh()
    return np.asarray(mesh.vert_properties, np.float32)[:, :3], np.asarray(mesh.tri_verts, np.int64)


def stl_bytes(man, name="lineart"):
    v, t = _mesh_arrays(man)
    tri = v[t]                                                     # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    head = name.encode()[:80].ljust(80, b" ")
    return head + struct.pack("<I", len(t)) + rec.tobytes()


def threemf_bytes(parts):
    """parts: [(name, manifold, '#RRGGBB')] -> a 3MF with one object per part (slicers
    offer to load them as one multi-part object; assign a filament per part)."""
    objs, items = [], []
    mats = "".join(f'<base name="{n}" displaycolor="{c}"/>' for n, _, c in parts)
    for i, (name, man, _) in enumerate(parts, start=2):
        v, t = _mesh_arrays(man)
        vs = "".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>' for a, b, c in v)
        ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in t)
        objs.append(f'<object id="{i}" name="{name}" type="model" pid="1" pindex="{i - 2}"><mesh>'
                    f'<vertices>{vs}</vertices><triangles>{ts}</triangles></mesh></object>')
        items.append(f'<item objectid="{i}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources><basematerials id="1">{mats}</basematerials>{"".join(objs)}</resources>'
             f'<build>{"".join(items)}</build></model>')
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
          '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def _raster(geom, bounds, px_mm, shape):
    """Fill a shapely (Multi)Polygon into a bool image (y down). Polygons are drawn largest
    first, each with its holes cleared, so islands inside holes survive."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = bounds
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    f = lambda ring: [((x - x0) / px_mm, (y1 - y) / px_mm) for x, y in ring.coords]
    polys = list(geom.geoms) if geom.geom_type in ("MultiPolygon", "GeometryCollection") else [geom]
    from shapely.geometry import Polygon
    for poly in sorted((q for q in polys if q.geom_type == "Polygon"), key=lambda q: -Polygon(q.exterior).area):
        d.polygon(f(poly.exterior), fill=255)
        for r in poly.interiors:
            d.polygon(f(r), fill=0)
    return np.asarray(im) > 127


def thickness_map(parts, px_mm=0.1):
    """Top view (y down): bool masks of the base, the raised part and the bridges/web."""
    x0, y0, x1, y1 = parts["body2d"].bounds
    shape = (int(math.ceil((y1 - y0) / px_mm)) + 1, int(math.ceil((x1 - x0) / px_mm)) + 1)
    b = (x0, y0, x1, y1)
    extra = parts.get("extra2d")
    ex = _raster(extra, b, px_mm, shape) if extra is not None and not extra.is_empty else np.zeros(shape, bool)
    return _raster(parts["body2d"], b, px_mm, shape), _raster(parts["raised2d"], b, px_mm, shape), ex


LIGHT, DARK = (247, 247, 244), (28, 28, 30)


def colours(cfg):
    """(base filament, top filament, line colour, background colour) as RGB. The lines are
    whatever shows the line colour: the top layers (fill=lines) or the base seen through the
    empty lines (fill=gaps)."""
    line_c, bg_c = (LIGHT, DARK) if cfg.light_lines else (DARK, LIGHT)
    base_c, top_c = (line_c, bg_c) if cfg.fill == "gaps" else (bg_c, line_c)
    return base_c, top_c, line_c, bg_c


def preview_views(cfg):
    """[(key, kwargs for preview_png, caption)] that make sense for this config."""
    base_c, top_c, line_c, bg_c = colours(cfg)
    name = lambda c: "white" if c == LIGHT else "dark"
    if cfg.mode == "open":
        wall = name(bg_c) if cfg.fill == "lines" else name(line_c)
        what = (f"{name(line_c)} lines on {name(bg_c)}" if cfg.fill == "lines" else
                f"{name(bg_c)} fill, the lines are slots: the {wall} wall shows through")
        return [("wall", dict(view="wall"), f"On a {wall} wall: {what}"),
                ("base", dict(view="base"), f"Base layers ({name(base_c)}): bridges{' and web' if cfg.web else ''} in red")]
    out = [("front", dict(view="front"), f"Front-lit: {name(line_c)} lines on {name(bg_c)}"
            + (" (grooves showing the base)" if cfg.fill == "gaps" else ""))]
    if base_c == LIGHT:
        out.append(("backlit", dict(view="backlit"), "Backlit" + (": the thin grooves glow, so the lines show bright"
                                                                    if cfg.fill == "gaps" else "")))
    else:
        out.append(("backlit", dict(view="backlit", single=True),
                    "Backlit, printed all in white (a dark base would block the light)"))
    return out


def preview_png(parts, cfg, px_mm=0.1, max_side=1400, view="front", single=False, **_):
    """view: front / wall (front-lit, on a wall), base (base layers, bridges/web in red),
    backlit (light through white PLA falls off ~exp(-thickness / 0.8 mm); dark filament
    blocks it; single = everything printed in white)."""
    from PIL import Image
    plate, raised, extra = thickness_map(parts, px_mm)
    r = cfg.line_spread_mm / 2 / px_mm
    if r >= 0.5:                                  # show the top at its printed (spread) width
        from scipy.ndimage import binary_dilation
        k = int(math.ceil(r))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        raised = binary_dilation(raised, structure=xx * xx + yy * yy <= r * r) & plate
    base_c, top_c, line_c, bg_c = colours(cfg)
    rgb = np.empty(plate.shape + (3,), np.uint8)
    if view == "base":
        rgb[:] = (226, 224, 219) if base_c == DARK else (40, 40, 46)
        rgb[plate] = base_c
        rgb[extra & plate] = (220, 40, 40)
    elif view == "backlit":
        t = plate * cfg.base_mm + raised * cfg.line_height_mm
        light = np.exp(-t / 0.8) / np.exp(-cfg.base_mm / 0.8)
        if not single:
            if top_c == DARK:
                light = np.where(raised, 0.03, light)
            if base_c == DARK:
                light = np.where(plate, 0.03, light)
        rgb[:] = (np.clip(light, 0, 1)[..., None] * 255).astype(np.uint8)
        rgb[~(plate | raised)] = (40, 40, 46)                         # wall behind
    else:
        wall = (bg_c if cfg.fill == "lines" else line_c) if cfg.mode == "open" else None
        rgb[:] = (226, 224, 219) if wall == LIGHT else (52, 52, 56) if wall == DARK else (128, 128, 132)
        rgb[plate] = base_c
        rgb[raised] = top_c
    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
