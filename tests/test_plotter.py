import math
import numpy as np

from plotter.gcode import write_gcode, order_polylines, PlotterProfile
from plotter import stringart as sa


def test_gcode_dialect_and_flip(tmp_path):
    out = tmp_path / "a.gcode"
    stats = write_gcode([[(10, 20), (30, 20)], [(30, 20), (30, 40)]], out, paper_height=100)
    lines = out.read_text().splitlines()
    assert lines[:4] == ["G21", "G90", "G0 Z5", "G4 P0.05"]
    assert "G0 X10.000 Y80.000" in lines          # y flipped
    assert lines.count("G0 Z0") == 1              # joined polylines: one pen-down
    assert math.isclose(stats["draw_mm"], 40)


def test_order_polylines_reverses():
    ordered = order_polylines([[(100, 0), (50, 0)], [(10, 0), (0, 0)]])
    assert ordered[0] == [(10, 0), (0, 0)][::-1]


def test_ink_saturates_and_stops():
    cfg = sa.StringArtConfig(nails=60, min_skip=5, max_lines=5000, pixel_mm=2.0, diameter_mm=100)
    size = 50
    target = np.full((size, size), 0.3)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    target *= (xx - c) ** 2 + (yy - c) ** 2 <= (size / 2) ** 2
    strokes, cov, _ = sa.solve(target, np.ones_like(target), cfg)
    n = sum(len(s) - 1 for s in strokes)
    assert 0 < n < 5000                          # stopped on its own
    assert cov.max() < 0.95                      # no solid black blob


def test_ink_cap_limits_overdraw_and_keeps_fixed():
    from plotter.gcode import limit_ink, ink_load
    # 40 identical 20 mm strokes on one spot = 40 covers with a 0.5 mm pen on a 0.5 mm grid
    pile = [[(0.0, 0.0), (20.0, 0.0)] for _ in range(40)]
    kept, st = limit_ink(pile, 0.5, 3.0, cell_mm=0.5)
    load, _ = ink_load(kept, 0.5, 0.5)
    assert load.max() <= 3.0 * 1.2 and 80 < st["ink_removed_pct"] < 98
    # a single stroke far from everything is untouched
    kept, st = limit_ink([[(0.0, 50.0), (20.0, 50.0)]], 0.5, 3.0)
    assert st["ink_removed_pct"] == 0 and len(kept) == 1
    # fixed strokes count but are never returned or cut: with 3 covers already there, nothing fits
    kept, st = limit_ink([[(0.0, 0.0), (20.0, 0.0)]], 0.5, 2.0, cell_mm=0.5, fixed=pile[:3])
    assert kept == [] and st["ink_removed_pct"] == 100


def test_compose_item_lands_in_its_frame_and_tone_thins(tmp_path):
    from PIL import Image
    from plotter.ui import App
    yy, xx = np.mgrid[:300, :300]
    img = (255 * (0.2 + 0.6 * (xx / 300))).astype(np.uint8)          # gradient, dark on the left
    Image.fromarray(img).save(tmp_path / "g.png")
    app = App(tmp_path / "work")
    up = app.upload((tmp_path / "g.png").read_bytes())
    lengths = []
    for tone in (1.0, 0.5):
        r = app.compose_item({"image": up["id"], "style": "hatch", "shape": "circle",
                              "frame": {"x": 30, "y": 50, "w": 60, "h": 60}, "crop": [0, 0, 300, 300],
                              "pen_mm": 0.3, "tone": tone, "params": {}})
        p = np.array([c for pl in r["polylines"] for c in pl]).reshape(-1, 2)
        assert p[:, 0].min() >= 29.9 and p[:, 0].max() <= 90.1 and p[:, 1].min() >= 49.9 and p[:, 1].max() <= 110.1
        lengths.append(r["stats"]["draw_m"])
    assert lengths[1] < 0.8 * lengths[0]                                   # lighter = fewer lines
    out = app.compose_export({"paper": {"w": 210, "h": 297}, "pen_mm": 0.3, "profile": {},
                              "parts": [{"key": r["key"], "dx": 10, "dy": 0}]})
    assert out["stats"]["draw_mm"] > 0


def test_scribble_size_and_dpi_set_working_resolution(tmp_path):
    from PIL import Image
    from plotter import scribble as scr
    Image.new("L", (1000, 500), 128).save(tmp_path / "p.png")
    cfg = scr.ScribbleConfig(size_mm=127.0, dpi=100.0)
    pix = scr.load_gray(str(tmp_path / "p.png"), cfg)
    assert pix.shape == (250, 500)                       # 127 mm at 100 dpi = 500 px on the long side
    assert abs(scr.paper_fit(*pix.shape, cfg)[0] - 0.254) < 1e-9
    small = scr.load_gray(str(tmp_path / "p.png"), scr.ScribbleConfig(size_mm=63.5, dpi=100.0))
    assert small.shape == (125, 250)                     # half the size -> half the pixels, same mm/px


def test_importance_mask_lines_up_and_weights(tmp_path):
    from PIL import Image
    from plotter.importance import importance_map
    photo = Image.new("L", (400, 300), 128)
    photo.save(tmp_path / "p.png")
    mask = Image.new("L", (400, 300), 0)
    mask.paste(255, (250, 100, 300, 150))                 # painted square, source pixels
    mask.save(tmp_path / "m.png")
    m, info = importance_map(str(tmp_path / "p.png"), (100, 0, 300, 300), 60, face=False,
                             mask_path=str(tmp_path / "m.png"))
    assert info["painted"] and m.shape == (60, 60)
    # square (250..300, 100..150) inside crop x 100..400 -> grid (150..200)/5 = 30..40, y 20..30
    assert m[25, 35] > 0.8 and m[25, 10] < 0.05 and m[50, 35] < 0.05

    cfg = sa.StringArtConfig(importance=3.0, face_detect=False, importance_mask=str(tmp_path / "m.png"),
                             crop=(100, 0, 300, 300), edge_boost=0)
    _, w, _, _ = sa.prepare_target(str(tmp_path / "p.png"), cfg, 60)
    assert w[25, 35] > 3.0 and abs(w[25, 10] - 1.0) < 0.1


def test_refine_lowers_error_and_chains_all_lines():
    from dataclasses import replace
    size = 50
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    inside = (xx - c) ** 2 + (yy - c) ** 2 <= (size / 2) ** 2
    target = (0.15 + 0.5 * (xx > c) + 0.2 * (np.abs(yy - c) < 6)) * inside   # two tones + a dark band
    w = np.ones_like(target)
    cfg = sa.StringArtConfig(nails=80, min_skip=8, pixel_mm=2.0, diameter_mm=100, pen_mm=0.3,
                             chords="warp", refine=0)

    def err(cov):
        r = target - cov
        return float(np.sum(np.where(r > 0, r * r, cfg.overshoot * r * r)))

    s0, cov0, _ = sa.solve(target, w, cfg)
    s1, cov1, _ = sa.solve(target, w, replace(cfg, refine=5))
    assert err(cov1) < err(cov0)

    # re-chained strokes contain each line exactly once and are continuous nail sequences
    lines = [(min(a, b), max(a, b), bi) for s in s1 for a, b, bi in zip(s, s[1:], s.bends)]
    assert len(lines) == len(set(lines))
    # the coverage the solver reports matches redrawing those lines
    table = sa.LineTable(cfg.nails, size, cfg.pixel_mm, cfg.pen_mm, sa.chord_bends(cfg))
    cov = np.zeros(size * size)
    for i, j, bi in lines:
        idx = table.line_cells(i, j, bi)
        cov[idx] = 1 - (1 - cov[idx]) * (1 - min(table.alpha[i, j, bi], 0.999))
    assert np.allclose(cov.reshape(size, size), cov1, atol=1e-6)


def test_flow_hatch_smoke(tmp_path):
    from PIL import Image
    from plotter import hatch as hz
    yy, xx = np.mgrid[:120, :120]
    img = (255 * np.clip(np.hypot(xx - 60, yy - 60) / 60, 0, 1)).astype(np.uint8)
    p = tmp_path / "g.png"
    Image.fromarray(img).save(p)
    cfg = hz.HatchConfig(diameter_mm=60, cell_mm=0.5, crop=(0, 0, 120, 120))
    layers, _, _ = hz.flow_hatch(p, cfg, progress=None)
    assert len(layers) == 4
    assert len(layers[0]) > len(layers[3]) > 0   # darker layers cover less area


def test_ui_backend_render(tmp_path):
    import io
    from PIL import Image
    from plotter.ui import App
    yy, xx = np.mgrid[:200, :200]
    img = (255 * np.clip(np.hypot(xx - 100, yy - 100) / 100, 0, 1)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
    app = App(tmp_path)
    up = app.upload(buf.getvalue())
    for style, params in [("hatch", {"layers": 2, "diameter_mm": 60}),
                          ("stringart", {"nails": 60, "max_lines": 200, "diameter_mm": 60, "pixel_mm": 2})]:
        r = app.render({"image": up["id"], "crop": up["crop"], "style": style, "params": params,
                        "profile": {"draw_feed": 2000, "header": ["G21", "G90"]}})
        assert r["polylines"] and r["stats"]["draw_mm"] > 0
        g = app.job_file(r["job"], "plot.gcode").read_text()
        assert "F2000" in g


def test_gcode_modal_feed_and_corner_rounding(tmp_path):
    from plotter.gcode import round_corners
    zig = [(i * 1.0, (i % 2) * 1.0) for i in range(20)]          # 90 deg corners
    r = round_corners(zig, 0.05)
    a, q = np.asarray(zig), np.asarray(r)
    # every rounded point stays within tol of the original path
    A, B = a[:-1], a[1:]
    AB = B - A
    t = np.clip(((q[:, None] - A) * AB).sum(-1) / (AB * AB).sum(-1), 0, 1)
    d = np.hypot(*(q[:, None] - (A + t[..., None] * AB)).transpose(2, 0, 1)).min(1)
    assert d.max() <= 0.05 + 1e-9
    assert r[0] == zig[0] and r[-1] == zig[-1]
    out = tmp_path / "z.gcode"
    st_round = write_gcode([zig], out, PlotterProfile(flip_y=False, draw_feed=6000))
    g1 = [l for l in out.read_text().splitlines() if l.startswith("G1")]
    assert sum("F" in l for l in g1) == 1                          # F written once (modal)
    st_sharp = write_gcode([zig], out, PlotterProfile(flip_y=False, draw_feed=6000, corner_tol_mm=0))
    assert st_round["draw_minutes"] < st_sharp["draw_minutes"]     # less braking at corners
    assert st_sharp["draw_minutes"] > st_sharp["draw_ideal_minutes"]


def test_time_estimate_straight_line():
    from plotter.gcode import estimate_time
    p = PlotterProfile(accel_mm_s2=1000)
    # 100 mm at 60 mm/s, accel 1000: 100/60 + 60/1000
    t = estimate_time([("xy", 100.0, 0.0, 3600.0)], p)
    assert math.isclose(t["total"], 100 / 60 + 60 / 1000, rel_tol=1e-6)


def test_bounds_check_refuses_and_writes_nothing(tmp_path):
    import pytest
    from plotter.gcode import OutOfBounds
    out = tmp_path / "b.gcode"
    # inside: fine
    write_gcode([[(10, 10), (200, 10)]], out, paper_height=210, paper_width=297)
    out.unlink()
    # machine is 320 wide: an A3-wide stroke at x=400 is refused
    with pytest.raises(OutOfBounds, match="machine area"):
        write_gcode([[(10, 10), (400, 10)]], out, paper_height=297, paper_width=420)
    assert not out.exists()
    # offset pushes it negative
    with pytest.raises(OutOfBounds):
        write_gcode([[(10, 10), (20, 10)]], out, PlotterProfile(offset_x=-50), paper_height=210)
    # off the paper
    with pytest.raises(OutOfBounds, match="paper"):
        write_gcode([[(10, 10), (300, 10)]], out, paper_height=210, paper_width=297)
    # footer moves are checked too
    with pytest.raises(OutOfBounds):
        write_gcode([[(10, 10), (20, 10)]], out, PlotterProfile(footer=["G0 X500 Y0"]), paper_height=210)
    # check can be switched off (paper check still applies only when paper_width is given)
    write_gcode([[(10, 10), (400, 10)]], out, PlotterProfile(check_bounds=False), paper_height=297)


def test_short_hop_for_short_travel(tmp_path):
    out = tmp_path / "h.gcode"
    pls = [[(10, 10), (20, 10)], [(22, 10), (30, 10)], [(100, 10), (110, 10)]]
    st = write_gcode(pls, out, PlotterProfile(hop_up="G0 Z2", hop_max_mm=10, dwell_s=0), paper_height=210)
    lines = out.read_text().splitlines()
    assert lines.count("G0 Z2") == 1            # 2 mm gap: hop
    assert lines.count("G0 Z5") == 3            # start, 70 mm travel, end
    assert st["pen_lifts"] == 4
    full = write_gcode(pls, out, PlotterProfile(dwell_s=0), paper_height=210)
    assert st["pen_lift_minutes"] < full["pen_lift_minutes"]
