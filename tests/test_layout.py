import math

import pytest

from plotter.gcode import write_gcode, PlotterProfile
from plotter.layout import parse_gcode, normalise, place, compose


def test_roundtrip_own_gcode(tmp_path):
    pls = [[(10, 20), (30, 20), (30, 40)], [(50, 50), (60, 55)]]
    out = tmp_path / "a.gcode"
    write_gcode(pls, out, PlotterProfile(corner_tol_mm=0, hop_up="G0 Z2"), paper_height=100)
    got, info = parse_gcode(out.read_text())
    assert info["pen"] == "Z" and len(got) == 2
    # file is y-up (flipped around paper height 100)
    assert got[0][0] == pytest.approx((10, 80)) and got[0][-1] == pytest.approx((30, 60))
    local, w, h = normalise(got)
    assert (w, h) == pytest.approx((50, 35))
    assert local[0][0] == pytest.approx((0, 0))          # top-left of the drawing, y-down


def test_servo_pen_and_arcs():
    g = """G21
G90
M5
G0 X0 Y0
M3 S90
G2 X10 Y0 I5 J0
M5
G0 X20 Y0
M3
G1 X30 Y0
M5
"""
    pls, info = parse_gcode(g)
    assert info["pen"] == "M3/M5" and len(pls) == 2
    # clockwise half circle from (0,0) to (10,0) around (5,0) passes through (5, 5) (y-up)
    assert max(p[1] for p in pls[0]) == pytest.approx(5, abs=0.05)
    assert pls[1] == [(20.0, 0.0), (30.0, 0.0)]


def test_rapids_never_draw_and_g91_refused():
    pls, _ = parse_gcode("G0 Z5\nG0 X0 Y0\nG0 Z0\nG1 X5\nG0 X50\nG1 X60\nG0 Z5\n")
    assert [p[0][0] for p in pls] == [0, 50]
    with pytest.raises(ValueError):
        parse_gcode("G91\nG0 Z0\nG1 X1\n")


def test_place_rotation_and_compose():
    pls = [[(0, 0), (20, 0), (20, 10)]]                    # w 20, h 10
    r90 = place(pls, 20, 10, 5, 5, 2, 90)
    # clockwise: top-left (0,0) -> top-right of the 10x20 box; scaled x2, shifted by 5
    assert r90[0][0] == pytest.approx((25, 5))
    assert r90[0][1] == pytest.approx((25, 45))
    xs = [p[0] for p in r90[0]]; ys = [p[1] for p in r90[0]]
    assert (max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx((20, 40))
    out = compose([{"id": "a", "x": 0, "y": 0, "scale": 1, "rot": 0},
                   {"id": "a", "x": 100, "y": 0, "scale": 0.5, "rot": 180}], {"a": (pls, 20, 10)})
    assert len(out) == 2 and out[1][0] == pytest.approx((110, 5))
    with pytest.raises(KeyError):
        compose([{"id": "zz"}], {})


def test_layout_endpoints(tmp_path):
    from plotter.ui import App
    app = App(tmp_path)
    jdir = tmp_path / "jobs" / ("a" * 32)
    jdir.mkdir(parents=True)
    write_gcode([[(10, 10), (40, 10), (40, 30)]], jdir / "plot.gcode", PlotterProfile(), paper_height=100)
    (jdir / "plot.json").write_text('{"style": "hatch", "params": {}, "stats": {"est_minutes": 1}}')
    assert app.jobs()[0]["job"] == "a" * 32
    src = app.layout_load({"job": "a" * 32})
    assert src["w"] == pytest.approx(30, abs=0.1) and src["h"] == pytest.approx(20, abs=0.1)
    up = app.layout_load({"gcode": (jdir / "plot.gcode").read_text(), "name": "x.gcode"})
    r = app.layout_export({"paper": {"w": 210, "h": 297}, "profile": {"dwell_s": 0},
                           "items": [{"id": src["id"], "x": 10, "y": 10, "scale": 2, "rot": 0},
                                     {"id": up["id"], "x": 100, "y": 200, "scale": 1, "rot": 90}]})
    assert math.isclose(r["stats"]["draw_mm"], 2 * 50 + 50, rel_tol=0.01)   # source corner was rounded
    assert app.job_file(r["job"], "plot.gcode").exists()
    # off the paper -> refused
    with pytest.raises(ValueError, match="paper"):
        app.layout_export({"paper": {"w": 210, "h": 297},
                           "items": [{"id": src["id"], "x": 200, "y": 10, "scale": 1, "rot": 0}]})
