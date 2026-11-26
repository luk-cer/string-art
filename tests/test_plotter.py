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
