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
