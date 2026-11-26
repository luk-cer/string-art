import numpy as np
from PIL import Image

from plotter import hatch as hz
from plotter.hatch import merge_strokes


def _radial(tmp_path):
    yy, xx = np.mgrid[:120, :120]
    img = (255 * np.clip(np.hypot(xx - 60, yy - 60) / 60, 0, 1)).astype(np.uint8)
    p = tmp_path / "g.png"
    Image.fromarray(img).save(p)
    return p


def test_merge_strokes_joins_aligned_ends():
    a = [(0, 0), (10, 0)]
    b = [(12, 0), (20, 0)]            # continues a, 2 mm gap
    c = [(30, 0), (30, 10)]           # far away
    d = [(22, 5), (22, 15)]           # near b's end but at 90 degrees -> not joined
    out = [list(map(tuple, s)) for s in merge_strokes([a, b, c, d], max_gap=3, max_angle_deg=40)]
    assert len(out) == 3
    joined = [(0, 0), (10, 0), (12, 0), (20, 0)]
    assert joined in out or joined[::-1] in out


def test_hatch_options_run(tmp_path):
    p = _radial(tmp_path)
    base = dict(diameter_mm=60, cell_mm=0.5, crop=(0, 0, 120, 120))
    plain, _, _ = hz.flow_hatch(p, hz.HatchConfig(**base), progress=None)
    cov, _, _ = hz.flow_hatch(p, hz.HatchConfig(tone_model="coverage", **base), progress=None)
    merged, _, _ = hz.flow_hatch(p, hz.HatchConfig(merge_gap_mm=4, **base), progress=None)
    outl, _, _ = hz.flow_hatch(p, hz.HatchConfig(outline_levels=(0.5,), outline_min_len_mm=5, **base),
                               progress=None)
    assert sum(map(len, merged)) <= sum(map(len, plain))
    assert len(outl) == len(plain) + 1 and len(outl[-1]) > 0
    assert sum(map(len, cov)) > 0


def test_default_is_deterministic(tmp_path):
    p = _radial(tmp_path)
    cfg = hz.HatchConfig(diameter_mm=60, cell_mm=0.5, crop=(0, 0, 120, 120))
    a, _, _ = hz.flow_hatch(p, cfg, progress=None)
    b, _, _ = hz.flow_hatch(p, cfg, progress=None)
    assert a == b
