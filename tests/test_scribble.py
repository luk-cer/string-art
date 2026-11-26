import numpy as np
from PIL import Image

from plotter.scribble import ScribbleConfig, scribble
from plotter.gcode import order_polylines, _order_kdtree


def _img(tmp_path):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:140, :110]
    a = 0.5 + 0.4 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    a[40:90, 30:70] *= 0.3                         # a dark block with edges
    a += 0.03 * rng.standard_normal(a.shape)
    p = tmp_path / "t.png"
    Image.fromarray((255 * np.clip(a, 0, 1)).astype(np.uint8)).save(p)
    return p


def test_fast_sampler_and_blur_match_exact(tmp_path):
    p = _img(tmp_path)
    exact, _, _ = scribble(p, ScribbleConfig(sampler="exact", separable_blur=False, workers=1, max_lines=400))
    fast, _, _ = scribble(p, ScribbleConfig(sampler="fast", separable_blur=True, workers=1, max_lines=400))
    assert len(exact) > 50
    # same distribution and same random stream: identical in practice
    assert exact == fast


def test_scribble_deterministic(tmp_path):
    p = _img(tmp_path)
    a, _, _ = scribble(p, ScribbleConfig(max_lines=300, workers=1))
    b, _, _ = scribble(p, ScribbleConfig(max_lines=300, workers=1))
    assert a == b


def test_kdtree_order_matches_bruteforce():
    rng = np.random.default_rng(0)
    pls = [[tuple(rng.random(2) * 200), tuple(rng.random(2) * 200)] for _ in range(1500)]
    brute = order_polylines(pls)               # n <= 2000 -> brute force path
    kd = _order_kdtree([list(p) for p in pls], (0.0, 0.0))
    assert [tuple(map(tuple, p)) for p in brute] == [tuple(map(tuple, p)) for p in kd]


def test_app_presets_run_and_keep_highlights_white(tmp_path):
    from plotter.scribble import preset
    p = _img(tmp_path)
    out = {}
    for name in ("c_v25", "app5", "app7"):
        lines, shape, sc = scribble(p, preset(name, max_lines=300, workers=1))
        assert lines
        out[name] = sc
    # app presets zero the lightest 20% of the tonal range -> no ink there
    assert (out["app5"].ipix == 0).sum() > (out["c_v25"].ipix == 0).sum()


def test_clean_hatching_and_momentum(tmp_path):
    import numpy as np
    from plotter.scribble import preset, Scribble, load_gray
    p = _img(tmp_path)
    for kw in (dict(hatch_mode="clean"), dict(hatch_mode="clean", main_momentum=0.5, smooth_main=3)):
        cfg = preset("app5", max_lines=300, workers=1, **kw)
        sc = Scribble(load_gray(p, cfg), cfg)
        sc.run()
        hatch = sc.lines[sc.n_main:]
        assert hatch
        for l in hatch:                       # no U-turns: consecutive steps never reverse
            a = np.diff(np.asarray(l, float), axis=0)
            if len(a) > 1:
                dots = (a[1:] * a[:-1]).sum(1)
                assert (dots > 0).all()
