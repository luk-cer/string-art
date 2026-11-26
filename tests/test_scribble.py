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


def test_ink_mode_hatches_less_where_main_lines_already_draw(tmp_path):
    import numpy as np
    from plotter.scribble import preset, Scribble, load_gray
    p = _img(tmp_path)
    res = {}
    for mode in ("clean", "ink"):
        cfg = preset("app5", max_lines=300, workers=1, hatch_mode=mode)
        sc = Scribble(load_gray(p, cfg), cfg)
        sc.run()
        res[mode] = sc
    ink = res["ink"]
    assert ink.hatch_target.max() <= 1 and ink.hatch_target.min() >= 0
    # main lines are identical; only the hatching differs
    assert res["clean"].lines[:res["clean"].n_main] == ink.lines[:ink.n_main]
    # the hatch target is reduced where main-line ink already is
    busy = ink.main_ink > 0.3
    if busy.any():
        assert ink.hatch_target[busy].mean() < ink.ipix[busy].mean()


def test_freehand_mode_runs_and_is_smooth(tmp_path):
    import numpy as np
    from plotter.scribble import preset, Scribble, load_gray
    p = _img(tmp_path)
    cfg = preset("app5", max_lines=200, workers=1, hatch_mode="freehand", freehand_density=0.5)
    sc = Scribble(load_gray(p, cfg), cfg)
    sc.run()
    strokes = sc.lines[sc.n_main:]
    assert strokes
    # momentum: consecutive pen steps never turn sharply (no jagged corners)
    for l in strokes[:50]:
        a = np.diff(np.asarray(l, float), axis=0)
        a = a[np.hypot(*a.T) > 1e-6]
        if len(a) > 2:
            cosang = (a[1:] * a[:-1]).sum(1) / (np.hypot(*a[1:].T) * np.hypot(*a[:-1].T))
            assert np.percentile(cosang, 5) > 0.3


def test_freehand_does_not_fill_enclosed_pockets_solid():
    import numpy as np
    from scipy import ndimage
    from plotter.freehand import FreehandConfig, freehand
    # light-grey field with a small walled pocket in the middle; target is uniform
    H = W = 120
    target = np.full((H, W), 0.35)
    walls = np.zeros((H, W), bool)
    walls[50:70, 50] = walls[50:70, 70] = walls[50, 50:71] = walls[70, 50:71] = True
    strokes, _ = freehand(target, np.zeros((H, W)), FreehandConfig(pen_px=1.0, seed=1), walls=walls)
    cov = np.zeros((H, W))
    for s in strokes:
        a = np.asarray(s).astype(int)
        cov[a[:, 0], a[:, 1]] = 1
    inside = cov[52:68, 52:68].mean()
    outside = cov[10:40, 10:40].mean()
    assert inside < 2.5 * outside + 0.05      # pocket not drawn much denser than the open field


def test_scribble_circle_mask_clips_everything(tmp_path):
    import numpy as np
    from plotter.scribble import preset, Scribble, load_gray
    p = _img(tmp_path)
    cfg = preset("app5", max_lines=200, workers=1, hatch_mode="clean", shape="circle", frame=True)
    sc = Scribble(load_gray(p, cfg), cfg)
    sc.run()
    cy, cx, R = sc.circle
    pts = np.array([q for l in sc.lines for q in l], float)
    assert np.hypot(pts[:, 0] - cy, pts[:, 1] - cx).max() <= R + 1e-6


def test_stringart_min_lines_forces_more_lines():
    import numpy as np
    from plotter import stringart as sa
    size = 50
    target = np.full((size, size), 0.05)          # very light: solver would stop almost at once
    cfg = sa.StringArtConfig(nails=60, min_skip=5, max_lines=500, pixel_mm=2.0, diameter_mm=100)
    free, _, _ = sa.solve(target, np.ones_like(target), cfg)
    cfg.min_lines = 200
    forced, _, _ = sa.solve(target, np.ones_like(target), cfg)
    n = lambda st: sum(len(s) - 1 for s in st)
    assert n(free) < 200 <= n(forced)


def test_stringart_arcs_stay_inside_and_dashes_taper():
    import math
    import numpy as np
    from plotter import stringart as sa
    for d in (0.3, 1.0, 2.5):
        p, q = (math.cos(-d / 2), math.sin(-d / 2)), (math.cos(d / 2), math.sin(d / 2))
        for b in (1.0, 0.5, -1.0):
            pts = sa.arc_points(p, q, (0, 0), 1.0, b, 0.01)
            assert np.hypot(*pts.T).max() <= 1.0 + 1e-9
    cfg = sa.StringArtConfig(dashed=True, gap_tone=0.2, taper=0.4)
    t = np.tile(np.linspace(0, 1, 50), (50, 1))
    duty = sa.drawable_mask(t, cfg)[25]
    assert duty[0] == 0 and duty[-1] == 1 and np.all(np.diff(duty) >= -1e-6)   # fades in monotonically
