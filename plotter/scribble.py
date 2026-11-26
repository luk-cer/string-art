"""
Scribble / gradient-tracing line art - a fast port of scribble-code/c_v25.py
(SketchConverter), keeping its look.

Algorithm (unchanged from c_v25):
1. Sobel gradients of the photo; magnitude below `gradient_threshold` dropped.
2. Main lines: pick a start pixel with probability ~ gradient magnitude, walk
   along the iso-brightness direction (gradient rotated 90 / 270 deg) while
   the image ahead is dark enough, then erase magnitude around the line
   (adaptive radius: small in dark/textured areas, large in light ones).
3. Cross-hatching: from points along the main lines, trace extra lines at
   fixed angles to a heavily smoothed gradient field, each angle only where
   the image is darker than its threshold.

What changed is only *how* it is computed:
* the 9x9 direction weights in `_direction_intensity` are built once instead
  of 324k times (meshgrid + norm per step was ~50% of the runtime);
* image patches are plain slices (wrap-around only at the border);
* the "line revisits a pixel" check is incremental instead of rebuilding a
  set of the whole line every step (was quadratic);
* cleanup stamps precomputed disks for all points of a line at once;
* start-point sampling: `exact` replays np.random.choice bit-for-bit;
  `fast` (default) samples the same distribution from block sums that are
  updated incrementally instead of renormalising the whole image per line;
* hatching uses a separable Gaussian (`fast`) instead of an 81x81 2-D one,
  and hatch traces (independent of each other) can run in parallel.

With sampler="exact", separable_blur=False and workers=1 the output is
identical, line for line, to c_v25 for the same np.random seed.
"""

from dataclasses import dataclass
import math
import os

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel, convolve, convolve1d


@dataclass
class ScribbleConfig:
    gradient_threshold: float = 0.05
    trace_kernel_size: int = 4          # d: patch radius for direction intensity, gradient blur size
    trace_kernel_sigma: float = 2.0
    max_lines: int = 10000
    min_magnitude_ratio: float = 1e-10
    light_percentile: float = 1.0
    cleanup_radius: int = 4
    remove_noise: bool = False          # non-local-means denoise (slow on big images)
    # cross-hatching
    hatch: bool = True
    hatch_min_line_length: int = 3
    hatch_sample_interval: int = 10
    hatch_max_length: int = 150
    hatch_angles: tuple = ((210, 0.5), (150, 0.7), (240, 0.85), (120, 0.95))
    hatch_adaptive_spacing: bool = True
    # variant knobs - see PRESETS (c_v25 defaults; app5 / app7 reproduce those scripts)
    white_cut: float = 0.0              # darkness below min + cut*range set to 0: light areas stay white
    cleanup_mode: str = "adaptive"      # adaptive (c_v25 circles) | quadrant (app5/7 boxes sized by darkness)
    main_max_steps: int = 1000
    split_main: bool = False            # keep forward/backward halves as separate lines (app5/7)
    main_min_points: int = 2            # drop main lines shorter than this (app5: 1)
    main_order: str = "none"            # none | sort (app5) | sort_merge (app7)
    legacy_gradient_sum: bool = False   # app5/7 compute sum(g * kernel) (~= g); only for bit-exact replay
    hatch_source: str = "all"           # all (c_v25) | longest (app5: 1000 longest) | merged (app7)
    hatch_keep_min: int = 4             # keep hatch lines with at least this many points (app5/7: 1)
    hatch_patch: int | None = None      # look-ahead patch radius for hatching (None = trace_kernel_size; app5/7: 5)
    # visual improvements (off by default = original look)
    hatch_mode: str = "legacy"          # legacy (original tracer) | clean (orientation field, spacing, no spirals)
                                        # | ink (clean + only adds the darkness main lines haven't drawn)
    hatch_spacing_min_px: float = 1.5   # clean: hatch line spacing in the darkest areas
    hatch_spacing_max_px: float = 4.0   # clean: spacing where a hatch layer fades in
    hatch_flow_smooth_px: float = 6.0   # clean: smoothing of the hatch direction field (orig. sigma 6)
    hatch_max_turn_deg: float = 90.0    # clean: stop a hatch line after turning this much in total
    hatch_clean_min_len: float = 6.0    # clean: drop hatch strokes shorter than this (px)
    hatch_threshold_shift: float = -0.1 # clean: start each hatch layer this much earlier (darker areas)
    hatch_ink_aware: bool = False       # same as hatch_mode="ink" (kept for scripts)
    hatch_view_px: float = 2.0          # ink-aware: blur (px) at which ink and target are compared ("viewing distance")
    hatch_ink_passes: int = 2           # ink-aware: 2 = hatch, measure achieved coverage, correct the request, redo
    hatch_fill_px: float = 30.0         # clean: flat areas borrow hatch direction from this larger scale (0 = off)
    smooth_main: int = 0                # moving-average window for main lines (0 = off): softens staircase
    main_momentum: float = 0.0          # 0..0.95: blend each step with the previous direction (and never
                                        # reverse onto itself) -> smoother, longer main lines; 0 = original
    # post
    min_points: int = 1                 # drop lines with fewer points (c_v25 keeps >1 for main lines)
    # performance
    sampler: str = "fast"               # fast | exact
    separable_blur: bool = True         # False = c_v25's direct 81x81 convolution
    workers: int = 1                    # processes for hatching (0 = all cores)
    seed: int | None = 0
    max_side: int | None = None         # downscale input so the longer side is at most this (pixels)
    crop: tuple | None = None           # (x, y, w, h) in source pixels
    # paper / output
    paper: str = "A4"
    portrait: bool = False
    padding_mm: float = 20.0
    pen_mm: float = 0.3
    rotate: str = "auto"                # auto | none | 90  (auto: rotate when it fits the page better)
    simplify_px: float = 0.5            # RDP tolerance in source pixels (like c_v25 max_rmse=0.5)
    join_px: float = 1.5                # join line ends closer than this (fewer pen lifts); 0 = off
    min_points_out: int = 5             # drop paths shorter than this many traced points (c_v25: 5)


PRESETS = {
    # scribble-code/c_v25.py __main__ settings
    "c_v25": {},
    # scribble-code/app5.py: white cut, sparse quadrant cleanup, hatch from the 1000 longest lines
    "app5": dict(white_cut=0.2, cleanup_mode="quadrant", cleanup_radius=20, main_max_steps=10000,
                 min_magnitude_ratio=1e-3, split_main=True, main_min_points=1, main_order="sort",
                 legacy_gradient_sum=True, hatch_source="longest", hatch_min_line_length=10,
                 hatch_adaptive_spacing=False, hatch_sample_interval=10, hatch_max_length=10000,
                 hatch_angles=((225, 0.4), (245, 0.6), (205, 0.75), (265, 0.9)), hatch_keep_min=1,
                 hatch_patch=5),
    # scribble-code/app7.py: as app5 but merges main lines first and hatches from all of them
    "app7": dict(white_cut=0.2, cleanup_mode="quadrant", cleanup_radius=20, main_max_steps=10000,
                 min_magnitude_ratio=1e-3, split_main=True, main_min_points=2, main_order="sort_merge",
                 legacy_gradient_sum=True, hatch_source="merged", hatch_min_line_length=5,
                 hatch_adaptive_spacing=False, hatch_sample_interval=10, hatch_max_length=10000,
                 hatch_angles=((225, 0.4), (245, 0.6), (205, 0.75), (265, 0.9)), hatch_keep_min=1,
                 hatch_patch=5),
}


def preset(name, **overrides):
    """ScribbleConfig for a named preset (c_v25 | app5 | app7) with overrides."""
    return ScribbleConfig(**{**PRESETS[name], **overrides})


def merge_lines(lines):
    """Port of scribble-code/gcode_utils.merge_lines (Eulerian-style joining of
    lines that share exact end points), without the prints."""
    from collections import defaultdict
    graph = defaultdict(list)
    degree = defaultdict(int)
    for idx, line in enumerate(lines):
        p1, p2 = tuple(line[0]), tuple(line[-1])
        graph[p1].append({"to": p2, "id": idx, "data": line})
        graph[p2].append({"to": p1, "id": idx, "data": line[::-1]})
        degree[p1] += 1
        degree[p2] += 1
    used = set()
    merged = []

    def trace_path(start):
        path = []
        curr = start
        while True:
            for edge in graph[curr]:
                if edge["id"] not in used:
                    used.add(edge["id"])
                    path.extend(edge["data"] if not path else edge["data"][1:])
                    curr = edge["to"]
                    break
            else:
                break
        return path

    nodes = sorted(graph.keys(), key=lambda n: degree[n] % 2, reverse=True)
    for node in nodes:
        if degree[node] % 2 != 0:
            p = trace_path(node)
            if p:
                merged.append(p)
    for node in nodes:
        p = trace_path(node)
        if p:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# helpers (bit-compatible with c_v25)
# ---------------------------------------------------------------------------

def gaussian_kernel_2d(size, sigma):
    if size % 2 == 0:
        size += 1
    x = np.arange(-(size // 2), size // 2 + 1)
    y = np.arange(-(size // 2), size // 2 + 1)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(arr, size, sigma, separable):
    if not separable:
        return convolve(arr, gaussian_kernel_2d(size, sigma))
    if size % 2 == 0:
        size += 1
    x = np.arange(-(size // 2), size // 2 + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return convolve1d(convolve1d(arr, g, axis=0), g, axis=1)


def _rescale(X):
    X = X.copy()
    X -= X.min()
    if X.max() > 0:
        X /= X.max()
    return X


def _direction_grid(d):
    x = np.arange(-d, d + 1, dtype=np.float64)
    w = np.stack(np.meshgrid(x, x))
    wn = np.linalg.norm(w, axis=0)
    wn[wn == 0] = 1.0
    w /= wn
    return w[0].copy(), w[1].copy()


def _sign(v):
    return (v > 0) - (v < 0)


class _Tracer:
    """Gradient-following line tracer (c_v25 `_trace_line`, same arithmetic)."""

    def __init__(self, ipix, d):
        self.ipix = ipix
        self.H, self.W = ipix.shape
        self.d = d
        self.U0, self.U1 = _direction_grid(d)
        self._xr = np.arange(-d, d + 1)

    def region(self, cx, cy):
        d = self.d
        if d <= cx < self.H - d and d <= cy < self.W - d:
            return self.ipix[cx - d:cx + d + 1, cy - d:cy + d + 1]
        return self.ipix.take(self._xr + cx, mode="wrap", axis=0).take(self._xr + cy, mode="wrap", axis=1)

    def direction_intensity(self, region, gx, gy):
        w = self.U0 * gx + self.U1 * gy
        w[w < 0] = 0
        s = w.sum()
        if s > 0:
            w /= s
        return np.multiply(region, w).sum()

    def trace(self, px, py, angle, gx, gy, max_steps=1000, min_intensity=0.0, gkernel=None, momentum=0.0):
        rad = angle * np.pi / 180.0
        crad = np.cos(rad)
        srad = np.sin(rad)
        H, W = self.H, self.W
        line = [(int(px), int(py))]
        seen = {line[0]}
        unique = True
        step = 0
        pdx = pdy = None                         # previous unit direction (momentum only)
        while -1 < px < H - 1 and -1 < py < W - 1 and unique and step < max_steps:
            cx, cy = int(px), int(py)
            a, b = gx[cx, cy], gy[cx, cy]
            if gkernel is not None:              # app5/7: np.sum(g[x, y] * w2)
                a, b = np.sum(a * gkernel), np.sum(b * gkernel)
            g_x = a * crad + b * srad
            g_y = -a * srad + b * crad
            if momentum > 0 and (g_x != 0 or g_y != 0):
                n = math.hypot(g_x, g_y)
                ux, uy = g_x / n, g_y / n
                if pdx is not None:
                    if ux * pdx + uy * pdy < 0:      # same orientation, keep heading the same way
                        ux, uy = -ux, -uy
                    ux = momentum * pdx + (1 - momentum) * ux
                    uy = momentum * pdy + (1 - momentum) * uy
                    n = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / n, uy / n
                pdx, pdy = ux, uy
                g_x, g_y = ux, uy
            pint = self.direction_intensity(self.region(cx, cy), g_x, g_y)
            if pint <= min_intensity:
                return line
            if max(abs(g_x), abs(g_y)) == abs(g_x):
                if g_x == 0:
                    return line
                g_y = np.sign(g_y) * abs(g_y / g_x)
                g_x = np.sign(g_x)
            else:
                if g_y == 0:
                    return line
                g_x = np.sign(g_x) * abs(g_x / g_y)
                g_y = np.sign(g_y)
            px = px + g_x
            py = py + g_y
            if px < 0 or py < 0:
                return line
            p = (int(px), int(py))
            if p in seen:
                unique = False
            else:
                seen.add(p)
            line.append(p)
            step += 1
        return line


class _BlockSampler:
    """Sample an index with probability ~ weights, with cheap point updates."""

    def __init__(self, weights, block=1024):
        self.w = weights.ravel()            # view: updates to the image are seen here
        self.B = block
        n = self.w.size
        self.nb = (n + block - 1) // block
        pad = self.nb * block - n
        self.sums = np.add.reduceat(self.w, np.arange(0, n, block)) if n else np.zeros(0)
        self._pad = pad

    def refresh(self, flat_idx):
        blocks = np.unique(flat_idx // self.B)
        for b in blocks:
            self.sums[b] = self.w[b * self.B:(b + 1) * self.B].sum()

    def refresh_range(self, lo, hi):
        """Refresh blocks covering flat indices lo..hi (inclusive)."""
        for b in range(lo // self.B, hi // self.B + 1):
            self.sums[b] = self.w[b * self.B:(b + 1) * self.B].sum()

    def sample(self):
        cs = np.cumsum(self.sums)
        total = cs[-1]
        if total <= 0:
            return None
        u = np.random.random_sample() * total
        b = int(np.searchsorted(cs, u, side="right"))
        b = min(b, self.nb - 1)
        lo = b * self.B
        seg = self.w[lo:lo + self.B]
        within = u - (cs[b - 1] if b > 0 else 0.0)
        i = int(np.searchsorted(np.cumsum(seg), within, side="right"))
        i = min(i, len(seg) - 1)
        while seg[i] == 0 and i > 0:      # guard against float edge at block end
            i -= 1
        return lo + i


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def paper_fit(H, W, cfg):
    """Page layout for an H x W image: (mm_per_px, rotated, paper_w, paper_h)."""
    from radon_renderer.config import PAPER_SIZES
    pw, ph = PAPER_SIZES[cfg.paper]
    if cfg.portrait:
        pw, ph = min(pw, ph), max(pw, ph)
    aw, ah = pw - 2 * cfg.padding_mm, ph - 2 * cfg.padding_mm
    fit = lambda w, h: min(aw / w, ah / h)
    rot = cfg.rotate == "90" or (cfg.rotate == "auto" and fit(H, W) > fit(W, H) * 1.05)
    cw, ch = (H, W) if rot else (W, H)
    return fit(cw, ch), rot, pw, ph


def ink_coverage(lines, shape, pen_px, supersample=4):
    """Fraction of each pixel covered by ink when `lines` ((row, col) points)
    are drawn with a pen `pen_px` pixels wide (opaque ink, overlaps count once)."""
    from PIL import ImageDraw
    H, W = shape
    k = supersample
    im = Image.new("L", (W * k, H * k), 0)
    d = ImageDraw.Draw(im)
    width = max(1, int(round(pen_px * k)))
    for l in lines:
        if len(l) > 1:
            d.line([((p[1] + 0.5) * k, (p[0] + 0.5) * k) for p in l], fill=255, width=width)
        elif l:
            r = width / 2
            x, y = (l[0][1] + 0.5) * k, (l[0][0] + 0.5) * k
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    a = np.asarray(im, dtype=float).reshape(H, k, W, k).mean(axis=(1, 3)) / 255.0
    return a

def load_gray(image_path, cfg):
    img = ImageOps.exif_transpose(Image.open(image_path))
    if cfg.crop:
        x, y, w, h = cfg.crop
        img = img.crop((x, y, x + w, y + h))
    if cfg.max_side and max(img.size) > cfg.max_side:
        s = cfg.max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                         Image.Resampling.LANCZOS)
    return np.array(img.convert("L"), dtype=float) / 255


class Scribble:
    def __init__(self, pix, cfg: ScribbleConfig):
        self.cfg = cfg
        self.pix = pix
        self.lines = []

    # -- preprocessing (c_v25 load_and_preprocess) --------------------------
    def preprocess(self):
        c = self.cfg
        if c.remove_noise:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma_est = estimate_sigma(self.pix)
            self.pix = denoise_nl_means(self.pix, h=1.15 * sigma_est, patch_size=7,
                                        patch_distance=11, fast_mode=False)
        self.ipix = 1 - self.pix
        if c.white_cut > 0:                       # app5/7 (via prob_mask aliasing ipix)
            ip = self.ipix
            ip[ip < ip.min() + (ip.max() - ip.min()) * c.white_cut] = 0
        self.edges_x = sobel(self.pix, 0)
        self.edges_y = sobel(self.pix, 1)
        mag = _rescale(np.sqrt(self.edges_x ** 2 + self.edges_y ** 2))
        mag[mag < c.gradient_threshold] = 0
        self.edges_x[mag == 0] = 0
        self.edges_y[mag == 0] = 0
        d = c.trace_kernel_size
        w2 = gaussian_kernel_2d(2 * d + 1, c.trace_kernel_sigma)
        self.gx = convolve(self.edges_x, w2)
        self.gy = convolve(self.edges_y, w2)

        ck = gaussian_kernel_2d(2 * c.cleanup_radius + 1, c.trace_kernel_sigma)
        ipix_smoothed = convolve(self.ipix, ck)
        self.ipix_perceptual = ipix_smoothed
        ipix_squared = convolve(ipix_smoothed ** 2, ck)
        self.ipix_variance = np.clip(np.sqrt(np.maximum(0, ipix_squared - ipix_smoothed ** 2)), 0, 1)
        lp = 0.85 * c.light_percentile
        pos = ipix_smoothed[ipix_smoothed > 0]
        var = self.ipix_variance[self.ipix_variance > 0]
        t_i = np.percentile(pos, lp) if pos.size else 0.0
        t_v = np.percentile(var, lp) if var.size else 0.0
        self.threshold = t_i + t_v - t_i * t_v

        s = mag.sum()
        if s > 0:
            mag /= s
        self.magnitude = mag
        self.tracer = _Tracer(self.ipix, d)
        self.gk_main = w2 if c.legacy_gradient_sum else None
        self._disks = {}
        return self

    # -- cleanup ------------------------------------------------------------
    def _disk(self, r):
        if r not in self._disks:
            yo, xo = np.ogrid[-r:r + 1, -r:r + 1]
            cx, cy = np.where(xo ** 2 + yo ** 2 <= r ** 2)
            self._disks[r] = (cx - r, cy - r)
        return self._disks[r]

    def _adaptive_radii(self, xs, ys, max_radius):
        """Vectorised c_v25 _calculate_adaptive_radius."""
        H, W = self.ipix_perceptual.shape
        inb = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)
        r = np.full(len(xs), max_radius, dtype=np.int64)
        if inb.any():
            a = self.ipix_perceptual[xs[inb], ys[inb]]
            v = self.ipix_variance[xs[inb], ys[inb]]
            h = a + v - a * v
            t = self.threshold
            factor = np.clip((h - t) / (1.0 - t), 0, 1)
            ratio = np.where(h < t, 1.0, 1 - factor)
            sr = (1 + ratio * (max_radius - 1)).astype(np.int64)     # int() truncation, values >= 0
            r[inb] = np.maximum(0, np.minimum(sr, max_radius))
        return r

    def _zero_magnitude(self, xs, ys, radii):
        H, W = self.magnitude.shape
        flat = []
        for r in np.unique(radii):
            sel = radii == r
            dx, dy = self._disk(int(r))
            X = np.clip(xs[sel, None] + dx[None, :], 0, H - 1)
            Y = np.clip(ys[sel, None] + dy[None, :], 0, W - 1)
            flat.append((X * W + Y).ravel())
        if not flat:
            return
        flat = np.unique(np.concatenate(flat))
        m = self.magnitude.ravel()
        self._nonzero -= int(np.count_nonzero(m[flat]))
        m[flat] = 0
        if self._sampler is not None:
            self._sampler.refresh(flat)

    def _zero_box(self, data, rs, cs, track):
        H, W = data.shape
        r0, r1, _ = rs.indices(H)
        c0, c1, _ = cs.indices(W)
        if r0 >= r1 or c0 >= c1:
            return
        view = data[r0:r1, c0:c1]
        if track:
            n = int(np.count_nonzero(view))
            if not n:
                return
            self._nonzero -= n
            view[...] = 0
            if self._sampler is not None:
                self._sampler.refresh_range(r0 * W + c0, (r1 - 1) * W + c1 - 1)
        else:
            view[...] = 0

    def _quadrant_cleanup(self, line, data, d, track):
        """app5/7 gradient_clean_up2: four boxes per point, each sized by the
        darkest pixel in that quadrant (dark -> 0, light -> d)."""
        ip = self.ipix
        mx, my = ip.shape
        for x, y in line:
            x, y = int(x), int(y)
            xdp = max(min(x + d + 1, mx), 1)
            xdn = max(x - d, 0)
            ydp = max(min(y + d + 1, my), 1)
            ydn = max(y - d, 0)
            pp = int((1 - np.max(ip[x:xdp, y:ydp])) * d)
            pn = int((1 - np.max(ip[x:xdp, ydn:y + 1])) * d)
            npp = int((1 - np.max(ip[xdn:x + 1, y:ydp])) * d)
            nn = int((1 - np.max(ip[xdn:x + 1, ydn:y + 1])) * d)
            self._zero_box(data, slice(x, x + pp + 1), slice(y, y + pp + 1), track)
            self._zero_box(data, slice(x, x + pn + 1), slice(y - pn, y + 1), track)
            self._zero_box(data, slice(x - npp, x + 1), slice(y, y + npp + 1), track)
            self._zero_box(data, slice(x - nn, x + 1), slice(y - nn, y + 1), track)

    def _cleanup_line(self, line):
        if self.cfg.cleanup_mode == "quadrant":
            arr = np.asarray(line, dtype=np.int64)
            self.gx[arr[:, 0], arr[:, 1]] = 0
            self.gy[arr[:, 0], arr[:, 1]] = 0
            self._quadrant_cleanup(line, self.magnitude, self.cfg.cleanup_radius, True)
            return
        arr = np.asarray(line, dtype=np.int64)
        xs, ys = arr[:, 0], arr[:, 1]
        radii = self._adaptive_radii(xs, ys, self.cfg.cleanup_radius)
        self.gx[xs, ys] = 0
        self.gy[xs, ys] = 0
        self._zero_magnitude(xs, ys, radii)

    # -- start point --------------------------------------------------------
    def _pick_point(self):
        if self.cfg.sampler == "exact":
            p = self.magnitude / self.magnitude.sum() if self.magnitude.sum() > 0 else self.magnitude
            cdf = p.ravel().cumsum()
            cdf /= cdf[-1]
            i = int(cdf.searchsorted(np.random.random_sample(), side="right"))
        else:
            i = self._sampler.sample()
            if i is None:
                return None
        return np.unravel_index(i, self.magnitude.shape)

    # -- main lines (c_v25 trace_main_lines) --------------------------------
    def trace_main_lines(self, progress=None):
        c = self.cfg
        points = self.magnitude.size
        self._nonzero = int(np.count_nonzero(self.magnitude))
        self._sampler = _BlockSampler(self.magnitude) if c.sampler == "fast" else None
        tr = self.tracer
        for i in range(c.max_lines):
            if self._nonzero / points < c.min_magnitude_ratio or self._nonzero == 0:
                break
            pt = self._pick_point()
            if pt is None:
                break
            px, py = int(pt[0]), int(pt[1])
            if self.ipix[px, py] < 0.01:
                if c.cleanup_mode == "quadrant":
                    self._quadrant_cleanup([(px, py)], self.magnitude, 3, True)
                else:
                    self._zero_magnitude(np.array([px]), np.array([py]), np.array([3]))
                continue
            fwd = tr.trace(px, py, 90, self.gx, self.gy, max_steps=c.main_max_steps, gkernel=self.gk_main,
                           momentum=c.main_momentum)
            bwd = tr.trace(px, py, 270, self.gx, self.gy, max_steps=c.main_max_steps, gkernel=self.gk_main,
                           momentum=c.main_momentum)
            if c.split_main:
                self.lines.append(fwd)
                self.lines.append(bwd)
                self._cleanup_line(bwd)
                self._cleanup_line(fwd)
            else:
                line = list(reversed(bwd[1:])) + fwd
                self.lines.append(line)
                self._cleanup_line(line)
            if progress and i % 500 == 0:
                progress(f"  main lines: {len(self.lines)}")
        self.lines = [l for l in self.lines if len(l) >= c.main_min_points]
        if c.main_order in ("sort", "sort_merge"):
            self.lines = sorted(self.lines, key=len, reverse=True)
        if c.main_order == "sort_merge":
            self.lines = merge_lines(self.lines)
        return self

    # -- cross-hatching (c_v25 add_crosshatching) ---------------------------
    def hatch_seeds(self):
        c = self.cfg
        seeds = []
        if c.hatch_source == "longest":          # app5: 1000 longest, stop at the first short one
            src = []
            for line in self.lines[:1000]:
                if len(line) < c.hatch_min_line_length:
                    break
                src.append(line)
        else:                                    # all (c_v25) / merged (app7: lines already merged)
            src = [l for l in self.lines if len(l) > c.hatch_min_line_length]
        for line in src:
            for i, (px, py) in enumerate(line):
                if c.hatch_adaptive_spacing:
                    interval = int(c.hatch_sample_interval * (1.5 - self.ipix[px, py]))
                else:
                    interval = c.hatch_sample_interval
                if i % interval == 0:
                    seeds.append((px, py))
        return seeds

    def add_crosshatching(self, progress=None):
        c = self.cfg
        d = c.trace_kernel_size
        gxs = gaussian_blur(self.edges_x, 10 * 2 * d + 1, 6, c.separable_blur)
        gys = gaussian_blur(self.edges_y, 10 * 2 * d + 1, 6, c.separable_blur)
        gk = gaussian_kernel_2d(10 * 2 * d + 1, 6) if c.legacy_gradient_sum else None
        extra = (c.hatch_keep_min, gk)
        dp = c.hatch_patch or d
        seeds = self.hatch_seeds()
        workers = c.workers or os.cpu_count() or 1
        if workers > 1 and len(seeds) > 200:
            from concurrent.futures import ProcessPoolExecutor
            chunks = [seeds[k::workers] for k in range(workers)]
            args = [(self.ipix, dp, gxs, gys, ch, tuple(c.hatch_angles), c.hatch_max_length) + extra
                    for ch in chunks]
            with ProcessPoolExecutor(workers) as ex:
                parts = list(ex.map(_hatch_chunk, args))
            # restore the original (seed-major) order
            per_seed = [None] * len(seeds)
            for k, part in enumerate(parts):
                for j, res in enumerate(part):
                    per_seed[k + j * workers] = res
            new = [l for res in per_seed for l in res]
        else:
            new = [l for res in _hatch_chunk((self.ipix, dp, gxs, gys, seeds, tuple(c.hatch_angles),
                                              c.hatch_max_length) + extra) for l in res]
        self.lines.extend(new)
        if progress:
            progress(f"  hatch lines: {len(new)} from {len(seeds)} seeds")
        return self

    def add_clean_hatching(self, progress=None):
        """Hatching without U-turns, spirals or clumps: hatch directions from a
        smoothed *orientation* field (sign kept consistent along a line), lines
        in one direction family kept apart by a tone-dependent spacing, total
        turning limited, sub-pixel steps. Starts from the same seed points on
        the main lines as the legacy hatching, then fills evenly."""
        from .hatch import HatchConfig, hatch_layer, direction_field, _Shared
        c = self.cfg
        angles = tuple(c.hatch_angles)
        hc = HatchConfig(cell_mm=1.0, spacing_min_mm=c.hatch_spacing_min_px,
                         spacing_max_mm=c.hatch_spacing_max_px, flow="gradient",
                         flow_smooth_mm=c.hatch_flow_smooth_px, flow_fill_mm=c.hatch_fill_px, step_mm=1.0,
                         max_len_mm=c.hatch_max_length, min_len_mm=c.hatch_clean_min_len,
                         min_len_spacings=2.0, tone_smooth_mm=1.0, seed_margin=0.03, overrun=0.05,
                         layer_thresholds=tuple(max(0.0, t + c.hatch_threshold_shift) for _, t in angles))
        darkness = self.ipix
        if c.hatch_ink_aware or c.hatch_mode == "ink":
            # what the main lines already draw, at the real pen width on paper
            from scipy.ndimage import gaussian_filter
            mm_per_px = paper_fit(*self.pix.shape, c)[0]
            pen_px = c.pen_mm / mm_per_px
            main_ink = ink_coverage(self.lines, self.pix.shape, pen_px)
            v = c.hatch_view_px
            T = np.clip(gaussian_filter(self.ipix, v), 0, 0.97)
            M = np.clip(gaussian_filter(main_ink, v), 0, 0.97)
            # ink combines like 1 - (1 - main)(1 - hatch) = target  ->  hatch = 1 - (1 - T)/(1 - M)
            darkness = np.clip(1 - (1 - T) / (1 - M), 0, 1)
            hc.tone_model = "coverage"           # spacing from real pen coverage of the remaining tone
            hc.pen_mm = pen_px                   # cell units = px
            hc.tone_smooth_mm = 0.0              # already at viewing scale
            self.main_ink, self.hatch_target = main_ink, darkness
            if progress:
                progress(f"  ink-aware: pen {pen_px:.2f} px, main lines cover {main_ink.mean()*100:.0f}% of the page, "
                         f"hatching asked for {darkness.mean()*100:.0f}% more")
        mask = np.ones(darkness.shape, dtype=bool)
        base = direction_field(self.pix, hc, 1.0)
        seeds = [(float(x), float(y)) for x, y in self.hatch_seeds()]

        def hatch_pass(dk):
            shared = _Shared(dk, mask, hc, 1.0)
            rng = np.random.default_rng(0 if c.seed is None else c.seed)
            out = []
            for (angle, _), thr in zip(angles, hc.layer_thresholds):
                theta = base + math.radians(angle)
                out += hatch_layer(dk, mask, theta, thr, hc, 1.0, rng, shared,
                                   seed_points=seeds, max_total_turn_deg=c.hatch_max_turn_deg)
            return out

        new = hatch_pass(darkness)
        if (c.hatch_ink_aware or c.hatch_mode == "ink") and c.hatch_ink_passes > 1:
            # Streamline hatching never reaches its nominal density (lines stop short of
            # each other, short strokes are dropped), so measure what was achieved and
            # ask for proportionally more where it fell short: per pixel, raise the
            # requested coverage r to 1 - (1 - r)^g with g = log(1-r)/log(1-achieved).
            from scipy.ndimage import gaussian_filter
            for _ in range(c.hatch_ink_passes - 1):
                got = np.clip(gaussian_filter(ink_coverage(new, self.pix.shape, hc.pen_mm), c.hatch_view_px), 0, 0.97)
                want = np.clip(self.hatch_target, 0, 0.97)
                g = np.log1p(-want) / np.minimum(np.log1p(-got), -1e-3)
                g = np.clip(gaussian_filter(np.where(want > 0.02, g, 1.0), 4 * c.hatch_view_px), 1.0, 3.0)
                darkness = np.clip(1 - (1 - want) ** g, 0, 0.97)
                new = hatch_pass(darkness)
            if progress:
                progress(f"  ink-aware: corrected request (gain {g.mean():.2f} avg)")
        self.lines.extend(new)
        if progress:
            progress(f"  clean hatch lines: {len(new)}")
        return self

    def run(self, progress=None):
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
        self.preprocess()
        self.trace_main_lines(progress)
        self.n_main = len(self.lines)
        if self.cfg.hatch:
            if self.cfg.hatch_mode in ("clean", "ink"):
                self.add_clean_hatching(progress)
            else:
                self.add_crosshatching(progress)
        if self.cfg.smooth_main and self.cfg.smooth_main > 1:
            self.lines[:self.n_main] = [_smooth_line(l, self.cfg.smooth_main) for l in self.lines[:self.n_main]]
        if self.cfg.min_points > 1:
            self.lines = [l for l in self.lines if len(l) >= self.cfg.min_points]
        return self.lines

    def render(self):
        out = np.ones(self.pix.shape, dtype=np.uint8) * 255
        pts = np.array([p for l in self.lines for p in l], dtype=float).astype(np.int64).reshape(-1, 2)
        if len(pts):
            out[np.clip(pts[:, 0], 0, out.shape[0] - 1), np.clip(pts[:, 1], 0, out.shape[1] - 1)] = 0
        return out


def _smooth_line(line, window):
    """Moving average with float output; end points kept (lines still meet)."""
    if len(line) <= 2:
        return line
    a = np.asarray(line, dtype=float)
    k = int(window)
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(ap[:, 0], ker, mode="valid"), np.convolve(ap[:, 1], ker, mode="valid")], 1)
    sm = sm[:len(a)]
    sm[0], sm[-1] = a[0], a[-1]
    return [tuple(p) for p in sm.tolist()]


def _hatch_chunk(args):
    ipix, d, gxs, gys, seeds, angles, max_len, keep_min, gk = args
    tr = _Tracer(ipix, d)
    out = []
    for px, py in seeds:
        res = []
        for angle, min_int in angles:
            l = tr.trace(px, py, angle, gxs, gys, max_steps=max_len, min_intensity=min_int, gkernel=gk)
            if len(l) >= keep_min:
                res.append(l)
        out.append(res)
    return out


def scribble(image_path, cfg: ScribbleConfig, progress=None):
    """Returns (lines in (row, col) pixel coords, image shape)."""
    pix = load_gray(image_path, cfg)
    s = Scribble(pix, cfg)
    s.run(progress)
    return s.lines, pix.shape, s
