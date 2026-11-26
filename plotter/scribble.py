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

    def trace(self, px, py, angle, gx, gy, max_steps=1000, min_intensity=0.0):
        rad = angle * np.pi / 180.0
        crad = np.cos(rad)
        srad = np.sin(rad)
        H, W = self.H, self.W
        line = [(int(px), int(py))]
        seen = {line[0]}
        unique = True
        step = 0
        while -1 < px < H - 1 and -1 < py < W - 1 and unique and step < max_steps:
            cx, cy = int(px), int(py)
            a, b = gx[cx, cy], gy[cx, cy]
            g_x = a * crad + b * srad
            g_y = -a * srad + b * crad
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

    def _cleanup_line(self, line):
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
                self._zero_magnitude(np.array([px]), np.array([py]), np.array([3]))
                continue
            fwd = tr.trace(px, py, 90, self.gx, self.gy)
            bwd = tr.trace(px, py, 270, self.gx, self.gy)
            line = list(reversed(bwd[1:])) + fwd
            self.lines.append(line)
            self._cleanup_line(line)
            if progress and i % 500 == 0:
                progress(f"  main lines: {len(self.lines)}")
        self.lines = [l for l in self.lines if len(l) > 1]
        return self

    # -- cross-hatching (c_v25 add_crosshatching) ---------------------------
    def hatch_seeds(self):
        c = self.cfg
        seeds = []
        for line in (l for l in self.lines if len(l) > c.hatch_min_line_length):
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
        seeds = self.hatch_seeds()
        workers = c.workers or os.cpu_count() or 1
        if workers > 1 and len(seeds) > 200:
            from concurrent.futures import ProcessPoolExecutor
            chunks = [seeds[k::workers] for k in range(workers)]
            args = [(self.ipix, d, gxs, gys, ch, tuple(c.hatch_angles), c.hatch_max_length) for ch in chunks]
            with ProcessPoolExecutor(workers) as ex:
                parts = list(ex.map(_hatch_chunk, args))
            # restore the original (seed-major) order
            per_seed = [None] * len(seeds)
            for k, part in enumerate(parts):
                for j, res in enumerate(part):
                    per_seed[k + j * workers] = res
            new = [l for res in per_seed for l in res]
        else:
            new = [l for res in _hatch_chunk((self.ipix, d, gxs, gys, seeds, tuple(c.hatch_angles),
                                              c.hatch_max_length)) for l in res]
        self.lines.extend(new)
        if progress:
            progress(f"  hatch lines: {len(new)} from {len(seeds)} seeds")
        return self

    def run(self, progress=None):
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
        self.preprocess()
        self.trace_main_lines(progress)
        if self.cfg.hatch:
            self.add_crosshatching(progress)
        if self.cfg.min_points > 1:
            self.lines = [l for l in self.lines if len(l) >= self.cfg.min_points]
        return self.lines

    def render(self):
        out = np.ones(self.pix.shape, dtype=np.uint8) * 255
        pts = np.array([p for l in self.lines for p in l], dtype=np.int64).reshape(-1, 2)
        if len(pts):
            out[np.clip(pts[:, 0], 0, out.shape[0] - 1), np.clip(pts[:, 1], 0, out.shape[1] - 1)] = 0
        return out


def _hatch_chunk(args):
    ipix, d, gxs, gys, seeds, angles, max_len = args
    tr = _Tracer(ipix, d)
    out = []
    for px, py in seeds:
        res = []
        for angle, min_int in angles:
            l = tr.trace(px, py, angle, gxs, gys, max_steps=max_len, min_intensity=min_int)
            if len(l) > 3:
                res.append(l)
        out.append(res)
    return out


def scribble(image_path, cfg: ScribbleConfig, progress=None):
    """Returns (lines in (row, col) pixel coords, image shape)."""
    pix = load_gray(image_path, cfg)
    s = Scribble(pix, cfg)
    s.run(progress)
    return s.lines, pix.shape, s
