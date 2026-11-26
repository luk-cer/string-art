# String Art - Radon-Based Line Art Renderer

Generates line art from images using Radon transform analysis. Outputs line coordinates as JSON for plotting or CNC/pen-plotter use.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python main.py <image> [options]
```

### Basic example

```bash
uv run python main.py photo.png
```

### Options

| Option | Default | Description |
|---|---|---|
| `image` | *(required)* | Input image path |
| `--lines` | `500` | Target number of lines |
| `--paper` | `A4` | Paper size (`A3`, `A4`, `A5`, `A6`, `Letter`, `Legal`) |
| `--pen` | `0.3` | Pen width in mm |
| `--padding` | `10.0` | Padding in mm |
| `--mode` | `fixed_6` | Rendering mode (see below) |
| `--output` | `output.json` | Output JSON path |
| `--save-target` | `target_image.png` | Save target image preview |
| `--save-rendered` | `rendered_image.png` | Save rendered line image preview |

### Rendering modes

| Mode | Description |
|---|---|
| `fixed_6` | Select 6 best-distributed angles, distribute lines evenly |
| `6_plus_refine` | Start with 6, add refinement angles if they improve quality |
| `pure_optimize` | Greedily add angles until quality threshold is met |
| `hierarchical` | Classify angles into primary/secondary/tertiary tiers with residual tracking |

```bash
# Fixed 6 angles, 800 lines
uv run python main.py photo.png --mode fixed_6 --lines 800

# Optimize with quality threshold
uv run python main.py photo.png --mode pure_optimize --quality-threshold 0.005 --max-lines 1000

# Hierarchical with cumulative strategy
uv run python main.py photo.png --mode hierarchical --hierarchical-strategy cumulative_contribution
```

### Metrics

**Angle selection** (`--angle-metric`):
- `radon_variance` - Radon transform variance (fast, default)
- `mse_reduction` - Actual MSE reduction per angle (slower, more accurate)

**Line spacing** (`--spacing-metric`):
- `adaptive` - Density-aware spacing (default)
- `even` - Uniform spacing
- `mse_optimized` - MSE-optimized placement (slow)

**Quality** (`--quality-metric`):
- `mse` - Mean squared error (default)
- `mae` - Mean absolute error

**Perception weights**:
- `--edge-weight` (default `0.75`) - Weight for edge detection
- `--intensity-weight` (default `0.25`) - Weight for tonal intensity

```bash
# Custom metrics and weights
uv run python main.py photo.png --angle-metric mse_reduction --quality-metric mae --edge-weight 0.6 --intensity-weight 0.4
```

## Output

The output JSON contains:
- `metadata` - Paper size, circle geometry, pen width, line count
- `rendering` - Mode, config, weights used
- `metrics` - Selected metrics and final error
- `lines` - Array of line segments with `x1, y1, x2, y2` coordinates in mm

## Project structure

```
radon_renderer/
  __init__.py      # Package exports
  config.py        # Paper size constants
  metrics.py       # Angle selection, line spacing, and quality metrics
  renderer.py      # AdvancedRadonRenderer class
  cli.py           # CLI argument parsing
main.py            # Entry point
```

## Preview UI

```bash
uv run python -m plotter ui        # opens http://127.0.0.1:8765
```

Upload a photo, drag the square crop, pick **String art / Flow hatch / Classic**,
tune the sliders and press **Render** (or tick *auto*). The preview draws every
line at true pen width; scroll to zoom, drag to pan, double-click to fit.
*Travel moves* shows pen-up moves, ▶ animates the plot order, *Target* /
*Simulation* show what the algorithm aims for vs. what it thinks it drew.
Download the `.gcode`, `.svg` or `.json` from the bottom bar. The plotter
profile (pen up/down, dwell, feed, header, Y flip…) is editable in the sidebar
and remembered in the browser. Files are kept in `out/ui/`.

## Plotter output (G-code) and string art

The `plotter` package writes G-code directly, in the same dialect as the
hand-edited files that plotted correctly (`G21`, `G90`, `G0 Z5` pen up,
`G0 Z0` pen down, `G4 P0.05` dwell, `G1 ... F1500`). Tweak it via a profile
instead of editing output files:

```bash
uv run python -m plotter profile --dump my_plotter.json   # edit, then pass --profile my_plotter.json
```

### Existing renders -> G-code (same look)

```bash
uv run python main.py photo.png --mode hierarchical --output render.json
uv run python -m plotter lines2gcode render.json -o out/render     # out/render.gcode + .svg
```

Lines are re-ordered (nearest neighbour, with reversal) to minimise pen-up travel.

### String art portrait

Nails on a circle, one line from nail to nail, chosen greedily with a
pen-width-aware ink model so dark areas saturate instead of turning into a
black mess.

```bash
uv run python -m plotter stringart photo.jpg -o out/portrait --paper A3 --pen 0.2 --nail-spread 1.5
```

Outputs `out/portrait.gcode`, `.svg`, `_preview.png` (true pen width),
`_compare.png` (target vs simulation) and `.json` (nail sequence + settings).

| Option | Default | Notes |
|---|---|---|
| `--nails` | 240 | nails on the circle |
| `--lines` | 3000 | max lines; stops earlier when no line improves the image |
| `--pen` | 0.3 | pen width mm – thinner pen = more lines = more detail |
| `--pixel` | 0.6 | simulation cell mm ("viewing distance"); 1.0–1.6 gives smoother tones |
| `--overshoot` | 3.0 | penalty for ink over light areas; lower = darker/denser, higher = cleaner highlights |
| `--gamma` | 1.0 | >1 lightens midtones |
| `--clahe` | 0.02 | local contrast boost (0 = off) |
| `--vignette` | 0 | fade busy backgrounds towards the rim |
| `--min-skip` | 15 | no chords between nails closer than this |
| `--nail-spread` | 0 | mm; spreads line ends around each nail to avoid ink blobs |
| `--continuous` | off | single unbroken path (default lifts the pen when stuck) |
| `--crop X Y W H` | auto | source crop in pixels; auto trims white borders |
| `--paper/--portrait/--diameter/--padding` | A4 | layout |

### Flow hatching (curved crosshatch)

Strokes follow the shapes in the image instead of running as straight parallel
lines; darker tones add crossing layers (across the flow, then diagonals).

```bash
uv run python -m plotter hatch photo.jpg -o out/hatch --pen 0.3
uv run python -m plotter hatch photo.jpg -o out/hatch_straight --flow fixed   # classic straight crosshatch
```

Optional improvements (off by default, so the default look is unchanged):

| Flag | Effect |
|---|---|
| `--tone-model coverage` | spacing chosen so the real ink coverage of the crossing layers matches the photo's darkness: richer tones, clearer face, ~45% more ink |
| `--merge-gap 4` | joins stroke ends that continue each other: ~18% fewer pen lifts, same look |
| `--outline 0.5` | adds smooth contour lines at that darkness level on top (defines jaw/nose/hair edges) |

Useful knobs: `--flow-smooth` (mm, calmer vs more detailed curves), `--layers 1-4`,
`--thresholds` (darkness where each layer starts), `--spacing-max/--spacing-min`,
`--min-len`, `--shape rect`.

### Scribble (gradient tracing)

Fast port of `scribble-code/c_v25.py` (branch `add-scribble-code`): lines follow
the image's contours, plus angled cross-hatching in darker areas. Same look,
identical lines for the same seed, ~25x faster (image35: 201 s -> 8 s).

```bash
uv run python -m plotter scribble photo.jpg -o out/scribble --max-side 800
```

Presets reproduce the different scripts exactly (same lines for the same seed):

| `--preset` | Look | image35 @ A4, 800 px |
|---|---|---|
| `c_v25` (default) | even scribble texture everywhere, lower contrast | 221 m ink, ~3.2 h |
| `app5` | high contrast: light areas stay white, darks solid; hatch from the 1000 longest lines | 153 m, ~2.3 h |
| `app7` | like app5, hatching from all (merged) lines: denser darks | 284 m, ~3.8 h |

Hatching versions (`--hatch-mode`, UI "Hatching"):

| mode | what it does |
|---|---|
| `legacy` (default) | the original tracer from the scripts |
| `clean` | no U-turns, spirals or clumped hatch lines; evenly spaced by tone; also fills flat dark areas |
| `ink` | clean, but hatches only the darkness the main lines haven't drawn yet (main-line ink measured at the real pen width, then hatch density calibrated in a second pass) |

Freehand scribble (`--hatch-mode freehand`): a simulated hand scribbling loops around a
wandering centre, with momentum on its turning and random drift in path, loop size, speed and
shape. It crosses itself and the contours, erases the darkness it has drawn as it goes, and after
a stroke of random length lifts and restarts where darkness is still missing. Knobs:
`--density` (0.55; lower = lighter), `--loops DARK LIGHT` (4 10 px), `--walls none|major|all`
(default `major`: don't cross silhouettes / big shape edges, found as drawn contours lying on
edges that survive at `--major-scale` px, 4 default, 6 = silhouette only), `--no-main` (scribble
only). The scribble only adds the darkness the contour lines haven't drawn (contour ink is measured
at the real pen width) and erases its own ink as it draws, so nothing is counted twice. Example:
`--preset app5 --hatch-mode freehand --density 0.5 --momentum 0.5 --smooth 3`.

Other opt-in improvements: `--momentum 0.5` (calmer main lines), `--smooth 3` (no pixel
staircase). Example: `--preset app5 --hatch-mode ink --momentum 0.5 --smooth 3`.

`--max-side` sets the tracing resolution, which sets line density on paper
(one traced pixel ~ one pen step): keep mm-per-pixel (printed at the end) at
or above the pen width or dark areas fill in solid. Other knobs mirror
`SketchConverter`: `--gradient-threshold`, `--cleanup-radius`, `--max-lines`,
`--hatch-angles 210:0.5,150:0.7,...`, `--no-hatch`, `--denoise`.

See `TODO.md` for the review of the existing code and the roadmap
(better crosshatching, preview UI, string-art quality).
