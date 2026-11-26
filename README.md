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

See `TODO.md` for the review of the existing code and the roadmap
(better crosshatching, preview UI, string-art quality).
