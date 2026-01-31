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
