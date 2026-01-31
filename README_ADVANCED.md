# Advanced Radon-Based Line Art Renderer

Complete implementation with pluggable metrics, 4 rendering modes, and human perception modeling.

## Overview

This system converts black and white images into plotter-ready line art using Radon transform analysis. It supports multiple rendering strategies and pluggable quality metrics for experimentation and optimization.

## Key Features

### Four Rendering Modes

1. **Fixed 6 Angles** (`fixed_6`)
   - Selects 6 dominant angles from Radon analysis
   - Distributes lines evenly across angles
   - Fast, predictable results
   - Best for: Simple images, quick previews

2. **6 Plus Refinement** (`6_plus_refine`)
   - Starts with 6 primary angles
   - Adds refinement angles if they improve quality
   - Stops when improvement < threshold
   - Best for: High-quality output with structured foundation

3. **Pure Optimization** (`pure_optimize`)
   - No preset angle count
   - Adds angles greedily until quality threshold reached
   - Uses as few or many angles as needed
   - Best for: Optimal quality regardless of complexity

4. **Hierarchical** (`hierarchical`)
   - Primary angles (dominant structure)
   - Secondary angles (major refinements)
   - Tertiary angles (fine details)
   - Two sub-strategies:
     - **Strength-based**: Classify by Radon strength thresholds
     - **Cumulative contribution**: Classify by percentage of total structure

### Pluggable Metrics

All metrics use abstract base classes that you can extend:

#### Angle Selection Metrics
- `RadonVarianceMetric`: Fast heuristic based on Radon variance (default)
- `MSEReductionMetric`: Slower but accurate, tests actual MSE reduction
- **Custom**: Extend `AngleSelectionMetric` base class

#### Line Spacing Metrics
- `EvenSpacingMetric`: Uniform spacing
- `AdaptiveSpacingMetric`: Adapts to image importance (default)
- `MSEOptimizedSpacingMetric`: Optimizes each line position
- **Custom**: Extend `LineSpacingMetric` base class

#### Quality Metrics
- `MSEQualityMetric`: Mean Squared Error (default)
- `MAEQualityMetric`: Mean Absolute Error
- **Custom**: Extend `QualityMetric` base class

### Human Perception Model

Based on visual perception research:
- **Edge weight**: 75% (default) - Preserves shapes and contours
- **Intensity weight**: 25% (default) - Maintains tonal values
- Configurable via command line

Research basis:
- Marr (1982): Edge detection is primary in human vision
- Hubel & Wiesel: 5-10x more edge-sensitive neurons in V1
- Campbell & Robson: Contrast Sensitivity Function

## Installation

```bash
pip install numpy pillow scikit-image scipy matplotlib
```

## Quick Start

```bash
# Mode 1: Fixed 6 angles
python advanced_radon_renderer.py image.jpg --mode fixed_6 --lines 500

# Mode 2: 6 + refinement
python advanced_radon_renderer.py image.jpg --mode 6_plus_refine --lines 400 --max-lines 600

# Mode 3: Pure optimization
python advanced_radon_renderer.py image.jpg --mode pure_optimize --quality-threshold 0.01

# Mode 4a: Hierarchical strength-based
python advanced_radon_renderer.py image.jpg --mode hierarchical \
    --hierarchical-strategy strength_based \
    --hierarchical-config '{"primary_min":0.8,"secondary_min":0.5}'

# Mode 4b: Hierarchical cumulative
python advanced_radon_renderer.py image.jpg --mode hierarchical \
    --hierarchical-strategy cumulative_contribution \
    --hierarchical-config '{"primary_percent":50,"secondary_percent":30}'
```

## Command Line Reference

### Basic Parameters
```
--lines LINES              Target number of lines (default: 500)
--paper {A3,A4,A5,...}     Paper size (default: A4)
--pen PEN                  Pen width in mm (default: 0.3)
--padding PADDING          Padding from edge in mm (default: 10.0)
```

### Rendering Mode
```
--mode {fixed_6,6_plus_refine,pure_optimize,hierarchical}
```

### Optimization Parameters
```
--quality-threshold FLOAT  Min improvement to continue (default: 0.01)
--max-lines INT           Maximum lines for optimization modes
```

### Hierarchical Parameters
```
--hierarchical-strategy {strength_based,cumulative_contribution}
--hierarchical-config JSON
```

**Strength-based config:**
```json
{
  "primary_min": 0.8,        // Min strength for primary (0-1)
  "secondary_min": 0.5,      // Min strength for secondary (0-1)
  "max_primary": 6,          // Max primary angles
  "max_secondary": 5         // Max secondary angles
}
```

**Cumulative config:**
```json
{
  "primary_percent": 60,     // % of structure for primary
  "secondary_percent": 25,   // % for secondary
  "tertiary_percent": 15     // % for tertiary
}
```

### Metric Selection
```
--angle-metric {radon_variance,mse_reduction}
--spacing-metric {even,adaptive,mse_optimized}
--quality-metric {mse,mae}
```

### Perception Weights
```
--edge-weight FLOAT        Weight for edges (default: 0.75)
--intensity-weight FLOAT   Weight for intensity (default: 0.25)
```

### Output
```
--output FILE              Output JSON (default: output.json)
--save-target FILE         Save target image (default: target_image.png)
--save-rendered FILE       Save rendered image (default: rendered_image.png)
```

## Output Format

### JSON Structure
```json
{
  "metadata": {
    "source_image": "image.jpg",
    "paper_size": "A4",
    "paper_dimensions_mm": {"width": 297, "height": 210},
    "circle": {
      "diameter_mm": 190.0,
      "center_x_mm": 148.5,
      "center_y_mm": 105.0,
      "radius_mm": 95.0
    },
    "padding_mm": 10.0,
    "pen_width_mm": 0.3,
    "total_lines": 300
  },
  "rendering": {
    "mode": "pure_optimize",
    "config": {...},
    "edge_weight": 0.75,
    "intensity_weight": 0.25
  },
  "metrics": {
    "angle_selection": "radon_variance",
    "line_spacing": "adaptive_spacing",
    "quality": "mse",
    "final_error": 0.041861
  },
  "lines": [
    {
      "x1": 148.5,
      "y1": 10.0,
      "x2": 148.5,
      "y2": 200.0,
      "angle": 90.0
    },
    ...
  ]
}
```

### Target Image

The `target_image.png` shows what the renderer is trying to approximate:
- White on black
- Combines edges (75%) and intensity (25%)
- Visual reference for understanding the optimization goal

### Rendered Image

The `rendered_image.png` shows the actual line rendering as raster:
- For quality comparison
- Helps visualize the result
- Can be compared to target

## Comparing Modes

Use the comparison tool to evaluate different modes:

```bash
python compare_modes.py \
    mode1.json mode2.json mode3.json mode4a.json mode4b.json \
    --labels "Fixed 6" "6+Refine" "Pure Opt" "Hier-Str" "Hier-Cum" \
    --target target_image.png \
    --save comparison.png
```

Outputs:
- Comparison table with metrics
- Detailed breakdown of each mode
- Visual comparison figure

## Mode Selection Guide

### When to Use Each Mode

**Fixed 6** - Best for:
- Quick previews
- Simple geometric images
- When you want predictable crosshatch pattern
- Artistic/stylized look

**6 Plus Refinement** - Best for:
- High-quality final output
- When you want both structure and detail
- Images with clear dominant orientations
- Balance between quality and visual coherence

**Pure Optimization** - Best for:
- Maximum quality regardless of complexity
- When line count is flexible
- Images with subtle details
- Experimental/research applications

**Hierarchical (Strength)** - Best for:
- Images with clear structural hierarchy
- When you want control over angle distribution
- Artistic control over primary/secondary structure

**Hierarchical (Cumulative)** - Best for:
- Balanced structure capture
- When you want percentage-based distribution
- Even coverage across different feature scales

## Example Results

From test image (star in circle):

| Mode | Lines | Angles | Error (MSE) | Notes |
|------|-------|--------|-------------|-------|
| Fixed 6 | 300 | 6 | 0.046809 | Dense crosshatch |
| 6+Refine | 300 | 6 | 0.046809 | No refinement needed |
| Pure Opt | 150 | 3 | 0.041861 | Lower error, fewer lines! |
| Hier-Str | 300 | 11 | 0.048662 | Many angles, more complexity |
| Hier-Cum | 300 | 11 | 0.043578 | Better error than fixed |

**Key insight**: Pure optimization achieved lower error with half the lines by focusing on most important angles.

## Extending with Custom Metrics

### Custom Angle Selection Metric

```python
from advanced_radon_renderer import AngleSelectionMetric

class MyCustomMetric(AngleSelectionMetric):
    def compute_angle_scores(self, image, edges, sinogram_edges, 
                            sinogram_intensity, theta, edge_weight, 
                            intensity_weight):
        # Your custom logic here
        # Return scores array (higher = better angle)
        scores = ...
        return scores
    
    def get_name(self):
        return "my_custom_metric"

# Use it
renderer = AdvancedRadonRenderer(
    image_path,
    angle_metric=MyCustomMetric()
)
```

### Custom Quality Metric

```python
from advanced_radon_renderer import QualityMetric

class PerceptualLossMetric(QualityMetric):
    def compute_error(self, target, generated):
        # Your perceptual loss calculation
        # Could use SSIM, LPIPS, etc.
        error = ...
        return error
    
    def get_name(self):
        return "perceptual_loss"
```

### Custom Line Spacing

```python
from advanced_radon_renderer import LineSpacingMetric

class ImportanceGuidedSpacing(LineSpacingMetric):
    def compute_line_positions(self, importance_map, angle_deg, 
                               num_lines, circle_radius, image_shape):
        # Your spacing algorithm
        positions = ...
        return positions
    
    def get_name(self):
        return "importance_guided"
```

## Performance Notes

### Speed Comparison

- **Radon Variance** (angle metric): ~0.3s
- **MSE Reduction** (angle metric): ~3-5s (evaluates actual line quality)
- **Adaptive Spacing**: Minimal overhead
- **MSE Optimized Spacing**: Much slower (not fully implemented)

### Memory Usage

- ~100-200 MB for typical images
- Scales with image size
- JSON output: 50-200 KB depending on line count

### Optimization Tips

1. Start with `radon_variance` for angle selection (fastest)
2. Use `adaptive` spacing (good balance)
3. Use `mse` quality metric (standard)
4. Test with low line counts first
5. Only use `mse_reduction` angle metric for final high-quality output

## Future Enhancements

Potential improvements:
- SSIM quality metric (perceptual)
- GPU acceleration for line rendering
- Multi-resolution analysis
- Color support (separate channels)
- Real-time preview GUI
- Direct G-code/HPGL export
- Line thickness variation
- Optimization for specific plotter constraints

## Troubleshooting

**Issue**: Mode 2 doesn't add any refinement angles
- **Cause**: Quality threshold too high or 6 angles already sufficient
- **Solution**: Lower `--quality-threshold` to 0.005 or less

**Issue**: Pure optimization uses too few angles
- **Cause**: Aggressive quality threshold
- **Solution**: Lower threshold or increase `--lines` target

**Issue**: Hierarchical has too many tertiary angles
- **Cause**: Thresholds too permissive
- **Solution**: Increase `primary_min` and `secondary_min` in config

**Issue**: High MSE error
- **Cause**: Not enough lines or wrong angles
- **Solution**: Increase line count or try different mode

**Issue**: Slow rendering with MSE reduction metric
- **Expected**: This metric is inherently slow (tests actual lines)
- **Solution**: Use for final output only, use radon_variance for testing

## References

### Computer Vision
- Deans, S. R. (1983). *The Radon Transform and Some of Its Applications*
- Canny, J. (1986). "A Computational Approach to Edge Detection"

### Human Perception
- Marr, D. (1982). *Vision: A Computational Investigation*
- Hubel & Wiesel (1968). "Receptive fields in monkey striate cortex"
- Campbell & Robson (1968). "Fourier analysis and visibility of gratings"

## License

MIT License

---

**Advanced Radon Line Art Renderer**
Combining computer vision, human perception research, and computational geometry
for high-quality plotter art generation.
