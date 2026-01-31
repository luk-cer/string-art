import argparse
import json
import time

import numpy as np
from PIL import Image

from .config import PAPER_SIZES
from .metrics import (
    RadonVarianceMetric,
    MSEReductionMetric,
    EvenSpacingMetric,
    AdaptiveSpacingMetric,
    MSEOptimizedSpacingMetric,
    MSEQualityMetric,
    MAEQualityMetric,
)
from .renderer import AdvancedRadonRenderer


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Radon-based line art renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('image', help='Input image path')
    parser.add_argument('--lines', type=int, default=500, help='Target lines (default: 500)')
    parser.add_argument('--paper', default='A4', choices=list(PAPER_SIZES.keys()))
    parser.add_argument('--pen', type=float, default=0.3, help='Pen width mm')
    parser.add_argument('--padding', type=float, default=10.0, help='Padding mm')

    # Rendering mode
    parser.add_argument('--mode', default='fixed_6',
                       choices=['fixed_6', '6_plus_refine', 'pure_optimize', 'hierarchical'],
                       help='Rendering mode')

    # Hierarchical options
    parser.add_argument('--hierarchical-strategy', default='strength_based',
                       choices=['strength_based', 'cumulative_contribution'])
    parser.add_argument('--hierarchical-config', type=str,
                       help='JSON config for hierarchical mode')

    # Quality thresholds
    parser.add_argument('--quality-threshold', type=float, default=0.01,
                       help='Min improvement threshold for optimization modes')
    parser.add_argument('--max-lines', type=int, help='Max lines for optimization modes')

    # Metric selection
    parser.add_argument('--angle-metric', default='radon_variance',
                       choices=['radon_variance', 'mse_reduction'])
    parser.add_argument('--spacing-metric', default='adaptive',
                       choices=['even', 'adaptive', 'mse_optimized'])
    parser.add_argument('--quality-metric', default='mse',
                       choices=['mse', 'mae'])

    # Perception weights
    parser.add_argument('--edge-weight', type=float, default=0.75)
    parser.add_argument('--intensity-weight', type=float, default=0.25)

    # Output
    parser.add_argument('--output', default='output.json')
    parser.add_argument('--save-target', default='target_image.png',
                       help='Save target image (edge+intensity mix)')
    parser.add_argument('--save-rendered', default='rendered_image.png',
                       help='Save rendered line image')

    args = parser.parse_args()

    print("="*70)
    print("ADVANCED RADON LINE ART RENDERER")
    print("="*70)

    # Create metrics
    angle_metric_map = {
        'radon_variance': RadonVarianceMetric(),
        'mse_reduction': MSEReductionMetric(None)  # Will set renderer later
    }

    spacing_metric_map = {
        'even': EvenSpacingMetric(),
        'adaptive': AdaptiveSpacingMetric(),
        'mse_optimized': MSEOptimizedSpacingMetric(None, None)
    }

    quality_metric_map = {
        'mse': MSEQualityMetric(),
        'mae': MAEQualityMetric()
    }

    # Create renderer
    renderer = AdvancedRadonRenderer(
        args.image,
        num_lines=args.lines,
        paper_size=args.paper,
        pen_width=args.pen,
        padding=args.padding,
        edge_weight=args.edge_weight,
        intensity_weight=args.intensity_weight,
        angle_metric=angle_metric_map[args.angle_metric],
        spacing_metric=spacing_metric_map[args.spacing_metric],
        quality_metric=quality_metric_map[args.quality_metric]
    )

    # Load and process
    start_time = time.time()
    renderer.load_and_preprocess()
    renderer.compute_edges()
    renderer.compute_target_image()
    renderer.save_target_image(args.save_target)

    # Analyze
    renderer.analyze_with_radon()

    # Select angles based on mode
    print(f"\nMode: {args.mode}")

    if args.mode == 'fixed_6':
        angles, lines_per_angle = renderer.select_angles_fixed_6()
        config = {'angles': angles, 'lines_per_angle': lines_per_angle}

    elif args.mode == '6_plus_refine':
        angles, lines_per_angle = renderer.select_angles_6_plus_refine(
            quality_threshold=args.quality_threshold,
            max_lines=args.max_lines
        )
        config = {'angles': angles, 'lines_per_angle': lines_per_angle,
                 'quality_threshold': args.quality_threshold}

    elif args.mode == 'pure_optimize':
        angles, lines_per_angle = renderer.select_angles_pure_optimize(
            quality_threshold=args.quality_threshold,
            max_lines=args.max_lines
        )
        config = {'angles': angles, 'lines_per_angle': lines_per_angle,
                 'quality_threshold': args.quality_threshold}

    elif args.mode == 'hierarchical':
        h_config = {}
        if args.hierarchical_config:
            print(f"jsorn arg {args.hierarchical_config}")
            h_config = json.loads(args.hierarchical_config)

        angles, lines_per_angle = renderer.select_angles_hierarchical(
            strategy=args.hierarchical_strategy,
            config=h_config
        )
        config = {'strategy': args.hierarchical_strategy,
                 'config': h_config,
                 'angles': angles,
                 'lines_per_angle': lines_per_angle}

    # Render (if not already done in optimization modes)
    if args.mode == 'fixed_6':
        print(f"\nGenerating {len(angles)} angle groups...")
        renderer.render_angles(angles, lines_per_angle)
    elif args.mode == 'hierarchical':
        # Already rendered with residual tracking in select_angles_hierarchical
        pass

    # Save rendered image
    rendered = renderer.render_lines_to_image(renderer.lines)
    rendered_8bit = (rendered * 255).astype(np.uint8)
    Image.fromarray(rendered_8bit).save(args.save_rendered)
    print(f"Saved rendered image to: {args.save_rendered}")

    # Final error
    final_error = renderer.compute_current_error()

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Angles used: {len(angles)}")
    print(f"Total lines: {len(renderer.lines)}")
    print(f"Final error ({args.quality_metric}): {final_error:.6f}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save output
    renderer.save_output(args.output, args.mode, config)
