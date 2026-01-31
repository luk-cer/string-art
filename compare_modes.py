#!/usr/bin/env python3
"""
Compare outputs from different rendering modes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys


def load_and_display_comparison(json_files, labels=None, target_image_path=None):
    """
    Display comparison of multiple rendering modes
    
    Args:
        json_files: List of JSON output files
        labels: Optional labels for each mode
        target_image_path: Optional path to target image
    """
    if labels is None:
        labels = [f"Mode {i+1}" for i in range(len(json_files))]
    
    n_modes = len(json_files)
    n_cols = min(3, n_modes)
    n_rows = (n_modes + n_cols - 1) // n_cols
    
    if target_image_path:
        n_rows += 1  # Extra row for target
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Show target image if provided
    if target_image_path:
        target_img = Image.open(target_image_path)
        axes[0, 0].imshow(target_img, cmap='gray')
        axes[0, 0].set_title('Target Image\n(Edge + Intensity Mix)', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Hide extra target row slots
        for col in range(1, n_cols):
            axes[0, col].axis('off')
        
        start_row = 1
    else:
        start_row = 0
    
    # Load and display each mode
    for i, (json_file, label) in enumerate(zip(json_files, labels)):
        row = start_row + (i // n_cols)
        col = i % n_cols
        ax = axes[row, col]
        
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        lines = data['lines']
        metrics = data.get('metrics', {})
        rendering = data.get('rendering', {})
        
        # Draw lines
        paper_w = metadata['paper_dimensions_mm']['width']
        paper_h = metadata['paper_dimensions_mm']['height']
        
        ax.set_xlim(0, paper_w)
        ax.set_ylim(0, paper_h)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_facecolor('white')
        
        for line in lines:
            ax.plot([line['x1'], line['x2']], [line['y1'], line['y2']],
                   'k-', linewidth=0.3, alpha=0.8)
        
        # Title with stats
        mode = rendering.get('mode', 'unknown')
        n_lines = metadata['total_lines']
        
        # Count unique angles
        angles = set(line['angle'] for line in lines)
        n_angles = len(angles)
        
        error = metrics.get('final_error', 0)
        
        title = f"{label}\n"
        title += f"{mode} | {n_lines} lines | {n_angles} angles\n"
        title += f"Error: {error:.6f}"
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    total_plots = n_modes + (1 if target_image_path else 0)
    for i in range(total_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if row < n_rows:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def print_comparison_table(json_files, labels=None):
    """Print comparison table of metrics"""
    if labels is None:
        labels = [f"Mode {i+1}" for i in range(len(json_files))]
    
    print("\n" + "="*80)
    print("MODE COMPARISON")
    print("="*80)
    
    # Header
    print(f"{'Mode':<20} {'Lines':>8} {'Angles':>8} {'Error':>12} {'Time':>8}")
    print("-"*80)
    
    for json_file, label in zip(json_files, labels):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        metrics = data.get('metrics', {})
        rendering = data.get('rendering', {})
        
        mode = rendering.get('mode', 'unknown')
        n_lines = metadata['total_lines']
        
        lines = data['lines']
        angles = set(line['angle'] for line in lines)
        n_angles = len(angles)
        
        error = metrics.get('final_error', 0)
        
        print(f"{label:<20} {n_lines:>8} {n_angles:>8} {error:>12.6f} {'N/A':>8}")
    
    print("="*80)
    
    # Detailed breakdown
    print("\nDETAILED BREAKDOWN:")
    print("-"*80)
    
    for json_file, label in zip(json_files, labels):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        rendering = data.get('rendering', {})
        metrics = data.get('metrics', {})
        
        print(f"\n{label}:")
        print(f"  Mode: {rendering.get('mode', 'unknown')}")
        print(f"  Edge weight: {rendering.get('edge_weight', 0):.2f}")
        print(f"  Intensity weight: {rendering.get('intensity_weight', 0):.2f}")
        print(f"  Angle selection: {metrics.get('angle_selection', 'N/A')}")
        print(f"  Line spacing: {metrics.get('line_spacing', 'N/A')}")
        print(f"  Quality metric: {metrics.get('quality', 'N/A')}")
        
        if 'config' in rendering:
            config = rendering['config']
            if 'quality_threshold' in config:
                print(f"  Quality threshold: {config['quality_threshold']}")
            if 'strategy' in config:
                print(f"  Hierarchical strategy: {config['strategy']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare rendering mode outputs')
    parser.add_argument('json_files', nargs='+', help='JSON output files to compare')
    parser.add_argument('--labels', nargs='+', help='Labels for each mode')
    parser.add_argument('--target', help='Path to target image')
    parser.add_argument('--save', help='Save comparison figure')
    
    args = parser.parse_args()
    
    # Print table
    print_comparison_table(args.json_files, args.labels)
    
    # Create visualization
    fig = load_and_display_comparison(args.json_files, args.labels, args.target)
    
    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved comparison to: {args.save}")
    
    plt.show()


if __name__ == '__main__':
    main()
