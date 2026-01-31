import numpy as np
import json
from PIL import Image
from skimage.transform import radon
from skimage import filters
from scipy.ndimage import sobel

from .config import PAPER_SIZES
from .metrics import (
    RadonVarianceMetric,
    AdaptiveSpacingMetric,
    MSEQualityMetric,
    MSEReductionMetric,
)


class AdvancedRadonRenderer:
    """Advanced renderer with pluggable metrics and multiple modes"""

    def __init__(self, image_path, num_lines=500, paper_size='A4',
                 pen_width=0.3, padding=10.0,
                 edge_weight=0.75, intensity_weight=0.25,
                 angle_metric=None, spacing_metric=None, quality_metric=None):
        """
        Initialize renderer

        Args:
            image_path: Path to input image
            num_lines: Target number of lines
            paper_size: Paper size
            pen_width: Pen width in mm
            padding: Padding in mm
            edge_weight: Weight for edges (default 0.75)
            intensity_weight: Weight for intensity (default 0.25)
            angle_metric: AngleSelectionMetric instance
            spacing_metric: LineSpacingMetric instance
            quality_metric: QualityMetric instance
        """
        self.image_path = image_path
        self.num_lines = num_lines
        self.paper_size = paper_size
        self.pen_width = pen_width
        self.padding = padding
        self.edge_weight = edge_weight
        self.intensity_weight = intensity_weight

        # Set default metrics if not provided
        self.angle_metric = angle_metric or RadonVarianceMetric()
        self.spacing_metric = spacing_metric or AdaptiveSpacingMetric()
        self.quality_metric = quality_metric or MSEQualityMetric()

        # For MSE reduction metric, pass renderer reference
        if isinstance(self.angle_metric, MSEReductionMetric):
            self.angle_metric.renderer = self

        # State
        self.image = None
        self.edges = None
        self.target_image = None
        self.current_rendered = None
        self.lines = []
        self.analysis = None

    def load_and_preprocess(self):
        """Load and preprocess image"""
        img = Image.open(self.image_path).convert('L')

        paper_w, paper_h = PAPER_SIZES[self.paper_size]
        available_size = min(paper_w, paper_h) - 2 * self.padding
        self.circle_diameter = available_size
        self.circle_radius = available_size / 2
        self.circle_center = (paper_w / 2, paper_h / 2)

        target_size = int(self.circle_diameter)

        # Crop to square
        size = min(img.size)
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Normalize (0=white, 1=black)
        self.image = np.array(img).astype(float) / 255.0
        self.image = 1.0 - self.image

        # Circular mask
        y, x = np.ogrid[:target_size, :target_size]
        center = target_size / 2
        mask = (x - center)**2 + (y - center)**2 <= (center)**2
        self.image = self.image * mask

        print(f"Loaded: {target_size}x{target_size}px, Paper: {self.paper_size}, "
              f"Circle: {self.circle_diameter:.1f}mm")

        return self.image

    def compute_edges(self):
        """Compute edge map"""
        edges_x = filters.sobel_h(self.image)
        edges_y = filters.sobel_v(self.image)
        self.edges = np.sqrt(edges_x**2 + edges_y**2)
        return self.edges

    def compute_target_image(self):
        """Create target image (edge + intensity mixture)"""
        edges_norm = self.edges / (self.edges.max() + 1e-8)
        intensity_norm = self.image / (self.image.max() + 1e-8)

        # Invert intensity BEFORE blending
        # intensity is currently: dark=1, light=0
        # We want: dark areas need lines, so invert to light=1, dark=0
        # Then blend: high values = need lines
        self.target_image = (self.edge_weight * edges_norm +
                            self.intensity_weight * (1.0 - intensity_norm))

        return self.target_image

    def save_target_image(self, filepath):
        """Save target image for inspection"""
        if self.target_image is None:
            self.compute_target_image()

        # Convert to 0-255 and save
        img_8bit = (self.target_image * 255).astype(np.uint8)
        Image.fromarray(img_8bit).save(filepath)
        print(f"Saved target image to: {filepath}")

    def analyze_with_radon(self, num_angles=180):
        """Perform Radon analysis"""
        print(f"\nRadon analysis (angle metric: {self.angle_metric.get_name()})...")

        theta = np.linspace(0., 180., num_angles, endpoint=False)
        sinogram_edges = radon(self.edges, theta=theta, circle=True)
        sinogram_intensity = radon(self.image, theta=theta, circle=True)

        # Use pluggable metric
        angle_scores = self.angle_metric.compute_angle_scores(
            self.image, self.edges, sinogram_edges, sinogram_intensity,
            theta, self.edge_weight, self.intensity_weight
        )

        self.analysis = {
            'theta': theta,
            'angle_scores': angle_scores,
            'sinogram_edges': sinogram_edges,
            'sinogram_intensity': sinogram_intensity
        }

        return self.analysis

    def select_angles_fixed_6(self):
        """Mode 1: Select fixed 6 angles"""
        theta = self.analysis['theta']
        scores = self.analysis['angle_scores']

        # Find top angles ensuring they're well-distributed
        sorted_indices = np.argsort(scores)[::-1]
        selected_angles = []

        for idx in sorted_indices:
            angle = theta[idx]

            if not selected_angles or all(abs(angle - a) > 15 for a in selected_angles):
                selected_angles.append(angle)
                if len(selected_angles) >= 6:
                    break

        # Distribute lines evenly
        lines_per_angle = [self.num_lines // len(selected_angles)] * len(selected_angles)
        lines_per_angle[0] += self.num_lines - sum(lines_per_angle)

        return selected_angles, lines_per_angle

    def select_angles_6_plus_refine(self, quality_threshold=0.01, max_lines=None):
        """Mode 2: Start with 6, add refinement angles"""
        if max_lines is None:
            max_lines = self.num_lines * 2

        # Start with 6
        primary_angles, primary_lines = self.select_angles_fixed_6()

        print(f"Primary 6 angles: {[f'{a:.1f}°' for a in primary_angles]}")

        # Generate initial lines
        self.render_angles(primary_angles, primary_lines)
        current_error = self.compute_current_error()

        print(f"Initial error: {current_error:.6f}")

        # Add refinement angles
        theta = self.analysis['theta']
        scores = self.analysis['angle_scores']
        used_angles = set(primary_angles)

        refinement_angles = []
        refinement_lines = []
        total_lines = sum(primary_lines)

        sorted_indices = np.argsort(scores)[::-1]

        for idx in sorted_indices:
            angle = theta[idx]

            if angle in used_angles:
                continue

            if total_lines >= max_lines:
                break

            # Try adding lines at this angle
            lines_to_add = min(20, max_lines - total_lines)
            test_lines = self.generate_chord_at_angle(angle, lines_to_add)

            # Compute error improvement
            old_error = current_error
            self.add_lines_to_render(test_lines)
            new_error = self.compute_current_error()
            improvement = (old_error - new_error) / old_error

            if improvement > quality_threshold:
                refinement_angles.append(angle)
                refinement_lines.append(lines_to_add)
                total_lines += lines_to_add
                current_error = new_error
                used_angles.add(angle)
                print(f"  Added angle {angle:.1f}° ({lines_to_add} lines): "
                      f"error={new_error:.6f}, improvement={improvement:.2%}")
            else:
                # Remove test lines
                self.lines = self.lines[:-lines_to_add]
                self.current_rendered = None

        all_angles = primary_angles + refinement_angles
        all_lines_per_angle = primary_lines + refinement_lines

        return all_angles, all_lines_per_angle

    def select_angles_pure_optimize(self, quality_threshold=0.01, max_lines=None):
        """Mode 3: Pure optimization - add angles until threshold"""
        if max_lines is None:
            max_lines = self.num_lines

        theta = self.analysis['theta']
        scores = self.analysis['angle_scores']

        selected_angles = []
        lines_per_angle = []
        total_lines = 0
        used_angles = set()

        sorted_indices = np.argsort(scores)[::-1]

        # Initialize with blank
        self.lines = []
        self.current_rendered = None
        current_error = self.compute_current_error()

        print(f"Initial error: {current_error:.6f}")

        for idx in sorted_indices:
            angle = theta[idx]

            if angle in used_angles or total_lines >= max_lines:
                continue

            # Determine lines to add (proportional to remaining budget and score)
            remaining = max_lines - total_lines
            lines_to_add = max(10, min(50, int(remaining * scores[idx] * 0.3)))
            lines_to_add = min(lines_to_add, remaining)

            # Generate and test
            test_lines = self.generate_chord_at_angle(angle, lines_to_add)

            old_error = current_error
            self.add_lines_to_render(test_lines)
            new_error = self.compute_current_error()
            improvement = (old_error - new_error) / (old_error + 1e-8)

            if improvement > quality_threshold or len(selected_angles) < 3:
                selected_angles.append(angle)
                lines_per_angle.append(lines_to_add)
                total_lines += lines_to_add
                current_error = new_error
                used_angles.add(angle)

                print(f"  Angle {angle:.1f}° ({lines_to_add} lines): "
                      f"error={new_error:.6f}, improvement={improvement:.2%}")
            else:
                # Remove test lines
                self.lines = self.lines[:-lines_to_add]
                self.current_rendered = None

                if len(selected_angles) >= 6:
                    print(f"  Improvement below threshold, stopping")
                    break

        print(f"Final: {len(selected_angles)} angles, {total_lines} lines")

        return selected_angles, lines_per_angle

    def select_angles_hierarchical(self, strategy='strength_based', config=None):
        """Mode 4: Hierarchical angle selection with residual tracking"""
        if config is None:
            config = {}

        theta = self.analysis['theta']
        scores = self.analysis['angle_scores']

        if strategy == 'strength_based':
            angles, lines_per_angle = self._hierarchical_strength_based(theta, scores, config)
        elif strategy == 'cumulative_contribution':
            angles, lines_per_angle = self._hierarchical_cumulative(theta, scores, config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Now render with residual tracking
        self._render_with_residual(angles, lines_per_angle)

        return angles, lines_per_angle

    def _render_with_residual(self, angles, lines_per_angle):
        """
        Render angles incrementally, subtracting each angle's contribution
        from the residual target
        """
        if self.target_image is None:
            self.compute_target_image()

        # Initialize residual with target
        residual = self.target_image.copy()

        self.lines = []

        print(f"\nRendering with residual tracking:")

        for i, (angle, n_lines) in enumerate(zip(angles, lines_per_angle)):
            # Generate lines for this angle based on CURRENT residual
            # Update the analysis to use residual instead of original
            old_image = self.image.copy()
            old_edges = self.edges.copy()

            # Temporarily replace with residual for line generation
            # (This makes sinogram-based placement work on what's left to approximate)
            self.image = residual
            self.edges = residual  # Residual shows what needs lines

            # Re-analyze with residual if needed
            # For now, use original sinogram but could re-compute here

            # Generate lines at this angle
            angle_lines = self.generate_chord_at_angle(angle, n_lines)

            # Restore original
            self.image = old_image
            self.edges = old_edges

            # Render these lines
            rendered = self.render_lines_to_image(angle_lines)

            # Subtract from residual
            residual = residual - rendered
            residual = np.maximum(residual, 0)  # Clamp to 0

            # Add to total lines
            self.lines.extend(angle_lines)

            # Compute current error on residual
            error = self.quality_metric.compute_error(residual, np.zeros_like(residual))

            print(f"  Angle {i+1}/{len(angles)}: {angle:.1f}° ({n_lines} lines) - "
                  f"Residual error: {error:.6f}")

        self.current_rendered = None  # Force recompute

    def _hierarchical_strength_based(self, theta, scores, config):
        """Hierarchical: strength-based classification"""
        primary_min = config.get('primary_min', 0.8)
        secondary_min = config.get('secondary_min', 0.5)
        max_primary = config.get('max_primary', 6)
        max_secondary = config.get('max_secondary', 5)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        primary_angles = []
        secondary_angles = []
        tertiary_angles = []

        used_angles = set()

        for idx in sorted_indices:
            angle = theta[idx]
            score = scores[idx]

            # Check angle spacing
            if any(abs(angle - a) < 15 for a in used_angles):
                continue

            if score >= primary_min and len(primary_angles) < max_primary:
                primary_angles.append(angle)
                used_angles.add(angle)
            elif score >= secondary_min and len(secondary_angles) < max_secondary:
                secondary_angles.append(angle)
                used_angles.add(angle)
            elif score >= 0.3 and len(tertiary_angles) < 10:
                tertiary_angles.append(angle)
                used_angles.add(angle)

        # Distribute lines
        total_angles = len(primary_angles) + len(secondary_angles) + len(tertiary_angles)
        if total_angles == 0:
            return [0.0], [self.num_lines]

        # Allocate: 50% to primary, 35% to secondary, 15% to tertiary
        primary_budget = int(self.num_lines * 0.5)
        secondary_budget = int(self.num_lines * 0.35)
        tertiary_budget = self.num_lines - primary_budget - secondary_budget

        lines_per_angle = []

        if primary_angles:
            per_primary = primary_budget // len(primary_angles)
            lines_per_angle.extend([per_primary] * len(primary_angles))

        if secondary_angles:
            per_secondary = secondary_budget // len(secondary_angles)
            lines_per_angle.extend([per_secondary] * len(secondary_angles))

        if tertiary_angles:
            per_tertiary = max(5, tertiary_budget // len(tertiary_angles))
            lines_per_angle.extend([per_tertiary] * len(tertiary_angles))

        # Adjust to match target
        difference = self.num_lines - sum(lines_per_angle)
        if lines_per_angle:
            lines_per_angle[0] += difference

        all_angles = primary_angles + secondary_angles + tertiary_angles

        print(f"Hierarchical (strength): {len(primary_angles)} primary, "
              f"{len(secondary_angles)} secondary, {len(tertiary_angles)} tertiary")

        return all_angles, lines_per_angle

    def _hierarchical_cumulative(self, theta, scores, config):
        """Hierarchical: cumulative contribution classification"""
        primary_percent = config.get('primary_percent', 60)
        secondary_percent = config.get('secondary_percent', 25)
        tertiary_percent = config.get('tertiary_percent', 15)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        # Calculate cumulative contribution
        total_score = scores.sum()
        cumsum = 0

        primary_angles = []
        secondary_angles = []
        tertiary_angles = []
        used_angles = set()

        for idx in sorted_indices:
            angle = theta[idx]
            score = scores[idx]

            # Check spacing
            if any(abs(angle - a) < 15 for a in used_angles):
                continue

            contribution = (score / total_score) * 100
            cumsum += contribution

            if cumsum <= primary_percent:
                primary_angles.append(angle)
            elif cumsum <= primary_percent + secondary_percent:
                secondary_angles.append(angle)
            elif cumsum <= 100:
                tertiary_angles.append(angle)
            else:
                break

            used_angles.add(angle)

        # Distribute lines by contribution
        all_angles = primary_angles + secondary_angles + tertiary_angles

        if not all_angles:
            return [0.0], [self.num_lines]

        lines_per_angle = []

        # Primary gets proportional to contribution
        primary_budget = int(self.num_lines * primary_percent / 100)
        secondary_budget = int(self.num_lines * secondary_percent / 100)
        tertiary_budget = self.num_lines - primary_budget - secondary_budget

        if primary_angles:
            lines_per_angle.extend([primary_budget // len(primary_angles)] * len(primary_angles))
        if secondary_angles:
            lines_per_angle.extend([secondary_budget // len(secondary_angles)] * len(secondary_angles))
        if tertiary_angles:
            lines_per_angle.extend([tertiary_budget // len(tertiary_angles)] * len(tertiary_angles))

        # Adjust
        difference = self.num_lines - sum(lines_per_angle)
        if lines_per_angle:
            lines_per_angle[0] += difference

        print(f"Hierarchical (cumulative): {len(primary_angles)} primary, "
              f"{len(secondary_angles)} secondary, {len(tertiary_angles)} tertiary")

        return all_angles, lines_per_angle

    def generate_chord_at_angle(self, angle_deg, num_lines_this_angle):
        """
        Generate chords at given angle, only where sinogram shows EDGES (transitions)

        Args:
            angle_deg: Angle in degrees (0=horizontal, 90=vertical)
            num_lines_this_angle: Number of parallel chords to generate

        Returns:
            List of line coordinates in mm
        """
        angle_rad = np.deg2rad(angle_deg)
        perp_angle = angle_rad + np.pi / 2
        dx_perp = np.cos(perp_angle)
        dy_perp = np.sin(perp_angle)

        # Get sinogram for this angle to know WHERE structure exists
        if self.analysis is not None and 'sinogram_edges' in self.analysis:
            theta = self.analysis['theta']
            # Find closest theta index
            angle_idx = np.argmin(np.abs(theta - angle_deg))

            # Get sinogram projection at this angle
            sinogram_edges = self.analysis['sinogram_edges'][:, angle_idx]
            sinogram_intensity = self.analysis['sinogram_intensity'][:, angle_idx]

            # Combine with perception weights
            combined_sinogram = (self.edge_weight * sinogram_edges +
                               self.intensity_weight * sinogram_intensity)

            # KEY CHANGE: Apply edge detection to sinogram
            # This finds TRANSITIONS (boundaries) rather than accumulations
            sinogram_gradient = np.abs(sobel(combined_sinogram))

            # Normalize
            sinogram_gradient = sinogram_gradient / (sinogram_gradient.max() + 1e-8)

            # Only place lines where sinogram GRADIENT is HIGH (transitions/edges)
            n_samples = len(sinogram_gradient)

            # Determine threshold: we want to place num_lines_this_angle lines
            # Find threshold that gives us approximately that many positions
            sorted_values = np.sort(sinogram_gradient)[::-1]

            # Ensure we have at least num_lines positions
            threshold_idx = min(int(num_lines_this_angle * 1.5), len(sorted_values) - 1)
            threshold = sorted_values[threshold_idx]

            # Find positions where gradient exceeds threshold
            high_positions_idx = np.where(sinogram_gradient >= threshold)[0]

            # Convert sinogram indices to actual positions in circle coordinates
            # Sinogram spans from -radius to +radius
            sinogram_positions = np.linspace(-self.circle_radius,
                                            self.circle_radius,
                                            n_samples)

            # Get positions where transitions exist
            candidate_positions = sinogram_positions[high_positions_idx]

            # If we have more candidates than needed, sample uniformly from them
            if len(candidate_positions) > num_lines_this_angle:
                indices = np.linspace(0, len(candidate_positions)-1,
                                    num_lines_this_angle, dtype=int)
                positions = candidate_positions[indices]
            else:
                positions = candidate_positions
        else:
            # Fallback: uniform spacing (shouldn't happen in normal flow)
            positions = np.linspace(-self.circle_radius * 0.95,
                                   self.circle_radius * 0.95,
                                   num_lines_this_angle)

        # Generate actual line coordinates
        lines = []
        line_dx = np.cos(angle_rad)
        line_dy = np.sin(angle_rad)

        for pos in positions:
            # Center point of the line in circle coordinates
            cx = pos * dx_perp
            cy = pos * dy_perp

            # Solve for circle intersection
            a = line_dx**2 + line_dy**2
            b = 2 * (cx * line_dx + cy * line_dy)
            c = cx**2 + cy**2 - self.circle_radius**2

            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                continue

            t1 = (-b + np.sqrt(discriminant)) / (2*a)
            t2 = (-b - np.sqrt(discriminant)) / (2*a)

            x1 = cx + t1 * line_dx
            y1 = cy + t1 * line_dy
            x2 = cx + t2 * line_dx
            y2 = cy + t2 * line_dy

            # Convert to paper coordinates
            x1_mm = self.circle_center[0] + x1
            y1_mm = self.circle_center[1] + y1
            x2_mm = self.circle_center[0] + x2
            y2_mm = self.circle_center[1] + y2

            lines.append({
                'x1': float(x1_mm),
                'y1': float(y1_mm),
                'x2': float(x2_mm),
                'y2': float(y2_mm),
                'angle': float(angle_deg)
            })

        return lines

    def render_lines_to_image(self, lines):
        """Render lines to raster image for quality measurement"""
        img = np.zeros_like(self.image)

        # Scale factor from mm to pixels
        scale = self.image.shape[0] / self.circle_diameter

        for line in lines:
            # Convert mm to pixels relative to circle center
            x1 = (line['x1'] - self.circle_center[0]) * scale + self.image.shape[1] / 2
            y1 = (line['y1'] - self.circle_center[1]) * scale + self.image.shape[0] / 2
            x2 = (line['x2'] - self.circle_center[0]) * scale + self.image.shape[1] / 2
            y2 = (line['y2'] - self.circle_center[1]) * scale + self.image.shape[0] / 2

            # Bresenham's line algorithm
            x0, y0 = int(x1), int(y1)
            x1_int, y1_int = int(x2), int(y2)

            dx = abs(x1_int - x0)
            dy = abs(y1_int - y0)
            sx = 1 if x0 < x1_int else -1
            sy = 1 if y0 < y1_int else -1
            err = dx - dy

            while True:
                if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
                    img[y0, x0] = min(1.0, img[y0, x0] + 0.1)  # Accumulate

                if x0 == x1_int and y0 == y1_int:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

        return img

    def render_angles(self, angles, lines_per_angle):
        """Render all angles"""
        self.lines = []
        for angle, n_lines in zip(angles, lines_per_angle):
            angle_lines = self.generate_chord_at_angle(angle, n_lines)
            self.lines.extend(angle_lines)

        self.current_rendered = None  # Force recompute
        return self.lines

    def add_lines_to_render(self, new_lines):
        """Add lines to current render"""
        self.lines.extend(new_lines)
        self.current_rendered = None  # Force recompute

    def compute_current_error(self):
        """Compute error of current lines vs target"""
        if self.current_rendered is None:
            self.current_rendered = self.render_lines_to_image(self.lines)

        if self.target_image is None:
            self.compute_target_image()

        error = self.quality_metric.compute_error(self.target_image, self.current_rendered)
        return error

    def save_output(self, filepath, mode, config=None):
        """Save output JSON"""
        paper_w, paper_h = PAPER_SIZES[self.paper_size]

        output = {
            'metadata': {
                'source_image': str(self.image_path),
                'paper_size': self.paper_size,
                'paper_dimensions_mm': {'width': paper_w, 'height': paper_h},
                'circle': {
                    'diameter_mm': float(self.circle_diameter),
                    'center_x_mm': float(self.circle_center[0]),
                    'center_y_mm': float(self.circle_center[1]),
                    'radius_mm': float(self.circle_radius)
                },
                'padding_mm': float(self.padding),
                'pen_width_mm': float(self.pen_width),
                'total_lines': len(self.lines)
            },
            'rendering': {
                'mode': mode,
                'config': config or {},
                'edge_weight': self.edge_weight,
                'intensity_weight': self.intensity_weight
            },
            'metrics': {
                'angle_selection': self.angle_metric.get_name(),
                'line_spacing': self.spacing_metric.get_name(),
                'quality': self.quality_metric.get_name(),
                'final_error': float(self.compute_current_error())
            },
            'lines': self.lines
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Saved output to: {filepath}")
        return output
