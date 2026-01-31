import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# ANGLE SELECTION METRICS
# ============================================================================

class AngleSelectionMetric(ABC):
    """Base class for angle selection strategies"""

    @abstractmethod
    def compute_angle_scores(self, image, edges, sinogram_edges, sinogram_intensity,
                            theta, edge_weight, intensity_weight):
        """
        Compute scores for each angle

        Returns:
            scores: Array of scores for each angle in theta
        """
        pass

    @abstractmethod
    def get_name(self):
        """Return metric name"""
        pass


class RadonVarianceMetric(AngleSelectionMetric):
    """Use Radon transform variance (fast, heuristic)"""

    def compute_angle_scores(self, image, edges, sinogram_edges, sinogram_intensity,
                            theta, edge_weight, intensity_weight):
        edge_strength = np.var(sinogram_edges, axis=0)
        intensity_strength = np.var(sinogram_intensity, axis=0)

        combined = (edge_weight * edge_strength +
                   intensity_weight * intensity_strength)
        return combined / (combined.max() + 1e-8)

    def get_name(self):
        return "radon_variance"


class MSEReductionMetric(AngleSelectionMetric):
    """Use actual MSE reduction (slower, more accurate)"""

    def __init__(self, renderer):
        self.renderer = renderer

    def compute_angle_scores(self, image, edges, sinogram_edges, sinogram_intensity,
                            theta, edge_weight, intensity_weight):
        """
        For each angle, estimate MSE reduction if we add lines at that angle
        This is expensive but accurate
        """
        scores = np.zeros(len(theta))

        # Create target image
        target = edge_weight * edges + intensity_weight * image

        # Sample subset of angles for speed
        sample_indices = np.linspace(0, len(theta)-1, min(36, len(theta)), dtype=int)

        for idx in sample_indices:
            angle = theta[idx]

            # Generate a few test lines at this angle
            test_lines = self.renderer.generate_chord_at_angle(angle, num_lines_this_angle=5)

            # Render these lines
            test_image = self.renderer.render_lines_to_image(test_lines)

            # Calculate MSE reduction
            current_mse = np.mean((target - 0) ** 2)  # Assuming starting from blank
            with_lines_mse = np.mean((target - test_image) ** 2)

            scores[idx] = current_mse - with_lines_mse  # Higher = better reduction

        # Interpolate for non-sampled angles
        for i in range(len(theta)):
            if i not in sample_indices:
                # Simple nearest neighbor
                nearest = sample_indices[np.argmin(np.abs(sample_indices - i))]
                scores[i] = scores[nearest] * 0.8  # Slightly lower than sampled

        return scores / (scores.max() + 1e-8)

    def get_name(self):
        return "mse_reduction"


# ============================================================================
# LINE SPACING METRICS
# ============================================================================

class LineSpacingMetric(ABC):
    """Base class for line spacing strategies"""

    @abstractmethod
    def compute_line_positions(self, importance_map, angle_deg, num_lines,
                              circle_radius, image_shape):
        """
        Compute positions for lines at given angle

        Returns:
            positions: Array of positions along perpendicular direction
        """
        pass

    @abstractmethod
    def get_name(self):
        """Return metric name"""
        pass


class EvenSpacingMetric(LineSpacingMetric):
    """Evenly spaced lines"""

    def compute_line_positions(self, importance_map, angle_deg, num_lines,
                              circle_radius, image_shape):
        return np.linspace(-circle_radius * 0.95, circle_radius * 0.95, num_lines)

    def get_name(self):
        return "even_spacing"


class AdaptiveSpacingMetric(LineSpacingMetric):
    """Adaptive spacing based on importance map"""

    def compute_line_positions(self, importance_map, angle_deg, num_lines,
                              circle_radius, image_shape):
        angle_rad = np.deg2rad(angle_deg)
        perp_angle = angle_rad + np.pi / 2
        dx_perp = np.cos(perp_angle)
        dy_perp = np.sin(perp_angle)

        # Sample importance along perpendicular
        n_samples = 200
        t = np.linspace(-circle_radius, circle_radius, n_samples)
        img_center = image_shape[0] / 2

        sample_importance = []
        for ti in t:
            x = img_center + ti * dx_perp
            y = img_center + ti * dy_perp

            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                importance = importance_map[int(y), int(x)]
                sample_importance.append(importance)
            else:
                sample_importance.append(0)

        sample_importance = np.array(sample_importance)

        # Base even spacing
        base_positions = np.linspace(-circle_radius * 0.95, circle_radius * 0.95, num_lines)

        # Add adaptive perturbations
        positions = []
        spacing = 2 * circle_radius * 0.95 / num_lines

        for pos in base_positions:
            idx = int((pos + circle_radius) / (2 * circle_radius) * (n_samples - 1))
            idx = np.clip(idx, 0, len(sample_importance) - 1)

            local_importance = sample_importance[idx]
            offset = (local_importance - 0.5) * spacing * 0.1
            positions.append(pos + offset)

        return np.array(positions)

    def get_name(self):
        return "adaptive_spacing"


class MSEOptimizedSpacingMetric(LineSpacingMetric):
    """Optimize each line position to minimize MSE (very slow but accurate)"""

    def __init__(self, renderer, target_image):
        self.renderer = renderer
        self.target_image = target_image

    def compute_line_positions(self, importance_map, angle_deg, num_lines,
                              circle_radius, image_shape):
        # Start with adaptive spacing as initial guess
        adaptive = AdaptiveSpacingMetric()
        initial_positions = adaptive.compute_line_positions(
            importance_map, angle_deg, num_lines, circle_radius, image_shape
        )

        # For now, just return initial (full optimization would be very slow)
        # TODO: Implement greedy optimization - add lines one at a time at position
        # that most reduces MSE
        return initial_positions

    def get_name(self):
        return "mse_optimized_spacing"


# ============================================================================
# QUALITY METRICS
# ============================================================================

class QualityMetric(ABC):
    """Base class for quality measurement"""

    @abstractmethod
    def compute_error(self, target, generated):
        """
        Compute error between target and generated images

        Returns:
            error: Scalar error value (lower is better)
        """
        pass

    @abstractmethod
    def get_name(self):
        """Return metric name"""
        pass


class MSEQualityMetric(QualityMetric):
    """Mean Squared Error"""

    def compute_error(self, target, generated):
        return np.mean((target - generated) ** 2)

    def get_name(self):
        return "mse"


class MAEQualityMetric(QualityMetric):
    """Mean Absolute Error"""

    def compute_error(self, target, generated):
        return np.mean(np.abs(target - generated))

    def get_name(self):
        return "mae"
