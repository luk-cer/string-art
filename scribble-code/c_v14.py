from PIL import Image
from cv2 import magnitude
import numpy as np 
from scipy.ndimage import sobel, convolve, binary_dilation
from matplotlib import pyplot as plt 
from scipy.sparse import coo_matrix
import numba
from functools import lru_cache
import tqdm


class SketchConverter:
    """Converts images to artistic line drawings using gradient-based tracing."""
    
    def __init__(self, image_path, 
                 gradient_threshold=0.05,
                 intensity_threshold=0.2,
                 trace_kernel_size=4,
                 trace_kernel_sigma=2,
                 max_lines=10000,
                 min_magnitude_ratio=0.0001):
        """
        Initialize the sketch converter.
        
        Args:
            image_path: Path to input image
            gradient_threshold: Minimum gradient magnitude to consider (0-1)
            intensity_threshold: Threshold for probability mask (0-1)
            trace_kernel_size: Size parameter for Gaussian kernel (d value)
            trace_kernel_sigma: Sigma for Gaussian smoothing
            max_lines: Maximum number of lines to trace
            min_magnitude_ratio: Stop when remaining magnitude is below this ratio
        """
        self.image_path = image_path
        self.gradient_threshold = gradient_threshold
        self.intensity_threshold = intensity_threshold
        self.d = trace_kernel_size
        self.kernel_sigma = trace_kernel_sigma
        self.max_lines = max_lines
        self.min_magnitude_ratio = min_magnitude_ratio
        
        # Cache for kernels
        self._kernel_cache = {}
        
        # Will be populated during processing
        self.image = None
        self.pix = None
        self.ipix = None
        self.edges_x = None
        self.edges_y = None
        self.magnitude = None
        self.gx = None
        self.gy = None
        self.lines = []
        
    def load_and_preprocess(self):
        """Load image and compute gradients."""
        self.image = Image.open(self.image_path)
        
        # Convert to grayscale and normalize
        self.pix = np.array(self.image.convert('L'), dtype=float) / 255
        self.ipix = 1 - self.pix
        
        # Compute edges
        self.edges_x = sobel(self.pix, 0)
        self.edges_y = sobel(self.pix, 1)
        
        # Compute magnitude
        self.magnitude = np.sqrt(self.edges_x**2 + self.edges_y**2)
        self.magnitude = self._rescale(self.magnitude)
        self.magnitude[self.magnitude < self.gradient_threshold] = 0
        self.edges_x[self.magnitude == 0] = 0
        self.edges_y[self.magnitude == 0] = 0
        
        # Smooth gradients
        w2 = self._get_kernel(2 * self.d + 1, self.kernel_sigma)
        self.gx = convolve(self.edges_x, w2)
        self.gy = convolve(self.edges_y, w2)
        
        # Pre-compute perceptual intensity map for adaptive cleanup
        gaussian_kernel = self._get_kernel(2 * self.d + 1, self.kernel_sigma)
        ipix_smoothed = convolve(self.ipix, gaussian_kernel)
        
        # Apply gamma correction for perceptual intensity
        self.ipix_perceptual = ipix_smoothed ** (1/2.2)
        
        # Pre-compute variance map for texture awareness
        ipix_squared = convolve(self.ipix ** 2, gaussian_kernel)
        self.ipix_variance = np.sqrt(np.maximum(0, ipix_squared - ipix_smoothed ** 2))
        # Normalize variance to 0-1 range
        if self.ipix_variance.max() > 0:
            self.ipix_variance = self.ipix_variance / self.ipix_variance.max()
        
        # Normalize magnitude for probabilistic sampling
        magnitude_sum = self.magnitude.sum()
        if magnitude_sum > 0:
            self.magnitude /= magnitude_sum
            
        return self
    
    def trace_main_lines(self, show_progress=True):
        """Trace main contour lines following gradients."""
        points = np.prod(self.magnitude.shape)
        initial_magnitude = self.magnitude.sum()
        
        iterator = range(self.max_lines)
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Tracing main lines")
        
        for i in iterator:
            # Early termination
            remaining_ratio = np.sum(self.magnitude > 0) / initial_magnitude
            if remaining_ratio < self.min_magnitude_ratio:
                print(f"Stopping tracing ITERATION {i}: remaining magnitude {remaining_ratio} below threshold {self.min_magnitude_ratio}.")
                break
            
            try:
                # Pick starting point
                px, py = self._pick_point(self.magnitude, prob=True)
                
                # Skip very light areas
                if self.ipix[px, py] < 0.0001:
                    self._cleanup_point(px, py, radius=3)
                    continue
                
                # Trace in both directions and combine into single line
                line_forward = self._trace_line(px, py, angle=90)
                line_backward = self._trace_line(px, py, angle=270)
                
                # Combine: backward (reversed, without start point) + forward
                combined_line = list(reversed(line_backward[1:])) + line_forward
                
                self.lines.append(combined_line)
                
                # Clean up traced areas
                self._cleanup_lines([combined_line], radius=self.d * 1.5)
                if i%500 ==0 :
                    plt.imshow(self.magnitude)
                    plt.show()
                
            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                break
        
        # Filter out short lines
        self.lines = [l for l in self.lines if len(l) > 1]
        return self
    
    def add_crosshatching(self, min_line_length=5, sample_interval=10, 
                         max_hatch_length=50, hatch_angles=None, 
                         adaptive_spacing=True, show_progress=True):
        """
        Add cross-hatching lines for shading with improved aesthetics.
        
        Args:
            min_line_length: Minimum length of main lines to add hatching to
            sample_interval: Spacing between hatch lines along main lines
            max_hatch_length: Maximum length for hatching lines (prevents long warps)
            hatch_angles: List of (angle, min_intensity) tuples for hatching directions
            adaptive_spacing: If True, vary spacing based on darkness
            show_progress: Show progress bar
        """
        if hatch_angles is None:
            # Default: fewer angles, more controlled
            hatch_angles = [
                (225, 0.5),   # Diagonal down-left
                (135, 0.7),   # Diagonal down-right (for cross-hatch)
            ]
        
        # Use larger kernel for smoother cross-hatching
        w2_large = self._get_kernel(10 * 2 * self.d + 1, 6)
        gx_smooth = convolve(self.edges_x, w2_large)
        gy_smooth = convolve(self.edges_y, w2_large)
        
        # Filter long lines
        lines_long = [l for l in self.lines if len(l) > min_line_length]
        
        new_lines = []
        
        iterator = lines_long
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Adding cross-hatching")
        
        for line in iterator:
            for i, point in enumerate(line):
                # Adaptive spacing based on darkness
                if adaptive_spacing:
                    px, py = point
                    darkness = self.ipix[px, py]
                    # Darker areas get denser hatching
                    current_interval = int(sample_interval * (1.5 - darkness))
                else:
                    current_interval = sample_interval
                
                if i % current_interval == 0:
                    px, py = point
                    
                    # Trace hatching lines at specified angles
                    for angle, min_intensity in hatch_angles:
                        hatch_line = self._trace_line(px, py, angle=angle, 
                                                     max_steps=max_hatch_length, 
                                                     min_intensity=min_intensity, 
                                                     gx=gx_smooth, gy=gy_smooth)
                        
                        # Only add if line is meaningful length
                        if len(hatch_line) > 3:
                            new_lines.append(hatch_line)
        
        self.lines.extend(new_lines)
        return self
    
    def add_stippling(self, density=0.001, min_intensity=0.3, show_progress=True):
        """
        Add stippling (dots) for texture in darker areas.
        
        Args:
            density: Proportion of pixels to potentially stipple
            min_intensity: Minimum darkness to add stipple
            show_progress: Show progress bar
        """
        h, w = self.ipix.shape
        num_points = int(h * w * density)
        
        points = []
        
        iterator = range(num_points)
        if show_progress:
            iterator = tqdm.tqdm(iterator, desc="Adding stippling")
        
        for _ in iterator:
            # Sample based on darkness
            px, py = self._pick_point(self.ipix, prob=True)
            
            if self.ipix[px, py] > min_intensity:
                # Create a small dot (single point or tiny cluster)
                points.append([(px, py)])
                # Optionally add tiny neighbors for slightly bigger dots
                if self.ipix[px, py] > 0.7:
                    if px + 1 < h:
                        points.append([(px + 1, py)])
                    if py + 1 < w:
                        points.append([(px, py + 1)])
        
        self.lines.extend(points)
        return self
    
    def smooth_lines(self, window_size=3):
        """
        Smooth lines using a moving average to reduce jitter.
        
        Args:
            window_size: Size of smoothing window
        """
        smoothed_lines = []
        
        for line in self.lines:
            if len(line) <= window_size:
                smoothed_lines.append(line)
                continue
            
            smoothed = []
            for i in range(len(line)):
                start = max(0, i - window_size // 2)
                end = min(len(line), i + window_size // 2 + 1)
                
                window = line[start:end]
                avg_x = int(np.mean([p[0] for p in window]))
                avg_y = int(np.mean([p[1] for p in window]))
                smoothed.append((avg_x, avg_y))
            
            smoothed_lines.append(smoothed)
        
        self.lines = smoothed_lines
        return self
    
    def filter_short_lines(self, min_length=3):
        """Remove very short lines that create noise."""
        self.lines = [l for l in self.lines if len(l) >= min_length]
        return self
    
    def render(self, use_fast_render=True):
        """
        Render lines to image array.
        
        Args:
            use_fast_render: If True, use optimized numpy rendering instead of sparse matrices
        """
        if use_fast_render:
            return self._render_fast()
        else:
            return self._render_sparse()
    
    def _render_fast(self):
        """Fast rendering using direct numpy indexing."""
        result = np.zeros(shape=self.pix.shape, dtype=np.uint8)
        
        # Collect all points from all lines at once
        all_x = []
        all_y = []
        
        for line in self.lines:
            if len(line) > 0:
                x_coords, y_coords = zip(*line)
                all_x.extend(x_coords)
                all_y.extend(y_coords)
        
        # Convert to numpy arrays
        all_x = np.array(all_x, dtype=np.int32)
        all_y = np.array(all_y, dtype=np.int32)
        
        # Clip to image bounds (safety check)
        all_x = np.clip(all_x, 0, self.pix.shape[0] - 1)
        all_y = np.clip(all_y, 0, self.pix.shape[1] - 1)
        
        # Set all points to 1 in single operation
        result[all_x, all_y] = 1
        
        return result.astype(float)
    
    def _render_sparse(self):
        """Original sparse matrix rendering (slower but kept for compatibility)."""
        result = np.zeros(shape=self.pix.shape)
        
        for line in self.lines:
            if len(line) > 1:
                try:
                    values = np.ones(len(line))
                    x_line, y_line = zip(*line)
                    line_image = coo_matrix((values, (x_line, y_line)), 
                                           shape=self.pix.shape).todense()
                    result += line_image
                except Exception as e:
                    print(f"Error rendering line: {e}")
        
        result[result > 0] = 1.0
        return result
    
    def show(self, title="Sketch"):
        """Display the rendered sketch."""
        sketch = self.render()
        plt.figure(figsize=(10, 10))
        plt.imshow(sketch, cmap='Greys')
        plt.title(title)
        plt.tight_layout()
        plt.axis('off')
        plt.show()
    
    def save(self, output_path):
        """Save the sketch to file."""
        sketch = self.render()
        sketch_uint8 = (sketch * 255).astype(np.uint8)
        Image.fromarray(sketch_uint8, mode='L').save(output_path)
    
    # ========== Helper Methods ==========
    
    def _get_kernel(self, size, sigma):
        """Get or create cached Gaussian kernel."""
        key = (size, sigma)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._gaussian_kernel_2d(size, sigma)
        return self._kernel_cache[key]
    
    @staticmethod
    def _gaussian_kernel_2d(size, sigma):
        """Create a 2D Gaussian kernel."""
        if size % 2 == 0:
            size += 1
        
        x = np.arange(-(size // 2), size // 2 + 1)
        y = np.arange(-(size // 2), size // 2 + 1)
        x, y = np.meshgrid(x, y)
        
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
    
    @staticmethod
    def _rescale(X):
        """Rescale array to 0-1 range."""
        X = X.copy()
        X -= X.min()
        if X.max() > 0:
            X /= X.max()
        return X
    
    def _pick_point(self, prob_mask, prob=True):
        """Pick a point based on probability mask or maximum."""
        if prob:
            p = prob_mask / prob_mask.sum() if prob_mask.sum() > 0 else prob_mask
            i = np.random.choice(np.arange(p.size), p=p.ravel())
            sample = np.unravel_index(i, p.shape)
        else:
            arg_max = np.where(prob_mask == np.max(prob_mask))
            sample = int(arg_max[0][0]), int(arg_max[1][0])
        return sample
    
    def _trace_line(self, px, py, angle=90, max_steps=1000, min_intensity=0, gx=None, gy=None):
        """Trace a line following the gradient field."""
        if gx is None:
            gx = self.gx
        if gy is None:
            gy = self.gy
            
        line = [(int(px), int(py))]
        step = 0
        
        while (-1 < px < self.pix.shape[0] - 1 and 
               -1 < py < self.pix.shape[1] - 1 and
               len(line) == len(set(line)) and
               step < max_steps):
            
            cx, cy = int(px), int(py)
            
            # Get gradient direction
            g_x, g_y = self._gradient(cx, cy, gx, gy, angle)
            
            # Check intensity in direction of gradient
            region_int = self._slice_array(self.ipix, cx, cy, self.d)
            pint = self._direction_intensity(region_int, [g_x, g_y])
            
            if pint <= min_intensity:
                break
            
            # Normalize gradient for stepping
            if max(abs(g_x), abs(g_y)) == abs(g_x):
                if g_x == 0:
                    break
                g_y = np.sign(g_y) * abs(g_y / g_x)
                g_x = np.sign(g_x)
            else:
                if g_y == 0:
                    break
                g_x = np.sign(g_x) * abs(g_x / g_y)
                g_y = np.sign(g_y)
            
            px = px + g_x
            py = py + g_y
            
            if px < 0 or py < 0:
                break
            
            line.append((int(px), int(py)))
            step += 1
        
        return line
    
    @staticmethod
    def _gradient(x, y, g_x, g_y, angle=90):
        """Get gradient at point rotated by angle."""
        rad = angle * np.pi / 180.0
        crad = np.cos(rad)
        srad = np.sin(rad)
        gx = g_x[x, y]
        gy = g_y[x, y]
        return gx * crad + gy * srad, -gx * srad + gy * crad
    
    @staticmethod
    def _direction_intensity(region, direction):
        """Calculate intensity in a given direction."""
        d = (max(region.shape) - 1) / 2
        x = np.arange(-d, d + 1, dtype=np.float64)
        w = np.stack(np.meshgrid(x, x))
        wn = np.linalg.norm(w, axis=0)
        wn[wn == 0] = 1.0
        w /= wn
        w = np.einsum('kij,k->ij', w, direction)
        w[w < 0] = 0
        if w.sum() > 0:
            w /= w.sum()
        return np.multiply(region, w).sum()
    
    @staticmethod
    def _slice_array(arr, x, y, d):
        """Slice array around point with wrapping."""
        xr = range(x - d, x + d + 1)
        yr = range(y - d, y + d + 1)
        return arr.take(xr, mode='wrap', axis=0).take(yr, mode='wrap', axis=1)
    
    def _cleanup_point(self, px, py, radius=3):
        """Clean up magnitude around a single point using vectorized circular mask."""
        adaptive_radius = self._calculate_adaptive_radius(px, py, radius)
        coords_x, coords_y = self._generate_circle_coords(px, py, adaptive_radius)
        
        if len(coords_x) > 0:
            coords_x = np.clip(coords_x, 0, self.magnitude.shape[0] - 1)
            coords_y = np.clip(coords_y, 0, self.magnitude.shape[1] - 1)
            self.magnitude[coords_x, coords_y] = 0
    
    def _cleanup_lines(self, lines, radius=5):
        """Clean up magnitude, gx, gy around traced lines using vectorized adaptive cleanup."""
        all_x = []
        all_y = []
        
        for line in lines:
            for x, y in line:
                # Get adaptive radius based on pre-computed perceptual intensity and variance
                adaptive_radius = self._calculate_adaptive_radius(x, y, radius)
                
                # Generate circular mask coordinates for this point
                coords_x, coords_y = self._generate_circle_coords(x, y, adaptive_radius)
                
                # Collect coordinates
                all_x.extend(coords_x)
                all_y.extend(coords_y)
        
        # Convert to numpy arrays and clip to image bounds
        if len(all_x) > 0:
            all_x = np.array(all_x, dtype=np.int32)
            all_y = np.array(all_y, dtype=np.int32)
            
            # Clip to valid range
            all_x = np.clip(all_x, 0, self.magnitude.shape[0] - 1)
            all_y = np.clip(all_y, 0, self.magnitude.shape[1] - 1)
            
            # Apply cleanup in single operation
            self.magnitude[all_x, all_y] = 0
            self.gx[all_x, all_y] = 0
            self.gy[all_x, all_y] = 0
    
    def _calculate_adaptive_radius(self, x, y, base_radius):
        """
        Calculate adaptive cleanup radius using pre-computed perceptual intensity and variance.
        Lighter areas get larger radius (sparser lines).
        Darker areas get smaller radius (denser lines).
        High variance areas preserve more detail.
        """
        # Bounds check
        if x < 0 or x >= self.ipix_perceptual.shape[0] or y < 0 or y >= self.ipix_perceptual.shape[1]:
            return base_radius
        
        # Get pre-computed values
        perceptual_intensity = self.ipix_perceptual[x, y]
        variance = self.ipix_variance[x, y]
        
        # Combine intensity and variance (Option B from discussion)
        # High variance reduces the intensity effect to preserve detail
        adaptive_factor = perceptual_intensity * (1 - 0.3 * variance)
        
        # Scale radius inversely with adaptive factor
        # Darker areas (high adaptive_factor) → smaller radius → denser lines
        # Lighter areas (low adaptive_factor) → larger radius → sparser lines
        scale_factor = 2.0 - 1.5 * adaptive_factor
        scaled_radius = int(base_radius * scale_factor)
        
        # Clamp to reasonable range
        return max(1, min(scaled_radius, base_radius * 3))
    
    def _generate_circle_coords(self, cx, cy, radius):
        """
        Generate coordinates for a circular mask around a center point.
        
        Returns:
            Two lists: x_coords, y_coords
        """
        # Create coordinate grids for circle
        y_offsets, x_offsets = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle_mask = x_offsets**2 + y_offsets**2 <= radius**2
        
        # Get relative coordinates within circle
        circle_x, circle_y = np.where(circle_mask)
        
        # Convert to absolute coordinates
        actual_x = cx + circle_x - radius
        actual_y = cy + circle_y - radius
        
        return actual_x.tolist(), actual_y.tolist()
    
    def _add_circular_mask(self, mask, cx, cy, radius):
        """
        Add a circular region to the mask.
        
        Args:
            mask: Boolean mask array to modify
            cx, cy: Circle center
            radius: Circle radius
        """
        mx, my = mask.shape
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = x_grid**2 + y_grid**2 <= radius**2
        
        # Calculate bounds
        x_min = max(0, cx - radius)
        x_max = min(mx, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(my, cy + radius + 1)
        
        # Adjust circle mask for edges
        mask_x_min = radius - (cx - x_min)
        mask_x_max = radius + (x_max - cx)
        mask_y_min = radius - (cy - y_min)
        mask_y_max = radius + (y_max - cy)
        
        adjusted_circle = circle[mask_x_min:mask_x_max, mask_y_min:mask_y_max]
        
        # Apply circular mask
        if adjusted_circle.shape == mask[x_min:x_max, y_min:y_max].shape:
            mask[x_min:x_max, y_min:y_max] |= adjusted_circle
    
    def _add_circular_mask(self, mask, cx, cy, radius):
        """
        Add a circular region to the mask.
        
        Args:
            mask: Boolean mask array to modify
            cx, cy: Circle center
            radius: Circle radius
        """
        mx, my = mask.shape
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = x_grid**2 + y_grid**2 <= radius**2
        
        # Calculate bounds
        x_min = max(0, cx - radius)
        x_max = min(mx, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(my, cy + radius + 1)
        
        # Adjust circle mask for edges
        mask_x_min = radius - (cx - x_min)
        mask_x_max = radius + (x_max - cx)
        mask_y_min = radius - (cy - y_min)
        mask_y_max = radius + (y_max - cy)
        
        adjusted_circle = circle[mask_x_min:mask_x_max, mask_y_min:mask_y_max]
        
        # Apply circular mask
        if adjusted_circle.shape == mask[x_min:x_max, y_min:y_max].shape:
            mask[x_min:x_max, y_min:y_max] |= adjusted_circle
    
    def _calc_intensity(self, x, y, d):
        """Calculate intensity-based cleanup distances (legacy - kept for compatibility)."""
        mx, my = self.ipix.shape
        xdp = max(min(x + d + 1, mx), 1)
        xdn = max(x - d, 0)
        ydp = max(min(y + d + 1, my), 1)
        ydn = max(y - d, 0)
        u = 0
        
        try:
            # Ensure slices are valid
            if xdp > x and ydp > y:
                ixpyp = int(u + (1 - np.max(self.ipix[x:xdp, y:ydp])) * (d - u))
            else:
                ixpyp = d
                
            if xdp > x and y >= ydn:
                ixpyn = int(u + (1 - np.max(self.ipix[x:xdp, ydn:y+1])) * (d - u))
            else:
                ixpyn = d
                
            if x >= xdn and ydp > y:
                ixnyp = int(u + (1 - np.max(self.ipix[xdn:x+1, y:ydp])) * (d - u))
            else:
                ixnyp = d
                
            if x >= xdn and y >= ydn:
                ixnyn = int(u + (1 - np.max(self.ipix[xdn:x+1, ydn:y+1])) * (d - u))
            else:
                ixnyn = d
        except Exception as e:
            ixpyp = ixpyn = ixnyp = ixnyn = d
        
        return ixpyp, ixpyn, ixnyp, ixnyn


# Example usage
if __name__ == "__main__":
    # Create converter
    converter = SketchConverter(
        'images/zuzka.jpg',
        gradient_threshold=0.05,
        intensity_threshold=0.2,
        trace_kernel_size=4,
        trace_kernel_sigma=2,
        max_lines=10000,
        min_magnitude_ratio=0.00001
    )
    
    # Process
    converter.load_and_preprocess()
    converter.trace_main_lines(show_progress=True)
    
    # Option 1: Controlled cross-hatching (cleaner, less warping)
    #converter.add_crosshatching(
    #    min_line_length=2, 
    #    sample_interval=10,  # Wider spacing reduces density
    #    max_hatch_length=100,  # Shorter lines prevent long warps
    #    hatch_angles=[(225, 0.6), (135, 0.75)],  # Fewer angles, higher thresholds
    #    adaptive_spacing=True
    #)
    
    # Option 2: Add stippling instead of/in addition to hatching
    # converter.add_stippling(density=0.001, min_intensity=0.5)
    
    # Optional: Smooth lines to reduce jitter
    #converter.smooth_lines(window_size=3)
    
    # Optional: Remove tiny fragments
    #converter.filter_short_lines(min_length=5)
    
    # Display and save
    converter.show()
    converter.save('output_sketch.jpg')
    
    print(f"Generated {len(converter.lines)} lines")