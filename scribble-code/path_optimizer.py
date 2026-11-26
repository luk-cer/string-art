import numpy as np
from scipy.spatial import KDTree

class PathOptimizer:
    def __init__(self, iterations=3, window=20):
        """
        iterations: Number of 2-Opt passes to perform.
        window: How far ahead to look for 2-Opt swaps (prevents O(N^2) slowdown).
        """
        self.iterations = iterations
        self.window = window

    def optimize(self, paths):
        """Main entry point: Greedy KD-Tree Sort followed by Local 2-Opt."""
        if len(paths) < 2:
            return paths
        
        # 1. Global Greedy Sort using KD-Tree
        sorted_paths = self._kd_tree_sort(paths)
        
        # 2. Local Refinement using Windowed 2-Opt
        refined_paths = self._refine_2opt(sorted_paths)
        
        return refined_paths

    def _kd_tree_sort(self, paths):
        # Prepare start and end points for all paths
        # Format: [x, y, path_index, point_type (0=start, 1=end)]
        points_data = []
        for idx, segments in enumerate(paths):
            p_start = self._get_start(segments[0])
            p_end = self._get_end(segments[-1])
            points_data.append([p_start[0], p_start[1], idx, 0])
            points_data.append([p_end[0], p_end[1], idx, 1])
        
        points_data = np.array(points_data)
        tree = KDTree(points_data[:, :2])
        
        visited = [False] * len(paths)
        ordered = []
        
        # Start at the first path in the list
        curr_idx = 0
        visited[curr_idx] = True
        ordered.append(paths[curr_idx])
        curr_pos = self._get_end(paths[curr_idx][-1])

        for _ in range(len(paths) - 1):
            # Query the closest 100 points
            dist, indices = tree.query(curr_pos, k=min(100, len(points_data)))
            
            found_next = False
            for tree_idx in indices:
                p_idx = int(points_data[tree_idx, 2])
                is_end_point = bool(points_data[tree_idx, 3])
                
                if not visited[p_idx]:
                    next_path = paths[p_idx]
                    # If the closest point is the end of the next path, flip it
                    if is_end_point:
                        next_path = self._reverse_path(next_path)
                    
                    ordered.append(next_path)
                    visited[p_idx] = True
                    curr_pos = self._get_end(next_path[-1])
                    found_next = True
                    break
            
            # Jump to nearest unvisited if cluster search failed
            if not found_next:
                for i, v in enumerate(visited):
                    if not v:
                        ordered.append(paths[i])
                        visited[i] = True
                        curr_pos = self._get_end(paths[i][-1])
                        break
        return ordered

    def _refine_2opt(self, paths):
        n = len(paths)
        refined = list(paths)
        
        for _ in range(self.iterations):
            improved = False
            for i in range(1, n - 2):
                # Search only within the local window
                for j in range(i + 1, min(i + self.window, n - 1)):
                    
                    # Distance: End(i-1)->Start(i) + End(j)->Start(j+1)
                    d_curr = self._dist(self._get_end(refined[i-1][-1]), self._get_start(refined[i][0])) + \
                             self._dist(self._get_end(refined[j][-1]), self._get_start(refined[j+1][0]))
                    
                    # Distance: End(i-1)->End(j) + Start(i)->Start(j+1)
                    d_swap = self._dist(self._get_end(refined[i-1][-1]), self._get_end(refined[j][-1])) + \
                             self._dist(self._get_start(refined[i][0]), self._get_start(refined[j+1][0]))
                    
                    if d_swap < d_curr:
                        # Reverse segment between i and j and all internal paths
                        sub_section = refined[i:j+1]
                        refined[i:j+1] = [self._reverse_path(p) for p in reversed(sub_section)]
                        improved = True
            if not improved:
                break
        return refined

    # --- Helpers ---
    def _get_start(self, seg):
        return np.array(seg["points"][0] if seg["type"] == "line" else seg["start"])

    def _get_end(self, seg):
        return np.array(seg["points"][-1] if seg["type"] == "line" else seg["end"])

    def _dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _reverse_path(self, segments):
        rev = []
        for s in reversed(segments):
            if s["type"] == "line":
                rev.append({"type": "line", "points": s["points"][::-1]})
            else:
                rev.append({
                    "type": "arc", "start": s["end"], "end": s["start"],
                    "center": s["center"], "radius": s["radius"]
                })
        return rev
    
    import numpy as np
from scipy.spatial import KDTree

class PathOptimizerV2:
    """
    Optimizes the sequence and direction of drawing paths to minimize 
    'pen-up' travel time. Combines a global Greedy KD-Tree search with 
    a local Windowed 2-Opt refinement.
    """
    def __init__(self, iterations=3, window=20):
        """
        :param iterations: Max passes for 2-Opt refinement.
        :param window: Limit for 2-Opt look-ahead to maintain O(N) performance.
        """
        self.iterations = iterations
        self.window = window

    def optimize(self, paths):
        """
        Executes the optimization pipeline.
        1. Greedy Sort: Reorders paths so the next path starts near the current end.
        2. 2-Opt: Swaps sub-sections to 'untangle' paths and reduce total travel.
        """
        if len(paths) < 2:
            return paths
        
        # Step 1: Global Greedy Sort using a Spatial Index
        sorted_paths = self._kd_tree_sort(paths)
        
        # Step 2: Local Refinement to fix greedy sub-optimalities
        refined_paths = self._refine_2opt(sorted_paths)
        
        return refined_paths

    def _kd_tree_sort(self, paths):
        """
        Sorts paths by always jumping to the nearest unvisited neighbor.
        Uses a KD-Tree to avoid O(N^2) exhaustive searching.
        """
        # We index both starts and ends of every path.
        # This allows the algorithm to decide if it's faster to draw a path 
        # in its original direction or reversed.
        points_data = []
        for idx, segments in enumerate(paths):
            try:
                p_start = self._get_start(segments[0])
            except Exception as e:
                print("Error in path:", segments)
                raise
            p_end = self._get_end(segments[-1])
            # [x, y, original_path_index, type_flag]
            points_data.append([p_start[0], p_start[1], idx, 0]) # 0 = start
            points_data.append([p_end[0], p_end[1], idx, 1])     # 1 = end
        
        points_data = np.array(points_data)
        tree = KDTree(points_data[:, :2])
        
        visited = [False] * len(paths)
        ordered = []
        
        # Start at the first path provided in the input list
        curr_idx = 0
        visited[curr_idx] = True
        ordered.append(paths[curr_idx])
        curr_pos = self._get_end(paths[curr_idx][-1])

        for _ in range(len(paths) - 1):
            # Query the 100 nearest points to the current pen position
            dist, indices = tree.query(curr_pos, k=min(100, len(points_data)))
            
            found_next = False
            for tree_idx in indices:
                p_idx = int(points_data[tree_idx, 2])
                is_end_point = bool(points_data[tree_idx, 3])
                
                if not visited[p_idx]:
                    next_path = paths[p_idx]
                    # If the closest point found in the tree was an 'end' point,
                    # we reverse the path so that the end becomes the new start.
                    if is_end_point:
                        next_path = self._reverse_path(next_path)
                    
                    ordered.append(next_path)
                    visited[p_idx] = True
                    # Update pen position to the end of the path we just added
                    curr_pos = self._get_end(next_path[-1])
                    found_next = True
                    break
            
            # If the local neighborhood (k=100) is fully visited, 
            # find the first available unvisited path (a 'long jump').
            if not found_next:
                for i, v in enumerate(visited):
                    if not v:
                        ordered.append(paths[i])
                        visited[i] = True
                        curr_pos = self._get_end(paths[i][-1])
                        break
        return ordered

    def _refine_2opt(self, paths):
        """
        Iteratively improves the path order by swapping pairs of edges.
        This untangles 'crossed' travel lines which greedy search often creates.
        """
        n = len(paths)
        refined = list(paths)
        
        for _ in range(self.iterations):
            improved = False
            for i in range(1, n - 2):
                # We limit the search to a 'window' to keep the algorithm fast.
                # In pen plotting, most improvements are found between nearby paths.
                for j in range(i + 1, min(i + self.window, n - 1)):
                    
                    # Current travel cost: gap between (i-1, i) and (j, j+1)
                    d_curr = self._dist(self._get_end(refined[i-1][-1]), self._get_start(refined[i][0])) + \
                             self._dist(self._get_end(refined[j][-1]), self._get_start(refined[j+1][0]))
                    
                    # Cost if we swap the sequence: gap between (i-1, j) and (i, j+1)
                    d_swap = self._dist(self._get_end(refined[i-1][-1]), self._get_end(refined[j][-1])) + \
                             self._dist(self._get_start(refined[i][0]), self._get_start(refined[j+1][0]))
                    
                    if d_swap < d_curr:
                        # If a swap reduces distance, reverse the sub-section [i:j]
                        # and flip every path inside that sub-section.
                        sub_section = refined[i:j+1]
                        refined[i:j+1] = [self._reverse_path(p) for p in reversed(sub_section)]
                        improved = True
            if not improved:
                break
        return refined

    # --- Helpers for Coordinate Extraction and Distance ---

    def _get_start(self, seg):
        """Extracts the XY starting coordinate of a Line or Arc segment."""
        return np.array(seg["points"][0] if seg["type"] == "line" else seg["start"])

    def _get_end(self, seg):
        """Extracts the XY ending coordinate of a Line or Arc segment."""
        return np.array(seg["points"][-1] if seg["type"] == "line" else seg["end"])

    def _dist(self, p1, p2):
        """Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def _reverse_path(self, segments):
        """
        Reverses a list of segments. Arcs must have their 
        start/end points swapped to remain valid.
        """
        rev = []
        for s in reversed(segments):
            if s["type"] == "line":
                rev.append({"type": "line", "points": s["points"][::-1]})
            else:
                # Arcs require flipping start/end. 
                # (Radius and Center remain constant).
                rev.append({
                    "type": "arc", "start": s["end"], "end": s["start"],
                    "center": s["center"], "radius": s["radius"]
                })
        return rev