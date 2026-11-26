import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

import tqdm

# --- 1. GEOMETRY UTILS ---
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- 2. MERGING (Eulerian Path Optimization) ---
def merge_lines(lines):
    def to_tuple(p): return tuple(p)
    graph = defaultdict(list)
    degree = defaultdict(int)
    
    print("Building graph for merging...")
    for idx, line in enumerate(lines):
        p1, p2 = to_tuple(line[0]), to_tuple(line[-1])
        graph[p1].append({'to': p2, 'id': idx, 'data': line})
        graph[p2].append({'to': p1, 'id': idx, 'data': line[::-1]})
        degree[p1] += 1
        degree[p2] += 1

    used_edges = set()
    merged = []

    def trace_path(start_node):
        path = []
        curr = start_node
        while True:
            found_next = False
            for edge in graph[curr]:
                if edge['id'] not in used_edges:
                    used_edges.add(edge['id'])
                    if not path: path.extend(edge['data'])
                    else: path.extend(edge['data'][1:])
                    curr = edge['to']
                    found_next = True
                    break
            if not found_next: break
        return path
    print("sorting paths...")
    nodes = sorted(graph.keys(), key=lambda n: degree[n] % 2, reverse=True)
    print("Tracing odd-degree paths...")
    for node in nodes:
        if degree[node] % 2 != 0:
            p = trace_path(node); 
            if p: merged.append(p)
    print("Tracing remaining paths...")
    for node in nodes:
        p = trace_path(node); 
        if p: merged.append(p)
    print(f"Merging complete: {len(merged)} paths formed.")
    return merged

# --- 3. REORDERING (Greedy + 2-Opt TSP) ---
def optimize_travel(paths):
    if not paths: return []
    # Greedy Sort
    ordered = [paths.pop(0)]
    while paths:
        last_p = ordered[-1][-1]
        best_dist, best_idx, reverse = float('inf'), -1, False
        for i, line in enumerate(paths):
            d_s, d_e = get_distance(last_p, line[0]), get_distance(last_p, line[-1])
            if d_s < best_dist: best_dist, best_idx, reverse = d_s, i, False
            if d_e < best_dist: best_dist, best_idx, reverse = d_e, i, True
        next_l = paths.pop(best_idx)
        ordered.append(next_l[::-1] if reverse else next_l)
    
    # 2-Opt Refinement
    print("Refining path order with 2-Opt...")
    for pass_idx in range(2): # Two passes of refinement
        print(f"Pass {pass_idx + 1}")
        for i in tqdm.tqdm(range(1, len(ordered) - 2)):
            for j in range(i + 1, len(ordered)):
                d_old = get_distance(ordered[i-1][-1], ordered[i][0]) + \
                        get_distance(ordered[j-1][-1], ordered[j][0])
                d_new = get_distance(ordered[i-1][-1], ordered[j-1][-1]) + \
                        get_distance(ordered[i][0], ordered[j][0])
                if d_new < d_old:
                    ordered[i:j] = reversed(ordered[i:j])
                    for k in range(i, j): ordered[k] = ordered[k][::-1]
    print("2-Opt refinement complete.")
    return ordered

# --- 4. APPROXIMATION (Lines & Arcs via RMSE) ---

import numpy as np

# 1. PRECOMPUTE ONCE (Outside the DP loop)
class PathSums:
    def __init__(self, pts):
        self.x = np.cumsum(pts[:, 0])
        self.y = np.cumsum(pts[:, 1])
        self.x2 = np.cumsum(pts[:, 0]**2)
        self.y2 = np.cumsum(pts[:, 1]**2)
        self.xy = np.cumsum(pts[:, 0] * pts[:, 1])

    def get_sums(self, j, i):
        """Returns sums for the range [j, i] in O(1)"""
        def diff(arr): return arr[i] - arr[j-1] if j > 0 else arr[i]
        n = i - j + 1
        return n, diff(self.x), diff(self.y), diff(self.x2), diff(self.y2), diff(self.xy)
    
class ArcSums:
    def __init__(self, pts):
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        x2, y2, xy = x*x, y*y, x*y
        z = x2 + y2
        
        # Base sums for the 3x3 system
        self.s_x = np.cumsum(x)
        self.s_y = np.cumsum(y)
        self.s_x2 = np.cumsum(x2)
        self.s_y2 = np.cumsum(y2)
        self.s_xy = np.cumsum(xy)
        
        # Cubic sums for the target vector
        self.s_x3 = np.cumsum(x*x2)
        self.s_y3 = np.cumsum(y*y2)
        self.s_x2y = np.cumsum(x2*y)
        self.s_xy2 = np.cumsum(x*y2)
        
        # Quartic sums for O(1) RMSE calculation
        self.s_z2 = np.cumsum(z*z)
        self.s_zx = np.cumsum(z*x)
        self.s_zy = np.cumsum(z*y)
        self.s_z = np.cumsum(z)

    def get_window(self, j, i):
        def d(arr): return arr[i] - arr[j-1] if j > 0 else arr[i]
        n = i - j + 1
        return {
            'n': n, 'x': d(self.s_x), 'y': d(self.s_y), 
            'x2': d(self.s_x2), 'y2': d(self.s_y2), 'xy': d(self.s_xy),
            'x3': d(self.s_x3), 'y3': d(self.s_y3), 'x2y': d(self.s_x2y), 'xy2': d(self.s_xy2),
            'z2': d(self.s_z2), 'zx': d(self.s_zx), 'zy': d(self.s_zy), 'z': d(self.s_z)
        }

class ArcSumsV2:
    def __init__(self, pts):
        # 1. Coordinate Normalization
        # Subtract the first point to keep values near the origin
        self.origin = pts[0].astype(np.float64)
        local_pts = pts.astype(np.float64) - self.origin
        
        x = local_pts[:, 0]
        y = local_pts[:, 1]
        
        x2, y2, xy = x*x, y*y, x*y
        z = x2 + y2
        
        # 2. Precompute Cumulative Sums (using float64)
        self.s_x = np.cumsum(x)
        self.s_y = np.cumsum(y)
        self.s_x2 = np.cumsum(x2)
        self.s_y2 = np.cumsum(y2)
        self.s_xy = np.cumsum(xy)
        
        self.s_x3 = np.cumsum(x*x2)
        self.s_y3 = np.cumsum(y*y2)
        self.s_x2y = np.cumsum(x2*y)
        self.s_xy2 = np.cumsum(x*y2)
        
        self.s_z2 = np.cumsum(z*z)
        self.s_zx = np.cumsum(z*x)
        self.s_zy = np.cumsum(z*y)
        self.s_z = np.cumsum(z)

    def get_window(self, j, i):
        """Returns the O(1) sums for the range [j, i]."""
        def d(arr): return arr[i] - arr[j-1] if j > 0 else arr[i]
        n = i - j + 1
        return {
            'n': n, 'x': d(self.s_x), 'y': d(self.s_y), 
            'x2': d(self.s_x2), 'y2': d(self.s_y2), 'xy': d(self.s_xy),
            'x3': d(self.s_x3), 'y3': d(self.s_y3), 'x2y': d(self.s_x2y), 'xy2': d(self.s_xy2),
            'z2': d(self.s_z2), 'zx': d(self.s_zx), 'zy': d(self.s_zy), 'z': d(self.s_z)
        }

def fit_arc_o1_normalized(j, i, arc_sums):
    s = arc_sums.get_window(j, i)
    if s['n'] < 3: return float('inf'), None, 0

    # Solve 3x3 as before...
    M = np.array([
        [s['x2'], s['xy'], s['x']],
        [s['xy'], s['y2'], s['y']],
        [s['x'],  s['y'],  s['n']]
    ])
    Y = np.array([-(s['x3'] + s['xy2']), -(s['x2y'] + s['y3']), -(s['x2'] + s['y2'])])

    try:
        A, B, C = np.linalg.solve(M, Y)
        local_cx, local_cy = -A / 2.0, -B / 2.0
        radius_sq = local_cx**2 + local_cy**2 - C
        if radius_sq <= 0: return float('inf'), None, 0
        radius = math.sqrt(radius_sq)

        # Calculate Algebraic RMSE
        err_sum = (s['z2'] + A*A*s['x2'] + B*B*s['y2'] + s['n']*C*C +
                   2*A*s['zx'] + 2*B*s['zy'] + 2*C*s['z'] +
                   2*A*B*s['xy'] + 2*A*C*s['x'] + 2*B*C*s['y'])
        rmse = math.sqrt(max(0, err_sum / s['n'])) / (2 * radius)

        # Denormalize: Add the origin back to the center
        global_cx = local_cx + arc_sums.origin[0]
        global_cy = local_cy + arc_sums.origin[1]
        
        return rmse, (global_cx, global_cy), radius
    except np.linalg.LinAlgError:
        return float('inf'), None, 0

# 2. THE O(1) RMSE FUNCTION (Inside the DP loop)
def get_line_rmse_o1(j, i, p_sums, pts):
    p1, p2 = pts[j], pts[i]
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    
    denom_sq = A*A + B*B
    if denom_sq == 0: return 0.0
    
    n, sx, sy, sx2, sy2, sxy = p_sums.get_sums(j, i)
    
    # Using the expanded polynomial formula
    sum_dist_sq = (A*A*sx2 + B*B*sy2 + n*C*C + 
                   2*A*B*sxy + 2*A*C*sx + 2*B*C*sy)
    
    mse = sum_dist_sq / (n * denom_sq)
    return math.sqrt(max(0, mse)) # max(0,...) handles tiny floating point errors

def get_arc_rmse(pts):
    if len(pts) < 3: return float('inf'), None, 0
    # Algebraic circle fit
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2
    try:
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = c[0]/2, c[1]/2
        radius = np.sqrt(c[2] + cx**2 + cy**2)
        distances = np.linalg.norm(pts - [cx, cy], axis=1)
        rmse = np.sqrt(np.mean((distances - radius)**2))
        return rmse, (cx, cy), radius
    except:
        return float('inf'), None, 0

def get_line_rmse_o1_old(j, i, p_sums, pts):
    p1, p2 = pts[j], pts[i]
    # Line eq: Ax + By + C = 0
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    
    denom_sq = A*A + B*B
    if denom_sq < 1e-9: return 0.0 # Points are identical
    
    n, sx, sy, sx2, sy2, sxy = p_sums.get_sums(j, i)
    
    # Polynomial expansion of Sum(Ax + By + C)^2
    sum_dist_sq = (A*A*sx2 + B*B*sy2 + n*C*C + 
                   2*A*B*sxy + 2*A*C*sx + 2*B*C*sy)
    
    mse = sum_dist_sq / (n * denom_sq)
    return math.sqrt(max(0, mse))


def simplify_geometry_dp_ultimate(path, max_rmse, window=150):
    pts = np.array(path)
    n = len(pts)
    if n < 2: return []

    # Precompute everything
    p_sums = ArcSumsV2(pts)
    
    dp = np.full(n, np.inf)
    parent = np.full(n, -1)
    seg_info = [None] * n
    dp[0] = 0
    
    for i in range(1, n):
        for j in range(max(0, i - window), i):
            # 1. Line check O(1)
            l_rmse = get_line_rmse_o1(j, i, p_sums, pts) # Re-use previous logic
            
            fit_type = None
            info = None
            
            if l_rmse <= max_rmse:
                fit_type = 'line'
                info = {"type": "line", "points": [pts[j].tolist(), pts[i].tolist()]}
            
            # 2. Arc check O(1) (Only if line isn't already a 'perfect' fit)
            if fit_type is None or l_rmse > (max_rmse * 0.4):
                a_rmse, center, rad = fit_arc_o1_normalized(j, i, p_sums)
                if a_rmse <= max_rmse and (fit_type is None or a_rmse < l_rmse):
                    fit_type = 'arc'
                    info = {"type": "arc", "start": pts[j].tolist(), "end": pts[i].tolist(), 
                            "center": center, "radius": rad}

            # 3. DP Update
            if fit_type:
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
                    seg_info[i] = info

    # Reconstruct path... (same backtracking as before)
    # 3. Backtrack
    result = []
    curr = n - 1
    while curr > 0:
        prev = parent[curr]
        if prev == -1: break
        result.append(seg_info[curr])
        curr = prev
        
    return result[::-1]

def simplify_geometry(path, max_rmse, window=100):
    pts = np.array(path)
    n = len(pts)
    if n < 2: return []
    print(f"Simplifying geometry for path of length {n} with max RMSE {max_rmse}...")
    # 1. Precompute sums
    p_sums = PathSums(pts)
    print("Precomputed path sums for RMSE calculations.")
    
    # 2. DP Tables
    dp = np.full(n, np.inf)      # Cost (number of segments)
    parent = np.full(n, -1)      # Previous split point
    seg_info = [None] * n        # Metadata for the segment
    
    dp[0] = 0
    
    for i in range(1, n):
        # Sliding window look-back
        for j in range(max(0, i - window), i):
            best_type = None
            current_info = None
            
            # --- STEP A: Check Line (O(1)) ---
            l_rmse = get_line_rmse_o1(j, i, p_sums, pts)
            
            if l_rmse <= max_rmse:
                best_type = 'line'
                current_info = {
                    "type": "line", 
                    "points": [pts[j].tolist(), pts[i].tolist()]
                }
            
            # --- STEP B: Check Arc (Only if Line isn't perfect) ---
            # We only try arc if line_rmse > threshold/2 to avoid over-segmenting arcs
            if best_type is None or l_rmse > (max_rmse * 0.5):
                # Use your optimized fit_arc_fast here
                center, radius, a_rmse = fit_arc(pts[j:i+1])
                
                if a_rmse <= max_rmse and (best_type is None or a_rmse < l_rmse):
                    best_type = 'arc'
                    current_info = {
                        "type": "arc",
                        "start": pts[j].tolist(),
                        "end": pts[i].tolist(),
                        "center": center,
                        "radius": radius
                    }

            # --- STEP C: Update DP ---
            if best_type is not None:
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
                    seg_info[i] = current_info

    # 3. Backtrack
    result = []
    curr = n - 1
    while curr > 0:
        prev = parent[curr]
        if prev == -1: break
        result.append(seg_info[curr])
        curr = prev
        
    return result[::-1]

def fit_arc_o1(j, i, sums):
    s = sums.get_window(j, i)
    if s['n'] < 3: return float('inf'), None, 0

    # Matrix M * [A, B, C]^T = Y
    M = np.array([
        [s['x2'], s['xy'], s['x']],
        [s['xy'], s['y2'], s['y']],
        [s['x'],  s['y'],  s['n']]
    ])
    
    Y = np.array([
        -(s['x3'] + s['xy2']),
        -(s['x2y'] + s['y3']),
        -(s['x2'] + s['y2'])
    ])

    try:
        # Solve the 3x3 system
        A, B, C = np.linalg.solve(M, Y)
        
        cx = -A / 2.0
        cy = -B / 2.0
        radius_sq = cx**2 + cy**2 - C
        if radius_sq <= 0: return float('inf'), None, 0
        radius = math.sqrt(radius_sq)

        # O(1) Algebraic Error Calculation
        # sum((z + Ax + By + C)^2) expanded:
        err_sum = (s['z2'] + A*A*s['x2'] + B*B*s['y2'] + s['n']*C*C +
                   2*A*s['zx'] + 2*B*s['zy'] + 2*C*s['z'] +
                   2*A*B*s['xy'] + 2*A*C*s['x'] + 2*B*C*s['y'])
        
        # Approximate geometric RMSE: algebraic_error / (2 * radius)
        rmse = math.sqrt(max(0, err_sum / s['n'])) / (2 * radius)
        
        return rmse, (cx, cy), radius
    except np.linalg.LinAlgError:
        return float('inf'), None, 0
    
import numpy as np
import math

def fit_arc(pts):
    """
    pts: (N, 2) numpy array slice (passing a slice is faster than creating one)
    """
    N = pts.shape[0]
    if N < 3: return None, 0, float('inf')

    x = pts[:, 0]
    y = pts[:, 1]
    x2 = x**2
    y2 = y**2
    xy = x * y
    sq_sum = x2 + y2

    # We solve the 3x3 system: (A.T @ A) * c = A.T @ b
    # This avoids the overhead of np.linalg.lstsq
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x2)
    sum_y2 = np.sum(y2)
    sum_xy = np.sum(xy)

    # Matrix A.T @ A construction
    ATA = np.array([
        [4 * sum_x2, 4 * sum_xy, 2 * sum_x],
        [4 * sum_xy, 4 * sum_y2, 2 * sum_y],
        [2 * sum_x,  2 * sum_y,  N]
    ])

    # Vector A.T @ b construction
    ATb = np.array([
        2 * np.sum(x * sq_sum),
        2 * np.sum(y * sq_sum),
        np.sum(sq_sum)
    ])

    try:
        cx, cy, C = np.linalg.solve(ATA, ATb)
        radius = math.sqrt(C + cx**2 + cy**2)
        
        # Optimized RMSE (np.sqrt is faster than np.linalg.norm for 2D points)
        dist_sq = (x - cx)**2 + (y - cy)**2
        rmse = np.sqrt(np.mean((np.sqrt(dist_sq) - radius)**2))
        
        return (cx, cy), radius, rmse
    except (np.linalg.LinAlgError, ValueError):
        return None, 0, float('inf')


# --- 5. G-CODE GENERATION ---
# --- 5. G-CODE GENERATION WITH DPI ---
def to_gcode(segments, dpi=96, feed=2000, z_up=5, z_down=0, delay=15):
    # scale factor: mm per pixel
    scale = 25.4 / dpi
    
    def sc(val): return val * scale

    gc = [f"G21 ; Units in mm\nG90 ; Absolute\nF{feed}\nG0 Z{z_up}\nG4 P{delay}"]
    
    for s in segments:
        start = s["points"][0] if s["type"]=="line" else s["start"]
        end = s["points"][-1] if s["type"]=="line" else s["end"]
        
        # Rapid to start
        gc.append(f"G0 X{sc(start[0]):.3f} Y{sc(start[1]):.3f}")
        # Pen Down
        gc.append(f"G1 Z{z_down}\nG4 P{delay}")
        
        if s["type"] == "line":
            gc.append(f"G1 X{sc(end[0]):.3f} Y{sc(end[1]):.3f}")
        else:
            # I and J are relative offsets from start to center
            i = sc(s["center"][0]) - sc(start[0])
            j = sc(s["center"][1]) - sc(start[1])
            gc.append(f"G2 X{sc(end[0]):.3f} Y{sc(end[1]):.3f} I{i:.3f} J{j:.3f}")
            
        # Pen Up
        gc.append(f"G0 Z{z_up}\nG4 P{delay}")
    
    gc.append("G0 X0 Y0 ; Return Home")
    return "\n".join(gc)

# --- 6. VISUALIZATION (IMSHOW) ---
def render_preview(segments, resolution=5):
    all_pts = []
    for s in segments:
        p = s["points"] if s["type"]=="line" else [s["start"], s["end"], s["center"]]
        all_pts.extend(p)
    all_pts = np.array(all_pts)
    min_v, max_v = np.min(all_pts, 0) - 5, np.max(all_pts, 0) + 5
    w, h = int((max_v[0]-min_v[0])*resolution), int((max_v[1]-min_v[1])*resolution)
    canvas = np.full((h, w), 255, dtype=np.uint8)
    
    for s in segments:
        if s["type"]=="line":
            p1 = ((np.array(s["points"][0])-min_v)*resolution).astype(int)
            p2 = ((np.array(s["points"][-1])-min_v)*resolution).astype(int)
            cv2.line(canvas, tuple(p1), tuple(p2), 0, 1)
        else:
            c = tuple(((np.array(s["center"])-min_v)*resolution).astype(int))
            r = int(s["radius"]*resolution)
            a1 = np.degrees(np.atan2(s["start"][1]-s["center"][1], s["start"][0]-s["center"][0]))
            a2 = np.degrees(np.atan2(s["end"][1]-s["center"][1], s["end"][0]-s["center"][0]))
            cv2.ellipse(canvas, c, (r, r), 0, a1, a2, 0, 1)

    plt.imshow(canvas, cmap='gray', origin='lower')
    plt.show()


def get_auto_scale(segments, target_w=170, target_h=257):
    # 1. Get bounding box of original pixels
    all_pts = []
    for s in segments:
        p = s["points"] if s["type"]=="line" else [s["start"], s["end"]]
        all_pts.extend(p)
    all_pts = np.array(all_pts)
    
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    
    pixel_w = max_x - min_x
    pixel_h = max_y - min_y
    
    # 2. Calculate scale factor to fit within target mm
    scale_w = target_w / pixel_w if pixel_w > 0 else 1
    scale_h = target_h / pixel_h if pixel_h > 0 else 1
    
    # Use the smaller scale to ensure it fits both dimensions
    final_scale = min(scale_w, scale_h)
    
    # Return scale and the offsets to center the drawing and account for 20mm margin
    return final_scale, min_x, min_y

def to_gcode_a4(segments, feed=2000, margin=20):
    scale, offset_x, offset_y = get_auto_scale(segments)
    
    gc = [f"G21 ; mm\nG90 ; Absolute\nF{feed}\nG0 Z5\nG4 P15"]
    
    for s in segments:
        # Scale and Shift coordinates
        # (val - offset) moves the drawing to 0,0
        # * scale resizes to mm
        # + margin moves it into the safe zone
        def transform(pt):
            nx = (pt[0] - offset_x) * scale + margin
            ny = (pt[1] - offset_y) * scale + margin
            return nx, ny

        start_pt = s["points"][0] if s["type"]=="line" else s["start"]
        end_pt = s["points"][-1] if s["type"]=="line" else s["end"]
        
        sx, sy = transform(start_pt)
        ex, ey = transform(end_pt)
        
        # Rapid to start
        gc.append(f"G0 X{sx:.3f} Y{sy:.3f}")
        # Pen Down
        gc.append("G1 Z0\nG4 P15")
        
        if s["type"] == "line":
            gc.append(f"G1 X{ex:.3f} Y{ey:.3f}")
        else:
            # Arcs require scaling the center and radius too
            cx, cy = transform(s["center"])
            i, j = cx - sx, cy - sy
            gc.append(f"G2 X{ex:.3f} Y{ey:.3f} I{i:.3f} J{j:.3f}")
            
        # Pen Up
        gc.append("G0 Z5\nG4 P15")
    
    gc.append("G0 X0 Y0")
    return "\n".join(gc)

def generate_final_gcode(paths_of_segments, size="A4", margin=20, feed=2000, z_up=5, z_down=0, delay=15):
    # Paper Config
    digits = 14
    paper_lib = {"A4": (297, 210), "A3": (420, 297)}
    paper_w, paper_h = paper_lib.get(size, (297, 210))
    target_w, target_h = paper_w - (margin * 2), paper_h - (margin * 2)

    # Calculate Bounding Box for scaling and rotation
    all_pts = []
    for path in paths_of_segments:
        for s in path:
            pts = s["points"] if s["type"]=="line" else [s["start"], s["end"]]
            all_pts.extend(pts)
    all_pts = np.array(all_pts)
    b_min = np.min(all_pts, axis=0)
    b_max = np.max(all_pts, axis=0)
    w, h = b_max - b_min
    
    rotate = h > w # Landscape logic: Rotate if tall
    scale = min(target_w/h, target_h/w) if rotate else min(target_w/w, target_h/h)

    def transform(p):
        nx, ny = p[0] - b_min[0], p[1] - b_min[1]
        if rotate: nx, ny = ny, -nx
        # Re-offset to ensure rotation stays in positive quadrant
        tx = nx * scale + margin
        ty = ny * scale + margin
        if rotate: tx += target_w if ny < 0 else 0 # Simple fix for negative rotation
        return tx, ty

    gc = [f"G21 ; mm\nG90 ; Absolute\nF{feed}\nG0 Z{z_up}\nG4 P{delay}"]

    for path in paths_of_segments:
        if not path: continue
        
        # Start of path
        first = path[0]
        start_pt = first["points"][0] if first["type"]=="line" else first["start"]
        sx, sy = transform(start_pt)

        # 1. Travel to Start and DROP PEN
        gc.append(f"G0 X{sx:.14f} Y{sy:.14f}")
        gc.append(f"G0 Z{z_down}\nG4 P{delay}")

        # 2. Draw entire path without lifting
        for s in path:
            end_pt = s["points"][-1] if s["type"]=="line" else s["end"]
            ex, ey = transform(end_pt)
            
            if s["type"] == "line":
                gc.append(f"G1 X{ex:.14f} Y{ey:.14f} F{feed}")
            else:
                # Arc with G2/G3 check
                cx, cy = transform(s["center"])
                # Get current point (start of this specific segment)
                cur_s = s["points"][0] if s["type"]=="line" else s["start"]
                csx, csy = transform(cur_s)
                
                # Cross product for direction
                v1 = (csx-cx, csy-cy)
                v2 = (ex-cx, ey-cy)
                cp = v1[0]*v2[1] - v1[1]*v2[0]
                cmd = "G3" if cp > 0 else "G2"
                
                gc.append(f"{cmd} X{ex:.14f} Y{ey:.14f} I{cx-csx:.14f} J{cy-csy:.14f}")

        # 3. Finish path and LIFT PEN
        gc.append(f"G0 Z{z_up}\nG4 P{delay}")

    gc.append("G0 X0 Y0 ; Home")
    return "\n".join(gc)

def estimate_duration(paths_of_segments, feed_rate_mm_min=2000, accel_mm_s2=500, rapid_rate_mm_min=3000, delay_ms=15):
    """
    Estimates time in seconds.
    feed_rate: Speed while drawing (mm/min)
    rapid_rate: Speed while traveling (mm/min)
    accel: Machine acceleration (mm/s^2)
    """
    draw_v = feed_rate_mm_min / 60.0  # convert to mm/s
    rapid_v = rapid_rate_mm_min / 60.0 # convert to mm/s
    delay_s = delay_ms / 1000.0
    
    total_time = 0
    total_lifts = 0
    prev_end = [0, 0] # Assume starts at home

    def time_for_dist(d, v_max, a):
        if d <= 0: return 0
        # Distance needed to reach v_max: d = v^2 / 2a
        dist_to_top = (v_max**2) / (2 * a)
        
        if d >= 2 * dist_to_top:
            # Reaches max speed: 2 * accel_time + constant_speed_time
            t_accel = v_max / a
            t_const = (d - 2 * dist_to_top) / v_max
            return (2 * t_accel) + t_const
        else:
            # Does not reach max speed: Triangular profile
            # v_peak = sqrt(a * d)
            return 2 * (math.sqrt(a * d) / a)

    for path in paths_of_segments:
        if not path: continue
        total_lifts += 1
        
        # 1. Travel to path start (G0 - Pen Up)
        first_seg = path[0]
        start_pt = first_seg["points"][0] if first_seg["type"]=="line" else first_seg["start"]
        travel_dist = get_distance(prev_end, start_pt)
        total_time += time_for_dist(travel_dist, rapid_v, accel_mm_s2)
        
        # 2. Pen Down Delay
        total_time += delay_s
        
        # 3. Drawing Moves (G1/G2/G3 - Pen Down)
        for s in path:
            end_pt = s["points"][-1] if s["type"]=="line" else s["end"]
            
            if s["type"] == "line":
                # For simplified paths, we treat the whole line as one move
                dist = get_distance(s["points"][0], s["points"][-1])
            else:
                # Arc length: radius * angle_delta
                # Simplified arc length for estimation
                dist = s["radius"] * abs(math.atan2(s["end"][1]-s["center"][1], s["end"][0]-s["center"][0]) - 
                                         math.atan2(s["start"][1]-s["center"][1], s["start"][0]-s["center"][0]))
            
            total_time += time_for_dist(dist, draw_v, accel_mm_s2)
            prev_end = end_pt

        # 4. Pen Up Delay
        total_time += delay_s

    return total_time, total_lifts

import numpy as np
import math

class GeometryOptimizer:
    def __init__(self, max_rmse=0.5, window=150):
        self.max_rmse = max_rmse
        self.window = window

    class PathSums:
        def __init__(self, pts):
            # Normalize to first point for numerical stability (prevents 4th power overflow)
            self.origin = pts[0].astype(np.float64)
            local_pts = pts.astype(np.float64) - self.origin
            
            x = local_pts[:, 0]
            y = local_pts[:, 1]
            x2, y2, xy = x*x, y*y, x*y
            z = x2 + y2
            
            # Linear & Quadratic (for Lines)
            self.s_x, self.s_y = np.cumsum(x), np.cumsum(y)
            self.s_x2, self.s_y2, self.s_xy = np.cumsum(x2), np.cumsum(y2), np.cumsum(xy)
            
            # Cubic & Quartic (for Arcs)
            self.s_x3, self.s_y3 = np.cumsum(x*x2), np.cumsum(y*y2)
            self.s_x2y, self.s_xy2 = np.cumsum(x2*y), np.cumsum(x*y2)
            self.s_z, self.s_z2 = np.cumsum(z), np.cumsum(z*z)
            self.s_zx, self.s_zy = np.cumsum(z*x), np.cumsum(z*y)

        def get_window(self, j, i):
            def d(arr): return arr[i] - arr[j-1] if j > 0 else arr[i]
            n = i - j + 1
            return {
                'n': n, 'x': d(self.s_x), 'y': d(self.s_y), 
                'x2': d(self.s_x2), 'y2': d(self.s_y2), 'xy': d(self.s_xy),
                'x3': d(self.s_x3), 'y3': d(self.s_y3), 'x2y': d(self.s_x2y), 'xy2': d(self.s_xy2),
                'z2': d(self.s_z2), 'zx': d(self.s_zx), 'zy': d(self.s_zy), 'z': d(self.s_z)
            }

    def _get_line_rmse_o1(self, j, i, p_sums, pts):
        # Line equation between start and end point of window
        p1, p2 = pts[j] - p_sums.origin, pts[i] - p_sums.origin
        A, B = p1[1] - p2[1], p2[0] - p1[0]
        C = p1[0]*p2[1] - p2[0]*p1[1]
        
        denom_sq = A*A + B*B
        if denom_sq < 1e-12: return 0.0
        
        s = p_sums.get_window(j, i)
        # Sum of (Ax + By + C)^2 expansion
        sum_dist_sq = (A*A*s['x2'] + B*B*s['y2'] + s['n']*C*C + 
                       2*A*B*s['xy'] + 2*A*C*s['x'] + 2*B*C*s['y'])
        
        return math.sqrt(max(0, sum_dist_sq / (s['n'] * denom_sq)))

    def _fit_arc_o1(self, j, i, p_sums):
        s = p_sums.get_window(j, i)
        if s['n'] < 3: return float('inf'), None, 0
        
        # Solve algebraic circle fit: x^2 + y^2 + Ax + By + C = 0
        M = np.array([[s['x2'], s['xy'], s['x']],
                      [s['xy'], s['y2'], s['y']],
                      [s['x'],  s['y'],  s['n']]])
        Y = np.array([-(s['x3'] + s['xy2']), -(s['x2y'] + s['y3']), -(s['x2'] + s['y2'])])

        try:
            params = np.linalg.solve(M, Y)
            A, B, C = params
            local_cx, local_cy = -A / 2.0, -B / 2.0
            radius_sq = local_cx**2 + local_cy**2 - C
            if radius_sq <= 0: return float('inf'), None, 0
            radius = math.sqrt(radius_sq)

            # Algebraic Error to Geometric RMSE approximation
            err_sum = (s['z2'] + A*A*s['x2'] + B*B*s['y2'] + s['n']*C*C +
                       2*A*s['zx'] + 2*B*s['zy'] + 2*C*s['z'] +
                       2*A*B*s['xy'] + 2*A*C*s['x'] + 2*B*C*s['y'])
            rmse = math.sqrt(max(0, err_sum / s['n'])) / (2 * radius)
            
            return rmse, (local_cx + p_sums.origin[0], local_cy + p_sums.origin[1]), radius
        except np.linalg.LinAlgError:
            return float('inf'), None, 0

    def optimize(self, path):
        pts = np.array(path)
        n = len(pts)
        if n < 2: return [{"type": "line", "points": [pts[0].tolist(), pts[0].tolist()]}]
        
        p_sums = self.PathSums(pts)
        dp = np.full(n, np.inf); parent = np.full(n, -1)
        seg_info = [None] * n
        dp[0] = 0
        
        for i in range(1, n):
            # Look back window
            for j in range(max(0, i - self.window), i):
                fit_type, info = None, None

                num_points = i - j + 1
                
                # Check Line first (cheaper)
                l_rmse = self._get_line_rmse_o1(j, i, p_sums, pts)
                if l_rmse <= self.max_rmse:
                    fit_type = 'line'
                    info = {"type": "line", "points": [pts[j].tolist(), pts[i].tolist()]}
                
                # Check Arc if line fails or for better curvature fit
                if num_points >= 6:
                    if fit_type is None or l_rmse > (self.max_rmse * 0.3):
                        a_rmse, center, rad = self._fit_arc_o1(j, i, p_sums)
                        if a_rmse <= self.max_rmse and (fit_type is None or a_rmse < l_rmse*0.3): # Prefer arc if significantly better
                            fit_type = 'arc'
                            info = {"type": "arc", "start": pts[j].tolist(), "end": pts[i].tolist(), 
                                    "center": center, "radius": rad}

                if fit_type and dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
                    seg_info[i] = info

        # Backtrack
        result = []; curr = n - 1
        while curr > 0:
            prev = parent[curr]
            if prev == -1: break
            result.append(seg_info[curr])
            curr = prev
        return result[::-1]
    

def estimate_duration(paths_of_segments, feed_rate_mm_min=2000, accel_mm_s2=500, rapid_rate_mm_min=3000, delay_ms=15):
    """
    Estimates time in seconds.
    feed_rate: Speed while drawing (mm/min)
    rapid_rate: Speed while traveling (mm/min)
    accel: Machine acceleration (mm/s^2)
    """
    draw_v = feed_rate_mm_min / 60.0  # convert to mm/s
    rapid_v = rapid_rate_mm_min / 60.0 # convert to mm/s
    delay_s = delay_ms / 1000.0
    
    total_time = 0
    total_lifts = 0
    prev_end = [0, 0] # Assume starts at home

    def time_for_dist(d, v_max, a):
        if d <= 0: return 0
        # Distance needed to reach v_max: d = v^2 / 2a
        dist_to_top = (v_max**2) / (2 * a)
        
        if d >= 2 * dist_to_top:
            # Reaches max speed: 2 * accel_time + constant_speed_time
            t_accel = v_max / a
            t_const = (d - 2 * dist_to_top) / v_max
            return (2 * t_accel) + t_const
        else:
            # Does not reach max speed: Triangular profile
            # v_peak = sqrt(a * d)
            return 2 * (math.sqrt(a * d) / a)

    for path in paths_of_segments:
        if not path: continue
        total_lifts += 1
        
        # 1. Travel to path start (G0 - Pen Up)
        first_seg = path[0]
        start_pt = first_seg["points"][0] if first_seg["type"]=="line" else first_seg["start"]
        travel_dist = get_distance(prev_end, start_pt)
        total_time += time_for_dist(travel_dist, rapid_v, accel_mm_s2)
        
        # 2. Pen Down Delay
        total_time += delay_s
        
        # 3. Drawing Moves (G1/G2/G3 - Pen Down)
        for s in path:
            end_pt = s["points"][-1] if s["type"]=="line" else s["end"]
            
            if s["type"] == "line":
                # For simplified paths, we treat the whole line as one move
                dist = get_distance(s["points"][0], s["points"][-1])
            else:
                # Arc length: radius * angle_delta
                # Simplified arc length for estimation
                dist = s["radius"] * abs(math.atan2(s["end"][1]-s["center"][1], s["end"][0]-s["center"][0]) - 
                                         math.atan2(s["start"][1]-s["center"][1], s["start"][0]-s["center"][0]))
            
            total_time += time_for_dist(dist, draw_v, accel_mm_s2)
            prev_end = end_pt

        # 4. Pen Up Delay
        total_time += delay_s

    return total_time, total_lifts

def to_gcode_continuous(paths_of_segments, feed=2000, margin=20, z_up=5, z_down=0, delay=15, scale_params=None):
    """
    paths_of_segments: A list where each element is a LIST of simplified segments 
                       (lines/arcs) that form one continuous path.
    """
    scale, r_min = scale_params # Passed from the scaling logic
    
    gc = [f"G21 ; mm\nG90 ; Absolute\nF{feed}\nG0 Z{z_up}\nG4 P{delay}"]

    for path in paths_of_segments:
        if not path: continue
        
        # 1. Get the very first point of the entire path
        first_seg = path[0]
        start_pt = first_seg["points"][0] if first_seg["type"]=="line" else first_seg["start"]
        
        def transform(p):
            fx = (p[0] - r_min[0]) * scale + margin
            fy = (p[1] - r_min[1]) * scale + margin
            return fx, fy

        sx, sy = transform(start_pt)
        
        # 2. Rapid to Start and Drop Pen
        gc.append(f"G0 X{sx:.3f} Y{sy:.3f}")
        gc.append(f"G1 Z{z_down}\nG4 P{delay}")

        # 3. Iterate through every segment in this path WITHOUT lifting
        for s in path:
            end_pt = s["points"][-1] if s["type"]=="line" else s["end"]
            ex, ey = transform(end_pt)
            
            if s["type"] == "line":
                # If the segment has multiple points, draw through all of them
                if len(s["points"]) > 2:
                    for p in s["points"][1:]: # Skip the first as we are already there
                        px, py = transform(p)
                        gc.append(f"G1 X{px:.3f} Y{py:.3f}")
                else:
                    gc.append(f"G1 X{ex:.3f} Y{ey:.3f}")
            
            elif s["type"] == "arc":
                cx, cy = transform(s["center"])
                # Calculate I and J based on current pen position (sx, sy)
                # Note: sx, sy must be updated to current end at each step
                curr_x, curr_y = transform(s["start"]) 
                i, j = cx - curr_x, cy - curr_y
                gc.append(f"G2 X{ex:.3f} Y{ey:.3f} I{i:.3f} J{j:.3f}")

        # 4. Path finished - Lift Pen
        gc.append(f"G0 Z{z_up}\nG4 P{delay}")
        
    gc.append("G0 X0 Y0")
    return "\n".join(gc)