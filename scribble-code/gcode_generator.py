import math
import numpy as np

class GCodeGenerator:
    # Standard Paper Sizes in mm (Width, Height)
    PAPER_SIZES = {
        "a4": (210, 297),
        "a3": (297, 420),
    }

    def __init__(self, size="a4", landscape=False, margins=10, 
                 feed_rate=3000, pen_up_z=5, pen_down_z=0, pen_delay_ms=15, precision=4):
        """
        :param size: "a3", "a4", or tuple (width, height)
        :param landscape: If True, swaps width and height
        :param margins: mm to keep clear from paper edges
        """
        # 1. Define Physical Workspace
        if isinstance(size, str):
            self.p_width, self.p_height = self.PAPER_SIZES.get(size.lower(), (210, 297))
        else:
            self.p_width, self.p_height = size # Custom (w, h)
            
        if landscape:
            self.p_width, self.p_height = self.p_height, self.p_width

        self.pen_delay_ms = pen_delay_ms    
        self.margins = margins
        self.limit_w = self.p_width - (2 * margins)
        self.limit_h = self.p_height - (2 * margins)

        # 2. Machine Parameters
        self.feed_rate = feed_rate
        self.up_z = pen_up_z
        self.down_z = pen_down_z
        self.prec = precision
        self.last_pos = (0.0, 0.0)

    def _get_bounds(self, paths):
        """Calculates the bounding box of all paths."""
        all_pts = []
        for path in paths:
            for seg in path:
                pts = seg["points"] if seg["type"] == "line" else [seg["start"], seg["end"]]
                all_pts.extend(pts)
        pts_arr = np.array(all_pts)
        return pts_arr[:,0].min(), pts_arr[:,0].max(), pts_arr[:,1].min(), pts_arr[:,1].max()

    def generate(self, optimized_paths):
        """Transforms paths to fit page and generates G-code."""
        if not optimized_paths: return ""

        # 1. Calculate Scaling and Centering
        min_x, max_x, min_y, max_y = self._get_bounds(optimized_paths)
        content_w = max_x - min_x
        content_h = max_y - min_y
        
        # Fit-to-page scale factor
        scale = min(self.limit_w / content_w, self.limit_h / content_h)
        
        # Centering offsets
        offset_x = self.margins + (self.limit_w - content_w * scale) / 2 - (min_x * scale)
        offset_y = self.margins + (self.limit_h - content_h * scale) / 2 - (min_y * scale)

        gcode = [
            f"; Page: {self.p_width}x{self.p_height}mm, Margins: {self.margins}mm",
            "G21 ; mm", "G90 ; Absolute",
            f"G0 Z{self.up_z}",
            f"G4 P{self.pen_delay_ms / 1000}" 
        ]

        def transform(pt):
            return (pt[0] * scale + offset_x, pt[1] * scale + offset_y)

        # 2. Path Generation Loop
        for path in optimized_paths:
            # Move to start
            if path[0]["type"]=="line":
                s_pt = transform(path[0]["points"][0])    
            else:
                s_pt = transform(path[0]["start"])
            gcode.append(f"G0 X{s_pt[0]:.{self.prec}f} Y{s_pt[1]:.{self.prec}f} F{self.feed_rate}")
            gcode.append(f"G0 Z{self.down_z}")
            gcode.append(f"G4 P{self.pen_delay_ms / 1000}")
            self.last_pos = s_pt

            for seg in path:
                if seg["type"] == "line":
                    end_pt = transform(seg["points"][-1])
                    gcode.append(f"G1 X{end_pt[0]:.{self.prec}f} Y{end_pt[1]:.{self.prec}f} F{self.feed_rate}")
                    self.last_pos = end_pt
                else:
                    gcode.append(self._format_arc(seg, transform))

            gcode.append(f"G0 Z{self.up_z}")
            gcode.append(f"G4 P{self.pen_delay_ms / 1000}")

        gcode.append("G0 X0 Y0")
        return "\n".join(gcode)

    def _format_arc(self, seg, transform_fn):
        """Sanitizes arc with scale/offset transformation applied."""
        sx, sy = self.last_pos
        ex, ey = transform_fn(seg["end"])
        cx_in, cy_in = transform_fn(seg["center"])

        # Perpendicular Bisector Sanitization
        mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
        dx, dy = ex - sx, ey - sy
        chord_len = math.sqrt(dx*dx + dy*dy)
        r_orig = math.sqrt((sx - cx_in)**2 + (sy - cy_in)**2)
        r = max(r_orig, chord_len / 2.0 + 1e-7)
        
        h = math.sqrt(max(0, r**2 - (chord_len/2)**2))
        ux, uy = -dy / chord_len, dx / chord_len
        
        c1 = (mx + h * ux, my + h * uy)
        c2 = (mx - h * ux, my - h * uy)
        dist1 = (c1[0] - cx_in)**2 + (c1[1] - cy_in)**2
        dist2 = (c2[0] - cx_in)**2 + (c2[1] - cy_in)**2
        cx, cy = c1 if dist1 < dist2 else c2

        i_rel, j_rel = cx - sx, cy - sy
        cross = (ex - sx) * (cy - sy) - (ey - sy) * (cx - sx)
        cmd = "G2" if cross < 0 else "G3"

        self.last_pos = (ex, ey)
        return f"{cmd} X{ex:.{self.prec}f} Y{ey:.{self.prec}f} I{i_rel:.{self.prec}f} J{j_rel:.{self.prec}f} F{self.feed_rate}"