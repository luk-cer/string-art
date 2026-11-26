G_CODE_HEADER="""G21 ; set units to millimeters
G90 ; use absolute coordinates
G0 Z5.00 ; lift pen
"""

def line_to_gcode(line, dpi=90, feed_rate=1500,up_height=5,z_delay_ms=15, round_digits=2):
    gcode = []
    #scale points from pixels to mm if DPI is given use that
    #use feedrate and movement commands repeat feed rate on every G1 commnsd
    #round all coordinates to given digits
    scale = 25.4 / dpi
    #rapid move to firtst point
    start_x = round(line[0][1] * scale, round_digits)
    start_y = round(line[0][0] * scale, round_digits)
    gcode.append(f'G0 X{start_x} Y{start_y}')   
    #lower pen to drawing height
    gcode.append(f'G0 Z{up_height}')   
    #wait for z_delay_ms
    gcode.append(f'G1 F{feed_rate}')
    #draw line at feed_rate
    #move through all points in line
    #if rounded coordinates are same as previous, skip
    prev_x = start_x
    prev_y = start_y    
    for point in line[1:]:
        x = round(point[1] * scale, round_digits)
        y = round(point[0] * scale, round_digits)   
        if x != prev_x or y != prev_y:
            gcode.append(f'G1 X{x} Y{y} F{feed_rate}')
        prev_x = x
        prev_y = y
    #lift pen to up_height
    gcode.append(f'G0 Z5.00')
    #return gcode commands as list of strings
    return gcode

def points_to_line_segemnts(points, rmse=0.1):
    #input is list of points formin a curve
    #output is list of linear spline nodes
    #minimize number of segments
    #where rmse between original points and linear spline is less than given rmse
    if len(points)<2:
        return []
    #interpolate so RMSE < 0.1 pixels
    #minimize number of segments
    #use dynamic programing to find optimal segments
    else:
        segments = [points[0]]
        start_idx = 0
        for end_idx in range(2, len(points)+1):
            segment = points[start_idx:end_idx]
            #fit line to segment
            x_coords, y_coords = zip(*segment)
            coeffs = np.polyfit(x_coords, y_coords, 1)
            #calculate RMSE
            fitted_y = np.polyval(coeffs, x_coords)
            rmse_val = np.sqrt(np.mean((np.array(y_coords) - fitted_y)**2))
            if rmse_val > rmse:
                #add last point before exceeding RMSE
                segments.append(points[end_idx-2])
                start_idx = end_idx - 2
        #add last point
        segments.append(points[-1])
        return segments
    
def minimize_travel_between_lines(lines, min_len=10):
    #always start with longest line
    #input is list of lines (list of points)
    #output is reordered list of lines to minimize travel distance
    #only consider lines longer than min_len
    lines = [line for line in lines if len(line)>=min_len]
    if len(lines)==0:
        return []
    #start with longest line
    lines = sorted(lines, key=len, reverse=True)
    ordered_lines = [lines[0]]
    remaining_lines = lines[1:]
    while remaining_lines:
        last_line = ordered_lines[-1]
        last_point = last_line[-1]
        #find closest line start or end point
        min_dist = float('inf')
        min_idx = -1
        reverse = False
        for idx, line in enumerate(remaining_lines):
            start_point = line[0]
            end_point = line[-1]
            dist_start = np.sqrt((last_point[0]-start_point[0])**2 + (last_point[1]-start_point[1])**2)
            dist_end = np.sqrt((last_point[0]-end_point[0])**2 + (last_point[1]-end_point[1])**2)
            if dist_start < min_dist:
                min_dist = dist_start
                min_idx = idx
                reverse = False
            if dist_end < min_dist:
                min_dist = dist_end
                min_idx = idx
                reverse = True
        next_line = remaining_lines.pop(min_idx)
        if reverse:
            next_line = next_line[::-1]
        ordered_lines.append(next_line)
    return ordered_lines

def minimize_number_of_lines(lines):
    #create a graph with line start and end as nodes and lines as edges
    #use dynamic programming to find the minimum number of lines that cover all points
    #ensure each edge/line is traversed just once 
    #reverse the line if needed
    nodes = []
    edges = []
    for line in lines:
        start = line[0]
        end = line[-1]
        nodes.append(start)
        nodes.append(end)
        edges.append((start, end, line))
    #count order of each node
    from collections import defaultdict
    node_order = defaultdict(int)
    for start, end, line in edges:
        node_order[start] += 1
        node_order[end] += 1
    #find nodes with odd order
    odd_nodes = [node for node, order in node_order.items() if order % 2 == 1]
    #if no odd nodes, return lines as is
    if len(odd_nodes)==0:
        return lines
    #do BFS to find paths between odd nodes
    from collections import deque
    def bfs(start, goal):
        queue = deque([start])
        visited = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path[::-1]
            for neighbor in [end for s, end, line in edges if s == current] + [start for start, e, line in edges if e == current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        return None
    #pair odd nodes and find paths
    paired_paths = []   
    while odd_nodes:
        start_node = odd_nodes.pop()
        min_path = None
        min_goal = None
        for goal_node in odd_nodes:
            path = bfs(start_node, goal_node)
            if path is not None and (min_path is None or len(path) < len(min_path)):
                min_path = path
                min_goal = goal_node
        if min_path is not None:
            paired_paths.append(min_path)
            odd_nodes.remove(min_goal)
    #add paths to edges
    for path in paired_paths:
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1], []))  #empty line for added path
    #reconstruct lines from edges
    used_edges = set()
    new_lines = []
    for start, end, line in edges:
        if (start, end) in used_edges or (end, start) in used_edges:
            continue
        current_line = line.copy()
        used_edges.add((start, end))
        current_end = end
        while True:
            next_edge = None
            for s, e, l in edges:
                if (s == current_end and (s, e) not in used_edges and (e, s) not in used_edges):
                    next_edge = (s, e, l)
                    break
                elif (e == current_end and (s, e) not in used_edges and (e, s) not in used_edges):
                    next_edge = (e, s, l[::-1])  #reverse line
                    break
            if next_edge is None:
                break
            _, next_end, next_line = next_edge
            current_line.extend(next_line[1:])  #avoid duplicating point
            used_edges.add((next_edge[0], next_edge[1]))
            current_end = next_end
        new_lines.append(current_line)
    return new_lines


    

#generate gcode for all lines
all_gcode = [G_CODE_HEADER]
min_line_length = 5 
optimized_lines = minimize_travel_between_lines(lines2, min_len=min_line_length)
for line in optimized_lines:
    simplified_line = points_to_line_segemnts(line, rmse=0.2)
    gcode_lines = line_to_gcode(simplified_line, dpi=90, feed_rate=1500, up_height=0, z_delay_ms=15, round_digits=2)
    all_gcode.extend(gcode_lines)