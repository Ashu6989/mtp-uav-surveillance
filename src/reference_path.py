"""
reference_path.py

Implements 3D A* pathfinding over the continuous/discrete warehouse environment 
to generate collision-free reference paths for UAVs to track towards their 
assigned viewpoints. Includes line-of-sight path simplification.
"""

from __future__ import annotations

import heapq
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Import existing domain models
from warehouse import Warehouse, create_test_warehouse
from visibility import has_line_of_sight
from viewpoints import (create_test_rois, build_roi_cells, 
                        generate_candidate_viewpoints, thin_candidates)


def get_nearest_valid_voxel(warehouse: Warehouse, point: np.ndarray, safety_distance: float) -> tuple[int, int, int] | None:
    """
    Finds the closest voxel to a given point that is both unoccupied and maintains 
    the safety distance. Used to snap invalid start/goal positions.
    Searches outward in a concentric grid pattern.
    """
    base_idx = warehouse.world_to_voxel(point)
    nx, ny, nz = warehouse.grid_shape
    
    # Check radii from 0 up to 4 voxels away
    for r in range(5):
        # Generate surface of the cube with radius r (L-infinity norm)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                        
                    idx = (base_idx[0] + dx, base_idx[1] + dy, base_idx[2] + dz)
                    if (0 <= idx[0] < nx) and (0 <= idx[1] < ny) and (0 <= idx[2] < nz):
                        pt = warehouse.voxel_to_world(idx)
                        if warehouse.is_valid_position(pt, safety_distance):
                            return idx
    return None


def astar_path(warehouse: Warehouse, start: np.ndarray, goal: np.ndarray, safety_distance: float) -> list[np.ndarray] | None:
    """
    Computes a 3D A* path from start to goal over the warehouse voxel grid.
    Cost is Euclidean distance. Voxels are valid only if they maintain safety_distance.
    """
    start_voxel = get_nearest_valid_voxel(warehouse, start, safety_distance)
    goal_voxel = get_nearest_valid_voxel(warehouse, goal, safety_distance)
    
    if start_voxel is None or goal_voxel is None:
        print("A* Error: Could not find valid safe start/goal voxels.")
        return None

    # Pre-generate 26-connectivity neighbor offsets and their Euclidean grid costs
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cost = np.linalg.norm(np.array([dx, dy, dz]) * warehouse.resolution)
                directions.append(((dx, dy, dz), cost))

    nx, ny, nz = warehouse.grid_shape
    goal_pos = warehouse.voxel_to_world(goal_voxel)
    
    # Priority queue: (f_score, tie_breaker, voxel_tuple)
    open_set = []
    counter = itertools.count()
    heapq.heappush(open_set, (0.0, next(counter), start_voxel))
    
    came_from = {}
    g_score = {start_voxel: 0.0}
    
    # Cache for validity checking to avoid redundant geometric obstacle loops
    validity_cache = {start_voxel: True, goal_voxel: True}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == goal_voxel:
            # Reconstruct path
            path_voxels = [current]
            while current in came_from:
                current = came_from[current]
                path_voxels.append(current)
            path_voxels.reverse()
            
            # Convert to continuous world coordinates
            path_world = [warehouse.voxel_to_world(v) for v in path_voxels]
            
            # Insert exact continuous start/goal if they don't perfectly overlap
            if np.linalg.norm(path_world[0] - start) > 1e-4:
                if warehouse.is_valid_position(start, safety_distance):
                    path_world.insert(0, np.array(start))
                else:
                    print("  [WARNING] True start position wasn't safety-compliant, so the path begins from the nearest safe point instead.")
                    
            if np.linalg.norm(path_world[-1] - goal) > 1e-4:
                if warehouse.is_valid_position(goal, safety_distance):
                    path_world.append(np.array(goal))
                else:
                    print("  [WARNING] True goal position wasn't safety-compliant, so the path ends at the nearest safe point instead.")
                    
            return path_world

        for (dx, dy, dz), move_cost in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Bounds check
            if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny and 0 <= neighbor[2] < nz):
                continue
                
            # Validity check (with caching)
            if neighbor not in validity_cache:
                pt = warehouse.voxel_to_world(neighbor)
                validity_cache[neighbor] = warehouse.is_valid_position(pt, safety_distance)
                
            if not validity_cache[neighbor]:
                continue
                
            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Heuristic: continuous Euclidean distance to the goal voxel
                neighbor_pos = warehouse.voxel_to_world(neighbor)
                h_score = np.linalg.norm(neighbor_pos - goal_pos)
                f_score = tentative_g + h_score
                
                heapq.heappush(open_set, (f_score, next(counter), neighbor))

    return None


def simplify_path(path: list[np.ndarray], warehouse: Warehouse, safety_distance: float) -> list[np.ndarray]:
    """
    Greedily simplifies a jagged A* path using line-of-sight shortcutting.
    Ensures straight-line shortcuts maintain the required safety clearance.
    """
    if len(path) <= 2:
        return path
        
    simplified = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        # Look backwards from the end of the path to find the furthest valid shortcut
        for j in range(len(path) - 1, i, -1):
            p1 = path[i]
            p2 = path[j]
            
            # 1. Broad phase: strict line intersection with obstacle bounding boxes
            if not has_line_of_sight(warehouse, p1, p2):
                continue
                
            # 2. Narrow phase: continuous sampling to ensure safety_distance clearance
            segment_vec = p2 - p1
            dist = np.linalg.norm(segment_vec)
            
            is_safe = True
            if dist > 0:
                # Sample every 0.1 meters
                num_samples = int(np.ceil(dist / 0.1))
                if num_samples > 0:
                    step_vec = segment_vec / num_samples
                    for step in range(1, num_samples):
                        sample_pt = p1 + step * step_vec
                        if warehouse.distance_to_obstacles(sample_pt) < safety_distance:
                            is_safe = False
                            break
                            
            if is_safe:
                simplified.append(p2)
                i = j
                break
                
    return simplified


def path_length(path: list[np.ndarray]) -> float:
    """Computes the total Euclidean length of a waypoint sequence."""
    if not path or len(path) < 2:
        return 0.0
    pts = np.array(path)
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def compute_all_reference_paths(uav_start_positions: list[np.ndarray], 
                                assigned_viewpoints: list[np.ndarray], 
                                warehouse: Warehouse, 
                                safety_distance: float) -> list[list[np.ndarray]]:
    """
    Generates simplified reference paths for all UAVs simultaneously.
    """
    all_paths = []
    
    print(f"\n--- Computing Reference Paths (Safety Clearance: {safety_distance}m) ---")
    for idx, (start, goal) in enumerate(zip(uav_start_positions, assigned_viewpoints)):
        print(f"UAV {idx}: Start {np.round(start, 1)} -> Goal {np.round(goal, 1)}")
        
        raw_path = astar_path(warehouse, start, goal, safety_distance)
        if raw_path is None:
            print(f"  [WARNING] Unreachable goal for UAV {idx}. Skipping.")
            all_paths.append([])
            continue
            
        sim_path = simplify_path(raw_path, warehouse, safety_distance)
        all_paths.append(sim_path)
        
        r_len = path_length(raw_path)
        s_len = path_length(sim_path)
        
        print(f"  A* Raw Path:      {len(raw_path):3d} nodes | {r_len:.2f}m")
        print(f"  Simplified Path:  {len(sim_path):3d} nodes | {s_len:.2f}m")
        
    return all_paths


def plot_reference_paths(warehouse: Warehouse, uav_start_positions: list[np.ndarray], 
                         assigned_viewpoints: list[np.ndarray], paths: list[list[np.ndarray]]):
    """
    Visualizes the warehouse environment and the resulting reference paths for each UAV.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("tab10")

    # Draw wireframe boxes for the environment
    def draw_wire_box(min_c, max_c, color, alpha=1.0):
        pts = list(itertools.product(*zip(min_c, max_c)))
        for p1, p2 in itertools.combinations(pts, 2):
            if sum(c1 != c2 for c1, c2 in zip(p1, p2)) == 1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color=color, alpha=alpha, linewidth=1.5)

    draw_wire_box((0, 0, 0), warehouse.dimensions, color='black', alpha=0.3)
    for obs in warehouse.obstacles:
        draw_wire_box(obs.min_corner, obs.max_corner, color='red', alpha=0.6)

    # Plot paths and start/goal markers
    for idx, path in enumerate(paths):
        if not path:
            continue
            
        color = cmap(idx % 10)
        start = uav_start_positions[idx]
        goal = assigned_viewpoints[idx]
        
        # Start marker (Circle)
        ax.scatter(*start, color=color, s=100, marker='o', edgecolors='black', label=f'UAV {idx} Start')
        # Goal marker (Star)
        ax.scatter(*goal, color=color, s=200, marker='*', edgecolors='black', label=f'UAV {idx} Goal')
        
        # Path line
        px, py, pz = zip(*path)
        ax.plot(px, py, pz, color=color, linewidth=2.5, linestyle='-', marker='.', markersize=8)

    ax.set_box_aspect(warehouse.dimensions)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("UAV Reference Paths (Obstacle Aware)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== 1. Initializing Environment ===")
    warehouse = create_test_warehouse()
    rois = create_test_rois(warehouse)
    roi_cells = build_roi_cells(rois)
    
    raw_candidates = generate_candidate_viewpoints(
        warehouse=warehouse, 
        roi_cells=roi_cells, 
        safety_distance=0.5, 
        standoff_distances=[1.5, 3.0, 5.0], 
        lateral_offsets=[-1.0, 0.0, 1.0]
    )
    candidates = thin_candidates(raw_candidates, min_spacing=1.0)
    
    print("\n=== 2. Setting up UAV Starts and Goals ===")
    uav_starts = [
        np.array([1.0, 7.5, 1.0]),
        np.array([19.0, 7.5, 1.0]),
        np.array([5.0, 3.0, 1.0])  # UAV 2: Obstacle avoidance test start
    ]
    
    # Pick viewpoints 106 and 37 safely based on array length
    if not candidates:
        raise ValueError("Candidate generation failed, cannot test paths.")
        
    vp_idx_1 = min(106, len(candidates) - 1)
    vp_idx_2 = min(37, len(candidates) - 1)
    
    assigned_goals = [
        candidates[vp_idx_1].position,
        candidates[vp_idx_2].position,
        np.array([5.0, 12.0, 1.0])  # UAV 2: Obstacle avoidance test goal
    ]
    
    print(f"UAV 0 assigned to Viewpoint Index {vp_idx_1}")
    print(f"UAV 1 assigned to Viewpoint Index {vp_idx_2}")
    print(f"UAV 2 assigned to Manual Goal (Obstacle Avoidance Test)")
    
    print("\n=== 3. Executing A* and Simplification ===")
    paths = compute_all_reference_paths(
        uav_start_positions=uav_starts, 
        assigned_viewpoints=assigned_goals, 
        warehouse=warehouse, 
        safety_distance=0.5
    )
    
    print("\n=== 4. Cost Comparison (Obstacle-Aware vs Straight-Line) ===")
    total_sim_len = 0.0
    total_sl_len = 0.0
    
    for idx, (path, start, goal) in enumerate(zip(paths, uav_starts, assigned_goals)):
        if path:
            sl_len = np.linalg.norm(goal - start)
            sim_len = path_length(path)
            
            total_sl_len += sl_len
            total_sim_len += sim_len
            
            overhead = sim_len - sl_len
            print(f"UAV {idx}:")
            print(f"  Straight-Line: {sl_len:.2f}m")
            print(f"  Path Routed:   {sim_len:.2f}m (Overhead: +{overhead:.2f}m)")
            
    print(f"\nFleet Combined Path Length: {total_sim_len:.2f}m")
    print(f"Fleet Combined Straight-Line: {total_sl_len:.2f}m")
    
    print("\n=== 5. Rendering Paths ===")
    plot_reference_paths(warehouse, uav_starts, assigned_goals, paths)