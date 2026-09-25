"""
visibility.py

Computes the visibility matrix for a multi-UAV path planning project.
Evaluates Range, Field of View (incidence angle), and Line-of-Sight constraints 
between candidate viewpoints and ROI cells.
"""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt

# Import existing domain models
from warehouse import Warehouse, Obstacle, create_test_warehouse
from viewpoints import (ROICell, Viewpoint, create_test_rois, build_roi_cells,
                        generate_candidate_viewpoints, thin_candidates)


def segment_intersects_obstacle(p1: np.ndarray, p2: np.ndarray, obstacle: Obstacle) -> bool:
    """
    Standard 3D segment-vs-AABB intersection test using the slab method.
    Returns True if the line segment from p1 to p2 passes strictly through 
    the obstacle's interior volume.
    """
    d = p2 - p1
    length = np.linalg.norm(d)
    
    # If the segment is practically a point, just check if it's inside
    if length < 1e-8:
        return obstacle.contains(p1)
        
    d_norm = d / length
    
    # Prevent division by zero for axis-aligned segments
    d_safe = np.where(np.abs(d_norm) < 1e-9, 1e-9, d_norm)
    
    t1 = (obstacle.min_corner - p1) / d_safe
    t2 = (obstacle.max_corner - p1) / d_safe
    
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    
    t_near = np.max(t_min)
    t_far = np.min(t_max)
    
    # We consider the segment blocked if it intersects the obstacle's volume 
    # STRICTLY between the viewpoint (t=0) and the ROI cell (t=length).
    # We use a small epsilon to ignore intersections at the exact endpoints, 
    # because ROI cells are typically located exactly on the boundary of an obstacle.
    eps = 1e-4
    if t_near <= t_far and t_near < (length - eps) and t_far > eps:
        return True
        
    return False


def has_line_of_sight(warehouse: Warehouse, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Returns True if the line segment between p1 and p2 does not intersect 
    any obstacles in the warehouse.
    """
    for obs in warehouse.obstacles:
        if segment_intersects_obstacle(p1, p2, obs):
            return False
    return True


def is_visible(viewpoint: Viewpoint, cell: ROICell, warehouse: Warehouse, 
               max_range: float, max_incidence_deg: float) -> bool:
    """
    Determines if an ROI cell is visible from a viewpoint.
    Checks (in order): Range, Incidence Angle (FoV proxy), and Line of Sight.
    """
    # 1. RANGE CHECK
    vec = viewpoint.position - cell.position
    dist = np.linalg.norm(vec)
    if dist > max_range:
        return False
        
    # 2. INCIDENCE ANGLE CHECK (FoV Constraint)
    if dist < 1e-6:
        return False
    # View direction is FROM the cell TO the viewpoint
    view_dir = vec / dist 
    max_cos = np.cos(np.radians(max_incidence_deg))
    
    # The normal points outward from the surface. The camera must look AT the surface.
    if np.dot(cell.normal, view_dir) < max_cos:
        return False
        
    # 3. LINE OF SIGHT CHECK (Raycast)
    # Expensive check done last
    if not has_line_of_sight(warehouse, viewpoint.position, cell.position):
        return False
        
    return True


def compute_visibility_matrix(viewpoints: list[Viewpoint], roi_cells: list[ROICell], 
                              warehouse: Warehouse, max_range: float, 
                              max_incidence_deg: float = 75.0) -> np.ndarray:
    """
    Computes the binary visibility matrix a_mg for all viewpoints and ROI cells.
    Returns a boolean numpy array of shape (len(viewpoints), len(roi_cells)).
    """
    P = len(viewpoints)
    G = len(roi_cells)
    a_matrix = np.zeros((P, G), dtype=bool)
    
    print(f"Computing visibility matrix for {P} viewpoints and {G} cells...")
    
    print_interval = max(1, P // 5)  # 20% intervals
    
    for m, vp in enumerate(viewpoints):
        if m % print_interval == 0 and m > 0:
            print(f"  Processed {m}/{P} viewpoints ({(m / P) * 100:.0f}%)")
            
        for g, cell in enumerate(roi_cells):
            a_matrix[m, g] = is_visible(vp, cell, warehouse, max_range, max_incidence_deg)
            
    print("  Processing complete.")
    
    # Calculate Diagnostics
    total_true = np.sum(a_matrix)
    avg_cells_per_vp = total_true / P if P > 0 else 0
    avg_vp_per_cell = total_true / G if G > 0 else 0
    zero_coverage_cells = np.sum(np.sum(a_matrix, axis=0) == 0)
    
    print("\n--- Visibility Statistics ---")
    print(f"Matrix shape: {a_matrix.shape}")
    print(f"Total visible pairs: {total_true}")
    print(f"Average cells visible per viewpoint: {avg_cells_per_vp:.2f}")
    print(f"Average viewpoints seeing each cell: {avg_vp_per_cell:.2f}")
    print(f"Cells with ZERO visibility coverage: {zero_coverage_cells} / {G}")
    
    return a_matrix


def plot_visibility_for_viewpoint(warehouse: Warehouse, viewpoints: list[Viewpoint], 
                                  roi_cells: list[ROICell], vis_matrix: np.ndarray, 
                                  viewpoint_id: int):
    """
    Visualizes the warehouse environment, highlighting a single viewpoint and coloring
    the ROI cells based on whether they are visible from that viewpoint.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Helper function to draw wireframe boxes
    def draw_wire_box(min_c, max_c, color, alpha=1.0):
        pts = list(itertools.product(*zip(min_c, max_c)))
        for p1, p2 in itertools.combinations(pts, 2):
            if sum(c1 != c2 for c1, c2 in zip(p1, p2)) == 1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        color=color, alpha=alpha, linewidth=1.5)

    # Draw warehouse bounds and obstacles
    draw_wire_box((0, 0, 0), warehouse.dimensions, color='black', alpha=0.3)
    for obs in warehouse.obstacles:
        draw_wire_box(obs.min_corner, obs.max_corner, color='red', alpha=0.6)

    # Plot the specific viewpoint
    vp = viewpoints[viewpoint_id]
    ax.scatter([vp.position[0]], [vp.position[1]], [vp.position[2]], 
               c='orange', s=150, edgecolor='black', label=f'Viewpoint {viewpoint_id}', zorder=10)

    # Separate visible and occluded cells for coloring
    vis_mask = vis_matrix[viewpoint_id, :]
    vis_cells = [c for i, c in enumerate(roi_cells) if vis_mask[i]]
    invis_cells = [c for i, c in enumerate(roi_cells) if not vis_mask[i]]
    
    if vis_cells:
        vx, vy, vz = zip(*[c.position for c in vis_cells])
        ax.scatter(vx, vy, vz, c='green', s=25, label='Visible Cells', zorder=5)
        
    if invis_cells:
        ix, iy, iz = zip(*[c.position for c in invis_cells])
        ax.scatter(ix, iy, iz, c='gray', s=15, alpha=0.4, label='Occluded/Out-of-Range', zorder=4)

    # Equal scaling
    ax.set_box_aspect(warehouse.dimensions)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Cell Visibility from Viewpoint {viewpoint_id} (Sees {len(vis_cells)} cells)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- 1. Building Environment ---")
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
    
    print("\n--- 2. Thinning Candidates ---")
    candidates = thin_candidates(raw_candidates, min_spacing=1.0)
    
    print("\n--- 3. Computing Visibility Matrix ---")
    a_matrix = compute_visibility_matrix(
        viewpoints=candidates,
        roi_cells=roi_cells,
        warehouse=warehouse,
        max_range=8.0,
        max_incidence_deg=75.0
    )
    
    # Find a median viewpoint to plot
    counts = a_matrix.sum(axis=1)
    
    # Filter to indices that see at least one cell, then sort by count
    valid_indices = np.where(counts > 0)[0]
    
    if len(valid_indices) > 0:
        sorted_valid = sorted(valid_indices, key=lambda idx: counts[idx])
        median_vp_id = sorted_valid[len(sorted_valid) // 2]
        
        print(f"\n--- 4. Rendering Visualization ---")
        print(f"Selected Viewpoint {median_vp_id} for plotting. It sees {counts[median_vp_id]} cells (median among active).")
        plot_visibility_for_viewpoint(warehouse, candidates, roi_cells, a_matrix, median_vp_id)
    else:
        print("\nNo viewpoints see any cells. Cannot render a meaningful diagnostic plot.")