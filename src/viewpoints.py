"""
viewpoints.py

Module for generating planar Regions of Interest (ROIs) and generating/thinning 
candidate UAV viewpoints for multi-UAV surveillance path planning.
Depends on the warehouse.py module for geometric and collision queries.
"""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import cKDTree

# Import dependencies from the warehouse module
from warehouse import Warehouse, Obstacle, create_test_warehouse


@dataclass
class ROICell:
    """
    Represents a single discretized patch of a Region of Interest.
    """
    position: np.ndarray  # 3D position (x, y, z)
    normal: np.ndarray    # 3D outward-facing unit normal vector
    roi_id: int           # ID of the parent ROI
    cell_id: int          # Sequential ID within the parent ROI


class PlanarROI:
    """
    Represents a rectangular surveillance patch on an obstacle (e.g., rack face).
    """
    def __init__(self, roi_id: int, corner: np.ndarray, edge_u: np.ndarray, edge_v: np.ndarray, cell_spacing: float):
        self.roi_id = roi_id
        self.corner = np.array(corner, dtype=float)
        self.edge_u = np.array(edge_u, dtype=float)
        self.edge_v = np.array(edge_v, dtype=float)
        self.cell_spacing = float(cell_spacing)
        
        # Compute the outward normal vector (cross product of spanning edges)
        cross_prod = np.cross(self.edge_u, self.edge_v)
        norm_length = np.linalg.norm(cross_prod)
        if norm_length < 1e-6:
            raise ValueError("edge_u and edge_v must not be collinear or zero-length.")
        self.normal = cross_prod / norm_length

    def generate_cells(self) -> list[ROICell]:
        """
        Discretizes the rectangular ROI into a grid of cells at cell_spacing resolution.
        Cells are placed at the center of their respective grid patches.
        """
        len_u = np.linalg.norm(self.edge_u)
        len_v = np.linalg.norm(self.edge_v)
        
        # Determine number of subdivisions (at least 1 cell per dimension)
        n_u = max(1, int(np.ceil(len_u / self.cell_spacing)))
        n_v = max(1, int(np.ceil(len_v / self.cell_spacing)))
        
        # Grid coordinates linearly spaced within the spanning vectors
        u_fracs = np.linspace(0.5 / n_u, 1.0 - 0.5 / n_u, n_u)
        v_fracs = np.linspace(0.5 / n_v, 1.0 - 0.5 / n_v, n_v)
        
        cells = []
        cell_id = 0
        for uf in u_fracs:
            for vf in v_fracs:
                # Interpolate 3D position from fractional weights
                pos = self.corner + uf * self.edge_u + vf * self.edge_v
                cells.append(ROICell(position=pos, normal=self.normal, roi_id=self.roi_id, cell_id=cell_id))
                cell_id += 1
                
        return cells


def build_roi_cells(rois: list[PlanarROI]) -> list[ROICell]:
    """
    Iterates over all provided ROIs, generates their cells, and flattens them into a single list.
    """
    all_cells = []
    for roi in rois:
        all_cells.extend(roi.generate_cells())
    return all_cells


@dataclass
class Viewpoint:
    """
    Represents a candidate 3D position for a UAV camera.
    """
    id: int
    position: np.ndarray
    region: int


def generate_candidate_viewpoints(warehouse: Warehouse, roi_cells: list[ROICell], 
                                  safety_distance: float, standoff_distances: list[float], 
                                  lateral_offsets: list[float]) -> list[Viewpoint]:
    """
    Generates candidate viewpoints by projecting outward from each ROI cell's normal.
    Tests various standoff distances and lateral grid offsets.
    Ensures candidates are collision-free and maintain the required safety distance.
    """
    candidates = []
    raw_count = 0
    vp_id = 0
    
    for cell in roi_cells:
        n_g = cell.normal
        
        # Compute two orthogonal basis vectors on the plane perpendicular to the normal.
        # Use Z-axis as the default reference "up" vector.
        ref = np.array([0.0, 0.0, 1.0])
        
        # If the normal is parallel/anti-parallel to Z, use X-axis as the reference instead.
        if np.abs(np.dot(n_g, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])
            
        b1 = np.cross(ref, n_g)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(n_g, b1) # Already normalized since n_g and b1 are orthogonal unit vectors
        
        for d in standoff_distances:
            for u in lateral_offsets:
                for v in lateral_offsets:
                    raw_count += 1
                    
                    # Generate candidate position
                    pt = cell.position + (d * n_g) + (u * b1) + (v * b2)
                    
                    # Keep if valid and safe
                    if warehouse.is_valid_position(pt, safety_distance):
                        region = warehouse.region_of(pt)
                        candidates.append(Viewpoint(id=vp_id, position=pt, region=region))
                        vp_id += 1
                        
    print(f"Candidate Generation: Tested {raw_count} geometry-projected points -> {len(candidates)} valid candidates survived.")
    return candidates


def thin_candidates(candidates: list[Viewpoint], min_spacing: float) -> list[Viewpoint]:
    """
    Greedily reduces candidate density to ensure manageable complexity for later stages.
    Enforces a minimum spatial distance between any two surviving candidates.
    """
    if not candidates:
        return []
        
    kept = []
    kept_positions = []
    
    for cand in candidates:
        if not kept_positions:
            kept.append(cand)
            kept_positions.append(cand.position)
            continue
            
        # Compute distance to all currently kept candidates
        dists = np.linalg.norm(np.array(kept_positions) - cand.position, axis=1)
        if np.min(dists) >= min_spacing:
            kept.append(cand)
            kept_positions.append(cand.position)
            
    # Re-assign sequential IDs to the thinned list
    for i, cand in enumerate(kept):
        cand.id = i
        
    print(f"Thinning Candidates: Reduced {len(candidates)} -> {len(kept)} viewpoints (min_spacing={min_spacing}m).")
    return kept


def plot_scene_3d(warehouse: Warehouse, roi_cells: list[ROICell], candidates: list[Viewpoint]):
    """
    Visualizes the warehouse geometry, ROI patches (with normals), and candidate viewpoints 
    colored by their connected-component region.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Helper function to draw wireframe boxes (borrowed from warehouse.py logic)
    def draw_wire_box(min_c, max_c, color, alpha=1.0):
        pts = list(itertools.product(*zip(min_c, max_c)))
        for p1, p2 in itertools.combinations(pts, 2):
            if sum(c1 != c2 for c1, c2 in zip(p1, p2)) == 1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=1.5)

    # Draw warehouse bounds
    draw_wire_box((0, 0, 0), warehouse.dimensions, color='black', alpha=0.3)
    
    # Draw obstacles
    for obs in warehouse.obstacles:
        draw_wire_box(obs.min_corner, obs.max_corner, color='red', alpha=0.6)

    # Scatter and orient ROI Cells
    if roi_cells:
        rx, ry, rz = zip(*[c.position for c in roi_cells])
        nx, ny, nz = zip(*[c.normal for c in roi_cells])
        
        ax.scatter(rx, ry, rz, color='green', s=10, label='ROI Cells', zorder=5)
        # Small quiver arrows indicating outward normals (what the camera must face)
        ax.quiver(rx, ry, rz, nx, ny, nz, length=0.5, color='green', alpha=0.8, arrow_length_ratio=0.3, normalize=True)

    # Scatter candidate viewpoints colored by region
    if candidates:
        vx, vy, vz = zip(*[c.position for c in candidates])
        regions = [c.region for c in candidates]
        
        scatter = ax.scatter(vx, vy, vz, c=regions, cmap='tab10', s=25, alpha=0.9, label='Candidate Viewpoints', zorder=4)
        
        # Generate legend for regions
        unique_regions = set(regions)
        if len(unique_regions) > 1:
            legend1 = ax.legend(*scatter.legend_elements(), title="Free Regions", loc="upper left")
            ax.add_artist(legend1)

    # Equal scaling
    ax.set_box_aspect(warehouse.dimensions)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Warehouse Surveillance ROI & Candidate Viewpoints")
    
    # Simple general legend
    handles, labels = ax.get_legend_handles_labels()
    # Dedup labels just in case
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()


def create_test_rois(warehouse: Warehouse) -> list[PlanarROI]:
    """
    Builds ROIs on the AISLE-FACING sides of the 4 racks from create_test_warehouse().
    The aisle is between y=5 and y=10.
    """
    rois = []
    
    # Front Row Racks (x in [2,10]/[12,18], y in [4,5], z in [0,3])
    # The aisle faces +Y. So normal should point to +Y (0, 1, 0)
    
    # Rack 1 Front Aisle Face
    roi1 = PlanarROI(
        roi_id=0,
        corner=np.array([2.0, 5.0, 0.0]),
        edge_u=np.array([0.0, 0.0, 3.0]),
        edge_v=np.array([8.0, 0.0, 0.0]),
        cell_spacing=0.5
    )
    rois.append(roi1)
    
    # Rack 2 Front Aisle Face
    roi2 = PlanarROI(
        roi_id=1,
        corner=np.array([12.0, 5.0, 0.0]),
        edge_u=np.array([0.0, 0.0, 3.0]),
        edge_v=np.array([6.0, 0.0, 0.0]),
        cell_spacing=0.5
    )
    rois.append(roi2)

    # Back Row Racks (x in [2,10]/[12,18], y in [10,11], z in [0,3])
    # The aisle faces -Y. So normal should point to -Y (0, -1, 0)

    # Rack 3 Back Aisle Face
    roi3 = PlanarROI(
        roi_id=2,
        corner=np.array([2.0, 10.0, 0.0]),
        edge_u=np.array([8.0, 0.0, 0.0]),  
        edge_v=np.array([0.0, 0.0, 3.0]), 
        cell_spacing=0.5
    )
    rois.append(roi3)
    
    # Rack 4 Back Aisle Face
    roi4 = PlanarROI(
        roi_id=3,
        corner=np.array([12.0, 10.0, 0.0]),
        edge_u=np.array([6.0, 0.0, 0.0]), 
        edge_v=np.array([0.0, 0.0, 3.0]), 
        cell_spacing=0.5
    )
    rois.append(roi4)
    
    return rois


if __name__ == "__main__":
    print("--- 1. Initializing Warehouse ---")
    warehouse = create_test_warehouse()
    
    print("\n--- 2. Generating ROIs ---")
    rois = create_test_rois(warehouse)
    roi_cells = build_roi_cells(rois)
    print(f"Total ROI Cells Generated: {len(roi_cells)}")
    
    print("\n--- 3. Generating Candidate Viewpoints ---")
    candidates = generate_candidate_viewpoints(
        warehouse=warehouse, 
        roi_cells=roi_cells, 
        safety_distance=0.5, 
        standoff_distances=[1.5, 3.0, 5.0], 
        lateral_offsets=[-1.0, 0.0, 1.0]
    )
    
    print("\n--- 4. Thinning Candidates ---")
    thinned_candidates = thin_candidates(candidates, min_spacing=1.0)
    
    # Print distinct region properties
    if thinned_candidates:
        unique_regions = set(c.region for c in thinned_candidates)
        print(f"Distinct connected regions spanned by candidates: {len(unique_regions)} {list(unique_regions)}")
    else:
        print("No candidates survived thinning (or generation)!")
    
    print("\n--- 5. Rendering 3D Scene ---")
    plot_scene_3d(warehouse, roi_cells, thinned_candidates)