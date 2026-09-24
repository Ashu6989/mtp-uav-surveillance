"""
warehouse.py

A modular 3D warehouse environment representation for multi-UAV path planning.
Handles continuous space geometry, voxelized occupancy grids, obstacle avoidance, 
and connected component mapping.
"""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.ndimage import label, generate_binary_structure


@dataclass
class Obstacle:
    """
    Represents a static axis-aligned cuboid obstacle (rack, column, wall) in the warehouse.
    """
    min_corner: np.ndarray
    max_corner: np.ndarray

    def __post_init__(self):
        # Ensure inputs are always numpy arrays of floats for vectorized operations
        self.min_corner = np.array(self.min_corner, dtype=float)
        self.max_corner = np.array(self.max_corner, dtype=float)

    def contains(self, point: np.ndarray) -> bool:
        """
        Checks if a given 3D point is inside the obstacle (including its boundary).
        """
        p = np.array(point)
        return bool(np.all(p >= self.min_corner) and np.all(p <= self.max_corner))

    def distance_to(self, point: np.ndarray) -> float:
        """
        Computes the shortest Euclidean distance from a point to the obstacle's surface.
        Returns 0.0 if the point is strictly inside the obstacle.
        """
        p = np.array(point)
        # Clamping the point to the bounding box gives the closest point on the box
        clamped_point = np.clip(p, self.min_corner, self.max_corner)
        return float(np.linalg.norm(p - clamped_point))


class Warehouse:
    """
    Core environment model representing the warehouse 3D volume, maintaining both 
    continuous geometric structures and a discrete voxel grid for planning.
    """
    def __init__(self, dimensions: tuple[float, float, float], obstacles: list[Obstacle], resolution: float = 0.25):
        self.dimensions = np.array(dimensions, dtype=float)
        self.obstacles = obstacles
        self.resolution = float(resolution)
        
        # Grid shape: ceil(dimension / resolution)
        self.grid_shape = (
            int(np.ceil(self.dimensions[0] / self.resolution)),
            int(np.ceil(self.dimensions[1] / self.resolution)),
            int(np.ceil(self.dimensions[2] / self.resolution))
        )
        
        self.occupancy_grid = self._build_occupancy_grid()
        self._region_labels: Optional[np.ndarray] = None

    def _build_occupancy_grid(self) -> np.ndarray:
        """
        Builds the 3D boolean occupancy array. 
        True = occupied (overlaps an obstacle or out of warehouse bounds). False = free.
        """
        nx, ny, nz = self.grid_shape
        grid = np.zeros((nx, ny, nz), dtype=bool)

        # 1D boundaries for all voxels along each axis
        x_min = np.arange(nx) * self.resolution
        x_max = x_min + self.resolution
        y_min = np.arange(ny) * self.resolution
        y_max = y_min + self.resolution
        z_min = np.arange(nz) * self.resolution
        z_max = z_min + self.resolution

        # Vectorized AABB collision check for each obstacle against all voxels
        for obs in self.obstacles:
            x_overlap = (x_min < obs.max_corner[0]) & (x_max > obs.min_corner[0])
            y_overlap = (y_min < obs.max_corner[1]) & (y_max > obs.min_corner[1])
            z_overlap = (z_min < obs.max_corner[2]) & (z_max > obs.min_corner[2])

            ix = np.where(x_overlap)[0]
            iy = np.where(y_overlap)[0]
            iz = np.where(z_overlap)[0]

            if len(ix) > 0 and len(iy) > 0 and len(iz) > 0:
                grid[np.ix_(ix, iy, iz)] = True

        # Enforce that voxels whose centers fall strictly outside warehouse dimensions are marked occupied
        cx = (np.arange(nx) + 0.5) * self.resolution
        cy = (np.arange(ny) + 0.5) * self.resolution
        cz = (np.arange(nz) + 0.5) * self.resolution
        
        grid[cx > self.dimensions[0], :, :] = True
        grid[:, cy > self.dimensions[1], :] = True
        grid[:, :, cz > self.dimensions[2]] = True

        return grid

    def world_to_voxel(self, point: np.ndarray) -> tuple[int, int, int]:
        """Converts a continuous 3D world coordinate into a discrete voxel grid index."""
        idx = np.floor(np.array(point) / self.resolution).astype(int)
        return tuple(idx.tolist())

    def voxel_to_world(self, voxel_idx: tuple[int, int, int]) -> np.ndarray:
        """Converts a discrete voxel grid index into a continuous 3D world coordinate (voxel center)."""
        return (np.array(voxel_idx) + 0.5) * self.resolution

    def is_free(self, point: np.ndarray) -> bool:
        """
        Checks if a continuous 3D point is within bounds and outside all obstacles.
        Uses exact geometric tests, independent of grid resolution.
        """
        p = np.array(point)
        # Check warehouse bounds
        if np.any(p < 0.0) or np.any(p > self.dimensions):
            return False
        # Check obstacles
        for obs in self.obstacles:
            if obs.contains(p):
                return False
        return True

    def distance_to_obstacles(self, point: np.ndarray) -> float:
        """
        Returns the shortest distance from the point to any obstacle.
        Returns a very large number (1e6) if the environment is entirely empty.
        """
        if not self.obstacles:
            return 1e6
        p = np.array(point)
        return min(obs.distance_to(p) for obs in self.obstacles)

    def is_valid_position(self, point: np.ndarray, safety_distance: float) -> bool:
        """
        Checks if a point is free AND maintains a safe clearance from all obstacles.
        """
        return self.is_free(point) and (self.distance_to_obstacles(point) >= safety_distance)

    def find_connected_components(self) -> np.ndarray:
        """
        Runs a 3D flood-fill (26-connectivity) to segment disconnected regions of free space.
        Returns an array where occupied space is -1, and free regions are labeled 0, 1, 2, ...
        Results are cached lazily.
        """
        if self._region_labels is not None:
            return self._region_labels

        # 3x3x3 connectivity matrix for 26-way adjacency
        struct = generate_binary_structure(3, 3)
        free_grid = ~self.occupancy_grid
        
        # scipy.ndimage.label labels regions starting from 1
        labels, num_features = label(free_grid, structure=struct)

        # Map occupied space to -1, and shift free space labels to start from 0
        self._region_labels = np.full(self.occupancy_grid.shape, -1, dtype=int)
        self._region_labels[free_grid] = labels[free_grid] - 1
        
        return self._region_labels

    def region_of(self, point: np.ndarray) -> int:
        if not self.is_free(point):
            return -1
            
        i, j, k = self.world_to_voxel(point)
        nx, ny, nz = self.grid_shape
        i = min(max(i, 0), nx - 1)
        j = min(max(j, 0), ny - 1)
        k = min(max(k, 0), nz - 1)
        labels = self.find_connected_components()

        return int(labels[i, j, k])


def plot_warehouse_3d(warehouse: Warehouse, show_grid: bool = False):
    """
    Renders a simple 3D wireframe and solid representation of the warehouse and obstacles.
    If show_grid=True, scatters sub-sampled free voxel centers.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Helper function to draw wireframe boxes
    def draw_wire_box(min_c, max_c, color, alpha=1.0):
        # Generate 8 corners
        pts = list(itertools.product(*zip(min_c, max_c)))
        # Connect vertices that differ by exactly 1 coordinate
        for p1, p2 in itertools.combinations(pts, 2):
            if sum(c1 != c2 for c1, c2 in zip(p1, p2)) == 1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=1.5)

    # Draw warehouse bounding box
    draw_wire_box((0, 0, 0), warehouse.dimensions, color='black', alpha=0.5)

    # Draw obstacles
    for obs in warehouse.obstacles:
        draw_wire_box(obs.min_corner, obs.max_corner, color='red', alpha=0.9)

    # Optionally draw free grid voxels
    if show_grid:
        free_indices = np.argwhere(~warehouse.occupancy_grid)
        # Vectorized lookup of centers
        free_centers = (free_indices + 0.5) * warehouse.resolution
        
        # Sub-sample to avoid crashing matplotlib
        max_pts = 5000
        if len(free_centers) > max_pts:
            sample_idxs = np.random.choice(len(free_centers), max_pts, replace=False)
            free_centers = free_centers[sample_idxs]
            
        if len(free_centers) > 0:
            ax.scatter(free_centers[:, 0], free_centers[:, 1], free_centers[:, 2], 
                       c='blue', s=2, alpha=0.1, label='Free Space (Subsampled)')
            ax.legend()

    # Scaling axes evenly
    ax.set_box_aspect(warehouse.dimensions)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Warehouse Topology ({warehouse.dimensions[0]}x{warehouse.dimensions[1]}x{warehouse.dimensions[2]}m)")
    plt.tight_layout()
    plt.show()


def create_test_warehouse() -> Warehouse:
    """
    Factory function producing a standard 20x15x5 test warehouse featuring 
    two parallel rows of racks with an intersecting main aisle.
    """
    # 4 standard racks: long and thin (8m x 1m x 3m). 
    obstacles = [
        # Front Row
        Obstacle(np.array([2.0, 4.0, 0.0]), np.array([10.0, 5.0, 3.0])),
        Obstacle(np.array([12.0, 4.0, 0.0]), np.array([18.0, 5.0, 3.0])),
        
        # Back Row
        Obstacle(np.array([2.0, 10.0, 0.0]), np.array([10.0, 11.0, 3.0])),
        Obstacle(np.array([12.0, 10.0, 0.0]), np.array([18.0, 11.0, 3.0])),
    ]
    return Warehouse(dimensions=(20.0, 15.0, 5.0), obstacles=obstacles, resolution=0.25)


if __name__ == "__main__":
    print("Building test warehouse...")
    warehouse = create_test_warehouse()
    
    print(f"Dimensions: {warehouse.dimensions} m")
    print(f"Resolution: {warehouse.resolution} m/voxel")
    print(f"Occupancy grid shape: {warehouse.occupancy_grid.shape}")
    
    total_voxels = np.prod(warehouse.grid_shape)
    free_voxels = np.sum(~warehouse.occupancy_grid)
    print(f"Space utilized by obstacles: {(1 - free_voxels/total_voxels)*100:.1f}%")
    
    labels = warehouse.find_connected_components()
    num_regions = labels.max() + 1
    print(f"Number of Connected Free Regions detected: {num_regions}")
    
    # Example Point Lookup
    test_pt = (5.0, 2.0, 1.0)
    print(f"\nEvaluating point {test_pt}:")
    print(f" - Is free? {warehouse.is_free(test_pt)}")
    print(f" - Dist to nearest obstacle: {warehouse.distance_to_obstacles(test_pt):.2f}m")
    print(f" - Region label: {warehouse.region_of(test_pt)}")

    print("\nRendering 3D visualization...")
    plot_warehouse_3d(warehouse, show_grid=True)