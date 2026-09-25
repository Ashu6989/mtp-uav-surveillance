"""
assignment_env.py

Gymnasium environment for assigning a fleet of UAVs to candidate viewpoints
to maximize ROI cell coverage, following a sequential MDP formulation.
Supports action masking for connectivity and mutual-exclusion constraints.
"""

from __future__ import annotations

import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines 3 imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Import domain modules
from warehouse import Warehouse, create_test_warehouse
from viewpoints import (
    ROICell, Viewpoint, create_test_rois, build_roi_cells,
    generate_candidate_viewpoints, thin_candidates
)
from visibility import compute_visibility_matrix


def compute_distance_vector(uav_position: np.ndarray, viewpoints: list[Viewpoint]) -> np.ndarray:
    """
    Computes straight-line Euclidean distance from the UAV start position 
    to every viewpoint. Designed to be swapped with A* grid distances later.
    """
    vp_positions = np.array([vp.position for vp in viewpoints])
    return np.linalg.norm(vp_positions - uav_position, axis=1)


class ViewpointAssignmentEnv(gym.Env):
    """
    Sequential MDP for assigning UAVs to viewpoints.
    One step = one UAV assignment. Episode terminates when all UAVs are assigned.
    """
    def __init__(self, uav_start_positions: list[np.ndarray], viewpoints: list[Viewpoint], 
                 roi_cells: list[ROICell], vis_matrix: np.ndarray, k_g: int | np.ndarray, 
                 warehouse: Warehouse, wC: float = 1.0, wL: float = 0.1, 
                 w_minus: float = 1.0, B: float = 50.0):
        super().__init__()
        
        self.uav_start_positions = uav_start_positions
        self.viewpoints = viewpoints
        self.roi_cells = roi_cells
        self.vis_matrix = vis_matrix
        self.warehouse = warehouse
        
        self.N = len(uav_start_positions)
        self.P = len(viewpoints)
        self.G = len(roi_cells)
        
        self.wC = wC
        self.wL = wL
        self.w_minus = w_minus
        self.B = B
        
        # Format required coverage array
        if isinstance(k_g, (int, float)):
            self.k_g = np.full(self.G, k_g, dtype=float)
        else:
            self.k_g = np.array(k_g, dtype=float)
            
        # Precompute region IDs for mask matching
        self.uav_regions = np.array([self.warehouse.region_of(pos) for pos in self.uav_start_positions])
        self.vp_regions = np.array([vp.region for vp in self.viewpoints])
        
        # State Arrays
        self.z = np.zeros(self.G, dtype=int)
        self.mu = np.zeros(self.P, dtype=int)
        self.current_uav_idx = 0
        
        # Observation Space: [z_hat_g (G), mu (P), s_u (3), d_u (P)]
        obs_dim = self.G + self.P + 3 + self.P
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action Space: Choose one viewpoint index
        self.action_space = spaces.Discrete(self.P)

    def _get_obs(self) -> np.ndarray:
        """Constructs the flattened float32 state observation."""
        # 1. Normalized coverage
        z_hat = np.minimum(self.z, self.k_g) / self.k_g
        
        # 2. Taken mask (self.mu)
        
        # 3 & 4. Current UAV position and distance vector
        if self.current_uav_idx < self.N:
            s_u = self.uav_start_positions[self.current_uav_idx]
            d_u = compute_distance_vector(s_u, self.viewpoints)
        else:
            # Dummy padded values for terminal state evaluation
            s_u = np.zeros(3)
            d_u = np.zeros(self.P)
            
        return np.concatenate([z_hat, self.mu, s_u, d_u]).astype(np.float32)

    def _get_info(self) -> dict:
        """Returns standard dict for tracking state inside evaluation callbacks."""
        C_K = np.sum(self.z >= self.k_g) / self.G if self.G > 0 else 0.0
        S_minus = np.sum(np.maximum(0, self.k_g - self.z))
        return {
            "C_K": C_K,
            "S_minus": S_minus,
            "z_g": self.z.copy()
        }

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean array of valid actions (viewpoints) for the active UAV.
        Valid = not taken by another UAV AND in the same connected free-space region.
        """
        if self.current_uav_idx >= self.N:
            return np.zeros(self.P, dtype=bool)
            
        active_uav_region = self.uav_regions[self.current_uav_idx]
        region_match = (self.vp_regions == active_uav_region)
        not_taken = (self.mu == 0)
        
        return region_match & not_taken

    def _advance_to_valid_uav(self):
        """Advances current_uav_idx past any UAVs that have 0 valid assignment options."""
        while self.current_uav_idx < self.N:
            if np.any(self.action_masks()):
                break
            else:
                warnings.warn(f"Skipping UAV {self.current_uav_idx} - no valid viewpoints in its region.")
                self.current_uav_idx += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z = np.zeros(self.G, dtype=int)
        self.mu = np.zeros(self.P, dtype=int)
        self.current_uav_idx = 0
        
        self._advance_to_valid_uav()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # Defensive check against invalid actions (e.g. if masking isn't strictly enforced)
        if self.current_uav_idx >= self.N or not self.action_masks()[action]:
            return self._get_obs(), -1000.0, True, False, self._get_info()
            
        active_uav_pos = self.uav_start_positions[self.current_uav_idx]
        d_u = compute_distance_vector(active_uav_pos, self.viewpoints)
        travel_dist = d_u[action]
        
        # Compute delta coverage
        old_clamped = np.minimum(self.z, self.k_g)
        
        vis_row = self.vis_matrix[action]
        self.z += vis_row
        
        new_clamped = np.minimum(self.z, self.k_g)
        delta_C_t = np.sum(new_clamped - old_clamped)
        
        # Step reward
        reward = self.wC * delta_C_t - self.wL * travel_dist
        
        # Finalize assignment
        self.mu[action] = 1
        self.current_uav_idx += 1
        self._advance_to_valid_uav()
        
        # Check termination
        terminated = (self.current_uav_idx >= self.N)
        
        # Compute terminal rewards if episode ends
        if terminated:
            info = self._get_info()
            if info["C_K"] == 1.0:
                reward += self.B
            else:
                reward -= self.w_minus * info["S_minus"]
                
        return self._get_obs(), reward, terminated, False, self._get_info()


def build_env_from_test_scenario() -> ViewpointAssignmentEnv:
    """
    Constructs the end-to-end environment by reusing previous setup logic.
    """
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
    
    vis_matrix = compute_visibility_matrix(
        viewpoints=candidates,
        roi_cells=roi_cells,
        warehouse=warehouse,
        max_range=8.0,
        max_incidence_deg=75.0
    )
    
    # Setup two UAVs at either end of the aisle (Y = 7.5)
    uav_starts = [
        np.array([1.0, 7.5, 1.0]),
        np.array([19.0, 7.5, 1.0])
    ]
    
    # Ensure they're valid based on warehouse safety checks
    for i, pos in enumerate(uav_starts):
        if not warehouse.is_valid_position(pos, 0.5):
            print(f"Warning: UAV {i} pos {pos} invalid! Proceeding anyway, but behavior may be undefined.")
        else:
            print(f"UAV {i} Start mapped successfully at {pos} (Region {warehouse.region_of(pos)})")

    # Wrap the base environment with an ActionMasker lambda expected by sb3_contrib
    def mask_fn(env: ViewpointAssignmentEnv) -> np.ndarray:
        return env.action_masks()

    env = ViewpointAssignmentEnv(
        uav_start_positions=uav_starts,
        viewpoints=candidates,
        roi_cells=roi_cells,
        vis_matrix=vis_matrix,
        k_g=1,  # Single coverage for this test
        warehouse=warehouse,
        wC=1.0, wL=0.1, w_minus=1.0, B=50.0
    )
    
    return ActionMasker(env, mask_fn)


if __name__ == "__main__":
    print("\n=== 1. Building Environment ===")
    masked_env = build_env_from_test_scenario()
    base_env = masked_env.unwrapped  # to access raw stats easily
    
    print(f"\nConfiguration:")
    print(f"UAVs: {base_env.N}")
    print(f"ROI Cells: {base_env.G}")
    print(f"Candidate Viewpoints: {base_env.P}")
    
    # -------------
    # RANDOM POLICY
    # -------------
    print("\n=== 2. Running Baseline (Random Masked Policy) ===")
    obs, info = masked_env.reset()
    done = False
    random_assignments = []
    total_rand_reward = 0.0
    
    while not done:
        mask = masked_env.action_masks()
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            action = masked_env.action_space.sample() # will fail cleanly and terminate
        else:
            action = np.random.choice(valid_actions)
            
        random_assignments.append(action)
        obs, reward, terminated, truncated, info = masked_env.step(action)
        total_rand_reward += reward
        done = terminated or truncated
        
    print(f"Random Actions Taken: {random_assignments}")
    print(f"Random Total Reward: {total_rand_reward:.2f}")
    print(f"Random Coverage fraction (C_K): {info['C_K']*100:.1f}%")
    
    # -------------
    # PPO TRAINING
    # -------------
    print("\n=== 3. Training MaskablePPO Agent ===")
    model = MaskablePPO("MlpPolicy", masked_env, gamma=0.99, verbose=1)
    
    # 20k steps is plenty for a toy 2-step MDP episode size to memorize best splits
    model.learn(total_timesteps=20_000) 
    
    # -------------
    # PPO EVALUATION
    # -------------
    print("\n=== 4. Evaluating Trained Policy ===")
    obs, info = masked_env.reset()
    done = False
    ppo_assignments = []
    total_ppo_reward = 0.0
    
    while not done:
        # Use deterministic=True for maximum expectation evaluation
        action, _states = model.predict(obs, action_masks=masked_env.action_masks(), deterministic=True)
        ppo_assignments.append(int(action))
        
        obs, reward, terminated, truncated, info = masked_env.step(action)
        total_ppo_reward += reward
        done = terminated or truncated
        
    print(f"Trained Actions Taken: {ppo_assignments}")
    print(f"Trained Total Reward: {total_ppo_reward:.2f}")
    print(f"Final C_K (Coverage Fraction): {info['C_K']*100:.1f}%")
    print(f"Final S_minus (Missing Coverage): {info['S_minus']}")
    print(f"Final Cell Coverage array (z_g):\n{info['z_g']}")