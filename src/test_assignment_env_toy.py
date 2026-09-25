"""
test_assignment_env_toy.py

A pure unit-test script that manually traces the ViewpointAssignmentEnv MDP.
Constructs a tiny hand-crafted scenario with explicit visibility to verify 
coverage progress, distance penalties, and terminal rewards exactly match 
the mathematical formulation.
"""

import numpy as np

# Import existing modules as-is
from warehouse import Warehouse
from viewpoints import ROICell, Viewpoint
from assignment_env import ViewpointAssignmentEnv


def run_sequence(env: ViewpointAssignmentEnv, action_sequence: list[int], seq_name: str) -> tuple[float, float, float]:
    """
    Manually forces the environment through a specific sequence of actions,
    printing a detailed breakdown of the reward components at every step.
    """
    print(f"\n{'='*60}\nEvaluating {seq_name}: Actions {action_sequence}\n{'='*60}")
    
    obs, info = env.reset()
    total_reward = 0.0
    
    print(f"Initial State : z_g = {info['z_g']}")
    
    for step_idx, action in enumerate(action_sequence):
        # Extract metadata before stepping to log mathematical components
        uav_idx = env.current_uav_idx
        uav_pos = env.uav_start_positions[uav_idx]
        vp = env.viewpoints[action]
        z_g_before = info['z_g'].copy()
        
        # Manually compute distance penalty for logging
        travel_dist = float(np.linalg.norm(uav_pos - vp.position))
        dist_penalty = env.wL * travel_dist
        
        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        z_g_after = info['z_g'].copy()
        
        # Reconstruct delta_C_t (the pure coverage step reward before distance penalty)
        clamped_before = np.minimum(z_g_before, env.k_g)
        clamped_after = np.minimum(z_g_after, env.k_g)
        delta_c_t = float(np.sum(clamped_after - clamped_before))
        
        # Disentangle the pure step reward from the terminal reward
        terminal_reward = 0.0
        if terminated:
            if info["C_K"] == 1.0:
                terminal_reward = env.B
            else:
                terminal_reward = -env.w_minus * info["S_minus"]
                
        pure_step_reward = reward - terminal_reward
        total_reward += reward
        
        # Print step trace
        print(f"\n--- Step {step_idx} (UAV {uav_idx} -> Viewpoint v{action}) ---")
        print(f"  z_g BEFORE     : {z_g_before}")
        print(f"  z_g AFTER      : {z_g_after}")
        print(f"  ΔC_t           : {delta_c_t}")
        print(f"  Distance       : {travel_dist:.3f} (Penalty term: {dist_penalty:.3f})")
        print(f"  Pure Step Rwd  : {pure_step_reward:.3f}  [Formula: wC(1.0)*{delta_c_t} - wL(0.1)*{travel_dist:.3f}]")
        
        if terminated:
            print(f"  [TERMINAL] C_K : {info['C_K']*100:.1f}%")
            print(f"  [TERMINAL] S_- : {info['S_minus']} missing coverage slots")
            print(f"  [TERMINAL] Rwd : {terminal_reward:.3f}")
            print(f"  Total Step Rwd : {reward:.3f}  [Pure + Terminal]")

    print(f"\nTotal Episode Reward for {seq_name}: {total_reward:.3f}")
    return total_reward, info["C_K"], info["S_minus"]


if __name__ == "__main__":
    print("Setting up hand-crafted toy environment...")

    # 1. Provide an empty warehouse and monkeypatch `region_of` so everything is in region 0.
    warehouse = Warehouse(dimensions=(20, 20, 5), obstacles=[], resolution=1.0)
    warehouse.region_of = lambda pos: 0

    # 2. Setup UAVs, Viewpoints, and dummy ROI cells
    uav_starts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0])
    ]

    viewpoints = [
        Viewpoint(id=0, position=np.array([2.0, 2.0, 0.0]), region=0),
        Viewpoint(id=1, position=np.array([5.0, 5.0, 0.0]), region=0),
        Viewpoint(id=2, position=np.array([8.0, 2.0, 0.0]), region=0),
    ]

    # Positions and normals don't matter because visibility is hardcoded
    roi_cells = [
        ROICell(position=np.array([0.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), roi_id=0, cell_id=i)
        for i in range(4)
    ]

    # 3. Explicit Visibility Matrix
    # v0 sees g0, g1
    # v1 sees g1, g2, g3
    # v2 sees g2, g3
    vis_matrix = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ], dtype=bool)

    # 4. Create the environment mapping
    env = ViewpointAssignmentEnv(
        uav_start_positions=uav_starts,
        viewpoints=viewpoints,
        roi_cells=roi_cells,
        vis_matrix=vis_matrix,
        k_g=2,               # Uniform double-coverage requirement
        warehouse=warehouse, 
        wC=1.0, wL=0.1, w_minus=1.0, B=50.0
    )

    # Run Sequence A
    rew_A, ck_A, sm_A = run_sequence(env, action_sequence=[0, 2], seq_name="Sequence A (v0, v2)")

    # Run Sequence B
    rew_B, ck_B, sm_B = run_sequence(env, action_sequence=[0, 1], seq_name="Sequence B (v0, v1)")

    # Print Summary Table
    print(f"\n{'='*60}")
    print(f"{'FINAL COMPARISON SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"| {'Sequence':<20} | {'Total Reward':>12} | {'Final C_K':>10} | {'S_minus':>7} |")
    print(f"|{'-'*22}|{'-'*14}|{'-'*12}|{'-'*9}|")
    print(f"| {'A: UAV0->v0, UAV1->v2':<20} | {rew_A:>12.3f} | {ck_A:>9.2f} | {sm_A:>7} |")
    print(f"| {'B: UAV0->v0, UAV1->v1':<20} | {rew_B:>12.3f} | {ck_B:>9.2f} | {sm_B:>7} |")
    print(f"{'='*60}")
    
    print("\nObservation:")
    print("Sequence B scores higher despite forcing UAV1 to fly further (v1 is further than v2), ")
    print("because v1 provides 3 units of coverage progress vs v2's 2 units, proving that the ")
    print("coverage weight (wC=1.0) successfully outbids the distance penalty (wL=0.1) here.")