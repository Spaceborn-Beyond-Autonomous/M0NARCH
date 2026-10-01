"""
Inverse Kinematics Solver (Rawan - Kinematics Days 3-4)

Solves IK using the Damped Least Squares (DLS) method: iteratively
adjusts joint angles to reduce the error between the current and
target end-effector position, using the Jacobian computed in Day 2.

DLS is more numerically stable than plain Jacobian pseudo-inverse,
especially near singularities (e.g., a fully extended leg/arm).
"""

import numpy as np
from kinematics.h1_chain import KinematicChain
from kinematics.jacobian import compute_jacobian


def solve_ik(chain: KinematicChain, target_position: np.ndarray,
             initial_angles: list = None, max_iterations: int = 200,
             damping: float = 0.05, tolerance: float = 1e-4,
             joint_limits: list = None) -> dict:
    """
    Damped Least Squares IK solver.

    Args:
        chain: the KinematicChain to solve for
        target_position: desired (x, y, z) end-effector position
        initial_angles: starting guess (defaults to all zeros)
        max_iterations: solver iteration cap
        damping: DLS damping factor (lambda) -- higher = more stable,
                 slower convergence; lower = faster, less stable near
                 singularities
        tolerance: stop when position error is below this (meters)
        joint_limits: optional list of (min_rad, max_rad) per joint

    Returns:
        dict with final_angles, final_position, error, converged, iterations
    """
    n = len(chain.joints)
    target = np.array(target_position, dtype=float)

    angles = np.array(initial_angles, dtype=float) if initial_angles else np.zeros(n)

    converged = False
    error_norm = float("inf")

    for iteration in range(max_iterations):
        current_pos = chain.end_effector_position(angles.tolist())
        error = target - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            converged = True
            break

        j = compute_jacobian(chain, angles.tolist())

        # Damped least squares: delta_theta = J^T (J J^T + lambda^2 I)^-1 * error
        jjt = j @ j.T
        damped = jjt + (damping ** 2) * np.eye(3)
        delta_angles = j.T @ np.linalg.solve(damped, error)

        angles = angles + delta_angles

        if joint_limits is not None:
            for i, (lo, hi) in enumerate(joint_limits):
                angles[i] = np.clip(angles[i], lo, hi)

    final_position = chain.end_effector_position(angles.tolist())

    return {
        "final_angles": angles.tolist(),
        "final_position": final_position,
        "error_m": round(float(error_norm), 6),
        "converged": converged,
        "iterations": iteration + 1,
    }


if __name__ == "__main__":
    from kinematics.h1_chain import H1Robot

    robot = H1Robot()

    print("=== IK Solver Validation (Days 3-4) ===\n")

    print("Test 1: Left leg IK -- reach the neutral standing position")
    neutral_pos = robot.left_leg.end_effector_position([0, 0, 0, 0, 0])
    result = solve_ik(robot.left_leg, target_position=neutral_pos,
                       initial_angles=[0.1, 0.1, 0.1, 0.1, 0.1])  # start off-target
    print(f"  Target: {np.round(neutral_pos, 4)}")
    print(f"  Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"  Final error: {result['error_m']} m")
    print(f"  Final angles (rad): {np.round(result['final_angles'], 4)}")

    print("\nTest 2: Left leg IK -- reach a forward+down position (step forward)")
    target = np.array([0.15, 0.2029, -0.85])  # foot steps forward and up slightly
    result = solve_ik(robot.left_leg, target_position=target)
    print(f"  Target: {target}")
    print(f"  Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"  Final position: {np.round(result['final_position'], 4)}")
    print(f"  Final error: {result['error_m']} m")

    print("\nTest 3: Left arm IK -- reach forward (e.g. to grasp an object)")
    target_arm = np.array([0.2, 0.25, 0.3])  # confirmed within reachable workspace
    result = solve_ik(robot.left_arm, target_position=target_arm,
                       max_iterations=300)
    print(f"  Target: {target_arm}")
    print(f"  Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"  Final position: {np.round(result['final_position'], 4)}")
    print(f"  Final error: {result['error_m']} m")

    print("\nTest 4: Unreachable target (too far away)")
    target_far = np.array([5.0, 5.0, 5.0])
    result = solve_ik(robot.left_arm, target_position=target_far)
    print(f"  Target: {target_far}")
    print(f"  Converged: {result['converged']} (expected False -- out of reach)")
    print(f"  Final error: {result['error_m']} m")
