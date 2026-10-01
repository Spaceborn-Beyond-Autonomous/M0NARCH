"""
Jacobian Computation (Rawan - Kinematics Day 2)

Computes the geometric Jacobian of a kinematic chain numerically,
using finite differences on the Forward Kinematics function. This
relates joint velocities to end-effector linear velocity:

    end_effector_velocity = J @ joint_velocities

Numerical differentiation is simple, robust, and chain-agnostic --
a good starting point before (optionally) deriving an analytical
Jacobian later for performance.
"""

import numpy as np
from kinematics.h1_chain import KinematicChain


def compute_jacobian(chain: KinematicChain, joint_angles: list,
                      epsilon: float = 1e-6) -> np.ndarray:
    """
    Numerically computes the 3xN positional Jacobian for a chain with
    N joints, using central differences for accuracy.

    Returns:
        3xN numpy array, where column i is d(position)/d(joint_angle_i)
    """
    n = len(joint_angles)
    jacobian = np.zeros((3, n))
    angles = np.array(joint_angles, dtype=float)

    for i in range(n):
        angles_plus = angles.copy()
        angles_minus = angles.copy()
        angles_plus[i] += epsilon
        angles_minus[i] -= epsilon

        pos_plus = chain.end_effector_position(angles_plus.tolist())
        pos_minus = chain.end_effector_position(angles_minus.tolist())

        jacobian[:, i] = (pos_plus - pos_minus) / (2 * epsilon)

    return jacobian


def end_effector_velocity(chain: KinematicChain, joint_angles: list,
                           joint_velocities: list) -> np.ndarray:
    """
    Given current joint angles and joint velocities, computes the
    instantaneous linear velocity of the end effector.
    """
    j = compute_jacobian(chain, joint_angles)
    return j @ np.array(joint_velocities)


if __name__ == "__main__":
    from kinematics.h1_chain import H1Robot

    robot = H1Robot()

    print("=== Jacobian Validation (Day 2) ===\n")

    print("Test 1: Left leg Jacobian at neutral pose")
    j = compute_jacobian(robot.left_leg, [0, 0, 0, 0, 0])
    print("  Jacobian (3x5):\n", np.round(j, 4))

    print("\nTest 2: Predicted vs actual end-effector velocity")
    angles = [0.0, 0.0, 0.0, 0.0, 0.0]
    velocities = [0.0, 0.0, 0.5, 0.0, 0.0]  # only hip_pitch moving at 0.5 rad/s
    dt = 0.001

    predicted_vel = end_effector_velocity(robot.left_leg, angles, velocities)
    print("  Predicted velocity (Jacobian):", np.round(predicted_vel, 5))

    # Validate by finite-difference over a small timestep
    pos_before = robot.left_leg.end_effector_position(angles)
    angles_after = [a + v * dt for a, v in zip(angles, velocities)]
    pos_after = robot.left_leg.end_effector_position(angles_after)
    actual_vel = (pos_after - pos_before) / dt
    print("  Actual velocity (finite diff):", np.round(actual_vel, 5))
    print("  Match:", np.allclose(predicted_vel, actual_vel, atol=1e-3))

    print("\nTest 3: Left arm Jacobian at neutral pose")
    j_arm = compute_jacobian(robot.left_arm, [0, 0, 0, 0])
    print("  Jacobian (3x4):\n", np.round(j_arm, 4))
