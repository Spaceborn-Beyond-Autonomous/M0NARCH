"""
H1 Kinematic Chain Definition (Rawan - Kinematics Day 1)

Defines the real kinematic chain of the Unitree H1 humanoid's left leg,
extracted directly from models/unitree_h1/h1.xml body positions and
joint axes. This is the foundation for Forward Kinematics (FK).

Joint order (base -> end effector):
  pelvis -> left_hip_yaw (Z) -> left_hip_roll (X) -> left_hip_pitch (Y)
  -> left_knee (Y) -> left_ankle (Y)

All positions in meters, matching h1.xml <body pos="..."> values exactly.
"""

import numpy as np
from kinematics.transforms import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    homogeneous_transform, transform_point,
)


class Joint:
    """
    A single revolute joint: a fixed offset from the parent frame,
    followed by a rotation about a specified axis.
    """

    def __init__(self, name: str, offset: np.ndarray, axis: str):
        self.name = name
        self.offset = np.array(offset, dtype=float)
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.axis = axis

    def local_transform(self, angle_rad: float) -> np.ndarray:
        """4x4 transform from parent frame to this joint's frame."""
        if self.axis == "x":
            r = rotation_matrix_x(angle_rad)
        elif self.axis == "y":
            r = rotation_matrix_y(angle_rad)
        else:
            r = rotation_matrix_z(angle_rad)
        return homogeneous_transform(r, self.offset)


class KinematicChain:
    """
    An ordered sequence of joints from a base frame to an end effector.
    """

    def __init__(self, name: str, joints: list):
        self.name = name
        self.joints = joints

    def forward_kinematics(self, joint_angles: list) -> np.ndarray:
        """
        Computes the end-effector's 4x4 transform in the base frame,
        given a list of joint angles (radians), one per joint, in order.
        """
        if len(joint_angles) != len(self.joints):
            raise ValueError(
                f"Expected {len(self.joints)} joint angles, got {len(joint_angles)}"
            )

        t = np.eye(4)
        for joint, angle in zip(self.joints, joint_angles):
            t = t @ joint.local_transform(angle)
        return t

    def end_effector_position(self, joint_angles: list) -> np.ndarray:
        """Convenience: just the (x, y, z) position of the end effector."""
        t = self.forward_kinematics(joint_angles)
        return t[:3, 3]


# --- Real H1 left leg chain, built from h1.xml body offsets ---
def build_left_leg_chain() -> KinematicChain:
    joints = [
        Joint("left_hip_yaw",   offset=[0.0,      0.0875,  -0.1742], axis="z"),
        Joint("left_hip_roll",  offset=[0.039468, 0.0,      0.0],    axis="x"),
        Joint("left_hip_pitch", offset=[0.0,      0.11536,  0.0],    axis="y"),
        Joint("left_knee",      offset=[0.0,      0.0,     -0.4],    axis="y"),
        Joint("left_ankle",     offset=[0.0,      0.0,     -0.4],    axis="y"),
    ]
    return KinematicChain("left_leg", joints)


if __name__ == "__main__":
    leg = build_left_leg_chain()

    print("=== H1 Left Leg Forward Kinematics (Day 1) ===\n")

    print("Test 1: All joints at 0 rad (neutral standing pose)")
    pos = leg.end_effector_position([0, 0, 0, 0, 0])
    print("  Ankle position relative to pelvis:", np.round(pos, 4))
    print("  Expected Y offset (hip abduction width): ~0.0875 + 0.11536 =", 0.0875 + 0.11536)
    print("  Expected Z offset (leg length): ~ -0.1742 - 0.4 - 0.4 =", -0.1742 - 0.4 - 0.4)

    print("\nTest 2: Knee bent 45 degrees")
    knee_angle = np.radians(45)
    pos = leg.end_effector_position([0, 0, 0, knee_angle, 0])
    print("  Ankle position with knee bent:", np.round(pos, 4))
    print("  (Z should be less negative than Test 1 -- leg shortened)")

    print("\nTest 3: Hip pitch 20 degrees forward (leg swings forward)")
    hip_pitch = np.radians(20)
    pos = leg.end_effector_position([0, 0, hip_pitch, 0, 0])
    print("  Ankle position with hip pitched forward:", np.round(pos, 4))
    print("  (X should become non-zero -- foot moves forward)")
