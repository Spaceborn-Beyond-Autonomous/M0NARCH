"""
H1 Kinematic Chain Definition (Rawan - Kinematics Day 1)

Defines the complete kinematic chains of the Unitree H1 humanoid
(both legs, both arms), extracted directly from
models/unitree_h1/h1.xml body positions, fixed orientations (quat),
and joint axes. This is the foundation for Forward Kinematics (FK).

Note: h1.xml has no wrist/hand body -- elbow_link is currently the
end-effector for each arm chain. A forearm+hand link must be added
to the model before wrist-level manipulation (Phase 5) is possible.
"""

import numpy as np
from kinematics.transforms import (
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    quaternion_to_rotation_matrix,
    homogeneous_transform, transform_point,
)


class Joint:
    def __init__(self, name, offset, axis, base_quat=None):
        self.name = name
        self.offset = np.array(offset, dtype=float)
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.axis = axis
        self.base_rotation = (
            quaternion_to_rotation_matrix(np.array(base_quat))
            if base_quat is not None else np.eye(3)
        )

    def local_transform(self, angle_rad):
        if self.axis == "x":
            r_joint = rotation_matrix_x(angle_rad)
        elif self.axis == "y":
            r_joint = rotation_matrix_y(angle_rad)
        else:
            r_joint = rotation_matrix_z(angle_rad)
        r_total = self.base_rotation @ r_joint
        return homogeneous_transform(r_total, self.offset)


class KinematicChain:
    def __init__(self, name, joints):
        self.name = name
        self.joints = joints

    def forward_kinematics(self, joint_angles):
        if len(joint_angles) != len(self.joints):
            raise ValueError(
                f"Expected {len(self.joints)} joint angles, got {len(joint_angles)}"
            )
        t = np.eye(4)
        for joint, angle in zip(self.joints, joint_angles):
            t = t @ joint.local_transform(angle)
        return t

    def end_effector_position(self, joint_angles):
        t = self.forward_kinematics(joint_angles)
        return t[:3, 3]


def build_left_leg_chain():
    joints = [
        Joint("left_hip_yaw",   offset=[0.0,      0.0875,  -0.1742], axis="z"),
        Joint("left_hip_roll",  offset=[0.039468, 0.0,      0.0],    axis="x"),
        Joint("left_hip_pitch", offset=[0.0,      0.11536,  0.0],    axis="y"),
        Joint("left_knee",      offset=[0.0,      0.0,     -0.4],    axis="y"),
        Joint("left_ankle",     offset=[0.0,      0.0,     -0.4],    axis="y"),
    ]
    return KinematicChain("left_leg", joints)


def build_right_leg_chain():
    joints = [
        Joint("right_hip_yaw",   offset=[0.0,      -0.0875,  -0.1742], axis="z"),
        Joint("right_hip_roll",  offset=[0.039468,  0.0,      0.0],    axis="x"),
        Joint("right_hip_pitch", offset=[0.0,      -0.11536,  0.0],    axis="y"),
        Joint("right_knee",      offset=[0.0,       0.0,     -0.4],    axis="y"),
        Joint("right_ankle",     offset=[0.0,       0.0,     -0.4],    axis="y"),
    ]
    return KinematicChain("right_leg", joints)


def build_left_arm_chain():
    joints = [
        Joint("left_shoulder_pitch", offset=[0.0055, 0.15535, 0.42999],
              axis="y", base_quat=[0.976296, 0.216438, 0.0, 0.0]),
        Joint("left_shoulder_roll", offset=[-0.0055, 0.0565, -0.0165],
              axis="x", base_quat=[0.976296, -0.216438, 0.0, 0.0]),
        Joint("left_shoulder_yaw", offset=[0.0, 0.0, -0.1343], axis="z"),
        Joint("left_elbow", offset=[0.0185, 0.0, -0.198], axis="y"),
    ]
    return KinematicChain("left_arm", joints)


def build_right_arm_chain():
    joints = [
        Joint("right_shoulder_pitch", offset=[0.0055, -0.15535, 0.42999],
              axis="y", base_quat=[0.976296, -0.216438, 0.0, 0.0]),
        Joint("right_shoulder_roll", offset=[-0.0055, -0.0565, -0.0165],
              axis="x", base_quat=[0.976296, 0.216438, 0.0, 0.0]),
        Joint("right_shoulder_yaw", offset=[0.0, 0.0, -0.1343], axis="z"),
        Joint("right_elbow", offset=[0.0185, 0.0, -0.198], axis="y"),
    ]
    return KinematicChain("right_arm", joints)


class H1Robot:
    def __init__(self):
        self.left_leg = build_left_leg_chain()
        self.right_leg = build_right_leg_chain()
        self.left_arm = build_left_arm_chain()
        self.right_arm = build_right_arm_chain()

    def all_chains(self):
        return {
            "left_leg": self.left_leg,
            "right_leg": self.right_leg,
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
        }

    def neutral_pose_positions(self):
        return {
            name: chain.end_effector_position([0.0] * len(chain.joints))
            for name, chain in self.all_chains().items()
        }


if __name__ == "__main__":
    robot = H1Robot()

    print("=== H1 Full-Body Forward Kinematics (Day 1) ===\n")

    print("Neutral pose end-effector positions (relative to pelvis):")
    for name, pos in robot.neutral_pose_positions().items():
        print(f"  {name:12s}: {np.round(pos, 4)}")

    print("\nSanity checks:")
    print("  Left/Right leg mirror in Y:",
          np.allclose(robot.neutral_pose_positions()["left_leg"][1],
                       -robot.neutral_pose_positions()["right_leg"][1]))
    print("  Left/Right arm mirror in Y:",
          np.allclose(robot.neutral_pose_positions()["left_arm"][1],
                       -robot.neutral_pose_positions()["right_arm"][1]))

    print("\nTest: Left arm with shoulder pitch rotated 45 degrees")
    shoulder_pitch_angle = np.radians(45)
    pos = robot.left_arm.end_effector_position([shoulder_pitch_angle, 0, 0, 0])
    print("  Elbow position with shoulder rotated:", np.round(pos, 4))
    print("  (Should differ from neutral [0.0185, 0.2135, 0.1066])")

    print("\nNote: h1.xml has no wrist/hand body -- elbow_link is the")
    print("current end-effector. A forearm+hand link must be added to")
    print("the model before wrist-level manipulation (Phase 5) is possible.")
