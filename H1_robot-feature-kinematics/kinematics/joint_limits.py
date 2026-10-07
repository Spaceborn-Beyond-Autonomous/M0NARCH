"""
Joint Limits (Rawan - Kinematics Days 5-6)

Real joint limits extracted directly from models/unitree_h1/h1.xml
<joint range="min max"> values (radians). Used to constrain IK
solutions to physically valid configurations.
"""

# (min_rad, max_rad) per joint, in the same order as each chain's
# joint list in kinematics/h1_chain.py
LEFT_LEG_LIMITS = [
    (-0.43, 0.43),   # left_hip_yaw
    (-0.43, 0.43),   # left_hip_roll
    (-1.57, 1.57),   # left_hip_pitch
    (-0.26, 2.05),   # left_knee
    (-0.87, 0.52),   # left_ankle
]

RIGHT_LEG_LIMITS = [
    (-0.43, 0.43),   # right_hip_yaw
    (-0.43, 0.43),   # right_hip_roll
    (-1.57, 1.57),   # right_hip_pitch
    (-0.26, 2.05),   # right_knee
    (-0.87, 0.52),   # right_ankle
]

LEFT_ARM_LIMITS = [
    (-2.87, 2.87),   # left_shoulder_pitch
    (-0.34, 3.11),   # left_shoulder_roll
    (-1.30, 4.45),   # left_shoulder_yaw
    (-1.25, 2.61),   # left_elbow
]

RIGHT_ARM_LIMITS = [
    (-2.87, 2.87),   # right_shoulder_pitch
    (-3.11, 0.34),   # right_shoulder_roll  (mirrored vs left)
    (-4.45, 1.30),   # right_shoulder_yaw   (mirrored vs left)
    (-1.25, 2.61),   # right_elbow
]

JOINT_LIMITS_BY_CHAIN = {
    "left_leg": LEFT_LEG_LIMITS,
    "right_leg": RIGHT_LEG_LIMITS,
    "left_arm": LEFT_ARM_LIMITS,
    "right_arm": RIGHT_ARM_LIMITS,
}


def is_within_limits(chain_name: str, angles: list) -> bool:
    """Check whether a set of joint angles respects the chain's limits."""
    limits = JOINT_LIMITS_BY_CHAIN[chain_name]
    return all(lo <= a <= hi for a, (lo, hi) in zip(angles, limits))


def clip_to_limits(chain_name: str, angles: list) -> list:
    """Clip a set of joint angles to the chain's valid range."""
    limits = JOINT_LIMITS_BY_CHAIN[chain_name]
    return [max(lo, min(a, hi)) for a, (lo, hi) in zip(angles, limits)]


if __name__ == "__main__":
    print("=== Joint Limits Validation (Days 5-6) ===\n")

    print("Test 1: Valid left leg pose (all zeros, within limits)")
    print("  Within limits:", is_within_limits("left_leg", [0, 0, 0, 0, 0]))

    print("\nTest 2: Invalid knee angle (3.0 rad, exceeds max 2.05)")
    test_angles = [0, 0, 0, 3.0, 0]
    print("  Within limits:", is_within_limits("left_leg", test_angles))
    print("  Clipped:", clip_to_limits("left_leg", test_angles))

    print("\nTest 3: Right arm mirrored limits check")
    print("  Left shoulder_roll limits:", LEFT_ARM_LIMITS[1])
    print("  Right shoulder_roll limits:", RIGHT_ARM_LIMITS[1])
    print("  Correctly mirrored:",
          LEFT_ARM_LIMITS[1] == (-RIGHT_ARM_LIMITS[1][1], -RIGHT_ARM_LIMITS[1][0]))
