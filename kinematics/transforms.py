"""
Transformation Utilities (Rawan - Kinematics Day 1)

Core math library for representing rotations, translations, and
homogeneous transformations between coordinate frames -- the
foundation for Forward/Inverse Kinematics.

Conventions:
- Rotation matrices: 3x3, right-handed, radians
- Quaternions: [w, x, y, z] (scalar-first)
- Homogeneous transforms: 4x4 matrices combining rotation + translation
"""

import numpy as np


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll-pitch-yaw (XYZ Euler angles, radians) to a rotation matrix.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def rotation_matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    Uses Shepperd's method for numerical stability.
    """
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)],
    ])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transformation matrix from a 3x3 rotation
    matrix and a 3-element translation vector.
    """
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def transform_point(t: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3D point."""
    point_h = np.append(point, 1.0)
    return (t @ point_h)[:3]


def invert_transform(t: np.ndarray) -> np.ndarray:
    """
    Efficiently invert a homogeneous transform (avoids full matrix
    inversion by exploiting the rotation/translation structure).
    """
    r = t[:3, :3]
    p = t[:3, 3]
    r_inv = r.T
    p_inv = -r_inv @ p
    return homogeneous_transform(r_inv, p_inv)


if __name__ == "__main__":
    print("=== Transformation Utilities Validation (Day 1) ===\n")

    # Test 1: Euler -> rotation matrix -> quaternion round-trip
    roll, pitch, yaw = np.radians([10, 20, 30])
    r = euler_to_rotation_matrix(roll, pitch, yaw)
    q = rotation_matrix_to_quaternion(r)
    r_back = quaternion_to_rotation_matrix(q)

    print("Test 1: Euler -> Rotation Matrix -> Quaternion -> Rotation Matrix")
    print("  Original R:\n", np.round(r, 4))
    print("  Quaternion [w,x,y,z]:", np.round(q, 4))
    print("  Recovered R:\n", np.round(r_back, 4))
    print("  Match:", np.allclose(r, r_back))

    # Test 2: Homogeneous transform + point transformation
    print("\nTest 2: Homogeneous transform of a point")
    translation = np.array([1.0, 0.0, 0.5])
    t = homogeneous_transform(r, translation)
    point = np.array([0.0, 0.0, 0.0])  # origin of child frame
    world_point = transform_point(t, point)
    print("  Transformed point (child origin in world frame):", np.round(world_point, 4))

    # Test 3: Transform inversion
    print("\nTest 3: Transform inversion (T^-1 @ T should be identity)")
    t_inv = invert_transform(t)
    identity_check = t_inv @ t
    print("  T^-1 @ T:\n", np.round(identity_check, 4))
    print("  Is identity:", np.allclose(identity_check, np.eye(4)))
