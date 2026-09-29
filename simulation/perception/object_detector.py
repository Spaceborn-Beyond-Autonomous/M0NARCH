"""
Object Detection Module
Detects target objects (e.g. manipulation objects) and computes 3D coordinates.
"""

from typing import Optional, Dict, Tuple, Any
import numpy as np


class ObjectDetector:
    def __init__(
        self,
        target_color_rgb: Tuple[int, int, int] = (200, 40, 40),
        color_threshold: float = 80.0,
    ):
        """
        Detects objects based on color signature and extracts 3D centroid.
        Compatible with both OpenCV (if installed) and pure NumPy.
        """
        self.target_color = np.array(target_color_rgb, dtype=np.float32)
        self.threshold = color_threshold

    def detect_in_image(
        self, rgb_image: np.ndarray, depth_image: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Detects target object in RGB-D frame.
        Returns 2D bounding box, centroid, and estimated 3D position relative to camera.
        """
        h, w, _ = rgb_image.shape

        # Color difference mask
        diff = np.linalg.norm(rgb_image.astype(np.float32) - self.target_color, axis=-1)
        mask = diff < self.threshold

        if not np.any(mask):
            return None

        # Calculate 2D centroid (u, v)
        v_indices, u_indices = np.where(mask)
        u_center = float(np.mean(u_indices))
        v_center = float(np.mean(v_indices))

        # Sample depth at centroid
        u_int = int(np.clip(u_center, 0, w - 1))
        v_int = int(np.clip(v_center, 0, h - 1))
        z_depth = float(depth_image[v_int, u_int])

        # Approximate pinhole camera deprojection (assuming 60-deg FOV)
        fov_rad = np.deg2rad(60.0)
        focal_length = (w / 2.0) / np.tan(fov_rad / 2.0)

        # 3D coordinates in camera frame: X (right), Y (down), Z (forward)
        x_cam = (u_center - (w / 2.0)) * z_depth / focal_length
        y_cam = (v_center - (h / 2.0)) * z_depth / focal_length

        return {
            "detected": True,
            "bbox_2d": (int(np.min(u_indices)), int(np.min(v_indices)), int(np.max(u_indices)), int(np.max(v_indices))),
            "centroid_2d": (u_center, v_center),
            "depth_meters": z_depth,
            "position_3d_camera": np.array([x_cam, y_cam, z_depth], dtype=np.float32),
        }

    def compute_6d_pose_in_torso(
        self,
        detection: Dict[str, Any],
        camera_offset_in_torso: np.ndarray = np.array([0.15, 0.0, 0.45], dtype=np.float32),
    ) -> Dict[str, Any]:
        """
        Transforms 3D camera optical coordinates into 6D pose in torso/base frame.
        Essential handoff for Manipulation, Inverse Kinematics (IK), and MoveIt 2.
        
        Optical frame: X=right, Y=down, Z=forward
        Torso/Base frame (REP-103): X=forward, Y=left, Z=up
        
        Args:
            detection: Dictionary returned by detect_in_image()
            camera_offset_in_torso: [x, y, z] camera mount translation relative to torso link
            
        Returns:
            Dictionary with:
              - 'position_torso': [x, y, z] coordinates in torso frame (meters)
              - 'orientation_quat_wxyz': [qw, qx, qy, qz] alignment quaternion
              - 'homogeneous_transform': (4, 4) transformation matrix T_torso_object
        """
        pos_cam = detection["position_3d_camera"]
        x_c, y_c, z_c = pos_cam[0], pos_cam[1], pos_cam[2]

        # Rotate optical -> robot body coordinates
        # X_robot = Z_cam, Y_robot = -X_cam, Z_robot = -Y_cam
        x_body = z_c
        y_body = -x_c
        z_body = -y_c

        # Add camera mount extrinsic translation
        pos_torso = np.array([x_body, y_body, z_body], dtype=np.float32) + camera_offset_in_torso

        # Construct 4x4 homogeneous transformation matrix
        T_torso_object = np.eye(4, dtype=np.float32)
        T_torso_object[0:3, 3] = pos_torso

        # Optical to body rotation matrix:
        # [ [0, -1,  0],
        #   [0,  0, -1],
        #   [1,  0,  0] ]
        # Identity orientation for object alignment in torso frame
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return {
            "position_torso": pos_torso,
            "orientation_quat_wxyz": quat_wxyz,
            "homogeneous_transform": T_torso_object,
            "frame_id": "torso_link",
        }
