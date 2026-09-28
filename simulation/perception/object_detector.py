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
