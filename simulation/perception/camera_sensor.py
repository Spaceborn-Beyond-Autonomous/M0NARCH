"""
Camera Sensor Module
Captures synthetic RGB and Metric Depth frames from simulated MuJoCo cameras.
"""

from typing import Optional, Dict, Tuple
import numpy as np
import mujoco


class CameraSensor:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: Optional[str] = None,
        width: int = 320,
        height: int = 240,
    ):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height

        # Resolve camera ID if name provided
        self.camera_id = -1
        if camera_name is not None:
            self.camera_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )

        # Initialize MuJoCo offscreen renderer
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def capture_rgb(self) -> np.ndarray:
        """Renders and returns an RGB image (H, W, 3) as uint8."""
        self.renderer.disable_depth_rendering()
        if self.camera_id >= 0:
            self.renderer.update_scene(self.data, camera=self.camera_id)
        else:
            self.renderer.update_scene(self.data)
        rgb = self.renderer.render()
        return rgb.copy()

    def capture_depth(self) -> np.ndarray:
        """
        Renders and returns metric depth map (H, W) in meters.
        MuJoCo depth is normalized [0, 1]; this converts to physical distance.
        """
        self.renderer.enable_depth_rendering()
        if self.camera_id >= 0:
            self.renderer.update_scene(self.data, camera=self.camera_id)
        else:
            self.renderer.update_scene(self.data)
        raw_depth = self.renderer.render()
        
        # Convert buffer depth to metric distance using camera clipping planes
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        metric_depth = near / (1.0 - raw_depth * (1.0 - near / far))
        return metric_depth.astype(np.float32)

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Returns synchronized RGB and Depth observations."""
        return {
            "rgb": self.capture_rgb(),
            "depth": self.capture_depth(),
        }

    def close(self):
        self.renderer.close()
