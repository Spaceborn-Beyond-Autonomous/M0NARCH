"""
Perception-Augmented RL Environment
Integrates visual and depth perception into the RL observation space for target tracking.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from gymnasium import spaces
import mujoco

import sys
from pathlib import Path

rl_path = str(Path(__file__).resolve().parents[1] / "rl")
if rl_path not in sys.path:
    sys.path.append(rl_path)

perception_path = str(Path(__file__).resolve().parent)
if perception_path not in sys.path:
    sys.path.append(perception_path)

from h1_locomotion_env import H1LocomotionEnv
try:
    from camera_sensor import CameraSensor
    from object_detector import ObjectDetector
except ImportError:
    from simulation.perception.camera_sensor import CameraSensor
    from simulation.perception.object_detector import ObjectDetector


class H1PerceptionEnv(H1LocomotionEnv):
    """
    Extends H1LocomotionEnv to provide perception-augmented observations.
    Observation Vector:
      [Joint Positions (17), Joint Velocities (19), Target 3D Relative Position (3)]
    """
    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        camera_name: Optional[str] = None,
    ):
        super().__init__(xml_path=xml_path, render_mode=render_mode)

        # Initialize perception sensors
        self.camera = CameraSensor(self.model, self.data, camera_name=camera_name, width=160, height=120)
        self.detector = ObjectDetector()

        # Expand observation space by 3 dimensions for Target 3D Position [dx, dy, dz]
        base_obs_dim = self.observation_space.shape[0]
        augmented_obs_dim = base_obs_dim + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(augmented_obs_dim,), dtype=np.float32
        )

        # Default virtual target position relative to robot (1.5m ahead, 0.8m height)
        self.target_pos = np.array([1.5, 0.0, 0.8], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Concatenates proprioceptive joint states with visual target perception."""
        base_obs = super()._get_obs()

        # Capture visual frame and run object detector
        frames = self.camera.get_observation()
        detection = self.detector.detect_in_image(frames["rgb"], frames["depth"])

        if detection is not None:
            # Use real detected 3D position from camera
            target_rel = detection["position_3d_camera"]
        else:
            # Fallback to simulated target relative to pelvis
            pelvis_pos = self.data.qpos[:3]
            target_rel = self.target_pos - pelvis_pos

        return np.concatenate([base_obs, target_rel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        # Add target-reaching bonus to reward
        target_dist = np.linalg.norm(obs[-3:])
        reaching_reward = 1.0 / (1.0 + target_dist)
        reward += reaching_reward
        info["target_distance"] = float(target_dist)

        return obs, reward, terminated, truncated, info

    def close(self):
        super().close()
        self.camera.close()
