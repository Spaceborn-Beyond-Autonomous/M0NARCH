"""
Unit tests for Perception Module (Camera, Image Processing, 3D Object Detection).
Runs with standard unittest or pytest.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import mujoco

from simulation.perception.camera_sensor import CameraSensor
from simulation.perception.image_processor import ImageProcessor
from simulation.perception.object_detector import ObjectDetector
from simulation.perception.perception_rl_env import H1PerceptionEnv


class TestPerceptionModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml_path = str(Path(__file__).resolve().parents[2] / "models" / "unitree_h1" / "scene.xml")
        cls.model = mujoco.MjModel.from_xml_path(cls.xml_path)
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_01_camera_sensor(self):
        camera = CameraSensor(self.model, self.data, width=160, height=120)
        rgb = camera.capture_rgb()
        self.assertEqual(rgb.shape, (120, 160, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        depth = camera.capture_depth()
        self.assertEqual(depth.shape, (120, 160))
        self.assertEqual(depth.dtype, np.float32)

        obs = camera.get_observation()
        self.assertIn("rgb", obs)
        self.assertIn("depth", obs)
        camera.close()

    def test_02_image_processor(self):
        processor = ImageProcessor(target_size=(84, 84))
        dummy_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy_depth = np.ones((240, 320), dtype=np.float32)

        processed = processor.process_for_rl(dummy_rgb, dummy_depth)
        self.assertEqual(processed.shape, (84, 84, 4))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)

    def test_03_object_detector(self):
        detector = ObjectDetector(target_color_rgb=(200, 40, 40), color_threshold=50.0)
        synthetic_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        synthetic_rgb[40:60, 70:90] = [200, 40, 40]
        synthetic_depth = np.full((120, 160), 2.0, dtype=np.float32)

        detection = detector.detect_in_image(synthetic_rgb, synthetic_depth)
        self.assertIsNotNone(detection)
        self.assertTrue(detection["detected"])
        self.assertIn("position_3d_camera", detection)
        self.assertEqual(len(detection["position_3d_camera"]), 3)
        self.assertAlmostEqual(detection["depth_meters"], 2.0, places=2)

        # Test 6D pose transformation to robot torso link (Rawan's handoff)
        pose_6d = detector.compute_6d_pose_in_torso(detection)
        self.assertIn("position_torso", pose_6d)
        self.assertIn("homogeneous_transform", pose_6d)
        self.assertEqual(pose_6d["frame_id"], "torso_link")
        self.assertEqual(pose_6d["homogeneous_transform"].shape, (4, 4))
        self.assertAlmostEqual(pose_6d["position_torso"][0], 2.0 + 0.15, places=2)  # X_torso = Z_cam + 0.15

    def test_04_perception_rl_env(self):
        env = H1PerceptionEnv()
        obs, info = env.reset()
        self.assertEqual(obs.shape, (59,))

        action = np.zeros(env.num_actuators, dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        self.assertEqual(next_obs.shape, (59,))
        self.assertIn("target_distance", step_info)
        env.close()


if __name__ == "__main__":
    unittest.main()
