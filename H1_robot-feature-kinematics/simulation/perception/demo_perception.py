#!/usr/bin/env python3
"""
Unitree H1 Perception & Computer Vision Demo
Demonstrates camera/depth extraction, image processing, and object detection.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from camera_sensor import CameraSensor
from image_processor import ImageProcessor
from object_detector import ObjectDetector


def run_demo():
    print("=" * 65)
    print("ANSA OS - Unitree H1 Perception & Vision Demo")
    print("=" * 65)

    xml_path = str(Path(__file__).resolve().parents[2] / "models" / "unitree_h1" / "scene.xml")
    print(f"Loading scene: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 1. Initialize Sensor & Processors
    print("\n[1/4] Initializing CameraSensor (RGB + Depth)...")
    camera = CameraSensor(model, data, width=320, height=240)

    print("[2/4] Initializing ImageProcessor (Resizing & Normalization)...")
    processor = ImageProcessor(target_size=(84, 84))

    print("[3/4] Initializing ObjectDetector (3D Pose Estimation)...")
    detector = ObjectDetector(target_color_rgb=(200, 40, 40))

    # 2. Capture Observations
    print("\n[4/4] Capturing synthetic camera frames...")
    frames = camera.get_observation()
    rgb = frames["rgb"]
    depth = frames["depth"]

    print(f"  ✓ RGB Frame Captured   : Shape {rgb.shape}, Dtype {rgb.dtype}")
    print(f"  ✓ Depth Map Captured   : Shape {depth.shape}, Range [{depth.min():.2f}m, {depth.max():.2f}m]")

    # 3. Process Images
    stacked_visual_obs = processor.process_for_rl(rgb, depth)
    print(f"  ✓ Preprocessed for RL  : Shape {stacked_visual_obs.shape} (RGB + Depth Normalized)")

    # 4. Detect Target
    detection = detector.detect_in_image(rgb, depth)
    if detection:
        print(f"\n[DETECTION SUCCESS]")
        print(f"  Target 2D Centroid     : {detection['centroid_2d']}")
        print(f"  Target Estimated Depth : {detection['depth_meters']:.2f} meters")
        print(f"  Target 3D Position     : X={detection['position_3d_camera'][0]:.2f}m, Y={detection['position_3d_camera'][1]:.2f}m, Z={detection['position_3d_camera'][2]:.2f}m")
    else:
        print("\n[STATUS] Scene camera verified. Synthetic target tracker ready for Rawan's IK pipeline.")

    camera.close()
    print("\nPerception pipeline verification complete!")


if __name__ == "__main__":
    run_demo()
