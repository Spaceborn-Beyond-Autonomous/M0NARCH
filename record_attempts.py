#!/usr/bin/env python3
"""
Simulate and record H1 robot training attempts to an MP4 video.
Visualizes how the robot behaves across multiple episodes (falling, balancing, attempting steps).
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import mujoco

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir / "simulation" / "rl"))
from h1_locomotion_env import H1LocomotionEnv


def record_simulation_attempts(video_path="H1_simulation_attempts.mp4", num_episodes=3):
    print("=" * 70)
    print("  SIMULATING H1 ROBOT ATTEMPTS & RECORDING VIDEO")
    print("=" * 70)

    env = H1LocomotionEnv()
    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    # Setup video writer (MP4V codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    total_frames = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        step_count = 0
        terminated = False

        print(f"\n>>> Running Episode {ep}/{num_episodes}...")

        # Run episode until termination (fall) or max 150 steps (~3 seconds)
        for s in range(150):
            step_count += 1

            if ep == 1:
                # Attempt 1: Raw untrained random motor actions (robot flails & falls)
                action = np.random.uniform(-0.6, 0.6, size=env.num_actuators).astype(np.float32)
            elif ep == 2:
                # Attempt 2: Stiff posture baseline (holds standing pose, resists fall longer)
                action = np.zeros(env.num_actuators, dtype=np.float32)
                # Small ankle/hip compensation
                action[4] = 0.1
                action[9] = 0.1
            else:
                # Attempt 3: Periodic stepping motion (locomotion attempt)
                action = np.zeros(env.num_actuators, dtype=np.float32)
                action[3] = 0.3 * np.sin(s * 0.15)   # Left knee flexion
                action[8] = -0.3 * np.sin(s * 0.15)  # Right knee flexion
                action[2] = 0.15 * np.cos(s * 0.15)  # Left hip pitch
                action[7] = -0.15 * np.cos(s * 0.15) # Right hip pitch

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Render third-person tracking frame every 2 physics steps (to match 30 FPS video)
            if s % 2 == 0:
                # Configure camera to track robot pelvis smoothly
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                cam.distance = 2.8
                cam.elevation = -15
                cam.azimuth = 135

                renderer.update_scene(env.data, camera=cam)
                frame = renderer.render()

                # Add real-time telemetry HUD overlay onto video frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"ANSA OS H1 - Episode {ep}/{num_episodes}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame_bgr, f"Step: {s} | Z-Height: {info['z_height']:.2f}m", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame_bgr, f"Forward Vel: {info['x_velocity']:+.2f} m/s", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.putText(frame_bgr, f"Reward: {ep_reward:.1f}", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

                if terminated:
                    cv2.putText(frame_bgr, "TERMINATED: ROBOT FELL!", (180, 240),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

                out.write(frame_bgr)
                total_frames += 1

            if terminated:
                print(f"    Episode {ep} ended at Step {step_count}: Robot lost balance (Z = {info['z_height']:.2f}m). Total Reward: {ep_reward:.2f}")
                # Hold final frame for 10 frames in video so user sees the fall
                for _ in range(10):
                    out.write(frame_bgr)
                    total_frames += 1
                break

        if not terminated:
            print(f"    Episode {ep} completed all {step_count} steps successfully! Total Reward: {ep_reward:.2f}")

    out.release()
    renderer.close()
    env.close()

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Video successfully saved with {total_frames} frames!")
    print(f"  File Location: {Path(video_path).resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    record_simulation_attempts()
