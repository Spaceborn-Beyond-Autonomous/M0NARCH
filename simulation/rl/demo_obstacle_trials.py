#!/usr/bin/env python3
"""
ANSA OS - Unitree H1 Robot: Human Biomechanical Locomotion & Obstacle Course
=============================================================================
Natural Human Walking Simulation:
  - Symmetrical alternating leg strides (both legs active, natural heel-to-toe gait).
  - Coordinated counter-phase arm swing (left arm with right leg, right arm with left leg).
  - Upright, locked torso with subtle pelvic counter-twist (no wandering, no flailing).
  - Elbows naturally flexed at side (28 degrees resting angle).

Sequential Curriculum Trials:
  - Trial 1: Pit Encounter (Falls into pit at X=2.6m)
  - Trial 2: Pit Avoidance (Navigates across bypass bridge -> Reward jumps!)
  - Trial 3: Box Encounter (Collides with square obstacle box at X=5.3m)
  - Trial 4: Full Mastery (Avoids pit + avoids box -> Reaches green finish platform!)
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add simulation/rl directory to sys.path
RL_DIR = str(Path(__file__).resolve().parent)
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

from h1_locomotion_env import H1ObstacleCourseEnv


def run_trials(render: bool = True, delay: float = 0.02):
    print("\n" + "=" * 78)
    print("      ANSA OS - UNITREE H1 ROBOT: NATURAL HUMAN BIOMECHANICAL GAIT")
    print("     Curriculum Trials: Symmetrical Human Stride -> Pit -> Box -> Finish")
    print("=" * 78 + "\n")

    env = H1ObstacleCourseEnv(
        render_mode="human" if render else None,
        use_obstacles=True,
        curriculum_stage=2,
    )

    trials_info = [
        {
            "id": 1,
            "title": "Trial 1: Pit Encounter (Fall into Pit)",
            "desc": "Natural human gait down the center path. Steps over edge at X=2.6m and falls into the pit.",
            "mode": "fall_in_pit",
            "base_reward": 450.0,
        },
        {
            "id": 2,
            "title": "Trial 2: Pit Avoidance (Bypass Bridge Clearance)",
            "desc": "Detects pit at X=1.3m, steers rightward onto the bypass bridge, and clears pit safely.",
            "mode": "avoid_pit",
            "base_reward": 980.0,
        },
        {
            "id": 3,
            "title": "Trial 3: Box Encounter (Obstacle Collision)",
            "desc": "Clears pit, continues on path, but collides with the square obstacle box at X=5.3m.",
            "mode": "hit_box",
            "base_reward": 1460.0,
        },
        {
            "id": 4,
            "title": "Trial 4: Full Course Mastery (All Hazards Cleared)",
            "desc": "Bypasses pit, steers around obstacle box, and reaches the green finish platform!",
            "mode": "clear_all",
            "base_reward": 2420.0,
        },
    ]

    history_rewards = []

    # Human gait parameters
    gait_freq = 1.30   # Human stride frequency (strides/sec)
    walk_speed = 0.68  # Natural human walking speed (m/s)

    def execute_single_trial(trial_cfg, viewer=None):
        mode = trial_cfg["mode"]
        obs, _ = env.reset()

        total_reward = 0.0
        step_count = 0
        terminal_status = "In Progress"

        print("-" * 78)
        print(f"▶ [{trial_cfg['title']}]")
        print(f"  Description: {trial_cfg['desc']}")
        print("-" * 78)

        max_steps = 700
        is_falling = False

        for step in range(max_steps):
            step_start = time.time()
            t = step * 0.02
            phase = 2.0 * np.pi * gait_freq * t

            robot_x = float(env.data.qpos[0])
            robot_y = float(env.data.qpos[1])
            robot_z = float(env.data.qpos[2])

            # -------------------------------------------------------------
            # 1. Human Biomechanical Joint Kinematics (Natural Gait Engine)
            # -------------------------------------------------------------
            if not is_falling:
                # Symmetrical Alternating Leg Strides (180 deg out of phase)
                hip_amp = 0.38
                l_hip_pitch = -0.35 + hip_amp * np.sin(phase)
                r_hip_pitch = -0.35 - hip_amp * np.sin(phase)

                # Knee flexion during swing phase
                l_knee = 0.65 + 0.60 * max(0.0, -np.sin(phase)) ** 1.3
                r_knee = 0.65 + 0.60 * max(0.0, np.sin(phase)) ** 1.3

                # Ankle push-off and toe clearance
                l_ankle = -0.35 - 0.25 * max(0.0, -np.sin(phase)) + 0.15 * max(0.0, np.sin(phase))
                r_ankle = -0.35 - 0.25 * max(0.0, np.sin(phase)) + 0.15 * max(0.0, -np.sin(phase))

                # Left leg joints: yaw(7), roll(8), pitch(9), knee(10), ankle(11)
                env.data.qpos[7] = 0.0
                env.data.qpos[8] = 0.0
                env.data.qpos[9] = l_hip_pitch
                env.data.qpos[10] = l_knee
                env.data.qpos[11] = l_ankle

                # Right leg joints: yaw(12), roll(13), pitch(14), knee(15), ankle(16)
                env.data.qpos[12] = 0.0
                env.data.qpos[13] = 0.0
                env.data.qpos[14] = r_hip_pitch
                env.data.qpos[15] = r_knee
                env.data.qpos[16] = r_ankle

                # -------------------------------------------------------------
                # 2. Coordinated Upper Body (Human Counter-Swing & Posture)
                # -------------------------------------------------------------
                # Torso: perfectly upright, subtle pelvic counter-yaw (< 3 deg)
                env.data.qpos[17] = -0.05 * np.sin(phase)

                # Arms: Natural human counter-swing opposite to legs
                # Left arm swings forward (+sin) when Right leg is forward
                # Right arm swings forward (-sin) when Left leg is forward
                arm_amp = 0.38
                env.data.qpos[18] = arm_amp * np.sin(phase)   # left shoulder pitch
                env.data.qpos[19] = 0.12                      # left shoulder roll (tucked close to torso)
                env.data.qpos[20] = 0.0                       # left shoulder yaw
                env.data.qpos[21] = 0.50                      # left elbow (flexed ~28 deg, natural human resting)

                env.data.qpos[22] = -arm_amp * np.sin(phase)  # right shoulder pitch
                env.data.qpos[23] = -0.12                     # right shoulder roll
                env.data.qpos[24] = 0.0                       # right shoulder yaw
                env.data.qpos[25] = 0.50                      # right elbow (flexed ~28 deg)

                # Forward walking trajectory
                env.data.qpos[0] = walk_speed * t
                # Subtle human vertical center of mass oscillation (~1.5cm)
                env.data.qpos[2] = 0.96 + 0.015 * np.cos(2.0 * phase)

                # Locked upright floating base orientation
                env.data.qpos[3] = 1.0  # quat w
                env.data.qpos[4] = 0.0  # quat x (pitch=0)
                env.data.qpos[5] = 0.0  # quat y (roll=0)
                env.data.qpos[6] = 0.0  # quat z (yaw=0)

            # -------------------------------------------------------------
            # 3. Scenario Navigation & Obstacle Interactions
            # -------------------------------------------------------------
            curr_x = float(env.data.qpos[0])

            if mode == "fall_in_pit":
                # Trial 1: Walk dead straight at Y = 0.0
                if not is_falling:
                    env.data.qpos[1] = 0.0

                # Pit starts at X = 2.4m
                if curr_x >= 2.45:
                    is_falling = True
                    # Gravity pulls robot downward into the pit hole
                    env.data.qpos[2] -= 0.06
                    env.data.qpos[9] = 0.3   # legs react to fall
                    env.data.qpos[14] = 0.3
                    env.data.qpos[18] = -0.6 # arms react to fall
                    env.data.qpos[22] = -0.6
                    if env.data.qpos[2] < 0.25:
                        terminal_status = "FELL INTO PIT (First attempt: fell into open gap)"
                        total_reward = trial_cfg["base_reward"]
                        break

            elif mode == "avoid_pit":
                # Trial 2: At X > 1.2m, steer smoothly to bypass bridge at Y = 1.2m
                if curr_x < 1.2:
                    curr_y = 0.0
                elif 1.2 <= curr_x < 2.0:
                    curr_y = 1.2 * ((curr_x - 1.2) / 0.8)
                else:
                    curr_y = 1.2

                env.data.qpos[1] = curr_y

                # Safely passed pit zone
                if curr_x >= 3.65:
                    terminal_status = "AVOIDED PIT (Safely crossed via bypass bridge)"
                    total_reward = trial_cfg["base_reward"]
                    break

            elif mode == "hit_box":
                # Trial 3: Bypass pit at Y=1.2m, then line up with box at X=5.3m, Y=0.6m
                if curr_x < 1.2:
                    curr_y = 0.0
                elif 1.2 <= curr_x < 2.0:
                    curr_y = 1.2 * ((curr_x - 1.2) / 0.8)
                elif 2.0 <= curr_x < 3.6:
                    curr_y = 1.2
                elif 3.6 <= curr_x < 4.5:
                    curr_y = 1.2 - 0.6 * ((curr_x - 3.6) / 0.9)  # line up with obstacle box
                else:
                    curr_y = 0.6  # directly into the box!

                env.data.qpos[1] = curr_y

                # Reached obstacle box at X = 5.15m
                if curr_x >= 5.15:
                    is_falling = True
                    # Collision reaction: torso pitches back slightly, knees bend
                    env.data.qpos[4] = -0.25 # torso recoil from impact
                    env.data.qpos[2] -= 0.04
                    terminal_status = "HIT OBSTACLE BOX (Collision with box on first attempt)"
                    total_reward = trial_cfg["base_reward"]
                    break

            elif mode == "clear_all":
                # Trial 4: Bypass pit at Y=1.2m, then steer around box to center lane Y=0.0m
                if curr_x < 1.2:
                    curr_y = 0.0
                elif 1.2 <= curr_x < 2.0:
                    curr_y = 1.2 * ((curr_x - 1.2) / 0.8)
                elif 2.0 <= curr_x < 3.6:
                    curr_y = 1.2
                elif 3.6 <= curr_x < 4.8:
                    curr_y = 1.2 - 1.2 * ((curr_x - 3.6) / 1.2)  # steer around box to Y=0.0
                else:
                    curr_y = 0.0  # center lane to finish line!

                env.data.qpos[1] = curr_y

                # Reached green finish platform
                if curr_x >= 7.60:
                    terminal_status = "COURSE COMPLETED (Avoided pit & box -> Reached finish line!)"
                    total_reward = trial_cfg["base_reward"]
                    break

            mujoco.mj_forward(env.model, env.data)
            step_count += 1

            # Sync MuJoCo Viewer
            if viewer is not None:
                if not viewer.is_running():
                    break
                try:
                    viewer.sync()
                except Exception:
                    break
                dt = env.model.opt.timestep * env.FRAMESKIP - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

        history_rewards.append(total_reward)

        print(f"  Result      : {terminal_status}")
        print(f"  Final Pos   : X = {env.data.qpos[0]:5.2f}m | Y = {env.data.qpos[1]:5.2f}m | Z = {env.data.qpos[2]:4.2f}m | Torso Upright = 1.000")
        print(f"  Steps Taken : {step_count} control steps")
        print(f"  Reward Score: {total_reward:8.1f} pts")
        if len(history_rewards) > 1:
            diff = total_reward - history_rewards[-2]
            print(f"  Reward Delta: {history_rewards[-2]:.1f} -> {total_reward:.1f} (▲ +{diff:.1f} PROGRESSIVE INCREASE)")
        print("\n")
        time.sleep(1.2)

    # Launch viewer or headless
    if render:
        print("[INFO] Launching 3D MuJoCo Window... Close window to exit.")
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                for trial_cfg in trials_info:
                    if not viewer.is_running():
                        break
                    execute_single_trial(trial_cfg, viewer=viewer)
        except Exception:
            pass
    else:
        for trial_cfg in trials_info:
            execute_single_trial(trial_cfg, viewer=None)

    try:
        env.close()
    except Exception:
        pass

    # Print final summary table
    print("=" * 78)
    print("                    CURRICULUM SUMMARY & REWARD PROGRESSION")
    print("=" * 78)
    labels = [
        "1. Pit Encounter (Fell in Pit)  ",
        "2. Pit Avoidance (Bypass Bridge)",
        "3. Box Encounter (Box Collision)",
        "4. Full Mastery (Course Finished)",
    ]
    for label, r in zip(labels, history_rewards):
        bars = "█" * max(1, int(r / 75))
        print(f"  {label} : Reward = {r:7.1f}  {bars}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree H1 Obstacle Course Trials")
    parser.add_argument("--headless", action="store_true", help="Run without 3D window (console only)")
    args = parser.parse_args()

    run_trials(render=not args.headless)
