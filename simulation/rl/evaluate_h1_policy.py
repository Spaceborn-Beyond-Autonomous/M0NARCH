#!/usr/bin/env python3
"""
عرض وتقييم الروبوت H1 في الـ 3D Viewer
========================================
بيعرض الروبوت وهو بيمشي ويتفادى العقبات في نافذة MuJoCo التفاعلية.

الاستخدام:
  python evaluate_h1_policy.py              # عرض على أرض مسطحة
  python evaluate_h1_policy.py --stage 1    # عرض مع الحفرة
  python evaluate_h1_policy.py --stage 2    # عرض مع الحفرة والبوكس
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from h1_locomotion_env import H1ObstacleCourseEnv


def evaluate(model_path: str, stage: int = 0, num_episodes: int = 5):
    stage_names = {
        0: "مشي على أرض مسطحة",
        1: "مشي + حفرة",
        2: "مشي + حفرة + بوكس",
    }
    
    print("=" * 65)
    print("  ANSA OS - Unitree H1 RL Visualizer & Evaluator")
    print(f"  Stage: {stage} - {stage_names.get(stage, 'Unknown')}")
    print("=" * 65)

    use_obstacles = (stage > 0)
    env = H1ObstacleCourseEnv(
        render_mode="human",
        use_obstacles=use_obstacles,
        curriculum_stage=stage,
    )
    obs, _ = env.reset()

    # محاولة تحميل الـ policy المتدربة
    policy = None
    
    # نجرب نلاقي أحسن موديل
    possible_paths = [
        model_path,
        f"checkpoints/h1_stage{stage}_policy.zip",
        "checkpoints/h1_walking_policy.zip",
        "checkpoints/h1_stage0_policy.zip",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                from stable_baselines3 import PPO
                print(f"[INFO] Loading policy from: {path}")
                policy = PPO.load(path)
                break
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    
    if policy is None:
        print("[INFO] No trained policy found. Showing neutral pose (zero action).")

    print("\nStarting 3D visualizer... (Close window to exit)")
    print("─" * 50)

    episode = 0
    episode_reward = 0.0
    episode_steps = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.num_actuators, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                episode += 1
                status = "✓ SURVIVED" if not terminated else "✗ FELL"
                pit = "✓" if info.get("passed_pit") else "✗"
                box = "✓" if info.get("passed_box") else "✗"
                
                print(f"  Episode {episode}: {status} | "
                      f"Steps: {episode_steps:4d} | "
                      f"Reward: {episode_reward:8.1f} | "
                      f"X: {info.get('x_position', 0):.2f}m | "
                      f"Pit: {pit} | Box: {box}")
                
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0

                if episode >= num_episodes:
                    print(f"\n  Completed {num_episodes} episodes. Restarting...")
                    episode = 0

            viewer.sync()

            # تحكم في سرعة العرض (real-time)
            dt = env.model.opt.timestep * env.FRAMESKIP - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H1 Policy Evaluator")
    parser.add_argument("--model", type=str, default="checkpoints/h1_walking_policy.zip",
                        help="Path to trained policy")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Which stage to evaluate (0=walk, 1=pit, 2=box)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--trials", action="store_true",
                        help="Run the 4-trial curriculum sequence (Pit fall -> Pit avoid -> Box hit -> Finish)")
    args = parser.parse_args()

    if args.trials:
        from demo_obstacle_trials import run_trials
        run_trials(render=True)
    else:
        evaluate(model_path=args.model, stage=args.stage, num_episodes=args.episodes)
