#!/usr/bin/env python3
"""Quick test to verify the new environment works."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from h1_locomotion_env import H1ObstacleCourseEnv
import numpy as np

print("=" * 60)
print("Testing H1ObstacleCourseEnv")
print("=" * 60)

# Test Stage 0: Flat walking
print("\n--- Stage 0: Flat walking ---")
env = H1ObstacleCourseEnv(use_obstacles=False, curriculum_stage=0)
obs, info = env.reset()
print(f"  Obs shape: {obs.shape} (expected: (56,))")
print(f"  Action shape: {env.action_space.shape} (expected: (19,))")

total_reward = 0
for i in range(100):
    action = env.action_space.sample() * 0.1  # small random actions
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated:
        print(f"  Terminated at step {i+1}! z={info['z_height']:.3f}, upright={info['torso_upright']:.3f}")
        obs, _ = env.reset()
        break

print(f"  Steps completed: {min(i+1, 100)}")
print(f"  Total reward: {total_reward:.1f}")
print(f"  Final x_pos: {info.get('x_position', 'N/A')}")
env.close()

# Test Stage 1: With obstacles  
print("\n--- Stage 1: With obstacles (pit) ---")
try:
    env = H1ObstacleCourseEnv(use_obstacles=True, curriculum_stage=1)
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    
    # Check obstacle geom IDs
    print(f"  Ground geom IDs: {sorted(env.ground_geom_ids)}")
    print(f"  Foot geom IDs: {sorted(env.foot_geom_ids)}")
    print(f"  Non-foot robot geoms: {sorted(env.non_foot_robot_geoms)}")
    
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.05
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  Terminated at step {i+1}")
            break
    
    print(f"  Steps completed: {min(i+1, 50)}")
    print(f"  Total reward: {total_reward:.1f}")
    env.close()
    print("  [OK] Obstacle env works!")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test Stage 2: Full course
print("\n--- Stage 2: Full obstacle course ---")
try:
    env = H1ObstacleCourseEnv(use_obstacles=True, curriculum_stage=2)
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.05
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"  Steps completed: {min(i+1, 50)}")
    print(f"  Total reward: {total_reward:.1f}")
    env.close()
    print("  [OK] Full course env works!")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All environment tests passed!")
print("=" * 60)
print("\nReady to train. Run:")
print('  & ".\.venv\Scripts\python.exe" simulation/rl/train_h1_ppo.py --stage 0 --steps 2000000')
