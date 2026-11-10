import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from stable_baselines3 import PPO
from envs.h1_pick_place_env import H1PickPlaceEnv
 
CHECKPOINT = "checkpoints/h1_pick_place_1850000_steps"
 
env = H1PickPlaceEnv(render_mode="human")
model = PPO.load(CHECKPOINT)
 
obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1/50)
    if terminated or truncated:
        obs, info = env.reset()
