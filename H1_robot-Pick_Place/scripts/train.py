import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.h1_pick_place_env import H1PickPlaceEnv
 
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run relative to project root
 
N_ENVS = 4                     # parallel sims; raise if you have CPU headroom
TOTAL_TIMESTEPS = 2_000_000    # realistic target — will NOT finish in 2 hrs, see Section 12
 
def make_env():
    return H1PickPlaceEnv()
 
if __name__ == "__main__":
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
 
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="checkpoints/",
        name_prefix="h1_pick_place",
    )
 
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="logs/",
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
 
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, tb_log_name="ppo_h1")
    model.save("checkpoints/h1_pick_place_final")
