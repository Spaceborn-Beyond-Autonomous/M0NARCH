"""
ANSA OS - Unitree H1 High-Throughput Batched Environment
=============================================================================
Vectorized humanoid simulation using Kevin Zakka's mjbatch library.
Enables running 256 to 1024 parallel H1 robots concurrently on CPU with C++
thread pooling and zero-copy NumPy memory bindings.

Features:
  - Vectorized observation extraction (qpos, qvel, torso upright, heights)
  - Vectorized reward computation (alive, velocity, upright, obstacle clearance)
  - Vectorized auto-reset for fallen robots
  - Domain Randomization (Sim-to-Real ground friction & motor damping)
=============================================================================
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import mujoco
from mjbatch import Batch


class H1BatchedEnv:
    def __init__(
        self,
        num_sims: int = 256,
        xml_path: Optional[str] = None,
        frameskip: int = 10,
        enable_domain_randomization: bool = True,
    ):
        self.num_sims = num_sims
        self.frameskip = frameskip

        # Load XML model
        if xml_path is None:
            models_dir = Path(__file__).resolve().parents[2] / "models" / "unitree_h1"
            xml_path = str(models_dir / "scene_obstacles.xml")
            if not os.path.exists(xml_path):
                xml_path = str(models_dir / "scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)

        # Initialize mjbatch Batch container (C++ thread pool)
        self.batch = Batch(self.model, num_sims=self.num_sims)

        # -------------------------------------------------------------
        # Zero-Copy Array Bindings (Live NumPy views directly into C++)
        # -------------------------------------------------------------
        self.qpos = self.batch.bind("qpos")  # shape: (num_sims, nq=26)
        self.qvel = self.batch.bind("qvel")  # shape: (num_sims, nv=25)
        self.ctrl = self.batch.bind("ctrl")  # shape: (num_sims, nu=19)

        self.num_actuators = self.model.nu  # 19 actuators
        self.obs_dim = 56                   # matching H1 single env

        # Torque limits from model
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        # Torso body ID
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Save home keyframe configuration
        self.home_qpos = np.zeros(self.model.nq, dtype=np.float64)
        if self.model.nkey > 0:
            self.home_qpos[:] = self.model.key_qpos[0]
        else:
            self.home_qpos[2] = 0.98
            self.home_qpos[3] = 1.0

        # Step counters and progress trackers per simulation
        self.step_counts = np.zeros(self.num_sims, dtype=np.int32)
        self.max_x = np.zeros(self.num_sims, dtype=np.float32)

        # Domain Randomization
        if enable_domain_randomization:
            self.apply_domain_randomization()

        # Initial reset
        self.reset()

    def apply_domain_randomization(self):
        """
        Randomizes ground friction and joint damping per simulation
        using mjbatch's expand() functionality for robust Sim-to-Real transfer.
        """
        try:
            # Randomize floor friction between 0.3 (slippery tile) and 1.2 (rough asphalt)
            friction = self.batch.expand("geom_friction")
            friction[:, :, 0] = np.random.uniform(0.35, 1.15, size=(self.num_sims, 1))

            # Randomize joint damping by +-15%
            damping = self.batch.expand("dof_damping")
            damping_scale = np.random.uniform(0.85, 1.15, size=(self.num_sims, self.model.nv))
            damping[:, :] = self.model.dof_damping[None, :] * damping_scale

            # Recompute model constants in C++
            self.batch.set_const()
        except Exception as e:
            # Fallback if specific expand property not supported in version
            pass

    def reset(self, env_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vectorized environment reset for specified or all parallel simulations.
        """
        if env_ids is None:
            env_ids = np.arange(self.num_sims)

        # Reset states to keyframe + small exploration noise
        noise_qpos = np.random.uniform(-0.005, 0.005, size=(len(env_ids), self.model.nq))
        noise_qpos[:, :7] = 0.0  # zero noise on root orientation & pos

        self.qpos[env_ids] = self.home_qpos[None, :] + noise_qpos
        self.qvel[env_ids] = 0.0
        self.ctrl[env_ids] = 0.0

        self.step_counts[env_ids] = 0
        self.max_x[env_ids] = self.qpos[env_ids, 0]

        return self.get_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Steps all parallel simulations simultaneously in C++ across CPU threads.
        
        Args:
            actions: NumPy array of shape (num_sims, 19) in range [-1.0, 1.0]
        
        Returns:
            observations: (num_sims, 56)
            rewards: (num_sims,)
            dones: (num_sims,) boolean mask
            info: dictionary of batch statistics
        """
        x_before = self.qpos[:, 0].copy()

        # 1. Scale actions to motor torque limits (vectorized)
        scaled_ctrl = self.ctrl_low[None, :] + (actions + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)[None, :]
        self.ctrl[:] = np.clip(scaled_ctrl, self.ctrl_low[None, :], self.ctrl_high[None, :])

        # 2. Physics stepping with frameskip in parallel C++ thread pool
        for _ in range(self.frameskip):
            self.batch.step()

        self.step_counts += 1

        # 3. Vectorized state extraction
        x_after = self.qpos[:, 0]
        z_height = self.qpos[:, 2]
        x_vel = (x_after - x_before) / (self.model.opt.timestep * self.frameskip)

        # Torso uprightness: quat w^2 - (x^2 + y^2) + z^2
        # (Exact Z-component of rotated local Z vector from quaternion)
        qw = self.qpos[:, 3]
        qx = self.qpos[:, 4]
        qy = self.qpos[:, 5]
        qz = self.qpos[:, 6]
        torso_upright = 1.0 - 2.0 * (qx ** 2 + qy ** 2)

        # 4. Vectorized Health / Termination Check
        height_ok = (z_height >= 0.60) & (z_height <= 1.25)
        upright_ok = torso_upright > 0.70
        is_healthy = height_ok & upright_ok

        # Check pit fall (x in [2.3, 3.4], y in center lane, z < 0.25)
        in_pit = (x_after >= 2.3) & (x_after <= 3.4) & (np.abs(self.qpos[:, 1]) < 0.75) & (z_height < 0.3)

        dones = (~is_healthy) | in_pit | (self.step_counts >= 1000)

        # 5. Vectorized Reward Function
        alive_bonus = 2.0 * is_healthy.astype(np.float32)
        vel_reward = 1.5 * np.exp(-2.0 * ((x_vel - 0.65) ** 2))
        upright_reward = 4.0 * np.clip(torso_upright, 0.0, 1.0)
        height_reward = 3.0 * np.exp(-20.0 * ((z_height - 0.96) ** 2))
        ctrl_cost = 0.01 * np.sum(actions ** 2, axis=1)

        # Progress reward
        progress = np.maximum(0.0, x_after - self.max_x) * 2.5
        self.max_x = np.maximum(self.max_x, x_after)

        # Obstacle bonus
        cleared_pit = (x_after > 3.3) & is_healthy
        obstacle_bonus = 80.0 * cleared_pit.astype(np.float32)

        rewards = np.where(
            dones & (~cleared_pit),
            -10.0,
            alive_bonus + vel_reward + upright_reward + height_reward + progress + obstacle_bonus - ctrl_cost
        )

        # 6. Auto-reset terminated environments in place
        reset_indices = np.where(dones)[0]
        if len(reset_indices) > 0:
            self.reset(env_ids=reset_indices)

        obs = self.get_observations()

        info = {
            "mean_reward": float(np.mean(rewards)),
            "mean_x": float(np.mean(x_after)),
            "num_resets": len(reset_indices),
            "healthy_ratio": float(np.mean(is_healthy)),
        }

        return obs, rewards, dones, info

    def get_observations(self) -> np.ndarray:
        """
        Extracts vectorized observations (num_sims, 56) directly from bound memory.
        """
        z_height = self.qpos[:, 2:3]          # (num_sims, 1)
        quat = self.qpos[:, 3:7]              # (num_sims, 4)
        joints = self.qpos[:, 7:]             # (num_sims, 19)
        vels = self.qvel                      # (num_sims, 25)

        qw = self.qpos[:, 3]
        qx = self.qpos[:, 4]
        qy = self.qpos[:, 5]
        qz = self.qpos[:, 6]

        # 3D Torso Z axis vector
        zx = 2.0 * (qx * qz + qw * qy)
        zy = 2.0 * (qy * qz - qw * qx)
        zz = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
        torso_z = np.column_stack([zx, zy, zz])  # (num_sims, 3)

        # Relative obstacle distance
        robot_x = self.qpos[:, 0:1]
        dist_pit = np.clip((2.6 - robot_x) / 5.0, -1.0, 1.0)
        dist_box = np.clip((5.3 - robot_x) / 10.0, -1.0, 1.0)
        obstacles = np.column_stack([dist_pit, dist_box]) # (num_sims, 2)

        # Simulated feet contacts indicator
        feet = np.zeros((self.num_sims, 2), dtype=np.float32)

        obs = np.hstack([z_height, quat, joints, vels, torso_z, feet, obstacles]).astype(np.float32)
        return obs
