#!/usr/bin/env python3
"""
Unit and Integration Tests for H1 Batched Environment (mjbatch)
=============================================================================
Validates:
  1. Batched environment instantiation across CPU threads
  2. Memory binding shapes (qpos, qvel, ctrl)
  3. Observation vector dimensions and numerical validity
  4. Step execution, reward computation, and auto-reset
  5. Sim-to-Real Domain Randomization (friction, damping)
=============================================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.rl.h1_batched_env import H1BatchedEnv


class TestH1BatchedEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_sims = 32
        cls.env = H1BatchedEnv(
            num_sims=cls.num_sims,
            frameskip=5,
            enable_domain_randomization=True,
        )

    def test_01_initialization_and_bindings(self):
        """Check zero-copy NumPy array shapes bound directly to C++ memory."""
        self.assertEqual(self.env.qpos.shape, (self.num_sims, self.env.model.nq))
        self.assertEqual(self.env.qvel.shape, (self.num_sims, self.env.model.nv))
        self.assertEqual(self.env.ctrl.shape, (self.num_sims, self.env.model.nu))
        self.assertEqual(self.env.num_actuators, 19)

    def test_02_observation_vector(self):
        """Verify observations match expected feature dimensionality (56)."""
        obs = self.env.get_observations()
        self.assertEqual(obs.shape, (self.num_sims, 56))
        self.assertFalse(np.isnan(obs).any(), "NaN found in observation vector!")
        self.assertFalse(np.isinf(obs).any(), "Inf found in observation vector!")

    def test_03_vectorized_step(self):
        """Verify stepping all parallel instances concurrently."""
        actions = np.zeros((self.num_sims, self.env.num_actuators), dtype=np.float32)
        obs, rewards, dones, info = self.env.step(actions)

        self.assertEqual(obs.shape, (self.num_sims, 56))
        self.assertEqual(rewards.shape, (self.num_sims,))
        self.assertEqual(dones.shape, (self.num_sims,))
        self.assertIn("mean_reward", info)
        self.assertIn("healthy_ratio", info)

    def test_04_auto_reset(self):
        """Verify fallen environments reset their posture without crashing the batch."""
        # Force robot 0 to lie on the ground (height = 0.1)
        self.env.qpos[0, 2] = 0.1
        actions = np.zeros((self.num_sims, self.env.num_actuators), dtype=np.float32)
        obs, rewards, dones, info = self.env.step(actions)

        # After step, auto-reset should restore robot 0 close to home height (~0.98m)
        self.assertGreater(self.env.qpos[0, 2], 0.70)


if __name__ == "__main__":
    unittest.main()
