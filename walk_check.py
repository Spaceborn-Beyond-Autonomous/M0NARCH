"""
Physics-walking check: REAL physics walking (mj_step, no qpos writes) on M0NARCH's H1.

Controller: Unitree's pretrained H1 walking policy (unitree_rl_gym, BSD-3-Clause),
run with NumPy by h1_policy.py (no PyTorch). Legs: policy -> PD torques. Torso + arms: PD hold.

Root cause found: M0NARCH's H1 feet collide through 3 thin capsules (heel is a 2.8 cm wide line,
foot ~18 cm long). The policy was trained on Unitree's foot (30 cm x 8 cm sole) and falls in
~2 s on the capsules. With one sole box of Unitree's foot size, it walks. Joint damping/armature
and the actuated arms turned out NOT to matter.

Run from ~/M0NARCH (mjlearn venv), with unitree_rl_gym cloned next to it:
  git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym ~/unitree_rl_gym
  python e9_check.py                 # ~1-2 min, prints one line per test
  python e9_check.py --view          # watch the 0.5 m/s walk in the MuJoCo viewer
"""
import os
import sys
import time

import mujoco
import numpy as np

from h1_policy import (ACTION_SCALE, ANG_VEL_SCALE, CMD_SCALE, DECIMATION, DEFAULT, DOF_VEL_SCALE, KDS, KPS,
                       PERIOD, H1Policy, gravity_orientation)

SCENE = "H1_robot-Physics-simulation/models/unitree_h1/scene.xml"
from physics_walk import policy_path
POLICY = policy_path() or os.path.expanduser("~/unitree_rl_gym/deploy/pre_train/h1/motion.pt")
UP_KP = np.array([300] + [60, 60, 60, 40] * 2, float)     # torso + 2 arms (4 joints each)


def load(sole_box=True):
    s = mujoco.MjSpec.from_file(SCENE)
    if sole_box:
        for side in ("left", "right"):
            b = s.body(side + "_ankle_link")
            for g in list(b.geoms):
                if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    s.delete(g)
            # same footprint as Unitree's foot collision mesh: x -0.10..0.20, y +-0.04, sole at z -0.07
            b.add_geom(name=side + "_sole", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.04, 0.01],
                       pos=[0.05, 0, -0.06], rgba=[0.2, 0.2, 0.2, 0.0], group=3)
    for k in list(s.keys):
        s.delete(k)
    return s.compile()


class PhysicsWalker:
    """cmd = [vx m/s, vy m/s, yaw_rate rad/s]. Call step() once per physics step (dt 0.002)."""

    def __init__(self, m, d):
        self.m, self.d, self.pol = m, d, H1Policy(POLICY)
        self.lo, self.hi = m.actuator_ctrlrange.T
        self.act = np.zeros(10, np.float32); self.tgt = DEFAULT.copy(); self.k = 0
        self.obs = np.zeros(41, np.float32); self.cmd = np.zeros(3, np.float32)

    def step(self):
        m, d = self.m, self.d
        tau = np.zeros(m.nu)
        tau[:10] = (self.tgt - d.qpos[7:17]) * KPS - d.qvel[6:16] * KDS
        tau[10:] = UP_KP * (0 - d.qpos[17:]) - UP_KP / 25 * d.qvel[16:]
        d.ctrl[:] = np.clip(tau, self.lo, self.hi)
        mujoco.mj_step(m, d)
        self.k += 1
        if self.k % DECIMATION == 0:
            ph = (self.k * m.opt.timestep) % PERIOD / PERIOD
            o = self.obs
            o[:3] = d.qvel[3:6] * ANG_VEL_SCALE; o[3:6] = gravity_orientation(d.qpos[3:7])
            o[6:9] = self.cmd * CMD_SCALE; o[9:19] = d.qpos[7:17] - DEFAULT
            o[19:29] = d.qvel[6:16] * DOF_VEL_SCALE; o[29:39] = self.act
            o[39:41] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
            self.act = self.pol(o).astype(np.float32); self.tgt = self.act * ACTION_SCALE + DEFAULT


def start(m):
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0; d.qpos[3] = 1; d.qpos[7:17] = DEFAULT
    mujoco.mj_forward(m, d)
    return d


def trial(m, cmd, T=10.0, push=None):
    d = start(m); w = PhysicsWalker(m, d); w.cmd[:] = cmd
    while d.time < T:
        d.xfrc_applied[1, :2] = push[1] if push and push[0] <= d.time < push[0] + 0.2 else 0
        w.step()
        if d.qpos[2] < 0.6:
            return f"FELL at {d.time:.1f} s"
    qw, qx, qy, qz = d.qpos[3:7]
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return f"walked  x={d.qpos[0]:+.2f} m  y={d.qpos[1]:+.2f} m  yaw={yaw:+.0f} deg"


def main():
    if not os.path.exists(POLICY):
        sys.exit(f"Missing {POLICY}\n  git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym ~/unitree_rl_gym")
    if "--view" in sys.argv:
        from mujoco import viewer as mjv
        m = load(); d = start(m); w = PhysicsWalker(m, d); w.cmd[:] = [0.5, 0, 0]
        with mjv.launch_passive(m, d) as v:
            while v.is_running():
                t0 = time.time()
                for _ in range(10):
                    w.step()
                v.sync(); time.sleep(max(0, 0.02 - (time.time() - t0)))
        return
    tests = [("original capsule feet, 0.5 m/s", False, [0.5, 0, 0], None),
             ("sole box, stand still (cmd 0)", True, [0, 0, 0], None),
             ("sole box, walk 0.5 m/s", True, [0.5, 0, 0], None),
             ("sole box, walk 0.8 m/s", True, [0.8, 0, 0], None),
             ("sole box, walk + turn 0.5 rad/s", True, [0.5, 0, 0.5], None),
             ("sole box, 100 N side push at 4 s", True, [0.5, 0, 0], (4.0, [0, 100])),
             ("sole box, 150 N side push at 4 s", True, [0.5, 0, 0], (4.0, [0, 150]))]
    models = {False: load(False), True: load(True)}
    print("Physics-walking check (mj_step only, 10 s each)\n")
    for name, box, cmd, push in tests:
        print(f"  {name:36s} {trial(models[box], cmd, push=push)}", flush=True)


if __name__ == "__main__":
    main()
