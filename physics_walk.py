"""
M0NARCH E9 fix: REAL physics walking for the Unitree H1 (mj_step only, no qpos writes).

Two parts:
  1. load_model(xml): loads the scene and gives each H1 foot a flat sole box with the
     footprint of Unitree's own H1 foot (x -0.10..0.20 m, y +-0.04 m, sole 7 cm below the
     ankle). The original 3 thin capsules (heel 2.8 cm wide) are removed: on them every
     balance controller falls within ~2 s. The box is invisible (group 3, alpha 0).
  2. PhysicsWalker: Unitree's pretrained H1 walking policy (unitree_rl_gym, BSD-3-Clause,
     deploy/pre_train/h1/motion.pt) run in NumPy by h1_policy.py. The policy sets PD
     targets for the 10 leg joints at 50 Hz; torso and arms are held by PD at 0.
     Command: [vx m/s, vy m/s, yaw_rate rad/s].

The policy file is looked for in this order:
  $M0NARCH_H1_POLICY, ./unitree_rl_gym/..., ~/unitree_rl_gym/deploy/pre_train/h1/motion.pt
"""
import math
import os

import mujoco
import numpy as np

from h1_policy import (ACTION_SCALE, ANG_VEL_SCALE, CMD_SCALE, DECIMATION, DEFAULT, DOF_VEL_SCALE, KDS, KPS,
                       PERIOD, H1Policy, gravity_orientation)

_REL = os.path.join("unitree_rl_gym", "deploy", "pre_train", "h1", "motion.pt")
_HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_CANDIDATES = [os.environ.get("M0NARCH_H1_POLICY", ""),
                     os.path.join(_HERE, "policies", "unitree_h1_motion.pt"),     # shipped with the repo
                     _REL, os.path.join(os.path.expanduser("~"), _REL)]

# sole box in the ankle frame (same footprint as Unitree's H1 foot collision mesh)
SOLE_HALF = (0.15, 0.04, 0.01)
SOLE_POS = (0.05, 0.0, -0.06)

UPPER_KP = np.array([300] + [60, 60, 60, 40] * 2, float)   # torso, then 4 joints per arm
UPPER_KD = UPPER_KP / 25

# arm swing (E13 kept under physics): each shoulder follows the OPPOSITE hip, like the
# kinematic gait did. 0 = arms held still. Upper order: torso, L sh pitch/roll/yaw, L elbow, R ...
ARM_SWING = 2.0      # shoulder pitch (rad) per rad of hip-pitch difference (~+-0.3 rad swing)
ARM_SWING_MAX = 0.4  # rad
ARM_POSE = np.array([0.0, 0.0, 0.12, 0.0, 0.5, 0.0, -0.12, 0.0, 0.5])   # arms slightly out, elbows bent

# heading -> command (used by the navigator)
V_MAX = 0.5          # m/s forward command (actual speed ~0.38 m/s)
YAW_GAIN = 1.5       # rad/s per rad of heading error
YAW_RATE_MAX = 0.6   # rad/s
TURN_IN_PLACE = 0.9  # rad: above this heading error, stop walking forward and turn


def policy_path():
    for p in POLICY_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def load_model(xml_path):
    """Load a scene with flat sole boxes on both H1 feet (keeps keyframes, sensors, everything else)."""
    return add_soles(mujoco.MjSpec.from_file(xml_path)).compile()


def add_soles(s):
    """Replace the foot capsules in an MjSpec with sole boxes. Returns the spec."""
    for side in ("left", "right"):
        b = s.body(side + "_ankle_link")
        if b is None:
            continue
        for g in list(b.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                s.delete(g)
        b.add_geom(name=side + "_sole", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(SOLE_HALF),
                   pos=list(SOLE_POS), rgba=[0.2, 0.2, 0.2, 0.0], group=3)
    return s


def sole(m, d, side):
    """(x, y) centre and lowest z of one foot's sole box."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, side + "_sole")
    R = d.geom_xmat[g].reshape(3, 3)
    corners = np.array([[sx * SOLE_HALF[0], sy * SOLE_HALF[1], -SOLE_HALF[2]] for sx in (-1, 1) for sy in (-1, 1)])
    w = corners @ R.T + d.geom_xpos[g]
    return d.geom_xpos[g][:2].copy(), float(w[:, 2].min())


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PhysicsWalker:
    def __init__(self, path=None):
        path = path or policy_path()
        if path is None:
            raise FileNotFoundError("H1 policy not found. Run:\n  git clone --depth 1 "
                                    "https://github.com/unitreerobotics/unitree_rl_gym ~/unitree_rl_gym")
        self.pol = H1Policy(path)
        self.cmd = np.zeros(3, np.float32)
        self.status = "idle"

    def reset(self, m, d, xy=(0.0, 0.0), yaw=0.0):
        """Put the robot upright in the policy's default leg pose, feet on the floor."""
        assert m.nu == 19, "expects the 19-motor H1 (10 legs, torso, 8 arm)"
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [xy[0], xy[1], 1.0]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[7:17] = DEFAULT
        mujoco.mj_forward(m, d)
        self.pol.reset()
        self.act = np.zeros(10, np.float32)
        self.tgt = DEFAULT.copy()
        self.obs = np.zeros(41, np.float32)
        self.k = 0
        self.cmd[:] = 0
        self.lo, self.hi = m.actuator_ctrlrange.T
        self.status = "walking"
        self._stand = None          # balance controller used after stop()
        self._stop_t0 = None

    def step(self, m, d, cmd=None):
        """One physics step (m.opt.timestep = 0.002 s)."""
        if cmd is not None:
            self.cmd[:] = cmd
        tau = np.empty(m.nu)
        tau[:10] = (self.tgt - d.qpos[7:17]) * KPS - d.qvel[6:16] * KDS
        up = ARM_POSE.copy()
        if self.status == "walking":
            diff = 0.5 * (d.qpos[14] - d.qpos[9])            # right hip ahead of left -> +, symmetric
            up[1] = np.clip(ARM_SWING * diff, -ARM_SWING_MAX, ARM_SWING_MAX)  # left arm follows right leg
            up[5] = -up[1]                                                     # right arm follows left leg
        tau[10:] = UPPER_KP * (up - d.qpos[17:]) - UPPER_KD * d.qvel[16:]
        d.ctrl[:] = np.clip(tau, self.lo, self.hi)
        mujoco.mj_step(m, d)
        self.k += 1
        if self.k % DECIMATION == 0:
            ph = (self.k * m.opt.timestep) % PERIOD / PERIOD
            o = self.obs
            o[:3] = d.qvel[3:6] * ANG_VEL_SCALE
            o[3:6] = gravity_orientation(d.qpos[3:7])
            o[6:9] = self.cmd * CMD_SCALE
            o[9:19] = d.qpos[7:17] - DEFAULT
            o[19:29] = d.qvel[6:16] * DOF_VEL_SCALE
            o[29:39] = self.act
            o[39:41] = (math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph))
            self.act = self.pol(o).astype(np.float32)
            self.tgt = self.act * ACTION_SCALE + DEFAULT
        if d.qpos[2] < 0.6:
            self.status = "fallen"
    
    def stop(self, m, d):
        """Stop at the goal and stand still: step in place until both feet are flat and level
        with each other, then hand over to the Stand balance controller (stand_fix.py)."""
        if self.status == "fallen":
            mujoco.mj_step(m, d)
            return
        if self.status == "standing":
            self._stand.apply_stand_control(m, d)
            mujoco.mj_step(m, d)
            if d.qpos[2] < 0.6:
                self.status = "fallen"
            return
        if self.status != "stopping":
            self.status, self._stop_t0 = "stopping", d.time
        self.step(m, d, (0.0, 0.0, 0.0))                      # step in place while slowing down
        if self.status == "fallen":
            return
        (xy_l, z_l), (xy_r, z_r) = sole(m, d, "left"), sole(m, d, "right")
        yaw = yaw_of(d)
        ahead = abs((xy_l[0] - xy_r[0]) * math.cos(yaw) + (xy_l[1] - xy_r[1]) * math.sin(yaw))
        feet_down = max(z_l, z_r) < 0.01
        slow = float(np.linalg.norm(d.qvel[:2])) < 0.15
        waited = d.time - self._stop_t0
        if waited > 1.0 and ((feet_down and ahead < 0.06 and slow) or waited > 4.0):
            from stand_fix import StandBalancer                  # same controller as Stand mode
            self._stand = StandBalancer(m, mujoco.MjData(m))     # scratch data: live robot untouched
            self.status = "standing"
    

    def follow(self, m, d, yaw_des):
        """Navigator mode: walk toward heading yaw_des (rad); None = stop (step in place)."""
        if yaw_des is None:
            self.step(m, d, (0.0, 0.0, 0.0))
            return
        err = wrap(yaw_des - yaw_of(d))
        rate = float(np.clip(YAW_GAIN * err, -YAW_RATE_MAX, YAW_RATE_MAX))
        vx = V_MAX * max(0.0, 1.0 - abs(err) / TURN_IN_PLACE)
        self.step(m, d, (vx, 0.0, rate))
