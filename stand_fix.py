"""
M0NARCH Stand fix (E15): PD + centre-of-mass balance for the full Unitree H1.

The original Stand controller fell after a 10 N push. This is the controller
tested bottom-up on legs -> torso -> full robot (h1_staged.py): it survives 60 N.

  1. Static balance: pick a pose with the CoM exactly over the middle of the feet,
                      and blend into it over 1.5 s (the 'home' pose has the CoM 78 mm behind the feet).
  2. Joint PD:        torque = KP (q_ref - q) - KD q_dot, clipped to motor limits.
  3. CoM feedback:    shift ankle (forward/back) and hip-roll (sideways) targets
                      toward where the CoM is heading (capture point, T = sqrt(h/g)).
Rule of thumb: total ankle stiffness (2 x 400) must exceed m*g*h (~480 for H1).
"""
import numpy as np
import mujoco

STAND_POSE = {"hip_pitch": -0.4, "knee": 0.8, "ankle": -0.4}   # hip + knee + ankle = 0 -> flat feet
GAINS = {"hip": 500, "knee": 600, "ankle": 400, "torso": 300, "shoulder": 60, "elbow": 40}
KD_RATIO = 1 / 25
K_BAL_X, K_BAL_Y, T_CP = 3.0, 2.0, 0.3
RAMP_S = 1.5          # s, blend from the starting pose into the balanced pose (a jump knocks it over)
FEET = ("left_ankle_link", "right_ankle_link")


class StandBalancer:
    def __init__(self, m, d):
        # Like the original controller: start the live robot in its 'home' standing pose.
        # (Without this, the server starts with straight legs at 1.06 m and the robot falls.)
        home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            d.qpos[:] = m.key_qpos[home]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        act_j = [m.actuator_trnid[i, 0] for i in range(m.nu)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in act_j])
        self.vadr = np.array([m.jnt_dofadr[j] for j in act_j])
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in act_j]
        self.kp = np.array([next((v for k, v in GAINS.items() if k in n), 100.0) for n in self.names], float)
        self.kd = self.kp * KD_RATIO
        self.lo, self.hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self.sag = [self.names.index(n) for n in ("left_ankle", "right_ankle")]
        self.lat = [self.names.index(n) for n in ("left_hip_roll", "right_hip_roll")]
        fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FEET]
        self.foot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in fb and m.geom_contype[g]]
        self.q_ref = self._solve_pose(m)
        self._t_last, self._q_start, self._t_start = None, None, 0.0

    def support_centre(self, d):
        return np.mean([d.geom_xpos[g] for g in self.foot_geoms], axis=0)

    def _solve_pose(self, m):
        """Hip/ankle split (feet flat, pelvis upright) that puts the CoM over the feet."""
        d = mujoco.MjData(m)                                # scratch copy, live data untouched
        q0 = np.array([next((v for k, v in STAND_POSE.items() if n.endswith(k)), 0.0) for n in self.names])
        hips = [i for i, n in enumerate(self.names) if n.endswith("hip_pitch")]
        ankles = [i for i, n in enumerate(self.names) if n.endswith("ankle")]
        def off(h):
            q = q0.copy(); q[hips] += h; q[ankles] -= h
            d.qpos[:] = 0; d.qpos[2] = 1.0; d.qpos[3] = 1.0; d.qpos[self.qadr] = q
            mujoco.mj_forward(m, d)
            return d.subtree_com[1][0] - self.support_centre(d)[0], q
        lo, hi = -0.6, 0.6
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if np.sign(off(mid)[0]) == np.sign(off(lo)[0]): lo = mid
            else: hi = mid
        return off(0.5 * (lo + hi))[1]

    def apply_stand_control(self, m, d):
        """Same call signature as the old H1BalanceController.apply_stand_control."""
        if self._t_last is None or d.time < self._t_last:      # first call, or Stand was re-clicked (time reset)
            self._q_start, self._t_start = d.qpos[self.qadr].copy(), d.time
        self._t_last = d.time
        s = min(1.0, (d.time - self._t_start) / RAMP_S)
        ref = self._q_start + (self.q_ref - self._q_start) * s
        err = d.subtree_com[1] - self.support_centre(d) + T_CP * d.subtree_linvel[1]
        w, x, y, z = d.qpos[3:7]                                  # rotate into the robot's heading frame
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fwd = np.cos(yaw) * err[0] + np.sin(yaw) * err[1]
        side = -np.sin(yaw) * err[0] + np.cos(yaw) * err[1]
        ref[self.sag] += K_BAL_X * fwd
        ref[self.lat] += K_BAL_Y * side
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        d.ctrl[:] = np.clip(self.kp * (ref - q) - self.kd * qd, self.lo, self.hi)
