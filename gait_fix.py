"""
M0NARCH gait fix (E10-E14): corrected kinematic walk for the Unitree H1.

Replaces the sine-wave WALK block in server.py. Still kinematic playback
(qpos + mj_forward, E9 = future work), but now:
  E10  stance/swing phases: knee lifts only while the leg swings forward
  E11  stance foot is pinned to the floor; the pelvis moves from it (no sliding)
  E12  pelvis height is computed so the stance sole sits exactly on the floor
  E13  each arm swings with the OPPOSITE leg
  E14  Walk Avoid steers by turning (yaw) instead of sliding the body sideways
"""
import math
import numpy as np
import mujoco

# ---- gait parameters --------------------------------------------------------
GAIT_FREQ   = 1.30    # steps cycles per second (same cadence as the original)
HIP_AMP     = 0.08    # rad, hip sweep each side of centre -> ~0.30 m/s
HIP_CENTRE  = -0.40   # rad, matches the H1 'home' keyframe
KNEE_STAND  = 0.80    # rad, knee bend in stance (home keyframe)
KNEE_LIFT   = 0.40    # rad, extra knee bend at mid-swing (foot clearance)
ARM_GAIN    = 0.30    # rad of shoulder pitch at full hip sweep
YAW_RATE    = 1.0     # rad/s, max turning rate in Walk Avoid
YAW_MAX     = 1.0     # rad, max heading away from straight ahead
LOOKAHEAD   = 0.5     # m, steer toward the path point this far ahead

_FEET = {}


def _foot_geoms(m):
    """Contact capsules of each foot (cached per model)."""
    if id(m) not in _FEET:
        ids = {}
        for side in ("left", "right"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_link")
            ids[side] = [g for g in range(m.ngeom)
                         if m.geom_bodyid[g] == b and m.geom_contype[g]]
        _FEET[id(m)] = ids
    return _FEET[id(m)]


def sole(m, d, side):
    """(x, y) centre and lowest z of one foot's contact capsules."""
    pts, zs = [], []
    for g in _foot_geoms(m)[side]:
        c = d.geom_xpos[g]
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
        r, h = m.geom_size[g][:2]
        for e in (c + ax * h, c - ax * h):
            pts.append(e[:2])
            zs.append(e[2] - r)
    return np.mean(pts, axis=0), min(zs)


def leg(u):
    """Hip, knee, ankle for one leg at its own phase u in [0,1).
    0-0.5 stance (hip sweeps back at constant rate), 0.5-1 swing (hip returns, knee lifts)."""
    if u < 0.5:
        hip = HIP_CENTRE - HIP_AMP + 2 * HIP_AMP * (u / 0.5)
        knee = KNEE_STAND
    else:
        w = (u - 0.5) / 0.5
        hip = HIP_CENTRE + HIP_AMP - HIP_AMP * (1 - math.cos(math.pi * w))
        knee = KNEE_STAND + KNEE_LIFT * math.sin(math.pi * w)
    return hip, knee, -(hip + knee)          # hip + knee + ankle = 0 -> flat foot


def avoid_target_y(x):
    """Original M0NARCH Walk Avoid path: bypass the pit on the right, return before the box."""
    if x < 1.2:
        return 0.0
    if x < 2.0:
        return 1.2 * (x - 1.2) / 0.8
    if x < 3.6:
        return 1.2
    if x < 4.8:
        return 1.2 - 1.2 * (x - 3.6) / 1.2
    return 0.0


def _set(m, d, name, value):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j >= 0:
        d.qpos[m.jnt_qposadr[j]] = value


def walk_step(m, d, sim, avoid=False, yaw_des=None):
    """One 2 ms step of the corrected walk. Call instead of the old WALK block.
    yaw_des: heading command (rad, world frame) from a planner, e.g. perception_nav.Navigator.
             If given, it overrides the built-in Walk Avoid path."""
    dt = m.opt.timestep
    if sim.walk_phase == 0.0 or not hasattr(sim, "gait_stance"):
        sim.gait_stance, sim.gait_anchor, sim.gait_yaw = None, None, 0.0

    sim.walk_phase += 2.0 * math.pi * GAIT_FREQ * dt
    u = (sim.walk_phase / (2.0 * math.pi)) % 1.0
    lh, lk, la = leg(u)
    rh, rk, ra = leg((u + 0.5) % 1.0)

    # legs
    for side, (h, k, a) in (("left", (lh, lk, la)), ("right", (rh, rk, ra))):
        _set(m, d, side + "_hip_yaw", 0.0)
        _set(m, d, side + "_hip_roll", 0.0)
        _set(m, d, side + "_hip_pitch", h)
        _set(m, d, side + "_knee", k)
        _set(m, d, side + "_ankle", a)
    _set(m, d, "torso", 0.0)

    # arms: each shoulder follows the OPPOSITE hip (E13)
    _set(m, d, "left_shoulder_pitch", ARM_GAIN * (rh - HIP_CENTRE) / HIP_AMP)
    _set(m, d, "right_shoulder_pitch", ARM_GAIN * (lh - HIP_CENTRE) / HIP_AMP)
    _set(m, d, "left_shoulder_roll", 0.12)
    _set(m, d, "right_shoulder_roll", -0.12)
    _set(m, d, "left_shoulder_yaw", 0.0)
    _set(m, d, "right_shoulder_yaw", 0.0)
    _set(m, d, "left_elbow", 0.50)
    _set(m, d, "right_elbow", 0.50)

    # heading: turn toward the path instead of sliding sideways (E14)
    if yaw_des is not None:                     # heading from the navigator
        err = math.atan2(math.sin(yaw_des - sim.gait_yaw), math.cos(yaw_des - sim.gait_yaw))
        yaw_des = sim.gait_yaw + err            # unwrap: take the short way round
    elif avoid:
        x, y = d.qpos[0], d.qpos[1]               # pure-pursuit: aim at a point ahead on the path
        yaw_des = float(np.clip(math.atan2(avoid_target_y(x + LOOKAHEAD) - y, LOOKAHEAD), -YAW_MAX, YAW_MAX))
    else:
        yaw_des = 0.0
    sim.gait_yaw += float(np.clip(yaw_des - sim.gait_yaw, -YAW_RATE * dt, YAW_RATE * dt))
    d.qpos[3:7] = [math.cos(sim.gait_yaw / 2), 0.0, 0.0, math.sin(sim.gait_yaw / 2)]

    # pin the stance foot (E11) and put its sole on the floor (E12)
    stance = "left" if u < 0.5 else "right"
    mujoco.mj_forward(m, d)
    sxy, sz = sole(m, d, stance)
    if stance != sim.gait_stance:               # touchdown: remember where this foot landed
        sim.gait_stance, sim.gait_anchor = stance, sxy.copy()
    d.qpos[0:2] += sim.gait_anchor - sxy
    d.qpos[2] -= sz

    d.qvel[:] = 0.0                              # still kinematic playback (E9: future work)
    mujoco.mj_forward(m, d)
    d.time += dt
