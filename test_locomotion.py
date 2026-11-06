"""
M0NARCH locomotion test suite (Joe's errors E10-E15, E17). Run from ~/M0NARCH with the monarch venv:

  python test_locomotion.py walk     # E10-E13: normal Walk, original vs fixed gait
  python test_locomotion.py avoid    # E14 + E17: Walk Avoid on the obstacle course, original vs fixed
  python test_locomotion.py stand    # E15: server's Stand mode, 60 simulated s + two 40 N pushes
  python test_locomotion.py stand --push=20   # try a different push force (N)
  python test_locomotion.py all
"""
import os, sys, math, time, types, numpy as np, mujoco
import gait_fix as G

NORMAL = "H1_robot-Physics-simulation/models/unitree_h1/scene.xml"
OBST   = "H1_robot-RL-Perception/models/unitree_h1/scene_obstacles.xml"
DT_CYCLES = 1.0 / G.GAIT_FREQ


# ---------------------------------------------------------------- original gait (copied from server.py)
def original_step(m, d, sim, avoid=False):
    dt = m.opt.timestep
    sim.walk_phase += 2 * math.pi * 1.30 * dt
    p = sim.walk_phase; s = math.sin(p)
    q = {"left_hip_pitch": -0.35 + 0.38 * s, "right_hip_pitch": -0.35 - 0.38 * s,
         "left_knee": 0.65 + 0.60 * max(0, -s) ** 1.3, "right_knee": 0.65 + 0.60 * max(0, s) ** 1.3,
         "left_ankle": -0.35 - 0.25 * max(0, -s) + 0.15 * max(0, s),
         "right_ankle": -0.35 - 0.25 * max(0, s) + 0.15 * max(0, -s),
         "left_shoulder_pitch": 0.38 * s, "right_shoulder_pitch": -0.38 * s,
         "left_shoulder_roll": 0.12, "right_shoulder_roll": -0.12,
         "left_elbow": 0.5, "right_elbow": 0.5, "torso": -0.05 * s}
    for n, v in q.items(): G._set(m, d, n, v)
    ty = G.avoid_target_y(d.qpos[0]) if avoid else 0.0
    d.qpos[0] += 0.30 * dt
    d.qpos[1] += (ty - d.qpos[1]) * 0.01
    d.qpos[2] = 0.96 + 0.015 * math.cos(2 * p)
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[:] = 0; mujoco.mj_forward(m, d); d.time += dt


# ---------------------------------------------------------------- measurement
def run(xml, stepper, avoid, seconds):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    d.qpos[0:3] = [0, 0, 1.05]; mujoco.mj_forward(m, d)
    sim = types.SimpleNamespace(walk_phase=0.0)
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_box")
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    elbow = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    rows, box_hits, pit_frames, min_box = [], 0, 0, 9.0
    fromto = np.zeros(6)
    for k in range(int(seconds / m.opt.timestep)):
        stepper(m, d, sim, avoid)
        (lxy, lz), (rxy, rz) = G.sole(m, d, "left"), G.sole(m, d, "right")
        rows.append([d.time, *d.qpos[:3], *lxy, lz, *rxy, rz, d.xpos[elbow][0]])
        if box >= 0:
            if any(box in (c.geom1, c.geom2) for c in d.contact[:d.ncon]): box_hits += 1
            if k % 25 == 0:
                for g in robot_geoms:
                    min_box = min(min_box, mujoco.mj_geomDistance(m, d, g, box, 1.0, fromto))
            for xy in (lxy, rxy):
                if 2.0 < xy[0] < 3.2 and abs(xy[1]) < 0.7: pit_frames += 1
    a = np.array(rows); t, px, py, pz, lx, ly, lz, rx, ry, rz, ex = a.T
    dt = t[1] - t[0]; sel = t > DT_CYCLES                       # skip the first cycle
    lv = np.hypot(np.gradient(lx, dt), np.gradient(ly, dt)); rv = np.hypot(np.gradient(rx, dt), np.gradient(ry, dt))
    left_down = lz <= rz
    st_v = np.where(left_down, lv, rv)[sel]; st_z = np.minimum(lz, rz)[sel]; sw_z = np.maximum(lz, rz)[sel]
    cycles = (t[sel][-1] - t[sel][0]) * G.GAIT_FREQ
    out = dict(
        speed=np.hypot(px[sel][-1] - px[sel][0], py[sel][-1] - py[sel][0]) / (t[sel][-1] - t[sel][0]),
        slip_per_cycle=st_v.sum() * dt / cycles,
        stance_z_mm=(st_z.min() * 1000, st_z.max() * 1000),
        swing_clear_mm=sw_z.max() * 1000,
        arm_leg_corr=np.corrcoef(lx - px, ex - px)[0, 1] if not avoid else float("nan"),
        distance_x=px[-1])
    if avoid:
        ty = np.array([G.avoid_target_y(x) for x in px])
        out.update(path_err_max=np.abs(py - ty)[sel].max(), feet_over_pit=pit_frames,
                   box_contacts=box_hits, min_box_gap_mm=min_box * 1000)
    return out


def show(title, before, after):
    print(f"\n=== {title} ===")
    print(f"{'metric':<20}{'ORIGINAL':>22}{'FIXED':>22}")
    for k in before:
        f = lambda v: f"{v[0]:.0f} to {v[1]:.0f}" if isinstance(v, tuple) else f"{v:.3f}"
        print(f"{k:<20}{f(before[k]):>22}{f(after[k]):>22}")


def test_walk():
    show("E10-E13  Walk (normal floor, 6 s)",
         run(NORMAL, original_step, False, 6), run(NORMAL, G.walk_step, False, 6))
    print("PASS if FIXED: slip_per_cycle < 0.01 m, stance_z 0 +-2 mm, swing_clear 50-100 mm, arm_leg_corr < 0")


def test_avoid():
    show("E14 + E17  Walk Avoid (obstacle course, 30 s)",
         run(OBST, original_step, True, 30), run(OBST, G.walk_step, True, 30))
    print("PASS if FIXED: slip < 0.01 m, feet_over_pit = 0, box_contacts = 0, min_box_gap > 0")


def test_stand(push=40.0, sim_seconds=60.0):
    """Runs the server's own STAND controller. The render thread is not started, so the
    physics runs at full speed (see E18), and all times below are SIMULATED seconds."""
    import threading
    os.environ.setdefault("MUJOCO_GL", "egl")
    _start = threading.Thread.start
    def start_no_render(self):
        if getattr(self, "_target", None) is not None and self._target.__name__ == "render_worker":
            return                                   # skip rendering for this test
        _start(self)
    threading.Thread.start = start_no_render
    sys.argv = [sys.argv[0]]
    import server                                   # starts the control thread only
    server.execute_command("stand")
    print("\n=== E15  Stand (server STAND mode, simulated time) ===")
    d = server.data_normal
    def now():
        with server.sim.lock: return float(d.time), float(d.qpos[2]), float(d.qpos[0])
    pushes = [(20.0, 0, "forward"), (40.0, 1, "sideways")]
    next_print, z_min = 0.0, 9.0
    while True:
        t, z, x = now()
        z_min = min(z_min, z)
        for (tp, axis, name) in pushes:
            with server.sim.lock:
                if tp <= d.time < tp + 0.2: d.xfrc_applied[1, axis] = push
                elif tp + 0.2 <= d.time < tp + 0.3: d.xfrc_applied[1, :] = 0.0
        if t >= next_print:
            print(f"sim t={t:5.1f}s  pelvis z={z:.3f} m  x={x:+.3f} m"); next_print += 5.0
        if t >= sim_seconds: break
        time.sleep(0.001)
    ok = z > 0.8 and z_min > 0.8
    print(f"pushes: {push:.0f} N forward at 20 s, {push:.0f} N sideways at 40 s; lowest pelvis z={z_min:.3f} m")
    print("RESULT E15:", "PASS (standing after 60 simulated s + 2 pushes)" if ok else "FAIL (fell)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("walk", "all"): test_walk()
    if what in ("avoid", "all"): test_avoid()
    push = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--push=")), 40))
    if what in ("stand", "all"): test_stand(push)
