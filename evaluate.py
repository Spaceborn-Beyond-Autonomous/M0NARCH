"""
M0NARCH evaluation: test STAND, WALK and NAVIGATION in any environment, then give a
REAL-WORLD READINESS score (0-100) for that environment.

  python evaluate.py                                    # warehouse course (scene_course2.xml)
  python evaluate.py --env=H1_robot-RL-Perception/models/unitree_h1/scene_obstacles.xml --goal=7.9,0
  python evaluate.py --quick                            # fewer trials
  python evaluate.py --nocam                            # LiDAR only (faster)

The environment needs: H1 with free base starting at (0, 0), floor top at z = 0, and a goal
(site "nav_goal" in the XML, or --goal=x,y).

WHY A READINESS SCORE: the simulator is perfect (exact sensors, exact position, exact masses).
The real world is not. So every test is repeated with realism added step by step, and the
score measures how much performance survives:

  level 0  perfect simulator
  level 1  + sensor noise: LiDAR range noise 2 cm, 2% dropped beams, camera depth noise 2%
  level 2  + localisation error: position drifts ~10 cm, heading ~2 deg (a real robot must
             estimate where it is; SLAM/odometry is never exact)
  level 3  + harsh: noise 4 cm, 5% dropouts, depth 4%, position ~20 cm, heading ~4 deg

Score parts (each 0..1):
  N  navigation   success rate, weighted toward the realistic levels (2 and 3)
  S  safety       closest approach to obstacles at the realistic levels vs a 0.20 m target
  B  balance      Stand mode survival with random mass (+-15%), friction (0.5-1.0),
                  motor strength (80-100%) and random pushes (20-80 N)
  L  locomotion   physics walking (E9 fix, physics_walk.py) under random mass (+-15%),
                  friction (0.5-1.0), motor strength (80-100%), random speed/turn commands and
                  random pushes (50-150 N): share of walks that stay up AND move at >= 60% of
                  the commanded speed
  READINESS = 100 * N^0.35 * S^0.15 * B^0.20 * L^0.30   (geometric: the weakest part dominates)

The score is a heuristic for comparing environments and versions, NOT a guarantee about
the real robot.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

import physics_walk as P
from perception_nav import Navigator, NavConfig, OccupancyGrid, Planner
from stand_fix import StandBalancer

DEFAULT_ENV = "H1_robot-Physics-simulation/models/unitree_h1/scene_course2.xml"
LEVELS = {
    0: dict(),
    1: dict(lidar_noise_m=0.02, lidar_dropout=0.02, depth_noise_frac=0.02),
    2: dict(lidar_noise_m=0.02, lidar_dropout=0.02, depth_noise_frac=0.02, odom_drift_m=0.03, odom_drift_deg=0.5),
    3: dict(lidar_noise_m=0.04, lidar_dropout=0.05, depth_noise_frac=0.04, odom_drift_m=0.06, odom_drift_deg=1.0),
}
LEVEL_WEIGHT = {0: 0.10, 1: 0.20, 2: 0.35, 3: 0.35}
LEVEL_NAME = {0: "perfect", 1: "sensor noise", 2: "+ position drift", 3: "harsh"}


def arg(key, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{key}=")), default)


# ------------------------------------------------------------------ environment analysis
def truth_map(m, cfg, start, goal):
    """Ground-truth occupancy from vertical rays (works for ANY environment)."""
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    grid = OccupancyGrid(cfg, start)
    lo = np.minimum(start, goal) - 6.0; hi = np.maximum(start, goal) + 6.0
    i0, j0 = np.clip(grid.to_idx(lo), 0, grid.n - 1); i1, j1 = np.clip(grid.to_idx(hi), 0, grid.n - 1)
    occ = np.ones((grid.n, grid.n), bool)                # outside the scanned box: treat as blocked
    groups = np.array([1, 1, 0, 0, 0, 0], np.uint8); gid = np.zeros(1, np.int32)
    down = np.array([0.0, 0.0, -1.0])
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            x, y = grid.to_xy((i, j))
            dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), down, groups, 1, -1, gid)
            z = 3.0 - dist if dist >= 0 else -99
            occ[i, j] = not (abs(z) <= cfg.obstacle_min_h)  # floor = free; obstacle, overhead, hole = blocked
    return grid, occ


def analyse_env(m, cfg, start, goal):
    grid, occ = truth_map(m, cfg, start, goal)
    pl = Planner(cfg)
    _, _, dist = pl.costmap(occ)
    blocked = dist < cfg.robot_radius
    raw = pl.astar(blocked, np.ones_like(blocked, float), tuple(grid.to_idx(start)), tuple(grid.to_idx(goal)))
    if raw is None:
        return None
    path = pl.shortcut(blocked, raw)
    pts = grid.to_xy(np.array(path)); pts[0] = start; pts[-1] = goal
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    cells = np.array(raw)
    narrow = float(2 * dist[cells[:, 0], cells[:, 1]].min())      # narrowest gap the robot passes (m)
    detour = length / float(np.linalg.norm(goal - start))
    return dict(shortest_m=length, straight_m=float(np.linalg.norm(goal - start)),
                detour=detour, narrowest_gap_m=narrow)


def difficulty(env):
    """0 (easy) .. 100 (very hard), from detour and narrowest passage."""
    narrow = np.clip((2.0 - env["narrowest_gap_m"]) / (2.0 - 0.9), 0, 1)   # 2 m wide = easy, 0.9 m = robot width
    det = np.clip((env["detour"] - 1.0) / 1.0, 0, 1)                        # 2x the straight line = hard
    return round(100 * (0.6 * narrow + 0.4 * det))


# ------------------------------------------------------------------ tests
def stand_trials(xml, n, rng):
    """Stand under random physics and random pushes (uses the real Stand controller)."""
    ok = 0; rows = []
    for k in range(n):
        m = P.load_model(xml); d = mujoco.MjData(m)
        ms, fr, mot = rng.uniform(0.85, 1.15), rng.uniform(0.5, 1.0), rng.uniform(0.8, 1.0)
        push, ang = rng.uniform(20, 80), rng.uniform(0, 2 * np.pi)
        m.body_mass[1:] *= ms
        m.geom_friction[:, 0] *= fr
        m.actuator_ctrlrange[:] *= mot
        bal = StandBalancer(m, d)
        for _ in range(int(8.0 / m.opt.timestep)):
            if 3.0 <= d.time < 3.2:
                d.xfrc_applied[1, :2] = push * np.array([math.cos(ang), math.sin(ang)])
            else:
                d.xfrc_applied[1, :2] = 0
            bal.apply_stand_control(m, d); mujoco.mj_step(m, d)
        up = d.qpos[2] > 0.8; ok += up
        rows.append(dict(mass=round(ms, 2), friction=round(fr, 2), motor=round(mot, 2), push_N=round(push), stood=bool(up)))
    return ok / n, rows


FLAT = "H1_robot-Physics-simulation/models/unitree_h1/scene.xml"


def walk_trials(n, rng):
    """Physics walking (E9 fix) on flat floor under random physics, commands and pushes.
    (Flat floor on purpose: this tests the legs, not the navigator. Walking blind in the
    course would just walk into its obstacles.)"""
    ok = 0; rows = []
    for k in range(n):
        m = P.load_model(FLAT); d = mujoco.MjData(m)
        ms, fr, mot = rng.uniform(0.85, 1.15), rng.uniform(0.5, 1.0), rng.uniform(0.8, 1.0)
        vx, rate = rng.uniform(0.3, 0.8), rng.uniform(-0.3, 0.3)
        push, ang, t_push = rng.uniform(50, 150), rng.uniform(0, 2 * np.pi), rng.uniform(2.0, 4.0)
        m.body_mass[1:] *= ms
        m.geom_friction[:, 0] *= fr
        m.actuator_ctrlrange[:] *= mot
        w = P.PhysicsWalker(); w.reset(m, d)
        dist = 0.0; prev = d.qpos[:2].copy()
        while d.time < 6.0 and w.status != "fallen":
            d.xfrc_applied[1, :2] = push * np.array([math.cos(ang), math.sin(ang)]) if t_push <= d.time < t_push + 0.2 else 0
            w.step(m, d, (vx, 0.0, rate))
            dist += float(np.linalg.norm(d.qpos[:2] - prev)); prev = d.qpos[:2].copy()
        speed = dist / max(d.time, 1e-3)
        good = w.status != "fallen" and speed >= 0.6 * vx
        ok += good
        rows.append(dict(mass=round(ms, 2), friction=round(fr, 2), motor=round(mot, 2), cmd_vx=round(vx, 2),
                         cmd_yaw_rate=round(rate, 2), push_N=round(push), speed=round(speed, 2),
                         fell=w.status == "fallen", ok=bool(good)))
    return ok / n, rows


def nav_run(xml, goal, level, seed, use_camera, t_max, **extra):
    cfg = NavConfig(use_camera=use_camera, noise_seed=seed, **{**LEVELS[level], **extra})
    m = P.load_model(xml); d = mujoco.MjData(m)
    w = P.PhysicsWalker(); w.reset(m, d)
    nav = Navigator(cfg); nav.live_view = False; nav.reset(m, d, goal)
    world = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_contype[g]]
    obstacles = []
    for g in world:                                      # obstacles = world geoms that rise above the floor
        top = d.geom_xpos[g][2] + (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else
                                   m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else 0)
        if top > 0.05:
            obstacles.append(g)
    obst = set(obstacles)
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    groups = np.array([1, 1, 0, 0, 0, 0], np.uint8); gid = np.zeros(1, np.int32); ft = np.zeros(6)
    coll = void = 0; gap = 9.0; k = 0
    while d.time < t_max and w.status != "fallen":
        yaw = nav.step(m, d)
        if yaw is None and nav.status == "goal reached":
            break
        w.follow(m, d, yaw)                              # physics walking; None = step in place
        if any(c.geom1 in obst or c.geom2 in obst for c in d.contact[:d.ncon]):
            coll += 1
        if k % 10 == 0:
            for s in ("left", "right"):
                xy, z = P.sole(m, d, s)
                r = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z + 0.05]), np.array([0, 0, -1.0]), groups, 1, -1, gid)
                void += (r < 0 or r > 0.15)
        if k % 125 == 0:
            for g in robot:
                for o in obstacles:
                    gap = min(gap, mujoco.mj_geomDistance(m, d, g, o, 2.0, ft))
        k += 1
    true_err = float(np.linalg.norm(d.qpos[:2] - goal))
    reached = nav.status == "goal reached" and true_err < cfg.goal_tol + 0.4
    if nav.sensors.renderer is not None:
        nav.sensors.renderer.close()
    fell = w.status == "fallen"
    return dict(level=level, seed=seed, ok=bool(reached and coll == 0 and void == 0 and not fell), reached=bool(reached),
                fell=fell,
                collisions=coll, void_steps=void, min_gap_m=gap, walked_m=nav.distance, time_s=d.time)


# ------------------------------------------------------------------ main
def main():
    xml = arg("env", DEFAULT_ENV)
    quick = "--quick" in sys.argv
    cam = "--nocam" not in sys.argv
    m = P.load_model(xml)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "nav_goal")
    if arg("goal", None):
        goal = np.array([float(v) for v in arg("goal", "").split(",")])
    elif sid >= 0:
        goal = m.site_pos[sid][:2].copy()
    else:
        sys.exit("No goal: add a site named nav_goal to the XML or pass --goal=x,y")
    start = np.zeros(2)
    cfg = NavConfig()
    t0 = time.time()
    print(f"M0NARCH evaluation  env={xml}  goal=({goal[0]:.1f}, {goal[1]:.1f})  sensors=LiDAR{' + camera' if cam else ''}\n")

    env = analyse_env(m, cfg, start, goal)
    if env is None:
        sys.exit("No safe path exists from start to goal in this environment (for a 0.45 m robot radius).")
    diff = difficulty(env)
    print(f"[ENV]   shortest safe path {env['shortest_m']:.1f} m (straight line {env['straight_m']:.1f} m, detour x{env['detour']:.2f}), "
          f"narrowest passage {env['narrowest_gap_m']:.2f} m  ->  difficulty {diff}/100")

    rng = np.random.default_rng(1)
    B, stand_rows = stand_trials(xml, 6 if quick else 12, rng)
    print(f"[STAND] {sum(r['stood'] for r in stand_rows)}/{len(stand_rows)} stood with random mass/friction/motor/push   B = {B:.2f}")

    L, walk_rows = walk_trials(6 if quick else 12, rng)
    fell = sum(r["fell"] for r in walk_rows)
    print(f"[WALK]  {sum(r['ok'] for r in walk_rows)}/{len(walk_rows)} physics walks OK with random mass/friction/motor/"
          f"command/push 50-150 N ({fell} fell)   L = {L:.2f}")

    t_max = env["shortest_m"] / 0.20 + 40.0
    print("\n[NAV]   level  realism            runs  success  collisions  void  falls  min_gap_m  path/shortest")
    seeds = [0, 1] if quick else [0, 1, 2]
    runs = []
    for lv in LEVELS:
        rs = [nav_run(xml, goal, lv, s, cam, t_max) for s in seeds]
        runs += rs
        succ = np.mean([r["ok"] for r in rs])
        good = [r for r in rs if r["ok"]]
        ratio = np.median([r["walked_m"] / max(env["shortest_m"] - cfg.goal_tol, 0.1) for r in good]) if good else float("nan")
        print(f"        {lv:>5}  {LEVEL_NAME[lv]:<17}{len(rs):>5}  {succ:>6.0%}  {sum(r['collisions'] for r in rs):>10}  "
              f"{sum(r['void_steps'] for r in rs):>4}  {sum(r['fell'] for r in rs):>5}  {min(r['min_gap_m'] for r in rs):>9.2f}  {ratio:>13.2f}", flush=True)

    # blind check
    # blind run: LiDAR and camera OFF, so the map stays empty -> proves the sensors drive the behaviour
    bl = nav_run(xml, goal, 0, 0, False, t_max, use_lidar=False)
    print(f"        blind  sensors OFF        1  {'100%' if bl['ok'] else '  0%'}  {bl['collisions']:>10}  {bl['void_steps']:>4}"
          f"  -> {'sensors NOT needed?!' if bl['ok'] else 'fails without sensors: navigation really uses LiDAR + camera'}")

    N = sum(LEVEL_WEIGHT[lv] * np.mean([r["ok"] for r in runs if r["level"] == lv]) for lv in LEVELS)
    real = [r["min_gap_m"] for r in runs if r["level"] >= 2]
    S = float(np.clip(np.median(real) / 0.20, 0, 1)) if real else 0.0
    parts = dict(N=N, S=S, B=B, L=L)
    score = 100 * math.exp(sum(w * math.log(max(parts[k], 1e-3)) for k, w in (("N", .35), ("S", .15), ("B", .20), ("L", .30))))
    band = ("LIKELY FEASIBLE" if score >= 80 else "PROMISING, needs work" if score >= 60 else
            "SIGNIFICANT GAPS" if score >= 40 else "NOT READY")
    weakest = min(parts, key=parts.get)
    names = dict(N="navigation under realistic noise", S="safety margin", B="balance", L="locomotion (physics walking)")
    print(f"\n[SCORE] N navigation {N:.2f}   S safety {S:.2f}   B balance {B:.2f}   L locomotion {L:.2f}")
    print(f"        REAL-WORLD READINESS = {score:.0f}/100  ->  {band}")
    print(f"        environment difficulty = {diff}/100;  weakest part: {names[weakest]}")
    print(f"        (heuristic for comparing environments and versions, not a guarantee)   [{time.time() - t0:.0f} s]")

    out = dict(env=xml, goal=goal.tolist(), environment=env, difficulty=diff, stand=stand_rows, walk=walk_rows,
               navigation=runs, blind=bl, parts=parts, readiness=round(score, 1), band=band)
    name = os.path.splitext(os.path.basename(xml))[0]
    with open(f"readiness_{name}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"        saved readiness_{name}.json")


if __name__ == "__main__":
    main()
