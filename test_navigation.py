"""
Random-world navigation tests for M0NARCH (LiDAR + camera -> map -> A* -> physics walking).

Each test builds a NEW random world around the H1: a floor with edges (drop-offs),
random boxes and cylinders, sometimes a pit, and a random goal. The robot gets only
the goal; it must find its way using its sensors. A world is used only if a safe path
exists (checked on the true map), so every failure is the navigator's fault.

  python test_navigation.py                 # 10 random worlds, LiDAR + camera
  python test_navigation.py --n=20 --seed=7
  python test_navigation.py --nocam         # LiDAR only (faster)
  python test_navigation.py --save          # also save each world's final map as PNG

Per world it measures:
  reached      robot got within 0.4 m of the goal in time
  collisions   physics steps where the robot's collision shapes touched an obstacle
  void_steps   steps with a foot over empty space (pit or off the floor edge)
  min_gap_mm   closest distance between robot and any obstacle
  path_ratio   distance walked / true shortest safe path (1.0 = perfect)
PASS = reached, did not fall, 0 collisions and 0 void steps.
"""
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

import physics_walk as P
from perception_nav import Navigator, NavConfig, Planner

SCENE = "H1_robot-Physics-simulation/models/unitree_h1/scene.xml"
FLOOR = (-3.0, 12.0, -5.0, 5.0)          # x0, x1, y0, y1 of the walkable floor


def make_world(rng):
    """Random world spec + ground-truth description."""
    spec = mujoco.MjSpec.from_file(SCENE)
    spec.delete(spec.geom("floor"))
    wb = spec.worldbody
    goal = np.array([rng.uniform(6.0, 9.0), rng.uniform(-2.0, 2.0)])
    x0, x1, y0, y1 = FLOOR
    hole = None
    if rng.random() < 0.6:                            # a pit somewhere between start and goal
        w, h = rng.uniform(0.6, 1.2), rng.uniform(1.0, 3.0)
        cx, cy = rng.uniform(2.5, goal[0] - 2.0), rng.uniform(-1.5, 1.5)
        hole = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    def floor_box(name, ax0, ax1, ay0, ay1, z=-0.05, hz=0.05):
        if ax1 - ax0 < 1e-3 or ay1 - ay0 < 1e-3:
            return
        g = wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[(ax1 - ax0) / 2, (ay1 - ay0) / 2, hz], pos=[(ax0 + ax1) / 2, (ay0 + ay1) / 2, z])
        g.rgba = [0.25, 0.35, 0.45, 1]
    if hole:
        hx0, hx1, hy0, hy1 = hole
        floor_box("floor_a", x0, hx0, y0, y1); floor_box("floor_b", hx1, x1, y0, y1)
        floor_box("floor_c", hx0, hx1, y0, hy0); floor_box("floor_d", hx0, hx1, hy1, y1)
        floor_box("pit_bottom", hx0, hx1, hy0, hy1, z=-0.65, hz=0.02)
    else:
        floor_box("floor_a", x0, x1, y0, y1)
    obstacles = []
    for i in range(rng.integers(4, 10)):
        for _ in range(50):
            c = np.array([rng.uniform(1.5, goal[0] + 1.0), rng.uniform(-3.5, 3.5)])
            r = rng.uniform(0.15, 0.6)
            if np.linalg.norm(c) < 1.4 + r or np.linalg.norm(c - goal) < 1.0 + r:
                continue
            break
        hgt = rng.uniform(0.2, 1.4)
        if rng.random() < 0.5:
            sx, sy, yaw = rng.uniform(0.15, 0.6), rng.uniform(0.15, 0.6), rng.uniform(0, np.pi)
            g = wb.add_geom(name=f"obs{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, hgt / 2],
                            pos=[c[0], c[1], hgt / 2], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            obstacles.append(("box", c, (sx, sy, yaw)))
        else:
            g = wb.add_geom(name=f"obs{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, hgt / 2, 0],
                            pos=[c[0], c[1], hgt / 2])
            obstacles.append(("cyl", c, (r,)))
        g.rgba = [0.8, 0.45, 0.2, 1]
    return spec, goal, hole, obstacles


def truth_grid(cfg, hole, obstacles, center):
    """Ground-truth occupancy (what a perfect map would contain)."""
    nav_grid = __import__("perception_nav").OccupancyGrid(cfg, center)
    n = nav_grid.n
    xs = nav_grid.origin[0] + (np.arange(n) + 0.5) * cfg.res
    ys = nav_grid.origin[1] + (np.arange(n) + 0.5) * cfg.res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x0, x1, y0, y1 = FLOOR
    occ = (X < x0) | (X > x1) | (Y < y0) | (Y > y1)
    if hole:
        occ |= (X > hole[0]) & (X < hole[1]) & (Y > hole[2]) & (Y < hole[3])
    for kind, c, p in obstacles:
        if kind == "cyl":
            occ |= (X - c[0]) ** 2 + (Y - c[1]) ** 2 < p[0] ** 2
        else:
            sx, sy, yaw = p
            u = (X - c[0]) * math.cos(yaw) + (Y - c[1]) * math.sin(yaw)
            v = -(X - c[0]) * math.sin(yaw) + (Y - c[1]) * math.cos(yaw)
            occ |= (np.abs(u) < sx) & (np.abs(v) < sy)
    return nav_grid, occ


def shortest_safe(cfg, grid, occ, start, goal):
    pl = Planner(cfg)
    _, _, dist = pl.costmap(occ)
    blocked = dist < cfg.robot_radius
    path = pl.astar(blocked, np.ones_like(blocked, float), tuple(grid.to_idx(start)), tuple(grid.to_idx(goal)))
    if path is None:
        return None
    path = pl.shortcut(blocked, path)
    pts = grid.to_xy(np.array(path)); pts[0] = start; pts[-1] = goal
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def run_world(seed, use_camera=True, save=False):
    rng = np.random.default_rng(seed)
    cfg = NavConfig(use_camera=use_camera)
    for _ in range(30):                                  # resample until a safe path exists
        spec, goal, hole, obstacles = make_world(rng)
        grid, occ = truth_grid(cfg, hole, obstacles, (0.0, 0.0))
        best = shortest_safe(cfg, grid, occ, np.zeros(2), goal)
        if best is not None:
            break
    m = P.add_soles(spec).compile(); d = mujoco.MjData(m)
    walker = P.PhysicsWalker(); walker.reset(m, d)            # E9 fix: real physics walking
    nav = Navigator(cfg); nav.live_view = save; nav.reset(m, d, goal)
    obst_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obs{i}") for i in range(len(obstacles))}
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    groups = np.array([1, 1, 0, 0, 0, 0], np.uint8); gid = np.zeros(1, np.int32)
    collisions = void = 0; min_gap = 9.0; fromto = np.zeros(6)
    t_max = best / 0.25 + 30.0                           # physics walk ~0.38 m/s
    k = 0
    while d.time < t_max and walker.status != "fallen":
        yaw = nav.step(m, d)
        if yaw is None and nav.status == "goal reached":
            break
        walker.follow(m, d, yaw)                              # None = step in place
        if any((c.geom1 in obst_ids) or (c.geom2 in obst_ids) for c in d.contact[:d.ncon]):
            collisions += 1
        if k % 10 == 0:
            for side in ("left", "right"):                # is there floor under each foot?
                xy, z = P.sole(m, d, side)
                dist = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z + 0.05]), np.array([0, 0, -1.0]),
                                     groups, 1, -1, gid)
                if dist < 0 or dist > 0.15:
                    void += 1
        if k % 125 == 0:
            for g in robot_geoms:
                for o in obst_ids:
                    min_gap = min(min_gap, mujoco.mj_geomDistance(m, d, g, o, 2.0, fromto))
        k += 1
    reached = nav.status == "goal reached"
    fell = walker.status == "fallen"
    ok = reached and collisions == 0 and void == 0 and not fell
    if save:
        cv2.imwrite(f"nav_world_{seed}.png", nav.map_image(d.qpos[:2]))
    if nav.sensors.renderer is not None:
        nav.sensors.renderer.close()
    return dict(seed=seed, ok=ok, reached=reached, fell=fell, collisions=collisions, void_steps=void,
                min_gap_mm=min_gap * 1000, path_ratio=nav.distance / max(best - cfg.goal_tol, 0.1), obstacles=len(obstacles),
                pit=hole is not None, sim_s=d.time)


def main():
    arg = lambda k, dflt: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), dflt)
    n, seed0 = int(arg("n", 10)), int(arg("seed", 0))
    cam, save = "--nocam" not in sys.argv, "--save" in sys.argv
    print(f"Navigation tests: {n} random worlds, sensors = LiDAR{' + camera' if cam else ''}\n")
    print(f"{'world':>5} {'obst':>4} {'pit':>4} {'reached':>8} {'fell':>5} {'collide':>8} {'void':>5} {'gap_mm':>7} {'path/best':>9} {'time_s':>7}  result")
    res = []
    for s in range(seed0, seed0 + n):
        t0 = time.time()
        r = run_world(s, cam, save); res.append(r)
        print(f"{s:>5} {r['obstacles']:>4} {'yes' if r['pit'] else 'no':>4} {str(r['reached']):>8} {'YES' if r['fell'] else 'no':>5} {r['collisions']:>8} "
              f"{r['void_steps']:>5} {r['min_gap_mm']:>7.0f} {r['path_ratio']:>9.2f} {r['sim_s']:>7.1f}  "
              f"{'PASS' if r['ok'] else 'FAIL'}   ({time.time() - t0:.0f} s)", flush=True)
    passed = sum(r["ok"] for r in res)
    good = [r["path_ratio"] for r in res if r["ok"]]
    print(f"\nPASSED {passed}/{n}  ({100 * passed / n:.0f}%)"
          + (f"   median path / shortest = {np.median(good):.2f}" if good else ""))
    sys.exit(0 if passed == n else 1)


if __name__ == "__main__":
    main()
