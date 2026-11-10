"""
M0NARCH perception + navigation: LiDAR + depth camera -> occupancy map -> A* shortest path -> heading.

Nothing here knows the course. The robot only uses what its sensors see:

  1. SENSORS   3D LiDAR on the head (MuJoCo ray casts, 180 x 9 beams, 8 m)
               depth camera on the head (MuJoCo depth render, looking down/ahead)
  2. MAPPING   both sensors update a 2D occupancy grid (log-odds), 0.1 m cells:
                 points above the floor   -> obstacle
                 points on the floor      -> free
                 points below the floor   -> hole / drop-off (also an obstacle)
               Robot pose comes from the simulator (mapping with known pose).
               Full SLAM would also estimate the pose from the sensors - future work.
  3. PLANNING  A* on the map, obstacles grown by the robot's radius, unknown = free
               (optimistic), then line-of-sight shortcutting -> shortest safe path.
               Replans every second and whenever the path becomes blocked.
  4. FOLLOW    pure pursuit: heading toward a point 0.8 m ahead on the path,
               sent to gait_fix.walk_step(yaw_des=...).

Usage (inside the simulation loop, every physics step):
    nav = Navigator()
    nav.reset(m, d, goal_xy=(8.0, 0.0))
    yaw = nav.step(m, d)          # None = stop (goal reached or no path)
"""
import heapq
import math
import threading
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class NavConfig:
    # sensor mount (robot-specific: H1 head is part of torso_link)
    sensor_body: str = "torso_link"
    lidar_offset: tuple = (0.05, 0.0, 0.65)       # m, in torso frame (~1.63 m above floor)
    camera_offset: tuple = (0.12, 0.0, 0.60)
    camera_pitch_deg: float = -30.0                # looking down-forward
    camera_fovy_deg: float = 70.0
    camera_size: tuple = (160, 120)                # depth used for mapping
    display_size: tuple = (640, 480)               # sharper picture for the dashboard only (not used for mapping)
    
    use_camera: bool = True
    camera_max_range: float = 4.5                  # m, depth is used only up to here
    camera_depth_error: float = 0.02               # sensor spec: depth error ~2% of distance (typical)
    camera_weight: float = 0.5                     # camera evidence per frame (x l_occ): needs agreement
    use_lidar: bool = True
    # LiDAR
    lidar_h_beams: int = 180                       # 2 deg steps, full circle
    lidar_v_angles_deg: tuple = (-40, -32, -25, -18, -12, -7, -3, 0, 4)
    lidar_range: float = 8.0
    world_groups: tuple = (0, 1)                   # geom groups the sensors can see (robot = 2, 3)
    # classification
    obstacle_min_h: float = 0.08                   # m above floor -> obstacle
    hole_depth: float = 0.12                       # m below floor -> hole
    max_h: float = 2.0                             # ignore points above this (ceilings)
    # map
    res: float = 0.10
    size_m: float = 40.0
    l_occ: float = 0.9
    l_free: float = -0.35
    l_clamp: float = 4.0
    occ_thresh: float = 0.6
    # planning
    lethal_radius: float = 0.35                    # never closer than this (body + swinging arms)
    robot_radius: float = 0.45                     # preferred margin: allowed, but expensive inside it
    margin_cost: float = 8.0
    soft_radius: float = 0.9                       # extra cost inside this distance (keeps margin)
    soft_weight: float = 1.5
    replan_s: float = 1.0
    sensor_hz: float = 10.0
    # following
    lookahead: float = 0.8
    goal_tol: float = 0.4
    # realism: 0 = perfect simulator. evaluate.py raises these to estimate real-world readiness
    lidar_noise_m: float = 0.0        # range noise (std, m)
    lidar_dropout: float = 0.0        # fraction of beams that return nothing
    depth_noise_frac: float = 0.0     # camera depth noise (std, fraction of distance)
    odom_drift_m: float = 0.0         # odometry drift: position random walk (m per sqrt(s))
    odom_drift_deg: float = 0.0       # odometry drift: heading random walk (deg per sqrt(s))
    noise_seed: int = 0
    # SLAM localisation: correct the drifting pose by matching each LiDAR scan to the map
    scan_matching: bool = True
    match_xy_m: float = 0.06          # search window +- per scan (drift per scan is small)
    match_xy_step: float = 0.02
    match_deg: float = 1.0            # search window +- per scan (deg)
    match_deg_step: float = 0.25
    match_min_points: int = 40        # need this many solid-obstacle hits to match
    match_prior: float = 0.15         # penalty for moving away from odometry (fraction of score)
    map_range: float = 6.0            # only map hits closer than this (far hits smear the most)
    # recovery when no path is found (like Nav2's "clear costmap"): fade local map, look again
    recovery_radius_m: float = 4.0
    recovery_fade: float = 0.3
    max_recoveries: int = 6


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class OccupancyGrid:
    def __init__(self, cfg, center_xy):
        self.cfg = cfg
        self.n = int(cfg.size_m / cfg.res)
        self.origin = np.array(center_xy, float) - cfg.size_m / 2
        self.L = np.zeros((self.n, self.n), np.float32)        # log-odds, [ix, iy]

    def to_idx(self, xy):
        ij = np.floor((np.asarray(xy) - self.origin) / self.cfg.res).astype(int)
        return ij

    def to_xy(self, ij):
        return self.origin + (np.asarray(ij) + 0.5) * self.cfg.res

    def _add(self, xy, val):
        if len(xy) == 0:
            return
        ij = self.to_idx(xy)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        ij = np.unique(ij[ok], axis=0)                          # one vote per cell per scan
        if len(ij):
            np.add.at(self.L, (ij[:, 0], ij[:, 1]), val)

    def update(self, free_xy, occ_xy):
        self._add(free_xy, self.cfg.l_free)
        self._add(occ_xy, self.cfg.l_occ)
        np.clip(self.L, -self.cfg.l_clamp, self.cfg.l_clamp, out=self.L)

    def occupied(self):
        return self.L > self.cfg.occ_thresh

    def known_free(self):
        return self.L < -self.cfg.occ_thresh


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Sensors:
    """3D LiDAR + depth camera, both from the robot's head. World-frame point clouds out."""

    def __init__(self, m, cfg):
        self.cfg = cfg
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg.sensor_body)
        if self.body < 0:
            raise ValueError(f"sensor body '{cfg.sensor_body}' not found")
        self.groups = np.array([1 if g in cfg.world_groups else 0 for g in range(6)], np.uint8)
        az = np.linspace(0, 2 * np.pi, cfg.lidar_h_beams, endpoint=False)
        el = np.radians(cfg.lidar_v_angles_deg)
        A, E = np.meshgrid(az, el)
        self.dirs_local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.nray = len(self.dirs_local)
        self.geomid = np.zeros(self.nray, np.int32)
        self.dist = np.zeros(self.nray, np.float64)
        self.renderer = None
        self.last_rgb = None
        self.display_renderer = None

    def display_rgb(self, m, d):
        """Same view as camera(), rendered larger and smoother for people to watch. Call after camera()."""
        if self.display_renderer is None:
            W = min(self.cfg.display_size[0], m.vis.global_.offwidth)
            H = min(self.cfg.display_size[1], m.vis.global_.offheight)
            self.display_renderer = mujoco.Renderer(m, H, W)
        r = self.display_renderer
        r.update_scene(d, self.cam, scene_option=self.opt)
        return r.render().copy()

    def _yaw(self, d):
        R = d.xmat[self.body].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def mount(self, d, offset):
        """Sensor position: body position + offset rotated by the body's heading only (gimbal-stabilised)."""
        return d.xpos[self.body] + _rotz(self._yaw(d)) @ np.array(offset)

    def lidar(self, m, d):
        o = self.mount(d, self.cfg.lidar_offset)
        dirs = (_rotz(self._yaw(d)) @ self.dirs_local.T).T          # beams rotate with the robot's heading
        mujoco.mj_multiRay(m, d, o, dirs.ravel(), self.groups, 1, -1,
                           self.geomid, self.dist, None, self.nray, self.cfg.lidar_range)
        hit = (self.geomid >= 0) & (self.dist > 0)
        dist = np.where(hit, self.dist, np.inf)
        rng = getattr(self, "rng", None)
        if rng is not None and self.cfg.lidar_noise_m > 0:
            dist = dist + rng.normal(0, self.cfg.lidar_noise_m, dist.shape)
        if rng is not None and self.cfg.lidar_dropout > 0:
            dist = np.where(rng.random(dist.shape) < self.cfg.lidar_dropout, np.inf, dist)
        return o, dirs, dist

    def camera(self, m, d):
        """Depth render from the head camera. Returns (origin, points Nx3, rgb image)."""
        cfg = self.cfg
        W, H = cfg.camera_size
        if self.renderer is None:
            m.vis.global_.fovy = cfg.camera_fovy_deg   # free camera uses the model's fovy
            self.renderer = mujoco.Renderer(m, H, W)
            self.opt = mujoco.MjvOption()
            self.opt.geomgroup[2] = 0                   # hide the robot itself
            self.opt.geomgroup[3] = 0
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        yaw = self._yaw(d)
        pos = self.mount(d, cfg.camera_offset)
        pitch = math.radians(cfg.camera_pitch_deg)
        fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        dist = 1.0
        self.cam.lookat[:] = pos + fwd * dist
        self.cam.distance = dist
        self.cam.azimuth = math.degrees(yaw)
        self.cam.elevation = cfg.camera_pitch_deg
        r = self.renderer
        r.update_scene(d, self.cam, scene_option=self.opt)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, self.cam, scene_option=self.opt)
        depth = r.render().copy()
        r.disable_depth_rendering()
        self.last_rgb = rgb
        gl = r.scene.camera[0]                                     # exact projection used by the render
        c_pos, c_fwd, c_up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
        c_right = np.cross(c_fwd, c_up)
        tan_y = gl.frustum_top / gl.frustum_near
        tan_x = tan_y * W / H
        step = 2
        vs, us = np.mgrid[0:H:step, 0:W:step]
        z = depth[vs, us]
        rng = getattr(self, "rng", None)
        if rng is not None and self.cfg.depth_noise_frac > 0:
            z = z * (1 + rng.normal(0, self.cfg.depth_noise_frac, z.shape))
        ok = (z > 0.05) & (z < self.cfg.lidar_range)
        x_n = (2 * (us + 0.5) / W - 1) * tan_x
        y_n = (1 - 2 * (vs + 0.5) / H) * tan_y
        rays = c_fwd[None, None] + x_n[..., None] * c_right + y_n[..., None] * c_up
        pts = c_pos + z[..., None] * rays
        return c_pos, pts[ok], rgb


def _classify_lidar(cfg, o, dirs, dist):
    """Turn LiDAR returns into free and occupied xy points (floor is z = 0)."""
    free, occ = [], []
    hit = np.isfinite(dist)
    p = o + dirs * np.where(hit, dist, 0)[:, None]
    down = dirs[:, 2] < -1e-3
    t_floor = np.where(down, o[2] / np.maximum(-dirs[:, 2], 1e-6), np.inf)   # where the beam would meet the floor
    # 1) obstacle hits
    is_obs = hit & (p[:, 2] > cfg.obstacle_min_h) & (p[:, 2] < cfg.max_h)
    occ.append(p[is_obs, :2])
    # 2) floor hits -> free space up to the hit
    is_floor = hit & (np.abs(p[:, 2]) <= cfg.obstacle_min_h)
    # 3) hole: beam goes below the floor, or a downward beam that should have hit the floor hits nothing
    is_hole = hit & (p[:, 2] < -cfg.hole_depth)                       # beam went below the floor
    missing = ~hit & down & (t_floor < cfg.lidar_range)                # floor expected, nothing came back
    # random dropouts are isolated; a real drop-off makes neighbouring beams go missing too
    mg = missing.reshape(-1, cfg.lidar_h_beams)
    missing = (mg & np.roll(mg, 1, axis=1) & np.roll(mg, -1, axis=1)).ravel()
    edge = o + dirs * np.where(np.isfinite(t_floor), t_floor, 0)[:, None]
    occ.append(edge[is_hole, :2])
    occ.append(p[is_hole, :2])
    # free = only where a beam actually LANDED on the floor. Cells the beam passed over are NOT
    # marked free: a beam flying over a pit or a low box proves nothing about them.
    free = p[is_floor, :2]
    # a missing return is weaker evidence (real LiDARs drop returns on dark or shiny surfaces)
    # solid = real surfaces above the floor: the only points used for scan matching
    return free, np.concatenate(occ), edge[missing, :2], p[is_obs, :2]


def estimate_depth_error(pts, origin, prior):
    """Self-calibration: the floor is flat, so the spread of floor-ish points around z = 0
    tells how noisy the depth camera really is (as a fraction of distance)."""
    rel = pts - origin
    r = np.linalg.norm(rel, axis=1)
    k = np.abs(rel[:, 2]) * 1.0 / np.maximum(r, 1e-6) * r            # metres of height error per unit depth error
    sel = (np.abs(pts[:, 2]) < 0.25) & (k > 0.3)
    if sel.sum() < 200:
        return prior
    ratio = pts[sel, 2] / k[sel]
    mad = np.median(np.abs(ratio - np.median(ratio))) * 1.4826       # robust standard deviation
    return float(max(prior, mad))


def _classify_points(cfg, pts, origin, depth_err=None):
    """Depth-camera points -> free / occupied. The height threshold grows with the expected
    depth error at that distance, so far noisy floor points are not mistaken for obstacles."""
    rel = pts - origin
    r = np.linalg.norm(rel, axis=1)
    keep = r < cfg.camera_max_range
    pts, rel, r = pts[keep], rel[keep], r[keep]
    z = pts[:, 2]
    err = cfg.camera_depth_error if depth_err is None else depth_err
    sigma_z = err * r * np.abs(rel[:, 2]) / np.maximum(r, 1e-6)
    tol_up = np.maximum(cfg.obstacle_min_h, 3 * sigma_z)
    tol_dn = np.maximum(cfg.hole_depth, 3 * sigma_z)
    occ = pts[((z > tol_up) & (z < cfg.max_h)) | (z < -tol_dn), :2]
    free = pts[np.abs(z) <= tol_up, :2]
    return free, occ


class Planner:
    """A* on an inflated occupancy grid + line-of-sight shortcutting (shortest safe path)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def costmap(self, occupied):
        """lethal cells (never), margin cells (expensive), soft zone (slightly expensive)."""
        c = self.cfg
        dist = distance_transform_edt(~occupied) * c.res               # m to nearest obstacle
        lethal = dist < c.lethal_radius
        soft = np.clip((c.soft_radius - dist) / (c.soft_radius - c.robot_radius), 0, 1)
        cost = 1.0 + c.soft_weight * soft + c.margin_cost * (dist < c.robot_radius)
        return lethal, cost, dist

    def astar(self, blocked, cost, start, goal):
        n0, n1 = blocked.shape
        if not (0 <= goal[0] < n0 and 0 <= goal[1] < n1):
            return None
        sx, sy = start
        if blocked[sx, sy]:                     # robot already too close: open a small escape disk
            blocked = blocked.copy()
            r = 3
            blocked[max(sx - r, 0):sx + r + 1, max(sy - r, 0):sy + r + 1] = False
        if blocked[goal]:
            return None
        nbr = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        h = lambda a: math.hypot(a[0] - goal[0], a[1] - goal[1])
        g = {start: 0.0}; came = {}; openq = [(h(start), start)]; closed = set()
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                return path[::-1]
            if cur in closed:
                continue
            closed.add(cur)
            cx, cy = cur
            for dx, dy, w in nbr:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < n0 and 0 <= ny < n1) or blocked[nx, ny]:
                    continue
                if dx and dy and (blocked[cx + dx, cy] or blocked[cx, cy + dy]):
                    continue                    # no corner cutting
                ng = g[cur] + w * cost[nx, ny]
                if ng < g.get((nx, ny), 1e18):
                    g[(nx, ny)] = ng; came[(nx, ny)] = cur
                    heapq.heappush(openq, (ng + h((nx, ny)), (nx, ny)))
        return None

    @staticmethod
    def line_free(blocked, a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        xs = np.round(np.linspace(a[0], b[0], n)).astype(int)
        ys = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return not blocked[xs, ys].any()

    def shortcut(self, blocked, path):
        if not path:
            return path
        out = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.line_free(blocked, path[i], path[j]):
                j -= 1
            out.append(path[j]); i = j
        return out


class Navigator:
    def __init__(self, cfg: NavConfig = None):
        self.cfg = cfg or NavConfig()
        self.sensors = None
        self.grid = None
        self.planner = Planner(self.cfg)
        self.path_xy = None
        self.goal = None
        self.status = "idle"
        self._lock = threading.Lock()
        self._cam_jpg = None
        self._map_jpg = None
        self._last_hits = np.zeros((0, 2))
        self._model_id = None
        self.live_view = True                       # set False in headless tests (faster)
        self.rng = np.random.default_rng(self.cfg.noise_seed)
        self._err = np.zeros(3)                     # localisation error: dx, dy, dyaw

    # -------------------------------------------------------------- control
    def reset(self, m, d, goal_xy):
        if self.sensors is None or self._model_id != id(m):
            self.sensors = Sensors(m, self.cfg)
            self._model_id = id(m)
        self.sensors.rng = self.rng
        self._err[:] = 0.0
        mujoco.mj_forward(m, d)
        self.grid = OccupancyGrid(self.cfg, d.qpos[:2].copy())
        self.goal = np.array(goal_xy, float)
        self.path_xy = None
        self.status = "starting"
        self._next_sense = d.time
        self._next_plan = d.time
        self.distance = 0.0
        self.recoveries = 0
        self._last_xy = d.qpos[:2].copy()
        self._yaw_cmd = 0.0

    def set_goal(self, goal_xy):
        self.goal = np.array(goal_xy, float)
        self.path_xy = None
        if self.status in ("goal reached", "no path"):
            self.status = "replanning"
        self.recoveries = 0

    def step(self, m, d):
        """Call every physics step. Returns heading (rad) or None to stop."""
        if self.grid is None or self.goal is None:
            return None
        true_xy = d.qpos[:2].copy()
        self.distance += float(np.linalg.norm(true_xy - self._last_xy)); self._last_xy = true_xy
        xy = true_xy + self._err[:2]                # where the robot THINKS it is
        if np.linalg.norm(self.goal - xy) < self.cfg.goal_tol:
            if self.status != "goal reached":
                self.status = "goal reached"
                self._render_map(xy)
            return None
        if d.time >= self._next_sense:
            self._next_sense = d.time + 1.0 / self.cfg.sensor_hz
            self._drift(1.0 / self.cfg.sensor_hz)
            xy = true_xy + self._err[:2]
            self.sense(m, d)
            if self.path_xy is not None and self._path_blocked():
                self._next_plan = d.time            # new obstacle on our path: replan now
            if d.time >= self._next_plan:
                self._next_plan = d.time + self.cfg.replan_s
                self.plan(xy)
            if self.live_view:
                self._render_map(xy)                # live map image for the UI (10 Hz)
        elif d.time >= self._next_plan:
            self._next_plan = d.time + self.cfg.replan_s
            self.plan(xy)
        if self.path_xy is None:
            return None
        self._yaw_cmd = self._pursue(xy)
        return self._yaw_cmd - self._err[2]          # heading error: the robot steers by its belief

    # -------------------------------------------------------------- pipeline
    def _drift(self, dt):
        """Odometry drift: without correction the pose error grows like a random walk."""
        c = self.cfg
        if c.odom_drift_m <= 0 and c.odom_drift_deg <= 0:
            return
        sig = np.array([c.odom_drift_m, c.odom_drift_m, math.radians(c.odom_drift_deg)])
        self._err += sig * math.sqrt(dt) * self.rng.standard_normal(3)

    def _scan_match(self, occ_b, d):
        """Correlative scan matching (the localisation half of SLAM).
        Try small shifts/rotations of the current scan; keep the one that lands the LiDAR
        hits on cells the map already knows are obstacles. Corrects the pose belief."""
        c = self.cfg
        occ_map = self.grid.occupied()
        if len(occ_b) < c.match_min_points or occ_map.sum() < 80:
            return
        if len(occ_b) > 400:
            occ_b = occ_b[self.rng.choice(len(occ_b), 400, replace=False)]
        field = np.exp(-(distance_transform_edt(~occ_map) * c.res) ** 2 / (2 * 0.08 ** 2))
        xy_b = d.qpos[:2] + self._err[:2]
        rel = occ_b - xy_b
        ths = np.radians(np.arange(-c.match_deg, c.match_deg + 1e-9, c.match_deg_step))
        ts = np.arange(-c.match_xy_m, c.match_xy_m + 1e-9, c.match_xy_step)
        best, best_s, base = (0.0, 0.0, 0.0), -1.0, None
        for th in ths:
            ct, st = math.cos(th), math.sin(th)
            rot = np.stack([rel[:, 0] * ct - rel[:, 1] * st, rel[:, 0] * st + rel[:, 1] * ct], 1) + xy_b
            for dx in ts:
                for dy in ts:
                    ij = np.floor((rot + (dx, dy) - self.grid.origin) / c.res).astype(int)
                    ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.grid.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.grid.n)
                    sc = field[ij[ok, 0], ij[ok, 1]].sum()
                    # prefer staying near the odometry estimate unless the scan clearly says otherwise
                    sc -= c.match_prior * len(rot) * ((dx / c.match_xy_m) ** 2 + (dy / c.match_xy_m) ** 2
                                                      + (th / max(math.radians(c.match_deg), 1e-9)) ** 2) / 3
                    if abs(th) < 1e-9 and abs(dx) < 1e-9 and abs(dy) < 1e-9:
                        base = sc
                    if sc > best_s:
                        best_s, best = sc, (dx, dy, th)
        if base is not None and best_s > base * 1.02:          # only correct on a clear improvement
            self._err[0] += best[0]; self._err[1] += best[1]; self._err[2] += best[2]

    def _to_belief(self, pts, d):
        """Sensor points are measured relative to the robot; with a pose error they land
        in the map shifted and rotated by that error."""
        if len(pts) == 0 or not self._err.any():
            return pts
        c0 = d.qpos[:2]
        R = np.array([[math.cos(self._err[2]), -math.sin(self._err[2])],
                      [math.sin(self._err[2]), math.cos(self._err[2])]])
        return (pts - c0) @ R.T + c0 + self._err[:2]

    def sense(self, m, d):
        if self.cfg.use_lidar:
            o, dirs, dist = self.sensors.lidar(m, d)
            free, occ, weak, solid = _classify_lidar(self.cfg, o, dirs, dist)
            near = lambda q: q[np.linalg.norm(q - d.qpos[:2], axis=1) < self.cfg.map_range]
            free, occ, weak, solid = near(free), near(occ), near(weak), near(solid)
            if self.cfg.scan_matching and self._err.any():
                self._scan_match(self._to_belief(solid, d), d)
            free, occ, weak = (self._to_belief(x, d) for x in (free, occ, weak))
            self.grid.update(free, occ)
            self.grid._add(weak, 0.5 * self.cfg.l_occ)
            np.clip(self.grid.L, -self.cfg.l_clamp, self.cfg.l_clamp, out=self.grid.L)
            self._last_hits = occ
        if self.cfg.use_camera:
            c_pos, pts, rgb = self.sensors.camera(m, d)
            est = estimate_depth_error(pts, c_pos, self.cfg.camera_depth_error)
            self.depth_err = 0.8 * getattr(self, "depth_err", est) + 0.2 * est      # smoothed estimate
            cf, co = _classify_points(self.cfg, pts, c_pos, self.depth_err)
            w = self.cfg.camera_weight
            self.grid._add(self._to_belief(cf, d), w * self.cfg.l_free)
            self.grid._add(self._to_belief(co, d), w * self.cfg.l_occ)
            np.clip(self.grid.L, -self.cfg.l_clamp, self.cfg.l_clamp, out=self.grid.L)
            if self.live_view:                         # dashboard picture: sharper render, same view
                rgb = self.sensors.display_rgb(m, d)
            self._store_cam(rgb)

    def plan(self, xy):
        blocked, cost, dist = self.planner.costmap(self.grid.occupied())
        s = tuple(np.clip(self.grid.to_idx(xy), 0, self.grid.n - 1))
        g = tuple(self.grid.to_idx(self.goal))
        path = self.planner.astar(blocked, cost, s, g)
        if path is None:
            self.path_xy = None
            if self.recoveries < self.cfg.max_recoveries:
                # recovery: weaken local evidence (noise and smear) and let the sensors rebuild it
                r = int(self.cfg.recovery_radius_m / self.cfg.res)
                i, j = s
                sl = (slice(max(i - r, 0), i + r + 1), slice(max(j - r, 0), j + r + 1))
                self.grid.L[sl] *= self.cfg.recovery_fade
                self.recoveries += 1
                self.status = f"recovering ({self.recoveries})"
            else:
                self.status = "no path"
            return
        margin = dist < self.cfg.robot_radius
        if margin[s]:
            margin = margin.copy(); margin[s] = False
        path = self.planner.shortcut(margin, path)       # straighten only where the margin is kept
        self.path_xy = self.grid.to_xy(np.array(path))
        self.path_xy[0] = xy
        self.path_xy[-1] = self.goal
        self._blocked = dist < self.cfg.robot_radius      # shown pink on the map
        self.status = "navigating"

    def _path_blocked(self):
        occ = self.grid.occupied()
        dist = distance_transform_edt(~occ) * self.cfg.res
        pts = []
        for a, b in zip(self.path_xy[:-1], self.path_xy[1:]):
            n = int(np.linalg.norm(b - a) / self.cfg.res) + 2
            pts.append(np.linspace(a, b, n))
        ij = np.clip(self.grid.to_idx(np.concatenate(pts)), 0, self.grid.n - 1)
        far = np.linalg.norm(np.concatenate(pts) - self.path_xy[0], axis=1) > 0.5   # ignore the robot's own spot
        return bool((dist[ij[:, 0], ij[:, 1]][far] < self.cfg.lethal_radius).any())

    def _pursue(self, xy):
        P = self.path_xy
        seg_d = [np.linalg.norm(P[i + 1] - P[i]) for i in range(len(P) - 1)]
        # closest point on the polyline
        best, best_s, acc = 1e9, 0.0, 0.0
        for i, L in enumerate(seg_d):
            a, b = P[i], P[i + 1]
            t = 0 if L < 1e-9 else np.clip(np.dot(xy - a, b - a) / L ** 2, 0, 1)
            dd = np.linalg.norm(a + t * (b - a) - xy)
            if dd < best:
                best, best_s = dd, acc + t * L
            acc += L
        target_s = min(best_s + self.cfg.lookahead, acc)
        acc = 0.0
        tgt = P[-1]
        for i, L in enumerate(seg_d):
            if acc + L >= target_s:
                t = (target_s - acc) / max(L, 1e-9)
                tgt = P[i] + t * (P[i + 1] - P[i]); break
            acc += L
        return math.atan2(tgt[1] - xy[1], tgt[0] - xy[0])

    # -------------------------------------------------------------- images for the UI
    def _store_cam(self, rgb):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if img.shape[1] < 320:                         # small perception image: enlarge for display
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            with self._lock:
                self._cam_jpg = jpg.tobytes()

    def map_image(self, xy, px=480, view_m=16.0):
        g = self.grid
        img = np.full((g.n, g.n, 3), 140, np.uint8)               # unknown = grey
        img[g.known_free()] = (235, 235, 235)                     # free = white
        if hasattr(self, "_blocked"):
            img[self._blocked & ~g.occupied()] = (200, 205, 245)  # grown safety zone = light red (BGR)
        img[g.occupied()] = (30, 30, 30)                          # obstacle = black
        # crop a window around the robot and the goal
        c = (xy + self.goal) / 2
        half = max(view_m / 2, np.abs(self.goal - xy).max() / 2 + 2.0)
        lo = np.clip(g.to_idx(c - half), 0, g.n - 1); hi = np.clip(g.to_idx(c + half), 1, g.n)
        crop = img[lo[0]:hi[0], lo[1]:hi[1]]
        crop = np.ascontiguousarray(np.flipud(np.transpose(crop, (1, 0, 2))))   # x right, y up
        scale = px / crop.shape[0]
        out = cv2.resize(crop, (int(crop.shape[1] * scale), px), interpolation=cv2.INTER_NEAREST)
        def to_px(p):
            ij = (np.asarray(p) - g.origin) / g.cfg.res - lo
            return int(ij[0] * scale), int(px - ij[1] * scale)
        for h in self._last_hits[::3]:
            cv2.circle(out, to_px(h), 1, (0, 140, 255), -1)       # latest LiDAR obstacle hits = orange
        if self.path_xy is not None:
            pts = np.array([to_px(p) for p in self.path_xy], np.int32)
            cv2.polylines(out, [pts], False, (230, 120, 0), 2)    # planned path = blue
        cv2.circle(out, to_px(self.goal), 7, (0, 0, 220), 2)      # goal = red ring
        rp = to_px(xy)
        cv2.circle(out, rp, 6, (0, 170, 0), -1)                   # robot = green
        tip = (int(rp[0] + 14 * math.cos(self._yaw_cmd)), int(rp[1] - 14 * math.sin(self._yaw_cmd)))
        cv2.line(out, rp, tip, (0, 120, 0), 2)
        cv2.putText(out, self.status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    def _render_map(self, xy):
        ok, jpg = cv2.imencode(".jpg", self.map_image(xy))
        if ok:
            with self._lock:
                self._map_jpg = jpg.tobytes()

    def camera_jpeg(self):
        with self._lock:
            return self._cam_jpg

    def map_jpeg(self):
        with self._lock:
            return self._map_jpg
