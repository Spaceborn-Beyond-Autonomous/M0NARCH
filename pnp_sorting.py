"""
M0NARCH Pick & Place = colour sorting with the H1, under real physics.

Three coloured blocks (red, green, blue) lie on a table; three open bins of the same colours
stand next to them. The robot looks with its head camera, finds every block and bin with OpenCV,
and puts each block into the bin of its colour. Positions and the colour-to-bin assignment are
random on every run, so nothing about the layout is known in advance.

How it works
  Scene      prepare_sorting_scene(spec): table at counter height (0.94 m), 3 blocks (free bodies,
             gravity, friction), 3 open bins, sole-box feet (as in physics walking), head camera.
  Body       the Stand balance controller (stand_fix.py) keeps the robot upright on its motors
             while the right arm works; mj_step only, no positions written while it runs.
  Arm        resolved-rate IK (damped least squares on the gripper's Jacobian) -> joint targets
             -> PD torque + gravity compensation on the 4 right-arm motors. The target moves at
             most 0.25 m/s toward each waypoint.
  Gripper    Robotiq 2F-85 motor closes on the block. When the fingers have closed with the block
             between them (the grasp check), a hold force keeps it at the fingertips while
             carrying, standing in for fingertip friction that a 4-joint arm cannot aim reliably.
             The block is released over the bin and falls in under gravity.
  Vision     pnp_vision.py: head camera colour + depth -> HSV blobs -> 3D points -> each blob is a
             BLOCK (about 5 cm wide) or a BIN (about 13 cm wide) by its measured size.
  Task       look -> choose the nearest unsorted block -> above -> descend -> close -> grasp check
             (retry on failure) -> lift -> carry over its bin -> lower -> release -> look again,
             until the camera sees every block inside its bin.
"""
import math
import threading

import cv2
import mujoco
import numpy as np

from physics_walk import add_soles, sole
from pnp_vision import COLOUR_RANGES, EyeCamera, add_eye_camera, detect_blobs, world_image
from stand_fix import StandBalancer

# ------------------------------------------------------------------ scene
TABLE_TOP_Z = 0.94                    # m, counter height for a 1.8 m humanoid
TABLE_HALF_T = 0.02
BLOCK_HALF = 0.025                    # 5 cm cubes
BLOCK_MASS = 0.08                     # kg
BIN_HALF = 0.055                      # outer half width of a bin (11 cm)
BIN_WALL = 0.005
BIN_WALL_HALF_H = 0.02                # 4 cm walls
COLOURS = {"red": (0.85, 0.10, 0.10, 1), "green": (0.10, 0.70, 0.15, 1), "blue": (0.10, 0.25, 0.90, 1)}
BIN_SLOTS = ((0.31, 0.09), (0.45, 0.09), (0.38, 0.205))   # where bins may stand (left side, within reach)
BLOCK_AREA = ((0.34, 0.52), (-0.25, -0.07))               # where blocks may lie (right side, within reach)
SLOT_JITTER = 0.01
MIN_BLOCK_GAP = 0.08

# ------------------------------------------------------------------ control
ARM_JOINTS = ("right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow")
ARM_KP = np.array([90.0, 90.0, 40.0, 45.0])
ARM_KD = np.array([4.0, 4.0, 2.0, 2.0])
IK_DAMPING = 0.05
V_MAX = 0.25                          # m/s, gripper speed limit
REACHED = 0.015                       # m
STEP_TIMEOUT = 5.0                    # s per motion step
HOLD_K, HOLD_D = 400.0, 6.0           # hold force at the fingertips (N/m, N s/m)
GRASP_CHECK = 0.035                   # m, block centre must be this close to the fingertips
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0
UPDATE_HZ = 10.0                      # camera rate
DROP_Z = TABLE_TOP_Z + 0.16           # m, release height over a bin (the block falls ~9 cm into it)
LOOK_POSE = np.array([0.25, -0.30, 1.10])   # gripper out of the camera's view while looking
MAX_TRIES = 2                         # grasp attempts per block before skipping it

# ------------------------------------------------------------------ vision classification
BLOCK_MAX_SIZE = 0.085                # m, a blob narrower than this is a block
BIN_MIN_SIZE = 0.09                   # m, wider than this is a bin
IN_BIN = 0.045                        # m, block centre this close to its bin centre = sorted
# A block inside its own (same colour) bin merges with it into one blob. It is found by height:
# bin floor and walls are low/at the edge, so tall points well inside the walls = a block in the bin.
IN_BIN_HEIGHT = 0.035                 # m above the table top (bin walls end at 0.046, a block in a bin at 0.056)
IN_BIN_MIN_PX = 30
LOOK_SETTLE = 0.3                     # s the arm rests out of view before the camera decides


def _set_table(spec):
    table = spec.body("table")
    if table is None:
        return
    table.pos = [table.pos[0], table.pos[1], TABLE_TOP_Z - TABLE_HALF_T]
    leg_half = (TABLE_TOP_Z - 2 * TABLE_HALF_T) / 2
    for g in table.geoms:
        if abs(g.pos[2]) > 1e-6:                                       # legs reach the floor
            g.size = [g.size[0], g.size[1], leg_half]
            g.pos = [g.pos[0], g.pos[1], -TABLE_HALF_T - leg_half]


def _add_block(spec, body, colour):
    if body is None:
        body = spec.body(f"block_{colour}")              # already added (scene prepared twice)
    if body is None:
        body = spec.worldbody.add_body(name=f"block_{colour}", pos=[0.4, -0.15, TABLE_TOP_Z + BLOCK_HALF])
        body.add_freejoint(name=f"block_{colour}_joint")
        body.add_geom(name=f"block_{colour}_geom", type=mujoco.mjtGeom.mjGEOM_BOX)
    g = body.geoms[0]
    g.size = [BLOCK_HALF] * 3
    g.rgba = list(COLOURS[colour])
    g.mass = BLOCK_MASS
    g.friction = [1.0, 0.01, 0.001]


def _add_bin(spec, colour):
    if spec.body(f"bin_{colour}") is not None:            # already added
        return
    b = spec.worldbody.add_body(name=f"bin_{colour}", pos=[0.4, 0.15, TABLE_TOP_Z])
    rgba = list(COLOURS[colour])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF, BIN_HALF, 0.003], pos=[0, 0, 0.003], rgba=rgba)
    w, h = BIN_WALL, BIN_WALL_HALF_H
    for (px, py, sx, sy) in ((BIN_HALF - w, 0, w, BIN_HALF), (-BIN_HALF + w, 0, w, BIN_HALF),
                             (0, BIN_HALF - w, BIN_HALF, w), (0, -BIN_HALF + w, BIN_HALF, w)):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, h], pos=[px, py, 0.006 + h], rgba=rgba)


def prepare_sorting_scene(spec):
    """Build the sorting scene into the Pick & Place model spec (call before spec.compile()). Safe to call twice."""
    _set_table(spec)
    _add_block(spec, spec.body("object"), "red")          # the scene's original block becomes the red one
    _add_block(spec, None, "green")
    _add_block(spec, None, "blue")
    for c in COLOURS:
        _add_bin(spec, c)
    for site in spec.sites:
        if site.name == "target_site":                     # old place marker: not used
            site.rgba = [0, 0, 0, 0]
    if not any(g.name == "left_sole" for g in spec.geoms):
        add_soles(spec)                                    # flat feet, as in physics walking
    add_eye_camera(spec)


class SortingTask:
    def __init__(self, m, seed=None):
        self.rng = np.random.default_rng(seed)
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_v = np.array([m.jnt_dofadr[j] for j in jid])
        self.arm_rng = np.array([m.jnt_range[j] for j in jid])
        self.arm_act = np.array([next(a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                                      and m.actuator_trnid[a, 0] == j) for j in jid])
        self.grip_act = next((a for a in range(m.nu) if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT), None)
        self.block_body = {"red": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")}
        for c in ("green", "blue"):
            self.block_body[c] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"block_{c}")
        self.block_jnt = {c: m.body_jntadr[b] for c, b in self.block_body.items()}
        self.bin_body = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bin_{c}") for c in COLOURS}
        self.camera = None
        self.lock = threading.Lock()
        self._jpg = b""
        self.stand = None
        self.state = "IDLE"

    # ------------------------------------------------------------------ new run
    def reset(self, m, d):
        """Random layout, robot standing on its feet, task from the start."""
        mujoco.mj_resetData(m, d)
        self.stand = StandBalancer(m, d)
        d.qpos[self.stand.qadr] = self.stand.q_ref
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        lowest = min(sole(m, d, "left")[1], sole(m, d, "right")[1])
        d.qpos[2] -= lowest - 0.0005                       # feet just touching the floor
        # random bins: shuffled colours on the bin slots, small offsets
        colours = list(COLOURS)
        self.rng.shuffle(colours)
        for c, (x, y) in zip(colours, BIN_SLOTS):
            m.body_pos[self.bin_body[c]] = [x + self.rng.uniform(-SLOT_JITTER, SLOT_JITTER),
                                            y + self.rng.uniform(-SLOT_JITTER, SLOT_JITTER), TABLE_TOP_Z]
        # random blocks: anywhere in the block area, apart from each other, random turn
        placed = []
        for c in COLOURS:
            for _ in range(200):
                p = np.array([self.rng.uniform(*BLOCK_AREA[0]), self.rng.uniform(*BLOCK_AREA[1])])
                if all(np.linalg.norm(p - q) > MIN_BLOCK_GAP for q in placed):
                    break
            placed.append(p)
            a, yaw = self.block_jnt[c], self.rng.uniform(0, math.pi / 2)
            qa = m.jnt_qposadr[a]
            d.qpos[qa:qa + 3] = [p[0], p[1], TABLE_TOP_Z + BLOCK_HALF + 0.001]
            d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.stand._t_last = None
        self.p_cmd = d.site_xpos[self.site].copy()
        self.home = self.p_cmd.copy()
        self.blocks, self.bins = {}, {}
        self.held = None
        self.target_colour = None
        self.tries = {c: 0 for c in COLOURS}
        self.skipped = set()
        self.sorted_count = 0
        self._t_cam = -1.0
        self._go("LOOK", d)

    def _go(self, state, d):
        self.state, self.t_state = state, d.time
        self._look_since = None

    # ------------------------------------------------------------------ one physics step
    def step(self, m, d):
        if self.stand is None:
            self.reset(m, d)
        if d.time - self._t_cam >= 1.0 / UPDATE_HZ or d.time < self._t_cam:
            self._t_cam = d.time
            self._perceive(m, d)
        target, grip = self._task(m, d)
        self.stand.apply_stand_control(m, d)               # legs, torso, left arm: balance
        self._arm(m, d, target)                            # right arm: reach
        if self.grip_act is not None:
            d.ctrl[self.grip_act] = grip
        self._hold(m, d)
        mujoco.mj_step(m, d)

    def _arm(self, m, d, target):
        dt = m.opt.timestep
        step = target - self.p_cmd
        n = np.linalg.norm(step)
        if n > V_MAX * dt:
            step *= V_MAX * dt / n
        self.p_cmd += step
        J = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, J, None, self.site)
        J = J[:, self.arm_v]
        err = self.p_cmd - d.site_xpos[self.site]
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(3), err)
        q, qd = d.qpos[self.arm_q], d.qvel[self.arm_v]
        q_ref = np.clip(q + dq, self.arm_rng[:, 0], self.arm_rng[:, 1])
        tau = ARM_KP * (q_ref - q) - ARM_KD * qd + d.qfrc_bias[self.arm_v]   # PD + gravity compensation
        lo, hi = m.actuator_ctrlrange[self.arm_act].T
        d.ctrl[self.arm_act] = np.clip(tau, lo, hi)

    def _hold(self, m, d):
        d.xfrc_applied[list(self.block_body.values())] = 0
        if self.held is None:
            return
        b = self.block_body[self.held]
        a = m.jnt_dofadr[self.block_jnt[self.held]]
        f = HOLD_K * (d.site_xpos[self.site] - d.xpos[b]) - HOLD_D * d.qvel[a:a + 3]
        f[2] += m.body_subtreemass[b] * 9.81
        d.xfrc_applied[b, :3] = f
        d.xfrc_applied[b, 3:] = -0.01 * d.qvel[a + 3:a + 6]

    def _true_block(self, d, c):
        return d.xpos[self.block_body[c]]

    # ------------------------------------------------------------------ task logic
    def _task(self, m, d):
        t = d.time - self.t_state
        s = self.state
        up = np.array([0, 0, 1.0])
        if s == "LOOK":
            at_pose = np.linalg.norm(d.site_xpos[self.site] - LOOK_POSE) < 0.05 or t > STEP_TIMEOUT
            if at_pose and self._look_since is None:
                self._look_since = d.time
            if not at_pose or d.time - self._look_since < LOOK_SETTLE:   # wait for fresh images
                return LOOK_POSE, GRIPPER_OPEN
            todo = [c for c in COLOURS if c in self.blocks and c in self.bins and c not in self.skipped
                    and np.linalg.norm(self.blocks[c][:2] - self.bins[c][:2]) > IN_BIN]
            if not todo:
                self._go("DONE", d)
                return LOOK_POSE, GRIPPER_OPEN
            here = d.site_xpos[self.site]
            self.target_colour = min(todo, key=lambda c: np.linalg.norm(self.blocks[c] - here))
            self.pick = self.blocks[self.target_colour].copy()
            self.place = self.bins[self.target_colour].copy()
            self._go("ABOVE", d)
        if s == "DONE":
            return LOOK_POSE, GRIPPER_OPEN
        pick, place = self.pick, self.place
        plan = {
            "ABOVE": (pick + 0.12 * up, GRIPPER_OPEN, "DESCEND"),
            "DESCEND": (pick, GRIPPER_OPEN, "CLOSE"),
            "LIFT": (pick + 0.15 * up, GRIPPER_CLOSED, "CARRY"),
            "CARRY": (np.array([place[0], place[1], TABLE_TOP_Z + 0.17]), GRIPPER_CLOSED, "LOWER"),
            "LOWER": (np.array([place[0], place[1], DROP_Z]), GRIPPER_CLOSED, "RELEASE"),
            "RETREAT": (np.array([place[0], place[1], TABLE_TOP_Z + 0.22]), GRIPPER_OPEN, "LOOK"),
            "BACK_OFF": (pick + 0.15 * up, GRIPPER_OPEN, "LOOK"),
        }
        if s in plan:
            target, grip, nxt = plan[s]
            if np.linalg.norm(d.site_xpos[self.site] - target) < REACHED or t > STEP_TIMEOUT:
                self._go(nxt, d)
            return target, grip
        if s == "CLOSE":
            if t > 0.6:                                    # fingers closed: is the block between them?
                c = self.target_colour
                if np.linalg.norm(self._true_block(d, c) - d.site_xpos[self.site]) < GRASP_CHECK:
                    self.held = c
                    self._go("LIFT", d)
                else:
                    self.tries[c] += 1
                    if self.tries[c] >= MAX_TRIES:
                        self.skipped.add(c)
                    self._go("BACK_OFF", d)
            return pick, GRIPPER_CLOSED
        if s == "RELEASE":
            self.held = None
            if t > 0.6:
                self._go("RETREAT", d)
            return np.array([place[0], place[1], DROP_Z]), GRIPPER_OPEN
        return self.home, GRIPPER_OPEN

    # ------------------------------------------------------------------ vision
    def _perceive(self, m, d):
        if self.camera is None:
            self.camera = EyeCamera(m)
        cam = self.camera
        bgr, depth = cam.capture(d)
        P = world_image(m, d, cam.cam, depth)                     # 3D world point of every pixel
        with np.errstate(invalid="ignore"):
            keep = np.where(P[..., 2] > TABLE_TOP_Z - 0.01, 255, 0).astype(np.uint8)   # table level, not floor
        drawn = []
        for colour, ranges in COLOUR_RANGES.items():
            got_bin = got_block = False
            in_bin = None
            for blob in detect_blobs(bgr, ranges, keep)[:4]:
                pts = P[blob["mask"] > 0]
                pts = pts[~np.isnan(pts[:, 0])]
                if len(pts) < 10:
                    continue
                lo, hi = np.percentile(pts[:, :2], 3, 0), np.percentile(pts[:, :2], 97, 0)
                size = float(np.max(hi - lo))
                if size >= BIN_MIN_SIZE and not got_bin:
                    got_bin = True
                    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, TABLE_TOP_Z])
                    self.bins[colour] = centre
                    drawn.append(("bin", colour, blob))
                    inside = np.all(np.abs(pts[:, :2] - centre[:2]) < BIN_HALF - 0.02, 1) & \
                        (pts[:, 2] > TABLE_TOP_Z + IN_BIN_HEIGHT)
                    in_bin = np.r_[np.median(pts[inside][:, :2], 0), TABLE_TOP_Z + BLOCK_HALF] \
                        if inside.sum() >= IN_BIN_MIN_PX else None
                elif size <= BLOCK_MAX_SIZE and not got_block:
                    got_block = True
                    surface = np.median(pts, 0)
                    ray = surface - d.cam_xpos[cam.cam]
                    self.blocks[colour] = surface + ray / np.linalg.norm(ray) * BLOCK_HALF
                    drawn.append(("block", colour, blob))
            if in_bin is not None and not got_block:     # no loose block of this colour: it is in its bin
                self.blocks[colour] = in_bin
        self.sorted_count = sum(1 for c in COLOURS if c in self.blocks and c in self.bins
                                and np.linalg.norm(self.blocks[c][:2] - self.bins[c][:2]) <= IN_BIN)
        self._draw(bgr, drawn)

    @staticmethod
    def bins_seen_now(drawn):
        return {c for k, c, _ in drawn if k == "bin"}

    @staticmethod
    def blocks_seen_now(drawn):
        return {c for k, c, _ in drawn if k == "block"}

    def _draw(self, bgr, drawn):
        H, W = bgr.shape[:2]
        ink = {"red": (60, 60, 255), "green": (60, 220, 60), "blue": (255, 160, 60)}
        for kind, colour, blob in drawn:
            thick = 2 if kind == "block" else 1
            cv2.drawContours(bgr, [blob["box"]], 0, (255, 255, 255) if kind == "block" else ink[colour], thick + 1)
            x0, y0 = blob["box"][:, 0].min(), max(blob["box"][:, 1].min() - 6, 14)
            cv2.putText(bgr, f"{colour} {kind}", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, f"{colour} {kind}", (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ink[colour], 1, cv2.LINE_AA)
        doing = f"{self.target_colour} block -> {self.target_colour} bin" if self.target_colour and \
            self.state not in ("LOOK", "DONE") else ("all blocks sorted" if self.state == "DONE" else "looking")
        cv2.rectangle(bgr, (0, H - 46), (W, H), (0, 0, 0), -1)
        cv2.putText(bgr, f"sorted {self.sorted_count}/{len(COLOURS)}   |   {doing}", (8, H - 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"{self.state}  |  targets from camera (OpenCV + depth)", (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.lock:
                self._jpg = jpg.tobytes()

    def jpeg(self):
        with self.lock:
            return self._jpg
