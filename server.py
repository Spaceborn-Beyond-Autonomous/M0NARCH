import os
import sys
import time
import math
import threading

import numpy as np
import cv2

# Force headless rendering to prevent X11/Wayland display errors
os.environ["MUJOCO_GL"] = "egl"

import mujoco
from gait_fix import walk_step
from physics_walk import PhysicsWalker, load_model as _load_physics_model

# E9 fix: real physics walking (physics_walk.py). Falls back to the kinematic gait if the
# Unitree policy file is missing.
try:
    walker = PhysicsWalker()
    load_h1 = _load_physics_model
    print("[WALK] physics walking ON (Unitree H1 policy + sole-box feet)")
except FileNotFoundError as _e:
    walker = None
    load_h1 = mujoco.MjModel.from_xml_path
    print(f"[WALK] physics walking OFF, using kinematic gait. {_e}")
WALK_CMD = (0.5, 0.0, 0.0)          # Walk mode: forward m/s, sideways m/s, turn rad/s
from stand_fix import StandBalancer
from perception_nav import Navigator, NavConfig

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ==============================================================================
# IMPORT LOCAL MODULES
# ==============================================================================

PHYSICS_DIR = os.path.join(
    os.path.dirname(__file__),
    "H1_robot-Physics-simulation",
    "simulation",
    "physics"
)

sys.path.append(PHYSICS_DIR)

import h1_balance_v16_1_validated as h1_balancer


RL_DIR = os.path.join(
    os.path.dirname(__file__),
    "H1_robot-RL-Perception",
    "simulation",
    "rl"
)

sys.path.append(RL_DIR)

from h1_locomotion_env import H1ObstacleCourseEnv


# ==============================================================================
# FASTAPI
# ==============================================================================

app = FastAPI(title="Spaceborn Simulation API")

os.makedirs("static", exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==============================================================================
# ENVIRONMENT 1: NORMAL MODE
# ==============================================================================

NORMAL_MODEL_PATH = (
    "H1_robot-Physics-simulation/models/unitree_h1/scene.xml"
)

model_normal = load_h1(  # E9: sole-box feet when physics walking is on
    NORMAL_MODEL_PATH
)

model_normal.vis.global_.offwidth = 1280
model_normal.vis.global_.offheight = 720

data_normal = mujoco.MjData(
    model_normal
)


# ==============================================================================
# ENVIRONMENT 2: PICK & PLACE MODE
# ==============================================================================

def load_pick_place_model():

    pnp_scene = os.path.join(
        "H1_robot-Pick_Place",
        "assets",
        "h1_pick_place_scene.xml"
    )

    gripper_path = os.path.join(
        "H1_robot-Pick_Place",
        "mujoco_menagerie",
        "robotiq_2f85",
        "2f85.xml"
    )

    if not os.path.exists(pnp_scene):

        pnp_scene = (
            "H1_robot-Pick_Place/"
            "assets/h1_pick_place_scene.xml"
        )

        gripper_path = (
            "H1_robot-Pick_Place/"
            "mujoco_menagerie/robotiq_2f85/2f85.xml"
        )

    spec = mujoco.MjSpec.from_file(
        pnp_scene
    )

    gripper_spec = mujoco.MjSpec.from_file(
        gripper_path
    )

    h1_assets = os.path.abspath(os.path.join(
        "H1_robot-Pick_Place", "mujoco_menagerie", "unitree_h1", "assets"
    ))
    for mesh in spec.meshes:
        mesh.file = os.path.join(h1_assets, os.path.basename(mesh.file))

    def find_body_recursive(
        bodies,
        target_name
    ):

        for b in bodies:

            if b.name == target_name:
                return b

            result = find_body_recursive(
                b.bodies,
                target_name
            )

            if result:
                return result

        return None

    elbow = find_body_recursive(
        spec.bodies,
        "right_elbow_link"
    )

    if elbow:

        mount_site = elbow.add_site(
            name="gripper_mount",
            pos=[0.25, 0, 0],
            euler=[1.57, 1.57, 0]
        )

        spec.attach(
            gripper_spec,
            site=mount_site,
            prefix="gripper_"
        )

        elbow.add_site(
            name="tcp",
            pos=[0.40, 0, 0]
        )

    m = spec.compile()

    m.vis.global_.offwidth = 1280
    m.vis.global_.offheight = 720

    d = mujoco.MjData(m)

    return m, d


model_pnp, data_pnp = load_pick_place_model()


# ==============================================================================
# ENVIRONMENT 3: OBSTACLE MODE
# ==============================================================================

h1_env = H1ObstacleCourseEnv(
    render_mode=None,
    use_obstacles=True,
    curriculum_stage=2
)

model_obs = h1_env.model

model_obs.vis.global_.offwidth = 1280
model_obs.vis.global_.offheight = 720

data_obs = h1_env.data

# Optional: run Obstacle Avoidance in another environment, e.g.
#   M0NARCH_NAV_SCENE=H1_robot-Physics-simulation/models/unitree_h1/scene_course2.xml python server.py
NAV_SCENE = os.environ.get("M0NARCH_NAV_SCENE", "H1_robot-RL-Perception/models/unitree_h1/scene_obstacles.xml")
if "M0NARCH_NAV_SCENE" in os.environ or walker is not None:   # E9: reload with sole-box feet
    model_obs = load_h1(NAV_SCENE)
    model_obs.vis.global_.offwidth = 1280
    model_obs.vis.global_.offheight = 720
    data_obs = mujoco.MjData(model_obs)
    print(f"[NAV] Obstacle Avoidance environment: {NAV_SCENE}")

# Sensor-based navigation (perception_nav.py): LiDAR + camera -> map -> A*.
# Change the goal at runtime with POST /api/nav/goal {"x": 7.9, "y": 0.0}
nav = Navigator(NavConfig())
NAV_GOAL = [7.9, 0.0]
_goal_site = mujoco.mj_name2id(model_obs, mujoco.mjtObj.mjOBJ_SITE, "nav_goal")
if _goal_site >= 0:                                  # the environment defines its own goal
    NAV_GOAL = [float(v) for v in model_obs.site_pos[_goal_site][:2]]


# ==============================================================================
# SIMULATION MANAGER
# ==============================================================================

class SimManager:

    def __init__(self):

        self.state = "STAND"

        self.lock = threading.Lock()

        self.running = True

        self.fps = 0.0

        self.time = 0.0

        self.frame_bytes = b""

        self.eye_frame_bytes = b""

        self.task_start_time = 0.0

        self.walk_phase = 0.0

        self.active_mode = "NORMAL"

        # Every Pick & Place button click requests a completely fresh run.
        self.pnp_restart_requested = False

        # ==============================================================
        # CAMERA STATE
        # ==============================================================

        self.camera_azimuth = 145.0

        self.camera_elevation = -15.0

        self.camera_distance = 3.5


sim = SimManager()


# ==============================================================================
# CAMERA REQUEST MODEL
# ==============================================================================

class CameraCommand(BaseModel):

    azimuth: float | None = None

    elevation: float | None = None

    distance: float | None = None


# ==============================================================================
# PICK & PLACE IDs
# ==============================================================================

mujoco.mj_forward(
    model_pnp,
    data_pnp
)

pnp_home_qpos = data_pnp.qpos.copy()


gripper_site_id = mujoco.mj_name2id(
    model_pnp,
    mujoco.mjtObj.mjOBJ_SITE,
    "tcp"
)

object_body_id = mujoco.mj_name2id(
    model_pnp,
    mujoco.mjtObj.mjOBJ_BODY,
    "object"
)

obj_jnt_id = mujoco.mj_name2id(
    model_pnp,
    mujoco.mjtObj.mjOBJ_JOINT,
    "object_joint"
)

obj_qpos_adr = (
    model_pnp.jnt_qposadr[obj_jnt_id]
    if obj_jnt_id != -1
    else 0
)

obj_dof_adr = (
    model_pnp.jnt_dofadr[obj_jnt_id]
    if obj_jnt_id != -1
    else 0
)


# ==============================================================================
# GRIPPER ACTUATOR
# ==============================================================================

gripper_actuator_id = None

for i in range(model_pnp.nu):

    act_name = mujoco.mj_id2name(
        model_pnp,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    if act_name and "gripper" in act_name:

        gripper_actuator_id = i

        break


# ==============================================================================
# ARM JOINTS
# ==============================================================================

possible_arm_joints = [

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint"

]


pnp_ik_dofs = []

pnp_ik_joint_ids = []


for j in possible_arm_joints:

    jid = mujoco.mj_name2id(
        model_pnp,
        mujoco.mjtObj.mjOBJ_JOINT,
        j
    )

    if jid != -1:

        pnp_ik_dofs.append(
            model_pnp.jnt_dofadr[jid]
        )

        pnp_ik_joint_ids.append(
            jid
        )


# ==============================================================================
# BALANCE CONTROLLER
# ==============================================================================

class H1BalanceController:

    def __init__(
        self,
        m,
        d
    ):

        self.actuator_map = (
            h1_balancer.actuator_dof_map(m)
        )

        self.actuator_limits = (
            h1_balancer.get_actuator_limits(m)
        )

        home_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_KEY,
            "home"
        )

        if home_id >= 0:

            d.qpos[:] = (
                m.key_qpos[home_id]
            )

            d.qvel[:] = 0.0

            mujoco.mj_forward(
                m,
                d
            )

        self.home_q = (
            h1_balancer.get_home_joint_state(m)
        )

        self.com_home = (
            h1_balancer.get_home_com(m, d)
        )


    def apply_stand_control(
        self,
        m,
        d
    ):

        mass, com, vcom = (
            h1_balancer.get_com_state(
                m,
                d
            )
        )

        contacts = (
            h1_balancer.get_contacts(
                m,
                d
            )
        )

        com_error = (
            self.com_home - com
        )

        a_com_des = (
            h1_balancer.COM_KP
            * com_error
            - h1_balancer.COM_KD
            * vcom
        )

        a_com_des = h1_balancer.clamp(
            a_com_des,
            np.array([
                -3.0,
                -3.0,
                -3.0
            ]),
            np.array([
                3.0,
                3.0,
                3.0
            ])
        )

        gravity = m.opt.gravity.copy()

        desired_force = (
            mass
            * (a_com_des - gravity)
        )

        base_ang_acc = (
            h1_balancer
            .base_orientation_acceleration(
                m,
                d
            )
        )

        I_eff = np.array([
            3.0,
            3.0,
            1.5
        ])

        desired_moment = (
            h1_balancer.clamp(
                I_eff * base_ang_acc,
                np.array([
                    -30.0,
                    -30.0,
                    -15.0
                ]),
                np.array([
                    30.0,
                    30.0,
                    15.0
                ])
            )
        )

        if len(contacts) > 0:

            lambdas, _ = (
                h1_balancer.solve_contact_qp(
                    contacts,
                    desired_force,
                    desired_moment,
                    com
                )
            )

            contact_forces = [
                lambdas[i]
                for i in range(
                    len(contacts)
                )
            ]

            contact_generalized_force = (
                h1_balancer
                .build_contact_generalized_force(
                    m,
                    d,
                    contacts,
                    contact_forces
                )
            )

        else:

            contact_generalized_force = (
                np.zeros(m.nv)
            )

        qdd_des = (
            h1_balancer.posture_acceleration(
                m,
                d,
                self.home_q
            )
        )

        qdd_des[0:3] = a_com_des

        qdd_des[3:6] = base_ang_acc

        tau, _ = (
            h1_balancer
            .inverse_dynamics_torque(
                m,
                d,
                qdd_des,
                contact_generalized_force,
                self.actuator_map
            )
        )

        d.ctrl[:] = h1_balancer.clamp(
            tau,
            -self.actuator_limits,
            self.actuator_limits
        )


# ==============================================================================
# SET JOINT QPOS
# ==============================================================================

def set_joint_qpos(
    m,
    d,
    name: str,
    value: float
):

    try:

        joint_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        if joint_id >= 0:

            qpos_adr = (
                m.jnt_qposadr[joint_id]
            )

            d.qpos[qpos_adr] = value

    except Exception:

        pass


# ==============================================================================
# CONTROL LOOP
# ==============================================================================

def control_loop():

    global data_normal
    global data_obs
    global data_pnp

    with sim.lock:

        balancer = StandBalancer(  # E15 fix (stand_fix.py)
            model_normal,
            data_normal
        )

    # --------------------------------------------------------------------------
    # PNP STATE MACHINE
    # --------------------------------------------------------------------------

    pnp_state_machine = "STAND"

    pnp_stand_timer = 0

    pnp_lift_target = None

    pnp_place_target = np.array([
        0.42,
        0.10,
        0.95
    ])

    pnp_grasp_timer = 0

    pnp_release_timer = 0

    pnp_initial_block_pos = np.array([
        0.42,
        -0.15,
        0.97
    ])

    pnp_transport_pos = None

    pnp_drop_timer = 0


    while sim.running:

        step_start = time.perf_counter()

        with sim.lock:

            # ==================================================================
            # STAND
            # ==================================================================

            if (
                sim.state == "STAND"
                and sim.active_mode == "NORMAL"
            ):

                balancer.apply_stand_control(
                    model_normal,
                    data_normal
                )

                mujoco.mj_step(
                    model_normal,
                    data_normal
                )


            # ==================================================================
            # PICK & PLACE
            # ==================================================================

            elif (
                sim.state == "PICK_AND_PLACE"
                and sim.active_mode == "PNP"
            ):

                # ==============================================================
                # RESTART THE COMPLETE PICK & PLACE SEQUENCE
                # ==============================================================
                # The MuJoCo data is reset by the API button handler. This flag
                # also resets the local state machine and all PNP timers so a
                # second, third, or later click starts from the same first frame.

                if sim.pnp_restart_requested:

                    pnp_state_machine = "STAND"
                    pnp_stand_timer = 0
                    pnp_grasp_timer = 0
                    pnp_release_timer = 0
                    pnp_drop_timer = 0
                    pnp_lift_target = None
                    pnp_transport_pos = None

                    data_pnp.qpos[:] = pnp_home_qpos.copy()
                    data_pnp.qvel[:] = 0.0

                    if obj_jnt_id != -1:

                        data_pnp.qpos[
                            obj_qpos_adr:
                            obj_qpos_adr + 3
                        ] = pnp_initial_block_pos

                        data_pnp.qpos[
                            obj_qpos_adr + 3:
                            obj_qpos_adr + 7
                        ] = [1, 0, 0, 0]

                        data_pnp.qvel[
                            obj_dof_adr:
                            obj_dof_adr + 6
                        ] = 0.0

                    mujoco.mj_forward(
                        model_pnp,
                        data_pnp
                    )

                    sim.pnp_restart_requested = False

                m = model_pnp

                d = data_pnp

                gripper_pos = (
                    d.site_xpos[
                        gripper_site_id
                    ]
                )

                reach_target = (
                    pnp_initial_block_pos.copy()
                )

                target_pos = (
                    reach_target
                )


                # ==============================================================
                # STAND BEFORE START
                # ==============================================================

                if pnp_state_machine == "STAND":

                    pnp_stand_timer += 1

                    if pnp_stand_timer > 400:

                        pnp_state_machine = (
                            "REACHING"
                        )


                # ==============================================================
                # REACH
                # ==============================================================

                elif pnp_state_machine == "REACHING":

                    target_pos = (
                        reach_target
                    )

                    if np.linalg.norm(
                        target_pos - gripper_pos
                    ) < 0.03:

                        pnp_state_machine = (
                            "GRASPING"
                        )


                # ==============================================================
                # GRASP
                # ==============================================================

                elif pnp_state_machine == "GRASPING":

                    target_pos = (
                        reach_target
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 255

                    pnp_grasp_timer += 1

                    if pnp_grasp_timer > 600:

                        pnp_lift_target = (
                            reach_target.copy()
                        )

                        pnp_state_machine = (
                            "LIFTING"
                        )


                # ==============================================================
                # LIFT
                # ==============================================================

                elif pnp_state_machine == "LIFTING":

                    if (
                        pnp_lift_target[2]
                        < reach_target[2] + 0.25
                    ):

                        # Very slow lift
                        pnp_lift_target[2] += (
                            0.00008
                        )

                    target_pos = (
                        pnp_lift_target
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 255

                    if (
                        pnp_lift_target[2]
                        >= reach_target[2] + 0.25
                    ):

                        pnp_state_machine = (
                            "PLACING"
                        )


                # ==============================================================
                # PLACE
                # ==============================================================

                elif pnp_state_machine == "PLACING":

                    if pnp_transport_pos is None:

                        pnp_transport_pos = (
                            gripper_pos.copy()
                        )

                    final_drop_pos = (
                        pnp_place_target
                        + np.array([
                            0,
                            0,
                            0.08
                        ])
                    )

                    # Very slow horizontal movement
                    pnp_transport_pos += (
                        final_drop_pos
                        - pnp_transport_pos
                    ) * 0.0005

                    target_pos = (
                        pnp_transport_pos
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 255

                    if np.linalg.norm(
                        final_drop_pos
                        - gripper_pos
                    ) < 0.04:

                        pnp_state_machine = (
                            "LOWERING"
                        )

                        pnp_drop_timer = 0


                # ==============================================================
                # LOWER
                # ==============================================================

                elif pnp_state_machine == "LOWERING":

                    pnp_transport_pos += (
                        pnp_place_target
                        - pnp_transport_pos
                    ) * 0.0005

                    target_pos = (
                        pnp_transport_pos
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 255

                    pnp_drop_timer += 1

                    if pnp_drop_timer >= 400:

                        pnp_state_machine = (
                            "RELEASE"
                        )


                # ==============================================================
                # RELEASE
                # ==============================================================

                elif pnp_state_machine == "RELEASE":

                    target_pos = (
                        pnp_place_target
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 0.0

                    pnp_release_timer += 1

                    if pnp_release_timer > 300:

                        # DO NOT RESET.
                        # Keep robot at final position.
                        pnp_state_machine = (
                            "HOLD_PLACED"
                        )


                # ==============================================================
                # HOLD FINAL POSITION
                # ==============================================================

                elif (
                    pnp_state_machine
                    == "HOLD_PLACED"
                ):

                    target_pos = (
                        pnp_place_target
                    )

                    if (
                        gripper_actuator_id
                        is not None
                    ):

                        d.ctrl[
                            gripper_actuator_id
                        ] = 0.0


                # ==============================================================
                # IK
                # ==============================================================

                if pnp_state_machine != "STAND":

                    err = (
                        target_pos
                        - gripper_pos
                    )

                    jacp = np.zeros(
                        (3, m.nv)
                    )

                    mujoco.mj_jacSite(
                        m,
                        d,
                        jacp,
                        None,
                        gripper_site_id
                    )

                    jacp_masked = (
                        np.zeros_like(jacp)
                    )

                    for dof in pnp_ik_dofs:

                        jacp_masked[
                            :,
                            dof
                        ] = jacp[
                            :,
                            dof
                        ]

                    damping = 0.05

                    J = jacp_masked

                    dq = (
                        J.T
                        @ np.linalg.inv(
                            J @ J.T
                            + np.eye(3)
                            * damping
                        )
                        @ err
                    )

                    mujoco.mj_integratePos(
                        m,
                        d.qpos,
                        dq,
                        0.0015
                    )


                # ==============================================================
                # KEEP OTHER JOINTS AT HOME
                # ==============================================================

                for i in range(m.njnt):

                    jnt_name = (
                        mujoco.mj_id2name(
                            m,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            i
                        )
                    )

                    is_gripper_joint = (
                        jnt_name
                        and "gripper"
                        in jnt_name
                    )

                    if (
                        pnp_state_machine == "STAND"
                        or (
                            i not in pnp_ik_joint_ids
                            and i != obj_jnt_id
                            and not is_gripper_joint
                        )
                    ):

                        q_idx = (
                            m.jnt_qposadr[i]
                        )

                        d_idx = (
                            m.jnt_dofadr[i]
                        )

                        q_len = (
                            7
                            if (
                                m.jnt_type[i]
                                == mujoco.mjtJoint.mjJNT_FREE
                            )
                            else 1
                        )

                        d_len = (
                            6
                            if (
                                m.jnt_type[i]
                                == mujoco.mjtJoint.mjJNT_FREE
                            )
                            else 1
                        )

                        d.qpos[
                            q_idx:q_idx + q_len
                        ] = pnp_home_qpos[
                            q_idx:q_idx + q_len
                        ]

                        d.qvel[
                            d_idx:d_idx + d_len
                        ] = 0


                # ==============================================================
                # ARM ACTUATORS
                # ==============================================================

                for i in range(m.nu):

                    if i == gripper_actuator_id:

                        continue

                    jid = (
                        m.actuator_trnid[
                            i,
                            0
                        ]
                    )

                    if (
                        jid in pnp_ik_joint_ids
                        and pnp_state_machine != "STAND"
                    ):

                        q_idx = (
                            m.jnt_qposadr[jid]
                        )

                        d.ctrl[i] = (
                            d.qpos[q_idx]
                        )


                # ==============================================================
                # STOP ARM VELOCITY
                # ==============================================================

                for dof in pnp_ik_dofs:

                    d.qvel[dof] = 0


                # ==============================================================
                # OBJECT
                # ==============================================================

                if obj_jnt_id != -1:

                    if pnp_state_machine in [
                        "LIFTING",
                        "PLACING",
                        "LOWERING"
                    ]:

                        d.qpos[
                            obj_qpos_adr:
                            obj_qpos_adr + 3
                        ] = gripper_pos

                        d.qvel[
                            obj_dof_adr:
                            obj_dof_adr + 6
                        ] = 0


                    elif pnp_state_machine in [
                        "RELEASE",
                        "HOLD_PLACED"
                    ]:

                        # Keep object permanently at
                        # the placed position.
                        d.qpos[
                            obj_qpos_adr:
                            obj_qpos_adr + 3
                        ] = pnp_place_target

                        d.qvel[
                            obj_dof_adr:
                            obj_dof_adr + 6
                        ] = 0


                    else:

                        d.qpos[
                            obj_qpos_adr:
                            obj_qpos_adr + 3
                        ] = pnp_initial_block_pos

                        d.qpos[
                            obj_qpos_adr + 3:
                            obj_qpos_adr + 7
                        ] = [
                            1,
                            0,
                            0,
                            0
                        ]

                        d.qvel[
                            obj_dof_adr:
                            obj_dof_adr + 6
                        ] = 0


                mujoco.mj_step(
                    m,
                    d
                )


            # ==================================================================
            # WALK
            # ==================================================================

            elif sim.state in [
                "WALK",
                "WALK_AVOID"
            ]:

                m = (
                    model_obs
                    if sim.active_mode == "OBSTACLE"
                    else model_normal
                )

                d = (
                    data_obs
                    if sim.active_mode == "OBSTACLE"
                    else data_normal
                )

                # E9 fix: real physics walking (physics_walk.py)
                if walker is not None:
                    if (
                        sim.state == "WALK_AVOID"
                        and sim.active_mode == "OBSTACLE"
                    ):
                        walker.follow(m, d, nav.step(m, d))   # None -> step in place
                    else:
                        walker.step(m, d, WALK_CMD)

                # E10-E14 fix: corrected kinematic gait (fallback, see gait_fix.py)
                elif (
                    sim.state == "WALK_AVOID"
                    and sim.active_mode == "OBSTACLE"
                ):
                    # sensor-based navigation: the robot finds its own path
                    yaw_cmd = nav.step(m, d)
                    if yaw_cmd is None:
                        d.time += m.opt.timestep   # stopped: goal reached or no path
                    else:
                        walk_step(m, d, sim, yaw_des=yaw_cmd)
                else:
                    walk_step(m, d, sim)


        # ======================================================================
        # CONTROL LOOP TIMING
        # ======================================================================

        time_until_next_step = (
            0.002
            - (
                time.perf_counter()
                - step_start
            )
        )

        if time_until_next_step > 0:

            time.sleep(
                max(
                    0,
                    time_until_next_step
                    - 0.001
                )
            )

            while (
                time.perf_counter()
                - step_start
            ) < 0.002:

                pass


# ==============================================================================
# RENDER WORKER
# ==============================================================================

def render_worker():

    renderer_normal = mujoco.Renderer(
        model_normal,
        height=720,
        width=1280
    )

    renderer_obs = mujoco.Renderer(
        model_obs,
        height=720,
        width=1280
    )

    renderer_pnp = mujoco.Renderer(
        model_pnp,
        height=720,
        width=1280
    )


    camera = mujoco.MjvCamera()


    while sim.running:

        with sim.lock:

            # ==============================================================
            # SELECT MODEL
            # ==============================================================

            if sim.active_mode == "OBSTACLE":

                m = model_obs

                d = data_obs

                r = renderer_obs

                camera.lookat = np.array([
                    d.qpos[0],
                    0.0,
                    0.9
                ])


            elif sim.active_mode == "PNP":

                m = model_pnp

                d = data_pnp

                r = renderer_pnp

                camera.lookat = np.array([
                    0.35,
                    -0.15,
                    0.9
                ])


            else:

                m = model_normal

                d = data_normal

                r = renderer_normal

                camera.lookat = np.array([
                    d.qpos[0],
                    0.0,
                    0.9
                ])


            # ==============================================================
            # APPLY USER CAMERA
            # ==============================================================

            camera.distance = (
                sim.camera_distance
            )

            camera.azimuth = (
                sim.camera_azimuth
            )

            camera.elevation = (
                sim.camera_elevation
            )


            # ==============================================================
            # MAIN CAMERA RENDER
            # ==============================================================

            r.update_scene(
                d,
                camera=camera
            )

            rgb_image = r.render()

            sim.time = d.time


            # ==============================================================
            # ROBOT EYE CAMERA
            # ==============================================================

            if sim.active_mode == "PNP":

                r.update_scene(
                    d,
                    camera="eye_cam"
                )

                eye_img = r.render()

                bgr_eye = cv2.cvtColor(
                    eye_img,
                    cv2.COLOR_RGB2BGR
                )

                hsv_frame = cv2.cvtColor(
                    bgr_eye,
                    cv2.COLOR_BGR2HSV
                )

                mask = cv2.inRange(
                    hsv_frame,
                    np.array([
                        0,
                        120,
                        70
                    ]),
                    np.array([
                        10,
                        255,
                        255
                    ])
                )

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:

                    largest = max(
                        contours,
                        key=cv2.contourArea
                    )

                    if (
                        cv2.contourArea(
                            largest
                        ) > 50
                    ):

                        x, y, w, h = (
                            cv2.boundingRect(
                                largest
                            )
                        )

                        cv2.rectangle(
                            bgr_eye,
                            (x, y),
                            (
                                x + w,
                                y + h
                            ),
                            (0, 255, 0),
                            2
                        )

                        cv2.circle(
                            bgr_eye,
                            (
                                x + w // 2,
                                y + h // 2
                            ),
                            4,
                            (255, 0, 0),
                            -1
                        )


                _, eye_jpeg = cv2.imencode(
                    ".jpg",
                    bgr_eye,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        80
                    ]
                )

                sim.eye_frame_bytes = (
                    eye_jpeg.tobytes()
                )

            else:

                sim.eye_frame_bytes = b""


        # ======================================================================
        # MAIN JPEG
        # ======================================================================

        bgr = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR
        )

        _, jpeg = cv2.imencode(
            ".jpg",
            bgr,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                70
            ]
        )

        sim.frame_bytes = (
            jpeg.tobytes()
        )


        time.sleep(0.033)


# ==============================================================================
# START THREADS
# ==============================================================================

threading.Thread(
    target=control_loop,
    daemon=True
).start()

threading.Thread(
    target=render_worker,
    daemon=True
).start()


# ==============================================================================
# COMMAND API
# ==============================================================================

@app.post("/api/command/{action}")
def execute_command(
    action: str
):

    valid_actions = [
        "stand",
        "walk",
        "walk_avoid",
        "pick_and_place"
    ]

    act_norm = action.lower()


    if act_norm in valid_actions:

        with sim.lock:

            # ==============================================================
            # STAND
            # ==============================================================

            if act_norm == "stand":

                sim.active_mode = "NORMAL"

                mujoco.mj_resetData(
                    model_normal,
                    data_normal
                )

                home_id = mujoco.mj_name2id(
                    model_normal,
                    mujoco.mjtObj.mjOBJ_KEY,
                    "home"
                )

                if home_id >= 0:

                    data_normal.qpos[:] = (
                        model_normal.key_qpos[
                            home_id
                        ]
                    )

                else:

                    data_normal.qpos[
                        0:3
                    ] = [
                        0.0,
                        0.0,
                        1.05
                    ]

                data_normal.qvel[:] = 0.0

                mujoco.mj_forward(
                    model_normal,
                    data_normal
                )


            # ==============================================================
            # WALK
            # ==============================================================

            elif act_norm == "walk":

                sim.active_mode = "NORMAL"

                mujoco.mj_resetData(
                    model_normal,
                    data_normal
                )

                data_normal.qpos[
                    0:3
                ] = [
                    0.0,
                    0.0,
                    1.05
                ]

                data_normal.qvel[:] = 0.0

                mujoco.mj_forward(
                    model_normal,
                    data_normal
                )
                if walker is not None:
                    walker.reset(model_normal, data_normal)   # E9: upright, policy leg pose


            # ==============================================================
            # WALK AVOID
            # ==============================================================

            elif act_norm == "walk_avoid":

                sim.active_mode = "OBSTACLE"

                mujoco.mj_resetData(
                    model_obs,
                    data_obs
                )

                data_obs.qpos[
                    0:3
                ] = [
                    0.0,
                    0.0,
                    1.05
                ]

                data_obs.qvel[:] = 0.0

                mujoco.mj_forward(
                    model_obs,
                    data_obs
                )
                if walker is not None:
                    walker.reset(model_obs, data_obs)      # E9: upright, policy leg pose
                nav.reset(model_obs, data_obs, NAV_GOAL)


            # ==============================================================
            # PICK & PLACE
            # ==============================================================

            elif act_norm == "pick_and_place":

                sim.active_mode = "PNP"

                # IMPORTANT:
                # Every click requests a brand-new complete PNP replay.
                sim.pnp_restart_requested = True

                mujoco.mj_resetData(
                    model_pnp,
                    data_pnp
                )

                data_pnp.qpos[:] = (
                    pnp_home_qpos.copy()
                )

                data_pnp.qvel[:] = 0.0

                mujoco.mj_forward(
                    model_pnp,
                    data_pnp
                )


            # ==============================================================
            # UPDATE STATE
            # ==============================================================

            sim.state = (
                act_norm.upper()
            )

            sim.task_start_time = (
                time.time()
            )

            if sim.state in [
                "WALK",
                "WALK_AVOID"
            ]:

                sim.walk_phase = 0.0


    return {
        "status": "ok",
        "state": sim.state
    }


# ==============================================================================
# CAMERA API
# ==============================================================================

@app.post("/api/camera")
def update_camera(
    command: CameraCommand
):

    with sim.lock:

        if command.azimuth is not None:

            sim.camera_azimuth = float(
                command.azimuth
            )

        if command.elevation is not None:

            sim.camera_elevation = max(
                -89.0,
                min(
                    89.0,
                    float(
                        command.elevation
                    )
                )
            )

        if command.distance is not None:

            sim.camera_distance = max(
                0.8,
                min(
                    8.0,
                    float(
                        command.distance
                    )
                )
            )

        return {
            "azimuth": sim.camera_azimuth,
            "elevation": sim.camera_elevation,
            "distance": sim.camera_distance
        }


# ==============================================================================
# CAMERA PRESETS
# ==============================================================================

@app.post("/api/camera/preset/{preset}")
def camera_preset(
    preset: str
):

    preset = preset.lower()

    presets = {

        "default": (
            145.0,
            -15.0,
            3.5
        ),

        "front": (
            180.0,
            -10.0,
            3.0
        ),

        "back": (
            0.0,
            -10.0,
            3.0
        ),

        "left": (
            90.0,
            -10.0,
            3.0
        ),

        "right": (
            270.0,
            -10.0,
            3.0
        ),

        "top": (
            180.0,
            -60.0,
            3.5
        )

    }

    if preset not in presets:

        return {
            "status": "error",
            "message": "Unknown camera preset"
        }

    azimuth, elevation, distance = (
        presets[preset]
    )

    with sim.lock:

        sim.camera_azimuth = azimuth

        sim.camera_elevation = elevation

        sim.camera_distance = distance

    return {
        "status": "ok",
        "preset": preset,
        "azimuth": azimuth,
        "elevation": elevation,
        "distance": distance
    }


# ==============================================================================
# TELEMETRY
# ==============================================================================

@app.get("/api/telemetry")
def get_telemetry():

    with sim.lock:

        return {
            "state": sim.state,
            "sim_time": round(
                sim.time,
                2
            ),
            "fps": sim.fps,
            "mode": sim.active_mode,
            "camera": {
                "azimuth": sim.camera_azimuth,
                "elevation": sim.camera_elevation,
                "distance": sim.camera_distance
            }
        }


# ==============================================================================
# VIDEO STREAM
# ==============================================================================

def video_stream_generator():

    while sim.running:

        if sim.frame_bytes:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + sim.frame_bytes
                + b"\r\n"
            )

        time.sleep(0.033)


# ==============================================================================
# EYE CAMERA STREAM
# ==============================================================================

def eye_stream_generator():

    while sim.running:

        if (
            sim.active_mode == "PNP"
            and sim.eye_frame_bytes
        ):

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + sim.eye_frame_bytes
                + b"\r\n"
            )

        time.sleep(0.033)


# ==============================================================================
# VIDEO ENDPOINT
# ==============================================================================

@app.get("/video_feed")
def video_feed():

    return StreamingResponse(
        video_stream_generator(),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ==============================================================================
# EYE CAMERA ENDPOINT
# ==============================================================================

@app.get("/eye_feed")
def eye_feed():

    return StreamingResponse(
        eye_stream_generator(),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ==============================================================================
# DASHBOARD
# ==============================================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def serve_dashboard():

    with open(
        "static/index.html",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# ==============================================================================
# START SERVER
# ==============================================================================


# ==============================================================================
# NAVIGATION VIEW: robot camera + live map (perception_nav.py)
# ==============================================================================
def _nav_placeholder(text):
    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.putText(img, text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return cv2.imencode(".jpg", img)[1].tobytes()

_NAV_WAIT = _nav_placeholder("click OBSTACLE AVOIDANCE")


def _nav_stream(get_frame):
    while sim.running:
        frame = get_frame() or _NAV_WAIT
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.1)


@app.get("/nav/camera_feed")
def nav_camera_feed():
    return StreamingResponse(_nav_stream(nav.camera_jpeg),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/nav/map_feed")
def nav_map_feed():
    return StreamingResponse(_nav_stream(nav.map_jpeg),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/nav/status")
def nav_status():
    return {"status": nav.status, "goal": NAV_GOAL,
            "walked_m": round(float(getattr(nav, "distance", 0.0)), 2)}


class NavGoal(BaseModel):
    x: float
    y: float


@app.post("/api/nav/goal")
def nav_set_goal(goal: NavGoal):
    NAV_GOAL[:] = [goal.x, goal.y]
    with sim.lock:
        nav.set_goal(NAV_GOAL)
    return {"goal": NAV_GOAL}


@app.get("/nav", response_class=HTMLResponse)
def nav_page():
    return """<!doctype html><html><head><title>M0NARCH Navigation</title>
<style>body{background:#0b1220;color:#cfe;font-family:monospace;margin:16px}
.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:#121a2b;border:1px solid #2a3550;
border-radius:10px;padding:10px}img{display:block;background:#555}</style></head><body>
<h3>M0NARCH Navigation (LiDAR + camera)</h3>
<div class=row>
<div class=card><div>ROBOT CAMERA (head)</div><img src="/nav/camera_feed" width=480 height=360></div>
<div class=card><div>LIVE MAP (occupancy grid + A* path)</div><img src="/nav/map_feed" height=360></div>
<div class=card><div>WORLD (3rd person)</div><img src="/video_feed" width=480 height=270></div>
</div>
<p id=st>status: ...</p>
<p>Goal x <input id=gx size=5 value=7.9> y <input id=gy size=5 value=0>
<button onclick="setGoal()">set goal</button>
<button onclick="fetch('/api/command/walk_avoid',{method:'POST'})">start</button></p>
<p>Map: white = free, black = obstacle or drop-off, pink = safety margin, grey = unknown,
blue = planned path, green = robot, red ring = goal, orange = latest LiDAR hits.</p>
<script>
function setGoal(){fetch('/api/nav/goal',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({x:+gx.value,y:+gy.value})})}
setInterval(()=>fetch('/api/nav/status').then(r=>r.json()).then(s=>{st.textContent=
'status: '+s.status+'   goal: ('+s.goal.join(', ')+')   walked: '+s.walked_m+' m'}),1000)
</script></body></html>"""



# ==============================================================================
# JOINT VALUES + READINESS SCORE (apply_extras.py)
# ==============================================================================
@app.get("/api/joints")
def api_joints():
    with sim.lock:
        if sim.active_mode == "OBSTACLE":
            m, d = model_obs, data_obs
        elif sim.active_mode == "PNP":
            m, d = model_pnp, data_pnp
        else:
            m, d = model_normal, data_normal
        physics = walker is not None or sim.state not in ("WALK", "WALK_AVOID")   # E9 fixed: physics walk
        joints = []
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            if m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT or j < 0:
                continue
            q = float(d.qpos[m.jnt_qposadr[j]])
            lo, hi = (float(v) for v in m.jnt_range[j])
            joints.append({
                "joint": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                "deg": round(math.degrees(q), 1),
                "range_pct": round(100 * (q - lo) / (hi - lo), 0) if hi > lo else None,
                "vel_dps": round(math.degrees(float(d.qvel[m.jnt_dofadr[j]])), 1) if physics else None,
                "torque_Nm": round(float(d.actuator_force[i]), 1) if physics else None,
            })
        return {"state": sim.state, "physics": physics, "sim_time": round(float(d.time), 2),
                "pelvis_xyz": [round(float(v), 3) for v in d.qpos[:3]], "joints": joints}


@app.get("/api/readiness")
def api_readiness():
    import json as _json
    name = os.path.splitext(os.path.basename(NAV_SCENE))[0]
    path = f"readiness_{name}.json"
    if not os.path.exists(path):
        return {"available": False, "hint": f"run: python evaluate.py --env={NAV_SCENE}"}
    r = _json.load(open(path))
    return {"available": True, "env": name, "readiness": r["readiness"], "band": r["band"],
            "difficulty": r["difficulty"], "parts": r["parts"]}


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
