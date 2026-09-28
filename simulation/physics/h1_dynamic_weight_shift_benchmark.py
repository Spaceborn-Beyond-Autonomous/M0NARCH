#!/usr/bin/env python3

from pathlib import Path
import time
import numpy as np
import mujoco
import osqp
from scipy import sparse


# ============================================================================
# UNITREE H1 — CONTACT-AWARE WHOLE-BODY BALANCE V16.1
#
# Real constrained contact-force QP + floating-base inverse dynamics.
#
# XML IS NOT MODIFIED.
# ============================================================================


XML = str(Path(__file__).resolve().parents[2] / "models" / "unitree_h1" / "scene.xml")

GROUND_GEOM = 0
FOOT_GEOMS = {14, 15, 16, 29, 30, 31}

DT = 0.002

# ---------------------------------------------------------------------------
# Balance task
# ---------------------------------------------------------------------------

COM_KP = np.array([35.0, 35.0, 70.0])
COM_KD = np.array([10.0, 10.0, 16.0])

BASE_KP = np.array([35.0, 35.0, 20.0])
BASE_KD = np.array([8.0, 8.0, 5.0])

POSTURE_KP = 12.0
POSTURE_KD = 2.0

# Contact friction.
# The XML contact diagnostics showed approximately mu = 1.
MU = 0.8

# Safety limits.
MIN_BASE_HEIGHT = 0.55
MAX_BASE_TILT = np.deg2rad(35.0)

# Dynamic weight-shift experiment
WEIGHT_SHIFT_START = 2.0
WEIGHT_SHIFT_DURATION = 1.0
WEIGHT_SHIFT_Y = 0.04000000

RIGHT_SHIFT_START = 4.0
RIGHT_SHIFT_DURATION = 1.0

# QP regularization.
LAMBDA_REG = 1e-4
WRENCH_REG = 1e-8


# ============================================================================
# BASIC HELPERS
# ============================================================================

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def actuator_dof_map(m):
    result = []

    for a in range(m.nu):
        joint_id = int(m.actuator_trnid[a, 0])
        dof = int(m.jnt_dofadr[joint_id])
        result.append((a, joint_id, dof))

    return result


def get_actuator_limits(m):
    limits = np.zeros(m.nu)

    for a in range(m.nu):
        # For the H1 torque actuators the actuator force range is the
        # meaningful torque limit.
        if m.actuator_forcerange[a, 1] > 0:
            limits[a] = min(
                abs(m.actuator_forcerange[a, 0]),
                abs(m.actuator_forcerange[a, 1])
            )
        elif m.actuator_ctrlrange[a, 1] > 0:
            limits[a] = min(
                abs(m.actuator_ctrlrange[a, 0]),
                abs(m.actuator_ctrlrange[a, 1])
            )
        else:
            limits[a] = 1e9

    return limits


def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def base_tilt(m, d):
    R = quat_to_rot(d.qpos[3:7])

    # Robot's local +Z axis in world coordinates.
    z_axis = R[:, 2]

    return np.arccos(clamp(z_axis[2], -1.0, 1.0))


def get_com_state(m, d):
    total_mass = 0.0
    com = np.zeros(3)

    for b in range(1, m.nbody):
        mass = m.body_mass[b]
        total_mass += mass
        com += mass * d.xipos[b]

    com /= total_mass

    # MuJoCo subtree_linvel[0] is the center-of-mass linear velocity
    # of the whole system.
    vcom = d.subtree_linvel[0].copy()

    return total_mass, com, vcom


def get_home_joint_state(m):
    home_id = mujoco.mj_name2id(
        m,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if home_id < 0:
        raise RuntimeError("home keyframe not found")

    q = m.key_qpos[home_id].copy()

    # Return the 19 actuated joint positions.
    values = []

    for a in range(m.nu):
        joint_id = int(m.actuator_trnid[a, 0])
        qadr = int(m.jnt_qposadr[joint_id])
        values.append(q[qadr])

    return np.asarray(values)


def get_home_com(m, d):
    saved_qpos = d.qpos.copy()
    saved_qvel = d.qvel.copy()

    home_id = mujoco.mj_name2id(
        m,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    d.qpos[:] = m.key_qpos[home_id]
    d.qvel[:] = 0.0

    mujoco.mj_forward(m, d)

    _, com, _ = get_com_state(m, d)

    d.qpos[:] = saved_qpos
    d.qvel[:] = saved_qvel

    mujoco.mj_forward(m, d)

    return com


# ============================================================================
# CONTACT EXTRACTION
# ============================================================================

def get_contacts(m, d):
    contacts = []

    for i in range(d.ncon):
        c = d.contact[i]

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        valid = (
            (g1 == GROUND_GEOM and g2 in FOOT_GEOMS) or
            (g2 == GROUND_GEOM and g1 in FOOT_GEOMS)
        )

        if not valid:
            continue

        # Contact force returned in contact coordinates.
        force_local = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force_local)

        # Contact frame rows are world axes expressed in contact frame.
        # Therefore transpose maps local force -> world force.
        Rcw = np.asarray(c.frame).reshape(3, 3)

        force_world = Rcw.T @ force_local[:3]

        # Determine foot body.
        foot_geom = g2 if g1 == GROUND_GEOM else g1
        foot_body = int(m.geom_bodyid[foot_geom])

        contacts.append({
            "index": i,
            "geom": foot_geom,
            "body": foot_body,
            "pos": c.pos.copy(),
            "force": force_world,
        })

    return contacts


def foot_vertical_forces(contacts):
    # H1 foot bodies:
    #   6  = left_ankle_link
    #   11 = right_ankle_link
    left_fz = 0.0
    right_fz = 0.0

    for c in contacts:
        fz = max(0.0, float(c["force"][2]))

        if c["body"] == 6:
            left_fz += fz
        elif c["body"] == 11:
            right_fz += fz

    return left_fz, right_fz


def contact_jacobian(m, d, body_id, point):
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))

    mujoco.mj_jac(
        m,
        d,
        jp,
        jr,
        point,
        body_id
    )

    return jp


# ============================================================================
# QP
# ============================================================================

def solve_contact_qp(contacts, desired_force, desired_moment, com):
    """
    Solve:

        minimize ||A lambda - wrench_des||² + regularization

    lambda contains [fx, fy, fz] for each active contact.

    Constraints:

        fz >= 0
        |fx| <= mu*fz
        |fy| <= mu*fz
    """

    n = len(contacts)

    if n == 0:
        raise RuntimeError("No active support contacts")

    nv_lambda = 3 * n

    A = np.zeros((6, nv_lambda))

    for k, c in enumerate(contacts):
        r = c["pos"] - com

        # Force contribution.
        A[0:3, 3*k:3*k+3] = np.eye(3)

        # Moment contribution r x F.
        skew_r = np.array([
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0]
        ])

        A[3:6, 3*k:3*k+3] = skew_r

    wrench_des = np.concatenate([
        desired_force,
        desired_moment
    ])

    # Quadratic objective:
    #
    #   1/2 lambda' P lambda + q' lambda
    #
    # where the least-squares target is:
    #
    #   ||A lambda - wrench||²
    #
    P = 2.0 * (
        A.T @ A +
        LAMBDA_REG * np.eye(nv_lambda)
    )

    q = -2.0 * A.T @ wrench_des

    # -----------------------------------------------------------------------
    # Friction pyramid:
    #
    # fx - mu*fz <= 0
    # -fx - mu*fz <= 0
    # fy - mu*fz <= 0
    # -fy - mu*fz <= 0
    # -fz <= 0
    # -----------------------------------------------------------------------

    rows = []
    lower = []
    upper = []

    for k in range(n):

        fx = 3*k
        fy = 3*k + 1
        fz = 3*k + 2

        row = np.zeros(nv_lambda)
        row[fx] = 1.0
        row[fz] = -MU
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(nv_lambda)
        row[fx] = -1.0
        row[fz] = -MU
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(nv_lambda)
        row[fy] = 1.0
        row[fz] = -MU
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(nv_lambda)
        row[fy] = -1.0
        row[fz] = -MU
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(nv_lambda)
        row[fz] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(np.inf)

    G = sparse.csc_matrix(np.vstack(rows))

    P = sparse.csc_matrix(P)

    solver = osqp.OSQP()

    solver.setup(
        P=P,
        q=q,
        A=G,
        l=np.asarray(lower),
        u=np.asarray(upper),
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=4000,
        polish=False
    )

    result = solver.solve()

    if result.x is None:
        raise RuntimeError(
            f"Contact QP failed: {result.info.status}"
        )

    if result.info.status not in (
        "solved",
        "solved inaccurate"
    ):
        raise RuntimeError(
            f"Contact QP status: {result.info.status}"
        )

    lam = result.x.reshape(n, 3)

    return lam, result


# ============================================================================
# WHOLE-BODY INVERSE DYNAMICS
# ============================================================================

def build_contact_generalized_force(
    m,
    d,
    contacts,
    lambda_world
):
    qfrc = np.zeros(m.nv)

    for c, force in zip(contacts, lambda_world):

        Jp = contact_jacobian(
            m,
            d,
            c["body"],
            c["pos"]
        )

        qfrc += Jp.T @ force

    return qfrc


def build_contact_jacobian(
    m,
    d,
    contacts
):
    """
    Stack exact translational contact Jacobians.
    """

    if not contacts:
        return np.zeros((0, m.nv))

    blocks = []

    for c in contacts:
        Jp = contact_jacobian(
            m,
            d,
            c["body"],
            c["pos"]
        )

        blocks.append(Jp)

    return np.vstack(blocks)


def inverse_dynamics_torque(
    m,
    d,
    qdd_des,
    contact_generalized_force,
    actuator_map
):
    """
    Dynamics:

        M qdd + qfrc_bias
          = qfrc_actuator + Jc^T lambda

    Therefore:

        qfrc_actuator =
            M qdd_des
            + qfrc_bias
            - Jc^T lambda

    Only actuated DOFs are converted to motor torques.
    """

    M = np.zeros((m.nv, m.nv), dtype=np.float64)

    # MuJoCo Python API:
    # mj_fullM(model, data, destination)
    mujoco.mj_fullM(
        m,
        d,
        M
    )

    required_generalized_force = (
        M @ qdd_des
        + d.qfrc_bias
        - contact_generalized_force
    )

    tau = np.zeros(m.nu)

    for a, joint_id, dof in actuator_map:
        tau[a] = required_generalized_force[dof]

    return tau, required_generalized_force


# ============================================================================
# POSTURE TASK
# ============================================================================

def posture_acceleration(
    m,
    d,
    home_q
):
    qdd = np.zeros(m.nv)

    for a in range(m.nu):

        joint_id = int(m.actuator_trnid[a, 0])

        qadr = int(m.jnt_qposadr[joint_id])
        dof = int(m.jnt_dofadr[joint_id])

        q = d.qpos[qadr]
        qd = d.qvel[dof]

        qdd[dof] = (
            POSTURE_KP * (home_q[a] - q)
            - POSTURE_KD * qd
        )

    return qdd


# ============================================================================
# ORIENTATION TASK
# ============================================================================

def base_orientation_acceleration(m, d):
    """
    Keep pelvis/torso orientation near the home orientation.

    This is intentionally a modest task. Balance is primarily controlled
    through COM acceleration and contact wrench distribution.
    """

    R = quat_to_rot(d.qpos[3:7])

    # Desired home orientation is identity.
    #
    # Small-angle orientation error can be extracted from the skew part.
    e = 0.5 * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    omega = d.qvel[3:6]

    return (
        -BASE_KP * e
        -BASE_KD * omega
    )


# ============================================================================
# SAFETY
# ============================================================================

def safety_state(m, d, contacts):
    height = float(d.qpos[2])
    tilt = float(base_tilt(m, d))

    if not np.all(np.isfinite(d.qpos)):
        return False, "NONFINITE_QPOS"

    if not np.all(np.isfinite(d.qvel)):
        return False, "NONFINITE_QVEL"

    if height < MIN_BASE_HEIGHT:
        return False, "LOW_BASE"

    if tilt > MAX_BASE_TILT:
        return False, "EXCESSIVE_TILT"

    if len(contacts) == 0:
        return False, "NO_SUPPORT"

    total_fz = sum(max(0.0, c["force"][2]) for c in contacts)

    # Do not demand exactly mg because contact forces are solver quantities
    # and transient dynamics can legitimately alter the vertical force.
    if total_fz < 0.25 * m.body_mass[1]:
        return False, "INSUFFICIENT_SUPPORT"

    return True, "OK"


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print(" UNITREE H1 — CONTACT-AWARE WHOLE-BODY BALANCE V16.1")
    print("=" * 80)

    print("\nMODEL")
    print("-" * 80)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)

    print("MuJoCo version :", mujoco.__version__)
    print("Bodies         :", m.nbody)
    print("Joints         :", m.njnt)
    print("DOFs           :", m.nv)
    print("Actuators      :", m.nu)

    actuator_map = actuator_dof_map(m)
    actuator_limits = get_actuator_limits(m)

    print("\nACTUATOR → DOF MAP")
    print("-" * 80)

    for a, joint_id, dof in actuator_map:
        name = mujoco.mj_id2name(
            m,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id
        )

        print(
            f"{a:2d}: "
            f"{name:<34} "
            f"dof={dof:2d} "
            f"limit=±{actuator_limits[a]:.1f}"
        )

    # -----------------------------------------------------------------------
    # HOME
    # -----------------------------------------------------------------------

    home_id = mujoco.mj_name2id(
        m,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if home_id < 0:
        raise RuntimeError("home keyframe not found")

    d.qpos[:] = m.key_qpos[home_id]
    d.qvel[:] = 0.0

    mujoco.mj_forward(m, d)

    # -----------------------------------------------------------------------
    # CONTACT INITIALIZATION / SETTLING
    # -----------------------------------------------------------------------
    # Let MuJoCo establish the initial foot-ground contact state before
    # enabling the V16.1 balance controller.
    d.ctrl[:] = 0.0

    for _ in range(20):
        mujoco.mj_step(m, d)

    mujoco.mj_forward(m, d)

    mass, com_home, _ = get_com_state(m, d)

    home_q = get_home_joint_state(m)

    print("\nINITIAL STATE")
    print("-" * 80)

    print(f"Mass       : {mass:.3f}")
    print(f"Base       : {d.qpos[:3]}")
    print(f"COM        : {com_home}")
    print(f"Contacts   : {d.ncon}")
    print(f"Base tilt  : {np.rad2deg(base_tilt(m, d)):.2f} deg")

    contacts = get_contacts(m, d)

    for i, c in enumerate(contacts):
        print(
            f" contact {i}: "
            f"geom={c['geom']} "
            f"body={c['body']} "
            f"pos={c['pos']} "
            f"F={c['force']}"
        )

    print("\nV16.1 MODE")
    print("-" * 80)
    print("True OSQP contact-force QP")
    print("Exact contact Jacobians")
    print("Floating-base inverse dynamics")
    print("Actuator torque saturation")
    print("Quiet-standing test only")
    print()
    print("Starting V16.1 controller...")

    start_wall = time.time()
    last_print = -1.0

    # -----------------------------------------------------------
    # DYNAMIC BENCHMARK METRICS
    # -----------------------------------------------------------
    max_com_y = 0.0
    max_tilt_deg = 0.0
    min_contacts = d.ncon
    min_fz = float("inf")
    max_tau = 0.0

    safety_stop = False
    safety_reason = ""
    safety_time = None

    try:

        # HEADLESS REGRESSION MODE
        # Same V16.1 controller and physics. GUI disabled.
        headless_duration = 8.0
        headless_start = time.time()

        while d.time < headless_duration:

                sim_t = float(d.time)

                # -----------------------------------------------------------
                # STATE
                # -----------------------------------------------------------

                mass, com, vcom = get_com_state(m, d)

                contacts = get_contacts(m, d)

                # -----------------------------------------------------------
                # DESIRED COM ACCELERATION
                #
                # Desired COM = HOME COM.
                # -----------------------------------------------------------

                # Dynamic lateral weight-shift target.
                # Positive Y shifts the desired CoM toward the left side.
                com_target = com_home.copy()

                # Shift load toward the left side
                if WEIGHT_SHIFT_START <= sim_t < (
                    WEIGHT_SHIFT_START + WEIGHT_SHIFT_DURATION
                ):
                    com_target[1] += WEIGHT_SHIFT_Y

                # Shift load toward the right side
                elif RIGHT_SHIFT_START <= sim_t < (
                    RIGHT_SHIFT_START + RIGHT_SHIFT_DURATION
                ):
                    com_target[1] -= WEIGHT_SHIFT_Y

                com_error = com_target - com

                a_com_des = (
                    COM_KP * com_error
                    - COM_KD * vcom
                )

                # Limit desired acceleration.
                a_com_des = clamp(
                    a_com_des,
                    np.array([-3.0, -3.0, -3.0]),
                    np.array([3.0, 3.0, 3.0])
                )

                # Net external force required:
                #
                #   m*a = F_contact + m*g
                #
                # so:
                #
                #   F_contact = m*(a - g)
                #

                gravity = m.opt.gravity.copy()

                desired_force = mass * (
                    a_com_des - gravity
                )

                # -----------------------------------------------------------
                # DESIRED BODY MOMENT
                #
                # Small orientation correction.
                # -----------------------------------------------------------

                base_ang_acc = base_orientation_acceleration(m, d)

                # Approximate rotational inertia from torso/pelvis bodies.
                # This is deliberately modest; contact wrench distribution
                # is still solved by the QP.
                I_eff = np.array([3.0, 3.0, 1.5])

                desired_moment = I_eff * base_ang_acc

                desired_moment = clamp(
                    desired_moment,
                    np.array([-30.0, -30.0, -15.0]),
                    np.array([30.0, 30.0, 15.0])
                )

                # -----------------------------------------------------------
                # CONTACT QP
                # -----------------------------------------------------------

                if len(contacts) > 0:

                    lambdas, qp_result = solve_contact_qp(
                        contacts,
                        desired_force,
                        desired_moment,
                        com
                    )

                    # Use optimized contact forces.
                    contact_forces = [
                        lambdas[i]
                        for i in range(len(contacts))
                    ]

                    # Exact J^T lambda.
                    contact_generalized_force = (
                        build_contact_generalized_force(
                            m,
                            d,
                            contacts,
                            contact_forces
                        )
                    )

                else:
                    contact_forces = []
                    contact_generalized_force = np.zeros(m.nv)
                    qp_result = None

                # -----------------------------------------------------------
                # WHOLE-BODY DESIRED ACCELERATION
                # -----------------------------------------------------------

                qdd_des = posture_acceleration(
                    m,
                    d,
                    home_q
                )

                # Floating base is governed by the balance task.
                #
                # qdd[0:3] = desired COM acceleration
                # qdd[3:6] = desired angular acceleration
                #
                # Approximation for the free-floating generalized state,
                # while contact forces satisfy the support wrench.
                qdd_des[0:3] = a_com_des
                qdd_des[3:6] = base_ang_acc

                # -----------------------------------------------------------
                # INVERSE DYNAMICS
                # -----------------------------------------------------------

                tau, required_generalized = inverse_dynamics_torque(
                    m,
                    d,
                    qdd_des,
                    contact_generalized_force,
                    actuator_map
                )

                # -----------------------------------------------------------
                # ACTUATOR SATURATION
                # -----------------------------------------------------------

                ctrl = clamp(
                    tau,
                    -actuator_limits,
                    actuator_limits
                )

                d.ctrl[:] = ctrl

                # -----------------------------------------------------------
                # STEP
                # -----------------------------------------------------------

                mujoco.mj_step(m, d)


                # -----------------------------------------------------------
                # SAFETY AFTER PHYSICS STEP
                # -----------------------------------------------------------

                mujoco.mj_forward(m, d)

                contacts_after = get_contacts(m, d)

                # -----------------------------------------------------------
                # UPDATE DYNAMIC BENCHMARK METRICS
                # -----------------------------------------------------------
                com_y_abs = abs(float(com[1] - com_home[1]))
                tilt_deg = float(np.rad2deg(base_tilt(m, d)))

                max_com_y = max(max_com_y, com_y_abs)
                max_tilt_deg = max(max_tilt_deg, tilt_deg)
                min_contacts = min(min_contacts, len(contacts_after))

                current_fz = sum(
                    max(0.0, float(c["force"][2]))
                    for c in contacts_after
                )
                min_fz = min(min_fz, current_fz)

                current_tau = (
                    float(np.max(np.abs(ctrl)))
                    if len(ctrl)
                    else 0.0
                )
                max_tau = max(max_tau, current_tau)

                safe, reason = safety_state(
                    m,
                    d,
                    contacts_after
                )

                if not safe:

                    safety_stop = True
                    safety_reason = str(reason)
                    safety_time = float(d.time)

                    print("\n")
                    print("=" * 80)
                    print("SAFETY STOP")
                    print("=" * 80)
                    print(f"Reason : {reason}")
                    print(f"Time   : {d.time:.6f}")
                    print(f"Height : {d.qpos[2]:.6f}")
                    print(
                        f"Tilt   : "
                        f"{np.rad2deg(base_tilt(m, d)):.3f} deg"
                    )
                    print(f"Contacts: {len(contacts_after)}")

                    break

                # -----------------------------------------------------------
                # STATUS
                # -----------------------------------------------------------

                if sim_t - last_print >= 0.25:

                    fz = sum(
                        max(0.0, f[2])
                        for f in contact_forces
                    )

                    left_fz, right_fz = foot_vertical_forces(
                        contacts_after
                    )

                    tau_max = (
                        float(np.max(np.abs(ctrl)))
                        if len(ctrl)
                        else 0.0
                    )

                    print(
                        f"t={d.time:7.2f}s "
                        f"z={d.qpos[2]:+.3f} "
                        f"COM=({com[0]:+.3f},"
                        f"{com[1]:+.3f},"
                        f"{com[2]:+.3f}) "
                        f"tilt={np.rad2deg(base_tilt(m,d)):5.1f}deg "
                        f"contacts={len(contacts_after):2d} "
                        f"L_Fz={left_fz:6.1f} "
                        f"R_Fz={right_fz:6.1f} "
                        f"QP_Fz={fz:7.1f} "
                        f"tau_max={tau_max:6.1f} "
                        f"state=OK"
                    )

                    last_print = sim_t

                # Keep real-time-ish pacing.
                elapsed = time.time() - start_wall
                target = float(d.time)

                sleep_time = target - elapsed

                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.005))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\n")
        print("=" * 80)
        print("DYNAMIC WEIGHT-SHIFT BENCHMARK RESULT")
        print("=" * 80)

        result = "FAIL" if safety_stop else "PASS"

        if min_fz == float("inf"):
            min_fz = 0.0

        print(f"Shift command       : +/- {WEIGHT_SHIFT_Y:.6f} m")
        print(f"Result              : {result}")
        print(f"Peak |CoM Y|        : {max_com_y:.6f} m")
        print(f"Peak base tilt      : {max_tilt_deg:.3f} deg")
        print(f"Minimum contacts    : {min_contacts}")
        print(f"Minimum total Fz    : {min_fz:.3f} N")
        print(f"Maximum torque      : {max_tau:.3f} N*m")
        print(f"Safety stop         : {safety_stop}")

        if safety_stop:
            print(f"Safety reason       : {safety_reason}")
            print(f"Safety time         : {safety_time:.6f} s")

        print(f"Final sim time      : {d.time:.3f} s")
        print("=" * 80)


if __name__ == "__main__":
    main()
