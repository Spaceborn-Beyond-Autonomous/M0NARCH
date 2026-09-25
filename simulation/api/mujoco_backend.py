from typing import Optional

import mujoco
import numpy as np

from .commands import CommandType, RobotCommand
from .robot_state import RobotState, empty_h1_state
from .simulation_backend import SimulationBackend


class MuJoCoBackend(SimulationBackend):
    """
    MuJoCo implementation of the ANSA SimulationBackend interface.

    This adapter provides the common simulation API without exposing
    MuJoCo-specific objects to higher-level modules.
    """

    def __init__(self, xml_path: str):
        self.xml_path = xml_path

        self.model = None
        self.data = None

        self.running = False
        self.last_command: Optional[RobotCommand] = None

        # IMU sensor IDs
        self.imu_accel_id: int = -1
        self.imu_gyro_id: int = -1

        # H1 foot contact geometry
        # Verified from models/unitree_h1/scene.xml
        self.floor_geom_id: int = 0
        self.left_foot_geoms = set(range(13, 17))
        self.right_foot_geoms = set(range(28, 32))

    def initialize(self) -> None:
        """Load the MuJoCo model and create simulation data."""

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Find IMU sensors defined in the H1 XML model.
        self.imu_accel_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_accel",
        )

        self.imu_gyro_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_gyro",
        )

        if self.imu_accel_id < 0:
            raise RuntimeError(
                "H1 IMU accelerometer sensor 'imu_accel' "
                "was not found in the MuJoCo model."
            )

        if self.imu_gyro_id < 0:
            raise RuntimeError(
                "H1 IMU gyro sensor 'imu_gyro' "
                "was not found in the MuJoCo model."
            )

        mujoco.mj_forward(self.model, self.data)

        self.running = True

    def reset(self) -> RobotState:
        """Reset MuJoCo simulation to the model's initial state."""

        if self.model is None or self.data is None:
            self.initialize()

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self.get_state()

    def step(self, dt: Optional[float] = None) -> RobotState:
        """
        Advance the MuJoCo simulation.

        The backend normally uses the timestep defined by the model.
        """

        if not self.running:
            raise RuntimeError("MuJoCo backend is not running.")

        if dt is not None:
            original_timestep = self.model.opt.timestep
            self.model.opt.timestep = float(dt)

            mujoco.mj_step(self.model, self.data)

            self.model.opt.timestep = original_timestep
        else:
            mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def _get_imu_state(self, state: RobotState) -> None:
        """
        Read IMU measurements from MuJoCo sensor outputs.

        The accelerometer and gyro are defined in the H1 XML model
        and attached to the 'imu' site.

        Noise and latency will be added later as a separate sensor layer.
        """

        if self.model is None or self.data is None:
            return

        if self.imu_accel_id < 0 or self.imu_gyro_id < 0:
            return

        # ---------------------------------------------------------
        # Accelerometer
        # ---------------------------------------------------------

        accel_adr = self.model.sensor_adr[self.imu_accel_id]

        state.imu_acceleration = self.data.sensordata[
            accel_adr : accel_adr + 3
        ].copy()

        # ---------------------------------------------------------
        # Gyroscope
        # ---------------------------------------------------------

        gyro_adr = self.model.sensor_adr[self.imu_gyro_id]

        state.imu_angular_velocity = self.data.sensordata[
            gyro_adr : gyro_adr + 3
        ].copy()

    def _get_contact_state(self, state: RobotState) -> None:
        """
        Read H1 left/right foot contact state and world-frame forces.

        contact_states:
            [left_foot_contact, right_foot_contact]

        contact_forces:
            shape (2, 3), world-frame force in Newtons.
        """

        if self.model is None or self.data is None:
            return

        left_contact = False
        right_contact = False

        left_force = np.zeros(3, dtype=float)
        right_force = np.zeros(3, dtype=float)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            g1 = contact.geom1
            g2 = contact.geom2

            # Only consider contacts involving the floor.
            if self.floor_geom_id not in (g1, g2):
                continue

            other_geom = (
                g2 if g1 == self.floor_geom_id else g1
            )

            # MuJoCo returns contact force in the contact frame.
            force_contact = np.zeros(6, dtype=float)

            mujoco.mj_contactForce(
                self.model,
                self.data,
                i,
                force_contact,
            )

            # Convert contact-frame force to world coordinates.
            rotation = contact.frame.reshape(3, 3)
            force_world = rotation.T @ force_contact[:3]

            if other_geom in self.left_foot_geoms:
                left_contact = True
                left_force += force_world

            elif other_geom in self.right_foot_geoms:
                right_contact = True
                right_force += force_world

        state.contact_states = [
            left_contact,
            right_contact,
        ]

        state.contact_forces = np.vstack(
            [left_force, right_force]
        )

    def get_state(self) -> RobotState:
        """Convert MuJoCo state into the common RobotState."""

        if self.model is None or self.data is None:
            raise RuntimeError(
                "MuJoCo backend is not initialized."
            )

        state = empty_h1_state()

        # ---------------------------------------------------------
        # Timestamp
        # ---------------------------------------------------------

        state.timestamp = float(self.data.time)

        # ---------------------------------------------------------
        # Floating-base position
        # ---------------------------------------------------------

        state.position = self.data.qpos[:3].copy()

        # ---------------------------------------------------------
        # Floating-base orientation
        # ---------------------------------------------------------

        if self.data.qpos.shape[0] >= 7:
            # MuJoCo quaternion convention:
            # [w, x, y, z]
            state.orientation = self.data.qpos[3:7].copy()

        # ---------------------------------------------------------
        # Floating-base velocity
        # ---------------------------------------------------------

        state.linear_velocity = self.data.qvel[:3].copy()

        if self.data.qvel.shape[0] >= 6:
            state.angular_velocity = self.data.qvel[3:6].copy()

        # ---------------------------------------------------------
        # Joint state
        # ---------------------------------------------------------

        joint_count = min(
            len(state.joint_position),
            self.model.nv - 6,
        )

        if joint_count > 0:
            # H1 has:
            #   qpos[0:3] = floating-base position
            #   qpos[3:7] = floating-base quaternion
            #   qpos[7:]  = joint positions
            #
            # qvel:
            #   qvel[0:6] = floating-base velocity
            #   qvel[6:]  = joint velocities

            state.joint_position[:joint_count] = self.data.qpos[
                7 : 7 + joint_count
            ]

            state.joint_velocity[:joint_count] = self.data.qvel[
                6 : 6 + joint_count
            ]

        # ---------------------------------------------------------
        # Joint effort / actuator command
        # ---------------------------------------------------------

        actuator_count = min(
            len(state.joint_effort),
            self.model.nu,
        )

        if actuator_count > 0:
            state.joint_effort[:actuator_count] = self.data.ctrl[
                :actuator_count
            ]

        # ---------------------------------------------------------
        # IMU
        # ---------------------------------------------------------

        self._get_imu_state(state)

        # ---------------------------------------------------------
        # Foot contact state
        # ---------------------------------------------------------

        self._get_contact_state(state)

        # ---------------------------------------------------------
        # Mode
        # ---------------------------------------------------------

        state.mode = (
            self.last_command.command_type.value
            if self.last_command is not None
            else "IDLE"
        )

        return state

    def send_command(self, command: RobotCommand) -> None:
        """
        Convert supported high-level commands into basic MuJoCo control.

        Detailed walking, balance, manipulation and planning controllers
        remain separate modules.
        """

        if self.model is None or self.data is None:
            raise RuntimeError(
                "MuJoCo backend is not initialized."
            )

        self.last_command = command.copy()

        if command.command_type == CommandType.STOP:
            self.data.ctrl[:] = 0.0

        elif command.command_type == CommandType.POSTURE:
            count = min(
                len(command.values),
                self.model.nu,
            )

            if count > 0:
                self.data.ctrl[:count] = command.values[:count]

    def apply_disturbance(
        self,
        body_id: int,
        force,
        torque,
    ) -> None:
        """Apply a world-frame external force and torque."""

        if self.model is None or self.data is None:
            raise RuntimeError(
                "MuJoCo backend is not initialized."
            )

        if body_id < 0 or body_id >= self.model.nbody:
            raise ValueError(
                f"Invalid body_id={body_id}; "
                f"valid range is 0..{self.model.nbody - 1}"
            )

        self.data.xfrc_applied[body_id, :3] = np.asarray(
            force,
            dtype=float,
        )

        self.data.xfrc_applied[body_id, 3:6] = np.asarray(
            torque,
            dtype=float,
        )

    def is_running(self) -> bool:
        """Return whether the backend is active."""

        return self.running

    def shutdown(self) -> None:
        """Stop the simulation backend."""

        self.running = False
        self.model = None
        self.data = None
        self.last_command = None
        self.imu_accel_id = -1
        self.imu_gyro_id = -1

    def close(self) -> None:
        """Close the backend."""

        self.shutdown()


