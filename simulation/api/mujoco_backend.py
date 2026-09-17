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

    def initialize(self) -> None:
        """Load the MuJoCo model and create simulation data."""

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

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

    def get_state(self) -> RobotState:
        """Convert MuJoCo state into the common RobotState."""

        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo backend is not initialized.")

        state = empty_h1_state()

        state.timestamp = float(self.data.time)

        state.position = self.data.qpos[:3].copy()

        if self.data.qpos.shape[0] >= 7:
            # MuJoCo quaternion is [w, x, y, z].
            state.orientation = self.data.qpos[3:7].copy()

        state.linear_velocity = self.data.qvel[:3].copy()

        if self.data.qvel.shape[0] >= 6:
            state.angular_velocity = self.data.qvel[3:6].copy()

        actuator_count = min(
            len(state.joint_position),
            self.model.nu,
        )

        if actuator_count > 0:
            state.joint_effort[:actuator_count] = self.data.ctrl[
                :actuator_count
            ]

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
            raise RuntimeError("MuJoCo backend is not initialized.")

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
            raise RuntimeError("MuJoCo backend is not initialized.")

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

    def close(self) -> None:
        """Close the backend."""

        self.shutdown()
