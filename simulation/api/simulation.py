from typing import Optional

from .commands import RobotCommand
from .robot_state import RobotState
from .simulation_backend import SimulationBackend
from simulation.sensors.sensor_model import SensorConfig, SensorModel


class Simulation:
    """
    Product-level simulation pipeline.

    Separates:
        backend physics state
        from
        simulated sensor state.
    """

    def __init__(
        self,
        backend: SimulationBackend,
        sensor_config: Optional[SensorConfig] = None,
    ):
        self.backend = backend
        self.sensor_model = SensorModel(sensor_config)

        self.initialized = False

    def initialize(self) -> RobotState:
        """Initialize the backend and return the sensor-visible state."""

        self.backend.initialize()
        self.sensor_model.reset()

        self.initialized = True

        return self.sensor_model.process(
            self.backend.get_state()
        )

    def reset(self) -> RobotState:
        """Reset physics and sensor history."""

        if not self.initialized:
            self.initialize()

        self.sensor_model.reset()

        return self.sensor_model.process(
            self.backend.reset()
        )

    def step(
        self,
        dt: Optional[float] = None,
    ) -> RobotState:
        """Advance physics and return the simulated sensor state."""

        if not self.initialized:
            raise RuntimeError(
                "Simulation is not initialized."
            )

        raw_state = self.backend.step(dt)

        return self.sensor_model.process(
            raw_state
        )

    def get_state(self) -> RobotState:
        """Return the current sensor-visible state."""

        if not self.initialized:
            raise RuntimeError(
                "Simulation is not initialized."
            )

        return self.sensor_model.process(
            self.backend.get_state()
        )

    def get_raw_state(self) -> RobotState:
        """
        Return the underlying physics state.

        This bypasses sensor noise and latency and is intended for
        physics validation and ground-truth evaluation.
        """

        if not self.initialized:
            raise RuntimeError(
                "Simulation is not initialized."
            )

        return self.backend.get_state()

    def send_command(
        self,
        command: RobotCommand,
    ) -> None:
        """Forward a simulator-independent command to the backend."""

        if not self.initialized:
            raise RuntimeError(
                "Simulation is not initialized."
            )

        self.backend.send_command(command)

    def apply_disturbance(
        self,
        body_id: int,
        force,
        torque,
    ) -> None:
        """Apply a physics disturbance through the backend."""

        if not self.initialized:
            raise RuntimeError(
                "Simulation is not initialized."
            )

        self.backend.apply_disturbance(
            body_id,
            force,
            torque,
        )

    def is_running(self) -> bool:
        """Return backend running state."""

        return (
            self.initialized
            and self.backend.is_running()
        )

    def shutdown(self) -> None:
        """Shutdown the backend and clear sensor history."""

        self.sensor_model.reset()
        self.backend.shutdown()
        self.initialized = False

    def close(self) -> None:
        """Close the simulation cleanly."""

        self.shutdown()
