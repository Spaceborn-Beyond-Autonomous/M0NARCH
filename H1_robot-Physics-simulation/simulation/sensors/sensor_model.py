from collections import deque
from dataclasses import dataclass

import numpy as np

from simulation.api.robot_state import RobotState


@dataclass
class SensorConfig:
    """
    Simulation-level sensor noise and latency configuration.

    All standard deviations are in the native units of the signal.
    Latency is specified in seconds.
    """

    imu_accel_std: float = 0.0
    imu_gyro_std: float = 0.0

    joint_position_std: float = 0.0
    joint_velocity_std: float = 0.0
    joint_effort_std: float = 0.0

    contact_force_std: float = 0.0

    latency_seconds: float = 0.0

    random_seed: int = 42


class SensorModel:
    """
    Simulation-level sensor model.

    Applies configurable measurement noise and latency to RobotState
    without modifying the underlying MuJoCo physics state.
    """

    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()

        if self.config.latency_seconds < 0.0:
            raise ValueError(
                "latency_seconds must be >= 0"
            )

        self.rng = np.random.default_rng(
            self.config.random_seed
        )

        self.buffer = deque()

    def _noise(
        self,
        value: np.ndarray,
        std: float,
    ) -> np.ndarray:
        """Add zero-mean Gaussian noise."""

        value = np.asarray(value, dtype=float)

        if std <= 0.0:
            return value.copy()

        return value + self.rng.normal(
            loc=0.0,
            scale=std,
            size=value.shape,
        )

    def _apply_noise(
        self,
        state: RobotState,
    ) -> RobotState:
        """Apply configured sensor noise to a state copy."""

        noisy = state.copy()

        noisy.imu_acceleration = self._noise(
            state.imu_acceleration,
            self.config.imu_accel_std,
        )

        noisy.imu_angular_velocity = self._noise(
            state.imu_angular_velocity,
            self.config.imu_gyro_std,
        )

        noisy.joint_position = self._noise(
            state.joint_position,
            self.config.joint_position_std,
        )

        noisy.joint_velocity = self._noise(
            state.joint_velocity,
            self.config.joint_velocity_std,
        )

        noisy.joint_effort = self._noise(
            state.joint_effort,
            self.config.joint_effort_std,
        )

        noisy.contact_forces = self._noise(
            state.contact_forces,
            self.config.contact_force_std,
        )

        return noisy

    def _get_delayed_state(
        self,
        current_time: float,
    ) -> RobotState:
        """
        Return the newest buffered state whose timestamp is old enough
        to satisfy the configured latency.
        """

        target_time = (
            current_time
            - self.config.latency_seconds
        )

        selected = None

        for timestamp, state in self.buffer:
            if timestamp <= target_time:
                selected = state
            else:
                break

        if selected is None:
            # Not enough history yet.
            return self.buffer[0][1].copy()

        return selected.copy()

    def process(
        self,
        state: RobotState,
    ) -> RobotState:
        """
        Apply sensor noise and latency to a RobotState.

        The input state is never modified.
        """

        noisy_state = self._apply_noise(state)

        if self.config.latency_seconds <= 0.0:
            return noisy_state

        self.buffer.append(
            (
                noisy_state.timestamp,
                noisy_state.copy(),
            )
        )

        # Keep only enough history for the requested latency.
        minimum_time = (
            state.timestamp
            - self.config.latency_seconds
            - 1.0
        )

        while (
            len(self.buffer) > 1
            and self.buffer[1][0] < minimum_time
        ):
            self.buffer.popleft()

        return self._get_delayed_state(
            state.timestamp
        )

    def reset(self) -> None:
        """Clear buffered sensor history and reset noise sequence."""

        self.buffer.clear()

        self.rng = np.random.default_rng(
            self.config.random_seed
        )
