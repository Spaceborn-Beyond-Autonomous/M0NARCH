from abc import ABC, abstractmethod
from typing import Optional

from .commands import RobotCommand
from .robot_state import RobotState


class SimulationBackend(ABC):
    """Simulator-independent interface for the ANSA humanoid module."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the simulator and robot model."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> RobotState:
        """Reset simulation and return initial robot state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: Optional[float] = None) -> RobotState:
        """Advance simulation by one step."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return current robot state."""
        raise NotImplementedError

    @abstractmethod
    def send_command(self, command: RobotCommand) -> None:
        """Send a simulator-independent robot command."""
        raise NotImplementedError

    @abstractmethod
    def apply_disturbance(
        self,
        body_id: int,
        force,
        torque,
    ) -> None:
        """Apply an external force and torque."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the simulator is running."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the simulator cleanly."""
        raise NotImplementedError

    def close(self) -> None:
        """Alias for shutdown()."""
        self.shutdown()
