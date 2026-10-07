from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class CommandType(str, Enum):
    """Standard ANSA OS humanoid command categories."""

    POSTURE = "POSTURE"
    WALK = "WALK"
    TURN = "TURN"
    STOP = "STOP"
    REACH = "REACH"
    GRASP = "GRASP"
    RELEASE = "RELEASE"
    MOVE_TO = "MOVE_TO"


@dataclass
class RobotCommand:
    """
    Simulator-independent humanoid command.

    The command contains only high-level intent/data.
    Simulator-specific execution belongs to the backend/controller.
    """

    command_type: CommandType

    timestamp: float = 0.0

    # Generic numeric command parameters.
    # Examples:
    #   WALK  -> [forward, lateral, yaw]
    #   TURN  -> [yaw_rate]
    #   POSTURE -> joint targets
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    # Optional target position in world coordinates.
    target_position: Optional[np.ndarray] = None

    # Optional target orientation [w, x, y, z].
    target_orientation: Optional[np.ndarray] = None

    # Optional named target, object, posture, etc.
    target_name: Optional[str] = None

    # Additional command-specific parameters.
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Command priority.
    priority: int = 0

    def copy(self) -> "RobotCommand":
        """Return an independent copy of the command."""

        return RobotCommand(
            command_type=self.command_type,
            timestamp=self.timestamp,
            values=self.values.copy(),
            target_position=(
                None
                if self.target_position is None
                else self.target_position.copy()
            ),
            target_orientation=(
                None
                if self.target_orientation is None
                else self.target_orientation.copy()
            ),
            target_name=self.target_name,
            parameters=dict(self.parameters),
            priority=self.priority,
        )


def stop_command() -> RobotCommand:
    """Create the standard emergency/normal STOP command."""

    return RobotCommand(
        command_type=CommandType.STOP,
        priority=100,
    )


def walk_command(
    forward: float = 0.0,
    lateral: float = 0.0,
    yaw: float = 0.0,
) -> RobotCommand:
    """Create a WALK command."""

    return RobotCommand(
        command_type=CommandType.WALK,
        values=np.array(
            [forward, lateral, yaw],
            dtype=float,
        ),
    )


def turn_command(yaw_rate: float) -> RobotCommand:
    """Create a TURN command."""

    return RobotCommand(
        command_type=CommandType.TURN,
        values=np.array(
            [yaw_rate],
            dtype=float,
        ),
    )
