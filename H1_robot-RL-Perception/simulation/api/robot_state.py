from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RobotState:
    """Simulator-independent humanoid robot state."""

    timestamp: float = 0.0

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    # Quaternion convention: [w, x, y, z]
    orientation: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
    )

    linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    joint_effort: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    joint_names: List[str] = field(default_factory=list)

    imu_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    imu_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

    contact_states: List[bool] = field(default_factory=list)

    contact_forces: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )

    camera: Optional[Any] = None
    depth: Optional[Any] = None

    power: Optional[float] = None

    mode: str = "IDLE"

    errors: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_position)

    @property
    def num_contacts(self) -> int:
        return len(self.contact_states)

    @property
    def active_contacts(self) -> int:
        return sum(self.contact_states)

    @property
    def has_error(self) -> bool:
        return bool(self.errors)

    def copy(self) -> "RobotState":
        return RobotState(
            timestamp=self.timestamp,
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            joint_position=self.joint_position.copy(),
            joint_velocity=self.joint_velocity.copy(),
            joint_effort=self.joint_effort.copy(),
            joint_names=list(self.joint_names),
            imu_acceleration=self.imu_acceleration.copy(),
            imu_angular_velocity=self.imu_angular_velocity.copy(),
            contact_states=list(self.contact_states),
            contact_forces=self.contact_forces.copy(),
            camera=self.camera,
            depth=self.depth,
            power=self.power,
            mode=self.mode,
            errors=list(self.errors),
            metadata=dict(self.metadata),
        )


def empty_h1_state() -> RobotState:
    """Create an empty H1-compatible RobotState."""

    return RobotState(
        joint_position=np.zeros(19, dtype=float),
        joint_velocity=np.zeros(19, dtype=float),
        joint_effort=np.zeros(19, dtype=float),
        joint_names=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
            "torso",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        ],
    )
