from dataclasses import dataclass, field
from typing import List

import numpy as np

from simulation.api.robot_state import RobotState


@dataclass
class PhysicsValidationResult:
    """Result of a physics-state validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.valid


@dataclass
class PhysicsValidator:
    """
    Product-level validation of a simulated RobotState.

    This validator checks the physical state exposed by the simulation
    API. It does not modify the simulation or controller.
    """

    expected_joints: int = 19

    min_base_height: float = 0.55
    max_base_height: float = 2.0

    max_base_tilt_deg: float = 35.0

    max_linear_velocity: float = 20.0
    max_angular_velocity: float = 50.0

    max_joint_position_abs: float = 10.0
    max_joint_velocity_abs: float = 100.0
    max_joint_effort_abs: float = 1000.0

    max_contact_force_abs: float = 10000.0

    def _error(
        self,
        errors: List[str],
        message: str,
    ) -> None:
        errors.append(message)

    def validate(
        self,
        state: RobotState,
    ) -> PhysicsValidationResult:
        """Validate one RobotState."""

        errors: List[str] = []
        warnings: List[str] = []

        # ---------------------------------------------------------
        # Position
        # ---------------------------------------------------------

        if state.position.shape != (3,):
            self._error(
                errors,
                f"Invalid position shape: {state.position.shape}",
            )
        elif not np.all(np.isfinite(state.position)):
            self._error(
                errors,
                "Position contains non-finite values.",
            )

        # ---------------------------------------------------------
        # Orientation
        # ---------------------------------------------------------

        if state.orientation.shape != (4,):
            self._error(
                errors,
                f"Invalid orientation shape: {state.orientation.shape}",
            )
        elif not np.all(np.isfinite(state.orientation)):
            self._error(
                errors,
                "Orientation contains non-finite values.",
            )
        else:
            quat_norm = np.linalg.norm(state.orientation)

            if quat_norm < 1e-8:
                self._error(
                    errors,
                    "Orientation quaternion has near-zero norm.",
                )
            elif abs(quat_norm - 1.0) > 1e-3:
                self._error(
                    errors,
                    f"Orientation quaternion is not normalized: "
                    f"norm={quat_norm:.6f}",
                )

        # ---------------------------------------------------------
        # Base height
        # ---------------------------------------------------------

        if state.position.shape == (3,):
            height = float(state.position[2])

            if height < self.min_base_height:
                self._error(
                    errors,
                    f"Base height below minimum: "
                    f"{height:.4f} m < {self.min_base_height:.4f} m",
                )

            if height > self.max_base_height:
                warnings.append(
                    f"Base height unusually high: "
                    f"{height:.4f} m > {self.max_base_height:.4f} m"
                )

        # ---------------------------------------------------------
        # Base velocity
        # ---------------------------------------------------------

        if not np.all(np.isfinite(state.linear_velocity)):
            self._error(
                errors,
                "Linear velocity contains non-finite values.",
            )
        elif np.linalg.norm(state.linear_velocity) > self.max_linear_velocity:
            self._error(
                errors,
                f"Linear velocity exceeds limit: "
                f"{np.linalg.norm(state.linear_velocity):.3f} m/s",
            )

        if not np.all(np.isfinite(state.angular_velocity)):
            self._error(
                errors,
                "Angular velocity contains non-finite values.",
            )
        elif np.linalg.norm(state.angular_velocity) > self.max_angular_velocity:
            self._error(
                errors,
                f"Angular velocity exceeds limit: "
                f"{np.linalg.norm(state.angular_velocity):.3f} rad/s",
            )

        # ---------------------------------------------------------
        # Base tilt
        # ---------------------------------------------------------

        if state.orientation.shape == (4,):
            q = state.orientation
            norm = np.linalg.norm(q)

            if norm > 1e-8:
                w, x, y, z = q / norm

                # Robot local +Z axis, expressed in world frame.
                z_world = np.array([
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ])

                tilt_rad = np.arccos(
                    np.clip(z_world[2], -1.0, 1.0)
                )

                tilt_deg = np.degrees(tilt_rad)

                if tilt_deg > self.max_base_tilt_deg:
                    self._error(
                        errors,
                        f"Base tilt exceeds limit: "
                        f"{tilt_deg:.3f} deg > "
                        f"{self.max_base_tilt_deg:.3f} deg",
                    )

        # ---------------------------------------------------------
        # Joint state
        # ---------------------------------------------------------

        if state.num_joints != self.expected_joints:
            self._error(
                errors,
                f"Unexpected joint count: "
                f"{state.num_joints} != {self.expected_joints}",
            )

        joint_arrays = (
            ("joint_position", state.joint_position, self.max_joint_position_abs),
            ("joint_velocity", state.joint_velocity, self.max_joint_velocity_abs),
            ("joint_effort", state.joint_effort, self.max_joint_effort_abs),
        )

        for name, values, limit in joint_arrays:
            if not np.all(np.isfinite(values)):
                self._error(
                    errors,
                    f"{name} contains non-finite values.",
                )
            elif values.size > 0 and np.max(np.abs(values)) > limit:
                self._error(
                    errors,
                    f"{name} exceeds configured safety limit: "
                    f"{np.max(np.abs(values)):.3f} > {limit:.3f}",
                )

        # ---------------------------------------------------------
        # Contact state
        # ---------------------------------------------------------

        if not isinstance(state.contact_states, list):
            self._error(
                errors,
                "contact_states must be a list.",
            )

        if state.contact_forces.ndim != 2 or (
            state.contact_forces.shape[1] != 3
            if state.contact_forces.ndim == 2
            else True
        ):
            self._error(
                errors,
                f"Invalid contact force shape: "
                f"{state.contact_forces.shape}",
            )

        if not np.all(np.isfinite(state.contact_forces)):
            self._error(
                errors,
                "Contact forces contain non-finite values.",
            )

        if (
            state.contact_forces.size > 0
            and np.max(np.abs(state.contact_forces))
            > self.max_contact_force_abs
        ):
            self._error(
                errors,
                "Contact force exceeds configured safety limit.",
            )

        # A contact force pointing downward at the floor is physically
        # suspicious. Small numerical noise is tolerated.
        if state.contact_forces.size > 0:
            vertical_forces = state.contact_forces[:, 2]

            for index, fz in enumerate(vertical_forces):
                if state.contact_states and index < len(state.contact_states):
                    if state.contact_states[index] and fz < -1e-3:
                        self._error(
                            errors,
                            f"Active contact {index} has negative "
                            f"vertical force: {fz:.6f} N",
                        )

        return PhysicsValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_or_raise(
        self,
        state: RobotState,
    ) -> PhysicsValidationResult:
        """Validate state and raise RuntimeError if invalid."""

        result = self.validate(state)

        if not result.valid:
            details = "\n".join(
                f"- {error}"
                for error in result.errors
            )

            raise RuntimeError(
                "Physics validation failed:\n"
                + details
            )

        return result
