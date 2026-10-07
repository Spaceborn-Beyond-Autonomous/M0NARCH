from pathlib import Path

import numpy as np

from simulation.api.mujoco_backend import MuJoCoBackend
from simulation.api.simulation import Simulation
from simulation.validation.physics_validator import PhysicsValidator


XML = str(
    Path(__file__).resolve().parents[2]
    / "models"
    / "unitree_h1"
    / "scene.xml"
)


def main():
    print("Physics validator validation")
    print("----------------------------")

    backend = MuJoCoBackend(XML)
    simulation = Simulation(backend)

    state = simulation.initialize()

    validator = PhysicsValidator()

    result = validator.validate(state)

    assert result.valid, result.errors

    print("Initial state: VALID")
    print(f"Joint count: {state.num_joints}")
    print(f"Base height: {state.position[2]:.6f} m")
    print(f"Contact shape: {state.contact_forces.shape}")

    for _ in range(100):
        state = simulation.step()

    result = validator.validate(state)

    assert result.valid, result.errors

    print("After 100 steps: VALID")
    print(f"Timestamp: {state.timestamp:.6f} s")
    print(f"Base height: {state.position[2]:.6f} m")
    print(f"Active contacts: {state.active_contacts}")

    # ---------------------------------------------------------
    # Negative test: deliberately corrupt the state.
    # ---------------------------------------------------------

    invalid_state = state.copy()
    invalid_state.position[2] = 0.1

    invalid_result = validator.validate(invalid_state)

    assert not invalid_result.valid
    assert any(
        "Base height below minimum" in error
        for error in invalid_result.errors
    )

    print("Invalid-state detection: PASSED")

    simulation.shutdown()

    print("----------------------------")
    print("Physics validator validation PASSED")


if __name__ == "__main__":
    main()
