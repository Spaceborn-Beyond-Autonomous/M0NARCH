from pathlib import Path

import numpy as np

from simulation.api.mujoco_backend import MuJoCoBackend


XML = str(
    Path(__file__).resolve().parents[2]
    / "models"
    / "unitree_h1"
    / "scene.xml"
)


def main():
    backend = MuJoCoBackend(XML)

    # Reset simulation.
    state = backend.reset()

    print("Contact-state validation")
    print("------------------------")
    print("Initial contact states:", state.contact_states)
    print("Initial contact forces:")
    print(state.contact_forces)

    # Let the robot settle and contacts develop.
    for _ in range(100):
        state = backend.step()

    print()
    print("After 100 simulation steps:")
    print("Contact states:", state.contact_states)
    print("Active contacts:", state.active_contacts)
    print("Contact forces:")
    print(state.contact_forces)

    # Basic structure checks.
    assert len(state.contact_states) == 2
    assert state.contact_forces.shape == (2, 3)

    # Numerical validity.
    assert np.all(
        np.isfinite(state.contact_forces)
    )

    # Both H1 feet should be in contact after settling.
    assert state.contact_states[0] is True
    assert state.contact_states[1] is True

    # World-frame vertical force should be positive.
    assert state.contact_forces[0, 2] > 0.0
    assert state.contact_forces[1, 2] > 0.0

    print("------------------------")
    print("Contact-state validation PASSED")

    backend.close()


if __name__ == "__main__":
    main()
