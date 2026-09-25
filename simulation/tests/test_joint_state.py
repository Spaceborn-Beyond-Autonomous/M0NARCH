from pathlib import Path
import numpy as np

from simulation.api.mujoco_backend import MuJoCoBackend

XML = str(
    Path(__file__).resolve().parents[2]
    / "models" / "unitree_h1" / "scene.xml"
)

EXPECTED_JOINTS = 19


def main():
    backend = MuJoCoBackend(XML)
    state = backend.reset()

    print("Joint-state validation")
    print("----------------------")
    print("Joint count:", state.num_joints)
    print("Position shape:", state.joint_position.shape)
    print("Velocity shape:", state.joint_velocity.shape)
    print("Effort shape:", state.joint_effort.shape)
    print("Joint names:", len(state.joint_names))

    assert state.num_joints == EXPECTED_JOINTS
    assert state.joint_position.shape == (EXPECTED_JOINTS,)
    assert state.joint_velocity.shape == (EXPECTED_JOINTS,)
    assert state.joint_effort.shape == (EXPECTED_JOINTS,)
    assert len(state.joint_names) == EXPECTED_JOINTS

    assert np.all(np.isfinite(state.joint_position))
    assert np.all(np.isfinite(state.joint_velocity))
    assert np.all(np.isfinite(state.joint_effort))

    print("----------------------")
    print("Joint-state validation PASSED")

    backend.close()


if __name__ == "__main__":
    main()
