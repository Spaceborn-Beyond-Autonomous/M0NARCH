from pathlib import Path
import numpy as np

from simulation.api.commands import CommandType, RobotCommand
from simulation.api.mujoco_backend import MuJoCoBackend


XML = str(Path(__file__).resolve().parents[2] / "models" / "unitree_h1" / "scene.xml")


def main():
    backend = MuJoCoBackend(XML)

    # Initialization
    backend.initialize()
    assert backend.is_running()

    assert backend.model.nbody == 21
    assert backend.model.njnt == 20
    assert backend.model.nv == 25
    assert backend.model.nu == 19

    # Reset and state
    state = backend.reset()

    assert state.num_joints == 19
    assert len(state.joint_names) == 19
    assert state.mode == "IDLE"

    # STOP command
    backend.send_command(
        RobotCommand(command_type=CommandType.STOP)
    )

    assert np.allclose(
        backend.data.ctrl,
        0.0,
    )

    # POSTURE command
    posture = RobotCommand(
        command_type=CommandType.POSTURE,
        values=np.ones(19) * 0.1,
    )

    backend.send_command(posture)

    assert np.allclose(
        backend.data.ctrl,
        0.1,
    )

    # Simulation step
    state = backend.step()

    assert state.timestamp > 0.0
    assert state.mode == "POSTURE"

    backend.shutdown()

    assert not backend.is_running()

    print("Simulation API validation PASSED")


if __name__ == "__main__":
    main()
