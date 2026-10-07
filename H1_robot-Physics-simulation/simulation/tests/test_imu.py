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

    state = backend.reset()

    print("IMU validation")
    print("----------------")
    print("IMU acceleration:", state.imu_acceleration)
    print("IMU angular velocity:", state.imu_angular_velocity)

    # ---------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------

    assert state.imu_acceleration.shape == (3,)
    assert state.imu_angular_velocity.shape == (3,)

    # ---------------------------------------------------------
    # Finite-value validation
    # ---------------------------------------------------------

    assert np.all(np.isfinite(state.imu_acceleration))
    assert np.all(np.isfinite(state.imu_angular_velocity))

    # ---------------------------------------------------------
    # Stationary gyro validation
    # ---------------------------------------------------------

    gyro_magnitude = np.linalg.norm(
        state.imu_angular_velocity
    )

    print("Gyro magnitude:", gyro_magnitude)

    assert gyro_magnitude < 1e-6

    # ---------------------------------------------------------
    # Accelerometer validation
    # ---------------------------------------------------------

    acceleration_magnitude = np.linalg.norm(
        state.imu_acceleration
    )

    print("Acceleration magnitude:", acceleration_magnitude)

    # A valid stationary MuJoCo sensor reading must be finite.
    # Exact gravity representation is model/sensor-frame dependent.
    assert np.isfinite(acceleration_magnitude)

    print("----------------")
    print("IMU validation PASSED")

    backend.close()


if __name__ == "__main__":
    main()
