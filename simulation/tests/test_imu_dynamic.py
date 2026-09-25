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

    state_initial = backend.reset()

    initial_accel = state_initial.imu_acceleration.copy()
    initial_gyro = state_initial.imu_angular_velocity.copy()

    print("Dynamic IMU validation")
    print("----------------------")
    print("Initial acceleration:", initial_accel)
    print("Initial gyro:", initial_gyro)

    # Advance the physics simulation.
    measurements = []

    for _ in range(100):
        state = backend.step()
        measurements.append(
            np.concatenate(
                [
                    state.imu_acceleration,
                    state.imu_angular_velocity,
                ]
            )
        )

    measurements = np.asarray(measurements)

    accel = measurements[:, :3]
    gyro = measurements[:, 3:]

    print("Acceleration range:")
    print("  min:", accel.min(axis=0))
    print("  max:", accel.max(axis=0))

    print("Gyro range:")
    print("  min:", gyro.min(axis=0))
    print("  max:", gyro.max(axis=0))

    # All sensor values must remain finite.
    assert np.all(np.isfinite(accel))
    assert np.all(np.isfinite(gyro))

    print("----------------------")
    print("Dynamic IMU validation PASSED")

    backend.close()


if __name__ == "__main__":
    main()
