import numpy as np

from simulation.api.robot_state import empty_h1_state
from simulation.sensors.sensor_model import (
    SensorConfig,
    SensorModel,
)


def main():
    print("Sensor model validation")
    print("-----------------------")

    # ---------------------------------------------------------
    # Test 1: Disabled noise/latency preserves the state
    # ---------------------------------------------------------

    state = empty_h1_state()
    state.timestamp = 1.0

    state.imu_acceleration[:] = [1.0, 2.0, 3.0]
    state.imu_angular_velocity[:] = [0.1, 0.2, 0.3]
    state.joint_position[:] = 0.5
    state.joint_velocity[:] = 0.2
    state.joint_effort[:] = 1.0
    state.contact_forces = np.array(
        [
            [10.0, 0.0, 100.0],
            [5.0, 0.0, 90.0],
        ],
        dtype=float,
    )

    model = SensorModel()

    output = model.process(state)

    assert np.allclose(
        output.imu_acceleration,
        state.imu_acceleration,
    )

    assert np.allclose(
        output.joint_position,
        state.joint_position,
    )

    assert np.allclose(
        output.contact_forces,
        state.contact_forces,
    )

    # ---------------------------------------------------------
    # Test 2: Noise is deterministic with fixed seed
    # ---------------------------------------------------------

    config = SensorConfig(
        imu_accel_std=0.1,
        imu_gyro_std=0.01,
        joint_position_std=0.001,
        joint_velocity_std=0.002,
        joint_effort_std=0.1,
        contact_force_std=1.0,
        random_seed=42,
    )

    model_a = SensorModel(config)
    model_b = SensorModel(config)

    output_a = model_a.process(state)
    output_b = model_b.process(state)

    assert np.allclose(
        output_a.imu_acceleration,
        output_b.imu_acceleration,
    )

    assert np.allclose(
        output_a.contact_forces,
        output_b.contact_forces,
    )

    # Noise should actually modify at least one signal.
    assert not np.allclose(
        output_a.imu_acceleration,
        state.imu_acceleration,
    )

    # ---------------------------------------------------------
    # Test 3: Latency
    # ---------------------------------------------------------

    latency_config = SensorConfig(
        latency_seconds=0.02,
        random_seed=42,
    )

    latency_model = SensorModel(latency_config)

    states = []

    for i in range(10):
        test_state = empty_h1_state()
        test_state.timestamp = i * 0.01
        test_state.joint_position[:] = float(i)

        delayed = latency_model.process(test_state)
        states.append(delayed)

    # At 0.09 s, a 0.02 s latency should return a state
    # from approximately 0.07 s or earlier.
    assert states[-1].timestamp <= 0.07 + 1e-9

    print("Noise disabled test: PASSED")
    print("Deterministic noise test: PASSED")
    print("Latency test: PASSED")
    print("-----------------------")
    print("Sensor model validation PASSED")


if __name__ == "__main__":
    main()
