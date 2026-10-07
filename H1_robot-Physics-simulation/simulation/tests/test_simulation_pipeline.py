from pathlib import Path

import numpy as np

from simulation.api.mujoco_backend import MuJoCoBackend
from simulation.api.simulation import Simulation
from simulation.sensors.sensor_model import SensorConfig


XML = str(
    Path(__file__).resolve().parents[2]
    / "models"
    / "unitree_h1"
    / "scene.xml"
)


def main():
    print("Simulation pipeline validation")
    print("------------------------------")

    backend = MuJoCoBackend(XML)

    config = SensorConfig(
        imu_accel_std=0.01,
        imu_gyro_std=0.01,
        joint_position_std=0.001,
        joint_velocity_std=0.001,
        joint_effort_std=0.01,
        contact_force_std=0.1,
        latency_seconds=0.0,
        random_seed=42,
    )

    simulation = Simulation(
        backend=backend,
        sensor_config=config,
    )

    state = simulation.initialize()

    assert simulation.is_running()
    assert state.num_joints == 19
    assert state.imu_acceleration.shape == (3,)
    assert state.imu_angular_velocity.shape == (3,)
    assert state.contact_forces.shape == (2, 3)

    raw_state = simulation.get_raw_state()

    assert raw_state.num_joints == 19
    assert raw_state.imu_acceleration.shape == (3,)
    assert raw_state.contact_forces.shape == (2, 3)

    stepped_state = simulation.step()

    assert stepped_state.num_joints == 19
    assert stepped_state.timestamp >= state.timestamp

    print(f"Initial sensor timestamp: {state.timestamp:.6f}")
    print(f"Stepped sensor timestamp: {stepped_state.timestamp:.6f}")
    print(f"Joint count: {stepped_state.num_joints}")
    print(f"Contact force shape: {stepped_state.contact_forces.shape}")
    print(f"Raw state timestamp: {raw_state.timestamp:.6f}")

    simulation.shutdown()

    assert not simulation.is_running()

    print("------------------------------")
    print("Simulation pipeline validation PASSED")


if __name__ == "__main__":
    main()
