# H1 Simulation - ROS 2 Humble Practical Setup

## Purpose

This document explains how to run the H1 simulation on Ubuntu 22.04 with ROS 2 Humble.

The H1 simulation core is ROS-independent. ROS 2 is an integration layer on top of the simulation.

## Architecture

    MuJoCo Physics
          |
          v
    MuJoCoBackend
          |
          v
       RobotState
          |
          v
      SensorModel
    (noise + latency)
          |
          v
     Simulation API
          |
          +------------------+
          |                  |
          v                  v
    PhysicsValidator     ROS 2 Adapter
                             |
                             v
                        ROS 2 Humble

## System requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- MuJoCo 3.12.0
- NumPy
- OSQP

## Clone the repository

    cd ~
    git clone -b Physics-simulation https://github.com/Spaceborn-Beyond-Autonomous/H1_robot.git H1_robot_physics
    cd ~/H1_robot_physics

If the repository already exists:

    cd ~/H1_robot_physics

## ROS 2 Humble

Source ROS 2 Humble:

    source /opt/ros/humble/setup.bash

Check:

    ros2 --version

The simulation core itself does not require ROS 2.

## Python environment

Check:

    python3 --version

Create:

    python3 -m venv .venv

If venv is missing:

    sudo apt update
    sudo apt install python3-venv

Activate:

    source .venv/bin/activate

## Install dependencies

    python -m pip install --upgrade pip
    python -m pip install numpy mujoco==3.12.0 osqp

Verify:

    python -c "import mujoco, numpy, osqp; print(MuJoCo:, mujoco.__version__); print(NumPy:, numpy.__version__); print(OSQP:, osqp.__version__)"

## Run the complete test suite

    cd ~/H1_robot_physics
    source .venv/bin/activate

    for test in simulation/tests/test_*.py; do
        module="${test%.py}"
        module="${module//\//.}"

        echo
        echo "========================================"
        echo "RUNNING: $test"
        echo "========================================"

        python -m "$module" || exit 1
    done

The test suite covers:

- IMU
- Dynamic IMU
- Joint state
- Contact state
- Sensor noise
- Sensor latency
- Simulation API
- Simulation pipeline
- Physics validation

## Run validated physics

    python simulation/physics/h1_balance_v16_1_validated.py

For the interactive viewer on Wayland/X11 systems:

    env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 GLFW_PLATFORM=x11 python simulation/physics/h1_balance_v16_1.py

Do not modify the validated physics/controller just to add ROS 2.

## Product-level Simulation API

Main file:

    simulation/api/simulation.py

Available operations:

- initialize()
- reset()
- step()
- get_state()
- get_raw_state()
- send_command()
- apply_disturbance()
- shutdown()

## Sensor layer

Main file:

    simulation/sensors/sensor_model.py

Supports:

- IMU noise
- Joint position noise
- Joint velocity noise
- Joint effort noise
- Contact-force noise
- Timestamp-based latency
- Deterministic random seed

## Physics validation

Main file:

    simulation/validation/physics_validator.py

The validator checks:

- finite numerical values
- quaternion validity
- base height
- base tilt
- linear/angular velocity
- joint count and limits
- contact-force shape
- contact-force direction
- contact-force magnitude

The validator does not modify the simulation.

## ROS 2 Humble integration

Recommended structure:

    H1_robot_physics/
    ├── simulation/
    └── ros2/
        └── h1_simulation_ros/
            ├── h1_simulation_ros/
            ├── launch/
            ├── config/
            ├── package.xml
            └── setup.py

Keep ROS-specific imports out of the core simulation modules.

The ROS 2 adapter should consume the existing Simulation API and RobotState.

## ROS 2 message mapping

RobotState joint data:

    sensor_msgs/JointState

IMU data:

    sensor_msgs/Imu

Robot pose:

    TF2

Contact information:

    ROS contact/state interface

The exact ROS message/interface should be selected by the ROS integration developer without changing the physics core.

## Humble and Jazzy compatibility

The simulation core should remain the same for both ROS 2 Humble and ROS 2 Jazzy.

Use separate ROS adapters if required.

Do not create separate physics implementations for Humble and Jazzy.

## Existing ROS 2 workspace

    source /opt/ros/humble/setup.bash
    cd ~/your_ros2_ws
    colcon build --symlink-install
    source install/setup.bash

The H1 simulation repository does not need to become a ROS package simply because ROS 2 is being used.

## Troubleshooting

Check Python:

    which python
    python --version

Check MuJoCo:

    python -c "import mujoco; print(mujoco.__version__)"

Expected:

    3.12.0

Check ROS:

    source /opt/ros/humble/setup.bash
    ros2 --version

If ROS integration fails, first run the simulation test suite independently.

## Development rule

Keep the layers separated:

    PHYSICS
    - MuJoCo
    - H1 V16.1 controller
    - dynamics
    - contacts
    - disturbances

    SIMULATION
    - MuJoCoBackend
    - RobotState
    - Simulation API
    - SensorModel
    - PhysicsValidator

    ROS 2
    - publishers
    - subscribers
    - TF2
    - ROS messages
    - launch files
    - parameters

A ROS 2 change should not require rewriting the validated physics.

## Current validated baseline

The current branch provides:

- H1 MuJoCo model
- 19 joints
- IMU acceleration and angular velocity
- joint position, velocity and effort
- left/right contact states
- contact forces
- configurable sensor noise
- configurable sensor latency
- Simulation API
- Simulation pipeline
- physics validation
- automated regression tests
