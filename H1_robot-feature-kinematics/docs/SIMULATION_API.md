# ANSA OS — Simulation API

## 1. Purpose

The Simulation API provides a simulator-independent interface for the ANSA humanoid robotics subsystem.

It allows higher-level control, ROS2 middleware, AI, and planning modules to interact with simulation without depending directly on MuJoCo or Gazebo.

Current backend: MuJoCo

Future-compatible backends: Gazebo / Ignition and other simulators.

---

## 2. Architecture

```text
ROS2 / AI / Planner
        |
        | RobotCommand / RobotState
        v
+----------------------+
|  SimulationBackend   |
|      Interface       |
+----------+-----------+
           |
     +-----+-----+
     |           |
     v           v
+---------+   +---------+
| MuJoCo  |   | Gazebo  |
| Backend |   | Backend |
+---------+   +---------+
```

---

## 3. SimulationBackend

File:

`simulation/api/simulation_backend.py`

The abstract interface provides:

- `initialize()`
- `reset() -> RobotState`
- `step(dt=None) -> RobotState`
- `get_state() -> RobotState`
- `send_command(command)`
- `apply_disturbance(body_id, force, torque)`
- `is_running() -> bool`
- `shutdown()`
- `close()`

---

## 4. RobotState

File:

`simulation/api/robot_state.py`

RobotState provides simulator-independent robot information:

- timestamp
- base position
- base orientation
- linear velocity
- angular velocity
- joint positions
- joint velocities
- joint effort
- joint names
- IMU acceleration
- IMU angular velocity
- contact states
- contact forces
- camera/depth data
- power
- operating mode
- errors
- metadata

The H1 interface contains 19 actuated joints.

---

## 5. RobotCommand

File:

`simulation/api/commands.py`

Supported commands:

- POSTURE
- WALK
- TURN
- STOP
- REACH
- GRASP
- RELEASE
- MOVE_TO

RobotCommand provides a simulator-independent way for higher-level modules to request robot actions.

---

## 6. MuJoCoBackend

File:

`simulation/api/mujoco_backend.py`

Current model:

`models/unitree_h1/scene.xml`

Validated H1 model:

- Bodies: 21
- Joints: 20
- DOFs: 25
- Actuators: 19

The MuJoCo backend supports:

- initialization
- reset
- simulation stepping
- state extraction
- command transmission
- external disturbances
- shutdown

---

## 7. Gazebo Compatibility

A future Gazebo backend should implement the same `SimulationBackend` interface.

Higher-level modules should communicate with:

`SimulationBackend`

rather than directly with MuJoCo or Gazebo APIs.

This keeps control, planning, AI and ROS2 modules simulator-independent.

---

## 8. API Validation

Validation test:

`simulation/tests/test_simulation_api.py`

Run from the workspace root:

```bash
python3 -m simulation.tests.test_simulation_api
```

Expected result:

```text
Simulation API validation PASSED
```

The validation covers:

- backend initialization
- H1 model dimensions
- state generation
- 19-joint interface
- STOP command
- POSTURE command
- control propagation
- simulation stepping
- mode reporting
- backend shutdown

---

## 9. Current Status

The simulation abstraction has been implemented and validated.

Completed:

- RobotState
- RobotCommand
- SimulationBackend
- MuJoCoBackend
- H1 integration
- API validation
- Gazebo-compatible interface definition

---
