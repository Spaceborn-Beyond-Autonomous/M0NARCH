# H1 Robot — Physics Simulation

Physics and simulation module for the **Unitree H1 humanoid robot**.

This branch contains the validated MuJoCo physics model, V16.1 whole-body balance controller, simulation API, disturbance tests, terrain tests, dynamic weight-shift benchmark, physics randomization, robustness benchmarks, and supporting documentation.

**Branch:** `Physics-simulation`

---

## 1. Repository Structure

```text
H1_robot/
├── models/
│   └── unitree_h1/
│       ├── scene.xml
│       ├── h1.xml
│       ├── assets/
│       └── LICENSE
│
├── simulation/
│   ├── api/
│   │   ├── commands.py
│   │   ├── mujoco_backend.py
│   │   ├── robot_state.py
│   │   └── simulation_backend.py
│   │
│   ├── physics/
│   │   ├── h1_balance_v16_1.py
│   │   ├── h1_balance_v16_1_headless.py
│   │   ├── h1_balance_v16_1_validated.py
│   │   ├── h1_balance_v16_1_disturbance_test.py
│   │   ├── h1_dynamic_weight_shift_benchmark.py
│   │   ├── h1_terrain_physics_benchmark.py
│   │   ├── h1_physics_disturbance_robustness.py
│   │   ├── h1_physics_randomization_benchmark.py
│   │   ├── h1_physics_randomization_runner.py
│   │   └── benchmark CSV files
│   │
│   └── tests/
│       └── test_simulation_api.py
│
├── docs/
│   ├── SIMULATION_API.md
│   └── H1_PHYSICS_ROBUSTNESS_REPORT.md
│
└── README.md
```

---

# 2. Requirements

Tested environment:

```text
Ubuntu 22.04
Python 3.10
MuJoCo 3.12.0
```

The physics module requires Python packages used by the simulation and controller, including:

```text
mujoco
numpy
osqp
scipy
```

Use the existing project virtual environment if available.

---

# 3. Setup

Clone the repository and switch to the physics branch:

```bash
git clone -b Physics-simulation https://github.com/Spaceborn-Beyond-Autonomous/H1_robot.git
cd H1_robot
```

Activate the Python environment:

```bash
source ~/humanoid_robot/.venv/bin/activate
```

Check MuJoCo:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Expected:

```text
3.12.0
```

If dependencies are missing, install them in the active virtual environment:

```bash
pip install mujoco numpy scipy osqp
```

---

# 4. Model Location

The H1 MuJoCo model is stored inside the repository:

```text
models/unitree_h1/
```

Main scene:

```text
models/unitree_h1/scene.xml
```

The physics scripts automatically locate the model relative to the repository:

```python
Path(__file__).resolve().parents[2] / "models" / "unitree_h1" / "scene.xml"
```

Therefore, the scripts do not depend on the user's current working directory.

---

# 5. Simulation API Test

Before running the physics controllers, validate the simulation API:

```bash
cd ~/H1_robot_physics
source ~/humanoid_robot/.venv/bin/activate

python -m simulation.tests.test_simulation_api
```

Expected:

```text
Simulation API validation PASSED
```

The API test verifies:

* MuJoCo model initialization
* H1 body count
* joint count
* DOF count
* actuator count
* state reset
* joint state handling
* command interface
* STOP command behavior

---

# 6. V16.1 Validated Balance Controller

The validated headless controller is:

```bash
python simulation/physics/h1_balance_v16_1_validated.py
```

This is the main regression test for the physics module.

Expected behavior:

```text
state=OK
contacts=8
V16.1 TERMINATED
```

Typical validated baseline:

```text
Simulation duration: ~60 seconds
H1 height:           ~0.975 m
COM:                 ~(+0.029,+0.001,+0.945)
Tilt:                ~0.4 deg
Contacts:            8
QP_Fz:               ~503 N
Maximum torque:      ~42 N·m
Safety stop:         none
```

A successful run ends with:

```text
V16.1 TERMINATED
```

---

# 7. Visual MuJoCo Simulation

To run the visual version:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 GLFW_PLATFORM=x11 \
python simulation/physics/h1_balance_v16_1.py
```

This launches the MuJoCo visualizer and runs the H1 balance simulation.

If the graphical window is not required, use the headless version instead.

---

# 8. Headless Balance Simulation

Run:

```bash
python simulation/physics/h1_balance_v16_1_headless.py
```

Use this when:

* running through SSH
* running without a graphical display
* performing automated tests
* collecting simulation data
* running repeated experiments

---

# 9. Disturbance Test

Run the validated disturbance test:

```bash
python simulation/physics/h1_balance_v16_1_disturbance_test.py
```

This evaluates the standing controller under an external disturbance.

---

# 10. Disturbance Robustness

Run:

```bash
python simulation/physics/h1_physics_disturbance_robustness.py
```

This evaluates the physics/controller response over the defined disturbance conditions and safety boundaries.

---

# 11. Dynamic Weight-Shift Benchmark

Run:

```bash
python simulation/physics/h1_dynamic_weight_shift_benchmark.py
```

This tests the H1 physics/controller response while changing the load distribution.

---

# 12. Terrain Physics Benchmark

Run:

```bash
python simulation/physics/h1_terrain_physics_benchmark.py
```

This evaluates standing/balance behavior under the configured terrain physics conditions.

---

# 13. Physics Robustness Benchmark

Run:

```bash
python simulation/physics/h1_robustness_benchmark.py
```

Results are stored in:

```text
simulation/physics/h1_robustness_benchmark.csv
```

---

# 14. Physics Randomization

Randomized physics parameters currently include:

```text
Friction:  0.60 – 1.00
Gravity:   9.50 – 10.10 m/s²
Damping:   0.98 – 1.02
```

Run the randomization benchmark:

```bash
python simulation/physics/h1_physics_randomization_benchmark.py
```

The randomization runner is:

```bash
python simulation/physics/h1_physics_randomization_runner.py
```

Results:

```text
simulation/physics/h1_physics_randomization_results.csv
```

The validated benchmark completed:

```text
10/10 episodes passed
0 safety stops
0 failures
```

---

# 15. Running the Main Validation Sequence

For a normal development check, run:

### Step 1 — API

```bash
python -m simulation.tests.test_simulation_api
```

### Step 2 — V16.1 regression

```bash
python simulation/physics/h1_balance_v16_1_validated.py
```

### Step 3 — Visual simulation

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
GLFW_PLATFORM=x11 \
python simulation/physics/h1_balance_v16_1.py
```

### Step 4 — Randomization benchmark

```bash
python simulation/physics/h1_physics_randomization_benchmark.py
```

For a quick check, Steps 1 and 2 are normally sufficient.

---

# 16. Simulation API

The simulation API provides a backend interface between the MuJoCo simulation and higher-level software.

Main components:

```text
simulation/api/commands.py
simulation/api/mujoco_backend.py
simulation/api/robot_state.py
simulation/api/simulation_backend.py
```

The API provides concepts such as:

```text
RobotCommand
CommandType
RobotState
SimulationBackend
MuJoCoBackend
```

Detailed API documentation:

```text
docs/SIMULATION_API.md
```

---

# 17. Integration With Other Team Modules

The physics module is intended to act as the **simulation/physics layer** rather than replacing ROS2, locomotion, perception, or kinematics modules.

Recommended architecture:

```text
                 ┌──────────────────────┐
                 │      ROS 2 Layer     │
                 │ topics / services    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Controller / Deryk   │
                 │ high-level commands  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Physics / Simulation │
                 │      MuJoCo H1       │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Robot State                   Sensor State
       pose / joints                 IMU / contacts
       velocities                    simulation data
```

The physics module should receive commands from higher-level control software and return simulated robot state/sensor information.

---

# 18. Integration With ROS2

The physics branch does not require ROS2 for the standalone MuJoCo validation.

ROS2 integration can be added above the simulation API.

Recommended direction:

```text
ROS2 node
   │
   │ command
   ▼
Simulation API
   │
   ▼
MuJoCo
   │
   │ state
   ▼
Simulation API
   │
   ▼
ROS2 topics
```

For example:

```text
/controller/command
        │
        ▼
   MuJoCo backend
        │
        ▼
/joint_states
/imu/data
/robot_state
/contact_state
```

The exact topic names should be agreed upon by the ROS2/control team before integration.

---

# 19. Integration With Locomotion / RL

The locomotion or RL module should not directly modify the MuJoCo model.

Recommended interface:

```text
Locomotion / RL
       │
       │ desired joint targets / actions
       ▼
Controller interface
       │
       ▼
Physics simulation
       │
       │ observations
       ▼
Locomotion / RL
```

Typical observations can include:

```text
joint positions
joint velocities
base position
base orientation
base velocity
contact state
IMU data
```

The physics module remains responsible for:

```text
gravity
contacts
collision
dynamics
actuation
friction
damping
terrain
sensor simulation
```

---

# 20. Integration With Kinematics / MoveIt2

The kinematics/MoveIt2 module can generate target joint configurations or end-effector trajectories.

Recommended flow:

```text
MoveIt2 / Kinematics
          │
          ▼
Target joint configuration
          │
          ▼
Controller
          │
          ▼
MuJoCo Physics
```

The physics simulation should be used to validate whether generated trajectories are dynamically feasible.

The physics module should not duplicate MoveIt2's planning functionality.

---

# 21. Integration With Sensors / Perception

The sensor/perception module can consume simulated measurements from the physics layer.

Example:

```text
                 MuJoCo
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       IMU       Contacts    Joint State
        │          │          │
        └──────────┼──────────┘
                   ▼
             Sensor Layer
                   │
                   ▼
              Perception
```

Sensor noise/randomization can be added later without changing the core H1 dynamics model.

---

# 22. Important Integration Rule

Do **not** modify the validated V16.1 controller simply to connect another module.

Prefer an interface around it:

```text
Other module
     │
     ▼
Integration interface
     │
     ▼
Validated physics/controller
```

This keeps the validated baseline reproducible.

If an integration change requires modifying the physics controller, create a separate commit and validate it against the baseline again.

---

# 23. Recommended Team Workflow

Each team member should work on their own branch.

Example:

```bash
git checkout -b my-feature
```

Do not directly develop on `main`.

For the physics work:

```text
Physics-simulation
```

contains Anand's validated physics implementation.

After another module needs integration:

```text
Physics-simulation
        │
        ▼
Integration branch / main
```

The stable physics branch should remain reproducible.

---

# 24. Updating From Main

Before integrating changes from the team:

```bash
git fetch origin
```

Check branches:

```bash
git branch -a
```

Update the local physics branch:

```bash
git pull origin Physics-simulation
```

Do not use:

```bash
git push origin main
```

from this branch.

---

# 25. Checking Your Branch

Check current branch:

```bash
git branch --show-current
```

Expected:

```text
Physics-simulation
```

Check repository state:

```bash
git status
```

Check recent commits:

```bash
git log --oneline --max-count=5
```

---

# 26. Commit Workflow

After making changes:

```bash
git status
```

Review:

```bash
git diff
```

Stage:

```bash
git add <files>
```

Commit:

```bash
git commit -m "Describe the change"
```

Push:

```bash
git push origin Physics-simulation
```

---

# 27. Reproducibility Check

Before submitting physics changes for integration, run:

```bash
python -m simulation.tests.test_simulation_api
```

Then:

```bash
python simulation/physics/h1_balance_v16_1_validated.py
```

The validated baseline should remain approximately:

```text
z       ≈ 0.975 m
tilt    ≈ 0.4 deg
contacts = 8
QP_Fz   ≈ 503 N
tau_max ≈ 42 N·m
state   = OK
```

If these values change significantly, investigate the change before merging.

---

# 28. Current Validated Physics Baseline

The current baseline consists of:

* Unitree H1 MuJoCo model
* 21 bodies
* 20 joints
* 25 velocity DOFs
* 19 actuators
* ~51.4 kg model mass
* V16.1 contact-aware whole-body balance controller
* OSQP contact-force QP
* contact Jacobians
* floating-base inverse dynamics
* actuator torque saturation
* standing stability validation
* disturbance testing
* dynamic weight-shift testing
* terrain physics testing
* physics robustness testing
* physics parameter randomization

The baseline should be treated as the reference physics implementation for integration.

---

# 29. Documentation

Additional documentation:

```text
docs/SIMULATION_API.md
docs/H1_PHYSICS_ROBUSTNESS_REPORT.md
```

The model's original documentation and license are preserved in:

```text
models/unitree_h1/README.md
models/unitree_h1/LICENSE
models/unitree_h1/CHANGELOG.md
```

---

# 30. Quick Start

For a teammate who just wants to verify the physics module:

```bash
git clone -b Physics-simulation https://github.com/Spaceborn-Beyond-Autonomous/H1_robot.git
cd H1_robot

source ~/humanoid_robot/.venv/bin/activate

python -c "import mujoco; print(mujoco.__version__)"

python -m simulation.tests.test_simulation_api

python simulation/physics/h1_balance_v16_1_validated.py
```

For the visual simulation:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
GLFW_PLATFORM=x11 \
python simulation/physics/h1_balance_v16_1.py
```

If the API test passes and the V16.1 regression terminates normally with stable values, the physics environment is ready for integration with the other H1 modules.
