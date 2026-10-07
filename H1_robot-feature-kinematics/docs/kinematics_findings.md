# Kinematics Findings (Rawan - Day 1)

## Model Gap: Missing Hand/Wrist

The current `models/unitree_h1/h1.xml` has no wrist or hand body --
`left_elbow_link` / `right_elbow_link` are the terminal links in each
arm chain. This means:

- Current FK end-effector for each arm is the elbow, not the hand.
- Phase 5 (Manipulation, MoveIt 2, grasping) as planned in the roadmap
  requires an end-effector (gripper or dexterous hand) to be added to
  the model before reach/grasp trajectories are meaningful.

**Action needed:** confirm with Anand (model owner) whether a
hand/gripper body will be added to h1.xml, and if so, get its joint
names and offsets so the arm kinematic chains can be extended.

## Day 1 Deliverables (Complete)

- `kinematics/transforms.py`: rotation matrices, quaternion conversions,
  homogeneous transforms -- validated via round-trip tests.
- `kinematics/h1_chain.py`: full-body kinematic chain (both legs, both
  arms) with Forward Kinematics, built directly from h1.xml body
  offsets and orientations. Left/right symmetry validated.
