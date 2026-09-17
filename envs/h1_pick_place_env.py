import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
 
SCENE_PATH = "assets/h1_pick_place_scene.xml"
 
# Right-arm actuator names to control (edit after checking Section 13's actuator listing)
ARM_ACTUATORS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]
 
EE_BODY = "right_elbow_link"     # end-effector body, must match scene XML
OBJECT_BODY = "object"
TARGET_SITE = "target_site"
GRASP_RADIUS = 0.05                   # meters, distance to trigger grasp
PLACE_TOLERANCE = 0.05                # meters, distance to count as "placed"
 
 
class H1PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
 
    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._viewer = None
 
        self.act_ids = [
            self.model.actuator(name).id for name in ARM_ACTUATORS
        ]
        self.ee_body_id = self.model.body(EE_BODY).id
        self.object_body_id = self.model.body(OBJECT_BODY).id
        self.object_joint_id = self.model.joint("object_joint").id
        self.object_qpos_adr = self.model.jnt_qposadr[self.object_joint_id]
        self.target_site_id = self.model.site(TARGET_SITE).id
        self.grasp_eq_id = self.model.equality("grasp_weld").id
 
        n_act = len(self.act_ids)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)
 
        obs_dim = n_act * 2 + 3 + 3 + 3 + 1   # joint pos+vel, ee pos, obj pos, target pos, grasped flag
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
 
        self.grasped = False
        self.max_steps = 500
        self.step_count = 0
 
    def _joint_adr(self, act_id, table):
        jid = self.model.actuator(act_id).trnid[0]
        return table[jid]
 
    def _get_obs(self):
        qpos = np.array([self.data.qpos[self._joint_adr(a, self.model.jnt_qposadr)]
                          for a in self.act_ids])
        qvel = np.array([self.data.qvel[self._joint_adr(a, self.model.jnt_dofadr)]
                          for a in self.act_ids])
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        obj_pos = self.data.xpos[self.object_body_id].copy()
        target_pos = self.data.site_xpos[self.target_site_id].copy()
        grasped_flag = np.array([1.0 if self.grasped else 0.0])
        full_obs = np.concatenate([qpos, qvel, ee_pos, obj_pos, target_pos, grasped_flag])
        return full_obs.astype(np.float32)
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
 
        # Randomize object start position within reach on the table
        rng = self.np_random
        obj_x = 0.55 + rng.uniform(-0.08, 0.08)
        obj_y = 0.0 + rng.uniform(-0.12, 0.12)
        self.data.qpos[self.object_qpos_adr:self.object_qpos_adr+3] = [obj_x, obj_y, 0.45]
        self.data.qpos[self.object_qpos_adr+3:self.object_qpos_adr+7] = [1, 0, 0, 0]
 
        # Randomize target position
        tgt_x = 0.55 + rng.uniform(-0.08, 0.08)
        tgt_y = 0.25 + rng.uniform(-0.08, 0.08)
        self.model.site_pos[self.target_site_id] = [tgt_x, tgt_y, 0.45]
 
        self.data.eq_active[self.grasp_eq_id] = 0
        self.grasped = False
        self.step_count = 0
 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
 
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, act_id in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)
 
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
 
        ee_pos = self.data.xpos[self.ee_body_id]
        obj_pos = self.data.xpos[self.object_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]
 
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - target_pos)
 
        reward = 0.0
        terminated = False
 
        if not self.grasped:
            reward -= dist_ee_obj                       # encourage reaching
            if dist_ee_obj < GRASP_RADIUS:
                self.grasped = True
                self.data.eq_active[self.grasp_eq_id] = 1
                reward += 5.0                            # grasp bonus
        else:
            reward -= dist_obj_target                    # encourage carrying to target
            reward += 0.5                                # small per-step bonus
            if dist_obj_target < PLACE_TOLERANCE:
                reward += 20.0                           # place bonus
                self.data.eq_active[self.grasp_eq_id] = 0
                terminated = True
 
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}
 
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
 
    def close(self):
        if self._viewer is not None:
            self._viewer.close()
