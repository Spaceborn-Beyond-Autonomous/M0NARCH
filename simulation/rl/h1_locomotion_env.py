"""
بيئة المشي للروبوت H1 مع عقبات (حفرة + صندوق)
=================================================
H1 Obstacle Course Locomotion Environment

المراحل (Curriculum):
  Stage 0: مشي على أرض مسطحة - الروبوت يتعلم يمشي مستقيم ومفرود
  Stage 1: حفرة في الطريق - الروبوت يتعلم يتفاداها أو يعديها
  Stage 2: صندوق (بوكس) مربع قدامه - يتعلم يتفاداه

كل مرحلة الريوارد بتزيد لما يعدي العقبة بنجاح.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class H1ObstacleCourseEnv(gym.Env):
    """
    بيئة Gymnasium مخصصة لروبوت Unitree H1.
    
    الروبوت بيتعلم:
      1. يمشي مفرود ومستقيم (الجسم الفوقاني مش راخي)
      2. يتفادى حفرة في الأرض
      3. يتفادى صندوق مربع قدامه
    
    Features:
      - Frameskip = 10 (50Hz control like real robots)
      - Torso orientation in observations (so policy SEES body tilt)
      - Strong upright posture reward (upper body stays straight)
      - Foot contact alternation reward (natural walking gait)
      - Joint-at-limit penalty (prevents hyperextension)
      - Curriculum stages with increasing difficulty
      - Progressive reward scaling per obstacle cleared
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ── ثوابت البيئة ──
    FRAMESKIP = 10           # عدد خطوات الفيزياء لكل action (50Hz control)
    TARGET_SPEED = 0.6       # سرعة المشي المطلوبة (m/s) - أبطأ = أسهل للتعلم
    MAX_EPISODE_STEPS = 1000 # أقصى عدد خطوات في الحلقة الواحدة

    # مواقع العقبات على المحور X
    PIT_START_X = 2.5   # بداية الحفرة
    PIT_END_X = 3.5     # نهاية الحفرة
    BOX_X = 6.0         # مكان الصندوق
    FINISH_X = 9.0      # خط النهاية

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        use_obstacles: bool = True,      # هل نستخدم بيئة العقبات ولا أرض مسطحة؟
        curriculum_stage: int = 0,        # مرحلة التعلم (0=مشي, 1=حفرة, 2=بوكس)
        **kwargs,
    ):
        super().__init__()

        # ── اختيار ملف المشهد ──
        if xml_path is None:
            models_dir = Path(__file__).resolve().parents[2] / "models" / "unitree_h1"
            if use_obstacles:
                xml_path = str(models_dir / "scene_obstacles.xml")
            else:
                xml_path = str(models_dir / "scene.xml")

        if not os.path.exists(xml_path):
            # fallback للمشهد الأساسي لو ملف العقبات مش موجود
            xml_path = str(models_dir / "scene.xml")

        self.render_mode = render_mode
        self.use_obstacles = use_obstacles
        self.curriculum_stage = curriculum_stage

        # ── تحميل موديل MuJoCo ──
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ── Action Space: 19 موتور ──
        # القيم من -1 لـ 1 وبنحولها لـ torque حقيقي
        self.num_actuators = self.model.nu  # 19
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32
        )

        # حفظ حدود الـ torque لكل موتور
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # ── Observation Space ──
        # المعلومات اللي الـ policy بتشوفها:
        #   - z_height (1): ارتفاع الحوض
        #   - quaternion (4): اتجاه الجسم  
        #   - joint_angles (19): زوايا المفاصل
        #   - joint_velocities (19+6=25): سرعات كل حاجة
        #   - torso_tilt (3): ميل الجزء العلوي من الجسم
        #   - feet_contact (2): هل الرجل اليمين/الشمال لامسة الأرض
        #   - obstacle_distance (2): المسافة للحفرة والبوكس
        # المجموع = 1 + 4 + 19 + 25 + 3 + 2 + 2 = 56
        obs_dim = 56
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── IDs المهمة ──
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # أرقام الـ geom بتاعة القدم (اللي مسموح تلمس الأرض)
        # بنكتشفها ديناميكياً من اسم الـ body عشان تشتغل مع أي scene
        self.foot_geom_ids = set()
        self.left_foot_geoms = set()
        self.right_foot_geoms = set()
        
        left_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
        right_ankle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
        
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if self.model.geom_contype[i] > 0 or self.model.geom_conaffinity[i] > 0:
                if body_id == left_ankle_id:
                    self.left_foot_geoms.add(i)
                    self.foot_geom_ids.add(i)
                elif body_id == right_ankle_id:
                    self.right_foot_geoms.add(i)
                    self.foot_geom_ids.add(i)

        # أي geom مرتبط بالـ worldbody (bodyid == 0) هو أرض أو عقبة في البيئة
        self.ground_geom_ids = set()
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == 0:
                self.ground_geom_ids.add(i)

        # تحديد الـ geom الخاص بالصندوق
        self.obstacle_box_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_box"
        )

        # الـ geoms بتاعة الروبوت (bodyid > 0) ما عدا الأقدام
        # لو أي جزء منها لمس الأرض أو العقبة = سقطة
        self.non_foot_robot_geoms = set()
        for i in range(self.model.ngeom):
            if (self.model.geom_contype[i] > 0 or self.model.geom_conaffinity[i] > 0):
                if self.model.geom_bodyid[i] > 0 and i not in self.foot_geom_ids:
                    self.non_foot_robot_geoms.add(i)

        # حفظ حالة الخطوات
        self._step_count = 0
        self._prev_left_contact = False
        self._prev_right_contact = False
        self._last_foot_switch = 0  # عداد من آخر تبديل قدم

        # أقصى x وصله الروبوت (لحساب مكافأة التقدم)
        self._max_x = 0.0
        # هل عدى العقبات؟
        self._passed_pit = False
        self._passed_box = False

        self.viewer = None

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        تحويل الـ action من [-1, 1] لقيم torque حقيقية
        بنستخدم تحويل خطي بسيط
        """
        low = self.ctrl_ranges[:, 0]
        high = self.ctrl_ranges[:, 1]
        scaled = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled, low, high)

    def _get_foot_contacts(self) -> Tuple[bool, bool]:
        """
        هل القدم الشمال/اليمين لامسة أي سطح أرضي؟
        مهم للمشي الطبيعي - لازم الأقدام تتبدل
        """
        left_contact = False
        right_contact = False
        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            g1, g2 = con.geom1, con.geom2
            # واحد من الـ geoms لازم يكون أرض/عقبة والتاني قدم
            for ground, foot in [(g1, g2), (g2, g1)]:
                if ground in self.ground_geom_ids:
                    if foot in self.left_foot_geoms:
                        left_contact = True
                    elif foot in self.right_foot_geoms:
                        right_contact = True
        return left_contact, right_contact

    def _check_body_collision(self) -> bool:
        """
        هل أي جزء من الجسم (غير القدم) لمس الأرض/عقبة؟
        لو أيوا = الروبوت وقع أو خبط
        """
        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            g1, g2 = con.geom1, con.geom2
            for ground, body_geom in [(g1, g2), (g2, g1)]:
                if ground in self.ground_geom_ids and body_geom in self.non_foot_robot_geoms:
                    return True
        return False

    def _get_obs(self) -> np.ndarray:
        """
        بناء الـ observation vector (56 عنصر)
        
        المعلومات دي هي اللي الـ neural network بتشوفها عشان تاخد قرار.
        كل ما المعلومات أكتر وأدق، كل ما الروبوت يتعلم أحسن.
        """
        # 1. ارتفاع الحوض (1)
        z_height = np.array([self.data.qpos[2]], dtype=np.float32)

        # 2. اتجاه الجسم - quaternion (4)
        quat = self.data.qpos[3:7].copy().astype(np.float32)

        # 3. زوايا كل المفاصل (19)
        joint_pos = self.data.qpos[7:].copy().astype(np.float32)

        # 4. كل السرعات (25 = 6 root + 19 joints)
        velocities = self.data.qvel.copy().astype(np.float32)

        # 5. ميل الجسم العلوي - أهم حاجة للمشي المستقيم! (3)
        # row 2 of rotation matrix = world-Z axis in body frame
        torso_rot = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_z_axis = torso_rot[2, :].astype(np.float32)  # [zx, zy, zz]

        # 6. حالة لمس القدمين (2)
        left_c, right_c = self._get_foot_contacts()
        feet = np.array([float(left_c), float(right_c)], dtype=np.float32)

        # 7. مسافة العقبات النسبية (2)
        robot_x = float(self.data.qpos[0])
        dist_to_pit = np.clip((self.PIT_START_X - robot_x) / 5.0, -1.0, 1.0)
        dist_to_box = np.clip((self.BOX_X - robot_x) / 10.0, -1.0, 1.0)
        obstacles = np.array([dist_to_pit, dist_to_box], dtype=np.float32)

        obs = np.concatenate([
            z_height,      # 1
            quat,          # 4
            joint_pos,     # 19
            velocities,    # 25
            torso_z_axis,  # 3
            feet,          # 2
            obstacles,     # 2
        ])  # Total = 56

        return obs

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        إعادة تعيين البيئة - الروبوت يرجع لوضع البداية
        بنضيف شوية noise صغيرة عشان يتعلم يتعامل مع أي وضع
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # تحميل وضع البداية من الـ keyframe
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # noise صغير للاستكشاف
        qpos_noise = self.np_random.uniform(-0.005, 0.005, size=self.model.nq)
        qpos_noise[:2] = 0  # مفيش noise في x,y
        qpos_noise[2] = 0   # مفيش noise في الارتفاع
        qpos_noise[3:7] = 0 # مفيش noise في الـ quaternion

        qvel_noise = self.np_random.uniform(-0.005, 0.005, size=self.model.nv)
        qvel_noise[:6] = 0  # مفيش noise في حركة الـ root

        self.data.qpos[:] += qpos_noise
        self.data.qvel[:] = qvel_noise

        mujoco.mj_forward(self.model, self.data)

        # إعادة تعيين المتغيرات الداخلية
        self._step_count = 0
        self._prev_left_contact = False
        self._prev_right_contact = False
        self._last_foot_switch = 0
        self._max_x = float(self.data.qpos[0])
        self._passed_pit = False
        self._passed_box = False

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        خطوة واحدة في البيئة.
        
        1. نطبق الـ action (torque على الموتورات)
        2. نشغل الفيزياء FRAMESKIP مرة (50Hz control)
        3. نحسب الـ reward
        4. نشوف لو الروبوت وقع أو خلص
        """
        self._step_count += 1
        x_before = float(self.data.qpos[0])

        # ──────────────────────────────────────────
        # 1. تطبيق الـ Action + Frameskip
        # ──────────────────────────────────────────
        scaled_torque = self._scale_action(action)
        self.data.ctrl[:] = scaled_torque
        for _ in range(self.FRAMESKIP):
            mujoco.mj_step(self.model, self.data)

        # ──────────────────────────────────────────
        # 2. قراءة حالة الروبوت بعد الخطوة
        # ──────────────────────────────────────────
        x_after = float(self.data.qpos[0])
        z_height = float(self.data.qpos[2])
        dt = self.model.opt.timestep * self.FRAMESKIP  # الوقت الفعلي للخطوة
        ctrl_cost = 0.01 * float(np.sum(np.square(action)))

        # سرعة الحركة للأمام
        x_vel = (x_after - x_before) / dt

        # ميل الجسم العلوي
        torso_rot = self.data.xmat[self.torso_body_id].reshape(3, 3)
        torso_upright = float(torso_rot[2, 2])  # 1.0 = مستقيم تماماً

        pelvis_rot = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        pelvis_upright = float(pelvis_rot[2, 2])

        # لمس القدمين
        left_contact, right_contact = self._get_foot_contacts()

        # هل جزء غير القدم لمس حاجة؟
        body_collision = self._check_body_collision()

        # ──────────────────────────────────────────
        # 3. شروط الصحة (هل الروبوت لسه واقف؟)
        # ──────────────────────────────────────────
        # تشديد شرط الارتفاع مع السماح بانثناء الركبة الطبيعي أثناء الخطو (الحد الأدنى 0.60m)
        height_ok = 0.60 <= z_height <= 1.25
        upright_ok = torso_upright > 0.75   # استقامة الجزء العلوي
        pelvis_ok = pelvis_upright > 0.75

        # فحص السقوط في الحفرة (بين x=2.0 و x=3.2 و y في المسار الأوسط)
        robot_y = float(self.data.qpos[1])
        fell_in_pit = (2.0 <= x_after <= 3.2 and abs(robot_y) < 0.75 and z_height < 0.1)
        fell_down = z_height < 0.4

        is_healthy = height_ok and upright_ok and pelvis_ok and (not body_collision) and (not fell_in_pit) and (not fell_down)

        # ──────────────────────────────────────────
        # 4. حساب الـ Reward (أهم جزء!)
        # ──────────────────────────────────────────

        if not is_healthy:
            # ── عقوبة السقوط ──
            reward = -10.0
        else:
            # ─── أ) مكافأة البقاء واقف (alive bonus) ───
            alive_bonus = 2.0

            # ─── ب) مكافأة سرعة المشي ───
            vel_error = x_vel - self.TARGET_SPEED
            vel_reward = 1.5 * float(np.exp(-2.0 * vel_error ** 2))

            # ─── ج) مكافأة الاستقامة الكاملة ومنع الرخاوة (الجسم مفرود!) ───
            # مكافأة استقامة الجذع والحوض
            upright_reward = 4.0 * float(np.clip(torso_upright, 0, 1))
            pelvis_reward = 2.0 * float(np.clip(pelvis_upright, 0, 1))

            # مكافأة الحفاظ على الارتفاع الطبيعي (0.95m - 1.0m)
            height_error = abs(z_height - 0.98)
            height_reward = 3.0 * float(np.exp(-20.0 * (height_error ** 2)))

            # عقوبة الرخاوة أو الانحناء لأسفل (Sagging Penalty)
            sagging_penalty = 0.0
            if z_height < 0.85:
                sagging_penalty = 5.0 * (0.85 - z_height)

            # ─── د) عقوبة استخدام الطاقة ───
            ctrl_cost = 0.01 * float(np.sum(np.square(action)))

            # ─── هـ) عقوبة الانحراف الجانبي ───
            lateral_penalty = 0.3 * abs(float(self.data.qvel[1]))

            # ─── و) مكافأة تبادل الخطوات (مشي طبيعي) ───
            gait_reward = 0.0
            if left_contact != right_contact:
                gait_reward = 0.5
                if (left_contact != self._prev_left_contact or 
                    right_contact != self._prev_right_contact):
                    gait_reward = 1.0
                    self._last_foot_switch = 0
            
            if left_contact and right_contact:
                self._last_foot_switch += 1
                if self._last_foot_switch > 20:
                    gait_reward = -0.3

            # ─── ز) عقوبة المفاصل عند الحدود ───
            joint_angles = self.data.qpos[7:]
            joint_limits_low = self.model.jnt_range[1:, 0]
            joint_limits_high = self.model.jnt_range[1:, 1]
            joint_range = joint_limits_high - joint_limits_low + 1e-8
            margin = 0.1
            at_lower = np.maximum(0, joint_limits_low + margin * joint_range - joint_angles)
            at_upper = np.maximum(0, joint_angles - (joint_limits_high - margin * joint_range))
            joint_limit_penalty = 0.5 * float(np.sum(at_lower + at_upper))

            # ─── ح) عقوبة الاهتزاز ───
            angular_vel_penalty = 0.002 * float(np.sum(np.square(self.data.qvel[6:])))

            # ─── ط) مكافأة التقدم ───
            progress_reward = 0.0
            if x_after > self._max_x:
                progress_reward = 2.5 * (x_after - self._max_x)
                self._max_x = x_after

            # ─── ي) مكافأة عبور العقبات التصاعدية ───
            obstacle_bonus = 0.0
            if self.use_obstacles:
                # 1. تخطي الحفرة بنجاح
                if not self._passed_pit and x_after > 3.2 and z_height > 0.7:
                    self._passed_pit = True
                    obstacle_bonus += 80.0

                # 2. تخطي الصندوق بنجاح
                if not self._passed_box and x_after > 5.8 and z_height > 0.7:
                    self._passed_box = True
                    obstacle_bonus += 120.0

                # 3. الوصول لخط النهاية
                if x_after > 7.5 and z_height > 0.7:
                    obstacle_bonus += 200.0

            # ─── المجموع النهائي ───
            reward = (
                alive_bonus
                + vel_reward
                + upright_reward
                + pelvis_reward
                + height_reward
                + gait_reward
                + progress_reward
                + obstacle_bonus
                - ctrl_cost
                - lateral_penalty
                - sagging_penalty
                - joint_limit_penalty
                - angular_vel_penalty
            )

        # تحديث حالة القدمين للخطوة الجاية
        self._prev_left_contact = left_contact
        self._prev_right_contact = right_contact

        # ──────────────────────────────────────────
        # 5. نهاية الحلقة
        # ──────────────────────────────────────────
        terminated = not is_healthy
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        if self.render_mode == "human":
            self.render()

        info = {
            "x_position": x_after,
            "x_velocity": x_vel,
            "z_height": z_height,
            "torso_upright": torso_upright,
            "is_healthy": is_healthy,
            "left_foot": left_contact,
            "right_foot": right_contact,
            "passed_pit": self._passed_pit,
            "passed_box": self._passed_box,
            "step_count": self._step_count,
            "ctrl_cost": ctrl_cost,
            "reward": reward,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """عرض المحاكاة في نافذة MuJoCo"""
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ─── ملخص الفروقات عن النسخة القديمة ───
# 1. Frameskip = 10 (50Hz بدل 500Hz)
# 2. الـ observation فيها معلومات أكتر (ميل الجسم + حالة القدمين + مسافة العقبات)
# 3. مكافأة الاستقامة عالية (الجسم لازم يبقى مفرود)
# 4. مكافأة تبادل الخطوات (مشي طبيعي)
# 5. عقبات (حفرة + بوكس) مع curriculum learning
# 6. مكافأة تقدمية (كل ما يتقدم أكتر، الريوارد أعلى)

# Backward compatibility alias
H1LocomotionEnv = H1ObstacleCourseEnv
