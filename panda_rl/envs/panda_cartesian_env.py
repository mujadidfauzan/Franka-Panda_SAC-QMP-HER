from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from panda_rl.controllers.ik_controller import DifferentialIK6DController
from panda_rl.utils.mujoco_utils import (
    actuator_ids,
    euler_xyz_to_mat,
    joint_dof_ids,
    joint_qpos_ids,
    mat_to_quat_wxyz,
    mj_name_to_id,
    orientation_error_vector,
    quat_wxyz_to_mat,
)


ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
ARM_ACTUATOR_NAMES = (
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
)
GRIPPER_ACTUATOR_NAME = "actuator8"
TARGET_BODY_NAME = "target_body"
TARGET_SITE_NAME = "target_site"


class PandaReachEnv(gym.Env):
    """Reach + orient task for Franka Panda with SAC-friendly continuous actions."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        model_path=None,
        randomize_target=True,
        frame_skip=10,
        max_steps=200,
        cartesian_scale=0.03,
        rotation_scale=0.10,
        success_pos_threshold=0.03,
        success_ori_threshold=0.15,
        success_bonus=10.0,
        action_penalty_weight=0.01,
        ik_max_joint_step=0.08,
    ):
        super().__init__()

        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else (
            project_root / "franka_emika_panda" / "scene_rl.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        self.randomize_target = randomize_target
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.cartesian_scale = cartesian_scale
        self.rotation_scale = rotation_scale
        self.success_pos_threshold = success_pos_threshold
        self.success_ori_threshold = success_ori_threshold
        self.success_bonus = success_bonus
        self.action_penalty_weight = action_penalty_weight
        self.ik_max_joint_step = ik_max_joint_step

        self.ee_site_name = "ee_site"
        self.ee_site_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.ee_site_name,
        )

        self.arm_qpos_ids = joint_qpos_ids(self.model, ARM_JOINT_NAMES)
        self.arm_dof_ids = joint_dof_ids(self.model, ARM_JOINT_NAMES)
        self.finger_qpos_ids = joint_qpos_ids(self.model, FINGER_JOINT_NAMES)
        self.arm_actuator_ids = actuator_ids(self.model, ARM_ACTUATOR_NAMES)
        self.gripper_actuator_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            GRIPPER_ACTUATOR_NAME,
        )
        self.target_body_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            TARGET_BODY_NAME,
        )
        self.target_site_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            TARGET_SITE_NAME,
        )
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        if self.target_mocap_id == -1:
            raise ValueError(f"Body '{TARGET_BODY_NAME}' must be a mocap body.")

        self.ik_controller = DifferentialIK6DController(
            model=self.model,
            site_name=self.ee_site_name,
            joint_names=list(ARM_JOINT_NAMES),
            pos_weight=1.0,
            ori_weight=0.35,
            damping=0.08,
            max_joint_step=self.ik_max_joint_step,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        obs_dim = 3 + 4 + 3 + 4 + 3 + 3 + 7 + 7 + len(self.finger_qpos_ids)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.default_target_pos = np.array([0.45, 0.25, 0.45], dtype=np.float64)
        self.default_target_mat = euler_xyz_to_mat([np.pi, 0.0, 0.0])
        self.target_pos_low = np.array([0.35, -0.30, 0.25], dtype=np.float64)
        self.target_pos_high = np.array([0.65, 0.30, 0.60], dtype=np.float64)
        self.target_rpy_low = np.array([-0.45, -0.45, -np.pi], dtype=np.float64)
        self.target_rpy_high = np.array([0.45, 0.45, np.pi], dtype=np.float64)
        self.command_pos_low = np.array([0.25, -0.45, 0.15], dtype=np.float64)
        self.command_pos_high = np.array([0.75, 0.45, 0.75], dtype=np.float64)

        self.target_pos = self.default_target_pos.copy()
        self.target_mat = self.default_target_mat.copy()
        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self._reset_robot()
        self._reset_target(options)

        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info(action_penalty=0.0)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.copy()

        self._apply_cartesian_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)

        terminated = reward_info["is_success"]
        truncated = self.current_step >= self.max_steps
        info = self._get_info(**reward_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _reset_robot(self):
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        if home_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.arm_qpos_ids] = np.array(
                [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
                dtype=np.float64,
            )
            self.data.qpos[self.finger_qpos_ids] = 0.04

        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_ids]
        self._set_gripper_open()
        mujoco.mj_forward(self.model, self.data)

    def _reset_target(self, options):
        if "target_pos" in options:
            self.target_pos = np.asarray(
                options["target_pos"],
                dtype=np.float64,
            ).reshape(3)
        elif self.randomize_target:
            self.target_pos = self.np_random.uniform(
                low=self.target_pos_low,
                high=self.target_pos_high,
            )
        else:
            self.target_pos = self.default_target_pos.copy()

        if "target_mat" in options:
            self.target_mat = np.asarray(
                options["target_mat"],
                dtype=np.float64,
            ).reshape(3, 3)
        elif "target_ori" in options:
            self.target_mat = np.asarray(
                options["target_ori"],
                dtype=np.float64,
            ).reshape(3, 3)
        elif "target_quat" in options:
            self.target_mat = quat_wxyz_to_mat(options["target_quat"])
        elif self.randomize_target:
            target_rpy = self.np_random.uniform(
                low=self.target_rpy_low,
                high=self.target_rpy_high,
            )
            self.target_mat = self.default_target_mat @ euler_xyz_to_mat(target_rpy)
        else:
            self.target_mat = self.default_target_mat.copy()

        self._sync_target_marker()

    def _sync_target_marker(self):
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos
        self.data.mocap_quat[self.target_mocap_id] = mat_to_quat_wxyz(self.target_mat)

    def _apply_cartesian_action(self, pose_action):
        delta_pos = pose_action[:3] * self.cartesian_scale
        delta_rot = pose_action[3:] * self.rotation_scale

        current_pos = self.data.site_xpos[self.ee_site_id].copy()
        current_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        command_pos = np.clip(
            current_pos + delta_pos,
            self.command_pos_low,
            self.command_pos_high,
        )
        command_mat = current_mat @ euler_xyz_to_mat(delta_rot)

        target_qpos, self.last_ik_info = self.ik_controller.compute_joint_target(
            data=self.data,
            target_pos=command_pos,
            target_mat=command_mat,
        )
        self.data.ctrl[self.arm_actuator_ids] = target_qpos

    def _set_gripper_open(self):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        self.data.ctrl[self.gripper_actuator_id] = (
            ctrl_max if ctrl_max > ctrl_min else 255.0
        )

    def _get_obs(self):
        ee_pos, ee_mat = self._get_ee_pose()
        ee_quat = mat_to_quat_wxyz(ee_mat)
        target_quat = mat_to_quat_wxyz(self.target_mat)
        position_error = self.target_pos - ee_pos
        orientation_error = orientation_error_vector(ee_mat, self.target_mat)

        obs = np.concatenate(
            [
                ee_pos,
                ee_quat,
                self.target_pos,
                target_quat,
                position_error,
                orientation_error,
                self.data.qpos[self.arm_qpos_ids],
                self.data.qvel[self.arm_dof_ids],
                self.data.qpos[self.finger_qpos_ids],
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action):
        ee_pos, ee_mat = self._get_ee_pose()
        position_distance = np.linalg.norm(self.target_pos - ee_pos)
        orientation_distance = np.linalg.norm(
            orientation_error_vector(ee_mat, self.target_mat)
        )
        action_penalty = self.action_penalty_weight * np.sum(np.square(action))

        is_success = (
            position_distance < self.success_pos_threshold
            and orientation_distance < self.success_ori_threshold
        )
        reward_position = -position_distance
        reward_orientation = -orientation_distance
        reward_action_penalty = -action_penalty
        reward_success_bonus = self.success_bonus if is_success else 0.0

        reward = (
            reward_position
            + reward_orientation
            + reward_action_penalty
            + reward_success_bonus
        )

        return float(reward), {
            "position_distance": float(position_distance),
            "orientation_distance": float(orientation_distance),
            "action_penalty": float(action_penalty),
            "reward_position": float(reward_position),
            "reward_orientation": float(reward_orientation),
            "reward_action_penalty": float(reward_action_penalty),
            "reward_success_bonus": float(reward_success_bonus),
            "reward_total": float(reward),
            "is_success": bool(is_success),
        }

    def _get_info(self, **extra):
        ee_pos, ee_mat = self._get_ee_pose()
        position_distance = np.linalg.norm(self.target_pos - ee_pos)
        orientation_distance = np.linalg.norm(
            orientation_error_vector(ee_mat, self.target_mat)
        )

        info = {
            "is_success": bool(
                position_distance < self.success_pos_threshold
                and orientation_distance < self.success_ori_threshold
            ),
            "position_distance": float(position_distance),
            "orientation_distance": float(orientation_distance),
            "target_pos": self.target_pos.copy(),
            "target_quat": mat_to_quat_wxyz(self.target_mat).astype(np.float32),
            "ee_pos": ee_pos.copy(),
            "ee_quat": mat_to_quat_wxyz(ee_mat).astype(np.float32),
        }
        info.update(extra)
        return info

    def _get_ee_pose(self):
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return ee_pos, ee_mat

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer as mujoco_viewer

                self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
