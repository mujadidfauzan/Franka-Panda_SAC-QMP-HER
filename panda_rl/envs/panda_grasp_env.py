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
CUBE_BODY_NAME = "cube"
CUBE_JOINT_NAME = "cube_freejoint"
CUBE_SITE_NAME = "cube_site"


class PandaGraspEnv(gym.Env):
    """Grasp a cube and lift it by 5 cm with Cartesian IK plus gripper action."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        model_path=None,
        randomize_object=True,
        frame_skip=10,
        max_steps=250,
        cartesian_scale=0.03,
        rotation_scale=0.10,
        lift_height=0.05,
        success_bonus=25.0,
        close_bonus=2.0,
        close_bonus_distance=0.01,
        action_penalty_weight=0.01,
        ik_max_joint_step=0.08,
    ):
        super().__init__()

        project_root = Path(__file__).resolve().parents[2]
        self.model_path = (
            Path(model_path)
            if model_path
            else (project_root / "franka_emika_panda" / "scene_grasp.xml")
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        self.randomize_object = randomize_object
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.cartesian_scale = cartesian_scale
        self.rotation_scale = rotation_scale
        self.lift_height = lift_height
        self.success_bonus = success_bonus
        self.close_bonus = close_bonus
        self.close_bonus_distance = close_bonus_distance
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

        self.cube_body_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            CUBE_BODY_NAME,
        )
        self.cube_joint_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            CUBE_JOINT_NAME,
        )
        self.cube_site_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            CUBE_SITE_NAME,
        )
        self.cube_qpos_id = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_dof_id = self.model.jnt_dofadr[self.cube_joint_id]

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
            shape=(7,),
            dtype=np.float32,
        )

        obs_dim = 3 + 4 + 3 + 4 + 3 + 3 + 1 + 7 + 7 + 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.default_cube_pos = np.array([0.48, 0.0, 0.02], dtype=np.float64)
        self.object_pos_low = np.array([0.30, -0.35, 0.02], dtype=np.float64)
        self.object_pos_high = np.array([0.72, 0.35, 0.02], dtype=np.float64)
        self.command_pos_low = np.array([0.25, -0.45, 0.05], dtype=np.float64)
        self.command_pos_high = np.array([0.75, 0.45, 0.75], dtype=np.float64)

        self.current_step = 0
        self.initial_cube_z = float(self.default_cube_pos[2])
        self.lift_target_pos = self.default_cube_pos + np.array(
            [0.0, 0.0, self.lift_height],
            dtype=np.float64,
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}
        self.last_auto_gripper_closed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self._reset_robot()
        self._reset_object(options)

        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}
        self.last_auto_gripper_closed = False

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

        self._apply_cartesian_action(action[:6])
        self._apply_auto_gripper()

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

    def _reset_object(self, options):
        if "object_pos" in options:
            cube_pos = np.asarray(options["object_pos"], dtype=np.float64).reshape(3)
        elif self.randomize_object:
            cube_pos = self.np_random.uniform(
                low=self.object_pos_low,
                high=self.object_pos_high,
            )
        else:
            cube_pos = self.default_cube_pos.copy()

        if "object_quat" in options:
            cube_quat = np.asarray(options["object_quat"], dtype=np.float64).reshape(4)
            cube_quat = cube_quat / np.linalg.norm(cube_quat)
        elif self.randomize_object:
            yaw = self.np_random.uniform(-np.pi, np.pi)
            cube_quat = mat_to_quat_wxyz(euler_xyz_to_mat([0.0, 0.0, yaw]))
        else:
            cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = cube_pos
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = cube_quat
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0

        self.initial_cube_z = float(cube_pos[2])
        self.lift_target_pos = cube_pos + np.array(
            [0.0, 0.0, self.lift_height],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)

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

    def _apply_gripper_action(self, gripper_action):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        gripper_ctrl = ctrl_min + (gripper_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl

    def _apply_auto_gripper(self):
        reach_distance = self._reach_distance()
        self.last_auto_gripper_closed = reach_distance <= self.close_bonus_distance

        if self.last_auto_gripper_closed:
            self._set_gripper_closed()
        else:
            self._set_gripper_open()

    def _set_gripper_open(self):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        self.data.ctrl[self.gripper_actuator_id] = (
            ctrl_max if ctrl_max > ctrl_min else 255.0
        )

    def _set_gripper_closed(self):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        self.data.ctrl[self.gripper_actuator_id] = (
            ctrl_min if ctrl_max > ctrl_min else 0.0
        )

    def _get_obs(self):
        ee_pos, ee_mat = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_quat = self.data.xquat[self.cube_body_id].copy()
        object_to_ee = cube_pos - ee_pos
        lift_error = self.lift_target_pos - cube_pos
        lift_progress = self._lift_progress(cube_pos)

        obs = np.concatenate(
            [
                ee_pos,
                mat_to_quat_wxyz(ee_mat),
                cube_pos,
                cube_quat,
                object_to_ee,
                lift_error,
                np.array([lift_progress], dtype=np.float64),
                self.data.qpos[self.arm_qpos_ids],
                self.data.qvel[self.arm_dof_ids],
                self.data.qpos[self.finger_qpos_ids],
                self.data.qvel[self.cube_dof_id : self.cube_dof_id + 3],
                self.data.qvel[self.cube_dof_id + 3 : self.cube_dof_id + 6],
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action):
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        reach_distance = self._reach_distance()
        lift_height = max(0.0, float(cube_pos[2] - self.initial_cube_z))
        lift_progress = self._lift_progress(cube_pos)
        action_penalty = self.action_penalty_weight * np.sum(np.square(action))

        is_success = lift_height >= self.lift_height
        reward_reach = -2.0 * reach_distance
        reward_lift = self.success_bonus if is_success else 0.0
        reward_close_bonus = (
            self.close_bonus if reach_distance <= self.close_bonus_distance else 0.0
        )
        reward_success_bonus = reward_lift

        reward = reward_reach + reward_lift + reward_close_bonus

        return float(reward), {
            "reach_distance": float(reach_distance),
            "lift_height": float(lift_height),
            "lift_progress": float(lift_progress),
            "action_penalty": float(action_penalty),
            "auto_gripper_closed": bool(self.last_auto_gripper_closed),
            "reward_reach": float(reward_reach),
            "reward_lift": float(reward_lift),
            "reward_close_bonus": float(reward_close_bonus),
            "reward_success_bonus": float(reward_success_bonus),
            "reward_total": float(reward),
            "is_success": bool(is_success),
        }

    def _get_info(self, **extra):
        ee_pos, _ = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        reach_distance = self._reach_distance()
        lift_height = max(0.0, float(cube_pos[2] - self.initial_cube_z))

        info = {
            "is_success": bool(lift_height >= self.lift_height),
            "reach_distance": float(reach_distance),
            "lift_height": float(lift_height),
            "lift_progress": float(self._lift_progress(cube_pos)),
            "auto_gripper_closed": bool(self.last_auto_gripper_closed),
            "cube_pos": cube_pos.copy(),
            "cube_quat": self.data.xquat[self.cube_body_id].copy(),
            "ee_pos": ee_pos.copy(),
            "lift_target_pos": self.lift_target_pos.copy(),
        }
        info.update(extra)
        return info

    def _lift_progress(self, cube_pos):
        lift_height = max(0.0, float(cube_pos[2] - self.initial_cube_z))
        return float(np.clip(lift_height / self.lift_height, 0.0, 1.0))

    def _reach_distance(self):
        ee_pos, _ = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        return float(np.linalg.norm(cube_pos - ee_pos))

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
