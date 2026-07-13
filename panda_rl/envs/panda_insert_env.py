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
SOCKET_BODY_NAME = "socket_body"
INSERT_TARGET_SITE_NAME = "insert_target_site"


class PandaInsertEnv(gym.Env):
    """Insert a grasped cube into a square socket with Cartesian IK control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        model_path=None,
        randomize_socket=True,
        randomize_start=True,
        frame_skip=10,
        max_steps=250,
        cartesian_scale=0.025,
        rotation_scale=0.08,
        insert_tolerance=0.01,
        success_bonus=100.0,
        action_penalty_weight=0.005,
        ik_max_joint_step=0.08,
        terminate_on_success=True,
    ):
        super().__init__()

        project_root = Path(__file__).resolve().parents[2]
        self.model_path = (
            Path(model_path)
            if model_path
            else (project_root / "franka_emika_panda" / "scene_insert.xml")
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        self.randomize_socket = randomize_socket
        self.randomize_start = randomize_start
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.cartesian_scale = cartesian_scale
        self.rotation_scale = rotation_scale
        self.insert_tolerance = insert_tolerance
        self.success_bonus = success_bonus
        self.action_penalty_weight = action_penalty_weight
        self.ik_max_joint_step = ik_max_joint_step
        self.terminate_on_success = terminate_on_success

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

        self.socket_body_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            SOCKET_BODY_NAME,
        )
        self.insert_target_site_id = mj_name_to_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            INSERT_TARGET_SITE_NAME,
        )
        self.socket_mocap_id = self.model.body_mocapid[self.socket_body_id]
        if self.socket_mocap_id == -1:
            raise ValueError(f"Body '{SOCKET_BODY_NAME}' must be a mocap body.")

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

        obs_dim = 3 + 4 + 3 + 4 + 3 + 3 + 3 + 7 + 7 + 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.cube_half_size = 0.02
        self.default_socket_pos = np.array([0.55, 0.0, 0.0], dtype=np.float64)
        self.socket_pos_low = np.array([0.35, -0.28, 0.0], dtype=np.float64)
        self.socket_pos_high = np.array([0.68, 0.28, 0.0], dtype=np.float64)
        self.start_xy_offset_low = np.array([-0.12, -0.12], dtype=np.float64)
        self.start_xy_offset_high = np.array([0.12, 0.12], dtype=np.float64)
        self.start_z_low = 0.14
        self.start_z_high = 0.22
        self.command_pos_low = np.array([0.25, -0.45, 0.03], dtype=np.float64)
        self.command_pos_high = np.array([0.75, 0.45, 0.75], dtype=np.float64)
        self.hold_finger_qpos = self.cube_half_size

        self.current_step = 0
        self.socket_pos = self.default_socket_pos.copy()
        self.insert_target_pos = self.default_socket_pos + np.array(
            [0.0, 0.0, self.cube_half_size],
            dtype=np.float64,
        )
        self.held_cube_start_pos = np.array([0.45, 0.0, 0.18], dtype=np.float64)
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self._reset_robot()
        self._reset_socket(options)
        self._reset_held_cube(options)

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

        self._apply_cartesian_action(action[:6])
        self._apply_gripper_action(action[6])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)

        terminated = bool(reward_info["is_success"] and self.terminate_on_success)
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

        self.data.qpos[self.finger_qpos_ids] = self.hold_finger_qpos
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qpos_ids]
        self._set_gripper_closed()
        mujoco.mj_forward(self.model, self.data)

    def _reset_socket(self, options):
        if "socket_pos" in options:
            socket_pos = np.asarray(options["socket_pos"], dtype=np.float64).reshape(3)
        elif self.randomize_socket:
            socket_pos = self.np_random.uniform(
                low=self.socket_pos_low,
                high=self.socket_pos_high,
            )
        else:
            socket_pos = self.default_socket_pos.copy()

        socket_pos = socket_pos.copy()
        socket_pos[2] = 0.0
        self.socket_pos = socket_pos
        self.insert_target_pos = socket_pos + np.array(
            [0.0, 0.0, self.cube_half_size],
            dtype=np.float64,
        )

        self.data.mocap_pos[self.socket_mocap_id] = self.socket_pos
        self.data.mocap_quat[self.socket_mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)

    def _reset_held_cube(self, options):
        if "held_cube_pos" in options:
            held_pos = np.asarray(options["held_cube_pos"], dtype=np.float64).reshape(3)
        elif self.randomize_start:
            offset_xy = self.np_random.uniform(
                low=self.start_xy_offset_low,
                high=self.start_xy_offset_high,
            )
            held_pos = np.array(
                [
                    self.insert_target_pos[0] + offset_xy[0],
                    self.insert_target_pos[1] + offset_xy[1],
                    self.np_random.uniform(self.start_z_low, self.start_z_high),
                ],
                dtype=np.float64,
            )
        else:
            held_pos = self.insert_target_pos + np.array(
                [0.0, 0.0, 0.16],
                dtype=np.float64,
            )

        held_pos = np.clip(held_pos, self.command_pos_low, self.command_pos_high)
        self.held_cube_start_pos = held_pos

        _, target_mat = self._get_ee_pose()
        self._set_arm_to_ee_pose(held_pos, target_mat)
        ee_pos, _ = self._get_ee_pose()

        if "cube_quat" in options:
            cube_quat = np.asarray(options["cube_quat"], dtype=np.float64).reshape(4)
            cube_quat = cube_quat / np.linalg.norm(cube_quat)
        else:
            yaw = self.np_random.uniform(-np.pi, np.pi)
            cube_quat = mat_to_quat_wxyz(euler_xyz_to_mat([0.0, 0.0, yaw]))

        self.data.qpos[self.cube_qpos_id : self.cube_qpos_id + 3] = ee_pos
        self.data.qpos[self.cube_qpos_id + 3 : self.cube_qpos_id + 7] = cube_quat
        self.data.qvel[self.cube_dof_id : self.cube_dof_id + 6] = 0.0
        self.data.qpos[self.finger_qpos_ids] = self.hold_finger_qpos
        self._set_gripper_closed()
        mujoco.mj_forward(self.model, self.data)

    def _set_arm_to_ee_pose(self, target_pos, target_mat, iterations=80):
        for _ in range(iterations):
            target_qpos, self.last_ik_info = self.ik_controller.compute_joint_target(
                data=self.data,
                target_pos=target_pos,
                target_mat=target_mat,
            )
            self.data.qpos[self.arm_qpos_ids] = target_qpos
            self.data.ctrl[self.arm_actuator_ids] = target_qpos
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

    def _set_gripper_closed(self):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        self.data.ctrl[self.gripper_actuator_id] = (
            ctrl_min if ctrl_max > ctrl_min else 0.0
        )

    def _get_obs(self):
        ee_pos, ee_mat = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_quat = self.data.xquat[self.cube_body_id].copy()
        cube_to_socket = self.insert_target_pos - cube_pos
        ee_to_cube = cube_pos - ee_pos

        obs = np.concatenate(
            [
                ee_pos,
                mat_to_quat_wxyz(ee_mat),
                cube_pos,
                cube_quat,
                self.insert_target_pos,
                cube_to_socket,
                ee_to_cube,
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
        insert_error = cube_pos - self.insert_target_pos
        insert_distance = float(np.linalg.norm(insert_error))
        xy_distance = float(np.linalg.norm(insert_error[:2]))
        z_error = float(abs(insert_error[2]))
        action_penalty = self.action_penalty_weight * np.sum(np.square(action))

        is_success = insert_distance <= self.insert_tolerance
        reward_insert = -5.0 * insert_distance
        reward_xy = -2.0 * xy_distance
        reward_z = -1.0 * z_error
        reward_action_penalty = -float(action_penalty)
        reward_success_bonus = self.success_bonus if is_success else 0.0

        reward = (
            reward_insert
            + reward_xy
            + reward_z
            + reward_action_penalty
            + reward_success_bonus
        )

        return float(reward), {
            "insert_distance": float(insert_distance),
            "xy_distance": float(xy_distance),
            "z_error": float(z_error),
            "action_penalty": float(action_penalty),
            "reward_insert": float(reward_insert),
            "reward_xy": float(reward_xy),
            "reward_z": float(reward_z),
            "reward_action_penalty": float(reward_action_penalty),
            "reward_success_bonus": float(reward_success_bonus),
            "reward_total": float(reward),
            "is_success": bool(is_success),
        }

    def _get_info(self, **extra):
        ee_pos, _ = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        insert_error = cube_pos - self.insert_target_pos
        insert_distance = float(np.linalg.norm(insert_error))
        xy_distance = float(np.linalg.norm(insert_error[:2]))
        z_error = float(abs(insert_error[2]))

        info = {
            "is_success": bool(insert_distance <= self.insert_tolerance),
            "insert_distance": insert_distance,
            "xy_distance": xy_distance,
            "z_error": z_error,
            "cube_pos": cube_pos.copy(),
            "cube_quat": self.data.xquat[self.cube_body_id].copy(),
            "ee_pos": ee_pos.copy(),
            "socket_pos": self.socket_pos.copy(),
            "insert_target_pos": self.insert_target_pos.copy(),
            "gripper_qpos": self.data.qpos[self.finger_qpos_ids].copy(),
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
