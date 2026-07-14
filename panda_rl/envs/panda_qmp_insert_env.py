import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from panda_rl.envs.panda_insert_env import PandaInsertEnv
from panda_rl.utils.mujoco_utils import euler_xyz_to_mat, mat_to_quat_wxyz


class PandaQMPInsertEnv(PandaInsertEnv):
    """Full sparse insert task used as the QMP-HER target environment."""

    def __init__(
        self,
        render_mode=None,
        model_path=None,
        randomize_object=True,
        randomize_socket=True,
        frame_skip=10,
        max_steps=350,
        cartesian_scale=0.025,
        rotation_scale=0.08,
        insert_tolerance=0.01,
        success_bonus=1.0,
        ik_max_joint_step=0.08,
        terminate_on_success=True,
    ):
        super().__init__(
            render_mode=render_mode,
            model_path=model_path,
            randomize_socket=randomize_socket,
            randomize_start=False,
            frame_skip=frame_skip,
            max_steps=max_steps,
            cartesian_scale=cartesian_scale,
            rotation_scale=rotation_scale,
            insert_tolerance=insert_tolerance,
            success_bonus=success_bonus,
            action_penalty_weight=0.0,
            ik_max_joint_step=ik_max_joint_step,
            terminate_on_success=terminate_on_success,
        )

        self.randomize_object = randomize_object
        self.default_cube_pos = np.array(
            [0.45, -0.12, self.cube_half_size],
            dtype=np.float64,
        )
        self.object_pos_low = np.array(
            [0.30, -0.35, self.cube_half_size],
            dtype=np.float64,
        )
        self.object_pos_high = np.array(
            [0.62, 0.35, self.cube_half_size],
            dtype=np.float64,
        )
        self.min_initial_socket_distance = 0.14

        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(45,),
            dtype=np.float32,
        )
        goal_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": state_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

        self.initial_cube_z = float(self.cube_half_size)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}

        self._reset_robot_open()
        self._reset_socket(options)
        self._reset_object_on_table(options)

        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.last_ik_info = {}

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info(reward_sparse=0.0, reward_success_bonus=0.0)

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

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        is_success = np.asarray(distance <= self.insert_tolerance, dtype=np.float32)
        return is_success * self.success_bonus

    def _reset_robot_open(self):
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

    def _reset_object_on_table(self, options):
        if "object_pos" in options:
            cube_pos = np.asarray(options["object_pos"], dtype=np.float64).reshape(3)
        elif self.randomize_object:
            cube_pos = self._sample_object_pos_away_from_socket()
        else:
            cube_pos = self.default_cube_pos.copy()

        cube_pos = cube_pos.copy()
        cube_pos[2] = self.cube_half_size

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
        mujoco.mj_forward(self.model, self.data)

    def _sample_object_pos_away_from_socket(self):
        cube_pos = self.default_cube_pos.copy()
        for _ in range(100):
            cube_pos = self.np_random.uniform(
                low=self.object_pos_low,
                high=self.object_pos_high,
            )
            xy_distance = np.linalg.norm(cube_pos[:2] - self.socket_pos[:2])
            if xy_distance >= self.min_initial_socket_distance:
                return cube_pos
        return cube_pos

    def _set_gripper_open(self):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        self.data.ctrl[self.gripper_actuator_id] = (
            ctrl_max if ctrl_max > ctrl_min else 255.0
        )

    def _get_obs(self):
        state_obs = PandaInsertEnv._get_obs(self)
        cube_pos = self.data.xpos[self.cube_body_id].copy().astype(np.float32)

        return {
            "observation": state_obs.astype(np.float32),
            "achieved_goal": cube_pos,
            "desired_goal": self.insert_target_pos.astype(np.float32),
        }

    def _compute_reward(self, action):
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        insert_error = cube_pos - self.insert_target_pos
        insert_distance = float(np.linalg.norm(insert_error))
        xy_distance = float(np.linalg.norm(insert_error[:2]))
        z_error = float(abs(insert_error[2]))
        reward = float(self.compute_reward(cube_pos, self.insert_target_pos, {}))
        is_success = reward > 0.0

        return reward, {
            "insert_distance": insert_distance,
            "xy_distance": xy_distance,
            "z_error": z_error,
            "reach_distance": float(self._reach_distance()),
            "cube_lift_height": max(0.0, float(cube_pos[2] - self.initial_cube_z)),
            "reward_sparse": reward,
            "reward_success_bonus": reward,
            "reward_total": reward,
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
            "reach_distance": float(self._reach_distance()),
            "cube_lift_height": max(0.0, float(cube_pos[2] - self.initial_cube_z)),
            "cube_pos": cube_pos.copy(),
            "cube_quat": self.data.xquat[self.cube_body_id].copy(),
            "ee_pos": ee_pos.copy(),
            "socket_pos": self.socket_pos.copy(),
            "insert_target_pos": self.insert_target_pos.copy(),
            "gripper_qpos": self.data.qpos[self.finger_qpos_ids].copy(),
        }
        info.update(extra)
        return info

    def _reach_distance(self):
        ee_pos, _ = self._get_ee_pose()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        return float(np.linalg.norm(cube_pos - ee_pos))
