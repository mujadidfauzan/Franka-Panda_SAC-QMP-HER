from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC


@dataclass
class PrimitiveCandidate:
    name: str
    model: object
    obs_projector: Callable
    action_adapter: Optional[Callable] = None
    deterministic: bool = True


def target_obs_to_state_array(target_obs):
    if isinstance(target_obs, dict):
        state_obs = target_obs["observation"]
    else:
        state_obs = target_obs

    state_obs = np.asarray(state_obs, dtype=np.float32)
    if state_obs.ndim == 1:
        state_obs = state_obs.reshape(1, -1)
    return state_obs


def target_obs_to_insert_obs(target_obs):
    return target_obs_to_state_array(target_obs)


def target_obs_to_grasp_obs(target_obs, lift_height=0.03, cube_half_size=0.02):
    state_obs = target_obs_to_state_array(target_obs)

    ee_pos = state_obs[:, 0:3]
    ee_quat = state_obs[:, 3:7]
    cube_pos = state_obs[:, 7:10]
    cube_quat = state_obs[:, 10:14]
    ee_to_cube = state_obs[:, 20:23]
    qpos_arm = state_obs[:, 23:30]
    qvel_arm = state_obs[:, 30:37]
    gripper_qpos = state_obs[:, 37:39]
    cube_linvel = state_obs[:, 39:42]
    cube_angvel = state_obs[:, 42:45]

    lift_error = np.zeros((state_obs.shape[0], 3), dtype=np.float32)
    lift_error[:, 2] = cube_half_size + lift_height - cube_pos[:, 2]
    lift_progress = np.clip(
        (cube_pos[:, 2] - cube_half_size) / max(lift_height, 1e-6),
        0.0,
        1.0,
    ).reshape(-1, 1)

    return np.concatenate(
        [
            ee_pos,
            ee_quat,
            cube_pos,
            cube_quat,
            ee_to_cube,
            lift_error,
            lift_progress,
            qpos_arm,
            qvel_arm,
            gripper_qpos,
            cube_linvel,
            cube_angvel,
        ],
        axis=1,
    ).astype(np.float32)


def grasp_auto_gripper_adapter(actions, target_obs, close_distance=0.01):
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    state_obs = target_obs_to_state_array(target_obs)
    ee_pos = state_obs[:, 0:3]
    cube_pos = state_obs[:, 7:10]
    reach_distance = np.linalg.norm(cube_pos - ee_pos, axis=1)

    actions[:, 6] = np.where(reach_distance <= close_distance, -1.0, 1.0)
    return actions


class QMPSAC(SAC):
    """SAC with Q-switch Mixture of Primitives action selection during rollouts."""

    def __init__(
        self,
        *args,
        primitive_policies: Optional[Sequence[PrimitiveCandidate]] = None,
        qmp_warmup_steps: int = 20_000,
        qmp_epsilon: float = 0.05,
        **kwargs,
    ):
        self.primitive_policies = list(primitive_policies or [])
        self.qmp_warmup_steps = int(qmp_warmup_steps)
        self.qmp_epsilon = float(qmp_epsilon)
        self.qmp_last_selected_names = []
        self.qmp_last_q_values = {}
        self.qmp_last_candidate_names = []
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["primitive_policies"]

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if not self.primitive_policies or not isinstance(self.action_space, spaces.Box):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        candidate_names, candidate_actions = self._primitive_candidate_actions(n_envs)
        if not candidate_actions:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        warmup_steps = max(int(learning_starts), self.qmp_warmup_steps)
        if self.num_timesteps < warmup_steps:
            selected_action = self._select_warmup_action(
                candidate_names,
                candidate_actions,
                n_envs,
            )
        else:
            target_action, _ = self.predict(self._last_obs, deterministic=False)
            target_action = self._ensure_action_batch(target_action, n_envs)
            candidate_names = ["target", *candidate_names]
            candidate_actions = [target_action, *candidate_actions]
            selected_action = self._select_q_action(
                candidate_names,
                candidate_actions,
                n_envs,
            )

        scaled_action = self.policy.scale_action(selected_action)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1.0, 1.0)
            selected_action = self.policy.unscale_action(scaled_action)

        return selected_action, scaled_action

    def _primitive_candidate_actions(self, n_envs):
        candidate_names = []
        candidate_actions = []

        for primitive in self.primitive_policies:
            primitive_obs = primitive.obs_projector(self._last_obs)
            action, _ = primitive.model.predict(
                primitive_obs,
                deterministic=primitive.deterministic,
            )
            action = self._ensure_action_batch(action, n_envs)

            if primitive.action_adapter is not None:
                action = primitive.action_adapter(action, self._last_obs)
                action = self._ensure_action_batch(action, n_envs)

            action = np.clip(action, self.action_space.low, self.action_space.high)
            candidate_names.append(primitive.name)
            candidate_actions.append(action.astype(np.float32))

        return candidate_names, candidate_actions

    def _select_warmup_action(self, candidate_names, candidate_actions, n_envs):
        action_stack = np.stack(candidate_actions, axis=1)
        selected_indices = np.random.randint(0, len(candidate_names), size=n_envs)
        selected_action = action_stack[np.arange(n_envs), selected_indices].copy()
        selected_names = [candidate_names[index] for index in selected_indices]

        selected_action, selected_names = self._apply_epsilon_random_actions(
            selected_action,
            selected_names,
        )
        self._store_qmp_selection(candidate_names, selected_names, None)
        return selected_action

    def _select_q_action(self, candidate_names, candidate_actions, n_envs):
        action_stack = np.stack(candidate_actions, axis=1)
        n_candidates = len(candidate_names)
        flat_actions = action_stack.reshape(n_envs * n_candidates, -1)
        flat_scaled_actions = self.policy.scale_action(flat_actions)

        obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
        repeated_obs = self._repeat_obs_tensor(obs_tensor, n_candidates)
        action_tensor = th.as_tensor(
            flat_scaled_actions,
            device=self.device,
            dtype=th.float32,
        )

        with th.no_grad():
            q_values = self.critic(repeated_obs, action_tensor)
            q_values = th.min(th.cat(q_values, dim=1), dim=1)[0]

        q_array = q_values.detach().cpu().numpy().reshape(n_envs, n_candidates)
        selected_indices = np.argmax(q_array, axis=1)
        selected_action = action_stack[np.arange(n_envs), selected_indices].copy()
        selected_names = [candidate_names[index] for index in selected_indices]

        selected_action, selected_names = self._apply_epsilon_random_actions(
            selected_action,
            selected_names,
        )
        mean_q_values = {
            name: float(np.mean(q_array[:, index]))
            for index, name in enumerate(candidate_names)
        }
        self._store_qmp_selection(candidate_names, selected_names, mean_q_values)
        return selected_action

    def _apply_epsilon_random_actions(self, selected_action, selected_names):
        if self.qmp_epsilon <= 0.0:
            return selected_action, selected_names

        for env_index in range(selected_action.shape[0]):
            if np.random.random() < self.qmp_epsilon:
                selected_action[env_index] = self.action_space.sample()
                selected_names[env_index] = "random"
        return selected_action, selected_names

    def _store_qmp_selection(self, candidate_names, selected_names, mean_q_values):
        self.qmp_last_candidate_names = list(candidate_names)
        self.qmp_last_selected_names = list(selected_names)
        self.qmp_last_q_values = mean_q_values or {}

    def _ensure_action_batch(self, action, n_envs):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape[0] != n_envs:
            action = np.repeat(action[:1], n_envs, axis=0)
        return action

    def _repeat_obs_tensor(self, obs_tensor, repeats):
        if isinstance(obs_tensor, dict):
            return {
                key: value.repeat_interleave(repeats, dim=0)
                for key, value in obs_tensor.items()
            }
        return obs_tensor.repeat_interleave(repeats, dim=0)
