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


def target_obs_object_held(
    target_obs,
    cube_half_size=0.02,
    min_lift_height=0.01,
    max_ee_distance=0.06,
    max_gripper_qpos=0.03,
):
    state_obs = target_obs_to_state_array(target_obs)
    ee_pos = state_obs[:, 0:3]
    cube_pos = state_obs[:, 7:10]
    gripper_qpos = state_obs[:, 37:39]

    lift_height = cube_pos[:, 2] - cube_half_size
    ee_distance = np.linalg.norm(cube_pos - ee_pos, axis=1)
    gripper_mean = np.mean(gripper_qpos, axis=1)
    return (
        (lift_height >= min_lift_height)
        & (ee_distance <= max_ee_distance)
        & (gripper_mean <= max_gripper_qpos)
    )


class QMPSAC(SAC):
    """SAC with Q-switch Mixture of Primitives action selection during rollouts."""

    def __init__(
        self,
        *args,
        primitive_policies: Optional[Sequence[PrimitiveCandidate]] = None,
        qmp_warmup_steps: int = 20_000,
        qmp_epsilon: float = 0.05,
        qmp_primitive_only_steps: int = 500_000,
        qmp_gate_steps: int = 500_000,
        qmp_commitment_min_steps: int = 10,
        qmp_commitment_max_steps: int = 30,
        qmp_held_min_lift_height: float = 0.01,
        qmp_held_max_ee_distance: float = 0.06,
        qmp_held_max_gripper_qpos: float = 0.03,
        qmp_stage_aware_mask: bool = True,
        qmp_target_q_margin: float = 0.05,
        qmp_bc_steps: int = 500_000,
        qmp_bc_coef: float = 1.0,
        qmp_bc_batch_size: int = 256,
        qmp_bc_buffer_size: int = 100_000,
        **kwargs,
    ):
        self.primitive_policies = list(primitive_policies or [])
        self.qmp_warmup_steps = int(qmp_warmup_steps)
        self.qmp_epsilon = float(qmp_epsilon)
        self.qmp_primitive_only_steps = int(qmp_primitive_only_steps)
        self.qmp_gate_steps = int(qmp_gate_steps)
        self.qmp_commitment_min_steps = int(qmp_commitment_min_steps)
        self.qmp_commitment_max_steps = int(qmp_commitment_max_steps)
        self.qmp_held_min_lift_height = float(qmp_held_min_lift_height)
        self.qmp_held_max_ee_distance = float(qmp_held_max_ee_distance)
        self.qmp_held_max_gripper_qpos = float(qmp_held_max_gripper_qpos)
        self.qmp_stage_aware_mask = bool(qmp_stage_aware_mask)
        self.qmp_target_q_margin = float(qmp_target_q_margin)
        self.qmp_bc_steps = int(qmp_bc_steps)
        self.qmp_bc_coef = float(qmp_bc_coef)
        self.qmp_bc_batch_size = int(qmp_bc_batch_size)
        self.qmp_bc_buffer_size = int(qmp_bc_buffer_size)
        if self.qmp_commitment_min_steps < 1:
            raise ValueError("qmp_commitment_min_steps must be at least 1.")
        if self.qmp_commitment_max_steps < self.qmp_commitment_min_steps:
            raise ValueError(
                "qmp_commitment_max_steps must be greater than or equal to "
                "qmp_commitment_min_steps."
            )
        if self.qmp_target_q_margin < 0.0:
            raise ValueError("qmp_target_q_margin must be non-negative.")
        if self.qmp_bc_steps < 0 or self.qmp_bc_coef < 0.0:
            raise ValueError("QMP behavior cloning steps and coefficient must be non-negative.")
        if self.qmp_bc_batch_size < 1 or self.qmp_bc_buffer_size < 1:
            raise ValueError("QMP behavior cloning batch and buffer sizes must be positive.")

        self.qmp_last_selected_names = []
        self.qmp_last_q_values = {}
        self.qmp_last_candidate_names = []
        self.qmp_last_object_held = []
        self.qmp_last_gate_active = False
        self.qmp_last_primitive_only_active = False
        self.qmp_last_commitment_remaining = []
        self.qmp_last_valid_candidate_names = []
        self.qmp_last_target_margin_fallback = []
        self.qmp_last_bc_loss = np.nan
        self._qmp_committed_names = []
        self._qmp_commitment_remaining = np.zeros(0, dtype=np.int64)
        self._qmp_was_primitive_only = False
        self._qmp_bc_observations = None
        self._qmp_bc_actions = None
        self._qmp_bc_position = 0
        self._qmp_bc_full = False
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "primitive_policies",
            "_qmp_bc_observations",
            "_qmp_bc_actions",
            "_qmp_bc_position",
            "_qmp_bc_full",
        ]

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        if (
            self.qmp_bc_coef > 0.0
            and self.num_timesteps < self.qmp_bc_steps
            and self._qmp_bc_size() >= self.qmp_bc_batch_size
        ):
            self._train_behavior_clone()

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        selected_action = self._select_qmp_action(
            learning_starts=learning_starts,
            n_envs=n_envs,
            deterministic_target=False,
            allow_random=True,
            record_q_values=False,
        )
        if selected_action is None:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        scaled_action = self.policy.scale_action(selected_action)
        primitive_names = {primitive.name for primitive in self.primitive_policies}
        primitive_mask = np.asarray(
            [name in primitive_names for name in self.qmp_last_selected_names],
            dtype=bool,
        )
        if self.num_timesteps < self.qmp_bc_steps and np.any(primitive_mask):
            self._store_behavior_clone_samples(
                self._last_obs,
                scaled_action,
                primitive_mask,
            )
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1.0, 1.0)
            selected_action = self.policy.unscale_action(scaled_action)

        return selected_action, scaled_action

    def _store_behavior_clone_samples(self, observations, actions, sample_mask):
        observation_batch = self._numpy_observation_batch(observations)
        actions = self._ensure_action_batch(actions, len(sample_mask))

        if self._qmp_bc_observations is None:
            self._qmp_bc_position = 0
            self._qmp_bc_full = False
            if isinstance(observation_batch, dict):
                self._qmp_bc_observations = {
                    key: np.empty(
                        (self.qmp_bc_buffer_size, *value.shape[1:]),
                        dtype=np.float32,
                    )
                    for key, value in observation_batch.items()
                }
            else:
                self._qmp_bc_observations = np.empty(
                    (self.qmp_bc_buffer_size, *observation_batch.shape[1:]),
                    dtype=np.float32,
                )
            self._qmp_bc_actions = np.empty(
                (self.qmp_bc_buffer_size, actions.shape[1]),
                dtype=np.float32,
            )

        for env_index in np.flatnonzero(sample_mask):
            position = self._qmp_bc_position
            if isinstance(observation_batch, dict):
                for key, value in observation_batch.items():
                    self._qmp_bc_observations[key][position] = value[env_index]
            else:
                self._qmp_bc_observations[position] = observation_batch[env_index]
            self._qmp_bc_actions[position] = actions[env_index]

            self._qmp_bc_position = (position + 1) % self.qmp_bc_buffer_size
            if self._qmp_bc_position == 0:
                self._qmp_bc_full = True

    def _train_behavior_clone(self):
        buffer_size = self._qmp_bc_size()
        indices = np.random.randint(
            0,
            buffer_size,
            size=self.qmp_bc_batch_size,
        )
        if isinstance(self._qmp_bc_observations, dict):
            observation_tensor = {
                key: th.as_tensor(value[indices], device=self.device)
                for key, value in self._qmp_bc_observations.items()
            }
        else:
            observation_tensor = th.as_tensor(
                self._qmp_bc_observations[indices],
                device=self.device,
            )
        expert_actions = th.as_tensor(
            self._qmp_bc_actions[indices],
            device=self.device,
        )

        predicted_actions = self.actor(observation_tensor, deterministic=True)
        bc_loss = th.nn.functional.mse_loss(predicted_actions, expert_actions)
        self.actor.optimizer.zero_grad()
        (self.qmp_bc_coef * bc_loss).backward()
        self.actor.optimizer.step()

        self.qmp_last_bc_loss = float(bc_loss.detach().cpu().item())
        self.logger.record("train/qmp_bc_loss", self.qmp_last_bc_loss)
        self.logger.record("train/qmp_bc_coef", self.qmp_bc_coef)
        self.logger.record("train/qmp_bc_buffer_size", float(buffer_size))

    def _qmp_bc_size(self):
        if self._qmp_bc_observations is None:
            return 0
        return (
            self.qmp_bc_buffer_size
            if self._qmp_bc_full
            else self._qmp_bc_position
        )

    @staticmethod
    def _numpy_observation_batch(observations):
        if isinstance(observations, dict):
            result = {}
            for key, value in observations.items():
                value = np.asarray(value, dtype=np.float32)
                result[key] = value.reshape(1, -1) if value.ndim == 1 else value
            return result

        observations = np.asarray(observations, dtype=np.float32)
        return observations.reshape(1, -1) if observations.ndim == 1 else observations

    def predict_qmp(self, observation, deterministic=True, episode_start=False):
        """Select one action with the rollout QMP schedule and return diagnostics."""
        state_obs = target_obs_to_state_array(observation)
        n_envs = state_obs.shape[0]
        is_vectorized = np.asarray(
            observation["observation"] if isinstance(observation, dict) else observation
        ).ndim > 1

        self._last_obs = observation
        self._last_episode_starts = np.full(n_envs, episode_start, dtype=bool)
        selected_action = self._select_qmp_action(
            learning_starts=self.learning_starts,
            n_envs=n_envs,
            deterministic_target=deterministic,
            allow_random=not deterministic,
            record_q_values=True,
        )
        if selected_action is None:
            action, _ = self.predict(observation, deterministic=deterministic)
            return action, {
                "selected_policy": "target",
                "q_values": {},
                "gate_active": False,
                "primitive_only_active": False,
                "object_held": False,
                "commitment_remaining": 0,
                "valid_candidates": ["target"],
                "target_margin_fallback": False,
            }

        diagnostics = {
            "selected_policy": self.qmp_last_selected_names[0],
            "q_values": dict(self.qmp_last_q_values),
            "gate_active": self.qmp_last_gate_active,
            "primitive_only_active": self.qmp_last_primitive_only_active,
            "object_held": bool(self.qmp_last_object_held[0]),
            "commitment_remaining": int(self.qmp_last_commitment_remaining[0]),
            "valid_candidates": list(self.qmp_last_valid_candidate_names[0]),
            "target_margin_fallback": bool(
                self.qmp_last_target_margin_fallback[0]
            ),
        }
        if not is_vectorized:
            selected_action = selected_action[0]
        return selected_action, diagnostics

    def _select_qmp_action(
        self,
        learning_starts,
        n_envs,
        deterministic_target,
        allow_random,
        record_q_values,
    ):
        if not self.primitive_policies or not isinstance(self.action_space, spaces.Box):
            return None

        self._prepare_commitment_state(n_envs)
        candidate_names, candidate_actions = self._primitive_candidate_actions(n_envs)
        if not candidate_actions:
            return None

        primitive_only_active = self.num_timesteps < self.qmp_primitive_only_steps
        gate_active = self.num_timesteps < self.qmp_gate_steps
        if self._qmp_was_primitive_only and not primitive_only_active:
            self._clear_commitments()
        self._qmp_was_primitive_only = primitive_only_active

        object_held = target_obs_object_held(
            self._last_obs,
            min_lift_height=self.qmp_held_min_lift_height,
            max_ee_distance=self.qmp_held_max_ee_distance,
            max_gripper_qpos=self.qmp_held_max_gripper_qpos,
        )
        self.qmp_last_object_held = object_held.astype(bool).tolist()
        self.qmp_last_gate_active = bool(gate_active)
        self.qmp_last_primitive_only_active = bool(primitive_only_active)
        self.qmp_last_target_margin_fallback = [False] * n_envs

        warmup_steps = max(int(learning_starts), self.qmp_warmup_steps)
        if gate_active:
            q_values = None
            if record_q_values:
                _, q_values = self._evaluate_candidate_q_values(
                    candidate_names,
                    candidate_actions,
                    n_envs,
                )
            selected_action = self._select_gated_primitive(
                candidate_names,
                candidate_actions,
                object_held,
                n_envs,
                q_values=q_values,
            )
            self.qmp_last_valid_candidate_names = [
                ["insert" if held else "grasp"]
                for held in object_held
            ]
        elif primitive_only_active and self.num_timesteps < warmup_steps:
            q_values = None
            if record_q_values:
                _, q_values = self._evaluate_candidate_q_values(
                    candidate_names,
                    candidate_actions,
                    n_envs,
                )
            selected_action = self._select_warmup_action(
                candidate_names,
                candidate_actions,
                n_envs,
                allow_random=False,
                q_values=q_values,
            )
        elif primitive_only_active:
            valid_mask, stage_names = self._build_stage_candidate_mask(
                candidate_names,
                object_held,
            )
            selected_action = self._select_q_action(
                candidate_names,
                candidate_actions,
                n_envs,
                allow_random=False,
                valid_candidate_mask=valid_mask,
                stage_primitive_names=stage_names,
            )
        else:
            target_action, _ = self.predict(
                self._last_obs,
                deterministic=deterministic_target,
            )
            target_action = self._ensure_action_batch(target_action, n_envs)
            candidate_names = ["target", *candidate_names]
            candidate_actions = [target_action, *candidate_actions]
            valid_mask, stage_names = self._build_stage_candidate_mask(
                candidate_names,
                object_held,
            )
            selected_action = self._select_q_action(
                candidate_names,
                candidate_actions,
                n_envs,
                allow_random=allow_random,
                valid_candidate_mask=valid_mask,
                stage_primitive_names=stage_names,
            )

        self.qmp_last_commitment_remaining = (
            self._qmp_commitment_remaining.astype(int).tolist()
        )
        return selected_action

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

    def _build_stage_candidate_mask(self, candidate_names, object_held):
        n_envs = len(object_held)
        valid_mask = np.ones((n_envs, len(candidate_names)), dtype=bool)
        stage_names = ["insert" if held else "grasp" for held in object_held]
        if not self.qmp_stage_aware_mask:
            self.qmp_last_valid_candidate_names = [list(candidate_names)] * n_envs
            return valid_mask, stage_names

        name_to_index = {name: index for index, name in enumerate(candidate_names)}
        valid_mask.fill(False)
        if "target" in name_to_index:
            valid_mask[:, name_to_index["target"]] = True
        for env_index, stage_name in enumerate(stage_names):
            if stage_name not in name_to_index:
                raise ValueError(
                    f"Stage-aware QMP requires primitive '{stage_name}'."
                )
            valid_mask[env_index, name_to_index[stage_name]] = True

        self.qmp_last_valid_candidate_names = [
            [
                name
                for candidate_index, name in enumerate(candidate_names)
                if valid_mask[env_index, candidate_index]
            ]
            for env_index in range(n_envs)
        ]
        return valid_mask, stage_names

    def _select_gated_primitive(
        self,
        candidate_names,
        candidate_actions,
        object_held,
        n_envs,
        q_values=None,
    ):
        action_stack = np.stack(candidate_actions, axis=1)
        name_to_index = {name: index for index, name in enumerate(candidate_names)}
        missing_names = {"grasp", "insert"} - set(name_to_index)
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise ValueError(
                f"Primitive gate requires grasp and insert policies; missing: {missing}."
            )

        selected_names = []
        selected_action = np.empty_like(action_stack[:, 0])
        for env_index in range(n_envs):
            selected_name = "insert" if object_held[env_index] else "grasp"
            selected_index = name_to_index[selected_name]
            selected_action[env_index] = action_stack[env_index, selected_index]
            selected_names.append(selected_name)

            if (
                self._qmp_committed_names[env_index] == selected_name
                and self._qmp_commitment_remaining[env_index] > 0
            ):
                self._qmp_commitment_remaining[env_index] -= 1
            else:
                self._start_commitment(env_index, selected_name)

        self._store_qmp_selection(candidate_names, selected_names, q_values)
        return selected_action

    def _select_warmup_action(
        self,
        candidate_names,
        candidate_actions,
        n_envs,
        allow_random,
        q_values=None,
    ):
        action_stack = np.stack(candidate_actions, axis=1)
        selected_indices = np.random.randint(0, len(candidate_names), size=n_envs)
        selected_action, selected_names = self._apply_primitive_commitment(
            candidate_names,
            action_stack,
            selected_indices,
            allow_random=allow_random,
        )
        self._store_qmp_selection(candidate_names, selected_names, q_values)
        return selected_action

    def _select_q_action(
        self,
        candidate_names,
        candidate_actions,
        n_envs,
        allow_random,
        valid_candidate_mask=None,
        stage_primitive_names=None,
    ):
        action_stack = np.stack(candidate_actions, axis=1)
        q_array, mean_q_values = self._evaluate_candidate_q_values(
            candidate_names,
            candidate_actions,
            n_envs,
        )
        if valid_candidate_mask is None:
            valid_candidate_mask = np.ones_like(q_array, dtype=bool)
            self.qmp_last_valid_candidate_names = [list(candidate_names)] * n_envs
        masked_q_array = np.where(valid_candidate_mask, q_array, -np.inf)
        selected_indices = np.argmax(masked_q_array, axis=1)

        margin_fallback = np.zeros(n_envs, dtype=bool)
        if (
            self.qmp_target_q_margin > 0.0
            and "target" in candidate_names
            and stage_primitive_names is not None
        ):
            target_index = candidate_names.index("target")
            for env_index, stage_name in enumerate(stage_primitive_names):
                stage_index = candidate_names.index(stage_name)
                if (
                    selected_indices[env_index] == target_index
                    and q_array[env_index, target_index]
                    < q_array[env_index, stage_index] + self.qmp_target_q_margin
                ):
                    selected_indices[env_index] = stage_index
                    margin_fallback[env_index] = True

        self.qmp_last_target_margin_fallback = margin_fallback.tolist()
        selected_action, selected_names = self._apply_primitive_commitment(
            candidate_names,
            action_stack,
            selected_indices,
            allow_random=allow_random,
            valid_candidate_mask=valid_candidate_mask,
        )
        self._store_qmp_selection(candidate_names, selected_names, mean_q_values)
        return selected_action

    def _evaluate_candidate_q_values(
        self,
        candidate_names,
        candidate_actions,
        n_envs,
    ):
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
        mean_q_values = {
            name: float(np.mean(q_array[:, index]))
            for index, name in enumerate(candidate_names)
        }
        return q_array, mean_q_values

    def _apply_primitive_commitment(
        self,
        candidate_names,
        action_stack,
        proposed_indices,
        allow_random,
        valid_candidate_mask=None,
    ):
        name_to_index = {name: index for index, name in enumerate(candidate_names)}
        selected_indices = np.asarray(proposed_indices, dtype=np.int64).copy()
        decision_mask = np.ones(selected_indices.shape[0], dtype=bool)
        if valid_candidate_mask is None:
            valid_candidate_mask = np.ones(
                (selected_indices.shape[0], len(candidate_names)),
                dtype=bool,
            )

        for env_index, committed_name in enumerate(self._qmp_committed_names):
            if (
                committed_name in name_to_index
                and valid_candidate_mask[env_index, name_to_index[committed_name]]
                and self._qmp_commitment_remaining[env_index] > 0
            ):
                selected_indices[env_index] = name_to_index[committed_name]
                self._qmp_commitment_remaining[env_index] -= 1
                decision_mask[env_index] = False
            else:
                self._clear_commitment(env_index)

        selected_action = action_stack[
            np.arange(selected_indices.shape[0]), selected_indices
        ].copy()
        selected_names = [candidate_names[index] for index in selected_indices]
        if allow_random:
            selected_action, selected_names = self._apply_epsilon_random_actions(
                selected_action,
                selected_names,
                eligible_mask=decision_mask,
            )

        primitive_names = {primitive.name for primitive in self.primitive_policies}
        for env_index in np.flatnonzero(decision_mask):
            selected_name = selected_names[env_index]
            if selected_name in primitive_names:
                self._start_commitment(env_index, selected_name)
            else:
                self._clear_commitment(env_index)

        return selected_action, selected_names

    def _apply_epsilon_random_actions(
        self,
        selected_action,
        selected_names,
        eligible_mask=None,
    ):
        if self.qmp_epsilon <= 0.0:
            return selected_action, selected_names

        if eligible_mask is None:
            eligible_mask = np.ones(selected_action.shape[0], dtype=bool)
        for env_index in range(selected_action.shape[0]):
            if eligible_mask[env_index] and np.random.random() < self.qmp_epsilon:
                selected_action[env_index] = self.action_space.sample()
                selected_names[env_index] = "random"
        return selected_action, selected_names

    def _prepare_commitment_state(self, n_envs):
        if len(getattr(self, "_qmp_committed_names", [])) != n_envs:
            self._qmp_committed_names = [None] * n_envs
            self._qmp_commitment_remaining = np.zeros(n_envs, dtype=np.int64)

        episode_starts = getattr(self, "_last_episode_starts", None)
        if episode_starts is not None:
            starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
            for env_index in np.flatnonzero(starts[:n_envs]):
                self._clear_commitment(env_index)

    def _start_commitment(self, env_index, primitive_name):
        duration = np.random.randint(
            self.qmp_commitment_min_steps,
            self.qmp_commitment_max_steps + 1,
        )
        self._qmp_committed_names[env_index] = primitive_name
        self._qmp_commitment_remaining[env_index] = duration - 1

    def _clear_commitment(self, env_index):
        self._qmp_committed_names[env_index] = None
        self._qmp_commitment_remaining[env_index] = 0

    def _clear_commitments(self):
        for env_index in range(len(self._qmp_committed_names)):
            self._clear_commitment(env_index)

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
