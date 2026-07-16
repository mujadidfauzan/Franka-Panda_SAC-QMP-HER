import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _install_dependency_stubs():
    if importlib.util.find_spec("gymnasium") is None:
        gymnasium = types.ModuleType("gymnasium")
        spaces = types.ModuleType("gymnasium.spaces")

        class Box:
            pass

        spaces.Box = Box
        gymnasium.spaces = spaces
        sys.modules["gymnasium"] = gymnasium
        sys.modules["gymnasium.spaces"] = spaces

    if importlib.util.find_spec("stable_baselines3") is None:
        stable_baselines3 = types.ModuleType("stable_baselines3")

        class SAC:
            pass

        stable_baselines3.SAC = SAC
        sys.modules["stable_baselines3"] = stable_baselines3


_install_dependency_stubs()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_rl.algorithms.qmp_sac import (  # noqa: E402
    QMPSAC,
    target_obs_grasp_verified,
    target_obs_object_lost,
    target_obs_to_grasp_obs,
    target_obs_to_insert_obs,
)


def make_target_obs(
    ee_pos=(0.45, 0.0, 0.30),
    cube_pos=(0.45, 0.0, 0.02),
    gripper_qpos=(0.04, 0.04),
    desired_goal=(0.60, 0.10, 0.02),
):
    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    cube_pos = np.asarray(cube_pos, dtype=np.float32)
    state = np.concatenate(
        [
            ee_pos,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            cube_pos,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            cube_pos - ee_pos,
            np.zeros(7, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.asarray(gripper_qpos, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
    )
    return {
        "observation": state,
        "achieved_goal": cube_pos.copy(),
        "desired_goal": np.asarray(desired_goal, dtype=np.float32),
    }


def make_selector():
    selector = QMPSAC.__new__(QMPSAC)
    selector.num_timesteps = 600_000
    selector.qmp_primitive_only_steps = 500_000
    selector.qmp_target_max_admission_probability = 0.0
    selector.qmp_target_admission_ramp_steps = 1_000_000
    selector.qmp_held_min_lift_height = 0.01
    selector.qmp_held_max_ee_distance = 0.06
    selector.qmp_held_max_gripper_qpos = 0.03
    selector.qmp_lost_min_ee_distance = 0.10
    selector.qmp_lost_min_gripper_qpos = 0.035
    selector.qmp_stage_loss_patience = 3
    selector.qmp_stage_commitment = True
    selector.qmp_commitment_min_steps = 10
    selector.qmp_commitment_max_steps = 30
    selector.qmp_stage_aware_mask = True
    selector.qmp_target_q_margin = 0.5
    selector.qmp_epsilon = 0.05
    selector.qmp_epsilon_end_steps = 500_000
    selector.qmp_bc_coef = 1.0
    selector.qmp_bc_anneal_start_steps = 500_000
    selector.qmp_bc_steps = 1_500_000
    selector._qmp_committed_names = []
    selector._qmp_commitment_remaining = np.zeros(0, dtype=np.int64)
    selector._qmp_stage_locked = np.zeros(0, dtype=bool)
    selector._qmp_stage_names = []
    selector._qmp_stage_loss_counts = np.zeros(0, dtype=np.int64)
    selector._qmp_target_stage_admitted = np.zeros(0, dtype=bool)
    selector._last_episode_starts = np.array([True])
    selector.qmp_last_valid_candidate_names = []
    selector.qmp_last_target_margin_fallback = []
    return selector


class ObservationProjectionTests(unittest.TestCase):
    def test_goal_agnostic_state_projects_to_legacy_insert_shape(self):
        obs = make_target_obs()
        projected = target_obs_to_insert_obs(obs)

        self.assertEqual(obs["observation"].shape, (39,))
        self.assertEqual(projected.shape, (1, 45))
        np.testing.assert_allclose(projected[0, 14:17], obs["desired_goal"])
        np.testing.assert_allclose(
            projected[0, 17:20],
            obs["desired_goal"] - obs["achieved_goal"],
        )

    def test_grasp_projection_keeps_primitive_shape(self):
        projected = target_obs_to_grasp_obs(make_target_obs())
        self.assertEqual(projected.shape, (1, 43))

    def test_legacy_insert_projection_is_unchanged(self):
        projected = target_obs_to_insert_obs(make_target_obs())
        round_trip = target_obs_to_insert_obs(projected)
        np.testing.assert_allclose(round_trip, projected)


class StageMachineTests(unittest.TestCase):
    def test_insert_stage_stays_latched_while_cube_is_lowered(self):
        selector = make_selector()
        selector._last_obs = make_target_obs()
        selector._prepare_selector_state(1)
        selector._last_episode_starts[:] = False

        selector._last_obs = make_target_obs(
            ee_pos=(0.45, 0.0, 0.04),
            cube_pos=(0.45, 0.0, 0.04),
            gripper_qpos=(0.02, 0.02),
        )
        selector._update_stage_state()
        self.assertEqual(selector._qmp_stage_names, ["insert"])

        selector._last_obs = make_target_obs(
            ee_pos=(0.60, 0.10, 0.02),
            cube_pos=(0.60, 0.10, 0.02),
            gripper_qpos=(0.02, 0.02),
        )
        verified = target_obs_grasp_verified(selector._last_obs)
        lost = target_obs_object_lost(selector._last_obs)
        selector._update_stage_state()

        self.assertFalse(bool(verified[0]))
        self.assertFalse(bool(lost[0]))
        self.assertEqual(selector._qmp_stage_names, ["insert"])

    def test_insert_stage_returns_to_grasp_only_after_loss_patience(self):
        selector = make_selector()
        selector._last_obs = make_target_obs(
            ee_pos=(0.45, 0.0, 0.04),
            cube_pos=(0.45, 0.0, 0.04),
            gripper_qpos=(0.02, 0.02),
        )
        selector._prepare_selector_state(1)
        selector._last_episode_starts[:] = False
        selector._update_stage_state()

        selector._last_obs = make_target_obs(gripper_qpos=(0.04, 0.04))
        selector._update_stage_state()
        selector._update_stage_state()
        self.assertEqual(selector._qmp_stage_names, ["insert"])

        selector._update_stage_state()
        self.assertEqual(selector._qmp_stage_names, ["grasp"])

    def test_stage_locked_primitive_cannot_be_interrupted_by_target(self):
        selector = make_selector()
        selector.primitive_policies = [SimpleNamespace(name="grasp")]
        selector._qmp_committed_names = ["grasp"]
        selector._qmp_commitment_remaining = np.array([0], dtype=np.int64)
        selector._qmp_stage_locked = np.array([True])

        action_stack = np.zeros((1, 2, 7), dtype=np.float32)
        action_stack[:, 0] = 1.0
        action_stack[:, 1] = -1.0
        selected_action, selected_names = selector._apply_primitive_commitment(
            ["target", "grasp"],
            action_stack,
            proposed_indices=np.array([0]),
            allow_random=False,
            valid_candidate_mask=np.ones((1, 2), dtype=bool),
        )

        self.assertEqual(selected_names, ["grasp"])
        np.testing.assert_allclose(selected_action, -1.0)

    def test_target_admission_masks_target_per_stage(self):
        selector = make_selector()
        selector.qmp_last_valid_candidate_names = []
        mask, stages = selector._build_stage_candidate_mask(
            ["target", "grasp", "insert"],
            np.array([False, True]),
            target_stage_admitted=np.array([False, True]),
        )

        self.assertEqual(stages, ["grasp", "insert"])
        np.testing.assert_array_equal(
            mask,
            np.array(
                [
                    [False, True, False],
                    [True, False, True],
                ]
            ),
        )

    def test_target_must_clear_q_margin_before_it_can_interrupt(self):
        selector = make_selector()
        selector.primitive_policies = [SimpleNamespace(name="grasp")]
        selector._qmp_committed_names = [None]
        selector._qmp_commitment_remaining = np.array([0], dtype=np.int64)
        selector._qmp_stage_locked = np.array([False])
        selector._evaluate_candidate_q_values = lambda *args: (
            np.array([[1.2, 1.0]], dtype=np.float32),
            {"target": 1.2, "grasp": 1.0},
        )

        actions = [
            np.ones((1, 7), dtype=np.float32),
            -np.ones((1, 7), dtype=np.float32),
        ]
        selected = selector._select_q_action(
            ["target", "grasp"],
            actions,
            n_envs=1,
            allow_random=False,
            valid_candidate_mask=np.ones((1, 2), dtype=bool),
            stage_primitive_names=["grasp"],
        )

        np.testing.assert_allclose(selected, -1.0)
        self.assertEqual(selector.qmp_last_selected_names, ["grasp"])
        self.assertEqual(selector.qmp_last_target_margin_fallback, [True])
        self.assertTrue(bool(selector._qmp_stage_locked[0]))


class ScheduleTests(unittest.TestCase):
    def test_bc_coefficient_anneals_after_primitive_phase(self):
        selector = make_selector()
        selector.num_timesteps = 500_000
        self.assertAlmostEqual(selector._current_bc_coef(), 1.0)
        selector.num_timesteps = 1_000_000
        self.assertAlmostEqual(selector._current_bc_coef(), 0.5)
        selector.num_timesteps = 1_500_000
        self.assertAlmostEqual(selector._current_bc_coef(), 0.0)

    def test_target_admission_ramps_and_random_epsilon_ends(self):
        selector = make_selector()
        selector.qmp_target_max_admission_probability = 0.2
        selector.num_timesteps = 500_000
        self.assertAlmostEqual(selector._current_target_admission_probability(), 0.0)
        selector.num_timesteps = 1_000_000
        self.assertAlmostEqual(selector._current_target_admission_probability(), 0.1)
        self.assertAlmostEqual(selector._current_qmp_epsilon(), 0.0)
        selector.num_timesteps = 1_500_000
        self.assertAlmostEqual(selector._current_target_admission_probability(), 0.2)


if __name__ == "__main__":
    unittest.main()
