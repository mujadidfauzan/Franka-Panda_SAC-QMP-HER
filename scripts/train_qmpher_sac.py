import argparse
import copy
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from panda_rl.algorithms import (
    PrimitiveCandidate,
    QMPSAC,
    grasp_auto_gripper_adapter,
    target_obs_to_grasp_obs,
    target_obs_to_insert_obs,
)
from panda_rl.envs import PandaQMPInsertEnv

STEP_PATTERN = re.compile(r"(\d+)")


class QMPHERTensorboardCallback(BaseCallback):
    """Log sparse insert metrics and QMP primitive selection diagnostics."""

    def _on_step(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if len(rewards) > 0:
            self.logger.record("reward/step_total", float(np.mean(rewards)))

        scalar_keys = {
            "reward_sparse": "reward/sparse_insert",
            "reward_success_bonus": "reward/success_bonus",
            "reward_total": "reward/env_total",
            "insert_distance": "metrics/insert_distance",
            "xy_distance": "metrics/xy_distance",
            "z_error": "metrics/z_error",
            "reach_distance": "metrics/reach_distance",
            "cube_lift_height": "metrics/cube_lift_height",
            "is_success": "metrics/is_success",
        }
        for info_key, log_key in scalar_keys.items():
            values = [float(info[info_key]) for info in infos if info_key in info]
            if values:
                self.logger.record(log_key, float(np.mean(values)))

        gripper_values = [
            float(np.mean(info["gripper_qpos"]))
            for info in infos
            if "gripper_qpos" in info
        ]
        if gripper_values:
            self.logger.record("metrics/gripper_qpos_mean", float(np.mean(gripper_values)))

        selected_names = getattr(self.model, "qmp_last_selected_names", [])
        candidate_names = list(getattr(self.model, "qmp_last_candidate_names", []))
        for name in [*candidate_names, "random"]:
            if selected_names:
                selected_ratio = np.mean([selected == name for selected in selected_names])
                self.logger.record(f"qmp/selected_{name}", float(selected_ratio))

        for name, q_value in getattr(self.model, "qmp_last_q_values", {}).items():
            self.logger.record(f"qmp/q_{name}", float(q_value))

        object_held = getattr(self.model, "qmp_last_object_held", [])
        if object_held:
            self.logger.record("qmp/object_held", float(np.mean(object_held)))
        self.logger.record(
            "qmp/gate_active",
            float(getattr(self.model, "qmp_last_gate_active", False)),
        )
        self.logger.record(
            "qmp/primitive_only_active",
            float(getattr(self.model, "qmp_last_primitive_only_active", False)),
        )
        commitment_remaining = getattr(
            self.model,
            "qmp_last_commitment_remaining",
            [],
        )
        if commitment_remaining:
            self.logger.record(
                "qmp/commitment_remaining",
                float(np.mean(commitment_remaining)),
            )

        return True


class PeriodicArtifactCallback(BaseCallback):
    """Save target model, one rolling HER buffer, and annotated QMP eval video."""

    def __init__(
        self,
        save_freq,
        checkpoint_dir,
        replay_buffer_dir,
        video_dir,
        video_length=350,
        video_fps=50,
        seed=0,
        save_video=True,
        terminate_on_success=True,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.replay_buffer_dir = Path(replay_buffer_dir)
        self.video_dir = Path(video_dir)
        self.video_length = video_length
        self.video_fps = video_fps
        self.seed = seed
        self.save_video = save_video
        self.terminate_on_success = terminate_on_success
        self.last_save_step = None

    def _init_callback(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.last_save_step is None:
            self.last_save_step = int(getattr(self.model, "num_timesteps", 0))

    def _on_step(self):
        if self.num_timesteps - self.last_save_step < self.save_freq:
            return True

        self.last_save_step = self.num_timesteps
        self._save_artifacts(self.num_timesteps)
        return True

    def _save_artifacts(self, step):
        step_label = str(step)
        model_path = self.checkpoint_dir / f"checkpoint_{step_label}.zip"
        replay_buffer_path = self.replay_buffer_dir / f"replay_buffer_{step_label}.pkl"

        self.model.save(str(model_path))
        self.model.save_replay_buffer(str(replay_buffer_path))
        self._remove_previous_replay_buffers(replay_buffer_path)

        if self.save_video:
            video_path = self.video_dir / f"eval_{step_label}_steps.mp4"
            try:
                self._save_eval_video(video_path, seed=self.seed + int(step))
            except Exception as exc:
                print(f"Skipping eval video at {step} timesteps: {exc}")

        if self.verbose:
            print(f"Saved QMP-HER SAC artifacts at {step} timesteps.")

    def _remove_previous_replay_buffers(self, current_path):
        for replay_path in self.replay_buffer_dir.glob("replay_buffer_*.pkl"):
            if replay_path != current_path:
                replay_path.unlink(missing_ok=True)

    def _save_eval_video(self, video_path, seed):
        try:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise ImportError(
                "Annotated video saving requires imageio, ffmpeg, and Pillow. "
                "Install dependencies from requirements.txt."
            ) from exc

        eval_env = PandaQMPInsertEnv(
            render_mode="rgb_array",
            randomize_object=True,
            randomize_socket=True,
            terminate_on_success=self.terminate_on_success,
        )
        frames = []
        selector_state = self._capture_qmp_selector_state()
        numpy_random_state = np.random.get_state()

        try:
            self._reset_qmp_selector_for_eval()
            obs, info = eval_env.reset(seed=seed)
            first_frame = eval_env.render()
            if first_frame is not None:
                frames.append(
                    self._annotate_eval_frame(
                        first_frame,
                        {
                            "selected_policy": "reset",
                            "q_values": {},
                            "gate_active": self.num_timesteps
                            < self.model.qmp_gate_steps,
                            "primitive_only_active": self.num_timesteps
                            < self.model.qmp_primitive_only_steps,
                            "object_held": False,
                            "commitment_remaining": 0,
                        },
                        eval_step=0,
                        training_step=self.num_timesteps,
                        reward=0.0,
                        info=info,
                        image_types=(Image, ImageDraw, ImageFont),
                    )
                )

            for eval_step in range(1, self.video_length + 1):
                action, diagnostics = self.model.predict_qmp(
                    obs,
                    deterministic=True,
                    episode_start=eval_step == 1,
                )
                obs, reward, terminated, truncated, info = eval_env.step(action)
                frame = eval_env.render()
                if frame is not None:
                    frames.append(
                        self._annotate_eval_frame(
                            frame,
                            diagnostics,
                            eval_step=eval_step,
                            training_step=self.num_timesteps,
                            reward=reward,
                            info=info,
                            image_types=(Image, ImageDraw, ImageFont),
                        )
                    )
                if terminated or truncated:
                    break
        finally:
            np.random.set_state(numpy_random_state)
            self._restore_qmp_selector_state(selector_state)
            eval_env.close()

        if frames:
            imageio.mimsave(video_path, frames, fps=self.video_fps)

    def _capture_qmp_selector_state(self):
        attribute_names = (
            "_last_obs",
            "_last_episode_starts",
            "_qmp_committed_names",
            "_qmp_commitment_remaining",
            "_qmp_was_primitive_only",
            "qmp_last_selected_names",
            "qmp_last_q_values",
            "qmp_last_candidate_names",
            "qmp_last_object_held",
            "qmp_last_gate_active",
            "qmp_last_primitive_only_active",
            "qmp_last_commitment_remaining",
        )
        return {
            name: copy.deepcopy(getattr(self.model, name))
            for name in attribute_names
        }

    def _restore_qmp_selector_state(self, state):
        for name, value in state.items():
            setattr(self.model, name, value)

    def _reset_qmp_selector_for_eval(self):
        self.model._qmp_committed_names = []
        self.model._qmp_commitment_remaining = np.zeros(0, dtype=np.int64)
        self.model._qmp_was_primitive_only = False

    @staticmethod
    def _annotate_eval_frame(
        frame,
        diagnostics,
        eval_step,
        training_step,
        reward,
        info,
        image_types,
    ):
        Image, ImageDraw, ImageFont = image_types
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font_size = max(13, min(18, image.width // 42))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
            bold_font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
            bold_font = font

        selected_policy = diagnostics.get("selected_policy", "unknown")
        q_values = diagnostics.get("q_values", {})

        def format_q(policy_name):
            value = q_values.get(policy_name)
            return "N/A" if value is None else f"{float(value):+.4f}"

        selected_q = format_q(selected_policy)
        phase = "GATED PRIMITIVES" if diagnostics.get("gate_active") else "Q SELECT"
        held = "YES" if diagnostics.get("object_held") else "NO"
        commitment = int(diagnostics.get("commitment_remaining", 0))
        insert_distance = float(info.get("insert_distance", np.nan))
        lines = [
            f"Eval step {eval_step:03d} | train {int(training_step):d}",
            f"Selected: {selected_policy.upper()} | Q: {selected_q}",
            f"Q target {format_q('target')} | grasp {format_q('grasp')}",
            f"Q insert {format_q('insert')}",
            f"Mode: {phase} | held: {held} | commit: {commitment}",
            f"Reward: {float(reward):+.2f} | insert dist: {insert_distance:.4f} m",
        ]

        padding = 10
        line_gap = 4
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_width = max(box[2] - box[0] for box in boxes)
        line_height = max(box[3] - box[1] for box in boxes)
        panel_width = min(image.width - 16, text_width + 2 * padding)
        panel_height = len(lines) * line_height + (len(lines) - 1) * line_gap + 2 * padding
        panel_xy = (8, 8, 8 + panel_width, 8 + panel_height)
        draw.rectangle(panel_xy, fill=(8, 12, 18, 210), outline=(235, 235, 235, 180))

        policy_colors = {
            "target": (80, 210, 255, 255),
            "grasp": (90, 235, 130, 255),
            "insert": (255, 190, 70, 255),
            "random": (255, 100, 100, 255),
            "reset": (220, 220, 220, 255),
        }
        y = 8 + padding
        for line_index, line in enumerate(lines):
            color = (
                policy_colors.get(selected_policy, (255, 255, 255, 255))
                if line_index == 1
                else (245, 245, 245, 255)
            )
            draw.text(
                (8 + padding, y),
                line,
                font=bold_font if line_index == 1 else font,
                fill=color,
            )
            y += line_height + line_gap

        return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def make_env(args, seed, monitor_dir, terminate_on_success):
    env = PandaQMPInsertEnv(
        randomize_object=not args.fixed_object,
        randomize_socket=not args.fixed_socket,
        max_steps=args.max_steps,
        insert_tolerance=args.insert_tolerance,
        success_bonus=args.success_bonus,
        terminate_on_success=terminate_on_success,
    )
    env = Monitor(env, filename=str(monitor_dir / "monitor.csv"))
    env.reset(seed=seed)
    return env


def clean_run_name(run_name):
    if run_name is None:
        return None
    return Path(str(run_name)).name.replace(" ", "_")


def step_from_path(path):
    matches = STEP_PATTERN.findall(Path(path).stem)
    return int(matches[-1]) if matches else -1


def latest_file(paths):
    paths = [Path(path) for path in paths]
    return (
        max(paths, key=lambda path: (step_from_path(path), path.stat().st_mtime))
        if paths
        else None
    )


def latest_run_dir(pattern):
    candidates = [
        path
        for path in (PROJECT_ROOT / "models").glob(pattern)
        if path.is_dir()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_model_path(explicit_path, run_pattern, label):
    if explicit_path:
        model_path = Path(explicit_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{label} model tidak ditemukan: {model_path}")
        return model_path

    run_dir = latest_run_dir(run_pattern)
    if run_dir is None:
        raise FileNotFoundError(
            f"Tidak ada run {label} di models/{run_pattern}. "
            f"Pakai --{label}-model untuk memilih checkpoint."
        )

    candidates = list((run_dir / "checkpoints").glob("*.zip"))
    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        candidates.append(final_model)

    model_path = latest_file(candidates)
    if model_path is None:
        raise FileNotFoundError(f"Tidak ada model .zip untuk {label} di {run_dir}.")
    return model_path


def infer_run_name_from_checkpoint(checkpoint_path):
    path = Path(checkpoint_path).resolve()
    try:
        relative = path.relative_to((PROJECT_ROOT / "models").resolve())
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) >= 3 else None


def resolve_resume_paths(args, run_name):
    if not (args.resume or args.resume_checkpoint or args.resume_replay_buffer):
        return None, None, run_name

    checkpoint_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    replay_buffer_path = (
        Path(args.resume_replay_buffer) if args.resume_replay_buffer else None
    )

    if checkpoint_path is None:
        run_dir = (
            PROJECT_ROOT / "models" / run_name
            if run_name
            else latest_run_dir("panda_qmpher_sac*")
        )
        if run_dir is None:
            raise FileNotFoundError("Tidak ada folder run panda_qmpher_sac di models/.")
        checkpoint_path = latest_file((run_dir / "checkpoints").glob("*.zip"))
        if checkpoint_path is None:
            raise FileNotFoundError(f"Tidak ada checkpoint di {run_dir / 'checkpoints'}.")
        if run_name is None:
            run_name = run_dir.name

    if run_name is None:
        run_name = infer_run_name_from_checkpoint(checkpoint_path)

    if replay_buffer_path is None and run_name:
        replay_buffer_path = latest_file(
            (PROJECT_ROOT / "models" / run_name / "replay_buffer").glob("*.pkl")
        )

    if replay_buffer_path is not None and not replay_buffer_path.exists():
        raise FileNotFoundError(f"Replay buffer tidak ditemukan: {replay_buffer_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    return checkpoint_path, replay_buffer_path, run_name


def build_primitive_policies(args):
    primitives = []

    if not args.no_grasp_primitive:
        grasp_path = resolve_model_path(
            args.grasp_model,
            "panda_grasp_sac*",
            "grasp",
        )
        grasp_model = SAC.load(str(grasp_path), device=args.device)

        def project_grasp_obs(target_obs):
            return target_obs_to_grasp_obs(
                target_obs,
                lift_height=args.grasp_lift_height,
                cube_half_size=0.02,
            )

        def adapt_grasp_action(actions, target_obs):
            return grasp_auto_gripper_adapter(
                actions,
                target_obs,
                close_distance=args.grasp_close_distance,
            )

        primitives.append(
            PrimitiveCandidate(
                name="grasp",
                model=grasp_model,
                obs_projector=project_grasp_obs,
                action_adapter=adapt_grasp_action,
                deterministic=True,
            )
        )
        print(f"Loaded grasp primitive: {grasp_path}")

    if not args.no_insert_primitive:
        insert_path = resolve_model_path(
            args.insert_model,
            "panda_insert_sac*",
            "insert",
        )
        insert_model = SAC.load(str(insert_path), device=args.device)
        primitives.append(
            PrimitiveCandidate(
                name="insert",
                model=insert_model,
                obs_projector=target_obs_to_insert_obs,
                deterministic=True,
            )
        )
        print(f"Loaded insert primitive: {insert_path}")

    if not primitives:
        raise ValueError("Minimal satu primitive harus aktif untuk QMP-HER.")

    if args.primitive_gate_steps > 0:
        primitive_names = {primitive.name for primitive in primitives}
        missing_names = {"grasp", "insert"} - primitive_names
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise ValueError(
                "Primitive gating membutuhkan model grasp dan insert. "
                f"Primitive yang belum tersedia: {missing}."
            )

    return primitives


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--grasp-model", type=str, default=None)
    parser.add_argument("--insert-model", type=str, default=None)
    parser.add_argument("--no-grasp-primitive", action="store_true")
    parser.add_argument("--no-insert-primitive", action="store_true")
    parser.add_argument("--fixed-object", action="store_true")
    parser.add_argument("--fixed-socket", action="store_true")
    parser.add_argument("--no-terminate-on-success", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--video-length", type=int, default=350)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--insert-tolerance", type=float, default=0.01)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--grasp-lift-height", type=float, default=0.03)
    parser.add_argument("--grasp-close-distance", type=float, default=0.01)
    parser.add_argument("--qmp-epsilon", type=float, default=0.05)
    parser.add_argument("--qmp-warmup-steps", type=int, default=None)
    parser.add_argument("--primitive-only-steps", type=int, default=500_000)
    parser.add_argument("--primitive-gate-steps", type=int, default=500_000)
    parser.add_argument("--primitive-commitment-min-steps", type=int, default=10)
    parser.add_argument("--primitive-commitment-max-steps", type=int, default=30)
    parser.add_argument("--held-min-lift-height", type=float, default=0.01)
    parser.add_argument("--held-max-ee-distance", type=float, default=0.06)
    parser.add_argument("--held-max-gripper-qpos", type=float, default=0.03)
    parser.add_argument("--her-n-sampled-goal", type=int, default=4)
    parser.add_argument("--her-goal-selection-strategy", type=str, default="future")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume dari checkpoint QMP-HER terbaru. Pakai --run-name untuk memilih run.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path checkpoint .zip target policy tertentu untuk dilanjutkan.",
    )
    parser.add_argument(
        "--resume-replay-buffer",
        type=str,
        default=None,
        help="Path HER replay buffer .pkl tertentu. Jika kosong, dipakai yang terbaru.",
    )
    parser.add_argument(
        "--fresh-timesteps",
        action="store_true",
        help="Reset counter timestep saat resume. Default resume melanjutkan counter.",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.primitive_gate_steps > args.primitive_only_steps:
        raise ValueError(
            "--primitive-gate-steps cannot exceed --primitive-only-steps."
        )
    if args.primitive_commitment_min_steps < 1:
        raise ValueError("--primitive-commitment-min-steps must be at least 1.")
    if args.primitive_commitment_max_steps < args.primitive_commitment_min_steps:
        raise ValueError(
            "--primitive-commitment-max-steps must be greater than or equal to "
            "--primitive-commitment-min-steps."
        )

    run_name = clean_run_name(args.run_name)
    resume_checkpoint, resume_replay_buffer, run_name = resolve_resume_paths(
        args,
        run_name,
    )
    run_name = run_name or datetime.now().strftime("panda_qmpher_sac_%Y%m%d_%H%M%S")

    run_log_dir = PROJECT_ROOT / "logs" / run_name
    checkpoint_dir = PROJECT_ROOT / "models" / run_name / "checkpoints"
    replay_buffer_dir = PROJECT_ROOT / "models" / run_name / "replay_buffer"
    video_dir = PROJECT_ROOT / "logs" / run_name / "videos"
    tensorboard_dir = PROJECT_ROOT / "logs" / "tensorboard"
    terminate_on_success = not args.no_terminate_on_success
    qmp_warmup_steps = (
        args.learning_starts if args.qmp_warmup_steps is None else args.qmp_warmup_steps
    )

    run_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args=args,
        seed=args.seed,
        monitor_dir=run_log_dir,
        terminate_on_success=terminate_on_success,
    )
    primitive_policies = build_primitive_policies(args)
    print(
        "QMP schedule: "
        f"primitive-only for {args.primitive_only_steps} steps, "
        f"grasp/insert gate for {args.primitive_gate_steps} steps, "
        "then target + grasp + insert candidates; "
        f"primitive commitment {args.primitive_commitment_min_steps}-"
        f"{args.primitive_commitment_max_steps} steps."
    )

    if resume_checkpoint is not None:
        model = QMPSAC.load(
            str(resume_checkpoint),
            env=env,
            tensorboard_log=str(tensorboard_dir),
            device=args.device,
            primitive_policies=primitive_policies,
            qmp_warmup_steps=qmp_warmup_steps,
            qmp_epsilon=args.qmp_epsilon,
            qmp_primitive_only_steps=args.primitive_only_steps,
            qmp_gate_steps=args.primitive_gate_steps,
            qmp_commitment_min_steps=args.primitive_commitment_min_steps,
            qmp_commitment_max_steps=args.primitive_commitment_max_steps,
            qmp_held_min_lift_height=args.held_min_lift_height,
            qmp_held_max_ee_distance=args.held_max_ee_distance,
            qmp_held_max_gripper_qpos=args.held_max_gripper_qpos,
            verbose=1,
        )
        if resume_replay_buffer is not None:
            model.load_replay_buffer(str(resume_replay_buffer))
        print(f"Resuming QMP-HER SAC from checkpoint: {resume_checkpoint}")
        if resume_replay_buffer is not None:
            print(f"Loaded HER replay buffer: {resume_replay_buffer}")
        else:
            print("Replay buffer tidak ditemukan/dipilih, lanjut tanpa replay buffer lama.")
    else:
        model = QMPSAC(
            policy="MultiInputPolicy",
            env=env,
            primitive_policies=primitive_policies,
            qmp_warmup_steps=qmp_warmup_steps,
            qmp_epsilon=args.qmp_epsilon,
            qmp_primitive_only_steps=args.primitive_only_steps,
            qmp_gate_steps=args.primitive_gate_steps,
            qmp_commitment_min_steps=args.primitive_commitment_min_steps,
            qmp_commitment_max_steps=args.primitive_commitment_max_steps,
            qmp_held_min_lift_height=args.held_min_lift_height,
            qmp_held_max_ee_distance=args.held_max_ee_distance,
            qmp_held_max_gripper_qpos=args.held_max_gripper_qpos,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": args.her_n_sampled_goal,
                "goal_selection_strategy": args.her_goal_selection_strategy,
            },
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            verbose=1,
            device=args.device,
        )

    callbacks = CallbackList(
        [
            QMPHERTensorboardCallback(),
            PeriodicArtifactCallback(
                save_freq=args.save_freq,
                checkpoint_dir=checkpoint_dir,
                replay_buffer_dir=replay_buffer_dir,
                video_dir=video_dir,
                video_length=args.video_length,
                video_fps=args.video_fps,
                seed=args.seed,
                save_video=not args.no_video,
                terminate_on_success=terminate_on_success,
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            tb_log_name=run_name,
            reset_num_timesteps=(resume_checkpoint is None or args.fresh_timesteps),
            progress_bar=False,
        )
    finally:
        final_model_path = PROJECT_ROOT / "models" / run_name / "final_model.zip"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(final_model_path))
        env.close()

    print(f"Training complete. Final target model saved to: {final_model_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run artifacts: {PROJECT_ROOT / 'models' / run_name}")


if __name__ == "__main__":
    main()
