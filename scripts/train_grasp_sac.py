import argparse
import sys
from datetime import datetime
from pathlib import Path
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from panda_rl.envs import PandaGraspEnv

STEP_PATTERN = re.compile(r"(\d+)")


class GraspTensorboardCallback(BaseCallback):
    """Log grasp reward components and task metrics from env info dicts."""

    def _on_step(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if len(rewards) > 0:
            self.logger.record("reward/step_total", float(np.mean(rewards)))

        scalar_keys = {
            "reward_reach": "reward/reach",
            "reward_lift": "reward/lift",
            "reward_close_bonus": "reward/close_bonus",
            "reward_success_bonus": "reward/success_bonus",
            "reward_total": "reward/env_total",
            "reach_distance": "metrics/reach_distance",
            "lift_height": "metrics/lift_height",
            "lift_progress": "metrics/lift_progress",
            "action_penalty": "metrics/action_penalty",
            "auto_gripper_closed": "metrics/auto_gripper_closed",
            "is_success": "metrics/is_success",
        }

        for info_key, log_key in scalar_keys.items():
            values = [float(info[info_key]) for info in infos if info_key in info]
            if values:
                self.logger.record(log_key, float(np.mean(values)))

        return True


class PeriodicArtifactCallback(BaseCallback):
    """Save model, one rolling replay buffer, and eval video periodically."""

    def __init__(
        self,
        save_freq,
        checkpoint_dir,
        replay_buffer_dir,
        video_dir,
        video_length=250,
        video_fps=50,
        seed=0,
        save_video=True,
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
            self._save_eval_video(video_path, seed=self.seed + int(step))

        if self.verbose:
            print(f"Saved SAC grasp artifacts at {step} timesteps.")

    def _remove_previous_replay_buffers(self, current_path):
        for replay_path in self.replay_buffer_dir.glob("replay_buffer_*.pkl"):
            if replay_path != current_path:
                replay_path.unlink(missing_ok=True)

    def _save_eval_video(self, video_path, seed):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Video saving requires imageio with ffmpeg support. "
                "Install dependencies from requirements.txt."
            ) from exc

        eval_env = PandaGraspEnv(render_mode="rgb_array", randomize_object=True)
        frames = []

        try:
            obs, _ = eval_env.reset(seed=seed)
            first_frame = eval_env.render()
            if first_frame is not None:
                frames.append(first_frame)

            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = eval_env.step(action)
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)
                if terminated or truncated:
                    break
        finally:
            eval_env.close()

        if frames:
            imageio.mimsave(video_path, frames, fps=self.video_fps)


def make_env(seed, monitor_dir, randomize_object=True):
    env = PandaGraspEnv(randomize_object=randomize_object)
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
    return max(paths, key=step_from_path) if paths else None


def latest_run_dir():
    candidates = [path for path in (PROJECT_ROOT / "models").glob("*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


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
        run_dir = PROJECT_ROOT / "models" / run_name if run_name else latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError("Tidak ada folder run di models/ untuk resume.")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--fixed-object", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-length", type=int, default=250)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume dari checkpoint terbaru. Pakai --run-name untuk memilih run.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path checkpoint .zip tertentu untuk dilanjutkan.",
    )
    parser.add_argument(
        "--resume-replay-buffer",
        type=str,
        default=None,
        help="Path replay buffer .pkl tertentu. Jika kosong, dipakai yang terbaru.",
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

    run_name = clean_run_name(args.run_name)
    resume_checkpoint, resume_replay_buffer, run_name = resolve_resume_paths(
        args,
        run_name,
    )
    run_name = run_name or datetime.now().strftime("panda_grasp_sac_%Y%m%d_%H%M%S")

    run_log_dir = PROJECT_ROOT / "logs" / run_name
    checkpoint_dir = PROJECT_ROOT / "models" / run_name / "checkpoints"
    replay_buffer_dir = PROJECT_ROOT / "models" / run_name / "replay_buffer"
    video_dir = PROJECT_ROOT / "logs" / run_name / "videos"
    tensorboard_dir = PROJECT_ROOT / "logs" / "tensorboard"

    run_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        seed=args.seed,
        monitor_dir=run_log_dir,
        randomize_object=not args.fixed_object,
    )

    if resume_checkpoint is not None:
        model = SAC.load(
            str(resume_checkpoint),
            env=env,
            tensorboard_log=str(tensorboard_dir),
            device=args.device,
            verbose=1,
        )
        if resume_replay_buffer is not None:
            model.load_replay_buffer(str(resume_replay_buffer))
        print(f"Resuming SAC grasp from checkpoint: {resume_checkpoint}")
        if resume_replay_buffer is not None:
            print(f"Loaded replay buffer: {resume_replay_buffer}")
        else:
            print("Replay buffer tidak ditemukan/dipilih, lanjut tanpa replay buffer lama.")
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            verbose=1,
            device=args.device,
        )

    callbacks = CallbackList(
        [
            GraspTensorboardCallback(),
            PeriodicArtifactCallback(
                save_freq=args.save_freq,
                checkpoint_dir=checkpoint_dir,
                replay_buffer_dir=replay_buffer_dir,
                video_dir=video_dir,
                video_length=args.video_length,
                video_fps=args.video_fps,
                seed=args.seed,
                save_video=not args.no_video,
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

    print(f"Training complete. Final model saved to: {final_model_path}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run artifacts: {PROJECT_ROOT / 'models' / run_name}")


if __name__ == "__main__":
    main()
