import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from panda_rl.envs import PandaReachEnv


class RewardTensorboardCallback(BaseCallback):
    """Log reward components and task metrics from env info dicts."""

    def _on_step(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if len(rewards) > 0:
            self.logger.record("reward/step_total", float(np.mean(rewards)))

        scalar_keys = {
            "reward_position": "reward/position",
            "reward_orientation": "reward/orientation",
            "reward_action_penalty": "reward/action_penalty",
            "reward_success_bonus": "reward/success_bonus",
            "reward_total": "reward/env_total",
            "position_distance": "metrics/position_distance",
            "orientation_distance": "metrics/orientation_distance",
            "action_penalty": "metrics/action_penalty",
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
        video_length=200,
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
        self.last_save_step = 0
        self.latest_replay_buffer_path = None

    def _init_callback(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps - self.last_save_step < self.save_freq:
            return True

        self.last_save_step = self.num_timesteps
        self._save_artifacts(self.num_timesteps)
        return True

    def _save_artifacts(self, step):
        step_label = f"{step:010d}"

        model_path = self.checkpoint_dir / f"sac_panda_reach_{step_label}_steps.zip"
        replay_buffer_path = (
            self.replay_buffer_dir / f"replay_buffer_{step_label}_steps.pkl"
        )

        self.model.save(str(model_path))
        self.model.save_replay_buffer(str(replay_buffer_path))
        self._remove_previous_replay_buffers(replay_buffer_path)
        self.latest_replay_buffer_path = replay_buffer_path

        if self.save_video:
            video_path = self.video_dir / f"eval_{step_label}_steps.mp4"
            self._save_eval_video(video_path)

        if self.verbose:
            print(f"Saved SAC artifacts at {step} timesteps.")

    def _remove_previous_replay_buffers(self, current_path):
        for replay_path in self.replay_buffer_dir.glob("replay_buffer_*_steps.pkl"):
            if replay_path != current_path:
                replay_path.unlink(missing_ok=True)

    def _save_eval_video(self, video_path):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Video saving requires imageio with ffmpeg support. "
                "Install dependencies from requirements.txt."
            ) from exc

        eval_env = PandaReachEnv(render_mode="rgb_array", randomize_target=False)
        frames = []

        try:
            obs, _ = eval_env.reset(seed=self.seed)
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


def make_env(seed, monitor_dir, randomize_target=True):
    env = PandaReachEnv(randomize_target=randomize_target)
    env = Monitor(env, filename=str(monitor_dir / "monitor.csv"))
    env.reset(seed=seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--fixed-target", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--video-fps", type=int, default=50)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
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

    run_name = args.run_name or datetime.now().strftime("panda_reach_sac_%Y%m%d_%H%M%S")
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
        randomize_target=not args.fixed_target,
    )

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
            RewardTensorboardCallback(),
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
