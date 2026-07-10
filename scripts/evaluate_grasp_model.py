import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC

from panda_rl.envs import PandaGraspEnv

STEP_PATTERN = re.compile(r"(\d+)")


def step_from_path(path):
    matches = STEP_PATTERN.findall(Path(path).stem)
    return int(matches[-1]) if matches else -1


def latest_file(paths):
    paths = [Path(path) for path in paths]
    return max(paths, key=step_from_path) if paths else None


def latest_run_dir():
    candidates = [path for path in (PROJECT_ROOT / "models").glob("*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_model_path(args):
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
        return model_path

    run_dir = PROJECT_ROOT / "models" / args.run_name if args.run_name else latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("Tidak ada folder run di models/.")

    candidates = list((run_dir / "checkpoints").glob("*.zip"))
    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        candidates.append(final_model)

    model_path = latest_file(candidates)
    if model_path is None:
        raise FileNotFoundError(f"Tidak ada model .zip di {run_dir}.")
    return model_path


def save_video(path, frames, fps):
    if not frames:
        return
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def evaluate_episode(model, env, seed, deterministic, record_video=False):
    obs, info = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    max_lift_height = float(info["lift_height"])
    min_reach_distance = float(info["reach_distance"])
    auto_close_steps = 0
    steps = 0

    if record_video:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        max_lift_height = max(max_lift_height, float(info["lift_height"]))
        min_reach_distance = min(min_reach_distance, float(info["reach_distance"]))
        auto_close_steps += int(bool(info.get("auto_gripper_closed", False)))

        if record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "success": bool(info["is_success"]),
        "final_reach_distance": float(info["reach_distance"]),
        "min_reach_distance": min_reach_distance,
        "final_lift_height": float(info["lift_height"]),
        "max_lift_height": max_lift_height,
        "auto_close_ratio": auto_close_steps / max(steps, 1),
        "frames": frames,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fixed-object", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args)
    deterministic = not args.stochastic
    render_mode = "human" if args.render else ("rgb_array" if args.video else None)

    print(f"Loading model: {model_path}")
    model = SAC.load(str(model_path), device=args.device)

    episode_results = []
    video_frames = []

    env = PandaGraspEnv(
        render_mode=render_mode,
        randomize_object=not args.fixed_object,
    )

    try:
        for episode in range(args.episodes):
            result = evaluate_episode(
                model=model,
                env=env,
                seed=args.seed + episode,
                deterministic=deterministic,
                record_video=args.video is not None,
            )
            episode_results.append(result)
            if args.video and episode == 0:
                video_frames = result["frames"]

            print(
                f"episode={episode + 1:03d}",
                f"reward={result['reward']:.2f}",
                f"success={result['success']}",
                f"min_reach={result['min_reach_distance']:.4f}",
                f"max_lift={result['max_lift_height']:.4f}",
                f"auto_close={result['auto_close_ratio']:.2f}",
                f"steps={result['steps']}",
            )
    finally:
        env.close()

    if args.video:
        save_video(args.video, video_frames, args.video_fps)
        print(f"Saved eval video: {args.video}")

    rewards = np.array([result["reward"] for result in episode_results])
    successes = np.array([result["success"] for result in episode_results], dtype=float)
    min_reaches = np.array([result["min_reach_distance"] for result in episode_results])
    max_lifts = np.array([result["max_lift_height"] for result in episode_results])
    auto_close = np.array([result["auto_close_ratio"] for result in episode_results])

    print("\nSummary")
    print(f"episodes          : {len(episode_results)}")
    print(f"success_rate      : {successes.mean():.3f}")
    print(f"reward_mean       : {rewards.mean():.2f}")
    print(f"min_reach_mean    : {min_reaches.mean():.4f} m")
    print(f"max_lift_mean     : {max_lifts.mean():.4f} m")
    print(f"max_lift_best     : {max_lifts.max():.4f} m")
    print(f"auto_close_mean   : {auto_close.mean():.3f}")


if __name__ == "__main__":
    main()
