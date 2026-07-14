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

from panda_rl.envs import PandaInsertEnv

STEP_PATTERN = re.compile(r"(\d+)")


def step_from_path(path):
    matches = STEP_PATTERN.findall(Path(path).stem)
    return int(matches[-1]) if matches else -1


def latest_file(paths):
    paths = [Path(path) for path in paths]
    return max(paths, key=step_from_path) if paths else None


def latest_run_dir():
    candidates = [
        path
        for path in (PROJECT_ROOT / "models").glob("panda_insert_sac*")
        if path.is_dir()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_model_path(args):
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
        return model_path

    run_dir = PROJECT_ROOT / "models" / args.run_name if args.run_name else latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("Tidak ada folder run panda_insert_sac di models/.")

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
    min_insert_distance = float(info["insert_distance"])
    min_xy_distance = float(info["xy_distance"])
    min_z_error = float(info["z_error"])
    gripper_qpos_values = [float(np.mean(info["gripper_qpos"]))]
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
        min_insert_distance = min(min_insert_distance, float(info["insert_distance"]))
        min_xy_distance = min(min_xy_distance, float(info["xy_distance"]))
        min_z_error = min(min_z_error, float(info["z_error"]))
        gripper_qpos_values.append(float(np.mean(info["gripper_qpos"])))

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
        "final_insert_distance": float(info["insert_distance"]),
        "min_insert_distance": min_insert_distance,
        "final_xy_distance": float(info["xy_distance"]),
        "min_xy_distance": min_xy_distance,
        "final_z_error": float(info["z_error"]),
        "min_z_error": min_z_error,
        "gripper_qpos_mean": float(np.mean(gripper_qpos_values)),
        "cube_pos": np.asarray(info["cube_pos"]).copy(),
        "insert_target_pos": np.asarray(info["insert_target_pos"]).copy(),
        "frames": frames,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fixed-socket", action="store_true")
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--no-terminate-on-success", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args)
    deterministic = not args.stochastic
    render_mode = "rgb_array" if args.video else ("human" if args.render else None)

    print(f"Loading model: {model_path}")
    model = SAC.load(str(model_path), device=args.device)

    episode_results = []
    video_frames = []

    env = PandaInsertEnv(
        render_mode=render_mode,
        randomize_socket=not args.fixed_socket,
        randomize_start=not args.fixed_start,
        terminate_on_success=not args.no_terminate_on_success,
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
                f"min_insert={result['min_insert_distance']:.4f}",
                f"final_insert={result['final_insert_distance']:.4f}",
                f"min_xy={result['min_xy_distance']:.4f}",
                f"min_z={result['min_z_error']:.4f}",
                f"gripper={result['gripper_qpos_mean']:.4f}",
                f"steps={result['steps']}",
            )
    finally:
        env.close()

    if args.video:
        save_video(args.video, video_frames, args.video_fps)
        print(f"Saved eval video: {args.video}")

    rewards = np.array([result["reward"] for result in episode_results])
    successes = np.array([result["success"] for result in episode_results], dtype=float)
    min_insert = np.array(
        [result["min_insert_distance"] for result in episode_results],
        dtype=float,
    )
    final_insert = np.array(
        [result["final_insert_distance"] for result in episode_results],
        dtype=float,
    )
    min_xy = np.array([result["min_xy_distance"] for result in episode_results])
    min_z = np.array([result["min_z_error"] for result in episode_results])
    steps = np.array([result["steps"] for result in episode_results], dtype=float)
    gripper = np.array([result["gripper_qpos_mean"] for result in episode_results])

    print("\nSummary")
    print(f"episodes              : {len(episode_results)}")
    print(f"success_rate          : {successes.mean():.3f}")
    print(f"reward_mean           : {rewards.mean():.2f}")
    print(f"min_insert_mean       : {min_insert.mean():.4f} m")
    print(f"min_insert_best       : {min_insert.min():.4f} m")
    print(f"final_insert_mean     : {final_insert.mean():.4f} m")
    print(f"min_xy_mean           : {min_xy.mean():.4f} m")
    print(f"min_z_mean            : {min_z.mean():.4f} m")
    print(f"steps_mean            : {steps.mean():.1f}")
    print(f"gripper_qpos_mean     : {gripper.mean():.4f}")


if __name__ == "__main__":
    main()
