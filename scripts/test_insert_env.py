import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_rl.envs import PandaInsertEnv


def main():
    env = PandaInsertEnv(randomize_socket=True, randomize_start=True)

    obs, info = env.reset(seed=0)
    print("Initial observation shape:", obs.shape)
    print("Observation space shape :", env.observation_space.shape)
    print("Action space shape      :", env.action_space.shape)
    print("Initial info            :", info)

    assert env.observation_space.contains(obs), "Reset observation is out of space."
    assert env.action_space.shape == (7,), "Insert action must be 7D."

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs), "Step observation is out of space."
        assert np.isfinite(reward), "Reward must be finite."

        print(
            f"step={step + 1:02d}",
            f"reward={reward:.4f}",
            f"insert_dist={info['insert_distance']:.4f}",
            f"xy={info['xy_distance']:.4f}",
            f"z={info['z_error']:.4f}",
            f"success={info['is_success']}",
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
