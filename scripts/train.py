"""
Main training entry point. Parametrised by config YAML.

Usage:
  python scripts/train.py --config configs/3v1_config.yaml
  python scripts/train.py --config configs/empty_config.yaml

Runs from repo root (CS5180-FinalProject/). Saves checkpoints and final
model under agents_dir from config (e.g. agents/baselines/ or agents/llm_augmented/).
"""

import os
import sys
import time
import argparse

# Repo root = parent of scripts/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Thread limits for parallel envs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import yaml
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from core.wrappers import CustomRewardWrapper, make_dense_reward_fn


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_id: str, rank: int, seed: int, reward_config: dict):
    def _init():
        env = football_env.create_environment(
            env_name=env_id,
            stacked=False,
            representation="simple115",
            rewards="scoring",
            write_video=False,
            write_full_episode_dumps=False,
            render=False,
        )
        env = Monitor(env)
        if reward_config.get("use_custom"):
            fn = make_dense_reward_fn(
                ball_dist_weight=reward_config.get("ball_dist_weight", 0.0),
                teammate_proximity_weight=reward_config.get("teammate_proximity_weight", 0.0),
                goal_bonus=reward_config.get("goal_bonus", 1.0),
                sparse_fallback=reward_config.get("sparse_fallback", True),
            )
            env = CustomRewardWrapper(env, reward_fn=fn, use_delta_features=True)
        env.seed(seed + rank)
        return env
    return _init


def find_latest_checkpoint(checkpoint_dir: str, prefix: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".zip") and f.startswith(prefix)
    ]
    if not files:
        return None

    def _steps(name):
        parts = name.replace(".zip", "").split("_")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    files.sort(key=_steps)
    return os.path.join(checkpoint_dir, files[-1])


def main():
    parser = argparse.ArgumentParser(description="Train PPO on GRF from config.")
    parser.add_argument("--config", type=str, default="configs/3v1_config.yaml", help="Path to YAML config.")
    parser.add_argument("--override", type=str, action="append", help="Override key=value (e.g. total_timesteps=1000).")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    config = load_config(config_path)

    for ov in args.override or []:
        if "=" in ov:
            k, v = ov.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if v.lower() in ("true", "false"):
                        v = v.lower() == "true"
            config[k.strip()] = v

    os.chdir(REPO_ROOT)

    env_id = config["env_id"]
    num_cpu = config.get("num_cpu", 16)
    total_timesteps = config["total_timesteps"]
    run_name = config["run_name"]
    checkpoint_prefix = config.get("checkpoint_prefix", run_name)
    baseline_zip = config.get("baseline_zip", f"{run_name}.zip")
    checkpoint_freq = config.get("checkpoint_freq", 100000)
    reward_config = config.get("reward_wrapper") or {}

    # Single output root for all run artifacts (TB, checkpoints, videos, agents)
    output_dir = config.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Agents live under <output_dir>/agents by default; can be overridden in config
    agents_dir_cfg = config.get("agents_dir")
    if agents_dir_cfg is None:
        agents_dir = os.path.join(output_dir, "agents")
    else:
        agents_dir = agents_dir_cfg
        if not os.path.isabs(agents_dir):
            agents_dir = os.path.join(REPO_ROOT, agents_dir)
    os.makedirs(agents_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    baseline_path = os.path.join(agents_dir, baseline_zip)

    if num_cpu > 1:
        env = SubprocVecEnv([make_env(env_id, i, 0, reward_config) for i in range(num_cpu)])
    else:
        env = DummyVecEnv([make_env(env_id, 0, 0, reward_config)])

    latest = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
    if latest is None and os.path.exists(baseline_path):
        latest = baseline_path

    # Build or resume model
    if latest:
        print(f"Loading existing PPO from: {latest}")
        model = PPO.load(latest, env=env)
    else:
        print("No checkpoint found. Building new PPO...")
        tb_log = os.path.join(output_dir, "tensorboard", run_name)
        os.makedirs(tb_log, exist_ok=True)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log,
            learning_rate=config.get("learning_rate", 0.0003),
            ent_coef=config.get("ent_coef", 0.01),
        )

    # Determine how many timesteps are already in the checkpoint and only
    # train for the remaining steps up to total_timesteps from config.
    already_trained = int(getattr(model, "num_timesteps", 0))
    target_timesteps = int(total_timesteps)
    remaining_timesteps = max(target_timesteps - already_trained, 0)
    print(
        f"Timesteps: already trained = {already_trained}, "
        f"target = {target_timesteps}, remaining = {remaining_timesteps}"
    )

    save_freq = max(checkpoint_freq // num_cpu, 1)
    cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=checkpoint_prefix,
    )

    if remaining_timesteps <= 0:
        print(
            "Configured total_timesteps already reached or exceeded; "
            "skipping additional training."
        )
        elapsed = 0.0
    else:
        print(
            f"Training {run_name} from {already_trained} "
            f"to {target_timesteps} timesteps "
            f"(remaining {remaining_timesteps})..."
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=remaining_timesteps, callback=cb)
        elapsed = time.perf_counter() - t0

    if elapsed > 0:
        h, r = divmod(int(elapsed), 3600)
        m, s = divmod(r, 60)
        print(f"Training complete. Wall-clock: {h}h {m}m {s}s ({elapsed:.1f}s)")

    final_path = os.path.join(agents_dir, run_name)
    model.save(final_path)
    env.close()
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
