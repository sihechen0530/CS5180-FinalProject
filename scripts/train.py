"""
Main training entry point. Parametrised by config YAML.

Usage:
  python scripts/train.py --config configs/3v1_config.yaml --output-dir outputs/my_run
  python scripts/train.py --config configs/empty_config.yaml --output-dir outputs/empty_001

Output directory must be specified; all artifacts (checkpoints, tensorboard, agents)
are written under it. Re-running with the same --output-dir and config resumes training.
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
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

from core.wrappers import CustomRewardWrapper, make_dense_reward_fn, ActionMaskWrapper, LLMDenseRewardWrapper
from core.callbacks import GRFEvalStoppingCallback, StopTrainingException
from core.coach import GroqCoach, DeepSeekCoach, MockDeepSeekCoach
from core.callbacks import CoachCallback


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", save_replay_buffer: bool = False, save_vecnormalize: bool = False, verbose: int = 0, starting_step: int = 0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.starting_step = starting_step
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps + self.starting_step}_steps")
            self.model.save(path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {path}")
            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self.num_timesteps + self.starting_step}_steps.pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")
            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                vec_normalize_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps + self.starting_step}_steps.pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")
        return True


class EnvBuilder:
    def __init__(self, env_id: str, rank: int, seed: int, reward_config: dict):
        self.env_id = env_id
        self.rank = rank
        self.seed = seed
        self.reward_config = reward_config

    def __call__(self):
        import gfootball.env as football_env
        from core.wrappers import CustomRewardWrapper, make_dense_reward_fn, ActionMaskWrapper, LLMDenseRewardWrapper
        from stable_baselines3.common.monitor import Monitor
        
        env = football_env.create_environment(
            env_name=self.env_id,
            stacked=False,
            representation="simple115",
            rewards="scoring",
            write_video=False,
            write_full_episode_dumps=False,
            render=False,
        )
        env = Monitor(env)
        if self.reward_config.get("use_custom"):
            fn = make_dense_reward_fn(
                ball_dist_weight=self.reward_config.get("ball_dist_weight", 0.0),
                teammate_proximity_weight=self.reward_config.get("teammate_proximity_weight", 0.0),
                goal_bonus=self.reward_config.get("goal_bonus", 1.0),
                sparse_fallback=self.reward_config.get("sparse_fallback", True),
            )
            env = CustomRewardWrapper(env, reward_fn=fn, use_delta_features=True)
            
        if self.reward_config.get("use_llm_reward"):
            # Load string formula representation from the config if available
            formula_str = self.reward_config.get("llm_reward_formula", None)
            reward_func = None
            if formula_str:
                local_env = {}
                try:
                    # Dynamically extract 'llm_reward_formula'
                    exec(formula_str, globals(), local_env)
                    if "llm_reward_formula" in local_env:
                        reward_func = local_env["llm_reward_formula"]
                except Exception as e:
                    print(f"Failed to compile LLM reward formula. Using default fallback. Error: {e}")
            env = LLMDenseRewardWrapper(env, reward_formula=reward_func)
            
        # Add the ActionMaskWrapper for Coach integration
        env = ActionMaskWrapper(env)
        env.seed(self.seed + self.rank)
        return env

def make_env(env_id: str, rank: int, seed: int, reward_config: dict):
    return EnvBuilder(env_id, rank, seed, reward_config)


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
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for all run artifacts (checkpoints, tensorboard, agents). Re-use to resume.")
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

    # Output directory: required; all artifacts live under it
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env_id = config["env_id"]
    num_cpu = config.get("num_cpu", 16)
    total_timesteps = config["total_timesteps"]
    run_name = config["run_name"]
    checkpoint_prefix = config.get("checkpoint_prefix", run_name)
    baseline_zip = config.get("baseline_zip", f"{run_name}.zip")
    checkpoint_freq = config.get("checkpoint_freq", 100000)
    reward_config = config.get("reward_wrapper") or {}

    agents_dir = os.path.join(output_dir, "agents")
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
        print(f"Loading existing MaskablePPO from: {latest}")
        model = MaskablePPO.load(latest, env=env)
    else:
        print("No checkpoint found. Building new MaskablePPO...")
        tb_log = os.path.join(output_dir, "tensorboard", run_name)
        os.makedirs(tb_log, exist_ok=True)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log,
            learning_rate=config.get("learning_rate", 0.0003),
            ent_coef=config.get("ent_coef", 0.01),
        )

    # Determine how many timesteps are already in the checkpoint and only
    # train for the remaining steps up to total_timesteps from config.
    already_trained = 0
    if latest:
        filename = os.path.basename(latest)
        parts = filename.replace(".zip", "").split("_")
        for p in reversed(parts):
            if p.isdigit():
                already_trained = int(p)
                break
    
    if already_trained == 0:
        already_trained = int(getattr(model, "num_timesteps", 0))
    target_timesteps = int(total_timesteps)
    remaining_timesteps = max(target_timesteps - already_trained, 0)
    print(
        f"Timesteps: already trained = {already_trained}, "
        f"target = {target_timesteps}, remaining = {remaining_timesteps}"
    )

    save_freq = max(checkpoint_freq // num_cpu, 1)
    callbacks = [
        CustomCheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=checkpoint_prefix,
            starting_step=already_trained,
        )
    ]

    # Eval-based early stopping (always on; override eval_stopping.* in config for tuning)
    eval_cfg = config.get("eval_stopping") or {}
    eval_freq = int(eval_cfg.get("eval_freq", 50000))
    n_eval_episodes = int(eval_cfg.get("n_eval_episodes", 100))
    target_win_rate = float(eval_cfg.get("target_win_rate", 0.95))
    callbacks.append(
        GRFEvalStoppingCallback(
            env_id=env_id,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            target_win_rate=target_win_rate,
            verbose=1,
        )
    )
    print(
        f"Eval stopping: eval_freq={eval_freq}, n_eval_episodes={n_eval_episodes}, "
        f"target_win_rate={target_win_rate:.0%}"
    )

    if config.get("use_action_masking", False):
        interval = config.get("coach_interval", 50)
        groq_api_key = os.environ.get("GROQ_API_KEY", None)
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        
        if groq_api_key:
            coach_client = GroqCoach(api_key=groq_api_key)
            print(f"Action Masking enabled via real GroqCoach API. Interval={interval}")
        elif deepseek_api_key:
            coach_client = DeepSeekCoach(api_key=deepseek_api_key)
            print(f"Action Masking enabled via real DeepSeekCoach API. Interval={interval}")
        else:
            coach_client = MockDeepSeekCoach()
            print(f"Action Masking enabled via MockDeepSeekCoach (No API key found). Interval={interval}")
            
        callbacks.append(CoachCallback(coach_client, coach_interval=interval, verbose=1))

    cb = CallbackList(callbacks)

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
        try:
            model.learn(total_timesteps=remaining_timesteps, callback=cb)
        except StopTrainingException as e:
            print(f"Early stopping: {e}")
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
