"""
Eureka-style iterative reward optimization for GRF.

Outer loop (configurable interval):
  1. LLM generates or refines reward function
  2. RL trains for N steps with that reward (no LLM during training)
  3. Collect eval stats → build reflection text
  4. Feed reflection back to LLM for next iteration

Inner loop: same as train.py — reward is pure Python, evaluated locally each step.

Usage:
  python scripts/train_iterative_reward.py \\
    --config configs/llm/reward/empty_config.yaml \\
    --output-dir outputs/iterative_empty \\
    --num-iterations 3 \\
    --steps-per-iteration 5000

  # Optional: put in config YAML under key "iterative"
  # iterative:
  #   num_iterations: 3
  #   steps_per_iteration: 10000
  #   eval_episodes: 50

  # Dry-run (no training; use fake stats for reflection, LLM still called if API key set)
  python scripts/train_iterative_reward.py --config ... --output-dir ... --dry-run
"""

import os
import sys
import time
import argparse
import importlib.util
import copy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_train_module():
    """Load train.py as a module to reuse EnvBuilder, make_env, load_config."""
    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(REPO_ROOT, "scripts", "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_training_segment(
    config: dict,
    reward_file_abs: str,
    steps: int,
    output_dir: str,
    segment_id: int,
    eval_episodes: int = 50,
) -> dict:
    """
    Run RL for `steps` with reward from `reward_file_abs`. No checkpoint resume; fresh model.
    Returns eval stats: eval_win_rate, eval_mean_return, eval_mean_episode_len, eval_n_episodes.
    """
    import yaml
    import gym
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    train_mod = _load_train_module()
    make_env = train_mod.make_env

    env_id = config["env_id"]
    num_cpu = config.get("num_cpu", 1)
    reward_config = copy.deepcopy(config.get("reward_wrapper") or {})
    reward_config["use_custom"] = False
    reward_config["use_llm_reward"] = True
    reward_config["llm_reward_file"] = reward_file_abs
    reward_config["llm_reward_formula"] = None

    # Use DummyVecEnv only: GRF (and wrappers) contain non-pickleable C extensions (PyCapsule),
    # so SubprocVecEnv would fail when spawning worker processes.
    env = DummyVecEnv([make_env(env_id, i, 0, reward_config) for i in range(num_cpu)])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=config.get("learning_rate", 0.0003),
        ent_coef=config.get("ent_coef", 0.01),
    )

    model.learn(total_timesteps=steps)
    # Eval: scoring-only env, deterministic
    eval_stats = _run_eval(model, env_id, eval_episodes)
    env.close()
    return eval_stats


def _run_eval(model, env_id: str, n_episodes: int) -> dict:
    """Run n_episodes with scoring-only env; return win rate, mean return, mean length."""
    import gfootball.env as football_env

    env = football_env.create_environment(
        env_name=env_id,
        stacked=False,
        representation="simple115",
        rewards="scoring",
        write_video=False,
        write_full_episode_dumps=False,
        render=False,
    )
    returns = []
    lengths = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
    env.close()
    wins = sum(1 for r in returns if r > 0)
    return {
        "eval_win_rate": wins / n_episodes,
        "eval_mean_return": sum(returns) / n_episodes,
        "eval_mean_episode_len": sum(lengths) / n_episodes,
        "eval_n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Eureka-style iterative reward optimization")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML (e.g. configs/llm/reward/empty_config.yaml)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for rewards and logs")
    parser.add_argument("--num-iterations", "-K", type=int, default=3, help="Number of reward refinement iterations")
    parser.add_argument("--steps-per-iteration", "-N", type=int, default=10000, help="RL steps per iteration")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Eval episodes for reflection stats")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "groq"])
    parser.add_argument("--dry-run", action="store_true", help="No training; generate/refine with fake stats only")
    parser.add_argument("--override", type=str, action="append", help="Override key=value (e.g. num_cpu=1)")
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

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rewards_dir = os.path.join(output_dir, "rewards")
    os.makedirs(rewards_dir, exist_ok=True)

    scenario = config.get("scenario") or config.get("env_id", "").replace("academy_", "").replace("_", " ")
    if "empty" in config.get("env_id", ""):
        scenario = "empty_goal"
    elif "3_vs_1" in config.get("env_id", ""):
        scenario = "3v1"

    iter_cfg = config.get("iterative") or {}
    K = iter_cfg.get("num_iterations", args.num_iterations)
    N = iter_cfg.get("steps_per_iteration", args.steps_per_iteration)
    eval_episodes = iter_cfg.get("eval_episodes", args.eval_episodes)

    api_key = os.environ.get("DEEPSEEK_API_KEY") if args.provider == "deepseek" else os.environ.get("GROQ_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: Set DEEPSEEK_API_KEY or GROQ_API_KEY for LLM calls")
        sys.exit(1)
    if args.dry_run:
        print("[Dry-run] No real training; using fake stats for reflection (LLM still called if API key set)")

    # Import reward_generator directly (avoid core.wrappers -> gym)
    _spec = importlib.util.spec_from_file_location(
        "reward_generator", os.path.join(REPO_ROOT, "core", "reward_generator.py")
    )
    _rg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rg)
    RewardGenerator = _rg.RewardGenerator
    build_reflection_text = _rg.build_reflection_text

    generator = RewardGenerator(api_key=api_key or "dry-run", provider=args.provider) if api_key else None
    previous_code = None
    reflection = None

    for i in range(K):
        print("\n" + "=" * 60)
        print(f"Iteration {i + 1}/{K}")
        print("=" * 60)

        if i == 0:
            if generator is None:
                # Dry-run without API key: use placeholder
                previous_code = """def llm_reward_formula(env_reward, features, prev_features):
    delta = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    return float(env_reward) + 0.05 * delta"""
            else:
                result = generator.generate(scenario=scenario, temperature=0.2)
                if result["error"]:
                    print(f"Generate failed: {result['error']}")
                    sys.exit(1)
                previous_code = result["code"]
        else:
            if generator is None:
                previous_code = previous_code + "\n# refined (dry-run, no API)"
            else:
                result = generator.refine(
                    previous_code,
                    reflection,
                    scenario=scenario,
                    iteration=i,
                    steps_per_iteration=N,
                    temperature=0.2,
                )
                if result["error"]:
                    print(f"Refine failed: {result['error']}")
                    sys.exit(1)
                previous_code = result["code"]

        reward_path = os.path.join(rewards_dir, f"iter_{i}.py")
        with open(reward_path, "w") as f:
            f.write(f'"""Iteration {i} reward"""\n\n')
            f.write(previous_code)
            f.write("\n")
        print(f"Saved reward to {reward_path}")

        if args.dry_run:
            stats = {
                "eval_win_rate": 0.3 + i * 0.15,
                "eval_mean_return": 0.2 + i * 0.2,
                "eval_mean_episode_len": 40.0 + i * 5,
                "eval_n_episodes": eval_episodes,
            }
            print(f"[Dry-run] Fake stats: win_rate={stats['eval_win_rate']:.1%}")
        else:
            reward_file_abs = os.path.abspath(reward_path)
            t0 = time.perf_counter()
            stats = run_training_segment(
                config, reward_file_abs, N, output_dir, i, eval_episodes=eval_episodes
            )
            elapsed = time.perf_counter() - t0
            print(f"Training: {N} steps in {elapsed:.1f}s")
            print(f"Eval: win_rate={stats['eval_win_rate']:.1%} mean_return={stats['eval_mean_return']:.3f}")

        reflection = build_reflection_text(stats, N)

    final_path = os.path.join(rewards_dir, "final.py")
    with open(final_path, "w") as f:
        f.write('"""Final reward after iterative refinement"""\n\n')
        f.write(previous_code)
        f.write("\n")
    print(f"\nFinal reward saved to {final_path}")
    print("Done.")


if __name__ == "__main__":
    main()
