"""
Evaluation entry point: load a trained agent, run one episode, optional video.

Usage:
  python scripts/evaluate.py --config configs/3v1_config.yaml --output-dir outputs/my_run
  python scripts/evaluate.py --config configs/empty_config.yaml --output-dir outputs/empty_001
  python scripts/evaluate.py --config configs/3v1_config.yaml --output-dir outputs/my_run --agent path/to/ppo.zip
  python scripts/evaluate.py --config configs/3v1_config.yaml --output-dir outputs/my_run --no-video

--output-dir is the run directory (same as used for training); agent is resolved from
output_dir/agents/<baseline_zip> or latest in output_dir/checkpoints/<run_name>. Videos go to output_dir/videos.
"""

import os
import sys
import argparse
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
import gym
import gfootball.env as football_env
from sb3_contrib import MaskablePPO

from core.wrappers import observation_to_features, ActionMaskWrapper

# Actions from gfootball
ACTION_SHORT_PASS = 11
ACTION_HIGH_PASS = 10
ACTION_LONG_PASS = 9
ACTION_SHOT = 12


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def extract_steps_from_model_path(path: str) -> int:
    """
    Best-effort extraction of training steps from a model filename.
    Expects names like ppo_3v1_100000_steps.zip; returns -1 if not found.
    """
    name = os.path.basename(path).replace(".zip", "")
    parts = name.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return -1


def get_ball_out_of_bounds(obs_flat: list) -> bool:
    features = observation_to_features(obs_flat)
    # The field is mostly bounded by [-1, 1] in X and [-0.42, 0.42] in Y roughly
    x, y = features.get("ball_x", 0.0), features.get("ball_y", 0.0)
    return abs(y) > 0.44 or abs(x) > 1.05

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GRF agent (Quantitative).")
    parser.add_argument("--config", type=str, default="configs/3v1_config.yaml")
    parser.add_argument("--output-dir", type=str, required=True, help="Run directory (same as used for training); agent and metrics resolved under it.")
    parser.add_argument("--agent", type=str, default=None, help="Override: path to .zip. If not set, use output_dir/agents/<baseline_zip> or latest checkpoint.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    config = load_config(config_path)

    os.chdir(REPO_ROOT)

    # Output directory: required; agent and videos under it
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env_id = config["env_id"]
    run_name = config["run_name"]
    checkpoint_prefix = config.get("checkpoint_prefix", run_name)
    baseline_zip = config.get("baseline_zip", f"{run_name}.zip")
    agents_dir = os.path.join(output_dir, "agents")
    checkpoint_dir = os.path.join(output_dir, "checkpoints", run_name)
    baseline_path = os.path.join(agents_dir, baseline_zip)

    # Resolve model path
    if args.agent:
        model_path = args.agent
        if not model_path.endswith(".zip"):
            model_path = f"{model_path}.zip"
        if not os.path.isabs(model_path):
            model_path = os.path.join(REPO_ROOT, model_path)
    else:
        model_path = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
        if model_path is None and os.path.exists(baseline_path):
            model_path = baseline_path
        if model_path is None:
            raise FileNotFoundError(
                f"No checkpoint or baseline found. Looked in {checkpoint_dir} and {baseline_path}."
            )

    env = football_env.create_environment(
        env_name=env_id,
        stacked=False,
        representation="simple115",
        rewards="scoring",
        render=False,
        write_video=False,
        write_full_episode_dumps=False,
    )
    env = ActionMaskWrapper(env)

    print(f"Loading agent: {model_path}")
    model = MaskablePPO.load(model_path)

    # Metrics
    num_episodes = args.episodes
    successes = 0
    total_steps = 0
    total_passes = 0
    shots_taken = 0
    total_shot_distance = 0.0
    out_of_bounds_count = 0
    goal_diff_total = 0

    print(f"Running evaluation for {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_passes = 0
        
        while not done:
            action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action_val = int(action)
            
            # Track passes
            if action_val in [ACTION_SHORT_PASS, ACTION_HIGH_PASS, ACTION_LONG_PASS]:
                ep_passes += 1
            
            # Track shot distance
            if action_val == ACTION_SHOT:
                shots_taken += 1
                features = observation_to_features(obs)
                ball_x, ball_y = features.get("ball_x", 0.0), features.get("ball_y", 0.0)
                # Goal is at x=1.0, y=0.0
                dist_sq = (1.0 - ball_x)**2 + (0.0 - ball_y)**2
                total_shot_distance += dist_sq
                
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            
        total_steps += ep_steps
        total_passes += ep_passes
        
        # Win / Out-of-bounds metrics
        if ep_reward > 0:
            successes += 1
        else:
            if get_ball_out_of_bounds(obs):
                out_of_bounds_count += 1
                
        # Goal difference
        if 'score_reward' in info:  # typically +1 for goal, -1 for concede
            goal_diff_total += info.get('score_reward', ep_reward)
        else:
            goal_diff_total += ep_reward

    env.close()

    metrics = {
        "evaluation_episodes": num_episodes,
        "Task_Performance": {
            "success_rate": successes / num_episodes,
            "average_goal_difference": goal_diff_total / num_episodes,
            "average_episode_length": total_steps / num_episodes
        },
        "Tactical_Behavioral": {
            "average_passes_per_episode": total_passes / num_episodes,
            "average_shot_distance_sq": (total_shot_distance / shots_taken) if shots_taken > 0 else 0.0,
            "out_of_bounds_rate": out_of_bounds_count / num_episodes
        }
    }

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation finished. Metrics saved to {metrics_path}:")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
