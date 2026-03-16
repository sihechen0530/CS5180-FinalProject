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
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO

import cv2


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


def concat_avi_files(avi_paths: list, output_path: str) -> None:
    """Concatenate multiple AVI files using ffmpeg."""
    list_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(list_path, "w") as f:
        for p in avi_paths:
            f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
            
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_path)


def convert_avi_to_mp4(input_path: str, output_path: str, overlay_text=None) -> None:
    """
    Convert AVI to MP4 using ffmpeg only, optionally drawing overlay text.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
    ]

    if overlay_text:
        # Avoid single quotes in the overlay text to keep ffmpeg args simple
        safe_text = str(overlay_text).replace("'", "")
        drawtext = (
            f"drawtext=text='{safe_text}':"
            "x=(w-text_w)/2:y=40:"
            "fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"
        )
        cmd += ["-vf", drawtext]

    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained GRF agent (Qualitative).")
    parser.add_argument("--config", type=str, default="configs/3v1_config.yaml")
    parser.add_argument("--output-dir", type=str, required=True, help="Run directory (same as used for training); agent and videos resolved under it.")
    parser.add_argument("--agent", type=str, default=None, help="Override: path to .zip. If not set, use output_dir/agents/<baseline_zip> or latest checkpoint.")
    parser.add_argument("--no-video", action="store_true", help="Disable video write.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to visualize (usually 3-5).")
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

    write_video = not args.no_video
    logdir = os.path.join(output_dir, "videos")
    os.makedirs(logdir, exist_ok=True)

    env = football_env.create_environment(
        env_name=env_id,
        stacked=False,
        representation="simple115",
        rewards="scoring",
        render=True,
        write_video=write_video,
        write_full_episode_dumps=write_video,
        logdir=logdir,
    )

    print(f"Loading agent: {model_path}")
    # The checkpoint was saved with MaskablePPO (sb3_contrib) which uses
    # MaskableActorCriticPolicy.  Two compat issues on load:
    #   1. Must load with MaskablePPO, not plain PPO.
    #   2. Older SB3 stored `use_sde` as a top-level model attribute and passed
    #      it as a kwarg to the policy constructor; MaskableActorCriticPolicy
    #      no longer accepts it → TypeError.  Patch __init__ to drop it.
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    _orig_policy_init = MaskableActorCriticPolicy.__init__
    def _compat_policy_init(self, *args, **kwargs):
        kwargs.pop("use_sde", None)
        _orig_policy_init(self, *args, **kwargs)
    MaskableActorCriticPolicy.__init__ = _compat_policy_init
    try:
        # Force CPU inference for visualization so we don't depend on GPU
        # CUDA capability matching the training environment.
        model = MaskablePPO.load(model_path, device="cpu")
    finally:
        MaskableActorCriticPolicy.__init__ = _orig_policy_init  # restore

    num_episodes = args.episodes
    print(f"Running visualization for {num_episodes} episodes...")

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep + 1}/{num_episodes} finished. Total reward: {total_reward}")

    env.close()

    # If we wrote a video, convert the most recent AVI files in logdir to a single MP4
    # and overlay some basic info (run name, mode, steps).
    if write_video:
        avi_files = [
            os.path.join(logdir, f)
            for f in os.listdir(logdir)
            if f.lower().endswith(".avi") and not f.startswith("merged_")
        ]
        if not avi_files:
            print(f"No .avi video files found in {logdir}; skipping MP4 conversion.")
        else:
            avi_files.sort(key=os.path.getmtime)
            recent_avis = avi_files[-num_episodes:] if len(avi_files) >= num_episodes else avi_files
            
            if len(recent_avis) > 1:
                latest_avi = recent_avis[-1]
                merged_avi = os.path.join(logdir, "merged_" + os.path.basename(latest_avi))
                print(f"Concatenating {len(recent_avis)} episodes into {merged_avi}...")
                concat_avi_files(recent_avis, merged_avi)
                final_avi = merged_avi
            else:
                final_avi = recent_avis[0]

            mp4_path = os.path.splitext(final_avi)[0] + ".mp4"

            print(f"Converting final video to MP4: {final_avi} -> {mp4_path}")
            convert_avi_to_mp4(final_avi, mp4_path)


if __name__ == "__main__":
    main()
