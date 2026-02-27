"""
Evaluation entry point: load a trained agent, run one episode, optional video.

Usage:
  python scripts/evaluate.py --config configs/3v1_config.yaml
  python scripts/evaluate.py --config configs/empty_config.yaml
  python scripts/evaluate.py --config configs/3v1_config.yaml --agent agents/llm_augmented/ppo_3v1
  python scripts/evaluate.py --config configs/3v1_config.yaml --no-video

Runs from repo root. Writes videos to configurable logdir (default videos/).
"""

import os
import sys
import argparse
import subprocess
import shutil

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
    parser = argparse.ArgumentParser(description="Evaluate a trained GRF agent.")
    parser.add_argument("--config", type=str, default="configs/3v1_config.yaml")
    parser.add_argument("--agent", type=str, default=None, help="Path to .zip or dir (e.g. agents/baselines/ppo_3v1). Overrides config.")
    parser.add_argument("--no-video", action="store_true", help="Disable video write.")
    parser.add_argument("--logdir", type=str, default=None, help="Where to write videos. Defaults to <output_dir>/videos.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)
    config = load_config(config_path)

    os.chdir(REPO_ROOT)

    env_id = config["env_id"]
    run_name = config["run_name"]
    checkpoint_prefix = config.get("checkpoint_prefix", run_name)
    baseline_zip = config.get("baseline_zip", f"{run_name}.zip")
    output_dir = config.get("output_dir", "outputs")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)

    # Same agents_dir logic as in train.py
    agents_dir_cfg = config.get("agents_dir")
    if agents_dir_cfg is None:
        agents_dir = os.path.join(output_dir, "agents")
    else:
        agents_dir = agents_dir_cfg
        if not os.path.isabs(agents_dir):
            agents_dir = os.path.join(REPO_ROOT, agents_dir)

    # Resolve model path
    if args.agent:
        model_path = args.agent
        if not model_path.endswith(".zip"):
            model_path = f"{model_path}.zip"
        if not os.path.isabs(model_path):
            model_path = os.path.join(REPO_ROOT, model_path)
    else:
        checkpoint_dir = os.path.join(output_dir, "checkpoints", run_name)
        baseline_path = os.path.join(agents_dir, baseline_zip)
        model_path = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
        if model_path is None and os.path.exists(baseline_path):
            model_path = baseline_path
        if model_path is None:
            raise FileNotFoundError(
                f"No checkpoint or baseline found. Looked in {checkpoint_dir} and {baseline_path}."
            )

    write_video = not args.no_video
    if args.logdir:
        logdir = args.logdir
        if not os.path.isabs(logdir):
            logdir = os.path.join(REPO_ROOT, logdir)
    else:
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
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    total_reward = 0

    print("Running evaluation episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward}")
    env.close()

    # If we wrote a video, convert the most recent AVI file in logdir to MP4
    # and overlay some basic info (run name, mode, steps).
    if write_video:
        avi_files = [
            os.path.join(logdir, f)
            for f in os.listdir(logdir)
            if f.lower().endswith(".avi")
        ]
        if not avi_files:
            print(f"No .avi video files found in {logdir}; skipping MP4 conversion.")
        else:
            latest_avi = max(avi_files, key=os.path.getmtime)
            mp4_path = os.path.splitext(latest_avi)[0] + ".mp4"

            steps = extract_steps_from_model_path(model_path)
            info_parts = [run_name, "mode=deterministic"]
            if steps >= 0:
                info_parts.append(f"steps={steps}")
            overlay_text = " | ".join(info_parts)

            print(
                f"Converting latest video to MP4: {latest_avi} -> {mp4_path} "
                f"with overlay '{overlay_text}'"
            )
            convert_avi_to_mp4(latest_avi, mp4_path, overlay_text=overlay_text)


if __name__ == "__main__":
    main()
