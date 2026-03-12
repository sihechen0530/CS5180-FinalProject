"""
Action Distribution Comparison — load matched checkpoints from two runs,
run evaluation episodes, and compare action selection distributions.

Requires: gfootball, sb3_contrib, stable_baselines3, torch, matplotlib, numpy

Usage:
    python analysis/action_distribution.py \\
        --run-a-ckpt-dir outputs/baseline/checkpoints \\
        --run-b-ckpt-dir outputs/masking/checkpoints \\
        --label-a "Baseline PPO" --label-b "MaskablePPO + LLM"

This script compares:
1. Action frequency: which actions each agent actually takes
2. Policy entropy: how "spread out" the policy is over actions
3. Action probability heatmaps: raw policy output for sample observations
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ACTION_NAMES = [
    "Idle", "Left", "TopLeft", "Top", "TopRight",
    "Right", "BtmRight", "Bottom", "BtmLeft",
    "LongPass", "HighPass", "ShortPass", "Shot",
    "Sprint", "RelDir", "RelSprint",
    "Sliding", "Dribble", "RelDribble"
]


def get_action_logits(model, obs):
    """Get raw action logits from the policy network, compatible across SB3 versions."""
    import torch
    obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(model.policy.device)
    with torch.no_grad():
        try:
            features = model.policy.extract_features(obs_t, model.policy.features_extractor)
        except TypeError:
            features = model.policy.extract_features(obs_t)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        logits = model.policy.action_net(latent_pi).cpu().numpy()[0]
    return logits


def find_checkpoint(ckpt_dir, prefix, target_step):
    """Find the checkpoint closest to target_step."""
    if not ckpt_dir.exists():
        return None, None
    candidates = []
    for f in ckpt_dir.iterdir():
        if f.suffix == ".zip" and f.name.startswith(prefix):
            parts = f.stem.split("_")
            for p in reversed(parts):
                if p.isdigit():
                    candidates.append((int(p), f))
                    break
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: abs(x[0] - target_step))
    return candidates[0]


def create_eval_env():
    import gfootball.env as football_env
    from core.wrappers import ActionMaskWrapper
    env = football_env.create_environment(
        env_name="academy_3_vs_1_with_keeper",
        stacked=False,
        representation="simple115",
        rewards="scoring",
        write_video=False,
        write_full_episode_dumps=False,
        render=False,
    )
    env = ActionMaskWrapper(env)
    return env


def run_episodes(model, env, n_episodes, collect_probs=True):
    """Run episodes and collect actions taken, observations, and optionally action probs."""
    import torch

    all_actions = []
    all_observations = []
    all_probs = []
    all_entropies = []
    wins = 0
    ep_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0
        while not done:
            all_observations.append(obs.copy())
            action, _ = model.predict(obs, deterministic=False)
            action_int = int(action)
            all_actions.append(action_int)

            if collect_probs:
                try:
                    logits = get_action_logits(model, obs)
                    probs = np.exp(logits - logits.max())
                    probs = probs / probs.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    all_probs.append(probs)
                    all_entropies.append(entropy)
                except Exception:
                    pass

            obs, reward, done, info = env.step(action_int)
            ep_len += 1
            ep_reward += reward

        if ep_reward > 0:
            wins += 1
        ep_lengths.append(ep_len)

    return {
        "actions": all_actions,
        "observations": np.array(all_observations),
        "probs": np.array(all_probs) if all_probs else None,
        "entropies": np.array(all_entropies) if all_entropies else None,
        "win_rate": wins / n_episodes,
        "mean_ep_length": np.mean(ep_lengths),
        "n_steps": len(all_actions),
    }


def fig1_action_frequency(run_a_result, run_b_result, step_label, label_a, label_b, figures_dir):
    """Bar chart comparing action selection frequency."""
    a_counts = Counter(run_a_result["actions"])
    b_counts = Counter(run_b_result["actions"])
    a_total = len(run_a_result["actions"])
    b_total = len(run_b_result["actions"])

    x = np.arange(19)
    a_freq = np.array([a_counts.get(i, 0) / a_total for i in range(19)])
    b_freq = np.array([b_counts.get(i, 0) / b_total for i in range(19)])

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.38
    ax.bar(x - width / 2, a_freq * 100, width, label=label_a, color="#1565C0",
           edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, b_freq * 100, width, label=label_b, color="#C62828",
           edgecolor="black", alpha=0.85)

    ax.set_xlabel("Action", fontsize=12)
    ax.set_ylabel("Selection Frequency (%)", fontsize=12)
    ax.set_title(f"Action Selection Frequency — {label_a} vs. {label_b} (at {step_label} steps)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{ACTION_NAMES[i]}" for i in range(19)], fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    a_nonzero = sum(1 for f in a_freq if f > 0.01)
    b_nonzero = sum(1 for f in b_freq if f > 0.01)
    ax.text(0.97, 0.95,
            f"{label_a}: {a_nonzero} actions used >1%\n"
            f"{label_b}: {b_nonzero} actions used >1%\n"
            f"Win rates: {run_a_result['win_rate']:.0%} vs {run_b_result['win_rate']:.0%}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(figures_dir / "action_frequency_comparison.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 1] Saved: action_frequency_comparison.png")


def fig2_entropy_comparison(run_a_result, run_b_result, step_label, label_a, label_b, figures_dir):
    """Histogram of per-step policy entropy for both agents."""
    if run_a_result["entropies"] is None or run_b_result["entropies"] is None:
        print("[Fig 2] Skipped (no entropy data available)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(run_a_result["entropies"], bins=60, alpha=0.6, color="#1565C0",
            label=f"{label_a} (mean={np.mean(run_a_result['entropies']):.3f})",
            density=True)
    ax.hist(run_b_result["entropies"], bins=60, alpha=0.6, color="#C62828",
            label=f"{label_b} (mean={np.mean(run_b_result['entropies']):.3f})",
            density=True)
    ax.set_xlabel("Policy Entropy (nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Policy Entropy Distribution — {label_a} vs. {label_b} ({step_label})",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "entropy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Saved: entropy_comparison.png")


def fig3_probability_heatmap(run_a_result, run_b_result, step_label, label_a, label_b,
                              figures_dir, n_samples=100):
    """Heatmap of action probabilities for a sample of observations."""
    if run_a_result["probs"] is None or run_b_result["probs"] is None:
        print("[Fig 3] Skipped (no probability data available)")
        return

    rng = np.random.RandomState(42)
    n_a = min(n_samples, len(run_a_result["probs"]))
    n_b = min(n_samples, len(run_b_result["probs"]))
    a_idx = rng.choice(len(run_a_result["probs"]), n_a, replace=False)
    b_idx = rng.choice(len(run_b_result["probs"]), n_b, replace=False)

    a_probs_sample = run_a_result["probs"][a_idx]
    b_probs_sample = run_b_result["probs"][b_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.imshow(a_probs_sample.T, aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
    ax1.set_xlabel("Sample Index", fontsize=11)
    ax1.set_ylabel("Action", fontsize=11)
    ax1.set_yticks(range(19))
    ax1.set_yticklabels([f"{i} {ACTION_NAMES[i]}" for i in range(19)], fontsize=8)
    ax1.set_title(f"{label_a} — Action Probabilities", fontsize=12)
    plt.colorbar(im1, ax=ax1, fraction=0.03)

    im2 = ax2.imshow(b_probs_sample.T, aspect="auto", cmap="Reds", vmin=0, vmax=0.5)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_ylabel("Action", fontsize=11)
    ax2.set_yticks(range(19))
    ax2.set_yticklabels([f"{i} {ACTION_NAMES[i]}" for i in range(19)], fontsize=8)
    ax2.set_title(f"{label_b} — Action Probabilities", fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.03)

    fig.suptitle(f"Raw Policy Action Probability Heatmaps ({step_label})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "action_probability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] Saved: action_probability_heatmap.png")


def fig4_mean_action_probs(run_a_result, run_b_result, step_label, label_a, label_b, figures_dir):
    """Bar chart of mean action probability across all sampled states."""
    if run_a_result["probs"] is None or run_b_result["probs"] is None:
        print("[Fig 4] Skipped (no probability data available)")
        return

    a_mean = run_a_result["probs"].mean(axis=0)
    b_mean = run_b_result["probs"].mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(19)
    width = 0.38
    ax.bar(x - width / 2, a_mean, width, label=label_a, color="#1565C0",
           edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, b_mean, width, label=label_b, color="#C62828",
           edgecolor="black", alpha=0.85)

    ax.set_xlabel("Action", fontsize=12)
    ax.set_ylabel("Mean Policy Probability", fontsize=12)
    ax.set_title(f"Mean Action Probability (Raw Policy Output) — {step_label}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{ACTION_NAMES[i]}" for i in range(19)], fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    a_eff = np.exp(-np.sum(a_mean * np.log(a_mean + 1e-10)))
    b_eff = np.exp(-np.sum(b_mean * np.log(b_mean + 1e-10)))
    ax.text(0.97, 0.95,
            f"Effective # of actions:\n"
            f"  {label_a}: {a_eff:.1f}\n"
            f"  {label_b}: {b_eff:.1f}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(figures_dir / "mean_action_probabilities.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 4] Saved: mean_action_probabilities.png")


def print_summary(run_a_result, run_b_result, step_label, label_a, label_b):
    print("\n" + "=" * 70)
    print(f"ACTION DISTRIBUTION SUMMARY ({step_label})")
    print("=" * 70)
    for name, res in [(label_a, run_a_result), (label_b, run_b_result)]:
        print(f"\n{name}:")
        print(f"  Win rate:         {res['win_rate']:.1%}")
        print(f"  Mean ep length:   {res['mean_ep_length']:.1f}")
        print(f"  Total steps:      {res['n_steps']:,}")
        counts = Counter(res["actions"])
        top5 = counts.most_common(5)
        n_steps = res["n_steps"]
        top5_str = ", ".join(
            f"{ACTION_NAMES[a]}({c / n_steps * 100:.1f}%)" for a, c in top5
        )
        print(f"  Top 5 actions:    {top5_str}")
        actions_used = sum(1 for a in range(19) if counts.get(a, 0) / res["n_steps"] > 0.01)
        print(f"  Actions >1% freq: {actions_used}/19")
        if res["entropies"] is not None:
            print(f"  Mean entropy:     {np.mean(res['entropies']):.4f}")
            print(f"  Entropy std:      {np.std(res['entropies']):.4f}")


def make_output_dir(output_dir=None):
    if output_dir:
        p = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = REPO_ROOT / "outputs" / f"analysis_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a-ckpt-dir", type=str, required=True,
                        help="Run A checkpoint directory")
    parser.add_argument("--run-b-ckpt-dir", type=str, required=True,
                        help="Run B checkpoint directory")
    parser.add_argument("--run-a-prefix", type=str, default="",
                        help="Run A checkpoint filename prefix (default: match all)")
    parser.add_argument("--run-b-prefix", type=str, default="",
                        help="Run B checkpoint filename prefix (default: match all)")
    parser.add_argument("--label-a", type=str, default="Run A",
                        help="Display label for run A (default: 'Run A')")
    parser.add_argument("--label-b", type=str, default="Run B",
                        help="Display label for run B (default: 'Run B')")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--step", type=int, default=5_000_000,
                        help="Target checkpoint step for comparison")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures (default: outputs/analysis_<timestamp>)")
    args = parser.parse_args()

    run_a_ckpt_dir = Path(args.run_a_ckpt_dir)
    run_b_ckpt_dir = Path(args.run_b_ckpt_dir)

    try:
        import gfootball
    except ImportError:
        print("ERROR: gfootball not installed. This script requires the GRF environment.")
        print("Install it or run this script on the training server.")
        sys.exit(1)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    a_step, a_path = find_checkpoint(run_a_ckpt_dir, args.run_a_prefix, args.step)
    b_step, b_path = find_checkpoint(run_b_ckpt_dir, args.run_b_prefix, args.step)

    if a_path is None or b_path is None:
        print("ERROR: Could not find checkpoints for both runs.")
        if a_path is None:
            print(f"  Missing run A checkpoint near step {args.step}")
        if b_path is None:
            print(f"  Missing run B checkpoint near step {args.step}")
        sys.exit(1)

    step_label = f"{args.step / 1e6:.0f}M"
    print(f"{args.label_a} checkpoint: {a_path.name} (step {a_step:,})")
    print(f"{args.label_b} checkpoint: {b_path.name} (step {b_step:,})")
    print(f"Running {args.n_episodes} episodes per model...\n")

    env = create_eval_env()

    load_kwargs = dict(
        env=env,
        custom_objects={"policy_class": MaskableActorCriticPolicy},
    )

    print(f"Loading and evaluating {args.label_a} model...")
    run_a_model = MaskablePPO.load(str(a_path), **load_kwargs)
    run_a_result = run_episodes(run_a_model, env, args.n_episodes)
    print(f"  {args.label_a}: win_rate={run_a_result['win_rate']:.1%}, "
          f"steps={run_a_result['n_steps']:,}")

    print(f"Loading and evaluating {args.label_b} model...")
    run_b_model = MaskablePPO.load(str(b_path), **load_kwargs)
    run_b_result = run_episodes(run_b_model, env, args.n_episodes)
    print(f"  {args.label_b}: win_rate={run_b_result['win_rate']:.1%}, "
          f"steps={run_b_result['n_steps']:,}")

    env.close()

    figures_dir = make_output_dir(args.output_dir)
    print(f"Output directory: {figures_dir}")

    fig1_action_frequency(run_a_result, run_b_result, step_label,
                          args.label_a, args.label_b, figures_dir)
    fig2_entropy_comparison(run_a_result, run_b_result, step_label,
                            args.label_a, args.label_b, figures_dir)
    fig3_probability_heatmap(run_a_result, run_b_result, step_label,
                             args.label_a, args.label_b, figures_dir)
    fig4_mean_action_probs(run_a_result, run_b_result, step_label,
                           args.label_a, args.label_b, figures_dir)
    print_summary(run_a_result, run_b_result, step_label, args.label_a, args.label_b)
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
