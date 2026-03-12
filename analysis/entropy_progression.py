"""
Entropy Progression Analysis — measure how the policy's normalized entropy
evolves across training checkpoints for two runs.

This helps validate the entropy_threshold parameter for entropy-gated masking
by showing when the agent transitions from confused (high entropy) to confident
(low entropy) during training.

Requires: gfootball, sb3_contrib, stable_baselines3, torch, matplotlib, numpy

Usage:
    python analysis/entropy_progression.py \\
        --run-a-ckpt-dir outputs/baseline/checkpoints \\
        --run-b-ckpt-dir outputs/masking/checkpoints \\
        --label-a "Baseline PPO" --label-b "MaskablePPO + LLM"
    python analysis/entropy_progression.py --n-episodes 20 --output-dir outputs/analysis_custom
"""

import math
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NUM_ACTIONS = 19
MAX_ENTROPY = math.log(NUM_ACTIONS)


ACTION_NAMES = [
    "Idle", "Left", "TopLeft", "Top", "TopRight",
    "Right", "BtmRight", "Bottom", "BtmLeft",
    "LongPass", "HighPass", "ShortPass", "Shot",
    "Sprint", "RelDir", "RelSprint",
    "Sliding", "Dribble", "RelDribble",
]


def list_checkpoints(ckpt_dir, prefix):
    """Return sorted list of (step, path) for all checkpoints in directory."""
    if not ckpt_dir.exists():
        return []
    results = []
    for f in ckpt_dir.iterdir():
        if f.suffix == ".zip" and f.name.startswith(prefix):
            parts = f.stem.split("_")
            for p in reversed(parts):
                if p.isdigit():
                    results.append((int(p), f))
                    break
    results.sort(key=lambda x: x[0])
    return results


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


def load_model(path, env):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    return MaskablePPO.load(
        str(path), env=env,
        custom_objects={"policy_class": MaskableActorCriticPolicy},
    )


def compute_entropy_stats(model, env, n_episodes):
    """Run episodes, compute normalized entropy at every step.
    Returns dict with per-step entropies and episode outcomes."""
    import torch

    all_entropies = []
    wins = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(model.policy.device)
            with torch.no_grad():
                try:
                    features = model.policy.extract_features(
                        obs_t, model.policy.features_extractor
                    )
                except TypeError:
                    features = model.policy.extract_features(obs_t)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                logits = model.policy.action_net(latent_pi)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()

            h_norm = entropy / MAX_ENTROPY
            all_entropies.append(h_norm)

            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, _ = env.step(int(action))
            ep_reward += reward

        if ep_reward > 0:
            wins += 1

    entropies = np.array(all_entropies)
    return {
        "entropies": entropies,
        "mean": float(entropies.mean()),
        "std": float(entropies.std()),
        "median": float(np.median(entropies)),
        "p10": float(np.percentile(entropies, 10)),
        "p25": float(np.percentile(entropies, 25)),
        "p75": float(np.percentile(entropies, 75)),
        "p90": float(np.percentile(entropies, 90)),
        "frac_above_06": float((entropies >= 0.6).mean()),
        "frac_above_05": float((entropies >= 0.5).mean()),
        "frac_above_04": float((entropies >= 0.4).mean()),
        "win_rate": wins / n_episodes,
        "n_steps": len(entropies),
    }


def fig1_entropy_over_training(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """Mean normalized entropy vs training step for both models."""
    fig, ax = plt.subplots(figsize=(13, 6))

    if run_a_data:
        steps = [s / 1e6 for s in sorted(run_a_data)]
        means = [run_a_data[s]["mean"] for s in sorted(run_a_data)]
        p25 = [run_a_data[s]["p25"] for s in sorted(run_a_data)]
        p75 = [run_a_data[s]["p75"] for s in sorted(run_a_data)]
        ax.plot(steps, means, "o-", color="#1565C0", linewidth=2, markersize=5,
                label=label_a)
        ax.fill_between(steps, p25, p75, color="#1565C0", alpha=0.15)

    if run_b_data:
        steps = [s / 1e6 for s in sorted(run_b_data)]
        means = [run_b_data[s]["mean"] for s in sorted(run_b_data)]
        p25 = [run_b_data[s]["p25"] for s in sorted(run_b_data)]
        p75 = [run_b_data[s]["p75"] for s in sorted(run_b_data)]
        ax.plot(steps, means, "s-", color="#C62828", linewidth=2, markersize=5,
                label=label_b)
        ax.fill_between(steps, p25, p75, color="#C62828", alpha=0.15)

    for thresh, color, ls in [(0.6, "#4CAF50", "--"), (0.5, "#FF9800", ":"), (0.4, "#9C27B0", ":")]:
        ax.axhline(y=thresh, color=color, linestyle=ls, alpha=0.6,
                   label=f"Threshold = {thresh}")

    ax.set_xlabel("Training Steps (millions)", fontsize=13)
    ax.set_ylabel("Normalized Entropy  H(pi) / log(19)", fontsize=13)
    ax.set_title("Policy Entropy Progression Over Training", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "entropy_over_training.png", dpi=150)
    plt.close(fig)
    print("[Fig 1] Saved: entropy_over_training.png")


def fig2_entropy_distributions(run_a_data, run_b_data, label_a, label_b, figures_dir,
                                highlight_steps=None):
    """Entropy distribution histograms at selected training steps."""
    if highlight_steps is None:
        all_steps = sorted(set(run_a_data) | set(run_b_data))
        if len(all_steps) <= 6:
            highlight_steps = all_steps
        else:
            indices = np.linspace(0, len(all_steps) - 1, 6, dtype=int)
            highlight_steps = [all_steps[i] for i in indices]

    n = len(highlight_steps)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    for i, step in enumerate(highlight_steps):
        ax = axes[i]
        step_label = f"{step / 1e6:.1f}M"

        if step in run_a_data:
            ax.hist(run_a_data[step]["entropies"], bins=40, alpha=0.55,
                    color="#1565C0", density=True, label=label_a)
        if step in run_b_data:
            ax.hist(run_b_data[step]["entropies"], bins=40, alpha=0.55,
                    color="#C62828", density=True, label=label_b)

        ax.axvline(x=0.6, color="#4CAF50", linestyle="--", alpha=0.7)
        ax.set_title(f"Step {step_label}", fontsize=11)
        ax.set_xlabel("H_norm" if i >= (n + 1) // 2 else "")
        ax.set_ylabel("Density" if i % ((n + 1) // 2) == 0 else "")
        ax.set_xlim(0, 1.05)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Entropy Distributions at Different Training Stages", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "entropy_distributions.png", dpi=150)
    plt.close(fig)
    print("[Fig 2] Saved: entropy_distributions.png")


def fig3_mask_activation_fraction(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """For various thresholds, what fraction of steps would trigger LLM masking?"""
    thresholds = np.arange(0.3, 0.85, 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for label, data, ax, color in [
        (label_a, run_a_data, ax1, "#1565C0"),
        (label_b, run_b_data, ax2, "#C62828"),
    ]:
        if not data:
            ax.set_visible(False)
            continue

        steps_sorted = sorted(data)
        for thresh in thresholds:
            fracs = []
            for s in steps_sorted:
                frac = float((data[s]["entropies"] >= thresh).mean())
                fracs.append(frac)
            ax.plot([s / 1e6 for s in steps_sorted], [f * 100 for f in fracs],
                    marker=".", linewidth=1.5, label=f"t={thresh:.2f}")

        ax.set_xlabel("Training Steps (millions)", fontsize=12)
        ax.set_ylabel("% Steps Where Mask Would Activate", fontsize=12)
        ax.set_title(f"{label}", fontsize=13)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Fraction of Steps Above Entropy Threshold Over Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_activation_fraction.png", dpi=150)
    plt.close(fig)
    print("[Fig 3] Saved: mask_activation_fraction.png")


def fig4_win_rate_vs_entropy(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """Scatter: mean entropy vs win rate at each checkpoint."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if run_a_data:
        means = [run_a_data[s]["mean"] for s in sorted(run_a_data)]
        wrs = [run_a_data[s]["win_rate"] for s in sorted(run_a_data)]
        steps = [s / 1e6 for s in sorted(run_a_data)]
        ax.scatter(means, [w * 100 for w in wrs], c=steps, cmap="Blues",
                   s=80, edgecolors="black", linewidths=0.5, label=label_a,
                   marker="o", vmin=0)
        for m, w, sl in zip(means, wrs, steps):
            ax.annotate(f"{sl:.0f}M", (m, w * 100), fontsize=7, ha="center",
                        va="bottom", color="#1565C0")

    if run_b_data:
        means = [run_b_data[s]["mean"] for s in sorted(run_b_data)]
        wrs = [run_b_data[s]["win_rate"] for s in sorted(run_b_data)]
        steps = [s / 1e6 for s in sorted(run_b_data)]
        ax.scatter(means, [w * 100 for w in wrs], c=steps, cmap="Reds",
                   s=80, edgecolors="black", linewidths=0.5, label=label_b,
                   marker="s", vmin=0)
        for m, w, sl in zip(means, wrs, steps):
            ax.annotate(f"{sl:.0f}M", (m, w * 100), fontsize=7, ha="center",
                        va="bottom", color="#C62828")

    ax.axvline(x=0.6, color="#4CAF50", linestyle="--", alpha=0.6, label="Threshold 0.6")
    ax.set_xlabel("Mean Normalized Entropy", fontsize=13)
    ax.set_ylabel("Win Rate (%)", fontsize=13)
    ax.set_title("Win Rate vs. Policy Entropy Across Training", fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "winrate_vs_entropy.png", dpi=150)
    plt.close(fig)
    print("[Fig 4] Saved: winrate_vs_entropy.png")


def print_summary(run_a_data, run_b_data, label_a, label_b):
    print("\n" + "=" * 80)
    print("ENTROPY PROGRESSION SUMMARY")
    print("=" * 80)

    for label, data in [(label_a, run_a_data), (label_b, run_b_data)]:
        if not data:
            print(f"\n{label}: no checkpoints found")
            continue
        print(f"\n{label}:")
        print(f"  {'Step':>10s}  {'Mean':>6s}  {'Std':>6s}  {'Med':>6s}  "
              f"{'P10':>6s}  {'P90':>6s}  {'>=0.6':>6s}  {'>=0.5':>6s}  "
              f"{'WinR':>6s}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  "
              f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for step in sorted(data):
            d = data[step]
            print(f"  {step/1e6:>8.1f}M  {d['mean']:>6.3f}  {d['std']:>6.3f}  "
                  f"{d['median']:>6.3f}  {d['p10']:>6.3f}  {d['p90']:>6.3f}  "
                  f"{d['frac_above_06']*100:>5.1f}%  {d['frac_above_05']*100:>5.1f}%  "
                  f"{d['win_rate']*100:>5.1f}%")


def make_output_dir(output_dir=None):
    if output_dir:
        p = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = REPO_ROOT / "outputs" / f"analysis_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how policy entropy evolves across training checkpoints."
    )
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
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Episodes per checkpoint for entropy measurement (default: 10)",
    )
    parser.add_argument(
        "--steps", type=int, nargs="*", default=None,
        help="Specific checkpoint steps to analyze (default: all available)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output figures (default: outputs/analysis_<timestamp>)",
    )
    args = parser.parse_args()

    run_a_ckpt_dir = Path(args.run_a_ckpt_dir)
    run_b_ckpt_dir = Path(args.run_b_ckpt_dir)

    try:
        import gfootball
    except ImportError:
        print("ERROR: gfootball not installed.")
        sys.exit(1)

    a_checkpoints = list_checkpoints(run_a_ckpt_dir, args.run_a_prefix)
    b_checkpoints = list_checkpoints(run_b_ckpt_dir, args.run_b_prefix)

    if args.steps:
        target = set(args.steps)
        a_checkpoints = [(s, p) for s, p in a_checkpoints if s in target]
        b_checkpoints = [(s, p) for s, p in b_checkpoints if s in target]

    print(f"{args.label_a} checkpoints found: {len(a_checkpoints)}")
    print(f"{args.label_b} checkpoints found:  {len(b_checkpoints)}")

    if not a_checkpoints and not b_checkpoints:
        print("ERROR: No checkpoints found.")
        sys.exit(1)

    env = create_eval_env()

    run_a_data = {}
    for step, path in a_checkpoints:
        print(f"\n[{args.label_a}] step {step/1e6:.1f}M  ({path.name})")
        model = load_model(path, env)
        stats = compute_entropy_stats(model, env, args.n_episodes)
        run_a_data[step] = stats
        print(f"  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
              f">=0.6: {stats['frac_above_06']*100:.1f}%  "
              f"win_rate={stats['win_rate']:.0%}")

    run_b_data = {}
    for step, path in b_checkpoints:
        print(f"\n[{args.label_b}] step {step/1e6:.1f}M  ({path.name})")
        model = load_model(path, env)
        stats = compute_entropy_stats(model, env, args.n_episodes)
        run_b_data[step] = stats
        print(f"  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
              f">=0.6: {stats['frac_above_06']*100:.1f}%  "
              f"win_rate={stats['win_rate']:.0%}")

    env.close()

    figures_dir = make_output_dir(args.output_dir)
    print(f"\nOutput directory: {figures_dir}")

    fig1_entropy_over_training(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
    fig2_entropy_distributions(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
    fig3_mask_activation_fraction(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
    fig4_win_rate_vs_entropy(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
    print_summary(run_a_data, run_b_data, args.label_a, args.label_b)
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
