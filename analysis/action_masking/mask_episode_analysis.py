"""
Mask Episode Analysis — run the masked agent through full episodes with the
GroqCoach to visualize mask behavior, staleness, and episode boundary issues.

Requires: gfootball, sb3_contrib, stable_baselines3, torch, matplotlib, numpy

Usage:
    python analysis/mask_episode_analysis.py [--n-episodes 5] [--step 5000000]
    python analysis/mask_episode_analysis.py --output-dir outputs/analysis_custom

This script demonstrates:
1. Mask heatmap over episode timesteps (which actions are available/forbidden)
2. Number of allowed actions over time within episodes
3. Episode boundary mask persistence (mask doesn't reset on env.reset())
4. What the agent would have done without masking vs with masking
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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


def run_episodes_with_coach(model, env, coach, n_episodes, coach_interval=100):
    """
    Run episodes simulating the training-time mask behavior:
    - Coach updates every coach_interval steps
    - Mask persists between updates
    - Mask does NOT reset on episode boundaries (replicating the bug)
    """
    import torch
    from core.wrappers import observation_to_features

    all_episode_data = []
    current_mask = np.ones(19, dtype=bool)
    steps_since_update = 0
    global_step = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_data = {
            "masks": [],
            "actions_masked": [],
            "actions_unmasked": [],
            "n_allowed": [],
            "roles": [],
            "steps_since_update": [],
            "obs_list": [],
            "reward": 0.0,
        }

        while not done:
            if global_step % coach_interval == 0:
                features = observation_to_features(obs)
                advice = coach.get_coaching_advice(features)
                forbidden = advice.get("Forbidden_Actions", [])
                current_mask = np.ones(19, dtype=bool)
                for a in forbidden:
                    if 0 <= a < 19:
                        current_mask[a] = False
                if not np.any(current_mask):
                    current_mask[0] = True
                steps_since_update = 0
                role = advice.get("Role", "Unknown")
            else:
                role = ep_data["roles"][-1] if ep_data["roles"] else "Unknown"

            ep_data["masks"].append(current_mask.copy())
            ep_data["n_allowed"].append(int(current_mask.sum()))
            ep_data["roles"].append(role)
            ep_data["steps_since_update"].append(steps_since_update)

            # Action WITH masking (what training sees)
            masked_logits = _get_masked_action(model, obs, current_mask)
            ep_data["actions_masked"].append(masked_logits)

            # Action WITHOUT masking (what the policy would prefer)
            unmasked_action = _get_unmasked_action(model, obs)
            ep_data["actions_unmasked"].append(unmasked_action)

            obs, reward, done, info = env.step(masked_logits)
            ep_data["reward"] += reward
            steps_since_update += 1
            global_step += 1

        all_episode_data.append(ep_data)

    return all_episode_data


def _get_masked_action(model, obs, mask):
    """Get action respecting the mask."""
    logits = get_action_logits(model, obs)
    logits[~mask] = -1e8
    return int(np.argmax(logits))


def _get_unmasked_action(model, obs):
    """Get the action the policy would take without any mask."""
    logits = get_action_logits(model, obs)
    return int(np.argmax(logits))


def fig1_mask_heatmap(episode_data, figures_dir, ep_idx=0):
    """Heatmap showing which actions are allowed (green) / forbidden (red) at each timestep."""
    ep = episode_data[ep_idx]
    masks = np.array(ep["masks"])  # (T, 19) bool array
    T = len(masks)

    fig, ax = plt.subplots(figsize=(min(16, max(10, T * 0.12)), 7))
    display = masks.T.astype(float)
    im = ax.imshow(display, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xlabel("Episode Timestep", fontsize=12)
    ax.set_ylabel("Action", fontsize=12)
    ax.set_yticks(range(19))
    ax.set_yticklabels([f"{i} {ACTION_NAMES[i]}" for i in range(19)], fontsize=9)
    ax.set_title(f"Action Mask Over Episode (Ep {ep_idx}, T={T}, "
                 f"reward={ep['reward']:.1f})", fontsize=13)

    green_patch = mpatches.Patch(color="#2ca02c", label="Allowed")
    red_patch = mpatches.Patch(color="#d62728", label="Forbidden")
    ax.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=10)

    # Mark coach update points
    update_steps = [i for i, s in enumerate(ep["steps_since_update"]) if s == 0]
    for us in update_steps:
        ax.axvline(x=us, color="blue", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(figures_dir / f"mask_heatmap_ep{ep_idx}.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 1] Saved: mask_heatmap_ep{ep_idx}.png")


def fig2_allowed_actions_over_time(episode_data, figures_dir):
    """Number of allowed actions over time, with episode boundaries marked."""
    fig, ax = plt.subplots(figsize=(14, 5))

    global_step = 0
    ep_boundaries = []
    all_steps = []
    all_allowed = []

    for ep_idx, ep in enumerate(episode_data):
        ep_boundaries.append(global_step)
        for n in ep["n_allowed"]:
            all_steps.append(global_step)
            all_allowed.append(n)
            global_step += 1

    ax.plot(all_steps, all_allowed, color="steelblue", linewidth=1, alpha=0.7)
    ax.axhline(y=19, color="green", linestyle=":", alpha=0.5, label="No masking (19)")
    for i, eb in enumerate(ep_boundaries):
        ax.axvline(x=eb, color="red", linestyle="--", alpha=0.6,
                   linewidth=1.5, label="Episode boundary" if i == 0 else "_nolegend_")

    ax.set_xlabel("Global Timestep", fontsize=12)
    ax.set_ylabel("Allowed Actions (out of 19)", fontsize=12)
    ax.set_title("Allowed Actions Over Time — Note Mask Persists Across Episode Boundaries",
                 fontsize=13)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "mask_allowed_over_episodes.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Saved: mask_allowed_over_episodes.png")


def fig3_mask_staleness(episode_data, figures_dir):
    """Distribution of mask staleness (steps since last coach update)."""
    all_staleness = []
    for ep in episode_data:
        all_staleness.extend(ep["steps_since_update"])

    fig, ax = plt.subplots(figsize=(10, 5))
    max_stale = max(all_staleness) if all_staleness else 100
    bins = range(0, min(max_stale + 2, 200))
    ax.hist(all_staleness, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Steps Since Last Mask Update", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Mask Staleness Distribution (How Old Is the Current Mask?)", fontsize=13)
    ax.axvline(x=np.mean(all_staleness), color="red", linestyle="--",
               label=f"Mean = {np.mean(all_staleness):.1f}")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "mask_staleness.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 3] Saved: mask_staleness.png")


def fig4_masked_vs_unmasked_actions(episode_data, figures_dir):
    """Compare what the policy wants to do vs what it's forced to do by the mask."""
    all_masked = []
    all_unmasked = []
    disagreements = 0

    for ep in episode_data:
        all_masked.extend(ep["actions_masked"])
        all_unmasked.extend(ep["actions_unmasked"])
        disagreements += sum(
            1 for m, u in zip(ep["actions_masked"], ep["actions_unmasked"]) if m != u
        )

    total = len(all_masked)
    disagree_rate = disagreements / total if total > 0 else 0

    masked_counts = Counter(all_masked)
    unmasked_counts = Counter(all_unmasked)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(19)
    width = 0.38

    m_freq = np.array([masked_counts.get(i, 0) / total for i in range(19)])
    u_freq = np.array([unmasked_counts.get(i, 0) / total for i in range(19)])

    ax1.bar(x - width / 2, u_freq * 100, width, label="Unmasked (preferred)", color="#4CAF50",
            edgecolor="black", alpha=0.85)
    ax1.bar(x + width / 2, m_freq * 100, width, label="Masked (forced)", color="#F44336",
            edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Action", fontsize=11)
    ax1.set_ylabel("Frequency (%)", fontsize=11)
    ax1.set_title("Policy Preferred vs. Mask-Forced Actions", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}\n{ACTION_NAMES[i]}" for i in range(19)], fontsize=7)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Confusion matrix: unmasked → masked redirection
    confusion = np.zeros((19, 19))
    for m, u in zip(all_masked, all_unmasked):
        confusion[u, m] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, where=row_sums > 0,
                               out=np.zeros_like(confusion))

    active_actions = [i for i in range(19)
                      if unmasked_counts.get(i, 0) > total * 0.005
                      or masked_counts.get(i, 0) > total * 0.005]
    if len(active_actions) > 2:
        sub = confusion_norm[np.ix_(active_actions, active_actions)]
        im = ax2.imshow(sub, cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(len(active_actions)))
        ax2.set_xticklabels([ACTION_NAMES[i] for i in active_actions], fontsize=8, rotation=45)
        ax2.set_yticks(range(len(active_actions)))
        ax2.set_yticklabels([ACTION_NAMES[i] for i in active_actions], fontsize=8)
        ax2.set_xlabel("Executed Action (masked)", fontsize=11)
        ax2.set_ylabel("Preferred Action (unmasked)", fontsize=11)
        ax2.set_title(f"Action Redirection Matrix\n(Disagree rate: {disagree_rate:.1%})",
                      fontsize=12)
        plt.colorbar(im, ax=ax2, fraction=0.04)

    fig.tight_layout()
    fig.savefig(figures_dir / "masked_vs_unmasked_actions.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 4] Saved: masked_vs_unmasked_actions.png")


def fig5_episode_boundary_persistence(episode_data, figures_dir):
    """Show that mask at end of episode N is the same as start of episode N+1."""
    if len(episode_data) < 2:
        print("[Fig 5] Skipped (need at least 2 episodes)")
        return

    fig, axes = plt.subplots(len(episode_data) - 1, 1,
                              figsize=(12, 3 * (len(episode_data) - 1)))
    if len(episode_data) - 1 == 1:
        axes = [axes]

    for i in range(len(episode_data) - 1):
        ax = axes[i]
        end_mask = episode_data[i]["masks"][-1]
        start_mask = episode_data[i + 1]["masks"][0]

        x = np.arange(19)
        width = 0.35
        ax.bar(x - width / 2, end_mask.astype(float), width,
               color="#FF9800", edgecolor="black", alpha=0.8, label=f"End of Ep {i}")
        ax.bar(x + width / 2, start_mask.astype(float), width,
               color="#9C27B0", edgecolor="black", alpha=0.8, label=f"Start of Ep {i+1}")

        identical = np.array_equal(end_mask, start_mask)
        ax.set_title(f"Episode {i} -> {i+1} Boundary - "
                     f"{'IDENTICAL (mask persists!)' if identical else 'Different'}",
                     fontsize=11, color="red" if identical else "green")
        ax.set_xticks(x)
        ax.set_xticklabels([ACTION_NAMES[j] for j in range(19)], fontsize=7, rotation=45)
        ax.set_ylabel("Allowed (1/0)")
        ax.set_ylim(-0.1, 1.3)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mask Persistence Across Episode Boundaries (Bug)", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_episode_boundary.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 5] Saved: mask_episode_boundary.png")


def print_summary(episode_data):
    print("\n" + "=" * 70)
    print("MASK EPISODE ANALYSIS SUMMARY")
    print("=" * 70)
    total_steps = sum(len(ep["masks"]) for ep in episode_data)
    total_disagree = sum(
        sum(1 for m, u in zip(ep["actions_masked"], ep["actions_unmasked"]) if m != u)
        for ep in episode_data
    )
    all_allowed = [n for ep in episode_data for n in ep["n_allowed"]]
    boundary_persist = 0
    for i in range(len(episode_data) - 1):
        if np.array_equal(episode_data[i]["masks"][-1], episode_data[i + 1]["masks"][0]):
            boundary_persist += 1

    print(f"Episodes: {len(episode_data)}")
    print(f"Total steps: {total_steps:,}")
    print(f"Mask-policy disagreement rate: {total_disagree / total_steps:.1%}")
    print(f"Mean allowed actions: {np.mean(all_allowed):.1f}")
    print(f"Min allowed actions: {np.min(all_allowed)}")
    print(f"Episode boundary mask persistence: "
          f"{boundary_persist}/{len(episode_data)-1} transitions")
    print(f"Wins: {sum(1 for ep in episode_data if ep['reward'] > 0)}/{len(episode_data)}")


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
    parser.add_argument("--masking-ckpt-dir", type=str, required=True,
                        help="Masking checkpoint directory")
    parser.add_argument("--masking-prefix", type=str, default="masking_ppo_3v1",
                        help="Masking checkpoint filename prefix (default: masking_ppo_3v1)")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--step", type=int, default=5_000_000)
    parser.add_argument("--coach-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures (default: outputs/analysis_<timestamp>)")
    args = parser.parse_args()

    masking_ckpt_dir = Path(args.masking_ckpt_dir)

    try:
        import gfootball
    except ImportError:
        print("ERROR: gfootball not installed. This script requires the GRF environment.")
        print("Install it or run this script on the training server.")
        sys.exit(1)

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from core.coach import GroqCoach

    m_step, m_path = find_checkpoint(masking_ckpt_dir, args.masking_prefix, args.step)
    if m_path is None:
        print("ERROR: Could not find masking checkpoint.")
        sys.exit(1)

    env = create_eval_env()

    print(f"Loading masking model: {m_path.name} (step {m_step:,})")
    model = MaskablePPO.load(
        str(m_path), env=env,
        custom_objects={"policy_class": MaskableActorCriticPolicy},
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)
    print(f"Using GroqCoach (coach_interval={args.coach_interval})")
    coach = GroqCoach(api_key=groq_api_key)

    print(f"Running {args.n_episodes} episodes...\n")

    episode_data = run_episodes_with_coach(
        model, env, coach, args.n_episodes, coach_interval=args.coach_interval
    )
    env.close()

    figures_dir = make_output_dir(args.output_dir)
    print(f"Output directory: {figures_dir}")

    fig1_mask_heatmap(episode_data, figures_dir, ep_idx=0)
    if len(episode_data) > 1:
        fig1_mask_heatmap(episode_data, figures_dir, ep_idx=1)
    fig2_allowed_actions_over_time(episode_data, figures_dir)
    fig3_mask_staleness(episode_data, figures_dir)
    fig4_masked_vs_unmasked_actions(episode_data, figures_dir)
    fig5_episode_boundary_persistence(episode_data, figures_dir)
    print_summary(episode_data)
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
