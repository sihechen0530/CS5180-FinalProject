"""
Mask Statistics Analysis — parse training logs to quantify LLM mask aggressiveness.

No GRF dependency required.

Usage:
    python analysis/mask_statistics.py
    python analysis/mask_statistics.py --output-dir outputs/analysis_custom
"""

import re
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ACTION_NAMES = {
    0: "Idle", 1: "Left", 2: "TopLeft", 3: "Top", 4: "TopRight",
    5: "Right", 6: "BtmRight", 7: "Bottom", 8: "BtmLeft",
    9: "LongPass", 10: "HighPass", 11: "ShortPass", 12: "Shot",
    13: "Sprint", 14: "RelDir", 15: "RelSprint",
    16: "Sliding", 17: "Dribble", 18: "RelDribble"
}

ACTION_GROUPS = {
    "Movement (1-8)": list(range(1, 9)),
    "Passing (9-11)": [9, 10, 11],
    "Shooting (12)": [12],
    "Modifiers (0,13-18)": [0, 13, 14, 15, 16, 17, 18],
}


COACH_PATTERN = re.compile(
    r'\[Coach Env (\d+)\] Step (\d+) - Role: (.+?) \| Masked: \[([^\]]*)\]'
)


def parse_coach_lines(log_path, step_offset=0):
    entries = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = COACH_PATTERN.search(line)
            if m:
                env_id = int(m.group(1))
                step = int(m.group(2)) + step_offset
                role = m.group(3).strip()
                masked_str = m.group(4).strip()
                masked = [int(x.strip()) for x in masked_str.split(",")] if masked_str else []
                entries.append({
                    "env_id": env_id,
                    "step": step,
                    "role": role,
                    "masked": set(masked),
                    "n_masked": len(masked),
                    "n_allowed": 19 - len(masked),
                })
    return entries


def fig1_allowed_actions_histogram(entries, figures_dir):
    """Distribution of number of allowed actions per mask update."""
    n_allowed = [e["n_allowed"] for e in entries]
    fig, ax = plt.subplots(figsize=(10, 5))
    counts, bins, patches = ax.hist(n_allowed, bins=range(0, 21), align="left",
                                     edgecolor="black", color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Number of Allowed Actions (out of 19)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Distribution of Allowed Actions per LLM Coach Mask Update", fontsize=14)
    ax.set_xticks(range(0, 20))
    ax.axvline(x=np.median(n_allowed), color="red", linestyle="--", linewidth=2,
               label=f"Median = {np.median(n_allowed):.0f}")
    ax.axvline(x=np.mean(n_allowed), color="orange", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(n_allowed):.1f}")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    extreme = sum(1 for n in n_allowed if n <= 1)
    very_low = sum(1 for n in n_allowed if n <= 4)
    ax.text(0.97, 0.95,
            f"Total mask updates: {len(n_allowed):,}\n"
            f"<=1 allowed (deadlock): {extreme:,} ({extreme/len(n_allowed)*100:.1f}%)\n"
            f"<=4 allowed: {very_low:,} ({very_low/len(n_allowed)*100:.1f}%)",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(figures_dir / "mask_allowed_actions_histogram.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 1] Saved: mask_allowed_actions_histogram.png")
    return n_allowed


def fig2_per_action_forbidden_heatmap(entries, figures_dir):
    """Heatmap of how frequently each action is forbidden, broken down by role."""
    role_counts = Counter(e["role"] for e in entries)
    top_roles = [r for r, _ in role_counts.most_common(8)]

    matrix = np.zeros((len(top_roles), 19))
    role_totals = defaultdict(int)
    for e in entries:
        if e["role"] in top_roles:
            idx = top_roles.index(e["role"])
            role_totals[e["role"]] += 1
            for a in e["masked"]:
                if 0 <= a < 19:
                    matrix[idx, a] += 1

    for i, role in enumerate(top_roles):
        if role_totals[role] > 0:
            matrix[i] /= role_totals[role]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(19))
    ax.set_xticklabels([f"{i}\n{ACTION_NAMES[i]}" for i in range(19)],
                        fontsize=8, rotation=0)
    ax.set_yticks(range(len(top_roles)))
    ax.set_yticklabels([f"{r} (n={role_totals[r]:,})" for r in top_roles], fontsize=10)
    ax.set_xlabel("Action ID", fontsize=12)
    ax.set_ylabel("Role assigned by LLM", fontsize=12)
    ax.set_title("Fraction of Time Each Action Is Forbidden, by LLM-Assigned Role", fontsize=13)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Forbidden Fraction", fontsize=11)

    fig.tight_layout()
    fig.savefig(figures_dir / "mask_per_action_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Saved: mask_per_action_heatmap.png")


def fig3_role_distribution(entries, figures_dir):
    """Role distribution pie/bar chart."""
    role_counts = Counter(e["role"] for e in entries)
    roles = [r for r, _ in role_counts.most_common(10)]
    counts = [role_counts[r] for r in roles]
    other = sum(role_counts[r] for r in role_counts if r not in roles)
    if other > 0:
        roles.append("Other")
        counts.append(other)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(roles)), counts, color=plt.cm.Set2(np.linspace(0, 1, len(roles))),
                    edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(roles, fontsize=11)
    ax.set_xlabel("Number of Mask Updates", fontsize=12)
    ax.set_title("LLM Coach Role Assignment Distribution", fontsize=14)
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({count / len(entries) * 100:.1f}%)",
                va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_role_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 3] Saved: mask_role_distribution.png")


def fig4_allowed_actions_timeseries(entries, figures_dir):
    """Time series: average allowed actions over training steps."""
    step_data = defaultdict(list)
    for e in entries:
        step_data[e["step"]].append(e["n_allowed"])

    steps = sorted(step_data.keys())
    means = [np.mean(step_data[s]) for s in steps]

    window = min(50, len(means) // 10 + 1)
    if window > 1:
        smoothed = np.convolve(means, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window // 2: window // 2 + len(smoothed)]
    else:
        smoothed = means
        smoothed_steps = steps

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter([s / 1e6 for s in steps], means, alpha=0.08, s=4, color="steelblue", label="Raw")
    ax.plot([s / 1e6 for s in smoothed_steps], smoothed, color="red", linewidth=2,
            label=f"Rolling avg (w={window})")
    ax.axhline(y=19, color="green", linestyle=":", alpha=0.5, label="All 19 allowed (no mask)")
    ax.set_xlabel("Training Step (millions)", fontsize=12)
    ax.set_ylabel("Avg. Allowed Actions (out of 19)", fontsize=12)
    ax.set_title("Average Number of Allowed Actions Over Training", fontsize=14)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_allowed_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 4] Saved: mask_allowed_timeseries.png")


def fig5_per_env_diversity(entries, figures_dir):
    """At each coach callback step, how many unique mask configurations across the 16 envs?"""
    step_masks = defaultdict(list)
    for e in entries:
        step_masks[e["step"]].append(frozenset(e["masked"]))

    steps = sorted(step_masks.keys())
    n_unique = [len(set(step_masks[s])) for s in steps]
    n_envs = [len(step_masks[s]) for s in steps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter([s / 1e6 for s in steps], n_unique, s=6, alpha=0.3, color="steelblue")

    window = min(50, len(n_unique) // 10 + 1)
    if window > 1 and len(n_unique) > window:
        smoothed = np.convolve(n_unique, np.ones(window) / window, mode="valid")
        smoothed_steps = steps[window // 2: window // 2 + len(smoothed)]
        ax.plot([s / 1e6 for s in smoothed_steps], smoothed, color="red", linewidth=2,
                label=f"Rolling avg (w={window})")

    ax.axhline(y=16, color="green", linestyle=":", alpha=0.5, label="Max diversity (16 envs)")
    ax.set_xlabel("Training Step (millions)", fontsize=12)
    ax.set_ylabel("Unique Mask Configs Across 16 Envs", fontsize=12)
    ax.set_title("Mask Diversity Across Parallel Environments", fontsize=14)
    ax.set_ylim(0, 17)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_env_diversity.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 5] Saved: mask_env_diversity.png")


def fig6_action_group_forbidden(entries, figures_dir):
    """Bar chart: how often each action group is COMPLETELY forbidden."""
    group_forbidden_counts = {g: 0 for g in ACTION_GROUPS}
    for e in entries:
        for group_name, actions in ACTION_GROUPS.items():
            if all(a in e["masked"] for a in actions):
                group_forbidden_counts[group_name] += 1

    total = len(entries)
    groups = list(ACTION_GROUPS.keys())
    fractions = [group_forbidden_counts[g] / total for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    bars = ax.bar(groups, [f * 100 for f in fractions], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("% of Mask Updates Where Group Entirely Forbidden", fontsize=11)
    ax.set_title("How Often Entire Action Groups Are Forbidden by LLM", fontsize=13)
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{frac:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(f * 100 for f in fractions) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "mask_group_forbidden.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 6] Saved: mask_group_forbidden.png")


def print_summary(entries, n_allowed):
    print("\n" + "=" * 70)
    print("MASK STATISTICS SUMMARY")
    print("=" * 70)
    print(f"Total mask update events parsed: {len(entries):,}")
    print(f"Unique roles assigned: {len(set(e['role'] for e in entries))}")
    print()
    print("Allowed actions per mask update:")
    print(f"  Mean:   {np.mean(n_allowed):.2f}")
    print(f"  Median: {np.median(n_allowed):.0f}")
    print(f"  Std:    {np.std(n_allowed):.2f}")
    print(f"  Min:    {np.min(n_allowed)}")
    print(f"  Max:    {np.max(n_allowed)}")
    print()
    buckets = [
        ("= 0 (all forbidden -> forced Idle)", lambda n: n == 0),
        ("= 1 (deadlock)", lambda n: n == 1),
        ("<= 2", lambda n: n <= 2),
        ("<= 4", lambda n: n <= 4),
        ("<= 6", lambda n: n <= 6),
        ("> 10", lambda n: n > 10),
        ("= 19 (no masking)", lambda n: n == 19),
    ]
    for label, cond in buckets:
        count = sum(1 for n in n_allowed if cond(n))
        print(f"  {label}: {count:,} ({count / len(n_allowed) * 100:.1f}%)")

    print()
    all_movement = list(range(1, 9))
    movement_fully_blocked = sum(
        1 for e in entries if all(a in e["masked"] for a in all_movement)
    )
    print(f"Movement (actions 1-8) fully blocked: "
          f"{movement_fully_blocked:,} ({movement_fully_blocked / len(entries) * 100:.1f}%)")

    passing_blocked = sum(
        1 for e in entries if all(a in e["masked"] for a in [9, 10, 11])
    )
    print(f"Passing (9-11) fully blocked: "
          f"{passing_blocked:,} ({passing_blocked / len(entries) * 100:.1f}%)")

    shot_blocked = sum(1 for e in entries if 12 in e["masked"])
    print(f"Shot (12) blocked: "
          f"{shot_blocked:,} ({shot_blocked / len(entries) * 100:.1f}%)")


def make_output_dir(output_dir=None):
    if output_dir:
        p = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = REPO_ROOT / "outputs" / f"analysis_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser(description="Mask statistics analysis.")
    parser.add_argument("--masking-log-dir", type=str, required=True,
                        help="Directory containing masking training log files (e.g. outputs/my_run/logs)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures (default: outputs/analysis_<timestamp>)")
    args = parser.parse_args()

    figures_dir = make_output_dir(args.output_dir)
    print(f"Output directory: {figures_dir}")

    masking_log_dir = Path(args.masking_log_dir)
    if not masking_log_dir.exists():
        print(f"ERROR: Log directory not found: {masking_log_dir}")
        sys.exit(1)

    log_files = sorted(masking_log_dir.glob("*.log"))
    if not log_files:
        print(f"ERROR: No *.log files found in {masking_log_dir}")
        sys.exit(1)

    print("Parsing masking training logs...")
    all_entries = []
    for f in log_files:
        entries = parse_coach_lines(f, step_offset=0)
        print(f"  {f.name}: {len(entries):,} coach entries")
        all_entries.extend(entries)

    if not all_entries:
        print("ERROR: No coach log entries found.")
        sys.exit(1)

    print(f"  Combined: {len(all_entries):,} total coach entries")
    print()

    n_allowed = fig1_allowed_actions_histogram(all_entries, figures_dir)
    fig2_per_action_forbidden_heatmap(all_entries, figures_dir)
    fig3_role_distribution(all_entries, figures_dir)
    fig4_allowed_actions_timeseries(all_entries, figures_dir)
    fig5_per_env_diversity(all_entries, figures_dir)
    fig6_action_group_forbidden(all_entries, figures_dir)

    print_summary(all_entries, n_allowed)
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
