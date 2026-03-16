"""
Learning Curve Comparison — extract eval win rates from training logs of two runs.

No GRF dependency required.

Usage:
    python analysis/learning_curves.py \\
        --run-a-log-dir outputs/baseline/logs \\
        --run-b-log-dir outputs/masking/logs \\
        --label-a "Baseline PPO" --label-b "MaskablePPO + LLM"
    python analysis/learning_curves.py --output-dir outputs/analysis_custom
"""

import re
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_PATTERN = re.compile(
    r'\[Eval\] step=(\d+) win_rate=([0-9.]+)%'
)
# SB3 training table metrics, e.g.:
# |    ep_len_mean          | 71.8        |
# |    ep_rew_mean          | 0.01        |
# |    total_timesteps      | 131072      |
TOTAL_STEPS_PATTERN = re.compile(r'total_timesteps\s*\|\s*([0-9]+)')
EP_LEN_PATTERN = re.compile(r'ep_len_mean\s*\|\s*([0-9.]+)')
EP_REW_PATTERN = re.compile(r'ep_rew_mean\s*\|\s*([0-9.+-eE]+)')


def parse_eval_lines(log_path, step_offset=0):
    """Extract (absolute_step, win_rate) tuples from a training log."""
    results = []
    if not log_path.exists():
        print(f"  WARNING: {log_path} not found, skipping.")
        return results
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EVAL_PATTERN.search(line)
            if m:
                step = int(m.group(1)) + step_offset
                win_rate = float(m.group(2)) / 100.0
                results.append((step, win_rate))
    return results


def load_all_eval_data(run_a_log_dir, run_b_log_dir):
    """Load eval data by scanning all *.log files in each log directory."""
    print("Loading evaluation data from training logs...")

    def scan_dir(log_dir, label):
        data = []
        log_dir = Path(log_dir)
        if not log_dir.exists():
            print(f"  WARNING: {log_dir} not found")
            return data
        log_files = sorted(log_dir.glob("*.log"))
        if not log_files:
            print(f"  WARNING: no *.log files in {log_dir}")
            return data
        for f in log_files:
            d = parse_eval_lines(f, step_offset=0)
            print(f"  {label} ({f.name}): {len(d)} eval points")
            data.extend(d)
        return data

    run_a_data = scan_dir(run_a_log_dir, "Run A")
    run_b_data = scan_dir(run_b_log_dir, "Run B")

    # De-duplicate and sort by step
    run_a_data = sorted(set(run_a_data), key=lambda x: x[0])
    run_b_data = sorted(set(run_b_data), key=lambda x: x[0])

    return run_a_data, run_b_data


def parse_episode_stats(log_path):
    """
    Extract (total_timesteps, ep_len_mean) and (total_timesteps, ep_rew_mean)
    from SB3 training tables in a log file.
    """
    len_data = []
    rew_data = []
    if not log_path.exists():
        return len_data, rew_data

    current_step = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_step = TOTAL_STEPS_PATTERN.search(line)
            if m_step:
                current_step = int(m_step.group(1))
                continue
            if current_step is None:
                continue
            m_len = EP_LEN_PATTERN.search(line)
            if m_len:
                try:
                    ep_len = float(m_len.group(1))
                    len_data.append((current_step, ep_len))
                except ValueError:
                    pass
                continue
            m_rew = EP_REW_PATTERN.search(line)
            if m_rew:
                try:
                    ep_rew = float(m_rew.group(1))
                    rew_data.append((current_step, ep_rew))
                except ValueError:
                    pass
                continue

    return len_data, rew_data


def load_all_episode_stats(run_a_log_dir, run_b_log_dir):
    """Scan logs and collect episode length / reward stats for both runs."""
    def scan_dir(log_dir):
        log_dir = Path(log_dir)
        all_len = []
        all_rew = []
        if not log_dir.exists():
            return all_len, all_rew
        for f in sorted(log_dir.glob("*.log")):
            l, r = parse_episode_stats(f)
            all_len.extend(l)
            all_rew.extend(r)
        # sort and deduplicate
        all_len = sorted(set(all_len), key=lambda x: x[0])
        all_rew = sorted(set(all_rew), key=lambda x: x[0])
        return all_len, all_rew

    a_len, a_rew = scan_dir(run_a_log_dir)
    b_len, b_rew = scan_dir(run_b_log_dir)
    return a_len, a_rew, b_len, b_rew


def fig1_learning_curves(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """Side-by-side win rate learning curves."""
    fig, ax = plt.subplots(figsize=(13, 6))

    a_steps = [s / 1e6 for s, _ in run_a_data]
    a_wr = [w * 100 for _, w in run_a_data]
    b_steps = [s / 1e6 for s, _ in run_b_data]
    b_wr = [w * 100 for _, w in run_b_data]

    ax.plot(a_steps, a_wr, color="#2196F3", linewidth=1.2, alpha=0.4, label="_nolegend_")
    ax.plot(b_steps, b_wr, color="#F44336", linewidth=1.2, alpha=0.4, label="_nolegend_")

    # Smoothed curves
    def smooth(values, window=7):
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid").tolist()

    if len(a_wr) > 10:
        w = 7
        a_smooth = smooth(a_wr, w)
        ax.plot(a_steps[w // 2: w // 2 + len(a_smooth)], a_smooth,
                color="#1565C0", linewidth=2.5, label=label_a)
    else:
        ax.plot(a_steps, a_wr, color="#1565C0", linewidth=2.5, label=label_a)

    if len(b_wr) > 10:
        w = 7
        b_smooth = smooth(b_wr, w)
        ax.plot(b_steps[w // 2: w // 2 + len(b_smooth)], b_smooth,
                color="#C62828", linewidth=2.5, label=label_b)
    else:
        ax.plot(b_steps, b_wr, color="#C62828", linewidth=2.5, label=label_b)

    ax.axhline(y=95, color="green", linestyle="--", alpha=0.6, label="Early stop target (95%)")
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.4, label="50% win rate")

    ax.set_xlabel("Training Steps (millions)", fontsize=13)
    ax.set_ylabel("Evaluation Win Rate (%)", fontsize=13)
    ax.set_title(f"Learning Curve Comparison: {label_a} vs. {label_b}", fontsize=14)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)

    # Annotate key differences
    if len(run_a_data) > 0:
        peak_a = max(run_a_data, key=lambda x: x[1])
        ax.annotate(f"{label_a} peak: {peak_a[1]*100:.0f}%",
                    xy=(peak_a[0] / 1e6, peak_a[1] * 100),
                    xytext=(peak_a[0] / 1e6 - 1.5, peak_a[1] * 100 + 8),
                    arrowprops=dict(arrowstyle="->", color="#1565C0"),
                    fontsize=10, color="#1565C0")
    if len(run_b_data) > 0:
        peak_b = max(run_b_data, key=lambda x: x[1])
        ax.annotate(f"{label_b} peak: {peak_b[1]*100:.0f}%",
                    xy=(peak_b[0] / 1e6, peak_b[1] * 100),
                    xytext=(peak_b[0] / 1e6 + 0.5, peak_b[1] * 100 + 10),
                    arrowprops=dict(arrowstyle="->", color="#C62828"),
                    fontsize=10, color="#C62828")

    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curves_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n[Fig 1] Saved: learning_curves_comparison.png")


def fig2_performance_gap(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """Performance gap over time at matched training steps."""
    all_steps = sorted(set(s for s, _ in run_a_data) | set(s for s, _ in run_b_data))
    common_steps = []
    a_at = []
    b_at = []

    a_sorted = sorted(run_a_data)
    b_sorted = sorted(run_b_data)

    for step in all_steps:
        aval = _interp_at(a_sorted, step)
        bval = _interp_at(b_sorted, step)
        if aval is not None and bval is not None:
            common_steps.append(step)
            a_at.append(aval)
            b_at.append(bval)

    if len(common_steps) < 2:
        print("[Fig 2] Not enough overlapping data to plot performance gap.")
        return

    gap = [a - b for a, b in zip(a_at, b_at)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[2, 1], sharex=True)

    x = [s / 1e6 for s in common_steps]
    ax1.plot(x, [v * 100 for v in a_at], color="#1565C0", linewidth=2, label=label_a)
    ax1.plot(x, [v * 100 for v in b_at], color="#C62828", linewidth=2, label=label_b)
    ax1.fill_between(x, [v * 100 for v in b_at], [v * 100 for v in a_at],
                     alpha=0.15, color="gray", label="Performance gap")
    ax1.set_ylabel("Win Rate (%)", fontsize=12)
    ax1.set_title(f"Performance Gap: {label_a} vs. {label_b}", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)

    ax2.bar(x, [g * 100 for g in gap], width=(x[1] - x[0]) * 0.8 if len(x) > 1 else 0.05,
            color=["#4CAF50" if g > 0 else "#F44336" for g in gap], alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Training Steps (millions)", fontsize=12)
    ax2.set_ylabel(f"Gap ({label_a} - {label_b}) %", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "performance_gap.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 2] Saved: performance_gap.png")


def fig3_convergence_speed(run_a_data, run_b_data, label_a, label_b, figures_dir):
    """Time-to-threshold comparison."""
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    def steps_to_threshold(data, thresh):
        for step, wr in sorted(data):
            if wr >= thresh:
                return step
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    a_steps_at = []
    b_steps_at = []
    valid_thresholds = []

    for t in thresholds:
        as_ = steps_to_threshold(run_a_data, t)
        bs = steps_to_threshold(run_b_data, t)
        if as_ is not None or bs is not None:
            valid_thresholds.append(t)
            a_steps_at.append(as_ / 1e6 if as_ else None)
            b_steps_at.append(bs / 1e6 if bs else None)

    x = np.arange(len(valid_thresholds))
    width = 0.35

    a_vals = [v if v is not None else 0 for v in a_steps_at]
    b_vals = [v if v is not None else 0 for v in b_steps_at]

    ax.bar(x - width / 2, a_vals, width, label=label_a,
           color="#1565C0", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, b_vals, width, label=label_b,
           color="#C62828", edgecolor="black", alpha=0.85)

    for i, (av, bv) in enumerate(zip(a_steps_at, b_steps_at)):
        if av is None:
            ax.text(i - width / 2, 0.1, "N/A", ha="center", fontsize=8, color="gray")
        if bv is None:
            ax.text(i + width / 2, 0.1, "Never", ha="center", fontsize=8, color="red",
                    fontweight="bold")

    ax.set_xlabel("Win Rate Threshold", fontsize=12)
    ax.set_ylabel("Training Steps to Reach Threshold (millions)", fontsize=12)
    ax.set_title("Convergence Speed: Steps Required to Reach Win Rate Thresholds", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0%}" for t in valid_thresholds], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "convergence_speed.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 3] Saved: convergence_speed.png")


def fig4_episode_length(run_a_len, run_b_len, label_a, label_b, figures_dir):
    """Episode length over time (helps diagnose reward hacking / exploration)."""
    if not run_a_len and not run_b_len:
        print("[Fig 4] No episode length data found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    if run_a_len:
        a_steps = [s / 1e6 for s, _ in run_a_len]
        a_len = [v for _, v in run_a_len]
        ax.plot(a_steps, a_len, color="#1565C0", linewidth=2, label=f"{label_a} ep_len_mean")

    if run_b_len:
        b_steps = [s / 1e6 for s, _ in run_b_len]
        b_len = [v for _, v in run_b_len]
        ax.plot(b_steps, b_len, color="#C62828", linewidth=2, label=f"{label_b} ep_len_mean")

    ax.set_xlabel("Training Steps (millions)", fontsize=12)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Episode Length Over Training", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(figures_dir / "episode_length.png", dpi=150)
    plt.close(fig)
    print(f"[Fig 4] Saved: episode_length.png")


def _interp_at(sorted_data, step):
    """Linear interpolation of win rate at a given step."""
    if not sorted_data:
        return None
    if step < sorted_data[0][0] or step > sorted_data[-1][0]:
        return None
    for i in range(len(sorted_data) - 1):
        s0, w0 = sorted_data[i]
        s1, w1 = sorted_data[i + 1]
        if s0 <= step <= s1:
            if s1 == s0:
                return w0
            t = (step - s0) / (s1 - s0)
            return w0 + t * (w1 - w0)
    return sorted_data[-1][1]


def print_summary(run_a_data, run_b_data, label_a, label_b):
    print("\n" + "=" * 70)
    print("LEARNING CURVE SUMMARY")
    print("=" * 70)

    for name, data in [(label_a, run_a_data), (label_b, run_b_data)]:
        if not data:
            print(f"\n{name}: No data")
            continue
        steps = [s for s, _ in data]
        wrs = [w for _, w in data]
        peak_wr = max(wrs)
        peak_step = steps[wrs.index(peak_wr)]
        final_wr = wrs[-1]
        final_step = steps[-1]

        print(f"\n{name}:")
        print(f"  Eval points: {len(data)}")
        print(f"  Step range:  {min(steps)/1e6:.2f}M - {max(steps)/1e6:.2f}M")
        print(f"  Peak win rate: {peak_wr:.1%} at step {peak_step/1e6:.2f}M")
        print(f"  Final win rate: {final_wr:.1%} at step {final_step/1e6:.2f}M")
        print(f"  Mean win rate (last 20 evals): {np.mean(wrs[-20:]):.1%}")

    if run_a_data and run_b_data:
        a_mean_last20 = np.mean([w for _, w in run_a_data[-20:]])
        b_mean_last20 = np.mean([w for _, w in run_b_data[-20:]])
        print(f"\n  Performance gap (last 20 evals): {(a_mean_last20 - b_mean_last20)*100:.1f}pp")


def make_output_dir(output_dir=None):
    if output_dir:
        p = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = REPO_ROOT / "outputs" / f"analysis_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser(description="Learning curve comparison.")
    parser.add_argument("--run-a-log-dir", type=str, required=True,
                        help="Directory containing run A training log files (e.g. outputs/my_run/logs)")
    parser.add_argument("--run-b-log-dir", type=str, required=True,
                        help="Directory containing run B training log files (e.g. outputs/my_run/logs)")
    parser.add_argument("--label-a", type=str, default="Run A",
                        help="Display label for run A (default: 'Run A')")
    parser.add_argument("--label-b", type=str, default="Run B",
                        help="Display label for run B (default: 'Run B')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output figures (default: outputs/analysis_<timestamp>)")
    args = parser.parse_args()

    figures_dir = make_output_dir(args.output_dir)
    print(f"Output directory: {figures_dir}")

    run_a_data, run_b_data = load_all_eval_data(args.run_a_log_dir, args.run_b_log_dir)
    a_len, a_rew, b_len, b_rew = load_all_episode_stats(args.run_a_log_dir, args.run_b_log_dir)

    if not run_a_data and not run_b_data:
        print("WARNING: No [Eval] lines found; win-rate plots will be empty.")
    else:
        fig1_learning_curves(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
        fig2_performance_gap(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
        fig3_convergence_speed(run_a_data, run_b_data, args.label_a, args.label_b, figures_dir)
        print_summary(run_a_data, run_b_data, args.label_a, args.label_b)

    # Always try to plot episode length; often more informative when win rate is flat.
    fig4_episode_length(a_len, b_len, args.label_a, args.label_b, figures_dir)
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
