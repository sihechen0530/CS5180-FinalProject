"""
Render Comparison — run both baseline and masking agents, find episodes where
baseline wins but masking fails, render videos, and log the full LLM coach
prompts + mask decisions for manual inspection.

Requires: gfootball, sb3_contrib, stable_baselines3, torch, numpy

Usage:
    python analysis/render_comparison.py --steps 1000000 3000000 5000000
    python analysis/render_comparison.py --steps 5000000 --n-episodes 10 --render-top 3
    python analysis/render_comparison.py --steps 5000000 --coach-interval 10
    python analysis/render_comparison.py --output-dir outputs/analysis_custom

Output per checkpoint step:
    <output_dir>/renders/<step>/baseline_win_ep<N>/           - GRF video files
    <output_dir>/renders/<step>/masking_loss_ep<N>/           - GRF video files
    <output_dir>/renders/<step>/masking_loss_ep<N>_coach.txt  - full coach prompt log
    <output_dir>/renders/<step>/summary.txt                   - episode outcomes
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
from textwrap import indent

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

ACTION_NAMES = {
    0: "Idle", 1: "Left", 2: "TopLeft", 3: "Top", 4: "TopRight",
    5: "Right", 6: "BottomRight", 7: "Bottom", 8: "BottomLeft",
    9: "LongPass", 10: "HighPass", 11: "ShortPass", 12: "Shot",
    13: "Sprint", 14: "ReleaseDirection", 15: "ReleaseSprint",
    16: "Sliding", 17: "Dribble", 18: "ReleaseDribble",
}

def get_action_logits(model, obs):
    """Get raw (unmasked) action logits from the policy network, compatible across SB3 versions."""
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


def create_env(write_video=False, logdir=None):
    import gfootball.env as football_env
    from core.wrappers import ActionMaskWrapper
    kwargs = dict(
        env_name="academy_3_vs_1_with_keeper",
        stacked=False,
        representation="simple115",
        rewards="scoring",
        render=write_video,
        write_video=write_video,
        write_full_episode_dumps=write_video,
        write_goal_dumps=False,
    )
    if logdir:
        kwargs["logdir"] = str(logdir)
    env = football_env.create_environment(**kwargs)
    env = ActionMaskWrapper(env)
    return env


def load_model(path, env):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    return MaskablePPO.load(
        str(path), env=env,
        custom_objects={"policy_class": MaskableActorCriticPolicy},
    )


def run_screening_episodes(model, n_episodes):
    """Run episodes without video to classify wins/losses. Returns list of outcomes."""
    env = create_env(write_video=False)
    outcomes = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        ep_len = 0
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            total_reward += reward
            ep_len += 1
        outcomes.append({"reward": total_reward, "win": total_reward > 0, "length": ep_len})
    env.close()
    return outcomes


def render_baseline_episode(model, logdir):
    """Render one baseline episode with video. Returns reward."""
    logdir.mkdir(parents=True, exist_ok=True)
    env = create_env(write_video=True, logdir=logdir)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    env.close()
    return total_reward


def render_masking_episode(model, coach, narrator, logdir, coach_interval=10):
    """
    Render one masking episode with video + full coach prompt log.
    Returns (reward, log_text).
    """
    from core.wrappers import observation_to_features
    from core.coach import COACH_SYSTEM_PROMPT, COACH_USER_PROMPT_TEMPLATE

    logdir.mkdir(parents=True, exist_ok=True)
    env = create_env(write_video=True, logdir=logdir)

    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    current_mask = np.ones(19, dtype=bool)
    current_role = "None (not yet assigned)"

    log_lines = []
    action_trace = []

    while not done:
        features = observation_to_features(obs)

        if step % coach_interval == 0:
            narrative = narrator.translate(features)
            # Replace escaped newlines for readable display
            narrative_display = narrative.replace("\\n", "\n")
            advice = coach.get_coaching_advice(features)
            forbidden = advice.get("Forbidden_Actions", [])
            current_role = advice.get("Role", "Unknown")

            current_mask = np.ones(19, dtype=bool)
            for a in forbidden:
                if 0 <= a < 19:
                    current_mask[a] = False
            if not np.any(current_mask):
                current_mask[0] = True

            allowed_ids = [i for i in range(19) if current_mask[i]]
            forbidden_ids = [i for i in range(19) if not current_mask[i]]

            user_prompt = COACH_USER_PROMPT_TEMPLATE.format(
                narrative=narrative_display
            )

            next_update = step + coach_interval
            full_prompt_display = (
                "=" * 72 + "\n"
                f"COACH UPDATE at frame {step}"
                f"  (mask applies to frames {step}-{next_update - 1})\n"
                "=" * 72 + "\n"
                "\n"
                f"[GAME STATE at frame {step}]\n"
                f"  Ball:       x={features.get('ball_x', 0):.2f}, "
                f"y={features.get('ball_y', 0):.2f}\n"
                f"  Possession: "
                f"{'Our team' if features.get('ball_owned_team') == 0 else 'Opponent' if features.get('ball_owned_team') == 1 else 'Loose'}\n"
                f"  Active:     Player {features.get('active_player_idx', -1)}\n"
                f"  Ball->Goal: {features.get('ball_dist_to_goal', 0):.2f}\n"
                "\n"
                "[SYSTEM PROMPT]\n"
                f"{COACH_SYSTEM_PROMPT}\n"
                "\n"
                "[USER PROMPT]\n"
                f"{user_prompt}\n"
                "\n"
                "[COACH RESPONSE]\n"
                f"  Role: {current_role}\n"
                f"  Forbidden Actions ({len(forbidden_ids)}): "
                f"{[f'{a}-{ACTION_NAMES[a]}' for a in forbidden_ids]}\n"
                f"  Allowed Actions  ({len(allowed_ids)}):  "
                f"{[f'{a}-{ACTION_NAMES[a]}' for a in allowed_ids]}\n"
            )
            log_lines.append(full_prompt_display)

        # Get the action the policy WANTS (unmasked greedy)
        logits = get_action_logits(model, obs)
        preferred_action = int(np.argmax(logits))

        # Apply mask to get the actual action
        masked_logits = logits.copy()
        masked_logits[~current_mask] = -1e8
        actual_action = int(np.argmax(masked_logits))

        overridden = preferred_action != actual_action
        override_flag = " ** OVERRIDDEN **" if overridden else ""

        action_trace.append(
            f"  frame {step:3d}: "
            f"preferred={preferred_action:2d}-{ACTION_NAMES[preferred_action]:<13s} | "
            f"executed={actual_action:2d}-{ACTION_NAMES[actual_action]:<13s} | "
            f"allowed={int(current_mask.sum()):2d}/19 | "
            f"role={current_role}"
            f"{override_flag}"
        )

        obs, reward, done, info = env.step(actual_action)
        total_reward += reward
        step += 1

    env.close()

    header = (
        "#" * 72 + "\n"
        f"# MASKING AGENT EPISODE LOG\n"
        f"# Outcome: {'WIN' if total_reward > 0 else 'LOSS'} (reward={total_reward})\n"
        f"# Total frames: {step}  (frame N = video frame N, 0-indexed)\n"
        f"# Coach interval: every {coach_interval} frames\n"
        f"# Coach updates at frames: "
        f"{', '.join(str(i) for i in range(0, step, coach_interval))}\n"
        f"#\n"
        f"# HOW TO USE: open the .avi video in a player, seek to the frame\n"
        f"# number shown in each COACH UPDATE section below, and compare\n"
        f"# the visual game state with the narrative and mask decision.\n"
        "#" * 72 + "\n\n"
    )

    action_summary = (
        "\n" + "=" * 72 + "\n"
        "FULL ACTION TRACE  (frame = video frame number)\n"
        "=" * 72 + "\n"
        + "\n".join(action_trace) + "\n"
    )

    overrides = sum(1 for line in action_trace if "OVERRIDDEN" in line)
    override_summary = (
        "\n" + "=" * 72 + "\n"
        "OVERRIDE SUMMARY\n"
        "=" * 72 + "\n"
        f"  Total frames: {step}\n"
        f"  Actions overridden by mask: {overrides} ({overrides/max(step,1)*100:.1f}%)\n"
    )

    full_log = header + "\n".join(log_lines) + action_summary + override_summary
    return total_reward, full_log


def process_step(step, args, output_base, baseline_ckpt_dir, masking_ckpt_dir):
    """Process one checkpoint step: screen, render, and log."""
    from core.coach import GRFNarrator, GroqCoach

    print(f"\n{'='*60}")
    print(f"CHECKPOINT STEP: {step:,}")
    print(f"{'='*60}")

    b_step, b_path = find_checkpoint(baseline_ckpt_dir, args.baseline_prefix, step)
    m_step, m_path = find_checkpoint(masking_ckpt_dir, args.masking_prefix, step)

    if b_path is None:
        print(f"  WARNING: No baseline checkpoint near step {step}, skipping.")
        return
    if m_path is None:
        print(f"  WARNING: No masking checkpoint near step {step}, skipping.")
        return

    print(f"  Baseline: {b_path.name} (step {b_step:,})")
    print(f"  Masking:  {m_path.name} (step {m_step:,})")

    render_dir = output_base / "renders" / f"step_{step}"
    render_dir.mkdir(parents=True, exist_ok=True)

    dummy_env = create_env(write_video=False)
    baseline_model = load_model(b_path, dummy_env)
    masking_model = load_model(m_path, dummy_env)
    dummy_env.close()

    # Screen episodes
    print(f"\n  Screening {args.n_episodes} episodes with baseline...")
    b_outcomes = run_screening_episodes(baseline_model, args.n_episodes)
    b_wins = sum(1 for o in b_outcomes if o["win"])
    print(f"    Baseline wins: {b_wins}/{args.n_episodes}")

    print(f"  Screening {args.n_episodes} episodes with masking agent...")
    m_outcomes = run_screening_episodes(masking_model, args.n_episodes)
    m_wins = sum(1 for o in m_outcomes if o["win"])
    print(f"    Masking wins: {m_wins}/{args.n_episodes}")

    # Write summary
    summary = (
        f"Checkpoint step: {step:,}\n"
        f"Baseline: {b_path.name} (step {b_step:,})\n"
        f"Masking:  {m_path.name} (step {m_step:,})\n"
        f"\n"
        f"Screening ({args.n_episodes} episodes each):\n"
        f"  Baseline win rate: {b_wins}/{args.n_episodes} "
        f"({b_wins/args.n_episodes*100:.0f}%)\n"
        f"  Masking win rate:  {m_wins}/{args.n_episodes} "
        f"({m_wins/args.n_episodes*100:.0f}%)\n"
        f"\n"
    )

    # Render baseline wins
    n_render = min(args.render_top, max(b_wins, 1))
    print(f"\n  Rendering {n_render} baseline WIN episodes...")
    rendered_b = 0
    for attempt in range(args.n_episodes * 2):
        if rendered_b >= n_render:
            break
        vid_dir = render_dir / f"baseline_win_ep{rendered_b}"
        reward = render_baseline_episode(baseline_model, vid_dir)
        if reward > 0:
            summary += f"  Rendered baseline WIN ep{rendered_b}: reward={reward}\n"
            print(f"    ep{rendered_b}: WIN (reward={reward})")
            rendered_b += 1
        else:
            # remove empty dir if loss
            import shutil
            shutil.rmtree(vid_dir, ignore_errors=True)

    # Render masking losses with full coach log
    narrator = GRFNarrator()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    coach = GroqCoach(api_key=groq_api_key)

    n_render_m = min(args.render_top, max(args.n_episodes - m_wins, 1))
    print(f"\n  Rendering {n_render_m} masking LOSS episodes with coach log...")
    print(f"  (coach_interval={args.coach_interval})")
    rendered_m = 0
    for attempt in range(args.n_episodes * 2):
        if rendered_m >= n_render_m:
            break
        vid_dir = render_dir / f"masking_loss_ep{rendered_m}"
        reward, coach_log = render_masking_episode(
            masking_model, coach, narrator, vid_dir,
            coach_interval=args.coach_interval,
        )
        if reward <= 0:
            log_path = render_dir / f"masking_loss_ep{rendered_m}_coach.txt"
            log_path.write_text(coach_log, encoding="utf-8")
            summary += (
                f"  Rendered masking LOSS ep{rendered_m}: reward={reward} "
                f"-> {log_path.name}\n"
            )
            print(f"    ep{rendered_m}: LOSS (reward={reward}) -> {log_path.name}")
            rendered_m += 1
        else:
            import shutil
            shutil.rmtree(vid_dir, ignore_errors=True)

    # Also render a masking win for comparison if any exist
    if m_wins > 0:
        print(f"\n  Rendering 1 masking WIN episode with coach log for comparison...")
        for attempt in range(args.n_episodes * 2):
            vid_dir = render_dir / "masking_win_ep0"
            reward, coach_log = render_masking_episode(
                masking_model, coach, narrator, vid_dir,
                coach_interval=args.coach_interval,
            )
            if reward > 0:
                log_path = render_dir / "masking_win_ep0_coach.txt"
                log_path.write_text(coach_log, encoding="utf-8")
                summary += (
                    f"  Rendered masking WIN ep0: reward={reward} "
                    f"-> {log_path.name}\n"
                )
                print(f"    ep0: WIN (reward={reward}) -> {log_path.name}")
                break
            else:
                import shutil
                shutil.rmtree(vid_dir, ignore_errors=True)

    summary_path = render_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\n  Summary written to: {summary_path}")
    print(f"  All renders at:     {render_dir}")


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
        description="Render side-by-side comparison of baseline vs masking agent "
                    "with full LLM coach prompt logs."
    )
    parser.add_argument("--baseline-ckpt-dir", type=str, required=True,
                        help="Baseline checkpoint directory")
    parser.add_argument("--masking-ckpt-dir", type=str, required=True,
                        help="Masking checkpoint directory")
    parser.add_argument("--baseline-prefix", type=str, default="ppo_3v1",
                        help="Baseline checkpoint filename prefix (default: ppo_3v1)")
    parser.add_argument("--masking-prefix", type=str, default="masking_ppo_3v1",
                        help="Masking checkpoint filename prefix (default: masking_ppo_3v1)")
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[3_000_000, 5_000_000],
        help="Checkpoint steps to compare (default: 3M 5M)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Episodes for screening win/loss (default: 10)",
    )
    parser.add_argument(
        "--render-top", type=int, default=2,
        help="Number of win/loss episodes to render with video (default: 2)",
    )
    parser.add_argument(
        "--coach-interval", type=int, default=10,
        help="Steps between coach mask updates during rendering (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output renders (default: outputs/analysis_<timestamp>)",
    )
    args = parser.parse_args()

    baseline_ckpt_dir = Path(args.baseline_ckpt_dir)
    masking_ckpt_dir = Path(args.masking_ckpt_dir)

    try:
        import gfootball
    except ImportError:
        print("ERROR: gfootball not installed.")
        sys.exit(1)

    output_base = make_output_dir(args.output_dir)
    print(f"Output directory: {output_base}")

    for step in args.steps:
        process_step(step, args, output_base, baseline_ckpt_dir, masking_ckpt_dir)

    print("\n" + "=" * 60)
    print(f"DONE. Review the renders at: {output_base / 'renders'}")
    print("Each masking loss episode has a *_coach.txt file showing")
    print("the full LLM prompt and mask decisions at every coach update.")
    print("=" * 60)


if __name__ == "__main__":
    main()
