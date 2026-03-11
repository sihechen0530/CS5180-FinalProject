#!/usr/bin/env python3
"""
Generate LLM-based reward functions for GRF training.

Usage:
  # Interactive mode (prompts for task description)
  python scripts/generate_reward.py

  # Direct task description with scenario context
  python scripts/generate_reward.py --task "Encourage shooting when near goal" --scenario 3v1

  # Save to snippet file (preserves YAML comments in config)
  python scripts/generate_reward.py --task "..." --snippet rewards/my_reward.py

  # Save to config file (overwrites, loses comments)
  python scripts/generate_reward.py --task "..." --output configs/llm/reward/my_config.yaml

  # Use Groq instead of DeepSeek
  python scripts/generate_reward.py --provider groq --task "..."

Environment variables:
  DEEPSEEK_API_KEY: Required for DeepSeek provider (default)
  GROQ_API_KEY: Required for Groq provider
"""

import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Direct import to avoid wrappers.py numpy dependency
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "reward_generator",
    os.path.join(REPO_ROOT, "core", "reward_generator.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
RewardGenerator = _module.RewardGenerator
get_available_scenarios = _module.get_available_scenarios


def load_existing_config(path: str) -> dict:
    """Load existing YAML config if it exists."""
    import yaml
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(path: str, code: str, task: str, scenario: str = None, base_config: dict = None):
    """Save or update a YAML config with the generated reward formula. (Note: loses YAML comments)"""
    import yaml

    config = base_config or {}

    if "reward_wrapper" not in config:
        config["reward_wrapper"] = {}

    config["reward_wrapper"]["use_custom"] = False
    config["reward_wrapper"]["use_llm_reward"] = True
    config["reward_wrapper"]["llm_reward_formula"] = code
    config["reward_wrapper"]["_task_description"] = task or "(default comprehensive reward)"
    if scenario:
        config["reward_wrapper"]["_scenario"] = scenario

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Config saved to: {path}")
    print("  (Note: YAML comments in original file are not preserved)")


def save_snippet(path: str, code: str, task: str = None, scenario: str = None):
    """Save reward function to a standalone Python file."""
    task_str = task or "(default comprehensive reward)"
    header = f'''"""
LLM-generated reward function for GRF.

Task: {task_str}
Scenario: {scenario or "not specified"}

Usage in config YAML:
  reward_wrapper:
    use_llm_reward: true
    llm_reward_file: {path}
"""

'''
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(header + code + "\n")

    print(f"Snippet saved to: {path}")
    print("  To use: set 'reward_wrapper.llm_reward_file' in your config YAML")


def main():
    available_scenarios = get_available_scenarios()

    parser = argparse.ArgumentParser(
        description="Generate LLM-based reward functions for GRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task description for reward generation. If not provided, enters interactive mode.",
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=None,
        help=f"Scenario name for environment context. Available: {', '.join(available_scenarios)}",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="deepseek",
        choices=["deepseek", "groq"],
        help="LLM provider to use (default: deepseek)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name override (default: provider's best model)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output YAML config path. WARNING: overwrites file, loses comments.",
    )
    parser.add_argument(
        "--snippet",
        type=str,
        default=None,
        help="Output path for standalone .py snippet file (recommended, preserves config comments).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2, lower = more deterministic)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the generated code, don't save to file",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for name in available_scenarios:
            print(f"  - {name}")
        sys.exit(0)

    # Get API key
    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("Error: DEEPSEEK_API_KEY environment variable not set")
            print("  export DEEPSEEK_API_KEY='your-api-key'")
            sys.exit(1)
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY environment variable not set")
            print("  export GROQ_API_KEY='your-api-key'")
            sys.exit(1)

    # Get scenario and optional task
    scenario = args.scenario
    task = args.task  # Can be None

    # Interactive mode if no scenario provided
    if not scenario and not args.print_only:
        print("=" * 60)
        print("LLM Reward Function Generator")
        print("=" * 60)
        print(f"\nAvailable scenarios: {', '.join(available_scenarios)}")
        scenario = input("Scenario (required): ").strip()
        if not scenario:
            print("Scenario is required. Exiting.")
            sys.exit(1)

        print("\n(Optional) Additional emphasis for the reward function.")
        print("Press Enter to use default comprehensive reward, or specify emphasis:")
        print("  - 'Extra emphasis on passing'")
        print("  - 'Higher weight on shooting opportunities'")
        print()
        task = input("Additional emphasis (or Enter for default): ").strip() or None

    if not scenario:
        print("Error: --scenario is required")
        print("  Example: --scenario 3v1")
        sys.exit(1)

    print(f"\nProvider: {args.provider}")
    print(f"Scenario: {scenario}")
    if task:
        print(f"Additional emphasis: {task}")
    else:
        print("Task: (default comprehensive reward)")
    print("\nGenerating reward function...")

    # Generate
    generator = RewardGenerator(api_key=api_key, provider=args.provider, model=args.model)
    result = generator.generate(task, scenario=scenario, temperature=args.temperature)

    # Display results
    print("\n" + "=" * 60)
    if result["error"]:
        print(f"ERROR: {result['error']}")
        if result["code"]:
            print("\nGenerated code (with errors):")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)
        sys.exit(1)

    print("SUCCESS! Generated reward function:")
    print("-" * 40)
    print(result["code"])
    print("-" * 40)

    if result["usage"]:
        print(f"\nTokens used: {result['usage'].get('total_tokens', 'N/A')}")

    # Save to file
    if args.print_only:
        pass
    elif args.snippet:
        save_snippet(args.snippet, result["code"], task, scenario)
    elif args.output:
        base_config = load_existing_config(args.output)
        save_config(args.output, result["code"], task, scenario, base_config)
    else:
        print("\nTo use this reward function:")
        print("Option 1 (recommended): Save as snippet file")
        print("  --snippet rewards/my_reward.py")
        print("  Then in config: reward_wrapper.llm_reward_file: rewards/my_reward.py")
        print("\nOption 2: Embed in config (loses YAML comments)")
        print("  --output configs/llm/reward/my_config.yaml")

    # Interactive: test with mock data
    if not args.task and result["function"]:
        print("\n" + "=" * 60)
        test = input("Test with mock data? [y/N]: ").strip().lower()
        if test == "y":
            mock_features = {
                "ball_x": 0.2, "ball_y": 0.1, "ball_z": 0.0,
                "ball_owned_team": 0, "ball_dist_to_goal": 0.9,
                "teammate_proximity_to_ball": 0.5, "active_player_idx": 3,
                "left_team_xs": [0.0] * 11, "left_team_ys": [0.0] * 11,
                "right_team_xs": [0.5] * 11, "right_team_ys": [0.0] * 11,
            }
            mock_prev = mock_features.copy()
            mock_prev["ball_x"] = 0.15
            mock_prev["ball_dist_to_goal"] = 0.95

            r = result["function"](0.0, mock_features, mock_prev)
            print(f"\nMock test result:")
            print(f"  env_reward=0.0, ball moved from x=0.15 to x=0.2")
            print(f"  Computed reward: {r:.4f}")

            r_goal = result["function"](1.0, mock_features, mock_prev)
            print(f"\nWith goal scored (env_reward=1.0):")
            print(f"  Computed reward: {r_goal:.4f}")


if __name__ == "__main__":
    main()
