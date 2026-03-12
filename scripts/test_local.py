#!/usr/bin/env python3
"""
Local testing script - NO GRF or heavy dependencies required.

Tests:
1. RewardGenerator code validation (without API calls)
2. Reward function logic with mock features
3. Config loading and formula compilation
4. (Optional) LLM API generation

Usage:
  python scripts/test_local.py           # Run all tests
  python scripts/test_local.py --api     # Also test LLM API (requires API key)

Dependencies: Only Python stdlib + yaml (for config test)
"""

import os
import sys
import argparse
import math

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_mock_features(ball_x=0.2, ball_dist_to_goal=0.9, ball_owned_team=0):
    """Create mock features dict (no numpy needed)."""
    return {
        "ball_x": ball_x,
        "ball_y": 0.1,
        "ball_z": 0.0,
        "ball_owned_team": ball_owned_team,
        "ball_dist_to_goal": ball_dist_to_goal,
        "teammate_proximity_to_ball": 0.5,
        "active_player_idx": 3,
        "left_team_xs": [0.0] * 11,
        "left_team_ys": [0.0] * 11,
        "right_team_xs": [0.5] * 11,
        "right_team_ys": [0.0] * 11,
    }


def test_reward_function_logic():
    """Test reward function logic with mock features (no numpy/gym needed)."""
    print("\n" + "=" * 50)
    print("TEST: Reward function logic (pure Python)")
    print("=" * 50)

    # Define a test reward function (like what LLM would generate)
    def test_reward_fn(env_reward, features, prev_features):
        delta_dist = prev_features["ball_dist_to_goal"] - features["ball_dist_to_goal"]
        return float(env_reward) + 0.05 * delta_dist

    # Test 1: ball moved toward goal
    prev = make_mock_features(ball_x=0.15, ball_dist_to_goal=0.95)
    curr = make_mock_features(ball_x=0.25, ball_dist_to_goal=0.85)
    reward = test_reward_fn(0.0, curr, prev)
    print(f"  Ball moved forward: reward = {reward:.4f} (expected > 0)")
    assert reward > 0, "Expected positive reward"

    # Test 2: goal scored
    reward_goal = test_reward_fn(1.0, curr, prev)
    print(f"  Goal scored: reward = {reward_goal:.4f} (expected > 1)")
    assert reward_goal > 1, "Expected reward > 1"

    # Test 3: first step (prev = curr, delta = 0)
    reward_first = test_reward_fn(0.0, curr, curr)
    print(f"  First step (delta=0): reward = {reward_first:.4f} (expected = 0)")
    assert abs(reward_first) < 0.001, "Expected ~0 reward"

    # Test 4: ball moved backward
    prev_back = make_mock_features(ball_dist_to_goal=0.80)
    curr_back = make_mock_features(ball_dist_to_goal=0.90)
    reward_back = test_reward_fn(0.0, curr_back, prev_back)
    print(f"  Ball moved backward: reward = {reward_back:.4f} (expected < 0)")
    assert reward_back < 0, "Expected negative reward"

    print("  ✓ PASSED")
    return True


def _load_reward_generator_module():
    """Load reward_generator module directly (avoids wrappers.py numpy dependency)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reward_generator",
        os.path.join(REPO_ROOT, "core", "reward_generator.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_validation():
    """Test the code validation logic in RewardGenerator."""
    print("\n" + "=" * 50)
    print("TEST: Code validation (RewardGenerator._compile_and_validate)")
    print("=" * 50)

    module = _load_reward_generator_module()
    RewardGenerator = module.RewardGenerator

    # Create a dummy generator (won't call API)
    class MockGenerator(RewardGenerator):
        def __init__(self):
            self.provider = "mock"
            self.api_key = "mock"
            self.client = None
            self.model = "mock"

    gen = MockGenerator()

    # Test 1: Valid code
    valid_code = """
def llm_reward_formula(env_reward, features, prev_features):
    delta = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    return float(env_reward) + 0.05 * delta
"""
    func, error = gen._compile_and_validate(valid_code.strip())
    print(f"  Valid code: {'✓ compiled' if func else '✗ failed'}")
    if error:
        print(f"    Error: {error}")
    assert func is not None, f"Valid code should compile: {error}"

    # Test 2: Wrong function name
    wrong_name = """
def my_reward(env_reward, features, prev_features):
    return float(env_reward)
"""
    func, error = gen._compile_and_validate(wrong_name.strip())
    print(f"  Wrong name: {'✗ rejected (good)' if error else '✓ accepted (bad)'}")
    assert error is not None, "Should reject wrong function name"

    # Test 3: Import statement (forbidden)
    with_import = """
import math
def llm_reward_formula(env_reward, features, prev_features):
    return float(env_reward)
"""
    func, error = gen._compile_and_validate(with_import.strip())
    print(f"  With import: {'✗ rejected (good)' if error else '✓ accepted (bad)'}")
    assert error is not None, "Should reject code with imports"

    # Test 4: Wrong number of parameters
    wrong_params = """
def llm_reward_formula(env_reward, features):
    return float(env_reward)
"""
    func, error = gen._compile_and_validate(wrong_params.strip())
    print(f"  Wrong params: {'✗ rejected (good)' if error else '✓ accepted (bad)'}")
    assert error is not None, "Should reject wrong parameter count"

    print("  ✓ PASSED")
    return True


def test_config_loading():
    """Test loading and compiling reward formula from config."""
    print("\n" + "=" * 50)
    print("TEST: Config loading and formula compilation")
    print("=" * 50)

    try:
        import yaml
    except ImportError:
        print("  yaml not installed (pip install pyyaml)")
        print("  ⚠ SKIPPED")
        return True

    config_path = os.path.join(REPO_ROOT, "configs/llm/reward/3v1_config.yaml")
    if not os.path.exists(config_path):
        print(f"  Config not found: {config_path}")
        print("  ⚠ SKIPPED")
        return True

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    reward_config = config.get("reward_wrapper", {})
    formula_str = reward_config.get("llm_reward_formula", "")

    print(f"  Config loaded: {config_path}")
    print(f"  use_llm_reward: {reward_config.get('use_llm_reward')}")

    if formula_str:
        local_env = {}
        try:
            exec(formula_str, {"__builtins__": __builtins__}, local_env)
            func = local_env.get("llm_reward_formula")
            if func:
                # Test with mock data
                mock_features = make_mock_features(ball_dist_to_goal=0.9)
                mock_prev = make_mock_features(ball_dist_to_goal=0.95)
                result = func(0.0, mock_features, mock_prev)
                print(f"  Formula compiled: ✓")
                print(f"  Test run result: {result:.4f}")
            else:
                print("  Formula compiled: ✗ (function not found)")
                return False
        except Exception as e:
            print(f"  Formula compiled: ✗ ({e})")
            return False
    else:
        print("  No formula in config")

    print("  ✓ PASSED")
    return True


def test_api_generation(provider="deepseek"):
    """Test actual LLM API call (optional, requires API key)."""
    print("\n" + "=" * 50)
    print(f"TEST: LLM API generation ({provider})")
    print("=" * 50)

    api_key = os.environ.get(f"{provider.upper()}_API_KEY")
    if not api_key:
        print(f"  {provider.upper()}_API_KEY not set")
        print("  ⚠ SKIPPED")
        return True

    module = _load_reward_generator_module()
    RewardGenerator = module.RewardGenerator

    gen = RewardGenerator(api_key=api_key, provider=provider)
    result = gen.generate(
        task_description="Encourage the agent to move the ball toward the goal",
        scenario="3v1",
        temperature=0.1
    )

    if result["error"]:
        print(f"  Generation failed: {result['error']}")
        return False

    print(f"  Generation: ✓")
    print(f"  Tokens used: {result['usage'].get('total_tokens', 'N/A')}")
    print(f"  Code preview:")
    for line in result["code"].split("\n")[:5]:
        print(f"    {line}")
    if len(result["code"].split("\n")) > 5:
        print("    ...")

    if result["function"]:
        mock_f = make_mock_features(ball_dist_to_goal=0.9)
        mock_p = make_mock_features(ball_dist_to_goal=0.95)
        r = result["function"](0.0, mock_f, mock_p)
        print(f"  Test execution: ✓ (result={r:.4f})")
    else:
        print("  Test execution: ✗ (function not compiled)")
        return False

    print("  ✓ PASSED")
    return True


def test_scenario_contexts():
    """Test that scenario contexts are properly defined."""
    print("\n" + "=" * 50)
    print("TEST: Scenario contexts")
    print("=" * 50)

    module = _load_reward_generator_module()
    SCENARIO_CONTEXTS = module.SCENARIO_CONTEXTS
    get_available_scenarios = module.get_available_scenarios

    scenarios = get_available_scenarios()
    print(f"  Available scenarios: {len(scenarios)}")
    for name in scenarios:
        ctx = SCENARIO_CONTEXTS.get(name, "")
        has_content = len(ctx.strip()) > 50
        print(f"    - {name}: {'✓' if has_content else '✗ empty'}")
        if not has_content:
            return False

    print("  ✓ PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Local tests without GRF")
    parser.add_argument("--api", action="store_true", help="Also test LLM API (requires API key)")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "groq"])
    args = parser.parse_args()

    print("=" * 50)
    print("LOCAL TESTS (no GRF/numpy required)")
    print("=" * 50)

    results = []

    results.append(("reward_function_logic", test_reward_function_logic()))
    results.append(("code_validation", test_code_validation()))
    results.append(("scenario_contexts", test_scenario_contexts()))
    results.append(("config_loading", test_config_loading()))

    if args.api:
        results.append(("api_generation", test_api_generation(args.provider)))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
