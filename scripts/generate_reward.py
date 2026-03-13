"""
LLM-based Reward Function Generator for Google Research Football.

This module generates dense reward functions from natural language task descriptions.
The LLM is called once to produce a Python function; the function is then executed
locally at each training step (no LLM calls during training).
"""

import re
import ast

# ---------------------------------------------------------------------------
# Scenario Contexts - environment-specific background for LLM
# ---------------------------------------------------------------------------

SCENARIO_CONTEXTS = {
    "empty_goal": """
## Scenario: Empty Goal (academy_empty_goal_close)
- Single agent vs empty goal (no goalkeeper, no opponents)
- Agent starts with the ball near the goal
- Objective: Learn to shoot and score
- Key insight: No need to consider opponents; focus on ball control and shooting accuracy
- Typical episode length: very short (< 50 steps)
- Episode ends on: goal scored, ball out of play, or max steps
""",

    "empty": """
## Scenario: Empty Goal (academy_empty_goal_close)
- Single agent vs empty goal (no goalkeeper, no opponents)
- Agent starts with the ball near the goal
- Objective: Learn to shoot and score
- Key insight: No need to consider opponents; focus on ball control and shooting accuracy
- Typical episode length: very short (< 50 steps)
- Episode ends on: goal scored, ball out of play, or max steps
""",

    "3v1": """
## Scenario: 3 vs 1 with Keeper (academy_3_vs_1_with_keeper)
- 3 agents (our team) vs 1 goalkeeper
- Our team starts with possession in the attacking third (ball_x ~ 0.5+)
- Objective: Coordinate passing and movement to beat the keeper and score
- Key insight: The keeper (right_team index 0) guards the goal; consider keeper position when shooting
- Useful features: right_team_xs[0], right_team_ys[0] give keeper position
- Typical episode length: short (< 100 steps)
- Episode ends on: goal scored, ball out of play, possession change, or max steps
- CRITICAL: Agent already starts in attacking zone with possession. Do NOT give
  constant rewards for "being in attacking zone" or "having possession" — the agent
  gets those for free and will learn to stall instead of scoring.
""",

    "academy_3_vs_1_with_keeper": """
## Scenario: 3 vs 1 with Keeper (academy_3_vs_1_with_keeper)
- 3 agents (our team) vs 1 goalkeeper
- Our team starts with possession in the attacking third (ball_x ~ 0.5+)
- Objective: Coordinate passing and movement to beat the keeper and score
- Key insight: The keeper (right_team index 0) guards the goal; consider keeper position when shooting
- Useful features: right_team_xs[0], right_team_ys[0] give keeper position
- Typical episode length: short (< 100 steps)
- Episode ends on: goal scored, ball out of play, possession change, or max steps
- CRITICAL: Agent already starts in attacking zone with possession. Do NOT give
  constant rewards for "being in attacking zone" or "having possession" — the agent
  gets those for free and will learn to stall instead of scoring.
""",

    "11v11": """
## Scenario: Full Game (11_vs_11_stochastic)
- Full 11 vs 11 match with stochastic opponent
- Complex coordination required across the entire pitch
- Objective: Score more goals than the opponent
- Key insight: Must handle both offense and defense; long episodes with varied situations
- Consider: possession transitions, defensive positioning, counter-attacks
- Typical episode length: 3000 steps (full match)
- Episode ends on: full match time
""",

    "counterattack": """
## Scenario: Counter Attack (academy_counterattack_easy/hard)
- Start from defensive position, opponent has lost the ball
- Fast transition from defense to attack
- Objective: Quickly advance and score before defense recovers
- Key insight: Speed is crucial; reward rapid ball advancement
- Typical episode length: medium (100-200 steps)
- Episode ends on: goal scored, ball out of play, possession change, or max steps
""",
}

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

REWARD_SYSTEM_PROMPT = """You are an expert reinforcement learning engineer designing dense reward functions for the Google Research Football (GRF) environment.

## Core Principle: Potential-Based Reward Shaping (PBRS)

All shaping rewards MUST be potential-based (using deltas between consecutive steps) or
event-triggered (one-time reward on state transitions). This is the single most important
rule to follow.

### WHY: The Reward Hacking Problem
A constant per-step reward (e.g., +0.01 every step you have the ball) teaches the agent
to MAXIMIZE EPISODE LENGTH instead of achieving the goal. For example:
- Per-step possession bonus → agent holds ball forever, never shoots
- Per-step "in attacking zone" bonus → agent circles near goal, never scores
- Accumulated shaping reward can EXCEED the sparse goal reward (+1), making scoring
  look like a bad deal (episode ends = no more free shaping reward)

### HOW: Correct Shaping Patterns

CORRECT (potential-based delta):
```python
delta = prev_features["ball_dist_to_goal"] - features["ball_dist_to_goal"]
reward += 0.05 * delta  # only fires when distance actually changes
```

CORRECT (event-triggered, one-time):
```python
if prev_features["ball_owned_team"] != 0 and features["ball_owned_team"] == 0:
    reward += 0.02  # one-time bonus for GAINING possession
```

CORRECT (threshold crossing, one-time):
```python
if features["ball_x"] > 0.7 and prev_features["ball_x"] <= 0.7:
    reward += 0.03  # one-time bonus for entering shooting zone
```

WRONG (constant per-step — causes reward hacking):
```python
if features["ball_owned_team"] == 0:
    reward += 0.01  # WRONG: accumulates every step, incentivizes stalling
```

WRONG (constant zone bonus):
```python
if features["ball_x"] > 0.33:
    reward += 0.01  # WRONG: free reward for staying in zone
```

## Required Components

Your reward function MUST include:

1. **Ball-to-goal progress** (potential-based delta):
   `prev_features["ball_dist_to_goal"] - features["ball_dist_to_goal"]`

2. **Possession change** (event-triggered, NOT per-step):
   Reward gaining possession, penalize losing it. Never reward holding it.

3. **Time penalty** (small negative per step):
   `-0.002` per step to discourage stalling. This is essential for short-episode
   academy scenarios where the agent can exploit shaping rewards by delaying.

The user's task description provides **additional emphasis or modifications**.
If the task says "emphasis on X", increase weights for X while keeping other components.
If the task describes a specific behavior, add it ON TOP of the required components.

## Available Features (observation dict)

The reward function receives two dicts: `features` (current step) and `prev_features` (previous step).
Both have identical structure with these keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| ball_x | float | [-1, 1] | Ball x-position. -1 = own goal, +1 = opponent goal |
| ball_y | float | [-0.42, 0.42] | Ball y-position. 0 = center |
| ball_z | float | [0, ~1] | Ball height above ground |
| ball_owned_team | int | {-1, 0, 1} | -1 = loose, 0 = our team, 1 = opponent |
| ball_dist_to_goal | float | [0, ~2.2] | Distance from ball to opponent goal center |
| teammate_proximity_to_ball | float | [0, ~2] | Mean distance of all teammates to ball |
| active_player_idx | int | [0, 10] | Index of currently controlled player |
| left_team_xs | list[float] | [-1, 1] | X-positions of all 11 teammates |
| left_team_ys | list[float] | [-0.42, 0.42] | Y-positions of all 11 teammates |
| right_team_xs | list[float] | [-1, 1] | X-positions of all 11 opponents |
| right_team_ys | list[float] | [-0.42, 0.42] | Y-positions of all 11 opponents |

Note: `prev_features` is NEVER None. On the first step, it equals `features` (so delta = 0).

## Function Signature

```python
def llm_reward_formula(env_reward: float, features: dict, prev_features: dict) -> float:
    # env_reward: sparse reward from environment (+1 for goal, 0 otherwise)
    # features: current observation dict
    # prev_features: previous observation dict (same structure, never None)
    # Returns: combined dense + sparse reward (float)
```

## Constraints (MUST follow)

1. Return type MUST be float
2. Dense shaping magnitude: keep each component in [0.01, 0.1] range per step
3. **ALL shaping MUST be potential-based (delta) or event-triggered (state transition)**
4. **NEVER use constant per-step bonuses** (if X: reward += c) — this causes reward hacking
5. DO NOT import any modules — only use built-in operations (+, -, *, /, min, max, abs, sum, len)
6. Always preserve the sparse signal: include `env_reward` in the final return
7. Use .get() for safe dict access with defaults
8. **ALWAYS include a small time penalty** (~-0.002 per step) to discourage stalling

## Output Format

Return ONLY the Python function definition. No explanation, no markdown code fences, no extra text.
Start directly with `def llm_reward_formula(` and end with `return reward`.
Keep the code concise (under 40 lines)."""


REWARD_FEW_SHOT_EXAMPLES = """
## Example 1: Ball progress + time penalty (minimal correct baseline)

Task: "Encourage the agent to move the ball toward the opponent's goal"

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    # Potential-based: reward reduction in ball-to-goal distance
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.05 * delta_dist
    # Time penalty to discourage stalling
    reward -= 0.002
    return reward
```

## Example 2: Possession events + ball progress (NOT per-step possession)

Task: "Reward maintaining possession while advancing the ball"

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    # Ball progress (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.05 * delta_dist
    # Possession CHANGE (event-triggered, NOT per-step)
    prev_own = prev_features.get("ball_owned_team", -1)
    curr_own = features.get("ball_owned_team", -1)
    if prev_own != 0 and curr_own == 0:
        reward += 0.02  # gained possession (one-time)
    elif prev_own == 0 and curr_own != 0:
        reward -= 0.03  # lost possession (one-time)
    # Time penalty
    reward -= 0.002
    return reward
```

## Example 3: Attacking with keeper avoidance (3v1 scenario)

Task: "Score against a keeper by positioning ball away from keeper"

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    # Ball-to-goal progress (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.08 * delta_dist
    # Possession change (event-triggered)
    prev_own = prev_features.get("ball_owned_team", -1)
    curr_own = features.get("ball_owned_team", -1)
    if prev_own != 0 and curr_own == 0:
        reward += 0.02
    elif prev_own == 0 and curr_own != 0:
        reward -= 0.03
    # Keeper avoidance in attacking zone (potential-based delta)
    ball_x = features.get("ball_x", 0.0)
    if ball_x > 0.5 and curr_own == 0:
        ball_y = features.get("ball_y", 0.0)
        keeper_y = features.get("right_team_ys", [0.0])[0]
        prev_ball_y = prev_features.get("ball_y", 0.0)
        prev_keeper_y = prev_features.get("right_team_ys", [0.0])[0]
        curr_dist = abs(ball_y - keeper_y)
        prev_dist = abs(prev_ball_y - prev_keeper_y)
        reward += 0.03 * (curr_dist - prev_dist)
    # Deep zone crossing (event-triggered threshold)
    prev_x = prev_features.get("ball_x", 0.0)
    if ball_x > 0.7 and prev_x <= 0.7 and curr_own == 0:
        reward += 0.03
    # Time penalty
    reward -= 0.002
    return reward
```
"""


REWARD_USER_PROMPT_TEMPLATE = """{scenario_context}{task_section}
Write a dense reward function that includes:
1. Ball-to-goal progress (potential-based delta)
2. Possession change events (NOT per-step bonus)
3. Time penalty
4. Any additional emphasis from the task above

Remember: EVERY shaping component must be a delta or event-triggered. NO constant per-step bonuses.

Follow all constraints and output ONLY the Python function."""


# ---------------------------------------------------------------------------
# RewardGenerator Class
# ---------------------------------------------------------------------------

class RewardGenerator:
    """
    Generates reward functions using LLM APIs (DeepSeek or Groq).
    """

    # Patterns that indicate per-step constant rewards (reward hacking risk)
    _HACKING_PATTERNS = [
        # pattern: (regex, description)
        (r'if\s+.*ball_owned_team.*==\s*0\s*:\s*\n\s*reward\s*\+=\s*[\d.]+',
         "Per-step possession bonus detected (should be event-triggered)"),
        (r'if\s+.*ball_x.*[>]\s*[\d.]+\s*:\s*\n\s*reward\s*\+=\s*[\d.]+(?!\s*\*)',
         "Per-step zone bonus detected (should be threshold-crossing event)"),
    ]

    def __init__(self, api_key: str, provider: str = "deepseek", model: str = None):
        """
        Args:
            api_key: API key for the LLM provider
            provider: "deepseek" or "groq"
            model: Model name (defaults to provider's best model)
        """
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider == "deepseek":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install `openai` package for DeepSeek API")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self.model = model or "deepseek-chat"
        elif self.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("Install `groq` package for Groq API")
            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.3-70b-versatile"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'deepseek' or 'groq'.")

    def generate(self, task_description: str = None, scenario: str = None,
                 temperature: float = 0.2, max_retries: int = 2) -> dict:
        """
        Generate a reward function, optionally with additional emphasis.

        Args:
            task_description: Optional additional emphasis (e.g., "Extra emphasis on passing").
                              If None, generates a default comprehensive reward.
            scenario: Optional scenario name (e.g., "3v1", "empty_goal") for context
            temperature: LLM sampling temperature (lower = more deterministic)
            max_retries: Number of retries if generated code fails validation

        Returns:
            dict with keys:
                - "code": the generated Python function as a string
                - "function": the compiled callable (or None if compilation failed)
                - "error": error message if any, else None
                - "warnings": list of non-fatal warnings
                - "usage": token usage info
                - "attempts": number of generation attempts used
        """
        # Build scenario context if provided
        scenario_context = ""
        if scenario:
            scenario_context = SCENARIO_CONTEXTS.get(
                scenario, SCENARIO_CONTEXTS.get(scenario.lower(), "")
            )
            if not scenario_context:
                available = ", ".join(SCENARIO_CONTEXTS.keys())
                print(f"Warning: Unknown scenario '{scenario}'. Available: {available}")

        # Build task section (optional)
        if task_description:
            task_section = f"Additional Emphasis: {task_description}\n\n"
        else:
            task_section = "(No additional emphasis - generate optimal comprehensive reward)\n\n"

        messages = [
            {"role": "system", "content": REWARD_SYSTEM_PROMPT + "\n" + REWARD_FEW_SHOT_EXAMPLES},
            {"role": "user", "content": REWARD_USER_PROMPT_TEMPLATE.format(
                scenario_context=scenario_context,
                task_section=task_section,
            )},
        ]

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1200,
                    temperature=temperature,
                )
                raw_code = response.choices[0].message.content.strip()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
            except Exception as e:
                return {
                    "code": None, "function": None, "error": f"API Error: {e}",
                    "warnings": [], "usage": {}, "attempts": attempt,
                }

            # Clean up code
            code = self._extract_code(raw_code)

            # Validate and compile
            func, error = self._compile_and_validate(code)

            # Check for reward hacking patterns
            warnings = self._check_hacking_patterns(code)

            if func is not None and not error:
                return {
                    "code": code, "function": func, "error": None,
                    "warnings": warnings, "usage": usage, "attempts": attempt,
                }

            # On failure, append error feedback and retry
            last_error = error
            messages.append({"role": "assistant", "content": raw_code})
            messages.append({
                "role": "user",
                "content": (
                    f"The generated code has an error: {error}\n"
                    + (f"Warnings: {'; '.join(warnings)}\n" if warnings else "")
                    + "Please fix it. Remember: NO per-step constant bonuses, "
                    "ALL shaping must be delta-based or event-triggered. "
                    "Output ONLY the corrected function."
                ),
            })
            # Slight temperature bump on retry for diversity
            temperature = min(temperature + 0.1, 0.7)

        return {
            "code": code if 'code' in dir() else None,
            "function": None,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "warnings": warnings if 'warnings' in dir() else [],
            "usage": usage if 'usage' in dir() else {},
            "attempts": max_retries,
        }

    def _extract_code(self, raw: str) -> str:
        """Extract Python code from potential markdown formatting."""
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _check_hacking_patterns(self, code: str) -> list:
        """Check code for common reward hacking patterns. Returns list of warnings."""
        warnings = []
        for pattern, description in self._HACKING_PATTERNS:
            if re.search(pattern, code, re.MULTILINE):
                warnings.append(description)

        # Check: does the code have a time penalty?
        if "0.002" not in code and "0.001" not in code and "0.003" not in code:
            has_step_penalty = bool(re.search(r'reward\s*-=\s*0\.0\d', code))
            if not has_step_penalty:
                warnings.append("No time penalty detected — agent may stall to accumulate shaping reward")

        return warnings

    def _compile_and_validate(self, code: str) -> tuple:
        """
        Compile code and validate it meets requirements.
        Returns (function, error_message).
        """
        # Check for forbidden imports
        if "import " in code:
            return None, "Code contains 'import' statement (forbidden)"

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, f"Syntax error: {e}"

        # Check function structure
        func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if len(func_defs) == 0:
            return None, "No function definition found"
        if len(func_defs) > 1:
            return None, f"Expected 1 function, found {len(func_defs)}"
        if func_defs[0].name != "llm_reward_formula":
            return None, f"Function must be named 'llm_reward_formula', got '{func_defs[0].name}'"

        # Check signature
        args = func_defs[0].args
        if len(args.args) != 3:
            return None, f"Function must have 3 parameters, got {len(args.args)}"

        # Compile and extract
        try:
            local_env = {}
            exec(code, {"__builtins__": __builtins__}, local_env)
            func = local_env.get("llm_reward_formula")
            if func is None:
                return None, "Function 'llm_reward_formula' not found after exec"
        except Exception as e:
            return None, f"Execution error: {e}"

        # Test with mock data
        test_error = self._test_function(func)
        if test_error:
            return None, test_error

        return func, None

    def _test_function(self, func) -> str:
        """Run the function with mock data to verify it works. Returns error message or None."""
        mock_features = {
            "ball_x": 0.2, "ball_y": 0.1, "ball_z": 0.0,
            "ball_owned_team": 0, "ball_dist_to_goal": 0.9,
            "teammate_proximity_to_ball": 0.5, "active_player_idx": 3,
            "left_team_xs": [0.0] * 11, "left_team_ys": [0.0] * 11,
            "right_team_xs": [0.5] * 11, "right_team_ys": [0.0] * 11,
        }
        mock_prev = {
            "ball_x": 0.15, "ball_y": 0.1, "ball_z": 0.0,
            "ball_owned_team": 0, "ball_dist_to_goal": 0.95,
            "teammate_proximity_to_ball": 0.55, "active_player_idx": 3,
            "left_team_xs": [0.0] * 11, "left_team_ys": [0.0] * 11,
            "right_team_xs": [0.5] * 11, "right_team_ys": [0.0] * 11,
        }

        try:
            result = func(0.0, mock_features, mock_prev)
            if not isinstance(result, (int, float)):
                return f"Function returned {type(result).__name__}, expected float"
            if abs(result) > 10:
                return f"Reward magnitude {result} too large (expected ~0.01-0.1 range)"
        except Exception as e:
            return f"Runtime error with mock data: {e}"

        # ---- Stalling test: same state for 100 steps should not accumulate large reward ----
        stall_reward = 0.0
        for _ in range(100):
            stall_reward += func(0.0, mock_features, mock_features)  # same state = no delta
        if stall_reward > 0.5:
            return (
                f"Stalling test failed: 100 identical steps accumulated reward = {stall_reward:.3f}. "
                f"This exceeds the sparse goal reward (+1) and will cause reward hacking. "
                f"Remove per-step constant bonuses."
            )

        return None


def format_for_yaml(code: str, indent: int = 4) -> str:
    """Format code for embedding in YAML config (with proper indentation)."""
    lines = code.split("\n")
    prefix = " " * indent
    return "\n".join(prefix + line for line in lines)


def get_available_scenarios() -> list:
    """Return list of available scenario names."""
    return list(SCENARIO_CONTEXTS.keys())