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
""",

    "empty": """
## Scenario: Empty Goal (academy_empty_goal_close)
- Single agent vs empty goal (no goalkeeper, no opponents)
- Agent starts with the ball near the goal
- Objective: Learn to shoot and score
- Key insight: No need to consider opponents; focus on ball control and shooting accuracy
- Typical episode length: very short (< 50 steps)
""",

    "3v1": """
## Scenario: 3 vs 1 with Keeper (academy_3_vs_1_with_keeper)
- 3 agents (our team) vs 1 goalkeeper
- Our team starts with possession in the attacking third
- Objective: Coordinate passing and movement to beat the keeper and score
- Key insight: The keeper (right_team index 0) guards the goal; consider keeper position when shooting
- Useful features: right_team_xs[0], right_team_ys[0] give keeper position
- Typical episode length: short (< 100 steps)
""",

    "academy_3_vs_1_with_keeper": """
## Scenario: 3 vs 1 with Keeper (academy_3_vs_1_with_keeper)
- 3 agents (our team) vs 1 goalkeeper
- Our team starts with possession in the attacking third
- Objective: Coordinate passing and movement to beat the keeper and score
- Key insight: The keeper (right_team index 0) guards the goal; consider keeper position when shooting
- Useful features: right_team_xs[0], right_team_ys[0] give keeper position
- Typical episode length: short (< 100 steps)
""",

    "11v11": """
## Scenario: Full Game (11_vs_11_stochastic)
- Full 11 vs 11 match with stochastic opponent
- Complex coordination required across the entire pitch
- Objective: Score more goals than the opponent
- Key insight: Must handle both offense and defense; long episodes with varied situations
- Consider: possession transitions, defensive positioning, counter-attacks
- Typical episode length: 3000 steps (full match)
""",

    "counterattack": """
## Scenario: Counter Attack (academy_counterattack_easy/hard)
- Start from defensive position, opponent has lost the ball
- Fast transition from defense to attack
- Objective: Quickly advance and score before defense recovers
- Key insight: Speed is crucial; reward rapid ball advancement
- Typical episode length: medium (100-200 steps)
""",
}

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

REWARD_SYSTEM_PROMPT = """You are an expert reinforcement learning engineer designing dense reward functions for the Google Research Football (GRF) environment.

## Default Behavior (ALWAYS include these baseline components)

Your reward function should ALWAYS include these fundamental components as a baseline:

1. **Ball Progress** (potential-based): Reward ball moving toward opponent goal
   - Use `prev_features["ball_dist_to_goal"] - features["ball_dist_to_goal"]`
   
2. **Possession**: Small bonus when our team controls the ball
   - Check `features["ball_owned_team"] == 0`

3. **Phase-Aware Logic**: Adjust rewards based on pitch position
   - Defensive third (ball_x < -0.33): emphasize safe possession
   - Midfield (-0.33 <= ball_x <= 0.33): emphasize progression
   - Attacking third (ball_x > 0.33): emphasize shooting opportunities

The user's task description provides **additional emphasis or modifications** to this baseline.
If the task says "emphasis on X", increase weights for X while keeping other components.
If the task describes a specific behavior, add it ON TOP of the baseline components.

## Available Features (observation dict)

The reward function receives two dicts: `features` (current step) and `prev_features` (previous step).
Both have identical structure with these keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| ball_x | float | [-1, 1] | Ball x-position. -1 = own goal, +1 = opponent goal (attacking direction) |
| ball_y | float | [-0.42, 0.42] | Ball y-position. 0 = center, ± = sidelines |
| ball_z | float | [0, ~1] | Ball height above ground |
| ball_owned_team | int | {-1, 0, 1} | -1 = loose ball, 0 = our team, 1 = opponent |
| ball_dist_to_goal | float | [0, ~2.2] | Euclidean distance from ball to opponent goal center |
| teammate_proximity_to_ball | float | [0, ~2] | Mean distance of all teammates to the ball |
| active_player_idx | int | [0, 10] | Index of the currently controlled player |
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
2. Dense shaping magnitude: keep each component in [0.01, 0.1] range per step to avoid overwhelming the sparse goal signal (+1)
3. Prefer potential-based shaping: use `prev_features[key] - features[key]` for approaching-goal rewards (ensures no artificial reward loops)
4. DO NOT import any modules - only use built-in operations (+, -, *, /, min, max, abs, sum, len)
5. Always preserve the sparse signal: include `env_reward` in the final return
6. Use .get() for safe dict access with defaults

## Output Format

Return ONLY the Python function definition. No explanation, no markdown code fences, no extra text.
Start directly with `def llm_reward_formula(` and end with `return reward`.
Keep the code concise (under 40 lines) - combine similar logic where possible."""


REWARD_FEW_SHOT_EXAMPLES = """
## Example 1: Simple potential-based (ball approaching goal)

Task: "Encourage the agent to move the ball toward the opponent's goal"

```python
def llm_reward_formula(env_reward, features, prev_features):
    # Potential-based: reward reduction in ball-to-goal distance
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    return float(env_reward) + 0.05 * delta_dist
```

## Example 2: Multi-factor (possession + ball progress)

Task: "Reward maintaining possession while advancing the ball"

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Possession bonus: small reward when our team has the ball
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    # Ball progress: potential-based shaping for moving ball forward
    delta_x = features.get("ball_x", 0.0) - prev_features.get("ball_x", 0.0)
    reward += 0.03 * delta_x  # positive delta_x = moving toward opponent goal
    
    return reward
```

## Example 3: Team coordination with boundary handling

Task: "Encourage teammates to spread out and support the ball carrier"

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Only apply team shaping when we have possession
    if features.get("ball_owned_team", -1) != 0:
        return reward
    
    # Reward teammates getting closer to ball (better support)
    prev_prox = prev_features.get("teammate_proximity_to_ball", 1.0)
    curr_prox = features.get("teammate_proximity_to_ball", 1.0)
    delta_prox = prev_prox - curr_prox  # positive = teammates moved closer
    reward += 0.02 * delta_prox
    
    # Small bonus for ball in attacking third (x > 0.33)
    ball_x = features.get("ball_x", 0.0)
    if ball_x > 0.33:
        reward += 0.01
    
    return reward
```
"""


REWARD_USER_PROMPT_TEMPLATE = """{scenario_context}{task_section}
Write a comprehensive dense reward function that includes:
1. The baseline components (ball progress, possession, phase-aware logic)
2. Any additional emphasis if specified above

Follow all constraints and output ONLY the Python function."""


# ---------------------------------------------------------------------------
# RewardGenerator Class
# ---------------------------------------------------------------------------

class RewardGenerator:
    """
    Generates reward functions using LLM APIs (DeepSeek or Groq).
    """

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

    def generate(self, task_description: str = None, scenario: str = None, temperature: float = 0.2) -> dict:
        """
        Generate a reward function, optionally with additional emphasis.

        Args:
            task_description: Optional additional emphasis (e.g., "Extra emphasis on passing").
                              If None, generates a default comprehensive reward.
            scenario: Optional scenario name (e.g., "3v1", "empty_goal") for context
            temperature: LLM sampling temperature (lower = more deterministic)

        Returns:
            dict with keys:
                - "code": the generated Python function as a string
                - "function": the compiled callable (or None if compilation failed)
                - "error": error message if any, else None
                - "usage": token usage info
        """
        # Build scenario context if provided
        scenario_context = ""
        if scenario:
            scenario_context = SCENARIO_CONTEXTS.get(scenario, SCENARIO_CONTEXTS.get(scenario.lower(), ""))
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
                task_section=task_section
            )},
        ]

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
            return {"code": None, "function": None, "error": f"API Error: {e}", "usage": {}}

        # Clean up code (remove markdown fences if present)
        code = self._extract_code(raw_code)

        # Validate and compile
        func, error = self._compile_and_validate(code)

        return {"code": code, "function": func, "error": error, "usage": usage}

    def _extract_code(self, raw: str) -> str:
        """Extract Python code from potential markdown formatting."""
        # Remove ```python ... ``` blocks
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no markdown, assume raw code
        return raw.strip()

    def _compile_and_validate(self, code: str) -> tuple:
        """
        Compile code and validate it meets requirements.
        Returns (function, error_message).
        """
        # Check for forbidden imports
        if "import " in code:
            return None, "Code contains 'import' statement (forbidden)"

        # Parse AST to check structure
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, f"Syntax error: {e}"

        # Check that it defines exactly one function named llm_reward_formula
        func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if len(func_defs) == 0:
            return None, "No function definition found"
        if len(func_defs) > 1:
            return None, f"Expected 1 function, found {len(func_defs)}"
        if func_defs[0].name != "llm_reward_formula":
            return None, f"Function must be named 'llm_reward_formula', got '{func_defs[0].name}'"

        # Check function signature (3 parameters)
        args = func_defs[0].args
        n_args = len(args.args)
        if n_args != 3:
            return None, f"Function must have 3 parameters, got {n_args}"

        # Try to compile and extract function
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
            "ball_x": 0.2,
            "ball_y": 0.1,
            "ball_z": 0.0,
            "ball_owned_team": 0,
            "ball_dist_to_goal": 0.9,
            "teammate_proximity_to_ball": 0.5,
            "active_player_idx": 3,
            "left_team_xs": [0.0] * 11,
            "left_team_ys": [0.0] * 11,
            "right_team_xs": [0.5] * 11,
            "right_team_ys": [0.0] * 11,
        }
        mock_prev = {
            "ball_x": 0.15,
            "ball_y": 0.1,
            "ball_z": 0.0,
            "ball_owned_team": 0,
            "ball_dist_to_goal": 0.95,
            "teammate_proximity_to_ball": 0.55,
            "active_player_idx": 3,
            "left_team_xs": [0.0] * 11,
            "left_team_ys": [0.0] * 11,
            "right_team_xs": [0.5] * 11,
            "right_team_ys": [0.0] * 11,
        }

        try:
            result = func(0.0, mock_features, mock_prev)
            if not isinstance(result, (int, float)):
                return f"Function returned {type(result).__name__}, expected float"
            # Check for reasonable magnitude (warn but don't fail)
            if abs(result) > 10:
                return f"Warning: reward magnitude {result} seems too large (expected ~0.01-0.1)"
        except Exception as e:
            return f"Runtime error with mock data: {e}"

        return None


def format_for_yaml(code: str, indent: int = 4) -> str:
    """Format code for embedding in YAML config (with proper indentation)."""
    lines = code.split("\n")
    prefix = " " * indent
    return "\n".join(prefix + line for line in lines)


def get_available_scenarios() -> list:
    """Return list of available scenario names."""
    return list(SCENARIO_CONTEXTS.keys())
