"""
LLM Coach: role assignment and action masking.

Direction B: Every N steps the "Coach" (LLM or rule-based) analyzes state,
assigns a role (Attacker, Supporter, Defender), and returns a boolean mask
of allowed actions. This trims exploration and accelerates convergence.

Usage (future):
  from core.coach import get_role_and_action_mask
  mask = get_role_and_action_mask(obs, step_count, coach_interval=50)
  # Use mask in SB3 via a custom env that implements action_mask in info.
"""

from typing import Tuple, List, Optional
import numpy as np

from core.wrappers import observation_to_features


# Roles for 3v1 / multi-agent
ROLE_ATTACKER = "Attacker"
ROLE_SUPPORTER = "Supporter"
ROLE_DEFENDER = "Defender"


def get_role_and_action_mask(
    obs: np.ndarray,
    step_count: int,
    coach_interval: int = 50,
    num_actions: int = 19,
) -> Tuple[str, Optional[np.ndarray]]:
    """
    Analyze state every coach_interval steps; assign role and return action mask.

    Args:
        obs: observation (115,) or (1, 115).
        step_count: current env step (used to decide if we call the coach).
        coach_interval: call coach every N steps.
        num_actions: size of GRF discrete action space (default 19).

    Returns:
        (role_name, mask): mask is bool array of shape (num_actions,) where
        True = allowed. If step_count % coach_interval != 0, returns (last_role, None)
        to mean "no masking this step".
    """
    # Stub: no LLM yet; rule-based role from ball position.
    # When step_count % coach_interval != 0, we would return cached role and None mask.
    features = observation_to_features(obs)
    ball_x = features.get("ball_x", 0.0)

    if ball_x > 0:
        role = ROLE_ATTACKER
    elif ball_x < -0.3:
        role = ROLE_DEFENDER
    else:
        role = ROLE_SUPPORTER

    # Stub mask: allow all actions (no masking). Replace with role-based mask later.
    mask = np.ones(num_actions, dtype=bool)

    # Example future logic: if role == ROLE_DEFENDER, mask out "Sprint" toward goal, etc.
    # if role == ROLE_DEFENDER:
    #     mask[LONG_PASS_ACTION_ID] = False  # when no teammate in range

    return role, mask
