"""
Custom Gym wrappers for GRF: reward shaping and (future) action masking.

Direction A: LLM-generated reward — use CustomRewardWrapper with a formula
that runs once (or is generated once by the LLM) and is applied at each step.
Direction B: Coach / action masking — see core/coach.py.
"""

import numpy as np
import gym


# ---------------------------------------------------------------------------
# Simple115 observation layout (GRF): 115 floats
# Indices: 0-21 left_team (11*2), 22-43 left_team_direction,
#          44-65 right_team, 66-87 right_team_direction,
#          88-90 ball (x,y,z), 91-93 ball_direction,
#          94-96 ball_owned_team one-hot, 97-107 active player one-hot,
#          108-114 game_mode one-hot
# ---------------------------------------------------------------------------
BALL_START = 88
BALL_DIM = 3
RIGHT_GOAL_X = 1.0   # attacking goal for left team
LEFT_GOAL_X = -1.0


def observation_to_features(obs: np.ndarray) -> dict:
    """
    Convert the 115-dim simple115 observation into a small feature dict
    suitable for LLM-generated reward formulas or logging.

    Args:
        obs: shape (115,) or (1, 115); will be flattened to (115,).

    Returns:
        dict with keys: ball_x, ball_y, ball_z, ball_owned_team,
                       left_team_xs, left_team_ys, right_team_xs, right_team_ys,
                       ball_dist_to_goal, active_player_idx.
    """
    o = np.asarray(obs).flatten()
    if o.size < 115:
        return _empty_features()

    ball = o[BALL_START : BALL_START + BALL_DIM]
    ball_x, ball_y = float(ball[0]), float(ball[1])

    # Left team: first 22 values (11 players * 2)
    left_x = o[0:22:2]
    left_y = o[1:22:2]
    # Right team: indices 44-65
    right_x = o[44:66:2]
    right_y = o[45:66:2]

    # Distance of ball to attacking goal (for left team, goal is at x=+1)
    ball_dist_to_goal = float(np.sqrt((RIGHT_GOAL_X - ball_x) ** 2 + (0 - ball_y) ** 2))
    # Teammate proximity to ball (mean distance of left team to ball)
    left_dists = np.sqrt((left_x - ball_x) ** 2 + (left_y - ball_y) ** 2)
    teammate_proximity_to_ball = float(np.mean(left_dists))

    active = -1
    for i in range(11):
        if o[97 + i] > 0.5:
            active = i
            break

    return {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_z": float(ball[2]) if len(ball) > 2 else 0.0,
        "ball_owned_team": int(np.argmax(o[94:97]) - 1) if np.any(o[94:97]) else -1,  # -1, 0, 1
        "left_team_xs": left_x.tolist(),
        "left_team_ys": left_y.tolist(),
        "right_team_xs": right_x.tolist(),
        "right_team_ys": right_y.tolist(),
        "ball_dist_to_goal": ball_dist_to_goal,
        "teammate_proximity_to_ball": teammate_proximity_to_ball,
        "active_player_idx": active,
    }


def _empty_features():
    return {
        "ball_x": 0.0, "ball_y": 0.0, "ball_z": 0.0,
        "ball_owned_team": -1,
        "left_team_xs": [], "left_team_ys": [],
        "right_team_xs": [], "right_team_ys": [],
        "ball_dist_to_goal": 1.0, "teammate_proximity_to_ball": 1.0,
        "active_player_idx": -1,
    }


# ---------------------------------------------------------------------------
# Custom reward wrapper: apply a custom formula on top of env reward.
# LLM can output a Python snippet that uses observation_to_features(obs)
# and returns a scalar; we wrap that in a callable and pass it here.
# ---------------------------------------------------------------------------

class CustomRewardWrapper(gym.Wrapper):
    """
    Applies a custom reward formula from the raw env reward and current (and
    optionally previous) observation. Use this to implement LLM-generated
    dense rewards without calling the LLM at training time.

    Formula signature: reward_fn(env_reward, features_dict, prev_features_dict or None)
    -> scalar. If you only need current state, ignore prev_features_dict.
    """

    def __init__(self, env, reward_fn=None, use_delta_features=True):
        super().__init__(env)
        self._reward_fn = reward_fn or _default_sparse  # pass-through
        self._use_delta = use_delta_features
        self._prev_features = None

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        obs_flat = np.asarray(obs).flatten()
        features = observation_to_features(obs_flat)
        prev = self._prev_features
        if self._use_delta:
            self._prev_features = features.copy() if isinstance(features, dict) else features
        custom_reward = self._reward_fn(env_reward, features, prev)
        return obs, float(custom_reward), done, info

    def reset(self, **kwargs):
        self._prev_features = None
        return self.env.reset(**kwargs)


def _default_sparse(env_reward, features, prev_features):
    """Default: keep the environment reward (e.g. +1 goal, 0 otherwise)."""
    return env_reward


def make_dense_reward_fn(
    ball_dist_weight=0.0,
    teammate_proximity_weight=0.0,
    goal_bonus=1.0,
    sparse_fallback=True,
):
    """
    Build a reward function suitable for CustomRewardWrapper.

    Example (LLM-style): 0.5 * (ball_distance_to_goal_delta) + 0.2 * (teammate_proximity).
    Here we use: goal_bonus * env_goal + ball_dist_weight * (-delta_dist) + teammate_proximity_weight * (-proximity).
    So closer ball to goal and closer teammates to ball give extra positive reward.
    """

    def fn(env_reward, features, prev_features):
        out = 0.0
        if sparse_fallback:
            out += goal_bonus * env_reward

        # Dense: encourage ball moving toward goal
        if ball_dist_weight != 0 and prev_features is not None:
            delta = prev_features.get("ball_dist_to_goal", 0) - features.get("ball_dist_to_goal", 0)
            out += ball_dist_weight * delta

        # Dense: encourage teammates close to ball (negative proximity = closer)
        if teammate_proximity_weight != 0:
            out -= teammate_proximity_weight * features.get("teammate_proximity_to_ball", 0)

        return out

    return fn
