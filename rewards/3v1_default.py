"""
Fixed reward function for 3v1 with keeper.

Key fixes:
1. Remove all non-potential-based (per-step) bonuses → eliminate reward hacking
2. Add explicit shooting incentive in attacking zone
3. Use time penalty to discourage stalling
4. Keep all shaping as deltas (potential-based)
"""

def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)

    ball_x = features.get("ball_x", 0.0)
    ball_owned = features.get("ball_owned_team", -1)

    # -------------------------------------------------------------------------
    # 1. Ball-to-goal progress (potential-based, always active)
    #    This is the primary shaping signal.
    # -------------------------------------------------------------------------
    delta_dist = (
        prev_features.get("ball_dist_to_goal", 1.0)
        - features.get("ball_dist_to_goal", 1.0)
    )
    reward += 0.08 * delta_dist

    # -------------------------------------------------------------------------
    # 2. Possession CHANGE reward (NOT per-step bonus)
    #    Only reward the moment of gaining possession, not holding it.
    # -------------------------------------------------------------------------
    prev_owned = prev_features.get("ball_owned_team", -1)
    if prev_owned != 0 and ball_owned == 0:
        reward += 0.02  # gained possession
    elif prev_owned == 0 and ball_owned != 0:
        reward -= 0.03  # lost possession

    # -------------------------------------------------------------------------
    # 3. Time/step penalty (small but crucial for 3v1)
    #    Forces agent to act decisively instead of stalling.
    # -------------------------------------------------------------------------
    reward -= 0.002

    # -------------------------------------------------------------------------
    # 4. Shooting zone incentive (potential-based, NOT constant)
    #    Reward ENTERING the shooting zone, not staying in it.
    #    Use ball_x crossing a threshold as a one-time-ish signal.
    # -------------------------------------------------------------------------
    prev_ball_x = prev_features.get("ball_x", 0.0)

    # Reward crossing into deep attacking zone (ball_x > 0.7)
    if ball_x > 0.7 and prev_ball_x <= 0.7 and ball_owned == 0:
        reward += 0.03

    # -------------------------------------------------------------------------
    # 5. Keeper-avoidance shaping (potential-based)
    #    In attacking third, reward moving ball to be far from keeper.
    #    Only when we have the ball to avoid rewarding random bounces.
    # -------------------------------------------------------------------------
    if ball_x > 0.5 and ball_owned == 0:
        ball_y = features.get("ball_y", 0.0)
        keeper_y = features.get("right_team_ys", [0.0])[0]

        curr_keeper_dist = abs(ball_y - keeper_y)
        prev_keeper_dist = abs(
            prev_features.get("ball_y", 0.0)
            - prev_features.get("right_team_ys", [0.0])[0]
        )
        reward += 0.03 * (curr_keeper_dist - prev_keeper_dist)

    # -------------------------------------------------------------------------
    # 6. Ball height penalty (discourage aimless lobs)
    #    Slightly penalize high balls that aren't shots on goal.
    # -------------------------------------------------------------------------
    ball_z = features.get("ball_z", 0.0)
    if ball_z > 0.5 and ball_x < 0.8:
        reward -= 0.005

    return reward