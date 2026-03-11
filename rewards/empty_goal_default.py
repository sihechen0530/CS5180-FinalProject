"""
LLM-generated reward function for GRF.

Task: (default comprehensive reward)
Scenario: empty_goal

Usage in config YAML:
  reward_wrapper:
    use_llm_reward: true
    llm_reward_file: rewards/empty_goal_default.py
"""

def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Ball progress toward empty goal (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.05 * delta_dist
    
    # Possession bonus when our team controls the ball
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    # Phase-aware logic: encourage shooting when close to goal
    ball_x = features.get("ball_x", 0.0)
    ball_dist = features.get("ball_dist_to_goal", 1.0)
    
    # Attacking third (close to empty goal) - emphasize shooting
    if ball_x > 0.33:
        # Bonus for being very close to goal
        if ball_dist < 0.2:
            reward += 0.02
        # Small penalty for moving away from goal in attacking area
        if delta_dist < 0:
            reward -= 0.01
    
    # Encourage ball movement toward goal center (y-axis accuracy)
    ball_y = abs(features.get("ball_y", 0.0))
    prev_ball_y = abs(prev_features.get("ball_y", 0.0))
    delta_y = prev_ball_y - ball_y  # positive = moving toward center
    reward += 0.02 * delta_y
    
    # Bonus for ball being in the air (z > 0) when close to goal - indicates shooting
    if ball_dist < 0.3 and features.get("ball_z", 0.0) > 0.1:
        reward += 0.01
    
    return reward
