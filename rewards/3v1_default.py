"""
LLM-generated reward function for GRF.

Task: (default comprehensive reward)
Scenario: 3v1

Usage in config YAML:
  reward_wrapper:
    use_llm_reward: true
    llm_reward_file: rewards/3v1_default.py
"""

def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # 1. Ball progress (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.05 * delta_dist
    
    # 2. Possession bonus
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    # 3. Phase-aware logic
    ball_x = features.get("ball_x", 0.0)
    
    # Defensive third (ball_x < -0.33): safe possession
    if ball_x < -0.33:
        # Small penalty for losing possession in defensive third
        if prev_features.get("ball_owned_team", -1) == 0 and features.get("ball_owned_team", -1) != 0:
            reward -= 0.02
    
    # Midfield (-0.33 <= ball_x <= 0.33): progression
    elif ball_x <= 0.33:
        # Extra reward for moving ball forward in midfield
        delta_x = features.get("ball_x", 0.0) - prev_features.get("ball_x", 0.0)
        if delta_x > 0:
            reward += 0.02 * delta_x
    
    # Attacking third (ball_x > 0.33): shooting opportunities
    else:
        # Bonus for being in shooting position
        reward += 0.01
        
        # Consider keeper position when shooting
        keeper_x = features.get("right_team_xs", [0.0])[0]
        keeper_y = features.get("right_team_ys", [0.0])[0]
        ball_y = features.get("ball_y", 0.0)
        
        # Reward moving ball away from keeper horizontally
        keeper_horiz_dist = abs(ball_y - keeper_y)
        prev_keeper_dist = abs(prev_features.get("ball_y", 0.0) - prev_features.get("right_team_ys", [0.0])[0])
        reward += 0.02 * (keeper_horiz_dist - prev_keeper_dist)
    
    # 4. Team coordination (3 vs 1 scenario)
    if features.get("ball_owned_team", -1) == 0:
        # Reward teammates supporting the ball carrier
        prev_prox = prev_features.get("teammate_proximity_to_ball", 1.0)
        curr_prox = features.get("teammate_proximity_to_ball", 1.0)
        delta_prox = prev_prox - curr_prox
        reward += 0.015 * delta_prox
        
        # Small penalty for clustering (encourage spreading)
        if curr_prox < 0.2:  # Very close clustering
            reward -= 0.005
    
    return reward
