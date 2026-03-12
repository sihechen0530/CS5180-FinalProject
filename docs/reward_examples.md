# LLM Reward Function Examples

This document contains example task descriptions and their generated reward functions.

## Usage

```bash
# Activate environment
source .venv_local/bin/activate  # local
# or
source environment.sh            # cluster

# Set API key
export DEEPSEEK_API_KEY='your-key'

# Generate reward function
python scripts/generate_reward.py \
  --scenario 3v1 \
  --task "Your task description here" \
  --snippet rewards/my_reward.py
```

---

## Example 1: Ball Progression + Shooting (3v1)

**Task:** `"Encourage moving the ball toward the goal and shooting when close"`

**Scenario:** `3v1`

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Ball progress toward goal (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.04 * delta_dist
    
    # Bonus when ball is close to goal
    ball_dist = features.get("ball_dist_to_goal", 1.0)
    if ball_dist < 0.3:
        reward += 0.02
    
    # Encourage shooting when close and keeper is not blocking center
    keeper_x = features.get("right_team_xs", [0.0])[0]
    keeper_y = features.get("right_team_ys", [0.0])[0]
    ball_x = features.get("ball_x", 0.0)
    
    if ball_dist < 0.4 and abs(keeper_y) > 0.1 and ball_x > 0.5:
        reward += 0.01
    
    # Small possession bonus
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    return reward
```

---

## Example 2: Possession + Team Coordination (3v1)

**Task:** `"Reward maintaining possession and teammates supporting the ball carrier"`

**Scenario:** `3v1`

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Possession bonus
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    # Ball progress toward goal (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.04 * delta_dist
    
    # Teammate support: reward teammates moving closer to ball when we have possession
    if features.get("ball_owned_team", -1) == 0:
        prev_prox = prev_features.get("teammate_proximity_to_ball", 1.0)
        curr_prox = features.get("teammate_proximity_to_ball", 1.0)
        delta_prox = prev_prox - curr_prox
        reward += 0.02 * delta_prox
    
    # Bonus for ball in attacking third
    ball_x = features.get("ball_x", 0.0)
    if ball_x > 0.5:
        reward += 0.01
    
    return reward
```

---

## Example 3: Beat the Keeper (3v1)

**Task:** `"Encourage finding shooting angles when the keeper is out of position"`

**Scenario:** `3v1`

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Ball progress toward goal (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.04 * delta_dist
    
    # Shooting angle bonus when keeper is out of position
    ball_x = features.get("ball_x", 0.0)
    ball_y = features.get("ball_y", 0.0)
    keeper_x = features.get("right_team_xs", [1.0])[0]
    keeper_y = features.get("right_team_ys", [0.0])[0]
    
    # Distance from ball to keeper
    keeper_dist_x = abs(ball_x - keeper_x)
    keeper_dist_y = abs(ball_y - keeper_y)
    keeper_dist = (keeper_dist_x**2 + keeper_dist_y**2)**0.5
    
    # Reward when ball is close to goal and keeper is far (good shooting angle)
    if ball_x > 0.5 and keeper_dist > 0.3:
        reward += 0.02
    
    # Small possession bonus
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    return reward
```

---

## Example 4: Empty Goal - Dribble and Shoot

**Task:** `"Learn to dribble and shoot into the empty goal"`

**Scenario:** `empty_goal`

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Ball progress toward goal (potential-based)
    delta_dist = prev_features.get("ball_dist_to_goal", 1.0) - features.get("ball_dist_to_goal", 1.0)
    reward += 0.05 * delta_dist
    
    # Bonus for forward movement
    delta_x = features.get("ball_x", 0.0) - prev_features.get("ball_x", 0.0)
    if delta_x > 0:
        reward += 0.02 * delta_x
    
    # Possession bonus
    if features.get("ball_owned_team", -1) == 0:
        reward += 0.01
    
    # Aim for goal center when close
    ball_x = features.get("ball_x", 0.0)
    ball_y = features.get("ball_y", 0.0)
    if ball_x > 0.5:
        reward -= 0.005 * abs(ball_y)
    
    # Close to goal bonus
    if features.get("ball_dist_to_goal", 1.0) < 0.3:
        reward += 0.01
    
    return reward
```

---

## Example 5: Fast Counter Attack

**Task:** `"Reward fast ball advancement and quick transitions from defense to attack"`

**Scenario:** `counterattack`

```python
def llm_reward_formula(env_reward, features, prev_features):
    reward = float(env_reward)
    
    # Fast ball advancement (strong emphasis)
    delta_x = features.get("ball_x", 0.0) - prev_features.get("ball_x", 0.0)
    reward += 0.06 * delta_x
    
    # Speed bonus for rapid advancement
    if delta_x > 0.02:
        reward += 0.03
    
    # Transition bonus: gaining possession in defensive half
    prev_possession = prev_features.get("ball_owned_team", -1)
    curr_possession = features.get("ball_owned_team", -1)
    ball_x = features.get("ball_x", 0.0)
    
    if prev_possession != 0 and curr_possession == 0 and ball_x < 0.0:
        reward += 0.04  # Start of counter
    
    # Penalty for backward movement during counter
    if curr_possession == 0 and delta_x < -0.01:
        reward -= 0.02
    
    # Bonus when entering final third
    if ball_x > 0.33 and curr_possession == 0:
        reward += 0.01
    
    return reward
```

---

## More Task Ideas

| Scenario | Task Description |
|----------|------------------|
| `3v1` | `"Encourage quick one-two passing to create space"` |
| `3v1` | `"Reward drawing the keeper out before passing to open teammate"` |
| `3v1` | `"Penalize losing possession and reward recovery"` |
| `empty_goal` | `"Encourage powerful shots from distance"` |
| `11v11` | `"Balance between possession retention and forward progression"` |
| `11v11` | `"Reward defensive shape when opponent has the ball"` |
| `counterattack` | `"Encourage long passes to break defensive lines"` |

---

## Tips for Writing Task Descriptions

1. **Be specific about the behavior** - "move ball toward goal" is better than "play well"
2. **Mention relevant features** - "when close to goal", "when keeper is out of position"
3. **Use action words** - "encourage", "reward", "penalize"
4. **Consider the scenario** - empty goal doesn't need keeper-related rewards
