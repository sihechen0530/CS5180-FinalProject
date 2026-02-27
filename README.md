# GRF LLM Coach — CS5180 Final Project

Repository for training and evaluating PPO agents on Google Research Football (GRF), with a path toward **LLM-generated reward functions** and **LLM-as-Coach action masking**.

## Structure

```
grf-llm-coach/
├── README.md
├── requirements.txt
├── .gitignore
├── agents/                 # Saved .zip models
│   ├── baselines/          # Pure PPO (3v1, empty)
│   └── llm_augmented/     # Models trained with LLM influence
├── core/                   # Custom logic
│   ├── wrappers.py         # CustomRewardWrapper, observation_to_features()
│   └── coach.py            # LLM Coach stub (role + action mask)
├── scripts/
│   ├── train.py            # Main training (config-driven)
│   └── evaluate.py         # Evaluation and video
├── configs/
│   ├── 3v1_config.yaml
│   └── empty_config.yaml
└── notebook/               # Analysis (TensorBoard, reward curves)
```

## Setup

1. **Python env** (3.8+). Install GRF from the sibling `football/` repo:
   ```bash
   pip install -e ../football
   ```
2. **This repo**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Train (3v1 or empty):**
```bash
python scripts/train.py --config configs/3v1_config.yaml
python scripts/train.py --config configs/empty_config.yaml
```
Checkpoints go to `agents/baselines/<run_name>_checkpoints/`; final model to `agents/baselines/<run_name>.zip`.

**Evaluate:**
```bash
python scripts/evaluate.py --config configs/3v1_config.yaml
python scripts/evaluate.py --config configs/empty_config.yaml --no-video
python scripts/evaluate.py --config configs/3v1_config.yaml --agent agents/llm_augmented/ppo_3v1
```

**Override config from CLI:**
```bash
python scripts/train.py --config configs/3v1_config.yaml --override total_timesteps=100000
```

## Development directions

### A — LLM-generated reward (dense reward)

- **Idea:** Replace sparse (+1/0) with a dense reward formula. The LLM writes a **reward snippet once**; we run it in a `RewardWrapper` at each step (no LLM at training time).
- **Hook:** `core/wrappers.py` — `CustomRewardWrapper` and `observation_to_features()`. Enable in config: `reward_wrapper.use_custom: true` and set weights (e.g. `ball_dist_weight`, `teammate_proximity_weight`).

### B — LLM as Coach (action masking)

- **Idea:** Every N steps the Coach assigns a **role** (Attacker / Supporter / Defender) and returns an **action mask** to prune bad actions (e.g. mask Long Pass when no teammate in range).
- **Hook:** `core/coach.py` — `get_role_and_action_mask()`. Stub is rule-based; next step is to plug in LLM or richer rules and an env wrapper that applies the mask (e.g. via SB3’s action masking support).

### Curriculum (suggested order)

1. **Stage 1:** Train with action masking only (faster empty & 3v1 basics).
2. **Stage 2:** Train with LLM-generated dense reward.
3. **Stage 3:** Combine both.

## Config

- **configs/3v1_config.yaml** — Academy 3v1 with keeper, 5M steps, 16 envs.
- **configs/empty_config.yaml** — Empty goal, 100k steps.

Keys: `env_id`, `agents_dir`, `run_name`, `total_timesteps`, `num_cpu`, `reward_wrapper`, `coach_interval`, `use_action_masking`, and optional PPO overrides.
