"""
Evaluation-based early stopping for GRF training.

Implements the "Threshold" strategy:
- Isolated eval env (rewards='scoring' only, no shaping; deterministic policy).
- Periodic evaluation every eval_freq steps: run n_eval_episodes episodes, compute win rate.
- Stop if win rate >= target_win_rate (e.g. 95%).
"""

import os
import sys

# Allow import when run from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gfootball.env as football_env
from stable_baselines3.common.callbacks import BaseCallback
from core.wrappers import observation_to_features


class StopTrainingException(Exception):
    """Raised by GRFEvalStoppingCallback when early-stop conditions are met."""

    pass

MAX_DEEPSEEK_TOKEN_BUDGET = 100_000_000

class GRFEvalStoppingCallback(BaseCallback):
    """
    Periodic evaluation in an isolated env (rewards='scoring' only, deterministic).
    Stops training when:
    1) Eval win rate >= target_win_rate (e.g. 95%).
    """

    def __init__(
        self,
        env_id: str,
        eval_freq: int = 50000,
        n_eval_episodes: int = 100,
        target_win_rate: float = 0.95,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.env_id = env_id
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.target_win_rate = target_win_rate

        self._last_eval_step = 0
        self._win_rate_history = []
        self._eval_env = None

    def _make_eval_env(self):
        """Isolated eval env: scoring-only reward, no shaping, no Monitor."""
        env = football_env.create_environment(
            env_name=self.env_id,
            stacked=False,
            representation="simple115",
            rewards="scoring",
            write_video=False,
            write_full_episode_dumps=False,
            render=False,
        )
        return env

    def _eval_win_rate(self):
        """Run n_eval_episodes deterministic episodes; return fraction with reward > 0 (win)."""
        if self._eval_env is None:
            self._eval_env = self._make_eval_env()

        wins = 0
        for _ in range(self.n_eval_episodes):
            obs = self._eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self._eval_env.step(action)
            # Win = we scored (goal); reward > 0 for scoring in GRF
            if reward > 0:
                wins += 1

        return wins / self.n_eval_episodes

    def _on_step(self) -> bool:
        total = self.model.num_timesteps
        if total - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = total
        win_rate = self._eval_win_rate()
        self._win_rate_history.append(win_rate)

        if self.verbose >= 1:
            print(
                f"[Eval] step={total} win_rate={win_rate:.1%} "
                f"(history={[f'{x:.1%}' for x in self._win_rate_history]})"
            )

        # Target reached: scenario mastered
        if win_rate >= self.target_win_rate:
            if self.verbose >= 1:
                print(
                    f"[Eval] Target win rate {self.target_win_rate:.0%} reached. "
                    f"Stopping training."
                )
            raise StopTrainingException(
                f"Win rate {win_rate:.1%} >= target {self.target_win_rate:.0%}"
            )



        return True

    def _on_training_end(self):
        if self._eval_env is not None:
            try:
                self._eval_env.close()
            except Exception:
                pass
            self._eval_env = None

class CoachCallback(BaseCallback):
    """
    Periodic callback that interfaces with the Coach API (or mock)
    to generate Action Masks for the environments.
    """
    def __init__(self, coach_client, coach_interval: int = 50, max_token_budget: int = MAX_DEEPSEEK_TOKEN_BUDGET, verbose: int = 0):
        super().__init__(verbose)
        self.coach_client = coach_client
        self.coach_interval = coach_interval
        self.max_token_budget = max_token_budget

    def _on_step(self) -> bool:
        """
        Called after every n_envs steps.
        Updates action masks periodically.
        """
        if self.n_calls % self.coach_interval == 0:
            if "new_obs" in self.locals:
                new_obs = self.locals["new_obs"]
                for i in range(self.training_env.num_envs):
                    obs_i = new_obs[i]
                    features = observation_to_features(obs_i)
                    
                    # Mock LLM API call
                    advice = self.coach_client.get_coaching_advice(features)
                    forbidden = advice.get("Forbidden_Actions", [])
                    
                    
                    if self.verbose >= 1:
                        hits = advice.get('_cache_hits', 0)
                        misses = advice.get('_cache_misses', 0)
                        total = hits + misses
                        hit_rate = (hits / total * 100) if total > 0 else 0.0
                        lat_latest = advice.get('_latest_api_latency', 0.0)
                        lat_avg = advice.get('_avg_api_latency', 0.0)
                        print(f"[Coach Env {i}] Step {self.num_timesteps} - Role: {advice.get('Role')} | Masked: {forbidden} | Cache: {hit_rate:.1f}% ({hits}/{total}) | Latency: {lat_latest:.2f}s (Avg: {lat_avg:.2f}s)")
                        
                    # Stop training if we exceed a designated API budget
                    total_tokens = advice.get("_total_tokens_used", 0)
                    if self.max_token_budget and total_tokens > self.max_token_budget:
                        print(f"[Coach Env {i}] DeepSeek API Token Budget Exceeded: {total_tokens} > {self.max_token_budget}. Stopping training early.")
                        raise StopTrainingException(f"API Token Budget Exceeded: {total_tokens}")

                    # Target the specific environment with the generated mask
                    self.training_env.env_method("update_mask", forbidden, indices=i)
        return True

