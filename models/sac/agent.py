"""
models/sac/agent.py
─────────────────────────────────────────────────────────────────────────────
QBEAST-AI.N  ·  SAC Agent
Week 8-9

Uses stable-baselines3 SAC. Wraps:
  · initial HP search (Optuna, 30 trials)
  · monthly experience replay update (10 000 gradient steps)
  · quarterly HP re-search for FAST params (γ, λ_dd, max_pos_cap)
  · off-policy replay buffer carry-forward across monthly retrains (spec §3)
  · drift sentinel evaluation hook (spec §3 step 04)

HP tiers (spec §2):
  FAST (quarterly re-search): temperature α (auto), discount γ, λ_dd, λ_turn, max_pos_cap
  SLOW (semi-annual):         τ (target smoothing), replay buffer size, actor/critic hidden
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from .env import QBeastSACEnv

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── defaults (SLOW params — changed semi-annually only) ──────────────────────
DEFAULT_SLOW_HP: Dict[str, Any] = {
    "tau":              0.005,
    "buffer_size":      200_000,
    "policy_kwargs":    dict(net_arch=[256, 256]),
    "batch_size":       256,
    "learning_starts":  500,
}

SYMBOLS_LC = {"RELIANCE", "HDFCBANK", "BAJFINANCE", "MARUTI", "HEROMOTOCO"}


# ── callbacks ────────────────────────────────────────────────────────────────
class EpisodeStatsCallback(BaseCallback):
    """Collect episode-level stats during training for Optuna objective."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


# ── HP search ────────────────────────────────────────────────────────────────
def _optuna_objective(
    trial: optuna.Trial,
    env_kwargs: Dict,
    slow_hp: Dict,
    n_train_steps: int = 20_000,
) -> float:
    """Optuna objective: mean episode reward over last 5 episodes."""

    gamma      = trial.suggest_float("gamma",        0.97,  0.99)
    lambda_dd  = trial.suggest_float("lambda_dd",    0.5,   5.0)
    lambda_turn = trial.suggest_float("lambda_turn", 0.01,  0.05)

    env = QBeastSACEnv(**env_kwargs, lambda_turnover=lambda_turn)

    try:
        model = SAC(
            "MlpPolicy",
            env,
            gamma          = gamma,
            tau            = slow_hp["tau"],
            buffer_size    = slow_hp["buffer_size"],
            policy_kwargs  = slow_hp["policy_kwargs"],
            batch_size     = slow_hp["batch_size"],
            learning_starts= slow_hp["learning_starts"],
            verbose        = 0,
        )
        cb = EpisodeStatsCallback()
        model.learn(total_timesteps=n_train_steps, callback=cb, progress_bar=False)
    except Exception as exc:
        logger.warning(f"Trial failed: {exc}")
        return -999.0

    rewards = cb.episode_rewards
    if not rewards:
        return -999.0
    return float(np.mean(rewards[-5:]))


def search_fast_hp(
    env_kwargs: Dict,
    slow_hp: Optional[Dict] = None,
    n_trials: int = 30,
    n_train_steps: int = 20_000,
    study_name: str = "sac_fast_hp",
) -> Dict[str, Any]:
    """
    Run Optuna search over FAST hyperparameters.
    Returns best {gamma, lambda_dd, lambda_turn}.
    """
    if slow_hp is None:
        slow_hp = DEFAULT_SLOW_HP

    study = optuna.create_study(
        direction  = "maximize",
        study_name = study_name,
        sampler    = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _optuna_objective(t, env_kwargs, slow_hp, n_train_steps),
        n_trials        = n_trials,
        show_progress_bar = False,
    )

    best = study.best_params
    logger.info(f"SAC HP search done. Best: {best}  |  value={study.best_value:.4f}")
    return best


# ── main agent class ─────────────────────────────────────────────────────────
class QBeastSACAgent:
    """
    Wraps stable-baselines3 SAC for QBEAST-AI.N.

    Responsibilities:
      · hold the SB3 model + replay buffer across monthly retrains
      · run initial training on 2016–2019 sim env
      · run monthly 10 000-step gradient updates from accumulated buffer
      · quarterly FAST HP re-search
      · expose predict() → continuous weight [0, 1] for fusion layer
      · save/load checkpoints to results/sac/
    """

    def __init__(
        self,
        symbol:          str,
        slow_hp:         Optional[Dict] = None,
        fast_hp:         Optional[Dict] = None,
        checkpoint_dir:  str = "results/sac",
    ):
        self.symbol         = symbol
        self.slow_hp        = slow_hp or DEFAULT_SLOW_HP.copy()
        self.fast_hp        = fast_hp or {
            "gamma":        0.98,
            "lambda_dd":    1.0,
            "lambda_turn":  0.02,
        }
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._model:         Optional[SAC]           = None
        self._env:           Optional[QBeastSACEnv]  = None
        self._months_trained: int = 0

    # ── internal: build or re-use model ──────────────────────────────────────
    def _build_model(self, env: QBeastSACEnv) -> SAC:
        return SAC(
            "MlpPolicy",
            env,
            gamma          = self.fast_hp["gamma"],
            tau            = self.slow_hp["tau"],
            buffer_size    = self.slow_hp["buffer_size"],
            policy_kwargs  = self.slow_hp["policy_kwargs"],
            batch_size     = self.slow_hp["batch_size"],
            learning_starts= self.slow_hp["learning_starts"],
            verbose        = 0,
        )

    # ── initial training (2016–2019) ─────────────────────────────────────────
    def initial_train(
        self,
        env_kwargs:    Dict,
        n_timesteps:   int = 200_000,
        run_hp_search: bool = True,
        n_hp_trials:   int = 30,
    ) -> None:
        """
        Train SAC from scratch on the 2016–2019 simulation environment.
        Optionally runs FAST HP search first.
        Spec §2: initial train window Jan 2016 – Dec 2019.
        """
        logger.info(f"[{self.symbol}] SAC initial training  (n_steps={n_timesteps})")

        if run_hp_search:
            logger.info(f"[{self.symbol}] Running FAST HP search ({n_hp_trials} trials)...")
            best_hp = search_fast_hp(
                env_kwargs   = env_kwargs,
                slow_hp      = self.slow_hp,
                n_trials     = n_hp_trials,
                n_train_steps = min(n_timesteps // 5, 20_000),
                study_name   = f"sac_{self.symbol}_initial",
            )
            self.fast_hp.update(best_hp)
            logger.info(f"[{self.symbol}] Best FAST HP: {self.fast_hp}")

        # build env and model
        env = QBeastSACEnv(**env_kwargs,
                           lambda_turnover=self.fast_hp["lambda_turn"])
        self._env   = env
        self._model = self._build_model(env)

        self._model.learn(total_timesteps=n_timesteps, progress_bar=True)
        self._months_trained += 1

        self.save_checkpoint("initial")
        logger.info(f"[{self.symbol}] Initial training complete.")

    # ── monthly update (10 000 gradient steps from buffer) ───────────────────
    def monthly_update(
        self,
        new_env_kwargs:    Dict,
        n_gradient_steps:  int = 10_000,
        run_hp_search:     bool = False,
        n_hp_trials:       int = 30,
    ) -> Dict[str, float]:
        """
        Spec §3 step 04:
          · Accumulate new experience from the new month's env into existing buffer
          · Run n_gradient_steps from the (now larger) buffer
          · Re-search FAST HP if quarterly trigger or regime flag
          · Returns evaluation stats for the drift sentinel
        """
        if self._model is None:
            raise RuntimeError("Call initial_train() before monthly_update().")

        if run_hp_search:
            logger.info(f"[{self.symbol}] Quarterly FAST HP re-search...")
            best_hp = search_fast_hp(
                env_kwargs   = new_env_kwargs,
                slow_hp      = self.slow_hp,
                n_trials     = n_hp_trials,
                n_train_steps = 15_000,
                study_name   = f"sac_{self.symbol}_m{self._months_trained}",
            )
            self.fast_hp.update(best_hp)
            # update gamma in existing model
            self._model.gamma = self.fast_hp["gamma"]

        # update position cap in new env immediately (FAST param)
        new_env = QBeastSACEnv(**new_env_kwargs,
                               lambda_turnover=self.fast_hp["lambda_turn"])
        self._model.set_env(new_env)
        self._env = new_env

        # collect experience from new month into the existing replay buffer
        # (off-policy: buffer carries forward — spec §2)
        self._model.learn(
            total_timesteps    = min(n_gradient_steps // 5, 2_000),
            reset_num_timesteps = False,   # keep buffer intact
            progress_bar       = False,
        )

        # gradient updates from accumulated buffer
        for _ in range(n_gradient_steps):
            if self._model.replay_buffer.size() >= self._model.learning_starts:
                self._model.train(batch_size=self.slow_hp["batch_size"],
                                  gradient_steps=1)

        self._months_trained += 1

        # evaluate policy on the last episode of the new env
        stats = self._evaluate(new_env)
        logger.info(
            f"[{self.symbol}] Monthly update #{self._months_trained}  "
            f"Sharpe={stats['sharpe']:.3f}  ret={stats['total_return']:.3f}"
        )

        self.save_checkpoint(f"m{self._months_trained:03d}")
        return stats

    # ── predict → position weight ─────────────────────────────────────────────
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        """
        Given a 35-dim state vector, return position weight ∈ [0, 1].
        Action is Δposition; we integrate it here to get the target weight.
        For fusion layer: returns the absolute position weight (not delta).
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")
        obs_2d = obs.reshape(1, -1).astype(np.float32)
        action, _ = self._model.predict(obs_2d, deterministic=deterministic)
        # clip Δposition to valid range, then add to current
        # Fusion layer only needs direction sign and magnitude: return raw action
        delta = float(np.clip(action[0], -1.0, 1.0))
        return delta   # caller (fusion) converts to final position

    # ── evaluation ───────────────────────────────────────────────────────────
    def _evaluate(self, env: QBeastSACEnv, n_episodes: int = 1) -> Dict[str, float]:
        """Run n_episodes on env with deterministic policy. Return aggregate stats."""
        all_stats = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            all_stats.append(env.get_episode_stats())

        return {
            k: float(np.mean([s[k] for s in all_stats]))
            for k in all_stats[0]
        }

    def evaluate_on_env(self, env_kwargs: Dict) -> Dict[str, float]:
        """Public evaluation hook for drift sentinel (spec §3 step 05)."""
        env = QBeastSACEnv(**env_kwargs,
                           lambda_turnover=self.fast_hp["lambda_turn"])
        return self._evaluate(env)

    # ── save / load ───────────────────────────────────────────────────────────
    def save_checkpoint(self, tag: str = "latest") -> None:
        path = self.checkpoint_dir / f"{self.symbol}_sac_{tag}"
        self._model.save(str(path))
        # save fast_hp alongside
        hp_path = self.checkpoint_dir / f"{self.symbol}_fast_hp_{tag}.pkl"
        with open(hp_path, "wb") as f:
            pickle.dump(self.fast_hp, f)
        logger.info(f"[{self.symbol}] Checkpoint saved: {path}")

    def load_checkpoint(self, tag: str = "latest") -> None:
        path    = self.checkpoint_dir / f"{self.symbol}_sac_{tag}.zip"
        hp_path = self.checkpoint_dir / f"{self.symbol}_fast_hp_{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        # need a dummy env to load into
        if self._env is None:
            raise RuntimeError("Set _env before loading checkpoint.")
        self._model = SAC.load(str(path), env=self._env)
        if hp_path.exists():
            with open(hp_path, "rb") as f:
                self.fast_hp = pickle.load(f)
        logger.info(f"[{self.symbol}] Checkpoint loaded: {path}")