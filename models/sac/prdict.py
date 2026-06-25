"""
models/sac/predict.py
─────────────────────────────────────────────────────────────────────────────
QBEAST-AI.N  ·  SAC predict interface for fusion layer
Week 8-9

Public API consumed by fusion/ensemble.py:
    sac_predict(symbol, obs, agent_registry) → (position_weight, direction_sign)

position_weight ∈ [0, 1]  — used as the size multiplier in fusion
direction_sign  ∈ {-1, 0, +1} — used as the direction vote in 2-of-3 rule
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .agent import QBeastSACAgent
from .env import REGIME_NAMES


def sac_predict(
    symbol:          str,
    obs:             np.ndarray,
    agent_registry:  Dict[str, QBeastSACAgent],
    current_pos:     float = 0.0,
) -> Tuple[float, int]:
    """
    Run inference for one symbol.

    Parameters
    ----------
    symbol          : NSE symbol string, e.g. "RELIANCE"
    obs             : 35-dim state vector (np.ndarray float32)
    agent_registry  : dict mapping symbol → QBeastSACAgent (one per symbol)
    current_pos     : current held position weight [0,1]

    Returns
    -------
    position_weight : float ∈ [0, 1]
        Target position as fraction of allocated capital.
        Used by fusion as the sizing multiplier: final_size = position_weight × regime_cap
    direction_sign  : int ∈ {-1, 0, +1}
        +1 if position_weight > 0.05 (going long)
         0 if abs(delta) ≤ 0.05    (flat / no change)
        -1 if position_weight < current_pos - 0.05 (reducing / exiting)
    """
    if symbol not in agent_registry:
        raise KeyError(f"No SAC agent registered for symbol '{symbol}'.")

    agent = agent_registry[symbol]
    delta = agent.predict(obs, deterministic=True)   # Δposition ∈ [-1, +1]

    # integrate delta → target position
    target_pos = float(np.clip(current_pos + delta, 0.0, 1.0))

    # derive direction vote for fusion
    if target_pos > current_pos + 0.05:
        direction_sign = 1
    elif target_pos < current_pos - 0.05:
        direction_sign = -1
    else:
        direction_sign = 0

    return target_pos, direction_sign


def build_sac_state(
    price_df,
    regime_df,
    svm_signals:   np.ndarray,
    cnn_forecasts: np.ndarray,
    t:             int,
    current_pos:   float,
    entry_price:   float | None,
    days_since_trade: int,
) -> np.ndarray:
    """
    Convenience builder for the 35-dim SAC state vector at bar t.
    Called from backtest/strategy.py on_bar event.

    This replicates QBeastSACEnv._build_state() but operates on raw arrays
    without instantiating a full env — used during live backtest inference.
    """
    closes = price_df["close"].values.astype(np.float64)

    # last 20 log-returns
    start = max(0, t - 20)
    vals  = closes[start:t + 1]
    if len(vals) >= 2:
        rets = np.log(vals[1:] / np.clip(vals[:-1], 1e-8, None))
    else:
        rets = np.array([], dtype=np.float64)
    ret_vec = np.zeros(20, dtype=np.float32)
    ret_vec[-len(rets):] = rets.astype(np.float32)

    # unrealised pnl %
    price = float(closes[t])
    if entry_price is not None and current_pos > 0.0:
        upnl = (price - entry_price) / (entry_price + 1e-8) * current_pos
    else:
        upnl = 0.0

    svm_sig  = float(svm_signals[t])
    cnn_fc   = float(cnn_forecasts[t])

    hmm_state = int(regime_df.iloc[t]["hmm_state"])
    vol_band  = int(regime_df.iloc[t]["vol_band"])

    reg_oh = np.zeros(5, dtype=np.float32)
    reg_oh[hmm_state] = 1.0
    vb_oh  = np.zeros(5, dtype=np.float32)
    vb_oh[vol_band] = 1.0

    days_norm = float(min(days_since_trade, 21)) / 21.0

    state = np.concatenate([
        ret_vec,
        np.array([current_pos, upnl, svm_sig, cnn_fc], dtype=np.float32),
        reg_oh,
        vb_oh,
        np.array([days_norm], dtype=np.float32),
    ]).astype(np.float32)

    return state