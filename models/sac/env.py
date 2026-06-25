"""
models/sac/env.py
─────────────────────────────────────────────────────────────────────────────
QBEAST-AI.N  ·  SAC Trading Environment
Week 8-9  ·  NautilusTrader-compatible Gym Env wrapper

State  : ~35 dims per symbol (spec §2)
Action : Δposition ∈ [−1, +1] → clipped to [0, 1] for long-only equity
Reward : r_t = (pnl_t − tx_cost_t) − λ_dd × max_drawdown_t − λ_turn × |Δpos_t|
Costs  : 0.11% deducted on BOTH buy and sell legs in the environment step
         (NOT as post-hoc adjustment — spec §2 & §4)

Position caps (spec §2):
  LC symbols (RELIANCE/HDFCBANK/BAJFINANCE/MARUTI/HEROMOTOCO) → up to 0.30
  MC symbols (ABB/BOSCHLTD/MOTHERSON/BRITANNIA/INDIGO)        → up to 0.20
  HV vol-band → halved cap

λ_dd regime scaling (spec §2):
  Rising  : 0.5   Rally   : 1.0
  Sideways: 1.2   Falling : 2.0   Crashing: 5.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any

# ── constants ────────────────────────────────────────────────────────────────
TX_RATE        = 0.0011          # 0.11% per leg, both buy and sell
STATE_DIM      = 35              # spec §2: ~35 dims per symbol
ACTION_DIM     = 1               # scalar Δposition

LARGE_CAP = {"RELIANCE", "HDFCBANK", "BAJFINANCE", "MARUTI", "HEROMOTOCO"}
SMALL_CAP = {"ABB", "BOSCHLTD", "MOTHERSON", "BRITANNIA", "INDIGO"}

# HMM state index → name (ordered by mean log-return ascending from fit)
REGIME_NAMES   = {0: "Rising", 1: "Rally", 2: "Sideways", 3: "Falling", 4: "Crashing"}

# λ_dd regime scalar  (spec §2)
LAMBDA_DD_MAP  = {0: 0.5, 1: 1.0, 2: 1.2, 3: 2.0, 4: 5.0}

# vol-band index: 0=LV, 1=MV, 2=HV
VOL_BAND_HV    = 2


# ── environment ──────────────────────────────────────────────────────────────
class QBeastSACEnv(gym.Env):
    """
    Single-symbol EOD trading environment.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain columns: date, open, high, low, close, volume
        Index: integer, sorted ascending by date.
    regime_df : pd.DataFrame
        Per-day columns: hmm_state (int 0-4), hmm_post_0..4 (float),
        vol_band (int 0-2), vol_band_lv, vol_band_mv, vol_band_hv (float 0/1).
        Aligned to price_df by date.
    svm_signals : np.ndarray  shape (T,)  values in {-1, 0, 1}
    cnn_forecasts : np.ndarray  shape (T,)  5d return forecast (float)
    symbol : str
    initial_capital : float  (total portfolio value, position is a fraction)
    lambda_turnover : float
    seed : int
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_df,
        regime_df,
        svm_signals: np.ndarray,
        cnn_forecasts: np.ndarray,
        symbol: str,
        initial_capital: float = 10_000_000.0,
        lambda_turnover: float  = 0.02,
        seed: int = 42,
    ):
        super().__init__()
        self.price_df        = price_df.reset_index(drop=True)
        self.regime_df       = regime_df.reset_index(drop=True)
        self.svm_signals     = svm_signals.astype(np.float32)
        self.cnn_forecasts   = cnn_forecasts.astype(np.float32)
        self.symbol          = symbol
        self.initial_capital = float(initial_capital)
        self.lambda_turnover = float(lambda_turnover)

        # ── spaces ──────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        # raw action: Δposition ∈ [−1, +1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self._rng = np.random.default_rng(seed)
        self._n   = len(self.price_df)
        self.reset()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _position_cap(self, vol_band: int) -> float:
        base = 0.30 if self.symbol in LARGE_CAP else 0.20
        if vol_band == VOL_BAND_HV:
            base = base * 0.5
        return base

    def _lambda_dd(self, hmm_state: int) -> float:
        return LAMBDA_DD_MAP.get(int(hmm_state), 1.0)

    def _close(self, t: int) -> float:
        return float(self.price_df.loc[t, "close"])

    def _get_returns(self, t: int, window: int = 20) -> np.ndarray:
        """Last `window` daily log-returns up to and including t."""
        closes = self.price_df["close"].values
        start  = max(0, t - window)
        vals   = closes[start : t + 1]
        if len(vals) < 2:
            return np.zeros(window, dtype=np.float32)
        rets = np.log(vals[1:] / np.clip(vals[:-1], 1e-8, None))
        # left-pad with zeros if we don't have a full window
        out = np.zeros(window, dtype=np.float32)
        out[-len(rets):] = rets
        return out.astype(np.float32)

    def _build_state(self, t: int) -> np.ndarray:
        """
        Build 35-dim state vector (spec §2):
          [0:20]  last 20 daily log-returns
          [20]    current_position (fraction of cap)
          [21]    unrealised PnL %
          [22]    SVM signal {-1,0,1} normalised to [-1,1]
          [23]    CNN-LSTM 5d return forecast (raw float)
          [24:29] regime one-hot (5 dims)
          [29:34] vol-band one-hot (3 dims) — padded to 5 for alignment
          [34]    days since last trade (normalised / 21)
        Total: 20 + 1 + 1 + 1 + 1 + 5 + 5 + 1 = 35
        """
        rets       = self._get_returns(t, 20)                      # 20 dims
        pos        = np.float32(self._current_pos)                 # 1
        upnl_pct   = np.float32(self._unrealised_pnl_pct(t))      # 1
        svm_sig    = np.float32(self.svm_signals[t])               # 1
        cnn_fc     = np.float32(self.cnn_forecasts[t])             # 1

        # regime one-hot 5 dims
        hmm_state  = int(self.regime_df.loc[t, "hmm_state"])
        reg_onehot = np.zeros(5, dtype=np.float32)
        reg_onehot[hmm_state] = 1.0                                # 5

        # vol-band one-hot 3 dims, padded to 5
        vb         = int(self.regime_df.loc[t, "vol_band"])
        vb_onehot  = np.zeros(5, dtype=np.float32)
        vb_onehot[vb] = 1.0                                        # 5 (last 2 always 0)

        days_norm  = np.float32(
            min(self._days_since_trade, 21) / 21.0
        )                                                           # 1

        state = np.concatenate([
            rets,
            [pos, upnl_pct, svm_sig, cnn_fc],
            reg_onehot,
            vb_onehot,
            [days_norm],
        ]).astype(np.float32)

        assert state.shape == (STATE_DIM,), f"State dim mismatch: {state.shape}"
        return state

    def _unrealised_pnl_pct(self, t: int) -> float:
        if self._entry_price is None or self._current_pos == 0.0:
            return 0.0
        cur_price = self._close(t)
        pnl_pct   = (cur_price - self._entry_price) / (self._entry_price + 1e-8)
        return float(pnl_pct * self._current_pos)

    def _tx_cost(self, price: float, delta_shares: float) -> float:
        """0.11% on both legs when |delta_shares| > 0."""
        return abs(delta_shares) * price * TX_RATE

    # ── gym interface ────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # start after warm-up (need at least 20 bars for returns)
        self._t               = 20
        self._current_pos     = 0.0      # fraction of capital [0, 1]
        self._entry_price     = None
        self._capital         = self.initial_capital
        self._peak_nav        = self.initial_capital
        self._realised_pnl    = 0.0
        self._cum_tx_cost     = 0.0
        self._days_since_trade = 0
        self._episode_returns = []

        obs  = self._build_state(self._t)
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        t          = self._t
        price      = self._close(t)
        hmm_state  = int(self.regime_df.loc[t, "hmm_state"])
        vol_band   = int(self.regime_df.loc[t, "vol_band"])

        # ── action → target position ─────────────────────────────────────
        delta_pos     = float(np.clip(action[0], -1.0, 1.0))
        pos_cap       = self._position_cap(vol_band)

        # Crashing regime: force flat (spec §2 override)
        if hmm_state == 4:
            new_pos = 0.0
        else:
            new_pos = float(np.clip(self._current_pos + delta_pos, 0.0, pos_cap))

        actual_delta  = new_pos - self._current_pos

        # ── transaction cost (both legs, in-loop) ────────────────────────
        capital_deployed = self._capital
        delta_shares     = actual_delta * capital_deployed / (price + 1e-8)
        tx_cost          = self._tx_cost(price, delta_shares)
        self._cum_tx_cost += tx_cost

        # ── pnl from position change ──────────────────────────────────────
        # Realise gain/loss on the portion closed (if reducing/closing pos)
        if actual_delta < 0 and self._entry_price is not None:
            closed_frac  = abs(actual_delta)
            realised_pnl = (
                (price - self._entry_price) * closed_frac * capital_deployed
                / (self._entry_price + 1e-8)
            ) - tx_cost
            self._realised_pnl += realised_pnl
        else:
            realised_pnl = -tx_cost   # only cost on entry

        # update entry price (VWAP-style average on adds)
        if actual_delta > 0:
            if self._entry_price is None or self._current_pos == 0.0:
                self._entry_price = price
            else:
                total = self._current_pos + actual_delta
                self._entry_price = (
                    self._entry_price * self._current_pos + price * actual_delta
                ) / (total + 1e-8)

        if new_pos == 0.0:
            self._entry_price = None

        self._current_pos = new_pos

        # ── NAV update ────────────────────────────────────────────────────
        # Unrealised portion: mark-to-market on open position
        if self._entry_price is not None and self._current_pos > 0:
            unrealised = (
                (price - self._entry_price)
                * self._current_pos * capital_deployed
                / (self._entry_price + 1e-8)
            )
        else:
            unrealised = 0.0

        nav         = capital_deployed + realised_pnl + unrealised
        self._capital = nav
        self._peak_nav = max(self._peak_nav, nav)

        # ── drawdown ──────────────────────────────────────────────────────
        max_dd     = (self._peak_nav - nav) / (self._peak_nav + 1e-8)
        lambda_dd  = self._lambda_dd(hmm_state)

        # ── reward ────────────────────────────────────────────────────────
        pnl_t = realised_pnl / (capital_deployed + 1e-8)
        reward = (
            pnl_t
            - (tx_cost / (capital_deployed + 1e-8))
            - lambda_dd * max_dd
            - self.lambda_turnover * abs(actual_delta)
        )

        # track daily return for episode stats
        daily_ret = (nav - capital_deployed) / (capital_deployed + 1e-8)
        self._episode_returns.append(daily_ret)

        # days since trade
        if abs(actual_delta) > 1e-6:
            self._days_since_trade = 0
        else:
            self._days_since_trade += 1

        # ── advance timestep ──────────────────────────────────────────────
        self._t += 1
        terminated = self._t >= self._n - 1
        truncated  = False

        if terminated:
            obs = self._build_state(self._n - 1)
        else:
            obs = self._build_state(self._t)

        info = {
            "t":               t,
            "price":           price,
            "position":        self._current_pos,
            "nav":             nav,
            "realised_pnl":    self._realised_pnl,
            "cum_tx_cost":     self._cum_tx_cost,
            "max_dd":          max_dd,
            "hmm_state":       hmm_state,
            "regime":          REGIME_NAMES.get(hmm_state, "Unknown"),
            "vol_band":        vol_band,
            "lambda_dd":       lambda_dd,
        }

        return obs, float(reward), terminated, truncated, info

    def get_episode_stats(self) -> Dict[str, float]:
        """Return Sharpe, total return, max drawdown for this episode."""
        rets = np.array(self._episode_returns)
        if len(rets) < 2:
            return {"sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0}
        rf_daily  = 0.065 / 252
        excess    = rets - rf_daily
        sharpe    = (np.mean(excess) * np.sqrt(252)) / (np.std(rets) + 1e-8)
        cum       = np.cumprod(1 + rets)
        peak      = np.maximum.accumulate(cum)
        dd        = np.max((peak - cum) / (peak + 1e-8))
        return {
            "sharpe":       float(sharpe),
            "total_return": float(cum[-1] - 1.0),
            "max_dd":       float(dd),
        }