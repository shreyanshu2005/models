"""
regime/inject.py
─────────────────────────────────────────────────────────────────────────────
QBEAST-AI.N  ·  Live Regime Injection
Week 8-9

This is the first task of Week 8-9: wire the regime channels that were
placeholder-zeros in the CNN-LSTM tensor into real values from RegimeHMM.

Responsibilities
----------------
1. inject_into_cnn_tensor()
   Replaces channels [14:19] (regime one-hot 5 dims) and [19:22]
   (vol-band one-hot 3 dims) in the existing (T, 22) CNN-LSTM feature matrix
   with real values from regime_df.

2. inject_into_svm_features()
   Fills columns post_0..4 and reg_0..4 in the SVM 25-dim feature matrix
   (already done in Week 2-3, provided here for completeness / re-injection
   after HMM refit).

3. RegimeInjector class
   Stateful wrapper used by backtest/strategy.py on_bar — builds the
   regime feature slice for a given date without re-reading files.

Channel layout for CNN-LSTM 22-channel tensor (spec §2):
  [0]    open
  [1]    high
  [2]    low
  [3]    close
  [4]    volume (log-normalised)
  [5]    SMA10
  [6]    SMA20
  [7]    RSI14
  [8]    ATR14 / close
  [9]    MACD-diff
  [10]   Bollinger %B
  [11]   OBV-z
  [12]   volume-z
  [13]   (spare / padding)
  [14:19] regime one-hot  (5 dims)  ← ZEROS in Week 6-7, NOW LIVE
  [19:22] vol-band one-hot (3 dims) ← ZEROS in Week 6-7, NOW LIVE
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# channel indices
REGIME_ONEHOT_START  = 14   # channels 14-18 (5 dims)
REGIME_ONEHOT_END    = 19
VOLBAND_ONEHOT_START = 19   # channels 19-21 (3 dims)
VOLBAND_ONEHOT_END   = 22
TOTAL_CHANNELS       = 22


def inject_into_cnn_tensor(
    tensor:     np.ndarray,          # shape (T, lookback, 22)
    regime_df:  pd.DataFrame,        # per-bar: hmm_state, vol_band, optional hmm_post_0..4
    use_posteriors: bool = False,     # if True, replace one-hot with soft posteriors
) -> np.ndarray:
    """
    Fill regime and vol-band channels in the CNN-LSTM input tensor in-place.

    Parameters
    ----------
    tensor      : (T, lookback, 22)  — modified in-place and returned
    regime_df   : aligned to T rows  (same index as tensor's time axis)
                  Required columns: hmm_state (int), vol_band (int)
                  Optional columns: hmm_post_0 .. hmm_post_4 (floats, sum ~1)
    use_posteriors : if True and hmm_post_* columns present, use soft probs
                     instead of one-hot for channels 14-18
    Returns
    -------
    tensor : same array, mutated
    """
    T, lookback, C = tensor.shape
    assert C == TOTAL_CHANNELS, f"Expected 22 channels, got {C}"
    assert len(regime_df) == T, (
        f"regime_df length {len(regime_df)} != tensor T {T}"
    )

    hmm_states = regime_df["hmm_state"].values.astype(int)
    vol_bands  = regime_df["vol_band"].values.astype(int)

    has_posteriors = all(f"hmm_post_{i}" in regime_df.columns for i in range(5))

    for i in range(T):
        # regime channels: broadcast same value across the lookback window
        reg_vec = np.zeros(5, dtype=np.float32)
        if use_posteriors and has_posteriors:
            for j in range(5):
                reg_vec[j] = float(regime_df.iloc[i][f"hmm_post_{j}"])
        else:
            reg_vec[hmm_states[i]] = 1.0

        vb_vec = np.zeros(3, dtype=np.float32)
        vb_vec[vol_bands[i]] = 1.0

        # apply to all lookback steps within this bar's window
        tensor[i, :, REGIME_ONEHOT_START:REGIME_ONEHOT_END]   = reg_vec
        tensor[i, :, VOLBAND_ONEHOT_START:VOLBAND_ONEHOT_END] = vb_vec

    n_nonzero = np.count_nonzero(tensor[:, 0, REGIME_ONEHOT_START:VOLBAND_ONEHOT_END])
    logger.info(
        f"Regime injection complete: {n_nonzero}/{T * 8} regime+vol-band "
        f"cells non-zero. use_posteriors={use_posteriors}"
    )
    return tensor


def inject_into_svm_features(
    feature_df:  pd.DataFrame,
    regime_df:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill SVM regime columns (reg_0..4 one-hot + post_0..4 posteriors) in-place.
    regime_df must be aligned to feature_df by date index.
    Returns updated feature_df.
    """
    aligned = regime_df.reindex(feature_df.index, method="ffill")

    # one-hot
    for s in range(5):
        feature_df[f"reg_{s}"] = (aligned["hmm_state"] == s).astype(np.float32)

    # posteriors (if available)
    for s in range(5):
        col = f"hmm_post_{s}"
        if col in aligned.columns:
            feature_df[f"post_{s}"] = aligned[col].astype(np.float32)
        else:
            feature_df[f"post_{s}"] = feature_df[f"reg_{s}"]

    logger.info(
        f"SVM regime injection: {feature_df[['reg_0','reg_1','reg_2','reg_3','reg_4']].sum().to_dict()}"
    )
    return feature_df


# ── stateful injector for live backtest ──────────────────────────────────────
class RegimeInjector:
    """
    Lightweight stateful wrapper for on_bar use in backtest/strategy.py.
    Pre-loads regime_df at construction time; provides O(1) lookups per bar.

    Usage
    -----
    injector = RegimeInjector(regime_df)
    # on each bar:
    regime_vec = injector.get_regime_vec(date)   # 10-dim: 5 one-hot + 5 posterior
    vol_vec    = injector.get_vol_vec(date)       # 3-dim one-hot
    """

    def __init__(self, regime_df: pd.DataFrame):
        """
        regime_df index: DatetimeIndex (or string dates).
        Required columns: hmm_state (int 0-4), vol_band (int 0-2)
        Optional columns: hmm_post_0 .. hmm_post_4
        """
        self._df = regime_df.copy()
        self._has_posteriors = all(
            f"hmm_post_{i}" in self._df.columns for i in range(5)
        )
        # build fast lookup dict: date-str → row dict
        self._lookup: dict = {}
        for idx, row in self._df.iterrows():
            key = str(idx)[:10]
            self._lookup[key] = row

    def get(self, date) -> Optional[pd.Series]:
        key = str(date)[:10]
        return self._lookup.get(key, None)

    def get_regime_vec(self, date) -> np.ndarray:
        """Return 10-dim vector: 5-dim one-hot + 5-dim HMM posteriors."""
        row = self.get(date)
        if row is None:
            return np.zeros(10, dtype=np.float32)

        # one-hot
        onehot = np.zeros(5, dtype=np.float32)
        onehot[int(row["hmm_state"])] = 1.0

        # posteriors
        if self._has_posteriors:
            post = np.array(
                [float(row[f"hmm_post_{i}"]) for i in range(5)],
                dtype=np.float32
            )
        else:
            post = onehot.copy()

        return np.concatenate([onehot, post])

    def get_vol_vec(self, date) -> np.ndarray:
        """Return 3-dim one-hot vol-band vector."""
        row = self.get(date)
        if row is None:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # default MV
        vb = np.zeros(3, dtype=np.float32)
        vb[int(row["vol_band"])] = 1.0
        return vb

    def get_hmm_state(self, date) -> int:
        row = self.get(date)
        if row is None:
            return 2   # Sideways default
        return int(row["hmm_state"])

    def get_vol_band(self, date) -> int:
        row = self.get(date)
        if row is None:
            return 1   # MV default
        return int(row["vol_band"])

    def get_lambda_dd(self, date) -> float:
        """Regime-scaled λ_dd scalar for SAC reward (spec §2)."""
        from models.sac.env import LAMBDA_DD_MAP
        return LAMBDA_DD_MAP.get(self.get_hmm_state(date), 1.0)

    def is_crashing(self, date) -> bool:
        return self.get_hmm_state(date) == 4