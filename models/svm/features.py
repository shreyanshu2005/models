"""
models/svm/features.py
======================
25-dim feature builder for the SVM trend-classification model.

Feature set (all z-scored vs rolling 252-day window, except regime dims):
  Returns       : ret_1d, ret_5d, ret_10d, ret_20d                  (4)
  Trend         : sma10_20_ratio, sma20_50_ratio, ema20_slope_5d    (3)
  Momentum      : rsi_14, macd_signal, roc_10                       (3)
  Volatility    : rvol_20d, atr_14_norm, bb_width                   (3)
  Volume        : obv_5d_z, vol_ratio                               (2)
  Regime        : 5 one-hot + 5 HMM posteriors                     (10)
  ──────────────────────────────────────────────────────────────────
  Total                                                             25

Causality guarantee: every value at row t uses only data up to and
including bar t.  No future information leaks through any calculation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants (overridable via config dict)
# ─────────────────────────────────────────────────────────────────────
_TX_RATE      = 0.0011   # 0.11 % per leg (buy + sell both deducted)
_ZSCORE_WIN   = 252      # rolling z-score window (trading days)
_FWD_DAYS     = 5        # label: forward holding period in trading days
_EPS          = 1e-10    # numerical stability floor

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average — causal, min_periods=span//2."""
    return s.ewm(span=span, min_periods=span // 2, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n // 2).mean()


def _rolling_zscore(s: pd.Series, window: int = _ZSCORE_WIN) -> pd.Series:
    """Causal rolling z-score: (x - μ) / σ over trailing `window` bars."""
    mu  = s.rolling(window, min_periods=window // 2).mean()
    sig = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / (sig + _EPS)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(period, min_periods=period // 2).mean()
    loss   = (-delta.clip(upper=0)).rolling(period, min_periods=period // 2).mean()
    rs     = gain / (loss + _EPS)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period // 2).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Returns the MACD signal line (not the MACD line itself)."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    return _ema(macd_line, signal)


def _bollinger_width(close: pd.Series, window: int = 20) -> pd.Series:
    mid = _sma(close, window)
    std = close.rolling(window, min_periods=window // 2).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (upper - lower) / (mid + _EPS)


# ─────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────

def build_svm_features(
    ohlcv: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    tx_rate: float = _TX_RATE,
    zscore_win: int = _ZSCORE_WIN,
    fwd_days: int = _FWD_DAYS,
) -> pd.DataFrame:
    """
    Build the full 25-dim SVM feature matrix for one symbol.

    Parameters
    ----------
    ohlcv : DataFrame with columns [open, high, low, close, volume]
            indexed by date (DatetimeIndex), sorted ascending.
    regime_df : DataFrame with columns [regime_0..4, post_0..4]
                aligned to the same date index (output of regime engine).
                If None, regime dims are filled with 0.
    tx_rate : per-leg transaction cost rate.
    zscore_win : rolling window for z-scoring (trading days).
    fwd_days : forward periods for label computation.

    Returns
    -------
    DataFrame: feature matrix (25 cols + 'label' + 'label_raw_return').
    """
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── Returns (raw log-returns, then z-scored) ──────────────────────
    log_ret = np.log(close / close.shift(1))

    feat: Dict[str, pd.Series] = {}
    for lag in [1, 5, 10, 20]:
        raw = np.log(close / close.shift(lag))
        feat[f"ret_{lag}d"] = _rolling_zscore(raw, zscore_win)

    # ── Trend ─────────────────────────────────────────────────────────
    sma10 = _sma(close, 10)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    ema20 = _ema(close, 20)

    feat["sma10_20_ratio"] = _rolling_zscore(sma10 / (sma20 + _EPS), zscore_win)
    feat["sma20_50_ratio"] = _rolling_zscore(sma20 / (sma50 + _EPS), zscore_win)
    ema20_slope = (ema20 - ema20.shift(5)) / (ema20.shift(5) + _EPS)
    feat["ema20_slope_5d"] = _rolling_zscore(ema20_slope, zscore_win)

    # ── Momentum ──────────────────────────────────────────────────────
    feat["rsi_14"]      = _rolling_zscore(_rsi(close, 14), zscore_win)
    feat["macd_signal"] = _rolling_zscore(_macd(close), zscore_win)
    roc10 = (close - close.shift(10)) / (close.shift(10) + _EPS)
    feat["roc_10"]      = _rolling_zscore(roc10, zscore_win)

    # ── Volatility ────────────────────────────────────────────────────
    rvol20 = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    feat["rvol_20d"]   = _rolling_zscore(rvol20, zscore_win)

    atr14 = _atr(high, low, close, 14)
    feat["atr_14_norm"] = _rolling_zscore(atr14 / (close + _EPS), zscore_win)
    feat["bb_width"]    = _rolling_zscore(_bollinger_width(close, 20), zscore_win)

    # ── Volume ────────────────────────────────────────────────────────
    obv        = _obv(close, volume)
    obv_5d_chg = obv.diff(5)
    feat["obv_5d_z"] = _rolling_zscore(obv_5d_chg, zscore_win)  # already z-scored; NOT included in second pass

    vol_avg20 = volume.rolling(20, min_periods=10).mean()
    feat["vol_ratio"] = volume / (vol_avg20 + _EPS)              # bounded ratio — kept as-is, not re-z-scored

    # ── Regime features (10 dims) ─────────────────────────────────────
    regime_cols = [f"regime_{i}" for i in range(5)] + [f"post_{i}" for i in range(5)]
    if regime_df is not None:
        regime_aligned = regime_df.reindex(df.index).fillna(0)
        for col in regime_cols:
            feat[col] = regime_aligned.get(col, pd.Series(0, index=df.index))
    else:
        warnings.warn("No regime_df provided — regime features will be zero.", UserWarning)
        for col in regime_cols:
            feat[col] = pd.Series(0.0, index=df.index)

    # ── Assemble feature matrix ───────────────────────────────────────
    feature_df = pd.DataFrame(feat, index=df.index)

    # ── Label: 5-day forward net-of-costs log-return ──────────────────
    fwd_log_ret = np.log(close.shift(-fwd_days) / (close + _EPS))
    net_fwd     = fwd_log_ret - 2 * tx_rate      # deduct both buy AND sell legs

    feature_df["label_raw_return"] = net_fwd

    # Ternary label: +1 Buy, -1 Sell, 0 Flat
    threshold = 2 * tx_rate   # must overcome round-trip cost to be profitable
    label = pd.Series(0, index=df.index, dtype=int)
    label[net_fwd >  threshold] =  1
    label[net_fwd < -threshold] = -1
    feature_df["label"] = label

    # NaN the last `fwd_days` rows — no valid label
    feature_df.loc[feature_df.index[-fwd_days:], ["label", "label_raw_return"]] = np.nan

    logger.debug(
        "Built SVM features: shape=%s, label counts=%s",
        feature_df.shape,
        feature_df["label"].value_counts().to_dict(),
    )
    return feature_df


def get_feature_columns() -> List[str]:
    """Returns the ordered list of 25 SVM feature column names."""
    base = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "sma10_20_ratio", "sma20_50_ratio", "ema20_slope_5d",
        "rsi_14", "macd_signal", "roc_10",
        "rvol_20d", "atr_14_norm", "bb_width",
        "obv_5d_z", "vol_ratio",
    ]
    regime = [f"regime_{i}" for i in range(5)] + [f"post_{i}" for i in range(5)]
    return base + regime  # 15 + 10 = 25