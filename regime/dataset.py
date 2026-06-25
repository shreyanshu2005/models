"""
models/cnn_lstm/dataset.py  (Week 8-9 update)
─────────────────────────────────────────────────────────────────────────────
QBEAST-AI.N  ·  CNN-LSTM tensor builder — live regime injection
Week 8-9 update: wires real regime channels into (batch, lookback, 22) tensors
                 replacing the placeholder zeros from Week 6-7.

22 channels per bar (spec §2):
  [0:5]   OHLCV (log-vol)
  [5]     SMA10 (z-scored)
  [6]     SMA20 (z-scored)
  [7]     RSI14 (scaled 0-1)
  [8]     ATR14/close
  [9]     MACD-diff (z-scored)
  [10]    Bollinger %B
  [11]    OBV-z
  [12]    volume-z
  [13]    (spare — zero; reserved for future)
  [14:19] regime one-hot OR posteriors (5 dims)  ← NOW LIVE
  [19:22] vol-band one-hot (3 dims)              ← NOW LIVE

Target: 5-day forward log-return (net of 0.11%×2 round-trip cost)
        winsorised at ±3σ per symbol on training set.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from regime.inject import inject_into_cnn_tensor

logger = logging.getLogger(__name__)

TX_RATE      = 0.0011
TOTAL_CH     = 22
LOOKBACK_DEF = 30
TARGET_DAYS  = 5     # 5-day forward return
WINSOR_SIGMA = 3.0


# ── feature computation ───────────────────────────────────────────────────────
def _zscore_rolling(series: pd.Series, window: int = 252) -> pd.Series:
    mu  = series.rolling(window, min_periods=20).mean()
    std = series.rolling(window, min_periods=20).std().replace(0, np.nan)
    return (series - mu) / std


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 13 base feature columns (channels 0-12, channel 13 = 0).
    Returns DataFrame with same index as df.
    df must have: open, high, low, close, volume (adj_close optional).
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    vol   = df["volume"].astype(float).replace(0, np.nan)

    # channels 0-4: OHLCV (log-normalised close, raw O/H/L ratios, log-vol)
    out["ch0_open"]   = np.log(df["open"].astype(float)  / close.replace(0, np.nan))
    out["ch1_high"]   = np.log(df["high"].astype(float)  / close.replace(0, np.nan))
    out["ch2_low"]    = np.log(df["low"].astype(float)   / close.replace(0, np.nan))
    out["ch3_close"]  = np.log(close / close.shift(1))
    out["ch4_vol"]    = _zscore_rolling(np.log(vol + 1))

    # channel 5: SMA10 z-scored
    sma10             = close.rolling(10).mean()
    out["ch5_sma10"]  = _zscore_rolling(close / sma10.replace(0, np.nan))

    # channel 6: SMA20 z-scored
    sma20             = close.rolling(20).mean()
    out["ch6_sma20"]  = _zscore_rolling(close / sma20.replace(0, np.nan))

    # channel 7: RSI14 scaled to [-1, 1]
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss.replace(0, np.nan)
    rsi14   = 100 - (100 / (1 + rs))
    out["ch7_rsi14"]  = (rsi14 / 50.0) - 1.0

    # channel 8: ATR14 / close
    hl     = df["high"].astype(float) - df["low"].astype(float)
    hc     = (df["high"].astype(float) - close.shift(1)).abs()
    lc     = (df["low"].astype(float)  - close.shift(1)).abs()
    tr     = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14  = tr.rolling(14).mean()
    out["ch8_atr14"]  = atr14 / close.replace(0, np.nan)

    # channel 9: MACD-diff z-scored
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    out["ch9_macd"]   = _zscore_rolling(macd - sig)

    # channel 10: Bollinger %B
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["ch10_bbpct"] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-8)
    out["ch10_bbpct"] = out["ch10_bbpct"].clip(-1, 2)

    # channel 11: OBV z-scored 5d change
    obv    = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    out["ch11_obv"]   = _zscore_rolling(obv.diff(5))

    # channel 12: volume z-scored
    out["ch12_volz"]  = _zscore_rolling(vol)

    # channel 13: spare / padding
    out["ch13_spare"] = 0.0

    return out


def build_target(
    df:             pd.DataFrame,
    tx_rate:        float = TX_RATE,
    winsor_sigma:   float = WINSOR_SIGMA,
    fit_winsor:     bool  = True,
    winsor_bounds:  Optional[Tuple[float, float]] = None,
) -> Tuple[pd.Series, Tuple[float, float]]:
    """
    5-day forward log-return net of round-trip tx cost.
    winsorised at ±winsor_sigma×std of training set.

    Returns (target_series, (lo_bound, hi_bound))
    """
    close       = df["close"].astype(float)
    fwd_ret     = np.log(close.shift(-TARGET_DAYS) / close.replace(0, np.nan))
    rt_cost     = 2 * tx_rate   # buy + sell
    fwd_net     = fwd_ret - rt_cost

    if fit_winsor:
        mu, sigma      = fwd_net.mean(), fwd_net.std()
        lo = mu - winsor_sigma * sigma
        hi = mu + winsor_sigma * sigma
    else:
        assert winsor_bounds is not None
        lo, hi = winsor_bounds

    fwd_clipped = fwd_net.clip(lo, hi)
    return fwd_clipped, (lo, hi)


# ── tensor builder ────────────────────────────────────────────────────────────
def build_cnn_tensor(
    df:             pd.DataFrame,
    regime_df:      pd.DataFrame,
    lookback:       int   = LOOKBACK_DEF,
    use_posteriors: bool  = True,
    target_days:    int   = TARGET_DAYS,
    tx_rate:        float = TX_RATE,
    fit_winsor:     bool  = True,
    winsor_bounds:  Optional[Tuple[float, float]] = None,
    sample_weight_recent_mult: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Build (X, y, w, winsor_bounds) where:
      X : (N, lookback, 22)  float32
      y : (N,)               float32  winsorised 5d net return
      w : (N,)               float32  sample weights (recent 2×)
      winsor_bounds : (lo, hi) tuple fitted on this df

    Regime channels 14-21 are filled with LIVE values from regime_df.

    Parameters
    ----------
    df         : price df for ONE symbol, sorted by date
    regime_df  : aligned to df by date
    lookback   : sequence length (default 30)
    use_posteriors : use HMM soft posteriors for channels 14-18
    """
    base = build_base_features(df)
    target_series, wb = build_target(
        df, tx_rate, WINSOR_SIGMA, fit_winsor, winsor_bounds
    )

    # align
    combined = base.join(target_series.rename("target"), how="left")
    combined = combined.join(
        regime_df[["hmm_state", "vol_band"] +
                   [f"hmm_post_{i}" for i in range(5)
                    if f"hmm_post_{i}" in regime_df.columns]],
        how="left"
    )
    combined["hmm_state"] = combined["hmm_state"].ffill().fillna(2).astype(int)
    combined["vol_band"]  = combined["vol_band"].ffill().fillna(1).astype(int)

    combined.dropna(subset=["target"], inplace=True)
    combined.dropna(subset=["ch3_close"], inplace=True)
    combined.fillna(0.0, inplace=True)

    # channel names in order
    base_cols = [f"ch{i}_{n}" for i, n in enumerate([
        "open","high","low","close","vol",
        "sma10","sma20","rsi14","atr14","macd","bbpct","obv","volz","spare"
    ])]

    # build raw tensor (regime channels = 0 initially, injected below)
    records = combined.reset_index(drop=True)
    T       = len(records)
    n_valid = 0

    X_list, y_list, idx_list = [], [], []

    for i in range(lookback, T):
        window_rows = records.iloc[i - lookback:i]
        x_base      = window_rows[base_cols].values.astype(np.float32)   # (lookback, 14)
        regime_pad  = np.zeros((lookback, 8), dtype=np.float32)          # channels 14-21
        x_full      = np.concatenate([x_base, regime_pad], axis=1)       # (lookback, 22)
        y_val       = float(records.iloc[i]["target"])
        X_list.append(x_full)
        y_list.append(y_val)
        idx_list.append(i)
        n_valid += 1

    if n_valid == 0:
        raise ValueError("No valid windows — check data length vs lookback.")

    X = np.stack(X_list, axis=0).astype(np.float32)    # (N, lookback, 22)
    y = np.array(y_list, dtype=np.float32)              # (N,)

    # ── live regime injection (Week 8-9 key step) ─────────────────────────
    # Build a sub-dataframe of regime values aligned to the N windows
    # (each window's regime = the regime at bar i, the target bar)
    window_regime = records.iloc[idx_list][
        ["hmm_state", "vol_band"] +
        [c for c in records.columns if c.startswith("hmm_post_")]
    ].reset_index(drop=True)

    X = inject_into_cnn_tensor(X, window_regime, use_posteriors=use_posteriors)

    # ── sample weights: 2× most recent N//4 bars ────────────────────────
    w = np.ones(n_valid, dtype=np.float32)
    recent_cutoff = max(0, n_valid - n_valid // 4)
    w[recent_cutoff:] *= sample_weight_recent_mult
    w /= w.sum() / n_valid   # normalise so sum = N

    logger.info(
        f"CNN tensor built: X={X.shape}, y mean={y.mean():.4f} std={y.std():.4f},"
        f" regime_nonzero={np.count_nonzero(X[:, 0, 14:22])}/{n_valid * 8}"
    )

    return X, y, w, wb


def pool_tensors(
    symbol_tensors: dict,   # {symbol: (X, y, w, wb)}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack tensors from all symbols for pooled cross-symbol training.
    Returns (X_pooled, y_pooled, w_pooled) — weights renormalised.
    """
    Xs, ys, ws = [], [], []
    for sym, (X, y, w, _) in symbol_tensors.items():
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        logger.info(f"  {sym}: {X.shape[0]} windows")

    X_pool = np.concatenate(Xs, axis=0)
    y_pool = np.concatenate(ys, axis=0)
    w_pool = np.concatenate(ws, axis=0)
    w_pool /= w_pool.sum() / len(w_pool)   # re-normalise after concat

    logger.info(f"Pooled tensor: X={X_pool.shape}")
    return X_pool, y_pool, w_pool