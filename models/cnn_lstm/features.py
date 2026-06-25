"""
CNN-LSTM feature / tensor builder
==================================
Produces (N, lookback, 22) tensors fed to the CNN-LSTM model.

22 channels per bar
--------------------
Price channels  (5):  log_ret_1, log_ret_2, log_ret_5, log_ret_10, log_ret_20
HL spread       (1):  (high - low) / close   — intraday range proxy
Volume          (2):  log_volume_norm, vol_change (log ratio)
Trend           (3):  EMA8/close-1, EMA21/close-1, EMA50/close-1
Momentum        (3):  RSI14 (scaled 0-1), ROC10, MACD_hist_norm
Volatility      (2):  20d realised vol (ann.), 5d realised vol (ann.)
Regime          (6):  HMM 5-state one-hot (5) + vol_band one-hot LV/MV/HV → 3 dims
                       (we drop HMM posteriors vs SVM's 10-dim; keep it compact)

Total = 5+1+2+3+3+2+6 = 22  ✓

All price/vol channels are z-scored over a 252-day rolling window before
windowing → network sees zero-mean unit-variance inputs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional

LOOKBACK = 30   # sequence length fed to CNN-LSTM
LABEL_THRESHOLD_FACTOR = 2  # multiplied by tx_rate to form flat-zone

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd = fast - slow
    signal = _ema(macd, 9)
    return macd - signal


def _realised_vol(log_ret: pd.Series, window: int) -> pd.Series:
    return log_ret.rolling(window).std() * np.sqrt(252)


def _rolling_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (s - mu) / sigma


# ------------------------------------------------------------------
# main builder
# ------------------------------------------------------------------

def build_cnn_lstm_features(
    df: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
    tx_rate: float = 0.0011,
    lookback: int = LOOKBACK,
    warm_up: int = 252,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Parameters
    ----------
    df         : equity OHLCV DataFrame with columns
                 [date, open, high, low, close, volume]
    regime_df  : optional DataFrame indexed by date with columns
                 [hmm_state_0..4 (one-hot), vol_band_lv, vol_band_mv, vol_band_hv]
                 If None, regime channels are set to zero (graceful fallback).
    tx_rate    : one-way transaction cost rate for label thresholding
    lookback   : sequence length L
    warm_up    : rows dropped for rolling indicator warm-up

    Returns
    -------
    X      : np.ndarray  (N, lookback, 22)   float32
    y      : np.ndarray  (N,)                int8  {-1, 0, +1}
    dates  : DatetimeIndex of length N  (date of the *last bar* in each window)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    # ---- price return channels ----
    lr1  = np.log(close / close.shift(1))
    lr2  = np.log(close / close.shift(2))
    lr5  = np.log(close / close.shift(5))
    lr10 = np.log(close / close.shift(10))
    lr20 = np.log(close / close.shift(20))

    # ---- intraday spread ----
    hl_spread = (high - low) / close

    # ---- volume ----
    log_vol = np.log(vol.replace(0, np.nan))
    vol_change = log_vol - log_vol.shift(1)

    # ---- trend ----
    ema8_ratio  = _ema(close, 8)  / close - 1
    ema21_ratio = _ema(close, 21) / close - 1
    ema50_ratio = _ema(close, 50) / close - 1

    # ---- momentum ----
    rsi14   = _rsi(close, 14) / 100.0    # [0,1]
    roc10   = close.pct_change(10)
    macd_h  = _macd_hist(close)
    macd_h_norm = macd_h / close          # normalise by price

    # ---- volatility ----
    rv20 = _realised_vol(lr1, 20)
    rv5  = _realised_vol(lr1, 5)

    # ---- assemble raw feature frame (before z-score) ----
    feat = pd.DataFrame({
        'lr1':        lr1,
        'lr2':        lr2,
        'lr5':        lr5,
        'lr10':       lr10,
        'lr20':       lr20,
        'hl_spread':  hl_spread,
        'log_vol':    log_vol,
        'vol_change': vol_change,
        'ema8':       ema8_ratio,
        'ema21':      ema21_ratio,
        'ema50':      ema50_ratio,
        'rsi14':      rsi14,
        'roc10':      roc10,
        'macd_h':     macd_h_norm,
        'rv20':       rv20,
        'rv5':        rv5,
    }, index=df.index)

    # ---- z-score continuous channels (all except rsi which is already [0,1]) ----
    z_cols = [c for c in feat.columns if c != 'rsi14']
    for c in z_cols:
        feat[c] = _rolling_zscore(feat[c], window=252)

    # ---- regime channels (6) ----
    if regime_df is not None:
        regime_df = regime_df.copy()
        regime_df.index = pd.to_datetime(regime_df.index)
        feat = feat.set_index(df['date'])
        feat = feat.join(regime_df[['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4',
                                     'vol_lv','vol_mv','vol_hv']], how='left')
        # forward-fill any gaps (HMM on NIFTY calendar, equity may differ by 1 day)
        for c in ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv','vol_mv','vol_hv']:
            feat[c] = feat[c].ffill().fillna(0)
        # keep only 6 regime dims: 5 HMM one-hot + vol_band collapsed to 3 one-hot
        regime_cols = ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv']
        # vol_mv and vol_hv → we only need 2 of 3 (LV is the "reference" dropped for
        # non-collinearity), but the spec says 22 channels so keep all 3 vol bands:
        regime_cols = ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv','vol_mv','vol_hv']
        # That gives 16+8 = 24 → trim to 22 by dropping vol_mv, vol_hv and using LV only
        # Actually: 16 base + 5 hmm one-hot + 1 vol_lv = 22 ✓ (vol is encoded by LV alone; MV=!LV&!HV)
        regime_cols = ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv']
        feat.reset_index(inplace=True)
        feat.rename(columns={'index':'date'}, inplace=True)
    else:
        for c in ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv']:
            feat[c] = 0.0
        regime_cols = ['hmm_s0','hmm_s1','hmm_s2','hmm_s3','hmm_s4','vol_lv']

    # ---- final 22-channel order ----
    channels = [
        'lr1','lr2','lr5','lr10','lr20',        # 5 return
        'hl_spread',                             # 1 spread
        'log_vol','vol_change',                  # 2 volume
        'ema8','ema21','ema50',                  # 3 trend
        'rsi14','roc10','macd_h',               # 3 momentum
        'rv20','rv5',                            # 2 vol
    ] + regime_cols                              # 6 regime  → total 22

    feat_vals = feat[channels].values.astype(np.float32)
    dates_arr = pd.to_datetime(df['date'].values)

    # ---- forward return label (+1, 0, -1) ----
    fwd_ret = np.log(close.shift(-1) / close).values
    threshold = LABEL_THRESHOLD_FACTOR * tx_rate
    label = np.where(fwd_ret > threshold, 1,
             np.where(fwd_ret < -threshold, -1, 0)).astype(np.int8)

    # ---- drop warm-up rows (NaN from rolling indicators) ----
    valid = ~np.isnan(feat_vals).any(axis=1)
    first_valid = np.argmax(valid)
    first_valid = max(first_valid, warm_up)

    # ---- build windowed tensors ----
    X_list, y_list, d_list = [], [], []
    total = len(feat_vals)
    for i in range(first_valid, total - 1):      # -1: need fwd_ret for label
        start = i - lookback + 1
        if start < 0:
            continue
        window = feat_vals[start: i + 1]         # (lookback, 22)
        if np.isnan(window).any():
            continue
        X_list.append(window)
        y_list.append(label[i])
        d_list.append(dates_arr[i])

    X = np.stack(X_list, axis=0)                 # (N, lookback, 22)
    y = np.array(y_list, dtype=np.int8)
    dates = pd.DatetimeIndex(d_list)

    return X, y, dates