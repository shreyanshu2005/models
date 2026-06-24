"""
QBEAST-AI.N  ·  data/feature_engineer.py
==========================================
Feature engineering pipeline.

Produces
--------
1. SVM feature matrix      — (T, 25) DataFrame per symbol, z-scored, no NaN.
2. CNN-LSTM channel tensor — (T, lookback, 22) NumPy array per symbol.
3. SAC state components    — return history, unrealised PnL stub, vol-band.
4. Target labels           — binary {-1, 0, +1} for SVM, 5-day forward
                              log-return (net of costs) for CNN-LSTM.

All computation is strictly causal (no look-ahead).

Regime features (10 dims) are injected from the regime module after the
HMM is trained.  Call `inject_regime_features()` to merge them in.

Usage
-----
    from data.feature_engineer import FeatureEngineer
    fe = FeatureEngineer(config)
    raw_features = fe.build_features(equity_df)          # per symbol
    label_df     = fe.build_svm_labels(equity_df)        # SVM targets
    cnn_tensor, cnn_targets = fe.build_cnn_tensors(equity_df, lookback=30)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TX_RATE = 0.0011   # 0.11% per leg — used for net-of-cost targets


class FeatureEngineer:
    """
    Stateless feature builder.  Pass the config dict at construction;
    each method operates on a single-symbol OHLCV DataFrame.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    """

    def __init__(self, config: dict):
        self.cfg          = config
        self.tx_rate      = config["costs"]["tx_rate_per_leg"]
        self.zscore_win   = config["features"]["zscore_window"]
        self.target_h     = config["features"]["cnn_lstm_target_horizon"]

    # ════════════════════════════════════════════════════════════════════════
    # 1.  Raw technical features  (SVM input — 15 base + 10 regime = 25 total)
    # ════════════════════════════════════════════════════════════════════════

    def build_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 15 base technical features per symbol.
        Regime features (10 dims) are added later via inject_regime_features().

        Inputs:  OHLCV DataFrame with DatetimeIndex.
        Returns: DataFrame same index, 15 columns, z-scored where appropriate.
                 NaN rows at the head (warm-up) are kept; caller should dropna.
        """
        feat = pd.DataFrame(index=df.index)

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── Returns ────────────────────────────────────────────────────────
        feat["ret_1d"]  = np.log(close / close.shift(1))
        feat["ret_5d"]  = np.log(close / close.shift(5))
        feat["ret_10d"] = np.log(close / close.shift(10))
        feat["ret_20d"] = np.log(close / close.shift(20))

        # ── Trend ──────────────────────────────────────────────────────────
        sma10  = close.rolling(10).mean()
        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        ema20  = close.ewm(span=20, adjust=False).mean()

        feat["sma10_20_ratio"] = sma10 / sma20 - 1.0
        feat["sma20_50_ratio"] = sma20 / sma50 - 1.0
        feat["ema20_slope_5d"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)

        # ── Momentum ───────────────────────────────────────────────────────
        feat["rsi_14"]     = self._rsi(close, 14)
        feat["macd_signal"] = self._macd_signal(close)
        feat["roc_10"]     = close / close.shift(10) - 1.0

        # ── Volatility ─────────────────────────────────────────────────────
        feat["rvol_20d"]   = feat["ret_1d"].rolling(20).std() * np.sqrt(252)
        feat["atr_14_norm"] = self._atr(high, low, close, 14) / (close + 1e-9)
        bb_upper, bb_lower = self._bollinger_bands(close, 20, 2)
        feat["bb_width"]   = (bb_upper - bb_lower) / (sma20 + 1e-9)

        # ── Volume ─────────────────────────────────────────────────────────
        obv             = self._obv(close, volume)
        feat["obv_5d_z"] = self._rolling_zscore(obv.diff(5), self.zscore_win)
        vol_avg20       = volume.rolling(20).mean()
        feat["vol_ratio"] = volume / (vol_avg20 + 1e-9)

        # ── Z-score continuous features ────────────────────────────────────
        # RSI is already bounded [0,100]; normalise to [−1, +1]
        feat["rsi_14"] = (feat["rsi_14"] - 50.0) / 50.0

        # Rolling z-score the unbounded features
        _to_zscore = [
            "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "sma10_20_ratio", "sma20_50_ratio", "ema20_slope_5d",
            "macd_signal", "roc_10", "rvol_20d", "atr_14_norm",
            "bb_width", "vol_ratio",
        ]
        for col in _to_zscore:
            feat[col] = self._rolling_zscore(feat[col], self.zscore_win)

        logger.debug("Built %d base features", len(feat.columns))
        feat = feat.ffill().bfill()   # fill NaNs from alignment gaps (≤8 rows)
        return feat

    # ════════════════════════════════════════════════════════════════════════
    # 2.  Regime feature injection
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def inject_regime_features(
        feat_df: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge regime one-hot (5 dims) + HMM posterior (5 dims) = 10 dims
        into the feature DataFrame.

        Parameters
        ----------
        feat_df    : output of build_raw_features()
        regime_df  : DataFrame with columns
                       regime_0..4 (one-hot) and
                       post_0..4   (soft posterior probs)

        Returns
        -------
        pd.DataFrame with 25 columns (15 base + 10 regime).
        """
        regime_cols = [f"regime_{i}" for i in range(5)] + \
                      [f"post_{i}"   for i in range(5)]
        available   = [c for c in regime_cols if c in regime_df.columns]
        merged = feat_df.join(regime_df[available], how="left")
        for col in available:
            merged[col] = merged[col].ffill()
        logger.debug(
            "Injected %d regime cols → total features = %d",
            len(available), len(merged.columns)
        )
        return merged

    # ════════════════════════════════════════════════════════════════════════
    # 3.  SVM target labels  {-1, 0, +1}
    # ════════════════════════════════════════════════════════════════════════

    def build_svm_labels(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold_pct: float = 0.005,
    ) -> pd.Series:
        """
        Ternary labels for SVM classification.

        Label at time t is based on the net-of-cost 5-day forward return:
          +1  if return >  +threshold  (Buy)
           0  if |return| <= threshold (Flat)
          -1  if return <  -threshold  (Sell)

        Both buy and sell cost legs are deducted from the raw return.
        """
        close = df["close"]
        fwd_raw   = np.log(close.shift(-forward_days) / close)   # look-ahead intentional (it's the label)
        # Net of costs: deduct buy leg (entry) + sell leg (exit)
        fwd_net   = fwd_raw - 2 * self.tx_rate

        labels = pd.Series(0, index=df.index, name="svm_label", dtype=np.int8)
        labels[fwd_net >  threshold_pct] =  1
        labels[fwd_net < -threshold_pct] = -1

        # Last `forward_days` rows have NaN future — mark as NaN
        labels.iloc[-forward_days:] = np.nan
        return labels

    # ════════════════════════════════════════════════════════════════════════
    # 4.  CNN-LSTM channel tensors
    # ════════════════════════════════════════════════════════════════════════

    def build_cnn_channels(
        self,
        df: pd.DataFrame,
        regime_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build the 22-channel DataFrame (vol-normalised) for the CNN-LSTM.

        Channels (22 total):
          OHLCV (5) + SMA10, SMA20, RSI14, ATR14, MACD-diff, BB-%B,
          OBV-z, volume-z  (8) + regime one-hot (5) + vol-band one-hot (3)

        Each channel is vol-normalised: divided by the rolling 20d realised vol.
        """
        ch = pd.DataFrame(index=df.index)
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        log_ret = np.log(close / close.shift(1))
        rvol20  = log_ret.rolling(20).std().replace(0, np.nan).ffill()

        def vol_norm(s: pd.Series) -> pd.Series:
            return s / (rvol20 + 1e-9)

        # ── OHLCV (vol-normalised returns / ratios) ────────────────────────
        ch["ch_ret_1d"]      = vol_norm(log_ret)
        ch["ch_high_ret"]    = vol_norm(np.log(high  / close.shift(1)))
        ch["ch_low_ret"]     = vol_norm(np.log(low   / close.shift(1)))
        ch["ch_open_ret"]    = vol_norm(np.log(df["open"] / close.shift(1)))
        ch["ch_volume_z"]    = self._rolling_zscore(np.log1p(volume), 20)

        # ── Trend channels ─────────────────────────────────────────────────
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        ch["ch_sma10"] = vol_norm((close - sma10) / (sma10 + 1e-9))
        ch["ch_sma20"] = vol_norm((close - sma20) / (sma20 + 1e-9))

        # ── Momentum / oscillator channels ─────────────────────────────────
        ch["ch_rsi14"]     = (self._rsi(close, 14) - 50.0) / 50.0
        ch["ch_atr14"]     = vol_norm(self._atr(high, low, close, 14) / (close + 1e-9))
        ch["ch_macd_diff"] = vol_norm(self._macd_diff(close))
        bb_upper, bb_lower = self._bollinger_bands(close, 20, 2)
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        ch["ch_bb_pctB"]   = (close - bb_lower) / (bb_range + 1e-9)

        # ── Volume channel ─────────────────────────────────────────────────
        obv = self._obv(close, volume)
        ch["ch_obv_z"]     = self._rolling_zscore(obv, 252)
        vol_avg20          = volume.rolling(20).mean()
        ch["ch_vol_ratio"] = self._rolling_zscore(volume / (vol_avg20 + 1e-9), 252)
        ch["ch_rvol20"]    = self._rolling_zscore(rvol20, 252)

        # ── Regime one-hot (5 dims) ────────────────────────────────────────
        for i in range(5):
            col = f"regime_{i}"
            ch[f"ch_reg_{i}"] = regime_df[col].reindex(df.index).ffill() \
                if (regime_df is not None and col in regime_df.columns) \
                else 0.0

        # ── Vol-band one-hot (3 dims: LV, MV, HV) ─────────────────────────
        for band, col_name in enumerate(["ch_vb_lv", "ch_vb_mv", "ch_vb_hv"]):
            src = f"vol_band_{band}"
            ch[col_name] = regime_df[src].reindex(df.index).ffill() \
                if (regime_df is not None and src in regime_df.columns) \
                else (1.0 if band == 1 else 0.0)

        assert len(ch.columns) == 22, f"Expected 22 channels, got {len(ch.columns)}"
        return ch

    def build_cnn_tensors(
        self,
        df: pd.DataFrame,
        lookback: int = 30,
        regime_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Slide a lookback window over the channel DataFrame to produce
        (N, lookback, 22) input tensor and (N,) target return array.

        Target: 5-day forward log-return, net of 2-leg costs,
                winsorised at ±3 sigma.

        Returns
        -------
        X      : np.ndarray, shape (N, lookback, 22)
        y      : np.ndarray, shape (N,)  — net-of-cost 5d forward return
        dates  : pd.DatetimeIndex of length N (date of the last bar in each window)
        """
        channels = self.build_cnn_channels(df, regime_df=regime_df)
        channels = channels.ffill().bfill()   # fill any residual NaNs

        close = df["close"].reindex(channels.index)
        fwd_ret = np.log(close.shift(-self.target_h) / close) - 2 * self.tx_rate

        ch_arr = channels.to_numpy(dtype=np.float32)
        fwd_arr = fwd_ret.to_numpy(dtype=np.float32)

        # Replace NaN / Inf
        ch_arr  = np.nan_to_num(ch_arr, nan=0.0, posinf=0.0, neginf=0.0)
        fwd_arr = np.nan_to_num(fwd_arr, nan=0.0)

        # Winsorise targets at ±3σ
        sigma = np.nanstd(fwd_arr[~np.isnan(fwd_arr)])
        fwd_arr = np.clip(fwd_arr, -3 * sigma, 3 * sigma)

        T  = len(ch_arr)
        N  = T - lookback - self.target_h + 1
        if N <= 0:
            raise ValueError(
                f"Not enough data to build tensors: T={T}, lookback={lookback}, "
                f"target_h={self.target_h}"
            )

        X     = np.stack([ch_arr[i : i + lookback] for i in range(N)])
        y     = fwd_arr[lookback - 1 : lookback - 1 + N]
        dates = channels.index[lookback - 1 : lookback - 1 + N]

        logger.info(
            "CNN tensors built: X=%s  y=%s  lookback=%d",
            X.shape, y.shape, lookback
        )
        return X, y, dates

    # ════════════════════════════════════════════════════════════════════════
    # 5.  SAC state builder
    # ════════════════════════════════════════════════════════════════════════

    def build_sac_state(
        self,
        df: pd.DataFrame,
        t: int,
        current_position: float = 0.0,
        unrealised_pnl_pct: float = 0.0,
        svm_signal: float = 0.0,
        cnn_return_forecast: float = 0.0,
        days_since_trade: int = 0,
        time_in_position: int = 0,
        regime_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Build the SAC state vector at time step t (~35 dims).

        Layout
        ------
        [0:20]   last 20 daily log-returns (from t-19 to t)
        [20]     current_position  (normalised 0→1)
        [21]     unrealised_pnl_pct
        [22]     svm_signal  (−1, 0, +1 normalised to −1..1)
        [23]     cnn_return_forecast  (vol-normalised)
        [24:29]  regime one-hot (5 dims)
        [29:32]  vol-band one-hot (3 dims)
        [32]     days_since_last_trade  (normalised by 21)
        [33]     time_in_position       (normalised by 63)
        [34]     current portfolio vol  (rolling 20d)
        """
        close   = df["close"].iloc[:t+1]
        returns = np.log(close / close.shift(1)).iloc[-20:].to_numpy(np.float32)
        # Pad if insufficient history
        if len(returns) < 20:
            returns = np.pad(returns, (20 - len(returns), 0), constant_values=0.0)

        rvol = float(np.nanstd(returns) * np.sqrt(252)) if len(returns) > 1 else 0.01

        # Regime features
        reg_onehot = np.zeros(5, dtype=np.float32)
        vb_onehot  = np.zeros(3, dtype=np.float32)
        if regime_df is not None and t < len(regime_df):
            idx = df.index[t]
            if idx in regime_df.index:
                row = regime_df.loc[idx]
                for i in range(5):
                    reg_onehot[i] = float(row.get(f"regime_{i}", 0.0))
                for i in range(3):
                    vb_onehot[i]  = float(row.get(f"vol_band_{i}", 0.0))
        else:
            vb_onehot[1] = 1.0   # default MV

        state = np.concatenate([
            returns,
            [current_position,
             unrealised_pnl_pct,
             float(svm_signal),
             float(cnn_return_forecast) / (rvol + 1e-9),
            ],
            reg_onehot,
            vb_onehot,
            [
                min(days_since_trade  / 21.0,  5.0),
                min(time_in_position  / 63.0,  5.0),
                rvol,
            ],
        ]).astype(np.float32)

        return state  # (35,)

    # ════════════════════════════════════════════════════════════════════════
    # 6.  Utility — technical indicator helpers
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _macd_signal(close: pd.Series,
                     fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
        return sig_line

    @staticmethod
    def _macd_diff(close: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - sig_line

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _bollinger_bands(
        close: pd.Series,
        window: int = 20,
        n_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series]:
        sma   = close.rolling(window).mean()
        std   = close.rolling(window).std()
        upper = sma + n_std * std
        lower = sma - n_std * std
        return upper, lower

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mu  = series.rolling(window).mean()
        sig = series.rolling(window).std()
        return (series - mu) / (sig + 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# Batch builder — process all symbols
# ════════════════════════════════════════════════════════════════════════════

def build_all_features(
    equities: Dict[str, pd.DataFrame],
    config: dict,
    regime_df: Optional[pd.DataFrame] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run FeatureEngineer.build_raw_features() for all symbols,
    optionally inject regime features and save to parquet.

    Returns
    -------
    dict: symbol → merged feature DataFrame (25 columns including regime)
    """
    fe = FeatureEngineer(config)
    result = {}
    for sym, df in equities.items():
        raw = fe.build_raw_features(df)
        if regime_df is not None:
            raw = FeatureEngineer.inject_regime_features(raw, regime_df)
        result[sym] = raw
        if save_dir:
            out_path = f"{save_dir}/{sym}_features.parquet"
            raw.to_parquet(out_path)
            logger.info("Saved features for %s → %s", sym, out_path)
    return result