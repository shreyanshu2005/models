"""
QBEAST-AI.N  ·  regime/vol_band.py
=====================================
Per-symbol rolling volatility band classifier.

Logic
-----
1. Compute 20-day realised vol for each symbol each day.
2. Rank that vol against a trailing 252-day rolling window to get a percentile.
3. Classify into 3 bands:
      LOW  (LV)  : percentile ≤ 33rd  →  [vol_band_0 = 1, 1, 0, 0]
      MED  (MV)  : 33rd < pct ≤ 67th  →  [vol_band_1 = 1, 0, 1, 0]
      HIGH (HV)  : percentile > 67th   →  [vol_band_2 = 1, 0, 0, 1]

Output columns per symbol:
  vol_20d      : 20-day annualised realised vol
  vol_pct_252  : rolling 252-day percentile rank of vol_20d
  vol_band     : int  {0=LV, 1=MV, 2=HV}
  vol_band_0   : float one-hot  (LV)
  vol_band_1   : float one-hot  (MV)
  vol_band_2   : float one-hot  (HV)

Position-cap mapping (from config.yaml):
  HV  →  position cap reduced by hv_reduction_factor (0.67)
  LV/MV → standard large_cap_max / mid_cap_max

Usage
-----
    from regime.vol_band import VolBandClassifier
    vb = VolBandClassifier(config)
    band_df = vb.classify(equity_df)                      # single symbol
    all_bands = vb.classify_all(equities_dict)            # dict of DFs
    cap = vb.position_cap(symbol, band_df, date)          # live cap lookup
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolBandClassifier:
    """
    Rolling-percentile volatility band classifier.

    Parameters
    ----------
    config : dict
        Parsed config.yaml — reads regime.vol_band.{low_pct, high_pct},
        position_limits.{large_cap_max, mid_cap_max, hv_reduction_factor}.
    """

    def __init__(self, config: dict):
        self.cfg          = config
        self.low_pct      = config["regime"]["vol_band"]["low_pct"]    # 33
        self.high_pct     = config["regime"]["vol_band"]["high_pct"]   # 67
        self.lc_max       = config["position_limits"]["large_cap_max"] # 0.30
        self.mc_max       = config["position_limits"]["mid_cap_max"]   # 0.20
        self.hv_factor    = config["position_limits"]["hv_reduction_factor"]  # 0.67
        self.lc_symbols   = set(config["universe"]["large_cap_symbols"])
        self.mc_symbols   = set(config["universe"]["mid_cap_symbols"])

    # ── Public API ───────────────────────────────────────────────────────────

    def classify(
        self,
        equity_df: pd.DataFrame,
        vol_window: int = 20,
        pct_window: int = 252,
    ) -> pd.DataFrame:
        """
        Classify a single-symbol OHLCV DataFrame into vol bands.

        Parameters
        ----------
        equity_df  : DatetimeIndex OHLCV DataFrame (must have 'close' column).
        vol_window : days to compute realised vol  (default 20).
        pct_window : days for rolling percentile rank (default 252).

        Returns
        -------
        DataFrame (same index) with columns:
          vol_20d, vol_pct_252, vol_band, vol_band_0, vol_band_1, vol_band_2
        """
        close   = equity_df["close"]
        ret     = np.log(close / close.shift(1))

        out = pd.DataFrame(index=equity_df.index)
        out["vol_20d"] = ret.rolling(vol_window).std() * np.sqrt(252)

        # Rolling percentile rank — strictly causal
        out["vol_pct_252"] = (
            out["vol_20d"]
            .rolling(pct_window, min_periods=pct_window // 2)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

        # Band classification
        out["vol_band"] = 1  # default MV
        out.loc[out["vol_pct_252"] <= self.low_pct  / 100.0, "vol_band"] = 0  # LV
        out.loc[out["vol_pct_252"] >  self.high_pct / 100.0, "vol_band"] = 2  # HV

        # One-hot
        for b, name in enumerate(["vol_band_0", "vol_band_1", "vol_band_2"]):
            out[name] = (out["vol_band"] == b).astype(np.float32)

        # NaN warm-up rows → default MV band
        warmup_mask = out["vol_pct_252"].isna()
        out.loc[warmup_mask, "vol_band"]   = 1
        out.loc[warmup_mask, "vol_band_0"] = 0.0
        out.loc[warmup_mask, "vol_band_1"] = 1.0
        out.loc[warmup_mask, "vol_band_2"] = 0.0

        return out

    def classify_all(
        self,
        equities: Dict[str, pd.DataFrame],
        vol_window: int = 20,
        pct_window: int = 252,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run classify() for all symbols in the equities dict.

        Returns
        -------
        dict: symbol → vol_band DataFrame
        """
        result = {}
        for sym, df in equities.items():
            band_df        = self.classify(df, vol_window, pct_window)
            result[sym]    = band_df
            band_counts    = band_df["vol_band"].value_counts().sort_index().to_dict()
            logger.info(
                "VolBand %-12s  LV=%d  MV=%d  HV=%d",
                sym,
                band_counts.get(0, 0),
                band_counts.get(1, 0),
                band_counts.get(2, 0),
            )
        return result

    def position_cap(
        self,
        symbol: str,
        band_df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> float:
        """
        Return the effective position cap (fraction of portfolio) for
        symbol at a given date, respecting the HV reduction factor.

        Parameters
        ----------
        symbol   : stock ticker string
        band_df  : output of classify() for this symbol
        date     : query date (must be in band_df.index)

        Returns
        -------
        float  —  e.g. 0.30 (large-cap, LV/MV), 0.20 (large-cap, HV),
                        0.20 (mid-cap, LV/MV),   0.13 (mid-cap, HV)
        """
        base = self.lc_max if symbol in self.lc_symbols else self.mc_max

        try:
            band = int(band_df.loc[date, "vol_band"])
        except KeyError:
            # Date not in index — use last available
            band = int(band_df["vol_band"].iloc[-1])

        if band == 2:   # HV
            return round(base * self.hv_factor, 4)
        return base

    def cnn_lookback(
        self,
        symbol: str,
        band_df: pd.DataFrame,
        date: pd.Timestamp,
        config: Optional[dict] = None,
    ) -> int:
        """
        Return the appropriate CNN-LSTM lookback window for this symbol/date
        based on cap-segment × vol-band interaction.

        Mapping (from config):
          Large-cap + LV  →  cnn_lstm_lookback_lc_lv  (60)
          Any    + HV     →  cnn_lstm_lookback_hv      (20)
          Default         →  cnn_lstm_lookback_default (30)
        """
        cfg = config or self.cfg
        try:
            band = int(band_df.loc[date, "vol_band"])
        except KeyError:
            band = int(band_df["vol_band"].iloc[-1])

        if band == 2:   # HV
            return cfg["features"]["cnn_lstm_lookback_hv"]
        if symbol in self.lc_symbols and band == 0:   # LC + LV
            return cfg["features"]["cnn_lstm_lookback_lc_lv"]
        return cfg["features"]["cnn_lstm_lookback_default"]

    def merge_regime_and_volband(
        self,
        regime_df: pd.DataFrame,
        band_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge HMM regime columns + vol-band columns on a common DatetimeIndex.
        Used to produce the unified conditioning DataFrame fed to all models.

        Parameters
        ----------
        regime_df : output of RegimeHMM.decode()
        band_df   : output of VolBandClassifier.classify() for one symbol

        Returns
        -------
        DataFrame with all regime_* + post_* + vol_band_* columns, forward-filled.
        """
        merged = regime_df.join(band_df[["vol_band", "vol_band_0", "vol_band_1", "vol_band_2"]], how="left")
        # Forward-fill any gaps (regime decoded on NIFTY dates, vol on equity dates)
        merged = merged.ffill()
        return merged


# ════════════════════════════════════════════════════════════════════════════
# Convenience: vol-band summary report
# ════════════════════════════════════════════════════════════════════════════

def vol_band_summary(
    all_bands: Dict[str, pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produce a summary table showing LV / MV / HV day-counts and percentages
    for each symbol, optionally filtered to [start, end].

    Returns
    -------
    pd.DataFrame  shape (n_symbols, 6):
      columns = [LV_days, MV_days, HV_days, LV_pct, MV_pct, HV_pct]
    """
    rows = {}
    for sym, df in all_bands.items():
        d = df["vol_band"]
        if start:
            d = d.loc[start:]
        if end:
            d = d.loc[:end]
        n   = len(d)
        lv  = int((d == 0).sum())
        mv  = int((d == 1).sum())
        hv  = int((d == 2).sum())
        rows[sym] = {
            "LV_days": lv, "MV_days": mv, "HV_days": hv,
            "LV_pct":  round(lv / n * 100, 1),
            "MV_pct":  round(mv / n * 100, 1),
            "HV_pct":  round(hv / n * 100, 1),
        }
    return pd.DataFrame(rows).T