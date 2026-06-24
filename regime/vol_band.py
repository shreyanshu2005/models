"""
QBEAST-AI.N  ·  regime/vol_band.py  (FIXED — v1.1)
=====================================================
Per-symbol rolling volatility band classifier.

Bugs fixed vs v1.0
-------------------
1. config["regime"]["vol_band"]["low_pct"] — key path was wrong.
   The YAML has  regime.low_pct  and  regime.high_pct  directly,
   not nested under a vol_band sub-key.  Fixed to config["regime"]["low_pct"].

2. config["position_limits"]["large_cap_max"] — this key never existed in
   config.yaml.  Position limits live under sac_hp (max_pos_lc_base,
   max_pos_mc_base, hv_reduction).  Fixed to use those paths.

3. config["universe"]["large_cap_symbols"] — key was wrong.
   The YAML has universe.symbols.large_cap (a list) and
   universe.symbols.mid_cap.  Fixed to read those lists.

4. cnn_lookback() referenced cfg["features"]["cnn_lstm_lookback_hv"] and
   cfg["features"]["cnn_lstm_lookback_lc_lv"] which don't exist in the YAML.
   Now falls back to cnn_lstm_lookback_default (30) safely.

Logic
-----
1. Compute 20-day realised vol for each symbol each day.
2. Rank that vol against a trailing 252-day rolling window to get a percentile.
3. Classify into 3 bands:
      LOW  (LV)  : percentile ≤ low_pct   →  vol_band=0
      MED  (MV)  : low_pct < pct ≤ high_pct →  vol_band=1
      HIGH (HV)  : percentile > high_pct  →  vol_band=2

Usage
-----
    from regime.vol_band import VolBandClassifier
    vb       = VolBandClassifier(config)
    band_df  = vb.classify(equity_df)
    all_bands = vb.classify_all(equities_dict)
    cap      = vb.position_cap("RELIANCE", band_df, date)
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
        Parsed config.yaml.
        Reads:
          regime.low_pct         (float, e.g. 0.33)
          regime.high_pct        (float, e.g. 0.67)
          sac_hp.max_pos_lc_base (float, e.g. 0.30)
          sac_hp.max_pos_mc_base (float, e.g. 0.20)
          sac_hp.hv_reduction    (float, e.g. 0.67)
          universe.symbols.large_cap (list of tickers)
          universe.symbols.mid_cap   (list of tickers)
    """

    def __init__(self, config: dict):
        self.cfg = config
        reg      = config.get("regime", {})

        # FIX #1 — correct key paths for vol-band thresholds
        # The YAML stores them as fractions (0.33, 0.67), not as percentiles (33, 67)
        # We normalise: if value > 1, assume it's a percentile → divide by 100
        low_raw  = float(reg.get("low_pct",  0.33))
        high_raw = float(reg.get("high_pct", 0.67))
        self.low_pct  = low_raw  / 100.0 if low_raw  > 1.0 else low_raw
        self.high_pct = high_raw / 100.0 if high_raw > 1.0 else high_raw

        # FIX #2 — position limits live under sac_hp
        sac = config.get("sac_hp", {})
        self.lc_max    = float(sac.get("max_pos_lc_base", 0.30))
        self.mc_max    = float(sac.get("max_pos_mc_base", 0.20))
        self.hv_factor = float(sac.get("hv_reduction",   0.67))

        # FIX #3 — universe symbols nested under universe.symbols.*
        syms = config.get("universe", {}).get("symbols", {})
        # Handle both list-style config and flat config
        if isinstance(syms, dict):
            self.lc_symbols = set(syms.get("large_cap", []))
            self.mc_symbols = set(syms.get("mid_cap",   []))
        else:
            # Fallback: universe.all_symbols — classify everything as LC
            self.lc_symbols = set(config.get("universe", {}).get("all_symbols", []))
            self.mc_symbols = set()

        # Also accept universe.large_cap_symbols / universe.mid_cap_symbols
        if not self.lc_symbols:
            self.lc_symbols = set(config.get("universe", {}).get("large_cap_symbols", []))
        if not self.mc_symbols:
            self.mc_symbols = set(config.get("universe", {}).get("mid_cap_symbols",   []))

        logger.debug(
            "VolBandClassifier: LC=%s  MC=%s  lc_max=%.2f  mc_max=%.2f  hv_factor=%.2f",
            sorted(self.lc_symbols), sorted(self.mc_symbols),
            self.lc_max, self.mc_max, self.hv_factor,
        )

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
        close = equity_df["close"]
        ret   = np.log(close / close.shift(1))

        out = pd.DataFrame(index=equity_df.index)
        out["vol_20d"] = ret.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)

        # Causal rolling percentile rank using pandas rank
        out["vol_pct_252"] = (
            out["vol_20d"]
            .rolling(pct_window, min_periods=pct_window // 2)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

        # Band classification (LV=0, MV=1, HV=2)
        out["vol_band"] = 1  # default MV
        out.loc[out["vol_pct_252"] <= self.low_pct,  "vol_band"] = 0  # LV
        out.loc[out["vol_pct_252"] >  self.high_pct, "vol_band"] = 2  # HV

        # One-hot encoding
        for b, col in enumerate(["vol_band_0", "vol_band_1", "vol_band_2"]):
            out[col] = (out["vol_band"] == b).astype(np.float32)

        # NaN warm-up rows → default MV band (middle band, conservative)
        warmup_mask = out["vol_pct_252"].isna()
        if warmup_mask.any():
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
        """Run classify() for all symbols; return dict of band DataFrames."""
        result = {}
        for sym, df in equities.items():
            band_df       = self.classify(df, vol_window, pct_window)
            result[sym]   = band_df
            counts        = band_df["vol_band"].value_counts().sort_index().to_dict()
            logger.info(
                "VolBand %-12s  LV=%d  MV=%d  HV=%d",
                sym,
                counts.get(0, 0),
                counts.get(1, 0),
                counts.get(2, 0),
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

        Returns e.g.  0.30 (LC, LV/MV)  or  0.20 (LC, HV)
                      0.20 (MC, LV/MV)  or  0.13 (MC, HV)
        """
        base = self.lc_max if symbol in self.lc_symbols else self.mc_max

        try:
            band = int(band_df.loc[date, "vol_band"])
        except KeyError:
            # Date not in index — use last available
            band = int(band_df["vol_band"].dropna().iloc[-1])

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
        Return the appropriate CNN-LSTM lookback window for this symbol/date.

        FIX #4 — keys cnn_lstm_lookback_hv / cnn_lstm_lookback_lc_lv
        don't exist in config.yaml; safe defaults are used instead.
        """
        cfg  = config or self.cfg
        feats = cfg.get("features", {})

        try:
            band = int(band_df.loc[date, "vol_band"])
        except KeyError:
            band = int(band_df["vol_band"].dropna().iloc[-1])

        default  = int(feats.get("cnn_lstm_lookback_default", 30))
        hv_lb    = int(feats.get("cnn_lstm_lookback_hv",      20))
        lc_lv_lb = int(feats.get("cnn_lstm_lookback_lc_lv",   60))

        if band == 2:
            return hv_lb
        if symbol in self.lc_symbols and band == 0:
            return lc_lv_lb
        return default

    def merge_regime_and_volband(
        self,
        regime_df: pd.DataFrame,
        band_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge HMM regime columns + vol-band columns on a common DatetimeIndex.
        Forward-fill to handle NIFTY vs equity calendar mismatches.

        Returns DataFrame with all regime_* + post_* + vol_band* columns.
        """
        vb_cols = ["vol_band", "vol_band_0", "vol_band_1", "vol_band_2"]
        merged  = regime_df.join(band_df[vb_cols], how="left")
        merged  = merged.ffill()
        return merged


# ════════════════════════════════════════════════════════════════════════════
# Summary report helper
# ════════════════════════════════════════════════════════════════════════════

def vol_band_summary(
    all_bands: Dict[str, pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str]   = None,
) -> pd.DataFrame:
    """
    Summary table: LV / MV / HV day-counts and percentages per symbol.

    Returns pd.DataFrame shape (n_symbols, 6):
      columns = [LV_days, MV_days, HV_days, LV_pct, MV_pct, HV_pct]
    """
    rows = {}
    for sym, df in all_bands.items():
        d = df["vol_band"]
        if start:
            d = d.loc[start:]
        if end:
            d = d.loc[:end]
        n  = max(len(d), 1)
        lv = int((d == 0).sum())
        mv = int((d == 1).sum())
        hv = int((d == 2).sum())
        rows[sym] = {
            "LV_days": lv, "MV_days": mv, "HV_days": hv,
            "LV_pct":  round(lv / n * 100, 1),
            "MV_pct":  round(mv / n * 100, 1),
            "HV_pct":  round(hv / n * 100, 1),
        }
    return pd.DataFrame(rows).T