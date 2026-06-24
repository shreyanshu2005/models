"""
QBEAST-AI.N  ·  data/loader.py
================================
Raw CSV ingestion layer.

Responsibilities
----------------
1. Load all 10 NSE equity CSVs into a common schema DataFrame.
2. Load and concatenate the four NIFTY50 benchmark CSV files into a
   single continuous series (2004-01-01 → 2026-06-22).
3. Normalise column names, parse dates, set DatetimeIndex, forward-fill
   any intra-week gaps, drop pre-universe rows.
4. Return clean DataFrames ready for feature engineering.

Usage
-----
    from data.loader import DataLoader
    loader = DataLoader(config)
    equities = loader.load_equities()   # dict: symbol → DataFrame
    nifty    = loader.load_nifty50()    # single DataFrame
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Column mappings ──────────────────────────────────────────────────────────

# Equity CSVs: these are the columns we keep
_EQUITY_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]

# NIFTY50 CSVs have a different schema
_NIFTY_RENAME = {
    "DateTime": "date",
    "Open":     "open",
    "High":     "high",
    "Low":      "low",
    "Close":    "close",
    "Volume":   "volume",
}


class DataLoader:
    """
    Loads raw CSVs and returns clean, aligned DataFrames.

    Parameters
    ----------
    config : dict
        Parsed content of config.yaml.
    data_root : str | Path, optional
        Override for the raw data directory. If None, resolved from config.
    """

    def __init__(self, config: dict, data_root: str | Path | None = None):
        self.cfg = config
        self.raw_dir = Path(data_root or config["paths"]["raw_data"])
        self.universe_start = pd.Timestamp(config["dates"]["universe_start"])
        self.backtest_end   = pd.Timestamp(config["dates"]["backtest_end"])
        self.symbols        = config["universe"]["all_symbols"]

    # ── Public API ───────────────────────────────────────────────────────────

    def load_equities(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 10 equity CSVs.

        Returns
        -------
        dict
            Keys are symbol strings (e.g. "RELIANCE"), values are DataFrames
            with DatetimeIndex and columns: open, high, low, close,
            adj_close, volume.  Index is timezone-naive UTC date.
        """
        out: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            path = self.raw_dir / f"{sym}.csv"
            if not path.exists():
                logger.error("Missing equity CSV: %s", path)
                raise FileNotFoundError(f"Missing CSV for {sym}: {path}")
            df = self._load_equity_csv(path, sym)
            out[sym] = df
            logger.info(
                "Loaded %-12s  rows=%5d  start=%s  end=%s",
                sym, len(df), df.index[0].date(), df.index[-1].date()
            )
        return out

    def load_nifty50(self) -> pd.DataFrame:
        """
        Concatenate the four NIFTY50 benchmark CSVs into one continuous
        series sorted by date.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (tz-naive), columns: open, high, low, close, volume.
            No adj_close for NIFTY — use close directly.
        """
        files = self.cfg["paths"]["nifty50_files"]
        frames = []
        for fpath in files:
            path = Path(fpath)
            if not path.exists():
                # Try relative to raw_dir parent
                path = self.raw_dir.parent.parent / fpath
            if not path.exists():
                logger.warning("NIFTY50 file not found at %s — skipping", fpath)
                continue
            df = self._load_nifty_csv(path)
            frames.append(df)
            logger.info(
                "Loaded NIFTY50  file=%-55s  rows=%5d  start=%s  end=%s",
                path.name, len(df), df.index[0].date(), df.index[-1].date()
            )

        nifty = (
            pd.concat(frames)
            .sort_index()
            .loc[~pd.Series(pd.concat(frames).sort_index().index).duplicated().values]
        )
        # De-duplicate (overlapping file edges)
        nifty = nifty[~nifty.index.duplicated(keep="first")]
        logger.info(
            "NIFTY50 combined: rows=%d  start=%s  end=%s",
            len(nifty), nifty.index[0].date(), nifty.index[-1].date()
        )
        return nifty

    def load_all(self):
        """Convenience: returns (equities_dict, nifty_df)."""
        return self.load_equities(), self.load_nifty50()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_equity_csv(self, path: Path, symbol: str) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            usecols=_EQUITY_COLS,
            parse_dates=["date"],
            dtype={
                "open": np.float64, "high": np.float64,
                "low":  np.float64, "close": np.float64,
                "adj_close": np.float64, "volume": np.float64,
            },
            low_memory=False,
        )
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()
        df.index = df.index.tz_localize(None)

        # Drop metadata column
        df = df.drop(columns=["symbol"], errors="ignore")

        # Filter to [universe_start, backtest_end]
        df = df.loc[self.universe_start : self.backtest_end]

        # Sanity: require OHLCV present
        _required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in _required if c not in df.columns]
        if missing:
            raise ValueError(f"{symbol}: missing columns {missing}")

        # Forward-fill up to 3 consecutive missing trading days
        # (handles exchange holidays creating gaps in some sources)
        df = df[~df.index.duplicated(keep="first")]
        df = df.replace(0, np.nan)

        # OHLC sanity: high >= low, close within [low, high]
        bad_mask = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
        if bad_mask.sum() > 0:
            logger.warning(
                "%s: %d rows with OHLC inconsistency — NaN-ing those rows",
                symbol, bad_mask.sum()
            )
            df.loc[bad_mask, ["open", "high", "low", "close", "adj_close"]] = np.nan

        df = df.ffill(limit=3)

        # Drop rows where close is still NaN after ffill
        df = df.dropna(subset=["close"])

        return df

    def _load_nifty_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df = df.rename(columns=_NIFTY_RENAME)

        # DateTime column may have timezone "+05:30" — strip it
        df["date"] = (
            pd.to_datetime(df["date"], utc=False, errors="coerce")
            .dt.tz_localize(None)  # already tz-aware strings, strip tz
        )
        # Handle already tz-aware parse
        try:
            df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
        except TypeError:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].astype(np.float64)

        df = df[~df.index.duplicated(keep="first")]
        df = df.replace(0, np.nan).ffill(limit=3).dropna(subset=["close"])
        return df


# ── Convenience function ─────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load YAML config from path."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_trading_calendar(
    equities: Dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Build a unified NSE trading calendar: union of all dates present in any
    of the equity DataFrames and NIFTY50.  This is used to align all series.

    Returns
    -------
    pd.DatetimeIndex sorted ascending.
    """
    all_dates = set(nifty.index.tolist())
    for df in equities.values():
        all_dates.update(df.index.tolist())
    cal = pd.DatetimeIndex(sorted(all_dates))
    logger.info("Trading calendar: %d days  %s → %s",
                len(cal), cal[0].date(), cal[-1].date())
    return cal


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    method: str = "ffill",
    limit: int = 3,
) -> pd.DataFrame:
    """
    Reindex a DataFrame to the shared trading calendar and
    forward-fill gaps up to `limit` days.
    """
    df = df.reindex(calendar)
    if method == "ffill":
        df = df.ffill(limit=limit)
    return df