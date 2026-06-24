"""
QBEAST-AI.N  ·  pipeline/calendar.py
=======================================
Trading calendar construction and alignment utilities.

Responsibilities
----------------
1. Build a unified NSE trading calendar from the union of all equity + NIFTY50 dates.
2. Reindex every DataFrame to the shared calendar with controlled forward-fill.
3. Provide split-boundary helpers for HP-train, walk-forward-val, and backtest windows.
4. Log calendar statistics (total trading days, holiday gaps, per-symbol coverage).

This module is called ONCE during Week 1 pipeline setup and the resulting
`trading_calendar` is reused throughout the entire project.

Usage
-----
    from pipeline.calendar import TradingCalendar
    tc = TradingCalendar(config)
    calendar = tc.build(equities, nifty)       # pd.DatetimeIndex
    equities_aligned = tc.align_all(equities, calendar)
    nifty_aligned    = tc.align(nifty, calendar)
    splits           = tc.get_splits()         # dict of date boundaries
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingCalendar:
    """
    Builds and manages the shared NSE trading calendar.

    Parameters
    ----------
    config : dict
        Parsed config.yaml — reads dates.*, paths.*
    """

    def __init__(self, config: dict):
        self.cfg             = config
        self.universe_start  = pd.Timestamp(config["dates"]["universe_start"])
        self.backtest_end    = pd.Timestamp(config["dates"]["backtest_end"])
        self.hmm_warmup_start = pd.Timestamp(config["dates"]["hmm_warmup_start"])
        self._calendar: Optional[pd.DatetimeIndex] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def build(
        self,
        equities: Dict[str, pd.DataFrame],
        nifty: pd.DataFrame,
        start_from: Optional[str] = None,
    ) -> pd.DatetimeIndex:
        """
        Build unified trading calendar as the union of all index dates across
        all equity DataFrames and NIFTY50, filtered to [universe_start, backtest_end].

        Parameters
        ----------
        equities   : dict of symbol → OHLCV DataFrame (already date-indexed)
        nifty      : NIFTY50 DataFrame (date-indexed)
        start_from : ISO string override for start; defaults to universe_start

        Returns
        -------
        pd.DatetimeIndex  — sorted ascending, tz-naive
        """
        start = pd.Timestamp(start_from) if start_from else self.universe_start

        # Union of all dates
        all_dates: set = set(nifty.index.tolist())
        for df in equities.values():
            all_dates.update(df.index.tolist())

        cal = pd.DatetimeIndex(sorted(all_dates))

        # Filter to [start, backtest_end]
        cal = cal[(cal >= start) & (cal <= self.backtest_end)]

        # Sanity: should be weekday-only (Mon–Fri)
        weekday_mask = cal.weekday < 5
        n_weekend = (~weekday_mask).sum()
        if n_weekend > 0:
            logger.warning(
                "Calendar contains %d weekend dates — dropping them", n_weekend
            )
            cal = cal[weekday_mask]

        self._calendar = cal

        # Log statistics
        logger.info(
            "Trading calendar built: %d days  %s → %s",
            len(cal), cal[0].date(), cal[-1].date(),
        )
        self._log_gap_stats(cal)
        return cal

    def build_nifty_calendar(self, nifty: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Build a calendar starting from hmm_warmup_start (2010) using NIFTY50 only.
        Used exclusively by the HMM regime engine which needs pre-2016 data.
        """
        cal = nifty.index
        cal = cal[(cal >= self.hmm_warmup_start) & (cal <= self.backtest_end)]
        cal = cal[cal.weekday < 5]
        logger.info(
            "NIFTY warmup calendar: %d days  %s → %s",
            len(cal), cal[0].date(), cal[-1].date(),
        )
        return pd.DatetimeIndex(sorted(cal))

    def align(
        self,
        df: pd.DataFrame,
        calendar: Optional[pd.DatetimeIndex] = None,
        ffill_limit: int = 5,
    ) -> pd.DataFrame:
        """
        Reindex a single DataFrame to the trading calendar with forward-fill.

        Parameters
        ----------
        df          : date-indexed DataFrame
        calendar    : target index; uses self._calendar if None
        ffill_limit : max consecutive days to forward-fill (handles holidays)

        Returns
        -------
        pd.DataFrame reindexed to calendar, NaN-filled up to ffill_limit
        """
        cal = calendar if calendar is not None else self._calendar
        if cal is None:
            raise RuntimeError("Call build() before align().")

        df_aligned = df.reindex(cal)
        df_aligned = df_aligned.ffill(limit=ffill_limit)
        return df_aligned

    def align_all(
        self,
        equities: Dict[str, pd.DataFrame],
        calendar: Optional[pd.DatetimeIndex] = None,
        ffill_limit: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all equity DataFrames to the trading calendar.

        Returns
        -------
        dict: symbol → aligned DataFrame
        """
        cal = calendar if calendar is not None else self._calendar
        result = {}
        for sym, df in equities.items():
            aligned = self.align(df, cal, ffill_limit = 5)
            n_nan_close = aligned["close"].isna().sum()
            if n_nan_close > 0:
                logger.warning(
                    "%s: %d NaN close prices after alignment (exceeds ffill limit "
                    "or data gap)", sym, n_nan_close
                )
            result[sym] = aligned
            logger.info(
                "Aligned %-12s  rows=%5d  nan_close=%d",
                sym, len(aligned), n_nan_close,
            )
        return result

    def get_splits(self) -> Dict[str, pd.Timestamp]:
        """
        Return all canonical date boundary timestamps from config.

        Keys:
            universe_start, hmm_warmup_start,
            hp_train_start, hp_train_end,
            val_start, val_end,
            backtest_start, backtest_end
        """
        d = self.cfg["dates"]
        return {k: pd.Timestamp(v) for k, v in d.items()}

    def get_trading_days_in_range(
        self,
        start: str,
        end: str,
        calendar: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DatetimeIndex:
        """Return subset of calendar between start and end (inclusive)."""
        cal = calendar if calendar is not None else self._calendar
        if cal is None:
            raise RuntimeError("Call build() before querying ranges.")
        t_start = pd.Timestamp(start)
        t_end   = pd.Timestamp(end)
        return cal[(cal >= t_start) & (cal <= t_end)]

    def month_ends(
        self,
        start: str,
        end: str,
        calendar: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DatetimeIndex:
        """
        Return the last trading day of each calendar month in [start, end].
        Used by the monthly retrain loop to identify refit trigger dates.
        """
        cal = self.get_trading_days_in_range(start, end, calendar)
        cal_series = pd.Series(cal, index=cal)
        month_end_dates = (
            cal_series
            .groupby([cal.year, cal.month])
            .last()
            .values
        )
        return pd.DatetimeIndex(month_end_dates)

    def month_starts(
        self,
        start: str,
        end: str,
        calendar: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DatetimeIndex:
        """
        Return the first trading day of each calendar month in [start, end].
        Used to identify when the monthly retrain loop fires.
        """
        cal = self.get_trading_days_in_range(start, end, calendar)
        cal_series = pd.Series(cal, index=cal)
        month_start_dates = (
            cal_series
            .groupby([cal.year, cal.month])
            .first()
            .values
        )
        return pd.DatetimeIndex(month_start_dates)

    def purge_window(
        self,
        fold_train_end: pd.Timestamp,
        purge_days: int = 30,
        embargo_days: int = 10,
        calendar: Optional[pd.DatetimeIndex] = None,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Compute the purge + embargo boundary after a walk-forward fold end.

        Returns
        -------
        (purge_end, val_start) where val_start = fold_train_end + purge + embargo
        in trading-day calendar terms.
        """
        cal = calendar if calendar is not None else self._calendar
        if cal is None:
            raise RuntimeError("Call build() before purge_window().")

        future_cal = cal[cal > fold_train_end]
        if len(future_cal) < purge_days + embargo_days:
            raise ValueError("Not enough future calendar days for purge + embargo.")

        purge_end  = future_cal[purge_days - 1]
        val_start  = future_cal[purge_days + embargo_days - 1]
        return purge_end, val_start

    def coverage_report(
        self,
        equities: Dict[str, pd.DataFrame],
        calendar: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Produce a per-symbol coverage report vs the trading calendar.

        Returns
        -------
        DataFrame with columns: first_date, last_date, n_rows, n_cal_days,
                                coverage_pct, n_nan_close
        """
        cal = calendar if calendar is not None else self._calendar
        rows = {}
        for sym, df in equities.items():
            n_cal   = len(cal[(cal >= df.index[0]) & (cal <= df.index[-1])])
            n_rows  = len(df)
            nan_cl  = df["close"].isna().sum()
            rows[sym] = {
                "first_date":    df.index[0].date(),
                "last_date":     df.index[-1].date(),
                "n_rows":        n_rows,
                "n_cal_days":    n_cal,
                "coverage_pct":  round(n_rows / max(n_cal, 1) * 100, 1),
                "n_nan_close":   int(nan_cl),
            }
        report = pd.DataFrame(rows).T
        logger.info("\nCoverage report:\n%s", report.to_string())
        return report

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _log_gap_stats(cal: pd.DatetimeIndex) -> None:
        """Log statistics about gaps > 1 business day (exchange holidays, etc.)."""
        diffs = pd.Series(cal).diff().dt.days.dropna()
        large_gaps = diffs[diffs > 3]
        if len(large_gaps) > 0:
            logger.info(
                "Calendar gaps > 3 days: %d occurrences  "
                "(expected: Diwali, Republic Day, Holi, etc.)",
                len(large_gaps),
            )
            for idx, gap in large_gaps.items():
                logger.debug(
                    "  Gap of %d days ending %s", int(gap), cal[idx].date()
                )
        else:
            logger.info("No anomalous calendar gaps detected.")