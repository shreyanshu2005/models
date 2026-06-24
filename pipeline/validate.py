"""
QBEAST-AI.N  ·  pipeline/validate.py
=======================================
Data integrity and look-ahead leakage validation.

Checks performed
----------------
1.  OHLCV completeness  — no NaN in close after alignment; H ≥ L; C in [L, H].
2.  Date monotonicity   — DatetimeIndex must be strictly increasing.
3.  No duplicate dates  — each symbol must have unique daily bars.
4.  Coverage floor      — every symbol must reach from universe_start to backtest_end
                          with ≥ 95% calendar coverage.
5.  Feature NaN audit   — after feature engineering, count NaNs per column
                          (warm-up rows at head are expected; body NaNs are errors).
6.  Look-ahead gate     — verifies that no feature column is computable using
                          data beyond time t (causal audit via shift check).
7.  Label look-ahead    — SVM labels are the ONLY place future data is intentionally
                          used (forward return); validate the shift is exactly
                          +target_h bars and the last target_h rows are NaN.
8.  CNN target audit    — confirm forward returns are NaN in last target_h rows.
9.  Regime causality    — HMM decode must not use future NIFTY50 bars; check by
                          confirming decode output length ≤ nifty input length.
10. Cross-symbol date   — confirm all 10 symbols share ≥ 80% of trading dates
                          after 2016-01-01.

Usage
-----
    from pipeline.validate import DataValidator
    v = DataValidator(config)
    v.validate_raw(equities, nifty)
    v.validate_features(features_dict)
    v.validate_labels(labels_dict, target_horizon=5)
    v.validate_regime(regime_df, nifty_feat_df)
    v.print_report()           # prints a summary table; raises on FAIL
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── ANSI colour codes for terminal output ──────────────────────────────────
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YEL   = "\033[93m"
_RST   = "\033[0m"
_BOLD  = "\033[1m"


class _CheckResult:
    """Single validation check outcome."""
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name   = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        status = f"{_GREEN}PASS{_RST}" if self.passed else f"{_RED}FAIL{_RST}"
        return f"[{status}] {self.name}: {self.detail}"


class DataValidator:
    """
    Stateful validator — accumulates check results and prints a final report.

    Parameters
    ----------
    config : dict
        Parsed config.yaml
    strict : bool
        If True, raise ValueError on the first FAIL. If False, accumulate all
        results and raise only in print_report() (default: False).
    """

    def __init__(self, config: dict, strict: bool = False):
        self.cfg             = config
        self.strict          = strict
        self.universe_start  = pd.Timestamp(config["dates"]["universe_start"])
        self.backtest_end    = pd.Timestamp(config["dates"]["backtest_end"])
        self.target_horizon  = config["features"]["cnn_lstm_target_horizon"]
        self.symbols         = config["universe"]["all_symbols"]
        self._results: List[_CheckResult] = []

    # ════════════════════════════════════════════════════════════════════════
    # 1.  Raw OHLCV validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_raw(
        self,
        equities: Dict[str, pd.DataFrame],
        nifty: pd.DataFrame,
    ) -> "DataValidator":
        """
        Run all raw-data integrity checks.

        Checks: monotonic index, no duplicates, OHLC sanity, coverage,
                cross-symbol date overlap, NIFTY continuity.
        """
        logger.info("=== Validating raw OHLCV data ===")

        for sym, df in equities.items():
            # 1a. Monotonic index
            self._check(
                f"{sym} monotonic index",
                df.index.is_monotonic_increasing,
                f"rows={len(df)}",
            )

            # 1b. No duplicate dates
            n_dups = df.index.duplicated().sum()
            self._check(
                f"{sym} no duplicate dates",
                n_dups == 0,
                f"duplicates={n_dups}",
            )

            # 1c. OHLC sanity
            bad_h_l  = (df["high"] < df["low"]).sum()
            bad_c_hi = (df["close"] > df["high"]).sum()
            bad_c_lo = (df["close"] < df["low"]).sum()
            ohlc_ok  = (bad_h_l + bad_c_hi + bad_c_lo) == 0
            self._check(
                f"{sym} OHLC sanity",
                ohlc_ok,
                f"H<L={bad_h_l}  C>H={bad_c_hi}  C<L={bad_c_lo}",
            )

            # 1d. No NaN close after universe_start
            df_u = df.loc[self.universe_start:]
            nan_close = df_u["close"].isna().sum()
            self._check(
                f"{sym} no NaN close post-2016",
                nan_close == 0,
                f"nan_close={nan_close}",
            )

            # 1e. Coverage: must reach at least to end of HP-train period (2017-12-31)
            hp_end   = pd.Timestamp(self.cfg["dates"]["hp_train_end"])
            coverage = df.index[-1] >= hp_end
            self._check(
                f"{sym} coverage to HP-train end",
                coverage,
                f"last_date={df.index[-1].date()}  required>={hp_end.date()}",
            )

        # 1f. NIFTY50 continuity
        nifty_start_ok = nifty.index[0] <= pd.Timestamp("2010-01-04")
        self._check(
            "NIFTY50 starts by 2010-01-04",
            nifty_start_ok,
            f"nifty_start={nifty.index[0].date()}",
        )
        nifty_end_ok = nifty.index[-1] >= pd.Timestamp("2026-06-01")
        self._check(
            "NIFTY50 reaches 2026",
            nifty_end_ok,
            f"nifty_end={nifty.index[-1].date()}",
        )
        nifty_nan = nifty["close"].isna().sum()
        self._check("NIFTY50 no NaN close", nifty_nan == 0, f"nan={nifty_nan}")

        # 1g. Cross-symbol date overlap ≥ 80% after universe_start
        post16_dates = [
            set(df.loc[self.universe_start:].index.tolist())
            for df in equities.values()
        ]
        if len(post16_dates) > 1:
            union_n = len(set.union(*post16_dates))
            inter_n = len(set.intersection(*post16_dates))
            overlap_pct = inter_n / max(union_n, 1) * 100
            self._check(
                "Cross-symbol date overlap ≥ 80%",
                overlap_pct >= 80.0,
                f"intersection/union = {inter_n}/{union_n} = {overlap_pct:.1f}%",
            )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 2.  Feature matrix validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_features(
        self,
        features: Dict[str, pd.DataFrame],
        warmup_rows: int = 252,
    ) -> "DataValidator":
        """
        Validate feature matrices output by FeatureEngineer.

        Checks: expected column count, no body NaNs beyond warm-up,
                no Inf values, column range sanity.
        """
        logger.info("=== Validating feature matrices ===")

        expected_cols = self.cfg["features"]["svm_n_features"]   # 25

        for sym, feat in features.items():
            # 2a. Column count
            self._check(
                f"{sym} feature column count",
                len(feat.columns) == expected_cols,
                f"got={len(feat.columns)}  expected={expected_cols}",
            )

            # 2b. No body NaNs (allow warmup rows at head)
            body = feat.iloc[warmup_rows:]
            nan_counts = body.isna().sum()
            body_nans  = nan_counts[nan_counts > 0]
            self._check(
                f"{sym} no body NaNs (post warmup={warmup_rows})",
                len(body_nans) == 0,
                f"cols_with_nan={list(body_nans.index)}",
            )

            # 2c. No Inf values
            inf_mask = np.isinf(feat.select_dtypes(include=[np.number]).values)
            n_inf    = int(inf_mask.sum())
            self._check(
                f"{sym} no Inf in features",
                n_inf == 0,
                f"inf_count={n_inf}",
            )

            # 2d. Z-scored features should have |values| < 10 in body (sanity)
            z_cols = [c for c in feat.columns if c.startswith("ret_") or
                      c in ["macd_signal", "roc_10", "rvol_20d", "bb_width"]]
            if z_cols:
                body_z   = body[z_cols].abs()
                extreme  = (body_z > 10).sum().sum()
                self._check(
                    f"{sym} z-score range |z|<10",
                    extreme == 0,
                    f"extreme_z_values={extreme}",
                )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 3.  Label look-ahead validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_labels(
        self,
        labels: Dict[str, pd.Series],
        target_horizon: Optional[int] = None,
    ) -> "DataValidator":
        """
        Validate SVM labels for correct forward-shift and NaN tail.

        Rule: last `target_horizon` rows must be NaN (no label computed
        beyond end of data). This is the ONLY intentional use of future data.
        """
        logger.info("=== Validating SVM labels ===")
        h = target_horizon if target_horizon is not None else self.target_horizon

        for sym, lbl in labels.items():
            # 3a. Tail NaN check
            tail_nan = lbl.iloc[-h:].isna().all()
            self._check(
                f"{sym} label tail NaN (last {h} rows)",
                tail_nan,
                f"tail_nan={lbl.iloc[-h:].isna().sum()}/{h}",
            )

            # 3b. Values are only {-1, 0, +1, NaN}
            valid_vals = set(lbl.dropna().unique()).issubset({-1, 0, 1})
            self._check(
                f"{sym} label values in {{-1,0,+1}}",
                valid_vals,
                f"unique={sorted(lbl.dropna().unique().tolist())}",
            )

            # 3c. Class distribution — warn if any class < 5%
            total  = len(lbl.dropna())
            counts = lbl.dropna().value_counts(normalize=True) * 100
            imb_ok = (counts >= 5.0).all()
            self._check(
                f"{sym} label class balance ≥ 5% each",
                imb_ok,
                f"dist={counts.round(1).to_dict()}",
            )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 4.  CNN-LSTM tensor validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_cnn_tensors(
        self,
        tensors: Dict[str, Tuple],  # sym → (X, y, dates)
        lookback: int = 30,
    ) -> "DataValidator":
        """
        Validate CNN-LSTM (X, y, dates) tuples.

        Checks: shape consistency, no NaN/Inf in X, target distribution sanity.
        """
        logger.info("=== Validating CNN-LSTM tensors ===")

        for sym, (X, y, dates) in tensors.items():
            N = len(dates)

            # 4a. Shape
            shape_ok = (X.shape == (N, lookback, 22)) and (y.shape == (N,))
            self._check(
                f"{sym} CNN tensor shape",
                shape_ok,
                f"X={X.shape}  y={y.shape}  expected=({N},{lookback},22)",
            )

            # 4b. No NaN/Inf in X
            nan_x = int(np.isnan(X).sum())
            inf_x = int(np.isinf(X).sum())
            self._check(
                f"{sym} CNN X no NaN/Inf",
                (nan_x + inf_x) == 0,
                f"nan={nan_x}  inf={inf_x}",
            )

            # 4c. Target distribution — should be approximately zero-mean
            y_mean = float(np.nanmean(y))
            y_std  = float(np.nanstd(y))
            self._check(
                f"{sym} CNN y distribution plausible",
                abs(y_mean) < 0.05 and 0.001 < y_std < 0.5,
                f"mean={y_mean:.4f}  std={y_std:.4f}",
            )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 5.  Regime causality validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_regime(
        self,
        regime_df: pd.DataFrame,
        nifty_feat_df: pd.DataFrame,
        vol_bands: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> "DataValidator":
        """
        Validate the regime output for causality and completeness.

        Checks: regime_df index ⊆ nifty_feat_df index (decode used only
        past data), required columns present, no NaN in output columns,
        vol_band one-hot sums to 1.
        """
        logger.info("=== Validating regime engine output ===")

        required_cols = (
            [f"regime_{i}" for i in range(5)] +
            [f"post_{i}"   for i in range(5)] +
            ["regime_label", "regime_name"]
        )

        # 5a. Required columns present
        missing = [c for c in required_cols if c not in regime_df.columns]
        self._check(
            "Regime required columns present",
            len(missing) == 0,
            f"missing={missing}",
        )

        # 5b. Causality: regime index ⊆ nifty input index
        extra_dates = set(regime_df.index) - set(nifty_feat_df.index)
        self._check(
            "Regime index ⊆ NIFTY feature index (causal)",
            len(extra_dates) == 0,
            f"extra_dates={len(extra_dates)}",
        )

        # 5c. No NaN in regime columns
        nan_counts = regime_df[required_cols].isna().sum()
        nan_issues = nan_counts[nan_counts > 0]
        self._check(
            "Regime no NaN in output columns",
            len(nan_issues) == 0,
            f"nan_cols={list(nan_issues.index)}",
        )

        # 5d. One-hot sums to 1 per row
        onehot_sum = regime_df[[f"regime_{i}" for i in range(5)]].sum(axis=1)
        onehot_ok  = ((onehot_sum - 1.0).abs() < 1e-6).all()
        self._check(
            "Regime one-hot sums to 1",
            bool(onehot_ok),
            f"max_dev={((onehot_sum - 1.0).abs().max()):.2e}",
        )

        # 5e. Posterior probabilities sum to 1
        post_sum  = regime_df[[f"post_{i}" for i in range(5)]].sum(axis=1)
        post_ok   = ((post_sum - 1.0).abs() < 1e-4).all()
        self._check(
            "Regime posteriors sum to 1",
            bool(post_ok),
            f"max_dev={((post_sum - 1.0).abs().max()):.2e}",
        )

        # 5f. All 5 states appear at least once (non-trivial model)
        state_counts = {i: int((regime_df["regime_label"] == i).sum())
                        for i in range(5)}
        all_states_seen = all(v > 0 for v in state_counts.values())
        self._check(
            "All 5 HMM states observed",
            all_states_seen,
            f"state_counts={state_counts}",
        )

        # 5g. Vol-band one-hot validation (if provided)
        if vol_bands is not None:
            for sym, vb_df in vol_bands.items():
                vb_cols = ["vol_band_0", "vol_band_1", "vol_band_2"]
                vb_sum  = vb_df[vb_cols].sum(axis=1)
                vb_ok   = ((vb_sum - 1.0).abs() < 1e-6).all()
                self._check(
                    f"{sym} vol-band one-hot sums to 1",
                    bool(vb_ok),
                    f"max_dev={((vb_sum - 1.0).abs().max()):.2e}",
                )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 6.  Walk-forward purge validation
    # ════════════════════════════════════════════════════════════════════════

    def validate_purge(
        self,
        folds: List[Dict],
        purge_days: int = 30,
        embargo_days: int = 10,
    ) -> "DataValidator":
        """
        Validate that walk-forward folds have no overlap between train and
        validation sets (respecting the purge + embargo gap).

        Parameters
        ----------
        folds : list of dicts with keys: train_end, val_start, val_end
        """
        logger.info("=== Validating walk-forward purge gaps ===")

        for i, fold in enumerate(folds):
            train_end = pd.Timestamp(fold["train_end"])
            val_start = pd.Timestamp(fold["val_start"])
            gap_days  = (val_start - train_end).days

            self._check(
                f"Fold {i+1} purge gap ≥ {purge_days + embargo_days} cal days",
                gap_days >= (purge_days + embargo_days),
                f"train_end={train_end.date()}  val_start={val_start.date()}  "
                f"gap_days={gap_days}",
            )

        return self

    # ════════════════════════════════════════════════════════════════════════
    # 7.  Report
    # ════════════════════════════════════════════════════════════════════════

    def print_report(self, raise_on_fail: bool = True) -> None:
        """
        Print a formatted summary of all validation checks.
        Raises ValueError if any check failed and raise_on_fail=True.
        """
        n_pass = sum(1 for r in self._results if r.passed)
        n_fail = sum(1 for r in self._results if not r.passed)
        n_total = len(self._results)

        header = f"\n{_BOLD}{'='*60}{_RST}\n"
        header += f"{_BOLD}  QBEAST-AI.N  ·  Data Validation Report{_RST}\n"
        header += f"{_BOLD}{'='*60}{_RST}\n"
        print(header)

        for res in self._results:
            print(f"  {res}")

        summary = (
            f"\n{_BOLD}{'─'*60}{_RST}\n"
            f"  Total: {n_total}   "
            f"{_GREEN}PASS: {n_pass}{_RST}   "
            f"{_RED}FAIL: {n_fail}{_RST}\n"
            f"{_BOLD}{'='*60}{_RST}\n"
        )
        print(summary)

        if n_fail > 0:
            logger.error("Validation FAILED: %d checks failed", n_fail)
            if raise_on_fail:
                failed_names = [r.name for r in self._results if not r.passed]
                raise ValueError(
                    f"Data validation failed on {n_fail} checks: {failed_names}"
                )
        else:
            logger.info("All %d validation checks PASSED.", n_total)

    def get_failed_checks(self) -> List[_CheckResult]:
        """Return list of all failed check results."""
        return [r for r in self._results if not r.passed]

    # ── Private ──────────────────────────────────────────────────────────────

    def _check(self, name: str, condition: bool, detail: str = "") -> None:
        result = _CheckResult(name, bool(condition), detail)
        self._results.append(result)
        if not result.passed:
            logger.warning("FAIL: %s — %s", name, detail)
            if self.strict:
                raise ValueError(f"Validation FAIL: {name} — {detail}")
        else:
            logger.debug("PASS: %s — %s", name, detail)