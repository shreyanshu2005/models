"""
models/svm/train.py
===================
SVM (RBF kernel) training with Optuna hyperparameter search and
purged walk-forward cross-validation.

Week 2–3 deliverable per §2 of the QBEAST-AI.N architecture spec.

Key design choices
──────────────────
• sklearn.svm.SVC with RBF kernel, class_weight='balanced' (fixed).
• Optuna minimises negative F1 macro on the purged OOS fold.
• Walk-forward CV: 5 folds, 30 trading-day purge + 10-day embargo.
• FAST params (C, gamma): re-searched quarterly or on regime drift.
• SLOW params (kernel, class_weight): fixed / re-searched semi-annually.
• Cap-segment priors: LC symbols search C in [10, 100]; MC in [0.1, 10].
• Expanding window refit on first trading day of each backtest month.
• No look-ahead: training set always ends at prior month-end.

Transaction costs
─────────────────
Labels already include cost deduction (from features.py).
Cost is baked into the label at feature-build time — not here.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.svm.features import build_svm_features, get_feature_columns

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
FEATURE_COLS    = get_feature_columns()
PURGE_DAYS      = 30    # trading days
EMBARGO_DAYS    = 10    # trading days
N_CV_FOLDS      = 5
LARGE_CAP_SYMS  = {"RELIANCE", "HDFCBANK", "BAJFINANCE", "MARUTI", "HEROMOTOCO"}


# ─────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SVMResult:
    symbol: str
    best_C: float
    best_gamma: float
    best_f1: float
    val_f1: float
    val_report: str
    model: SVC
    scaler: StandardScaler
    train_end: str
    hp_search_type: str   # 'full' | 'fast' | 'reuse'


@dataclass
class SVMRegistry:
    """Holds the current best SVM model per symbol — updated monthly."""
    models: Dict[str, SVMResult] = field(default_factory=dict)

    def update(self, result: SVMResult) -> None:
        self.models[result.symbol] = result

    def get(self, symbol: str) -> Optional[SVMResult]:
        return self.models.get(symbol)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SVMRegistry":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────
# Purged walk-forward CV
# ─────────────────────────────────────────────────────────────────────

def _make_wf_splits(
    dates: pd.DatetimeIndex,
    n_folds: int = N_CV_FOLDS,
    purge_days: int = PURGE_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged walk-forward splits.

    Each fold uses an expanding training window and a fixed-size OOS
    validation window.  `purge_days` bars are dropped at the train/val
    boundary to prevent label leakage through overlapping feature windows.
    `embargo_days` are dropped at the start of the next fold's train set.

    Returns list of (train_idx, val_idx) index arrays.
    """
    n = len(dates)
    fold_size  = n // (n_folds + 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(1, n_folds + 1):
        val_end   = min((k + 1) * fold_size, n)
        val_start = k * fold_size
        train_end = val_start - purge_days

        if train_end <= 0 or val_start >= n:
            continue

        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(val_start, val_end)

        # Embargo: remove first `embargo_days` of each next fold from
        # training to prevent contamination from the previous fold's OOS
        if k > 1:
            train_idx = train_idx[train_idx >= embargo_days * (k - 1)]

        splits.append((train_idx, val_idx))

    logger.debug("Generated %d walk-forward folds.", len(splits))
    return splits


# ─────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────

def _make_objective(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    symbol: str,
    c_low: float,
    c_high: float,
    gamma_low: float,
    gamma_high: float,
    fast_only: bool = False,
) -> callable:
    splits = _make_wf_splits(dates)

    def objective(trial: optuna.Trial) -> float:
        C     = trial.suggest_float("C",     c_low,     c_high,     log=True)
        gamma = trial.suggest_float("gamma", gamma_low, gamma_high, log=True)

        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx],   y[val_idx]

            # Drop NaN labels and NaN features (defensive — X should
            # already be NaN-free after the upstream filter in train_svm,
            # but checking here too avoids a silent 1.0 fallback score).
            tr_mask = ~np.isnan(y_tr) & ~np.isnan(X_tr).any(axis=1)
            va_mask = ~np.isnan(y_va) & ~np.isnan(X_va).any(axis=1)
            if tr_mask.sum() < 50 or va_mask.sum() < 10:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[tr_mask])
            X_va_s = scaler.transform(X_va[va_mask])

            clf = SVC(
                C=C, gamma=gamma, kernel="rbf",
                class_weight="balanced", cache_size=500,
                random_state=42, max_iter=5000,
            )
            try:
                clf.fit(X_tr_s, y_tr[tr_mask].astype(int))
                preds = clf.predict(X_va_s)
                score = f1_score(y_va[va_mask].astype(int), preds,
                                 average="macro", zero_division=0)
                fold_scores.append(score)
            except Exception as e:
                logger.debug("Trial fold failed: %s", e)
                continue

        return -np.mean(fold_scores) if fold_scores else 1.0  # minimise negative F1

    return objective


# ─────────────────────────────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────────────────────────────

def train_svm(
    symbol: str,
    feature_df: pd.DataFrame,
    train_start: str = "2016-01-01",
    train_end:   str = "2019-12-31",
    val_start:   str = "2018-01-01",
    val_end:     Optional[str] = None,
    n_trials: int = 50,
    fast_only: bool = False,
    prior_result: Optional[SVMResult] = None,
    cfg: Optional[dict] = None,
) -> SVMResult:
    """
    Train (or incrementally refit) the SVM for one symbol.

    Parameters
    ----------
    symbol       : NSE ticker string.
    feature_df   : Output of build_svm_features() — 25 features + 'label'.
    train_start  : Inclusive start date for training data.
    train_end    : Inclusive end date for training data.
    val_start    : Inclusive start date for the held-out OOS validation
                   window. Independent of train_end — during the initial
                   HP search, train_end is "2017-12-31" (in-sample) while
                   validation runs 2018-2019, so these must NOT be tied
                   together or the validation window collapses to empty.
    val_end      : Inclusive end date for OOS validation. Defaults to
                   "2019-12-31" (the initial-search OOS period) if not
                   given. During the monthly backtest loop, pass the
                   current month-end explicitly instead.
    n_trials     : Number of Optuna trials.
    fast_only    : If True, only tune C and gamma (FAST params);
                   otherwise do full HP search.
    prior_result : Previous SVMResult to fall back on if gate fails.
    cfg          : Optional config dict (overrides module-level defaults).

    Returns
    -------
    SVMResult with best model, scaler, and metrics.
    """
    cfg = cfg or {}
    c_low   = cfg.get("C_low",     0.01)
    c_high  = cfg.get("C_high",    200.0)
    g_low   = cfg.get("gamma_low", 1e-4)
    g_high  = cfg.get("gamma_high",2.0)

    # Cap-segment prior: narrow search range for large-caps
    if symbol in LARGE_CAP_SYMS:
        c_low  = max(c_low,  cfg.get("large_cap_C_min", 10.0))
        c_high = min(c_high, cfg.get("large_cap_C_max", 100.0))
    else:
        c_low  = max(c_low,  cfg.get("mid_cap_C_min", 0.1))
        c_high = min(c_high, cfg.get("mid_cap_C_max", 10.0))

    logger.info(
        "[%s] SVM training: %s → %s | trials=%d | %s | C=[%.2f, %.2f]",
        symbol, train_start, train_end, n_trials,
        "fast" if fast_only else "full", c_low, c_high,
    )

    # ── Slice training window ─────────────────────────────────────────
    mask = (
        (feature_df.index >= pd.Timestamp(train_start)) &
        (feature_df.index <= pd.Timestamp(train_end))
    )
    df_train = feature_df.loc[mask].copy()

    # Drop rows where the label is NaN (no forward-return target) AND
    # rows where ANY feature is NaN. Feature NaNs come from rolling
    # warm-up windows (252-day z-score, ATR-14, etc.) at the start of
    # the series — these survive a label-only filter since the label
    # is forward-looking and becomes valid well before the slowest
    # feature's warm-up period ends. Letting them through silently
    # corrupts every CV fold (StandardScaler/SVC on NaN) and crashes
    # the final model.fit() call.
    valid_mask = df_train["label"].notna() & df_train[FEATURE_COLS].notna().all(axis=1)
    n_dropped_feat_nan = (df_train["label"].notna() & ~df_train[FEATURE_COLS].notna().all(axis=1)).sum()
    if n_dropped_feat_nan > 0:
        logger.info(
            "[%s] Dropping %d rows with valid label but NaN feature(s) "
            "(rolling-window warm-up period).", symbol, n_dropped_feat_nan,
        )
    df_train   = df_train.loc[valid_mask]

    if len(df_train) < 100:
        raise ValueError(
            f"[{symbol}] Insufficient training data: {len(df_train)} rows after filtering."
        )

    X = df_train[FEATURE_COLS].values.astype(np.float32)
    y = df_train["label"].values.astype(float)
    dates = df_train.index

    # ── Optuna study ──────────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Warm-start with prior best HP if available
    if prior_result is not None and fast_only:
        study.enqueue_trial({"C": prior_result.best_C, "gamma": prior_result.best_gamma})

    objective = _make_objective(
        X, y, dates, symbol, c_low, c_high, g_low, g_high, fast_only
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_C     = study.best_params["C"]
    best_gamma = study.best_params["gamma"]
    best_f1    = -study.best_value

    logger.info("[%s] Best HP: C=%.4f, gamma=%.6f, CV F1=%.4f",
                symbol, best_C, best_gamma, best_f1)

    # ── Final model: retrain on full training window ──────────────────
    valid_mask_np = ~np.isnan(y)
    X_clean = X[valid_mask_np]
    y_clean = y[valid_mask_np].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    final_model = SVC(
        C=best_C, gamma=best_gamma, kernel="rbf",
        class_weight="balanced", probability=True,
        cache_size=500, random_state=42, max_iter=10000,
    )
    final_model.fit(X_scaled, y_clean)

    # ── Validation report on held-out OOS window ───────────────────────
    # Spec requires a 30-trading-day purge gap between train_end and
    # val_start (calendar-adjacent dates leak via rolling-window features
    # that span the boundary, e.g. a 252-day z-score computed on
    # 2018-01-01 still reaches back into December 2017 training data).
    # Compute the purge using actual trading-day positions in the full
    # index, not calendar days, since calendar-day counts don't match
    # trading-day counts (weekends/holidays).
    all_dates = feature_df.index.sort_values()
    train_end_ts = pd.Timestamp(train_end)
    pos_after_train_end = all_dates.searchsorted(train_end_ts, side="right")
    purge_pos = pos_after_train_end + PURGE_DAYS
    if purge_pos < len(all_dates):
        purged_val_start_ts = all_dates[purge_pos]
    else:
        purged_val_start_ts = pd.Timestamp(val_start)

    requested_val_start_ts = pd.Timestamp(val_start)
    effective_val_start_ts = max(requested_val_start_ts, purged_val_start_ts)
    if effective_val_start_ts > requested_val_start_ts:
        logger.info(
            "[%s] val_start pushed from %s to %s to enforce %d-day purge "
            "gap after train_end=%s.",
            symbol, val_start, effective_val_start_ts.date(), PURGE_DAYS, train_end,
        )

    val_end_resolved = val_end or "2019-12-31"
    val_mask  = (
        (feature_df.index >= effective_val_start_ts) &
        (feature_df.index <= pd.Timestamp(val_end_resolved)) &
        feature_df["label"].notna() &
        feature_df[FEATURE_COLS].notna().all(axis=1)
    )
    val_df = feature_df.loc[val_mask]
    val_f1 = 0.0
    val_report = ""

    if len(val_df) > 30:
        X_val = scaler.transform(val_df[FEATURE_COLS].values.astype(np.float32))
        y_val = val_df["label"].values.astype(int)
        y_pred = final_model.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        val_report = classification_report(y_val, y_pred,
                                            target_names=["Sell", "Flat", "Buy"],
                                            zero_division=0)
        logger.info("[%s] Val F1 (OOS): %.4f", symbol, val_f1)
    else:
        logger.warning(
            "[%s] OOS validation window %s→%s produced only %d rows "
            "(need >30) — val_f1 left at 0.0, not a real measurement.",
            symbol, val_start, val_end_resolved, len(val_df),
        )

    return SVMResult(
        symbol       = symbol,
        best_C       = best_C,
        best_gamma   = best_gamma,
        best_f1      = best_f1,
        val_f1       = val_f1,
        val_report   = val_report,
        model        = final_model,
        scaler       = scaler,
        train_end    = train_end,
        hp_search_type = "fast" if fast_only else "full",
    )


# ─────────────────────────────────────────────────────────────────────
# Monthly incremental refit (called by monthly_loop.py)
# ─────────────────────────────────────────────────────────────────────

def monthly_refit_svm(
    symbol: str,
    feature_df: pd.DataFrame,
    registry: SVMRegistry,
    current_month_end: str,
    trigger_hp_search: bool = False,
    cfg: Optional[dict] = None,
) -> SVMResult:
    """
    Monthly incremental SVM refit on expanding window.

    Logic:
      - Always refit weights on expanding window (Jan 2016 → current_month_end).
      - HP re-search (Optuna) only if `trigger_hp_search=True`
        (quarterly calendar OR regime drift flag set).
      - Otherwise reuse prior C, gamma and skip Optuna entirely.

    Parameters
    ----------
    symbol           : NSE ticker.
    feature_df       : Full feature matrix up to current_month_end.
    registry         : SVMRegistry holding prior best result.
    current_month_end: End date for training (prior month-end in backtest).
    trigger_hp_search: True → run Optuna; False → reuse prior HP.

    Returns
    -------
    Updated SVMResult (new weights, potentially new HP).
    """
    prior = registry.get(symbol)
    cfg   = cfg or {}

    n_trials = cfg.get("n_trials_fast", 30) if trigger_hp_search else 0

    if trigger_hp_search or prior is None:
        # OOS check window: trailing ~63 trading days (~90 calendar days)
        # up to current_month_end, per the drift-sentinel spec (§3 step 05).
        # Without this, train_svm would silently fall back to the stale
        # 2018-2019 initial-search validation window every single month.
        month_end_ts = pd.Timestamp(current_month_end)
        val_start_roll = (month_end_ts - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

        result = train_svm(
            symbol       = symbol,
            feature_df   = feature_df,
            train_start  = "2016-01-01",
            train_end    = current_month_end,
            val_start    = val_start_roll,
            val_end      = current_month_end,
            n_trials     = n_trials if n_trials > 0 else 30,
            fast_only    = True,
            prior_result = prior,
            cfg          = cfg,
        )
    else:
        # Reuse prior HP — just refit on expanded data
        logger.info("[%s] Monthly refit: reusing HP C=%.4f gamma=%.6f | end=%s",
                    symbol, prior.best_C, prior.best_gamma, current_month_end)

        mask = (
            (feature_df.index >= pd.Timestamp("2016-01-01")) &
            (feature_df.index <= pd.Timestamp(current_month_end)) &
            feature_df["label"].notna() &
            feature_df[FEATURE_COLS].notna().all(axis=1)
        )
        df_tr  = feature_df.loc[mask]
        X      = df_tr[FEATURE_COLS].values.astype(np.float32)
        y      = df_tr["label"].values.astype(int)

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        model = SVC(
            C=prior.best_C, gamma=prior.best_gamma, kernel="rbf",
            class_weight="balanced", probability=True,
            cache_size=500, random_state=42, max_iter=10000,
        )
        model.fit(X_s, y)

        result = SVMResult(
            symbol         = symbol,
            best_C         = prior.best_C,
            best_gamma     = prior.best_gamma,
            best_f1        = prior.best_f1,
            val_f1         = prior.val_f1,
            val_report     = prior.val_report,
            model          = model,
            scaler         = scaler,
            train_end      = current_month_end,
            hp_search_type = "reuse",
        )

    registry.update(result)
    return result


# ─────────────────────────────────────────────────────────────────────
# Initial HP search: Jan 2016 – Dec 2019 for all symbols
# ─────────────────────────────────────────────────────────────────────

def run_initial_hp_search(
    all_features: Dict[str, pd.DataFrame],
    output_dir: Path,
    cfg: Optional[dict] = None,
) -> SVMRegistry:
    """
    Run full Optuna HP search for all 10 symbols on 2016–2019 data.
    This is the Week 2–3 deliverable.

    Parameters
    ----------
    all_features : {symbol: feature_df} from build_svm_features().
    output_dir   : Where to save SVMRegistry pickle.
    cfg          : Config dict.

    Returns
    -------
    SVMRegistry populated with initial best results for all symbols.
    """
    cfg = cfg or {}
    registry = SVMRegistry()

    for symbol, feat_df in all_features.items():
        logger.info("=" * 60)
        logger.info("[%s] Starting initial HP search (2016–2019)", symbol)
        try:
            result = train_svm(
                symbol      = symbol,
                feature_df  = feat_df,
                train_start = "2016-01-01",
                train_end   = "2017-12-31",   # HP search in-sample
                n_trials    = cfg.get("n_trials_full", 50),
                fast_only   = False,
                cfg         = cfg,
            )
            registry.update(result)
            logger.info(
                "[%s] ✓ HP search complete | C=%.4f | gamma=%.6f | "
                "CV F1=%.4f | Val F1=%.4f",
                symbol, result.best_C, result.best_gamma,
                result.best_f1, result.val_f1,
            )
        except Exception as e:
            logger.error("[%s] HP search FAILED: %s", symbol, e, exc_info=True)

    # Save registry
    out_path = output_dir / "svm_registry_initial.pkl"
    registry.save(out_path)
    logger.info("SVMRegistry saved → %s", out_path)

    # Print summary table
    _print_summary(registry)
    return registry


def _print_summary(registry: SVMRegistry) -> None:
    """Pretty-print the HP search summary table."""
    rows = []
    for sym, res in registry.models.items():
        cap = "LC" if sym in LARGE_CAP_SYMS else "MC"
        rows.append({
            "Symbol":       sym,
            "Cap":          cap,
            "C":            f"{res.best_C:.4f}",
            "gamma":        f"{res.best_gamma:.6f}",
            "CV F1 (macro)":f"{res.best_f1:.4f}",
            "Val F1 (OOS)": f"{res.val_f1:.4f}",
            "HP Type":      res.hp_search_type,
            "Train end":    res.train_end,
        })

    df = pd.DataFrame(rows)
    logger.info("\n%s", df.to_string(index=False))
    print("\n" + "═" * 80)
    print("SVM Initial HP Search Results — QBEAST-AI.N Week 2–3")
    print("═" * 80)
    print(df.to_string(index=False))
    print("═" * 80 + "\n")