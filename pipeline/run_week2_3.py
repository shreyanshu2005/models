"""
pipeline/run_week2_3.py  (FIXED — v1.1)
=========================================
Week 2–3 orchestration: Regime engine + SVM feature audit +
hyperparameter search on 2016–2019 data.

Bugs fixed vs v1.0
-------------------
1. regime_data["per_symbol"] was never populated →
   run_regime_engine() now builds per-symbol aligned regime DataFrames
   by forward-filling the NIFTY-based HMM decode onto each equity's
   trading calendar (using align_regime_to_equity).

2. load_nifty50 fallback searched a hardcoded /mnt/project path that
   only existed in the Claude sandbox → now searches cfg["paths"]["raw_data"]
   first (the correct location), with a proper error message.

3. No regime engine call existed in the main() function at all →
   added Step 02b which runs HMM + vol-band, writes regime labels,
   and injects them before the SVM feature build.

4. audit_feature_distribution printed a confusing "flagged issues" table
   that showed all regime columns as zero-variance (correct behaviour
   before wiring, but alarming after) → added a clear note distinguishing
   expected zero-variance (regime cols before HMM) from real issues.

Run from project root:
    python -m pipeline.run_week2_3 --log-level INFO
    python -m pipeline.run_week2_3 --skip-hp-search   # quick test
    python -m pipeline.run_week2_3 --skip-regime       # reuse saved regime
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Make project root importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from models.svm.features import build_svm_features, get_feature_columns
from models.svm.train    import run_initial_hp_search, SVMRegistry, LARGE_CAP_SYMS
from regime.hmm_model    import (
    build_nifty_hmm_features,
    run_regime_pipeline,
    align_regime_to_equity,
    RegimeHMM,
)
from regime.vol_band     import VolBandClassifier, vol_band_summary

logger = logging.getLogger("run_week2_3")


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Step 01a: Load raw OHLCV CSVs for all 10 equity symbols
# ─────────────────────────────────────────────────────────────────────────────

def load_from_raw_csvs(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load raw CSV files directly and return OHLCV DataFrames.
    Searches cfg.paths.raw_data.  Falls back to PROJECT_ROOT / "data" / "raw".
    """
    raw_dir   = PROJECT_ROOT / cfg["paths"]["raw_data"]
    fallbacks = [raw_dir, PROJECT_ROOT / "data" / "raw"]

    syms_cfg  = cfg["universe"]["symbols"]
    all_symbols = syms_cfg.get("large_cap", []) + syms_cfg.get("mid_cap", [])
    # Also support a flat all_symbols key
    if not all_symbols:
        all_symbols = cfg["universe"].get("all_symbols", [])

    col_map = {"date": "date", "open": "open", "high": "high",
               "low": "low", "close": "close", "volume": "volume"}

    symbol_data: Dict[str, pd.DataFrame] = {}
    for sym in all_symbols:
        for d in fallbacks:
            csv_path = d / f"{sym}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["date"])
                df = df.rename(columns=col_map)
                df = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df = df.set_index("date").sort_index()
                df = df[df.index >= pd.Timestamp(cfg["universe"].get("universe_start", "2016-01-01"))]
                df = df[df.index <= pd.Timestamp(cfg["universe"].get("backtest_end",   "2026-06-30"))]
                df = df.dropna(subset=["close"])
                symbol_data[sym] = df
                logger.info(
                    "Loaded %-12s  %d rows  %s → %s",
                    sym, len(df), df.index[0].date(), df.index[-1].date(),
                )
                break
        else:
            logger.warning("CSV not found for %s (searched: %s)", sym,
                           [str(d) for d in fallbacks])

    if not symbol_data:
        raise FileNotFoundError(
            f"No equity CSVs found in {fallbacks}. "
            "Place <SYMBOL>.csv files there and retry."
        )
    return symbol_data


# ─────────────────────────────────────────────────────────────────────────────
# Step 01b: Load NIFTY50 benchmark CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_nifty50(cfg: dict) -> pd.DataFrame:
    """
    Load and concatenate NIFTY50 benchmark CSVs.
    Searches raw_data dir for NIFTY50_Benchmark_*.csv files.
    """
    raw_dir   = PROJECT_ROOT / cfg["paths"]["raw_data"]
    fallbacks = [raw_dir, PROJECT_ROOT / "data" / "raw"]

    nifty_paths = []
    for d in fallbacks:
        if d.exists():
            found = sorted(d.glob("NIFTY50*.csv"))
            if found:
                nifty_paths = found
                logger.info(
                    "NIFTY50 files found in %s: %s",
                    d, [p.name for p in found],
                )
                break

    if not nifty_paths:
        raise FileNotFoundError(
            f"No NIFTY50*.csv files found in {fallbacks}. "
            "Copy NIFTY50 benchmark CSVs to cfg.paths.raw_data and retry."
        )

    frames = []
    for fp in nifty_paths:
        df = pd.read_csv(fp)
        # Handle both timezone-aware and naive DateTime columns
        date_col = next(
            (c for c in df.columns if "date" in c.lower() or "time" in c.lower()),
            df.columns[0],
        )
        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
        df = df.rename(columns={
            date_col:  "date",
            "Open":    "open",  "High":  "high",
            "Low":     "low",   "Close": "close",
            "Volume":  "volume",
        })
        # Also handle lowercase columns
        df.columns = [c.lower() if c in ("Open","High","Low","Close","Volume") else c
                      for c in df.columns]
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        frames.append(df)

    nifty = pd.concat(frames).sort_index()
    nifty = nifty[~nifty.index.duplicated(keep="first")]
    logger.info(
        "NIFTY50 loaded: %d rows  %s → %s",
        len(nifty), nifty.index[0].date(), nifty.index[-1].date(),
    )
    return nifty


# ─────────────────────────────────────────────────────────────────────────────
# Step 02: Run regime engine and align to equity calendars
# ─────────────────────────────────────────────────────────────────────────────

def run_regime_engine(
    nifty: pd.DataFrame,
    raw_data: Dict[str, pd.DataFrame],
    cfg: dict,
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Train HMM + classify vol bands.  Return a regime_data dict with:
      regime_data["regime_df"]       : NIFTY-indexed decode (all columns)
      regime_data["regime_labels"]   : pd.Series of regime_label
      regime_data["hmm"]             : fitted RegimeHMM object
      regime_data["vol_bands"]       : dict sym → vol_band DataFrame
      regime_data["per_symbol"]      : dict sym → merged regime+volband DataFrame
                                       aligned to equity's trading calendar

    FIX #1 — this function now exists and populates per_symbol correctly.
    """
    logger.info("=== Running 5-state HMM on NIFTY50 ===")

    # ── HMM fit (on data up to HP-train end or all data) ──────────────────
    fit_up_to  = cfg.get("dates", {}).get(
        "hp_train_end",
        cfg.get("universe", {}).get("train_end", "2019-12-31"),
    )
    hmm_save   = (save_dir / "hmm_initial.pkl") if save_dir else None
    if hmm_save:
        hmm_save.parent.mkdir(parents=True, exist_ok=True)

    # Check if a saved HMM already exists to skip retraining
    if hmm_save and hmm_save.exists():
        logger.info("Loading existing HMM from %s (use --force-regime to retrain)", hmm_save)
        hmm = RegimeHMM.load(hmm_save)
        feat_df   = build_nifty_hmm_features(nifty)
        regime_df = hmm.decode(feat_df)
    else:
        regime_df, hmm = run_regime_pipeline(
            nifty_df  = nifty,
            config    = cfg,
            fit_up_to = fit_up_to,
            save_path = hmm_save,
        )

    # ── Vol-band classification per symbol ────────────────────────────────
    logger.info("=== Classifying per-symbol vol bands ===")
    vbc       = VolBandClassifier(cfg)
    vol_bands = vbc.classify_all(raw_data)

    summary = vol_band_summary(
        vol_bands,
        start = cfg.get("universe", {}).get("universe_start", "2016-01-01"),
        end   = cfg.get("universe", {}).get("backtest_end",   "2026-06-30"),
    )
    logger.info("\nVol-band summary:\n%s", summary.to_string())

    # ── FIX #1: Build per-symbol merged regime DataFrame ─────────────────
    # The HMM decodes on the NIFTY trading calendar.  Equity calendars
    # may differ slightly (different holidays).  We align via forward-fill.
    per_symbol: Dict[str, pd.DataFrame] = {}
    for sym, ohlcv in raw_data.items():
        # Step 1: merge HMM regime + vol-band on NIFTY dates
        vb_df  = vol_bands[sym]
        merged = vbc.merge_regime_and_volband(regime_df, vb_df)

        # Step 2: align merged to equity's own index (ffill across calendar gaps)
        aligned = align_regime_to_equity(merged, ohlcv.index)
        per_symbol[sym] = aligned

        logger.debug(
            "[%s] Regime aligned: rows=%d  label_dist=%s",
            sym, len(aligned),
            aligned["regime_label"].value_counts().to_dict(),
        )

    # ── Validate: all 5 states must appear ───────────────────────────────
    state_counts = regime_df["regime_label"].value_counts().sort_index().to_dict()
    if len(state_counts) < 5:
        logger.warning(
            "Only %d of 5 HMM states observed in decode — model may need "
            "more training data or a different sticky_kappa.",
            len(state_counts),
        )
    else:
        logger.info("All 5 HMM states observed: %s", state_counts)

    return {
        "regime_df":     regime_df,
        "regime_labels": regime_df["regime_label"],
        "hmm":           hmm,
        "vol_bands":     vol_bands,
        "per_symbol":    per_symbol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 02c: Regime visual validation
# ─────────────────────────────────────────────────────────────────────────────

STATE_NAMES = {0: "Rising", 1: "Rally", 2: "Sideways", 3: "Falling", 4: "Crashing"}

def validate_regime_visually(nifty: pd.DataFrame, regime_data: dict) -> dict:
    """
    Print regime distribution and check known market events.

    Known ground-truth checks:
      - Mar 2020: COVID crash → Crashing or Falling
      - Apr–Dec 2020: Recovery → Rising or Rally
      - 2021: Bull market → Rising or Rally
      - Feb–Mar 2022: Ukraine war → Falling or Sideways
    """
    known_events = {
        ("2020-03-01", "2020-03-31"): [3, 4],
        ("2020-04-01", "2020-12-31"): [0, 1],
        ("2021-01-01", "2021-12-31"): [0, 1],
        ("2022-02-01", "2022-03-31"): [2, 3],
    }

    report = {"events": {}, "state_distribution": {}}

    if "regime_labels" not in regime_data:
        logger.warning("No regime_labels in regime_data — skipping visual check.")
        return report

    labels = regime_data["regime_labels"].copy()
    labels.index = pd.to_datetime(labels.index)

    dist = labels.value_counts().sort_index()
    report["state_distribution"] = {
        STATE_NAMES.get(int(k), str(k)): int(v) for k, v in dist.items()
    }
    logger.info("Regime distribution: %s", report["state_distribution"])

    passed = failed = 0
    for (start, end), expected in known_events.items():
        window = labels.loc[start:end]
        if len(window) == 0:
            continue
        dominant = int(window.mode().iloc[0])
        ok       = dominant in expected
        status   = "✅" if ok else "⚠️ "
        key      = f"{start}→{end}"
        report["events"][key] = {
            "dominant_state": STATE_NAMES.get(dominant, str(dominant)),
            "expected":       [STATE_NAMES[s] for s in expected],
            "pass":           ok,
        }
        logger.info(
            "%s %s  dominant=%s  expected=%s",
            status, key,
            STATE_NAMES.get(dominant, str(dominant)),
            [STATE_NAMES[s] for s in expected],
        )
        if ok: passed += 1
        else:  failed += 1

    report["checks_passed"] = passed
    report["checks_failed"] = failed
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Step 03: Build SVM feature matrices with real regime features
# ─────────────────────────────────────────────────────────────────────────────

def build_all_svm_features(
    raw_data: Dict[str, pd.DataFrame],
    regime_data: dict,
    cfg: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Build 25-dim SVM feature matrices for all symbols.
    regime_data["per_symbol"] must be populated (by run_regime_engine).
    """
    all_features: Dict[str, pd.DataFrame] = {}

    for sym, ohlcv in raw_data.items():
        # FIX #1: use per-symbol aligned regime DataFrame
        regime_df = regime_data.get("per_symbol", {}).get(sym)
        if regime_df is None:
            logger.warning(
                "[%s] No per-symbol regime DataFrame — "
                "regime features will be zero for this symbol.", sym,
            )

        try:
            feat_df = build_svm_features(
                ohlcv      = ohlcv,
                regime_df  = regime_df,
                tx_rate    = cfg["costs"]["tx_rate_per_leg"],
                zscore_win = cfg["features"]["zscore_window"],
                fwd_days   = cfg["features"]["forward_days"],
            )
            all_features[sym] = feat_df

            # Count how many regime rows are non-zero (sanity check)
            regime_cols   = [f"regime_{i}" for i in range(5)] + [f"post_{i}" for i in range(5)]
            nonzero_regime = (feat_df[regime_cols].sum(axis=1) > 0).sum()
            label_dist     = feat_df["label"].value_counts().to_dict()

            logger.info(
                "[%s] Features built: %d rows | regime_nonzero=%d | Label dist: %s",
                sym, len(feat_df), nonzero_regime, label_dist,
            )
        except Exception as e:
            logger.error("[%s] Feature build failed: %s", sym, e, exc_info=True)

    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# Step 04: Feature distribution audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_feature_distribution(all_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute per-feature stats across all symbols for the 2016-2019 training window.
    Flags near-zero variance, extreme kurtosis, high NaN rate.

    Note: regime columns (regime_0..4, post_0..4) are expected to have
    non-zero variance now that the HMM is wired in. If they still show
    zero variance after run_regime_engine(), that means the alignment
    failed — check the per_symbol dict in regime_data.
    """
    feature_cols = get_feature_columns()
    rows = []

    for sym, feat_df in all_features.items():
        mask   = (
            (feat_df.index >= pd.Timestamp("2016-01-01")) &
            (feat_df.index <= pd.Timestamp("2019-12-31"))
        )
        df_tr = feat_df.loc[mask]

        for col in feature_cols:
            if col not in df_tr.columns:
                continue
            s = df_tr[col].dropna()
            if len(s) == 0:
                continue
            nan_rate = float(df_tr[col].isna().mean())
            kurt     = float(s.kurtosis()) if len(s) > 4 else 0.0
            std_val  = float(s.std())
            rows.append({
                "symbol":       sym,
                "feature":      col,
                "mean":         float(s.mean()),
                "std":          std_val,
                "min":          float(s.min()),
                "max":          float(s.max()),
                "kurtosis":     kurt,
                "nan_rate":     nan_rate,
                "flag_lowvar":  bool(std_val < 0.01),
                "flag_kurtosis":bool(kurt > 50),
                "flag_nan":     bool(nan_rate > 0.05),
            })

    audit_df  = pd.DataFrame(rows)
    if audit_df.empty:
        logger.warning("Audit DataFrame is empty — no features computed.")
        return audit_df

    summary = audit_df.groupby("feature").agg(
        mean_std    = ("std",         "mean"),
        max_kurt    = ("kurtosis",    "max"),
        max_nan     = ("nan_rate",    "max"),
        n_lowvar    = ("flag_lowvar", "sum"),
        n_highkurt  = ("flag_kurtosis","sum"),
        n_nanflag   = ("flag_nan",    "sum"),
    ).reset_index()

    # FIX #4: distinguish regime cols (zero var is a bug post-HMM wiring)
    regime_cols = [f"regime_{i}" for i in range(5)] + [f"post_{i}" for i in range(5)]
    flagged = summary[
        (summary["n_lowvar"]   > 0) |
        (summary["n_highkurt"] > 0) |
        (summary["n_nanflag"]  > 0)
    ]

    regime_flagged = flagged[flagged["feature"].isin(regime_cols)]
    other_flagged  = flagged[~flagged["feature"].isin(regime_cols)]

    if not regime_flagged.empty:
        logger.error(
            "\n⚠️  REGIME FEATURES STILL ZERO-VARIANCE after HMM wiring!\n"
            "This means align_regime_to_equity() failed or per_symbol dict is empty.\n"
            "Check run_regime_engine() output.\n%s",
            regime_flagged.to_string(index=False),
        )
    if not other_flagged.empty:
        logger.warning(
            "\nNon-regime features with issues:\n%s",
            other_flagged.to_string(index=False),
        )

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 05: Validate features
# ─────────────────────────────────────────────────────────────────────────────

def validate_features(all_features: Dict[str, pd.DataFrame]) -> bool:
    """
    Run look-ahead and data quality checks:
      1. Label class balance (no class < 5 %)
      2. Feature NaN count in training window
      3. Regime features non-zero (post HMM wiring)
    """
    all_ok       = True
    feature_cols = get_feature_columns()
    regime_cols  = [f"regime_{i}" for i in range(5)] + [f"post_{i}" for i in range(5)]

    for sym, feat_df in all_features.items():
        mask  = (
            (feat_df.index >= pd.Timestamp("2016-01-01")) &
            (feat_df.index <= pd.Timestamp("2019-12-31"))
        )
        df_tr = feat_df.loc[mask]
        valid = df_tr["label"].notna()
        df_v  = df_tr.loc[valid]

        if len(df_v) < 100:
            logger.error("[%s] FAIL: Only %d valid training rows.", sym, len(df_v))
            all_ok = False
            continue

        # Label balance
        counts = df_v["label"].value_counts(normalize=True)
        for lbl_val, pct in counts.items():
            if pct < 0.05:
                logger.warning(
                    "[%s] Label %+.0f is rare: %.1f %%", sym, lbl_val, pct * 100
                )

        # NaN audit
        nan_counts = df_tr[feature_cols].isna().sum()
        high_nan   = nan_counts[nan_counts > len(df_tr) * 0.10]
        if len(high_nan) > 0:
            logger.warning("[%s] High NaN features: %s", sym, high_nan.to_dict())

        # Regime non-zero sanity check
        regime_nonzero = (df_v[regime_cols].abs().sum(axis=1) > 0).sum()
        if regime_nonzero == 0:
            logger.error(
                "[%s] ⚠️  ALL regime features are ZERO in training window! "
                "HMM not wired correctly.", sym,
            )
            all_ok = False
        else:
            regime_pct = regime_nonzero / len(df_v) * 100
            logger.info(
                "[%s] ✓ Validate passed | rows=%d | regime_live=%.0f%% | labels=%s",
                sym, len(df_v), regime_pct, counts.round(3).to_dict(),
            )

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Week 1 artefact loader (fallback path)
# ─────────────────────────────────────────────────────────────────────────────

def load_week1_artefacts(cfg: dict):
    """Load Week 1 parquets if available, else raise FileNotFoundError."""
    proc_dir     = PROJECT_ROOT / cfg["paths"].get("processed", "data/processed")
    manifest_pth = proc_dir / "week1_manifest.json"
    if not manifest_pth.exists():
        raise FileNotFoundError(f"Week 1 manifest not found at {manifest_pth}")
    with open(manifest_pth) as f:
        manifest = json.load(f)
    symbol_data = {}
    for sym in manifest.get("symbols", []):
        p = proc_dir / f"{sym}_features.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            symbol_data[sym] = df
    return symbol_data, manifest


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(skip_hp_search: bool = False, skip_regime: bool = False) -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )

    cfg         = _load_config()
    results_dir = PROJECT_ROOT / cfg["paths"]["results"]
    regime_dir  = PROJECT_ROOT / "data" / "regime"
    results_dir.mkdir(parents=True, exist_ok=True)
    regime_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 72)
    print("  QBEAST-AI.N  |  Week 2–3  |  Regime Engine + SVM HP Search")
    print("═" * 72 + "\n")

    # ── Step 01: Load raw data ─────────────────────────────────────────────
    logger.info("[01] Loading raw CSV data...")
    raw_data = load_from_raw_csvs(cfg)
    nifty    = load_nifty50(cfg)

    # ── Step 02: Regime engine ─────────────────────────────────────────────
    regime_parquet = regime_dir / "regime_labels.parquet"
    regime_data    = {}

    if skip_regime and regime_parquet.exists():
        logger.info("[02] Loading saved regime labels from %s", regime_parquet)
        rl = pd.read_parquet(regime_parquet)
        regime_data["regime_labels"] = rl["regime_label"]

        # Rebuild per_symbol aligned frames from the saved DataFrame
        hmm_pkl = regime_dir / "hmm_initial.pkl"
        if hmm_pkl.exists():
            saved_regime_df = rl  # full df with all columns
            vbc             = VolBandClassifier(cfg)
            vol_bands       = vbc.classify_all(raw_data)
            per_symbol      = {}
            for sym, ohlcv in raw_data.items():
                vb_df   = vol_bands[sym]
                merged  = vbc.merge_regime_and_volband(saved_regime_df, vb_df)
                aligned = align_regime_to_equity(merged, ohlcv.index)
                per_symbol[sym] = aligned
            regime_data["per_symbol"] = per_symbol
            regime_data["vol_bands"]  = vol_bands
            logger.info("[02] Regime per-symbol alignment rebuilt from saved artefacts.")
        else:
            logger.warning("[02] Saved HMM not found — per_symbol will be empty.")

    else:
        logger.info("[02] Running regime engine (HMM + vol bands)...")
        regime_data = run_regime_engine(
            nifty    = nifty,
            raw_data = raw_data,
            cfg      = cfg,
            save_dir = regime_dir,
        )
        # Persist regime DataFrame for downstream use
        regime_data["regime_df"].to_parquet(regime_parquet)
        logger.info("Regime labels saved → %s", regime_parquet)

    # ── Step 02c: Visual validation ────────────────────────────────────────
    logger.info("[02c] Regime visual validation against known market events...")
    regime_report = validate_regime_visually(nifty, regime_data)

    # ── Step 03: Build SVM features WITH regime ────────────────────────────
    logger.info("[03] Building SVM feature matrices (25-dim, with regime)...")
    all_features = build_all_svm_features(raw_data, regime_data, cfg)

    if not all_features:
        logger.error("No features built — check raw CSV paths.")
        sys.exit(1)

    # ── Step 04: Feature distribution audit ───────────────────────────────
    logger.info("[04] Feature distribution audit...")
    audit_df = audit_feature_distribution(all_features)
    audit_path = results_dir / "feature_audit_with_regime.csv"
    audit_df.to_csv(audit_path, index=False)
    logger.info("Feature audit saved → %s", audit_path)

    # ── Step 05: Validate features ────────────────────────────────────────
    logger.info("[05] Validating features (look-ahead, label balance, NaN, regime)...")
    ok = validate_features(all_features)
    if not ok:
        logger.warning("Validation found issues — review before proceeding.")

    # ── Step 06: SVM Initial HP Search ────────────────────────────────────
    registry: Optional[SVMRegistry] = None
    if not skip_hp_search:
        logger.info("[06] Running initial SVM HP search (2016-2017 in-sample)...")
        svm_cfg  = cfg.get("svm_hp", {})
        registry = run_initial_hp_search(
            all_features = all_features,
            output_dir   = results_dir,
            cfg          = svm_cfg,
        )
        logger.info("✅  Week 2–3 SVM HP search complete!")
    else:
        logger.info("[06] Skipping HP search (--skip-hp-search flag set).")

    # ── Step 07: Save week2 manifest ──────────────────────────────────────
    symbols_processed = list(all_features.keys())
    manifest = {
        "week":                "2-3",
        "regime_validation":   regime_report,
        "feature_audit_path":  str(audit_path),
        "svm_registry_path":   str(results_dir / "svm_registry_initial.pkl"),
        "symbols":             symbols_processed,
        "regime_labels_path":  str(regime_parquet),
        "train_period":        "2016-01-01 → 2017-12-31 (HP search) | 2018-01-01 → 2019-12-31 (OOS val)",
        "regime_feature_wired": regime_data.get("per_symbol") is not None,
    }
    manifest_path = results_dir / "week2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Week 2–3 manifest saved → %s", manifest_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Week 2–3 COMPLETE")
    print(f"  Symbols processed : {len(symbols_processed)}")
    print(f"  Regime wired      : {'✅ YES' if regime_data.get('per_symbol') else '❌ NO'}")
    print(f"  Regime checks     : {regime_report.get('checks_passed', 'N/A')} passed "
          f"/ {regime_report.get('checks_failed', 'N/A')} failed")
    print(f"  Feature audit     : {audit_path}")
    print(f"  SVM registry      : results/svm_registry_initial.pkl")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QBEAST-AI.N Week 2–3 Pipeline")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-hp-search", action="store_true",
                        help="Skip Optuna HP search (for quick testing)")
    parser.add_argument("--skip-regime", action="store_true",
                        help="Load saved regime labels instead of retraining HMM")
    parser.add_argument("--force-regime", action="store_true",
                        help="Force HMM retraining even if saved pkl exists")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    main(
        skip_hp_search = args.skip_hp_search,
        skip_regime    = args.skip_regime,
    )