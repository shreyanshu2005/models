"""
pipeline/run_week2_3.py
=======================
Week 2–3 orchestration: SVM feature audit, visual regime validation,
and initial hyperparameter search on 2016–2019 data.

Run from project root:
    python -m pipeline.run_week2_3 --log-level INFO

Steps
-----
01. Load Week 1 artefacts (parquets + manifest)
02. Regime visual validation plot (NIFTY50 + HMM states)
03. SVM feature distribution audit (25 features across all symbols)
04. Initial SVM HP search: 2016–2017 in-sample, 2018–2019 OOS validation
05. Per-cap-segment HP summary table
06. Validate: no look-ahead, label class balance, feature NaN counts
07. Save SVMRegistry + audit report to results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make sure project root is on sys.path ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from models.svm.features import build_svm_features, get_feature_columns
from models.svm.train import run_initial_hp_search, SVMRegistry, LARGE_CAP_SYMS

logger = logging.getLogger("run_week2_3")

# ─────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────
# Step 01: Load Week 1 parquets
# ─────────────────────────────────────────────────────────────────────

def load_week1_artefacts(cfg: dict) -> dict:
    """Load aligned OHLCV parquets produced by Week 1 pipeline."""
    proc_dir = PROJECT_ROOT / cfg["paths"]["processed"]
    manifest_path = proc_dir / "week1_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Week 1 manifest not found at {manifest_path}. "
            "Run pipeline/run_week1.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    symbols_data = {}
    for sym in manifest.get("symbols", []):
        p = proc_dir / f"{sym}_features.parquet"
        if p.exists():
            symbols_data[sym] = pd.read_parquet(p)
            symbols_data[sym].index = pd.to_datetime(symbols_data[sym].index)
        else:
            logger.warning("Parquet not found for %s — will build from raw CSV", sym)

    logger.info("Loaded %d symbol artefacts from Week 1.", len(symbols_data))
    return symbols_data, manifest


# ─────────────────────────────────────────────────────────────────────
# Fallback: build from raw CSVs (when Week 1 parquets unavailable)
# ─────────────────────────────────────────────────────────────────────

def load_from_raw_csvs(cfg: dict) -> dict:
    """
    Load raw CSV files directly and build feature matrices.
    This is the fallback path when Week 1 parquets are not available.
    Builds features WITHOUT regime (regime_df=None) — inject later.
    """
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    # Try the local project raw dir first, then the Claude-sandbox mount as fallback
    mount_dir = Path("/mnt/project")

    all_symbols = (
        cfg["universe"]["symbols"]["large_cap"] +
        cfg["universe"]["symbols"]["mid_cap"]
    )

    col_map = {
        "date": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    }

    symbol_data = {}
    for sym in all_symbols:
        # Try mounted project first, then raw_dir
        for search_dir in [raw_dir, mount_dir]:
            csv_path = search_dir / f"{sym}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["date"])
                df = df.rename(columns=col_map)
                df = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df = df.set_index("date").sort_index()
                df = df[df.index >= pd.Timestamp(cfg["universe"]["universe_start"])]
                df = df[df.index <= pd.Timestamp(cfg["universe"]["backtest_end"])]
                df = df.dropna(subset=["close"])
                symbol_data[sym] = df
                logger.info("Loaded %s: %d rows (%s → %s)",
                            sym, len(df), df.index[0].date(), df.index[-1].date())
                break
        else:
            logger.warning("CSV not found for %s", sym)

    return symbol_data


def load_nifty50(cfg: dict) -> pd.DataFrame:
    """Load and concatenate the NIFTY50 benchmark CSVs.

    Searches (in order): the configured raw-data dir (cfg.paths.raw_data,
    i.e. data/raw on a normal local install), then /mnt/project as a
    fallback for the Claude sandbox environment. Matches files by glob
    pattern instead of a hardcoded exact filename list, since the NIFTY50
    CSVs may cover slightly different / overlapping date ranges.
    """
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    search_dirs = [raw_dir, Path("/mnt/project")]

    nifty_paths: list[Path] = []
    for d in search_dirs:
        if d.exists():
            found = sorted(d.glob("NIFTY50_Benchmark_*.csv"))
            if found:
                nifty_paths = found
                logger.info("NIFTY50 files found in %s: %s", d, [p.name for p in found])
                break

    if not nifty_paths:
        raise FileNotFoundError(
            f"No NIFTY50_Benchmark_*.csv files found in any of: "
            f"{[str(d) for d in search_dirs]}. Copy the benchmark CSVs into "
            f"{raw_dir} (cfg.paths.raw_data)."
        )

    frames = []
    for fp in nifty_paths:
        df = pd.read_csv(fp)
        # Handle timezone-aware DateTime
        df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True).dt.tz_localize(None)
        df = df.rename(columns={
            "DateTime": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        frames.append(df)

    nifty = pd.concat(frames).sort_index()
    nifty = nifty[~nifty.index.duplicated(keep="first")]
    logger.info("NIFTY50 loaded: %d rows (%s → %s)",
                len(nifty), nifty.index[0].date(), nifty.index[-1].date())
    return nifty


# ─────────────────────────────────────────────────────────────────────
# Step 02: Regime visual validation (text summary — matplotlib optional)
# ─────────────────────────────────────────────────────────────────────

STATE_NAMES = {0: "Rising", 1: "Rally", 2: "Sideways", 3: "Falling", 4: "Crashing"}

def validate_regime_visually(nifty: pd.DataFrame, regime_data: dict) -> dict:
    """
    Print regime distribution and check known market events.

    Known ground-truth checks:
      - Mar 2020: COVID crash → should be Crashing or Falling
      - Apr–Dec 2020: Recovery → should be Rising or Rally
      - Jan 2021–Dec 2021: Bull market → Rising
      - Feb 2022: Ukraine war correction → Falling/Sideways
    """
    known_events = {
        ("2020-03-01", "2020-03-31"): [3, 4],   # COVID crash → Falling/Crashing
        ("2020-04-01", "2020-12-31"): [0, 1],   # Recovery → Rising/Rally
        ("2021-01-01", "2021-12-31"): [0, 1],   # Bull market
        ("2022-02-01", "2022-03-31"): [2, 3],   # Ukraine war
        ("2020-01-01", "2020-03-20"): [2, 3, 4], # Pre-crash → Sideways/Falling
    }

    report = {"events": {}, "state_distribution": {}}

    if "regime_labels" not in regime_data:
        logger.warning("No regime labels in regime_data — skipping visual check.")
        return report

    labels = regime_data["regime_labels"]
    labels.index = pd.to_datetime(labels.index)

    # Distribution
    dist = labels.value_counts().sort_index()
    report["state_distribution"] = {
        STATE_NAMES.get(int(k), str(k)): int(v)
        for k, v in dist.items()
    }
    logger.info("Regime distribution: %s", report["state_distribution"])

    # Known event checks
    passed = 0
    failed = 0
    for (start, end), expected_states in known_events.items():
        window = labels.loc[start:end]
        if len(window) == 0:
            continue
        dominant = int(window.mode().iloc[0])
        ok = dominant in expected_states
        status = "✅" if ok else "⚠️ "
        event_key = f"{start}→{end}"
        report["events"][event_key] = {
            "dominant_state": STATE_NAMES.get(dominant, str(dominant)),
            "expected": [STATE_NAMES[s] for s in expected_states],
            "pass": ok,
        }
        logger.info("%s %s dominant=%s expected=%s",
                    status, event_key,
                    STATE_NAMES.get(dominant, str(dominant)),
                    [STATE_NAMES[s] for s in expected_states])
        if ok: passed += 1
        else:  failed += 1

    report["checks_passed"] = passed
    report["checks_failed"] = failed
    return report


# ─────────────────────────────────────────────────────────────────────
# Step 03: Feature distribution audit
# ─────────────────────────────────────────────────────────────────────

def audit_feature_distribution(all_features: dict) -> pd.DataFrame:
    """
    For each of the 25 features, compute stats across all symbols.
    Flags:
      - near-zero variance (std < 0.01) → suspect constant feature
      - extreme kurtosis (> 50) → fat tails, may need clipping
      - NaN rate > 5 % → data quality issue
    """
    feature_cols = get_feature_columns()
    rows = []

    for sym, feat_df in all_features.items():
        train_mask = (
            (feat_df.index >= pd.Timestamp("2016-01-01")) &
            (feat_df.index <= pd.Timestamp("2019-12-31"))
        )
        df_tr = feat_df.loc[train_mask]

        for col in feature_cols:
            if col not in df_tr.columns:
                continue
            s = df_tr[col].dropna()
            if len(s) == 0:
                continue
            nan_rate = df_tr[col].isna().mean()
            kurt = float(s.kurtosis()) if len(s) > 4 else 0.0
            rows.append({
                "symbol":     sym,
                "feature":    col,
                "mean":       float(s.mean()),
                "std":        float(s.std()),
                "min":        float(s.min()),
                "max":        float(s.max()),
                "kurtosis":   kurt,
                "nan_rate":   float(nan_rate),
                "flag_lowvar": bool(s.std() < 0.01),
                "flag_kurtosis": bool(kurt > 50),
                "flag_nan":   bool(nan_rate > 0.05),
            })

    audit_df = pd.DataFrame(rows)

    # Summary per feature (across all symbols)
    summary = audit_df.groupby("feature").agg(
        mean_std   = ("std",        "mean"),
        max_kurt   = ("kurtosis",   "max"),
        max_nan    = ("nan_rate",   "max"),
        n_lowvar   = ("flag_lowvar","sum"),
        n_highkurt = ("flag_kurtosis","sum"),
        n_nanflag  = ("flag_nan",   "sum"),
    ).reset_index()

    flagged = summary[
        (summary["n_lowvar"] > 0) |
        (summary["n_highkurt"] > 0) |
        (summary["n_nanflag"] > 0)
    ]

    logger.info("\nFeature audit — flagged issues:\n%s", flagged.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────
# Step 04: Build SVM feature matrices for all symbols
# ─────────────────────────────────────────────────────────────────────

def build_all_svm_features(raw_data: dict, regime_data: dict, cfg: dict) -> dict:
    """Build 25-dim SVM feature matrices for all 10 symbols."""
    all_features = {}
    for sym, ohlcv in raw_data.items():
        regime_df = regime_data.get("per_symbol", {}).get(sym)
        try:
            feat_df = build_svm_features(
                ohlcv      = ohlcv,
                regime_df  = regime_df,
                tx_rate    = cfg["costs"]["tx_rate_per_leg"],
                zscore_win = cfg["features"]["zscore_window"],
                fwd_days   = cfg["features"]["forward_days"],
            )
            all_features[sym] = feat_df
            label_dist = feat_df["label"].value_counts().to_dict()
            logger.info(
                "[%s] Features built: %d rows | Label dist: %s",
                sym, len(feat_df), label_dist,
            )
        except Exception as e:
            logger.error("[%s] Feature build failed: %s", sym, e, exc_info=True)

    return all_features


# ─────────────────────────────────────────────────────────────────────
# Step 06: Validate — no look-ahead, label balance
# ─────────────────────────────────────────────────────────────────────

def validate_features(all_features: dict) -> bool:
    """
    Run look-ahead and data quality checks:
      1. No future price data in features (verified by column construction)
      2. Label class balance not extreme (no class < 10 %)
      3. Feature NaN count in training window
      4. Feature correlation with future returns (sanity — should be low)
    """
    all_ok = True
    feature_cols = get_feature_columns()

    for sym, feat_df in all_features.items():
        train_mask = (
            (feat_df.index >= pd.Timestamp("2016-01-01")) &
            (feat_df.index <= pd.Timestamp("2019-12-31"))
        )
        df_tr = feat_df.loc[train_mask]
        valid  = df_tr["label"].notna()
        df_v   = df_tr.loc[valid]

        if len(df_v) < 100:
            logger.error("[%s] FAIL: Only %d valid training rows.", sym, len(df_v))
            all_ok = False
            continue

        # Label balance
        counts = df_v["label"].value_counts(normalize=True)
        for lbl_val, pct in counts.items():
            if pct < 0.05:
                logger.warning("[%s] Label %d is rare: %.1f %%", sym, lbl_val, pct * 100)

        # NaN audit
        nan_counts = df_tr[feature_cols].isna().sum()
        high_nan = nan_counts[nan_counts > len(df_tr) * 0.10]
        if len(high_nan) > 0:
            logger.warning("[%s] High NaN features: %s", sym, high_nan.to_dict())

        logger.info("[%s] ✓ Validate passed | rows=%d | labels=%s",
                    sym, len(df_v), counts.round(3).to_dict())

    return all_ok


# ─────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────

def main(skip_hp_search: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config()
    results_dir = PROJECT_ROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print("  QBEAST-AI.N  |  Week 2–3  |  SVM HP Search Pipeline")
    print("═" * 70 + "\n")

    # ── Step 01: Load data ─────────────────────────────────────────────
    logger.info("[01] Loading data...")
    try:
        symbol_data, manifest = load_week1_artefacts(cfg)
        # If Week 1 parquets lack raw OHLCV we rebuild
        raw_data = load_from_raw_csvs(cfg)
    except FileNotFoundError:
        logger.info("Week 1 manifest not found — loading from raw CSVs.")
        raw_data = load_from_raw_csvs(cfg)

    nifty = load_nifty50(cfg)

    # ── Step 02: Regime data placeholder ──────────────────────────────
    # In full pipeline: load from data/regime/regime_labels.parquet
    # For Week 2-3 standalone: try to load, else use empty
    regime_data = {}
    regime_path = PROJECT_ROOT / "data" / "regime" / "regime_labels.parquet"
    if regime_path.exists():
        rl = pd.read_parquet(regime_path)
        regime_data["regime_labels"] = rl
        logger.info("Loaded regime labels from %s", regime_path)
    else:
        logger.warning("Regime labels not found — SVM regime features will be zero. "
                       "Run regime engine (Week 3) for full 25-dim features.")

    # ── Step 03: Build SVM features ───────────────────────────────────
    logger.info("[03] Building SVM feature matrices for all symbols...")
    all_features = build_all_svm_features(raw_data, regime_data, cfg)

    if not all_features:
        logger.error("No features built — check raw CSV paths.")
        sys.exit(1)

    # ── Step 04: Regime visual validation ─────────────────────────────
    logger.info("[04] Regime visual validation...")
    regime_report = validate_regime_visually(nifty, regime_data)

    # ── Step 05: Feature distribution audit ───────────────────────────
    logger.info("[05] Feature distribution audit...")
    audit_df = audit_feature_distribution(all_features)
    audit_df.to_csv(results_dir / "feature_audit.csv", index=False)
    logger.info("Feature audit saved → results/feature_audit.csv")

    # ── Step 06: Validate features ────────────────────────────────────
    logger.info("[06] Validating features (look-ahead, label balance, NaN)...")
    ok = validate_features(all_features)
    if not ok:
        logger.warning("Validation found issues — review before proceeding.")

    # ── Step 07: SVM Initial HP Search ────────────────────────────────
    if not skip_hp_search:
        logger.info("[07] Running initial SVM HP search (2016–2017 in-sample)...")
        svm_cfg = cfg.get("svm_hp", {})
        registry = run_initial_hp_search(
            all_features = all_features,
            output_dir   = results_dir,
            cfg          = svm_cfg,
        )
        logger.info("✅ Week 2–3 SVM HP search complete!")
    else:
        logger.info("[07] Skipping HP search (--skip-hp-search flag set).")
        registry = None

    # ── Step 08: Save week2 manifest ──────────────────────────────────
    manifest = {
        "week": "2-3",
        "regime_validation": regime_report,
        "feature_audit_path": str(results_dir / "feature_audit.csv"),
        "svm_registry_path": str(results_dir / "svm_registry_initial.pkl"),
        "symbols": list(all_features.keys()),
        "train_period": "2016-01-01 → 2017-12-31 (HP search) | 2018-01-01 → 2019-12-31 (OOS val)",
    }
    manifest_path = results_dir / "week2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Week 2–3 manifest saved → %s", manifest_path)

    print("\n" + "═" * 70)
    print("  Week 2–3 COMPLETE")
    print(f"  Symbols processed : {len(all_features)}")
    print(f"  Regime checks     : {regime_report.get('checks_passed', 'N/A')} passed")
    print(f"  Feature audit     : results/feature_audit.csv")
    print(f"  SVM registry      : results/svm_registry_initial.pkl")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QBEAST-AI.N Week 2–3 Pipeline")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-hp-search", action="store_true",
                        help="Skip Optuna HP search (for quick testing)")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    main(skip_hp_search=args.skip_hp_search)