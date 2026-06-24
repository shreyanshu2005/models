"""
QBEAST-AI.N  ·  pipeline/run_week1.py
========================================
Week 1 End-to-End Pipeline Orchestrator
----------------------------------------
Executes the full Week 1 build sequence as defined in the project spec (§7):

    Step 1  Load all 10 equity CSVs + 4 NIFTY50 benchmark CSVs
    Step 2  Build unified NSE trading calendar
    Step 3  Align all symbols to calendar; produce coverage report
    Step 4  Build feature matrices (25 features) for all symbols
    Step 5  Train 5-state HMM on NIFTY50 (2010–2019 warmup+train)
    Step 6  Classify per-symbol vol bands (20d vol / 252d percentile)
    Step 7  Inject regime features into all feature matrices
    Step 8  Build SVM labels  {-1, 0, +1}
    Step 9  Build CNN-LSTM tensors  (N, lookback, 22)
    Step 10 Run full validation suite; print report
    Step 11 Save processed artefacts to data/processed/ and data/regime/

All outputs are saved as Parquet (DataFrames) or .npz (NumPy tensors).
A JSON manifest (data/processed/week1_manifest.json) records paths,
shapes, date ranges, and validation status for downstream steps.

Usage (CLI)
-----------
    python -m pipeline.run_week1                       # uses config.yaml in cwd
    python -m pipeline.run_week1 --config path/to/config.yaml
    python -m pipeline.run_week1 --data-root /path/to/csvs --validate-only

Usage (Python)
--------------
    from pipeline.run_week1 import Week1Pipeline
    pipe = Week1Pipeline("config.yaml")
    pipe.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ── Project-local imports ────────────────────────────────────────────────────
# These imports assume run_week1.py lives in pipeline/ inside qbeast_trimodel/
# Adjust sys.path if running from a different working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loader          import DataLoader, load_config
from data.feature_engineer import FeatureEngineer, build_all_features
from pipeline.calendar    import TradingCalendar
from pipeline.validate    import DataValidator
from regime.hmm_model     import RegimeHMM, build_nifty_hmm_features, run_regime_pipeline
from regime.vol_band      import VolBandClassifier, vol_band_summary

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week1")


# ════════════════════════════════════════════════════════════════════════════
# Week1Pipeline  —  main orchestrator
# ════════════════════════════════════════════════════════════════════════════

class Week1Pipeline:
    """
    Orchestrates the full Week 1 data + regime pipeline.

    Parameters
    ----------
    config_path : str | Path
        Path to config.yaml
    data_root   : str | Path | None
        Override raw data directory (default: from config)
    save_outputs : bool
        If True (default), persist parquet / npz artefacts to disk.
    validate     : bool
        If True (default), run full DataValidator suite and raise on failure.
    """

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        data_root: Optional[str | Path] = None,
        save_outputs: bool = True,
        validate: bool = True,
    ):
        self.config_path  = Path(config_path)
        self.cfg          = load_config(config_path)
        self.data_root    = Path(data_root) if data_root else None
        self.save_outputs = save_outputs
        self.validate     = validate

        # Derived paths
        self.processed_dir = Path(self.cfg["paths"]["processed_data"])
        self.regime_dir    = Path(self.cfg["paths"]["regime_data"])
        self.artifacts_dir = Path(self.cfg["paths"]["model_artifacts"])

        # Pipeline artefacts (filled by run())
        self.equities:        Dict[str, pd.DataFrame] = {}
        self.nifty:           pd.DataFrame = pd.DataFrame()
        self.calendar:        pd.DatetimeIndex = pd.DatetimeIndex([])
        self.equities_aligned: Dict[str, pd.DataFrame] = {}
        self.nifty_aligned:   pd.DataFrame = pd.DataFrame()
        self.features:        Dict[str, pd.DataFrame] = {}
        self.svm_labels:      Dict[str, pd.Series]    = {}
        self.cnn_tensors:     Dict[str, Tuple]         = {}   # sym → (X, y, dates)
        self.regime_df:       pd.DataFrame = pd.DataFrame()
        self.vol_bands:       Dict[str, pd.DataFrame] = {}
        self.hmm:             Optional[RegimeHMM] = None
        self.manifest:        Dict = {}

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self) -> "Week1Pipeline":
        """Execute all 11 steps sequentially. Returns self for chaining."""
        t0 = time.time()
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║   QBEAST-AI.N  ·  Week 1 Pipeline  ·  Starting      ║")
        logger.info("╚══════════════════════════════════════════════════════╝")

        self._step1_load()
        self._step2_calendar()
        self._step3_align()
        self._step4_features()      # base features (15 dims) — no regime yet
        self._step5_hmm()
        self._step6_vol_bands()
        self._step7_inject_regime() # merge regime → 25-dim features
        self._step8_svm_labels()
        self._step9_cnn_tensors()
        if self.validate:
            self._step10_validate()
        if self.save_outputs:
            self._step11_save()

        elapsed = time.time() - t0
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║   Week 1 Pipeline COMPLETE  ·  %.1fs elapsed        ║", elapsed)
        logger.info("╚══════════════════════════════════════════════════════╝")
        return self

    # ── Steps ────────────────────────────────────────────────────────────────

    def _step1_load(self) -> None:
        logger.info("─── Step 1: Loading raw CSVs ───")
        loader = DataLoader(self.cfg, data_root=self.data_root)
        self.equities, self.nifty = loader.load_all()
        logger.info(
            "Loaded %d equity symbols + NIFTY50 (%d rows)",
            len(self.equities), len(self.nifty),
        )

    def _step2_calendar(self) -> None:
        logger.info("─── Step 2: Building trading calendar ───")
        tc = TradingCalendar(self.cfg)
        self.calendar = tc.build(self.equities, self.nifty)
        self._tc = tc   # store for later use
        logger.info(
            "Calendar: %d trading days  %s → %s",
            len(self.calendar),
            self.calendar[0].date(),
            self.calendar[-1].date(),
        )

    def _step3_align(self) -> None:
        logger.info("─── Step 3: Aligning to calendar + coverage report ───")
        self.equities_aligned = self._tc.align_all(self.equities, self.calendar)
        self.nifty_aligned    = self._tc.align(self.nifty, self.calendar)

        # Coverage report (printed and stored)
        coverage = self._tc.coverage_report(self.equities_aligned, self.calendar)
        logger.info("\n%s", coverage.to_string())

    def _step4_features(self) -> None:
        logger.info("─── Step 4: Building raw feature matrices (15 base dims) ───")
        fe = FeatureEngineer(self.cfg)
        raw_feats: Dict[str, pd.DataFrame] = {}
        for sym, df in self.equities_aligned.items():
            raw = fe.build_raw_features(df)
            raw_feats[sym] = raw
            logger.info(
                "  %-12s  shape=%s  warmup_nans=%d",
                sym, raw.shape,
                raw.iloc[:252].isna().any(axis=1).sum(),
            )
        self._raw_features = raw_feats   # 15-dim; regime injection in step 7

    def _step5_hmm(self) -> None:
        logger.info("─── Step 5: Training 5-state HMM on NIFTY50 (2010–2019) ───")
        nifty_warmup = self._tc.align(
            self.nifty,
            self._tc.build_nifty_calendar(self.nifty),
        )
        nifty_feat_df = build_nifty_hmm_features(nifty_warmup)
        self._nifty_feat_df = nifty_feat_df

        # Train only on data up to 2019-12-31 (HP-train + validation window)
        fit_up_to = self.cfg["dates"]["val_end"]
        save_path = self.artifacts_dir / "regime" / "hmm_initial.pkl" \
                    if self.save_outputs else None
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        self.regime_df, self.hmm = run_regime_pipeline(
            nifty_warmup,
            self.cfg,
            fit_up_to=fit_up_to,
            save_path=save_path,
        )

        # Log state distribution
        vc = self.regime_df["regime_name"].value_counts()
        logger.info("HMM state distribution:\n%s", vc.to_string())

    def _step6_vol_bands(self) -> None:
        logger.info("─── Step 6: Classifying per-symbol vol bands ───")
        vbc = VolBandClassifier(self.cfg)
        self.vol_bands = vbc.classify_all(self.equities_aligned)

        summary = vol_band_summary(
            self.vol_bands,
            start=self.cfg["dates"]["universe_start"],
            end=self.cfg["dates"]["backtest_end"],
        )
        logger.info("\nVol-band summary:\n%s", summary.to_string())
        self._vbc = vbc

    def _step7_inject_regime(self) -> None:
        logger.info("─── Step 7: Injecting regime features → 25-dim feature matrices ───")
        fe = FeatureEngineer(self.cfg)
        self.features = {}

        for sym, raw in self._raw_features.items():
            # Merge HMM regime + vol-band for this symbol
            vb_df     = self.vol_bands[sym]
            merged_rg = self._vbc.merge_regime_and_volband(self.regime_df, vb_df)

            # Inject regime (10 dims: one-hot + posterior)
            feat25 = FeatureEngineer.inject_regime_features(raw, merged_rg)

            # Trim to [universe_start, backtest_end] and drop pure warm-up NaN rows
            feat25 = feat25.loc[
                self.cfg["dates"]["universe_start"] :
                self.cfg["dates"]["backtest_end"]
            ]
            # Drop head rows that are all-NaN (pre-feature warm-up)
            feat25 = feat25.dropna(how="all")
            # Forward-fill any residual NaN from regime alignment
            feat25 = feat25.ffill()

            self.features[sym] = feat25
            logger.info(
                "  %-12s  shape=%s  cols=%d",
                sym, feat25.shape, len(feat25.columns),
            )

    def _step8_svm_labels(self) -> None:
        logger.info("─── Step 8: Building SVM labels {-1, 0, +1} ───")
        fe = FeatureEngineer(self.cfg)
        self.svm_labels = {}
        for sym, df in self.equities_aligned.items():
            lbl = fe.build_svm_labels(
                df,
                forward_days=self.cfg["features"]["cnn_lstm_target_horizon"],
                threshold_pct=self.cfg["cnn_lstm_hp"]["signal_threshold_pct"],
            )
            # Align label index to feature index
            lbl = lbl.reindex(self.features[sym].index)
            self.svm_labels[sym] = lbl

            vc = lbl.dropna().value_counts().sort_index()
            logger.info("  %-12s  labels=%s", sym, vc.to_dict())

    def _step9_cnn_tensors(self) -> None:
        logger.info("─── Step 9: Building CNN-LSTM tensors ───")
        fe = FeatureEngineer(self.cfg)
        self.cnn_tensors = {}

        for sym, df in self.equities_aligned.items():
            vb_df     = self.vol_bands[sym]
            merged_rg = self._vbc.merge_regime_and_volband(self.regime_df, vb_df)
            lookback  = self.cfg["features"]["cnn_lstm_lookback_default"]

            try:
                X, y, dates = fe.build_cnn_tensors(
                    df,
                    lookback=lookback,
                    regime_df=merged_rg,
                )
                self.cnn_tensors[sym] = (X, y, dates)
                logger.info("  %-12s  X=%s  y=%s", sym, X.shape, y.shape)
            except ValueError as e:
                logger.error("  %-12s  CNN tensor FAILED: %s", sym, e)

    def _step10_validate(self) -> None:
        logger.info("─── Step 10: Running validation suite ───")
        v = DataValidator(self.cfg, strict=False)

        v.validate_raw(self.equities, self.nifty)
        v.validate_features(self.features)
        v.validate_labels(self.svm_labels)
        v.validate_cnn_tensors(
            self.cnn_tensors,
            lookback=self.cfg["features"]["cnn_lstm_lookback_default"],
        )
        v.validate_regime(self.regime_df, self._nifty_feat_df, self.vol_bands)
        v.print_report(raise_on_fail=True)

    def _step11_save(self) -> None:
        logger.info("─── Step 11: Saving processed artefacts ───")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.regime_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "config_path":   str(self.config_path),
            "symbols":       {},
            "regime":        {},
            "calendar":      {
                "n_days":    len(self.calendar),
                "start":     str(self.calendar[0].date()),
                "end":       str(self.calendar[-1].date()),
            },
        }

        # Save feature matrices
        for sym, feat in self.features.items():
            pq_path = self.processed_dir / f"{sym}_features.parquet"
            feat.to_parquet(pq_path)

            lbl_path = self.processed_dir / f"{sym}_svm_labels.parquet"
            self.svm_labels[sym].to_frame().to_parquet(lbl_path)

            # CNN tensors → .npz
            if sym in self.cnn_tensors:
                X, y, dates = self.cnn_tensors[sym]
                npz_path = self.processed_dir / f"{sym}_cnn_tensors.npz"
                np.savez_compressed(
                    npz_path,
                    X=X, y=y,
                    dates=np.array(dates.astype(str)),
                )
                cnn_info = {
                    "X_shape": list(X.shape),
                    "y_shape": list(y.shape),
                    "path":    str(npz_path),
                }
            else:
                cnn_info = {}

            # Vol bands
            vb_path = self.regime_dir / f"{sym}_vol_band.parquet"
            self.vol_bands[sym].to_parquet(vb_path)

            manifest["symbols"][sym] = {
                "features_path":   str(pq_path),
                "features_shape":  list(feat.shape),
                "labels_path":     str(lbl_path),
                "vol_band_path":   str(vb_path),
                "cnn":             cnn_info,
            }
            logger.info("  Saved %s → %s", sym, pq_path)

        # Save regime DataFrame
        regime_path = self.regime_dir / "nifty_regime.parquet"
        self.regime_df.to_parquet(regime_path)
        manifest["regime"] = {
            "path":        str(regime_path),
            "shape":       list(self.regime_df.shape),
            "state_counts": self.regime_df["regime_name"].value_counts().to_dict(),
        }

        # Save manifest JSON
        manifest_path = self.processed_dir / "week1_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Manifest saved → %s", manifest_path)

        self.manifest = manifest


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QBEAST-AI.N  ·  Week 1 Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override raw data directory from config",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving artefacts to disk (dry run)",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation suite",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Re-configure logging at requested level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    pipe = Week1Pipeline(
        config_path=args.config,
        data_root=args.data_root,
        save_outputs=not args.no_save,
        validate=not args.no_validate,
    )
    pipe.run()


if __name__ == "__main__":
    main()