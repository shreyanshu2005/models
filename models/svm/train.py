"""
QBEAST-AI.N  ·  models/svm/train.py
=====================================
SVM hyperparameter search and model training.

Pipeline
--------
1. For each symbol:
   a. Load purged walk-forward CV splits (SVMDataLoader.build_hp_splits).
   b. Run Optuna over (C, γ) search space — 30 trials, 5-fold purged CV.
      Objective: mean balanced-accuracy across folds (weighted by class).
   c. Refit final SVC on full train window (2016–2017) with best HP.
   d. Evaluate on purged OOS val (2018–2019).
   e. Save fitted model + HP record to artifacts/svm/.

Cap-segment HP ranges (from config):
   Large-cap  C ∈ [10, 100],  γ ∈ [1e-4, 1.0]
   Mid-cap    C ∈ [0.1, 10],  γ ∈ [1e-4, 1.0]

HV regime modifier: if vol_band_2 (HV) dominates the train window,
reduce C_high by 50% and γ_high by 50% for a smoother hyperplane.

Usage (CLI)
-----------
    python -m models.svm.train                         # all 10 symbols
    python -m models.svm.train --symbols RELIANCE HDFCBANK
    python -m models.svm.train --symbols RELIANCE --trials 50

Usage (Python)
--------------
    from models.svm.train import SVMTrainer
    trainer = SVMTrainer(config)
    results = trainer.train_all()
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import yaml
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.svm.features import SVMDataLoader

logger = logging.getLogger(__name__)


class SVMTrainer:
    """
    Per-symbol SVM trainer with Optuna HP search.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    artifacts_dir : str | Path | None
        Override for artifacts/svm/. Uses config if None.
    n_trials : int
        Optuna trials per symbol (default from config: 30).
    n_folds : int
        Walk-forward CV folds (default from config: 5).
    """

    def __init__(
        self,
        config: dict,
        artifacts_dir: str | Path | None = None,
        n_trials: Optional[int] = None,
        n_folds: Optional[int] = None,
    ):
        self.cfg          = config
        self.art_dir      = Path(artifacts_dir or config["paths"]["model_artifacts"]) / "svm"
        self.n_trials     = n_trials or config["svm_hp"]["optuna_trials"]   # 30
        self.n_folds      = n_folds  or config["svm_hp"]["cv_folds"]        # 5
        self.loader       = SVMDataLoader(config)
        self.symbols      = config["universe"]["all_symbols"]
        self.lc_symbols   = set(config["universe"]["large_cap_symbols"])
        self.results_: Dict[str, dict] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def train_symbol(self, symbol: str) -> dict:
        """
        Full HP search + final fit + OOS evaluation for one symbol.

        Returns
        -------
        dict with keys:
            symbol, cap_segment, best_C, best_gamma, val_balanced_acc,
            val_report, model_path, scaler_path, hp_path
        """
        t0 = time.time()
        logger.info("══════  SVM: %s  ══════", symbol)

        cap_seg    = self.loader.cap_segment(symbol)
        C_low, C_high = self.loader.get_C_bounds(symbol)
        g_low      = self.cfg["svm_hp"]["gamma_low"]
        g_high     = self.cfg["svm_hp"]["gamma_high"]

        # ── Step 1: HP search via Optuna ────────────────────────────────────
        splits = self.loader.build_hp_splits(symbol, n_folds=self.n_folds)

        # HV regime modifier: check if HV band dominates training data
        feat_all, _ = self.loader.load_symbol(symbol)
        train_feat  = feat_all.loc[
            self.cfg["dates"]["hp_train_start"] : self.cfg["dates"]["hp_train_end"]
        ]
        hv_cols = [c for c in train_feat.columns if "vol_band_2" in c]
        if hv_cols:
            hv_ratio = float(train_feat[hv_cols[0]].mean())
            if hv_ratio > 0.40:   # >40% of days in HV band
                C_high  = C_high  * 0.5
                g_high  = g_high  * 0.5
                logger.info(
                    "%s: HV ratio=%.1f%% → reducing C_high=%.1f  g_high=%.4f",
                    symbol, hv_ratio * 100, C_high, g_high,
                )

        def objective(trial: optuna.Trial) -> float:
            C     = trial.suggest_float("C",     C_low, C_high, log=True)
            gamma = trial.suggest_float("gamma", g_low, g_high, log=True)

            model = SVC(
                C=C, gamma=gamma,
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
                cache_size=500,
            )

            fold_scores = []
            for X_tr, y_tr, X_vl, y_vl in splits:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_vl_s = scaler.transform(X_vl)
                model.fit(X_tr_s, y_tr)
                preds  = model.predict(X_vl_s)
                score  = balanced_accuracy_score(y_vl, preds)
                fold_scores.append(score)

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"svm_{symbol}",
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_C     = study.best_params["C"]
        best_gamma = study.best_params["gamma"]
        best_cv    = study.best_value

        logger.info(
            "%s: HP search done  best_C=%.4f  best_gamma=%.6f  cv_bal_acc=%.4f",
            symbol, best_C, best_gamma, best_cv,
        )

        # ── Step 2: Final fit on full train window ───────────────────────────
        X_train, y_train, X_val, y_val = self.loader.build_initial_train_val(symbol)

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        final_model = SVC(
            C=best_C,
            gamma=best_gamma,
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
            cache_size=500,
            probability=True,    # enable predict_proba for soft signals
        )
        final_model.fit(X_tr_s, y_train)

        # ── Step 3: OOS evaluation (2018–2019, purged) ─────────────────────
        val_preds     = final_model.predict(X_val_s)
        val_bal_acc   = balanced_accuracy_score(y_val, val_preds)
        val_report    = classification_report(
            y_val, val_preds,
            target_names=["Sell(-1)", "Flat(0)", "Buy(+1)"],
            output_dict=True,
            zero_division=0,
        )

        logger.info(
            "%s: OOS val  bal_acc=%.4f  buy_f1=%.3f  sell_f1=%.3f",
            symbol,
            val_bal_acc,
            val_report.get("Buy(+1)", {}).get("f1-score", 0.0),
            val_report.get("Sell(-1)", {}).get("f1-score", 0.0),
        )

        # ── Step 4: Save artefacts ──────────────────────────────────────────
        self.art_dir.mkdir(parents=True, exist_ok=True)

        model_path  = self.art_dir / f"{symbol}_svm.pkl"
        scaler_path = self.art_dir / f"{symbol}_scaler.pkl"
        hp_path     = self.art_dir / f"{symbol}_hp.json"

        with open(model_path,  "wb") as f: pickle.dump(final_model, f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

        hp_record = {
            "symbol":       symbol,
            "cap_segment":  cap_seg,
            "best_C":       best_C,
            "best_gamma":   best_gamma,
            "cv_bal_acc":   best_cv,
            "val_bal_acc":  val_bal_acc,
            "n_trials":     self.n_trials,
            "n_folds":      self.n_folds,
            "C_search_range":     [C_low, C_high],
            "gamma_search_range": [g_low, g_high],
            "train_rows":   int(X_train.shape[0]),
            "val_rows":     int(X_val.shape[0]),
            "val_report":   val_report,
            "elapsed_s":    round(time.time() - t0, 1),
        }
        with open(hp_path, "w") as f:
            json.dump(hp_record, f, indent=2)

        logger.info(
            "%s: saved model→%s  scaler→%s  hp→%s  (%.1fs)",
            symbol, model_path.name, scaler_path.name, hp_path.name,
            time.time() - t0,
        )

        self.results_[symbol] = hp_record
        return hp_record

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Train SVM for all symbols (or a subset).

        Returns
        -------
        dict: symbol → result dict
        """
        targets = symbols or self.symbols
        t0      = time.time()

        logger.info(
            "╔══════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║   QBEAST-AI.N  ·  SVM Training  ·  %d symbols       ║",
            len(targets),
        )
        logger.info(
            "╚══════════════════════════════════════════════════════╝"
        )

        for sym in targets:
            try:
                self.train_symbol(sym)
            except Exception as e:
                logger.error("FAILED %s: %s", sym, e, exc_info=True)

        # Save combined summary
        summary_path = self.art_dir / "svm_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.results_, f, indent=2)

        elapsed = time.time() - t0
        logger.info(
            "SVM training complete: %d/%d symbols  %.1fs total",
            len(self.results_), len(targets), elapsed,
        )
        logger.info("Summary → %s", summary_path)
        return self.results_

    @staticmethod
    def load_model(symbol: str, artifacts_dir: str | Path) -> Tuple:
        """
        Load a saved SVM model + scaler for inference.

        Returns
        -------
        (model, scaler)
        """
        art_dir     = Path(artifacts_dir) / "svm"
        model_path  = art_dir / f"{symbol}_svm.pkl"
        scaler_path = art_dir / f"{symbol}_scaler.pkl"

        with open(model_path,  "rb") as f: model  = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
        return model, scaler


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QBEAST-AI.N · SVM HP Search + Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    type=str, default="config.yaml")
    p.add_argument("--symbols",   nargs="+", default=None,
                   help="Symbols to train (default: all 10)")
    p.add_argument("--trials",    type=int, default=None,
                   help="Optuna trials per symbol (default: from config)")
    p.add_argument("--folds",     type=int, default=None,
                   help="CV folds (default: from config)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = SVMTrainer(
        config,
        n_trials=args.trials,
        n_folds=args.folds,
    )
    trainer.train_all(symbols=args.symbols)


if __name__ == "__main__":
    main()