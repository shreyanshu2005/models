"""
QBEAST-AI.N  ·  models/svm/features.py
========================================
SVM feature preparation layer.

Responsibilities
----------------
1. Load 25-dim feature parquets (output of Week 1) for each symbol.
2. Load corresponding SVM labels {-1, 0, +1}.
3. Build purged walk-forward train/val splits respecting:
      - 30-day purge gap between train end and val start
      - 10-day embargo after each fold
      - Strict time ordering (no shuffling)
4. Return (X_train, y_train, X_val, y_val) arrays ready for sklearn SVC.

The feature matrices already contain the 25 dims (15 base + 10 regime).
No additional feature engineering is done here — that was Week 1's job.

Usage
-----
    from models.svm.features import SVMDataLoader
    loader = SVMDataLoader(config)
    splits = loader.build_hp_splits("RELIANCE")   # HP search splits (2016–2017)
    X_tr, y_tr, X_val, y_val = loader.build_initial_train_val("RELIANCE")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SVMDataLoader:
    """
    Loads Week 1 parquet artefacts and prepares purged train/val arrays
    for SVM hyperparameter search and initial training.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    processed_dir : str | Path | None
        Override for data/processed/. Uses config if None.
    """

    def __init__(self, config: dict, processed_dir: str | Path | None = None):
        self.cfg           = config
        self.proc_dir      = Path(processed_dir or config["paths"]["processed_data"])
        self.purge_days    = config["svm_hp"]["purge_days"]          # 30
        self.embargo_days  = config["walk_forward"]["embargo_days"]  # 10
        self.hp_train_start = pd.Timestamp(config["dates"]["hp_train_start"])
        self.hp_train_end   = pd.Timestamp(config["dates"]["hp_train_end"])
        self.val_start      = pd.Timestamp(config["dates"]["val_start"])
        self.val_end        = pd.Timestamp(config["dates"]["val_end"])
        self.lc_symbols     = set(config["universe"]["large_cap_symbols"])
        self.mc_symbols     = set(config["universe"]["mid_cap_symbols"])

    # ── Public API ───────────────────────────────────────────────────────────

    def load_symbol(self, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load feature matrix and SVM labels for a single symbol.

        Returns
        -------
        (features_df, labels_series)
            features_df : (T, 25) float DataFrame, DatetimeIndex
            labels_series : (T,) int8 Series {-1, 0, +1}, NaN at tail
        """
        feat_path  = self.proc_dir / f"{symbol}_features.parquet"
        label_path = self.proc_dir / f"{symbol}_svm_labels.parquet"

        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature parquet not found: {feat_path}\n"
                f"Run pipeline.run_week1 first."
            )
        if not label_path.exists():
            raise FileNotFoundError(
                f"Label parquet not found: {label_path}\n"
                f"Run pipeline.run_week1 first."
            )

        feat  = pd.read_parquet(feat_path)
        label = pd.read_parquet(label_path).squeeze()   # DataFrame → Series

        # Align on common index (should be identical from Week 1)
        common = feat.index.intersection(label.index)
        feat   = feat.loc[common]
        label  = label.loc[common]

        logger.info(
            "Loaded %-12s  features=%s  labels=%d (non-NaN)",
            symbol, feat.shape, label.dropna().shape[0]
        )
        return feat, label

    def build_hp_splits(
        self,
        symbol: str,
        n_folds: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build n_folds purged walk-forward CV splits on the HP-train window
        (2016-01-01 → 2017-12-31) for Optuna cross-validation.

        Each fold:
          - train : expanding window from hp_train_start → fold_end
          - gap   : purge_days trading days (no data used)
          - val   : embargo_days + next fold_step trading days

        Parameters
        ----------
        symbol  : ticker string
        n_folds : number of CV folds (default 5, from config)

        Returns
        -------
        List of (X_train, y_train, X_val, y_val) numpy arrays.
        NaN labels are dropped before returning.
        """
        feat, label = self.load_symbol(symbol)

        # Restrict to HP-train window
        mask  = (feat.index >= self.hp_train_start) & (feat.index <= self.hp_train_end)
        feat  = feat.loc[mask]
        label = label.loc[mask].dropna()

        # Align feat to valid label rows
        feat = feat.loc[label.index]

        T     = len(feat)
        step  = T // (n_folds + 1)   # approx fold size
        splits = []

        for fold in range(n_folds):
            train_end_idx  = step * (fold + 1)
            purge_end_idx  = min(train_end_idx + self.purge_days, T)
            val_start_idx  = min(purge_end_idx + self.embargo_days, T)
            val_end_idx    = min(val_start_idx + step, T)

            if val_start_idx >= T or val_end_idx <= val_start_idx:
                logger.warning("Fold %d: not enough data — skipping", fold)
                continue

            X_tr = feat.iloc[:train_end_idx].values.astype(np.float32)
            y_tr = label.iloc[:train_end_idx].values.astype(np.int8)
            X_vl = feat.iloc[val_start_idx:val_end_idx].values.astype(np.float32)
            y_vl = label.iloc[val_start_idx:val_end_idx].values.astype(np.int8)

            splits.append((X_tr, y_tr, X_vl, y_vl))
            logger.debug(
                "Fold %d/%d  train=%d  val=%d  purge_gap=%d  embargo=%d",
                fold + 1, n_folds,
                len(X_tr), len(X_vl),
                self.purge_days, self.embargo_days,
            )

        logger.info(
            "%s: built %d/%d purged CV folds on HP-train window",
            symbol, len(splits), n_folds,
        )
        return splits

    def build_initial_train_val(
        self,
        symbol: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the full initial train (2016–2017) and OOS val (2018–2019)
        split with purge gap between them.

        Used for final model selection after HP search is complete.

        Returns
        -------
        (X_train, y_train, X_val, y_val) numpy arrays.
        """
        feat, label = self.load_symbol(symbol)
        label = label.dropna()
        feat  = feat.loc[label.index]

        train_mask = (feat.index >= self.hp_train_start) & (feat.index <= self.hp_train_end)
        val_mask   = (feat.index >= self.val_start)      & (feat.index <= self.val_end)

        # Apply purge: drop val rows within purge_days of train end
        train_end_date = feat.loc[train_mask].index[-1]
        val_feat       = feat.loc[val_mask]
        val_label      = label.loc[val_mask]

        # Count trading days from train end — drop first purge_days rows of val
        days_from_end  = np.arange(len(val_feat))
        val_feat       = val_feat.iloc[self.purge_days:]
        val_label      = val_label.iloc[self.purge_days:]

        X_train = feat.loc[train_mask].values.astype(np.float32)
        y_train = label.loc[train_mask].values.astype(np.int8)
        X_val   = val_feat.values.astype(np.float32)
        y_val   = val_label.values.astype(np.int8)

        logger.info(
            "%s: initial train/val split  X_train=%s  X_val=%s  "
            "(purge_days=%d applied)",
            symbol, X_train.shape, X_val.shape, self.purge_days,
        )
        return X_train, y_train, X_val, y_val

    def cap_segment(self, symbol: str) -> str:
        """Return 'large_cap' or 'mid_cap' for the symbol."""
        return "large_cap" if symbol in self.lc_symbols else "mid_cap"

    def get_C_bounds(self, symbol: str) -> Tuple[float, float]:
        """Return (C_low, C_high) for the symbol's cap segment."""
        seg = self.cap_segment(symbol)
        return (
            self.cfg["svm_hp"][seg]["C_low"],
            self.cfg["svm_hp"][seg]["C_high"],
        )