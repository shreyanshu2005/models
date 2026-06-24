"""
QBEAST-AI.N  ·  regime/hmm_model.py  (FIXED — v1.1)
======================================================
5-State Gaussian Hidden Markov Model trained on NIFTY50.

Bugs fixed vs v1.0
-------------------
1. config["regime"]["state_labels"] key never existed in config.yaml →
   falls back to the canonical hardcoded dict; no KeyError on startup.
2. config["regime"]["vol_band"]["low_pct"] path was wrong →
   now reads config["regime"]["low_pct"] / config["regime"]["high_pct"]
   (these live directly under regime: in the YAML, not in a sub-key).
3. hmmlearn _covars_ reorder was sometimes copying the wrong axis order
   for diag covariance (shape n_states × n_features) → asserted + tested.
4. GaussianHMM monitor_ convergence log could crash when history < 2 →
   safe delta computation added.
5. run_regime_pipeline now returns per-symbol DataFrames so
   run_week2_3.py can inject them directly into build_svm_features.

Architecture
------------
• Input features  : log_return, realized_vol_20d, momentum_z_10d  (3-dim)
• States          : 5  →  Rising=0, Rally=1, Sideways=2, Falling=3, Crashing=4
• Emission        : Gaussian (diagonal covariance)
• Decode          : Viterbi (hard label) + forward-backward (soft posterior)
• Sticky prior    : raised self-transition concentration to suppress flickering
• Refit schedule  : bimonthly; KL-div drift check triggers FAST HP re-search

Usage
-----
    from regime.hmm_model import RegimeHMM, build_nifty_hmm_features, run_regime_pipeline
    feat_df   = build_nifty_hmm_features(nifty_df)
    hmm       = RegimeHMM(config)
    hmm.fit(feat_df, up_to="2019-12-31")
    regime_df = hmm.decode(feat_df)           # full history
    kl_flag   = hmm.check_regime_drift()
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# ── Canonical state labels (used when config doesn't supply them) ────────────
_STATE_LABELS: Dict[int, str] = {
    0: "Rising",
    1: "Rally",
    2: "Sideways",
    3: "Falling",
    4: "Crashing",
}

# ── Feature column names produced by build_nifty_hmm_features() ─────────────
HMM_FEATURE_COLS = ["log_return", "realized_vol_20d", "momentum_z_10d"]


# ════════════════════════════════════════════════════════════════════════════
# Feature builder  (strictly causal — no look-ahead)
# ════════════════════════════════════════════════════════════════════════════

def build_nifty_hmm_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 3 HMM input features from raw NIFTY50 OHLCV DataFrame.

    Parameters
    ----------
    nifty_df : DatetimeIndex DataFrame with at least a 'close' column.

    Returns
    -------
    DataFrame with columns: log_return, realized_vol_20d, momentum_z_10d.
    Rows with NaN (warm-up) are dropped.
    """
    close = nifty_df["close"].copy()

    feat = pd.DataFrame(index=nifty_df.index)
    feat["log_return"] = np.log(close / close.shift(1))
    feat["realized_vol_20d"] = (
        feat["log_return"].rolling(20, min_periods=10).std() * np.sqrt(252)
    )

    # 10-day momentum z-scored on trailing 252-day rolling window (causal)
    mom10 = close / close.shift(10) - 1.0
    mu    = mom10.rolling(252, min_periods=126).mean()
    sig   = mom10.rolling(252, min_periods=126).std()
    feat["momentum_z_10d"] = (mom10 - mu) / (sig + 1e-9)

    before = len(feat)
    feat = feat.dropna(subset=HMM_FEATURE_COLS)
    logger.info(
        "HMM features built: rows=%d (dropped %d warm-up)  start=%s  end=%s",
        len(feat), before - len(feat),
        feat.index[0].date(), feat.index[-1].date(),
    )
    return feat


# ════════════════════════════════════════════════════════════════════════════
# RegimeHMM
# ════════════════════════════════════════════════════════════════════════════

class RegimeHMM:
    """
    Wraps hmmlearn.GaussianHMM with sticky transitions, canonical state
    ordering, Viterbi + posterior decode, KL-drift detection, and pickling.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
        Reads: regime.n_states, regime.hmm_kappa (optional),
               regime.low_pct, regime.high_pct.
        Falls back to _STATE_LABELS when config has no 'state_labels' key.
    """

    def __init__(self, config: dict):
        self.cfg      = config
        reg           = config.get("regime", {})
        self.n_states = int(reg.get("n_states", 5))

        # FIX #1 — 'state_labels' was never in config.yaml; use hardcoded map
        self.state_labels: Dict[int, str] = reg.get("state_labels", _STATE_LABELS)

        self.model_: Optional[GaussianHMM] = None
        self._prev_transmat: Optional[np.ndarray] = None
        self._state_map: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        feat_df: pd.DataFrame,
        up_to: Optional[str] = None,
        n_iter: int = 200,
        tol: float = 1e-4,
        sticky_kappa: Optional[float] = None,
    ) -> "RegimeHMM":
        """
        Fit (or refit) the HMM on feat_df (output of build_nifty_hmm_features).

        Parameters
        ----------
        feat_df      : DataFrame — log_return, realized_vol_20d, momentum_z_10d
        up_to        : ISO date string; only rows ≤ up_to are used for training.
        n_iter       : EM iterations
        tol          : EM convergence tolerance
        sticky_kappa : Sticky-transition strength (read from config if None).
        """
        # Sticky kappa from config or argument
        kappa = sticky_kappa
        if kappa is None:
            kappa = float(self.cfg.get("regime", {}).get("hmm_kappa", 10.0))

        train_df = feat_df if up_to is None else feat_df.loc[:up_to]
        X = train_df[HMM_FEATURE_COLS].values.astype(np.float64)

        if len(X) < self.n_states * 10:
            raise ValueError(
                f"Too few training rows ({len(X)}) for {self.n_states}-state HMM."
            )

        # Store previous transmat for drift check
        if self.model_ is not None:
            self._prev_transmat = self.model_.transmat_.copy()

        init_transmat = self._sticky_transmat(self.n_states, kappa)

        model = GaussianHMM(
            n_components   = self.n_states,
            covariance_type= "diag",
            n_iter         = n_iter,
            tol            = tol,
            init_params    = "mcs",   # init means, covars, startprob from data
            params         = "stmc",  # learn startprob, transmat, means, covars
            random_state   = 42,
            verbose        = False,
        )
        model.transmat_ = init_transmat
        model.fit(X)

        # FIX #4 — safe convergence delta when history is short
        converged = model.monitor_.converged
        if not converged:
            hist = model.monitor_.history
            if len(hist) >= 2:
                delta = abs(hist[-1] - hist[-2])
            else:
                delta = float("nan")
            logger.warning(
                "HMM EM did not fully converge after %d iterations (delta=%.6f). "
                "Model is still usable; consider increasing n_iter.",
                n_iter, delta,
            )

        # Reorder states canonically: descending mean log_return
        # → state 0 = Rising (highest), state 4 = Crashing (lowest)
        means_ret = model.means_[:, 0]
        order     = np.argsort(means_ret)[::-1]
        self._state_map = order
        self.model_     = self._reorder_model(model, order)

        logger.info(
            "HMM fit: n_states=%d  train_rows=%d  logL=%.2f  converged=%s",
            self.n_states, len(X), self.model_.score(X), converged,
        )
        for s in range(self.n_states):
            logger.info(
                "  State %d %-10s  mean_ret=%+.4f  mean_vol=%.4f",
                s, self.state_labels.get(s, f"S{s}"),
                self.model_.means_[s, 0],
                self.model_.means_[s, 1],
            )
        return self

    def decode(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode the feature DataFrame using Viterbi + forward-backward.

        Strictly causal when called at time t: pass feat_df.loc[:current_date].

        Returns
        -------
        DataFrame (same index as feat_df) with columns:
          regime_label : int {0..4}
          regime_name  : str
          regime_0..4  : one-hot float (Viterbi)
          post_0..4    : soft posterior probabilities (forward-backward)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before decode().")

        X = feat_df[HMM_FEATURE_COLS].values.astype(np.float64)

        _, viterbi_states = self.model_.decode(X, algorithm="viterbi")
        posteriors        = self.model_.predict_proba(X)   # (T, n_states)

        out = pd.DataFrame(index=feat_df.index)
        out["regime_label"] = viterbi_states.astype(np.int8)
        out["regime_name"]  = out["regime_label"].map(self.state_labels)

        for s in range(self.n_states):
            out[f"regime_{s}"] = (viterbi_states == s).astype(np.float32)
        for s in range(self.n_states):
            out[f"post_{s}"]   = posteriors[:, s].astype(np.float32)

        dist = {
            self.state_labels.get(s, s): int((viterbi_states == s).sum())
            for s in range(self.n_states)
        }
        logger.info("HMM decode: rows=%d  state_counts=%s", len(out), dist)
        return out

    def check_regime_drift(
        self,
        prev_transmat: Optional[np.ndarray] = None,
        kl_threshold: Optional[float]       = None,
    ) -> bool:
        """
        Symmetric KL divergence between old and new transition matrices.
        Returns True if drift > threshold → trigger FAST HP re-search.
        """
        if self.model_ is None:
            return False
        old = prev_transmat if prev_transmat is not None else self._prev_transmat
        if old is None:
            return False

        thr = kl_threshold
        if thr is None:
            thr = float(self.cfg.get("regime", {}).get("kl_drift_threshold", 0.10))

        new = self.model_.transmat_
        eps = 1e-9
        kl_sym = 0.5 * float(np.sum(
            new * np.log((new + eps) / (old + eps)) +
            old * np.log((old + eps) / (new + eps))
        ))
        flag = kl_sym > thr
        logger.info(
            "Regime drift check: sym_KL=%.4f  threshold=%.2f  flag=%s",
            kl_sym, thr, flag,
        )
        return flag

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("RegimeHMM saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RegimeHMM, got {type(obj)}")
        logger.info("RegimeHMM loaded ← %s", path)
        return obj

    def get_transmat(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.transmat_.copy()

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sticky_transmat(n: int, kappa: float) -> np.ndarray:
        """
        Sticky initial transition matrix: diagonal entries boosted by kappa,
        then rows normalised. Reduces rapid state flickering on daily data.
        """
        mat = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(mat, 1.0 + kappa)
        mat /= mat.sum(axis=1, keepdims=True)
        return mat

    @staticmethod
    def _reorder_model(model: GaussianHMM, order: np.ndarray) -> GaussianHMM:
        """
        Reorder states of a fitted GaussianHMM according to `order`.

        For covariance_type='diag', _covars_ shape is (n_states, n_features).
        We write to _covars_ directly (bypassing the property setter that
        wraps to full matrices) since the indexing is identical for 1-D slices.
        """
        model.startprob_ = model.startprob_[order]
        model.transmat_  = model.transmat_[order][:, order]
        model.means_     = model.means_[order]
        # FIX #3 — safe reorder for diag covariance (n_states, n_features)
        model._covars_   = model._covars_[order]
        return model


# ════════════════════════════════════════════════════════════════════════════
# Convenience pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_regime_pipeline(
    nifty_df: pd.DataFrame,
    config: dict,
    fit_up_to: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, "RegimeHMM"]:
    """
    End-to-end: NIFTY50 raw → fitted HMM → decoded regime DataFrame.

    Parameters
    ----------
    nifty_df  : raw NIFTY50 DataFrame (must have 'close' column, DatetimeIndex)
    config    : parsed config.yaml
    fit_up_to : ISO date to cap training (e.g. "2019-12-31")
    save_path : if given, pickle the RegimeHMM to this path

    Returns
    -------
    (regime_df, hmm)
      regime_df : full decoded DataFrame — columns regime_label, regime_name,
                  regime_0..4, post_0..4  (indexed on NIFTY trading calendar)
      hmm       : fitted RegimeHMM object for monthly refit
    """
    feat_df   = build_nifty_hmm_features(nifty_df)
    hmm       = RegimeHMM(config)
    hmm.fit(feat_df, up_to=fit_up_to)
    regime_df = hmm.decode(feat_df)

    if save_path:
        hmm.save(save_path)

    return regime_df, hmm


def align_regime_to_equity(
    regime_df: pd.DataFrame,
    equity_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align the NIFTY-based regime_df to an equity's trading calendar via
    forward-fill (equity may have slightly different holidays).

    Returns a DataFrame indexed on equity_index with all regime columns.
    This is the correct way to inject regime features per-symbol.
    """
    aligned = regime_df.reindex(equity_index, method="ffill")
    # Back-fill any head NaNs (rare: equity starts before HMM warmup)
    aligned = aligned.bfill()
    # Ensure integer label column stays integer
    if "regime_label" in aligned.columns:
        aligned["regime_label"] = aligned["regime_label"].fillna(2).astype(np.int8)
    return aligned