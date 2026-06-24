"""
QBEAST-AI.N  ·  regime/hmm_model.py
=====================================
5-State Gaussian Hidden Markov Model trained on NIFTY50.

Architecture
------------
• Input features  : log_return, realized_vol_20d, momentum_z_10d  (3-dim)
• States          : 5  →  Rising, Rally, Sideways, Falling, Crashing
• Emission        : Gaussian (diagonal covariance)
• Decode          : Viterbi for hard label, forward-backward for soft posterior
• Sticky prior    : raised self-transition concentration to suppress flickering

Pipeline position
-----------------
    nifty_df  →  RegimeHMM.fit()  →  .decode(nifty_df)
    →  regime_df  (columns: regime_label, regime_0..4, post_0..4, vol_band_0..2)
    →  FeatureEngineer.inject_regime_features()  →  all three model inputs

Refit schedule : bimonthly (every 2 months) during monthly retrain loop.
KL-divergence  : if transition matrix shifts > 10% → flag FAST HP re-search
                 for SVM, CNN-LSTM lookback, and SAC γ.

Usage
-----
    from regime.hmm_model import RegimeHMM, build_nifty_hmm_features
    feat_df  = build_nifty_hmm_features(nifty_df)
    hmm      = RegimeHMM(config)
    hmm.fit(feat_df, up_to="2019-12-31")
    regime_df = hmm.decode(feat_df)
    kl_flag   = hmm.check_regime_drift(prev_transmat)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# ── State labels (canonical order) ──────────────────────────────────────────
STATE_LABELS = {
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
    close = nifty_df["close"]

    feat              = pd.DataFrame(index=nifty_df.index)
    feat["log_return"]      = np.log(close / close.shift(1))
    feat["realized_vol_20d"] = feat["log_return"].rolling(20).std() * np.sqrt(252)

    # 10-day momentum z-scored on a 252-day rolling window
    mom10 = close / close.shift(10) - 1.0
    mu    = mom10.rolling(252).mean()
    sig   = mom10.rolling(252).std()
    feat["momentum_z_10d"] = (mom10 - mu) / (sig + 1e-9)

    feat = feat.dropna(subset=HMM_FEATURE_COLS)
    logger.info(
        "HMM features built: rows=%d  start=%s  end=%s",
        len(feat), feat.index[0].date(), feat.index[-1].date(),
    )
    return feat


# ════════════════════════════════════════════════════════════════════════════
# RegimeHMM  —  main class
# ════════════════════════════════════════════════════════════════════════════

class RegimeHMM:
    """
    Wraps hmmlearn.GaussianHMM with:
    • Sticky self-transition prior
    • Canonical state ordering (sort by mean log_return ascending)
    • Viterbi hard labels + forward-backward soft posteriors
    • KL-divergence drift detection
    • Pickle save / load for monthly refit cycle

    Parameters
    ----------
    config : dict
        Parsed config.yaml — reads regime.n_states, regime.vol_band.*
    """

    def __init__(self, config: dict):
        self.cfg        = config
        self.n_states   = config["regime"]["n_states"]           # 5
        self.state_labels = config["regime"]["state_labels"]     # {0: "Rising", …}
        self._vol_low_pct  = config["regime"]["vol_band"]["low_pct"]   # 33
        self._vol_high_pct = config["regime"]["vol_band"]["high_pct"]  # 67
        self.model_: Optional[GaussianHMM] = None
        self._prev_transmat: Optional[np.ndarray] = None
        # State index remapping after canonical sort
        self._state_map: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        feat_df: pd.DataFrame,
        up_to: Optional[str] = None,
        n_iter: int = 200,
        tol: float = 1e-4,
        sticky_kappa: float = 10.0,
    ) -> "RegimeHMM":
        """
        Fit (or refit) the HMM on feat_df (output of build_nifty_hmm_features).

        Parameters
        ----------
        feat_df       : features DataFrame — log_return, realized_vol_20d, momentum_z_10d
        up_to         : ISO date string; only rows ≤ up_to are used for training.
                        Pass None to use all rows (warmup fit).
        n_iter        : EM iterations
        tol           : EM convergence tolerance
        sticky_kappa  : concentration added to diagonal of initial transmat prior
                        to enforce state persistence (reduce flickering).

        Returns
        -------
        self (for chaining)
        """
        train_df = feat_df if up_to is None else feat_df.loc[:up_to]
        X = train_df[HMM_FEATURE_COLS].values.astype(np.float64)

        # Save previous transmat for KL check (if refitting)
        if self.model_ is not None:
            self._prev_transmat = self.model_.transmat_.copy()

        # Build sticky initial transition matrix
        init_transmat = self._sticky_transmat(self.n_states, sticky_kappa)

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=tol,
            init_params="mcs",   # init means, covars, startprob from data
            params="stmc",       # learn startprob, transmat, means, covars
            random_state=42,
            verbose=False,
        )
        # Inject sticky prior into transmat
        model.transmat_ = init_transmat
        model.fit(X)

        # Warn explicitly if EM did not converge (non-fatal — model is still usable)
        if not model.monitor_.converged:
            logger.warning(
                "HMM EM did not fully converge after %d iterations "
                "(delta=%.6f). Model is still usable; increase n_iter or "
                "set a larger tol to suppress this warning.",
                n_iter, abs(model.monitor_.history[-1] - model.monitor_.history[-2])
                if len(model.monitor_.history) >= 2 else float("nan"),
            )

        # Re-order states canonically by mean log_return descending:
        # state 0 = highest return (Rising) … state 4 = most negative (Crashing)
        means_ret = model.means_[:, 0]          # first feature = log_return
        order     = np.argsort(means_ret)[::-1] # descending
        self._state_map = order
        self.model_     = self._reorder_model(model, order)

        # Score using self.model_ (post-reorder) not the local pre-reorder model
        logger.info(
            "HMM fit: n_states=%d  train_rows=%d  logL=%.2f  converged=%s",
            self.n_states, len(X), self.model_.score(X), model.monitor_.converged,
        )
        for s in range(self.n_states):
            logger.info(
                "  State %d %-10s  mean_ret=%.4f  mean_vol=%.4f",
                s, self.state_labels.get(s, "?"),
                self.model_.means_[s, 0],
                self.model_.means_[s, 1],
            )
        return self

    def decode(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode the full feature DataFrame (train + OOS) using Viterbi + forward-backward.

        Strictly causal when called at time t: only data ≤ t is passed in.
        In the backtest monthly loop, always call with feat_df.loc[:current_date].

        Returns
        -------
        DataFrame indexed same as feat_df with columns:
          regime_label  : int {0..4}
          regime_name   : str  "Rising" | "Rally" | …
          regime_0..4   : one-hot float (hard Viterbi state)
          post_0..4     : soft posterior probabilities (forward-backward)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before decode().")

        X = feat_df[HMM_FEATURE_COLS].values.astype(np.float64)

        # Viterbi hard labels
        _, viterbi_states = self.model_.decode(X, algorithm="viterbi")

        # Soft posteriors (forward-backward)
        posteriors = self.model_.predict_proba(X)   # (T, n_states)

        out = pd.DataFrame(index=feat_df.index)
        out["regime_label"] = viterbi_states.astype(np.int8)
        out["regime_name"]  = out["regime_label"].map(self.state_labels)

        # One-hot from hard Viterbi
        for s in range(self.n_states):
            out[f"regime_{s}"] = (viterbi_states == s).astype(np.float32)

        # Soft posteriors
        for s in range(self.n_states):
            out[f"post_{s}"] = posteriors[:, s].astype(np.float32)

        logger.info(
            "HMM decode: rows=%d  state_counts=%s",
            len(out),
            {self.state_labels.get(s, s): int((viterbi_states == s).sum())
             for s in range(self.n_states)},
        )
        return out

    def check_regime_drift(
        self,
        prev_transmat: Optional[np.ndarray] = None,
        kl_threshold: float = 0.10,
    ) -> bool:
        """
        Compute symmetric KL divergence between old and new transition matrices.
        Returns True if drift exceeds kl_threshold → signals FAST HP re-search.

        Parameters
        ----------
        prev_transmat   : previous transition matrix (n_states, n_states).
                          If None, uses the internally stored _prev_transmat.
        kl_threshold    : default from config (0.10 = 10% KL)

        Returns
        -------
        bool : True = regime shift detected, trigger FAST HP re-search.
        """
        if self.model_ is None:
            return False
        old = prev_transmat if prev_transmat is not None else self._prev_transmat
        if old is None:
            return False

        new = self.model_.transmat_
        eps = 1e-9

        # Symmetrised KL  sum_ij [ new * log(new/old) + old * log(old/new) ] / 2
        kl_sym = 0.5 * np.sum(
            new * np.log((new + eps) / (old + eps)) +
            old * np.log((old + eps) / (new + eps))
        )
        drift_flag = kl_sym > kl_threshold
        logger.info(
            "Regime drift check: sym_KL=%.4f  threshold=%.2f  flag=%s",
            kl_sym, kl_threshold, drift_flag,
        )
        return drift_flag

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("RegimeHMM saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        """Load a pickled RegimeHMM."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RegimeHMM, got {type(obj)}")
        logger.info("RegimeHMM loaded ← %s", path)
        return obj

    def get_transmat(self) -> np.ndarray:
        """Return current transition matrix (n_states × n_states)."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.transmat_.copy()

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sticky_transmat(n: int, kappa: float) -> np.ndarray:
        """
        Build a sticky (Dirichlet-style) initial transition matrix.
        Diagonal entries are boosted by kappa, then rows normalised.
        This biases the HMM toward staying in the current state,
        reducing rapid state flickering on noisy daily data.
        """
        mat = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(mat, 1.0 + kappa)
        mat /= mat.sum(axis=1, keepdims=True)
        return mat

    @staticmethod
    def _reorder_model(model: GaussianHMM, order: np.ndarray) -> GaussianHMM:
        """
        Reorder states of a fitted GaussianHMM in-place according to `order`.
        order[i] = old state index that becomes new state i.

        Note: model._covars_ (private, shape n_states × n_features) is written
        directly to bypass hmmlearn's covars_ property setter, which re-validates
        shape using the expanded (n_states, n_features, n_features) form and raises
        ValueError: 'diag' covars must have shape (n_components, n_dim).
        """
        model.startprob_ = model.startprob_[order]
        model.transmat_  = model.transmat_[order][:, order]
        model.means_     = model.means_[order]
        model._covars_   = model._covars_[order]   # bypass property setter
        return model


# ════════════════════════════════════════════════════════════════════════════
# Convenience: full regime pipeline in one call
# ════════════════════════════════════════════════════════════════════════════

def run_regime_pipeline(
    nifty_df: pd.DataFrame,
    config: dict,
    fit_up_to: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, RegimeHMM]:
    """
    End-to-end: NIFTY50 raw → fitted HMM → decoded regime DataFrame.

    Parameters
    ----------
    nifty_df  : raw NIFTY50 DataFrame from DataLoader.load_nifty50()
    config    : parsed config.yaml
    fit_up_to : ISO date to cap training (e.g. "2019-12-31")
    save_path : if given, pickle the RegimeHMM to this path

    Returns
    -------
    (regime_df, hmm)
      regime_df  : full decoded DataFrame (including OOS rows)
      hmm        : fitted RegimeHMM object (for monthly refit)
    """
    feat_df   = build_nifty_hmm_features(nifty_df)
    hmm       = RegimeHMM(config)
    hmm.fit(feat_df, up_to=fit_up_to)
    regime_df = hmm.decode(feat_df)

    if save_path:
        hmm.save(save_path)

    return regime_df, hmm