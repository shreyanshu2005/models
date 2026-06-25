"""
pipeline/run_week6_7.py
========================
Week 6–7 orchestrator: CNN-LSTM training pipeline.
Run exactly like the previous week scripts:

    python pipeline/run_week6_7.py

Optional flags:
    --phase 1          run Phase 1 (pooled backbone) only
    --phase 2          run Phase 2 (per-symbol fine-tune) only
    --symbol HDFCBANK  fine-tune a single symbol (Phase 2 only)
    --epochs 40        override Phase 1 epoch count
    --data-dir path    override raw CSV directory  (default: data/raw)
    --results-dir path override results output dir (default: results/cnn_lstm)

Prerequisites
-------------
- Week 1 pipeline completed  (data/raw/*.csv exist)
- Week 2-3 pipeline completed (results/svm_registry_initial.pkl exists)

Outputs
-------
  results/cnn_lstm/cnn_lstm_backbone.pt        shared backbone weights
  results/cnn_lstm/cnn_lstm_{SYMBOL}.pt        per-symbol fine-tuned model
  results/cnn_lstm/cnn_lstm_registry.pkl       registry dict for fusion layer
  results/cnn_lstm/week6_7_manifest.json       artefact manifest
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import copy
import pickle
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# resolve repo root so imports work regardless of cwd
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.dirname(_SCRIPT_DIR)
_CNN_LSTM_DIR = os.path.join(_REPO_ROOT, "models", "cnn_lstm")

# validate the folder exists before inserting into path
if not os.path.isdir(_CNN_LSTM_DIR):
    print(f"\n[ERROR] Cannot find models/cnn_lstm/ at: {_CNN_LSTM_DIR}")
    print("  Make sure your repo looks like:")
    print("    models/")
    print("      cnn_lstm/")
    print("        features.py   (rename from cnn_lstm_features.py)")
    print("        model.py      (rename from cnn_lstm_model.py)")
    print("        train.py      (rename from cnn_lstm_train.py)")
    sys.exit(1)

# check individual files exist with helpful rename hint
for _fname, _download_name in [
    ("features.py", "cnn_lstm_features.py"),
    ("model.py",    "cnn_lstm_model.py"),
    ("train.py",    "cnn_lstm_train.py"),
]:
    if not os.path.exists(os.path.join(_CNN_LSTM_DIR, _fname)):
        # check if user forgot to rename
        if os.path.exists(os.path.join(_CNN_LSTM_DIR, _download_name)):
            print(f"\n[ERROR] Found '{_download_name}' but need '{_fname}'")
            print(f"  Rename it:  {_download_name}  →  {_fname}")
        else:
            print(f"\n[ERROR] Missing file: models/cnn_lstm/{_fname}")
        sys.exit(1)

sys.path.insert(0, _CNN_LSTM_DIR)

from features import build_cnn_lstm_features
from model    import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("week6_7")

SYMBOLS = [
    "HDFCBANK", "BAJFINANCE", "RELIANCE", "BRITANNIA",
    "INDIGO", "ABB", "MOTHERSON", "HEROMOTOCO", "BOSCHLTD", "MARUTI",
]
CAP_MAP = {
    "HDFCBANK": "LC", "BAJFINANCE": "LC", "RELIANCE": "LC",
    "HEROMOTOCO": "LC", "MARUTI": "LC",
    "BRITANNIA": "MC", "INDIGO": "MC", "ABB": "MC",
    "MOTHERSON": "MC", "BOSCHLTD": "MC",
}

TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END   = pd.Timestamp("2017-12-31")
VAL_START   = pd.Timestamp("2018-02-14")
VAL_END     = pd.Timestamp("2019-12-31")

BATCH_SIZE    = 256
PHASE1_EPOCHS = 40
PHASE2_EPOCHS = 15
LR_PHASE1     = 3e-4
LR_PHASE2     = 3e-5
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def _load_equity(symbol, data_dir):
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, usecols=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _load_regime(data_dir):
    pkl_path = os.path.normpath(os.path.join(data_dir, "..", "regime", "hmm_initial.pkl"))
    if not os.path.exists(pkl_path):
        log.warning("Regime pkl not found — regime channels will be zero (non-blocking)")
        return None

    log.info(f"Regime pkl found at {pkl_path}")

    # The pkl was saved by Week 1/2-3 and contains a RegimeHMM object.
    # The CNN-LSTM feature builder expects a per-symbol DataFrame or None.
    # We don't use the raw HMM object here — regime channels are handled
    # inside build_cnn_lstm_features directly from the equity data.
    # Return a sentinel so the caller knows the pkl exists but can't be used as a dict.
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        # if it's a dict keyed by symbol name, use it directly
        if isinstance(obj, dict) and any(s in obj for s in ["HDFCBANK", "RELIANCE"]):
            log.info("Regime pkl is a per-symbol dict — injecting HMM channels")
            return obj
        # otherwise it's a RegimeHMM model object — not directly usable here
        log.info(
            f"Regime pkl contains a {type(obj).__name__} object (not a per-symbol dict). "
            "Regime channels in CNN-LSTM will be zero — non-blocking."
        )
        return None
    except Exception as e:
        log.warning(f"Could not load regime pkl ({e}) — falling back to zero regime channels")
        return None


def _class_weights(y):
    counts = np.bincount(y + 1, minlength=3).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = counts.sum() / (3 * counts)
    return torch.tensor(w, dtype=torch.float32)


def _to_loader(X, y, shuffle):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.int64) + 1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)


def _run_epoch(model, loader, criterion, optimizer, train):
    model.train(train)
    total_loss, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    return total_loss / len(loader.dataset), f1_score(trues, preds, average="macro", zero_division=0)


# ── Step 1 ────────────────────────────────────────────────────────────
def step_build_tensors(data_dir, regime_data):
    log.info("─" * 55)
    log.info("STEP 1  Building 22-channel tensors")
    tensors = {}
    for sym in SYMBOLS:
        df        = _load_equity(sym, data_dir)
        regime_df = regime_data.get(sym) if regime_data else None
        X, y, dates = build_cnn_lstm_features(df, regime_df=regime_df)
        tensors[sym] = (X, y, dates)
        dist = {k: int((y == k).sum()) for k in [-1, 0, 1]}
        log.info(f"  {sym:12s}  N={len(y):5d}  shape={X.shape}  labels={dist}")
    return tensors


# ── Step 2 ────────────────────────────────────────────────────────────
def step_phase1(tensors, results_dir, epochs):
    log.info("─" * 55)
    log.info(f"STEP 2  Phase 1: pooled training  ({epochs} epochs, device={DEVICE})")

    X_tr_all, y_tr_all, X_vl_all, y_vl_all = [], [], [], []
    for sym in SYMBOLS:
        X, y, dates = tensors[sym]
        tr = (dates >= TRAIN_START) & (dates <= TRAIN_END)
        vl = (dates >= VAL_START)   & (dates <= VAL_END)
        X_tr_all.append(X[tr]); y_tr_all.append(y[tr])
        X_vl_all.append(X[vl]); y_vl_all.append(y[vl])

    X_tr = np.concatenate(X_tr_all); y_tr = np.concatenate(y_tr_all)
    X_vl = np.concatenate(X_vl_all); y_vl = np.concatenate(y_vl_all)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    log.info(f"  Pooled train={X_tr.shape}  val={X_vl.shape}  label_dist={np.bincount(y_tr+1)}")

    train_loader = _to_loader(X_tr, y_tr, True)
    val_loader   = _to_loader(X_vl, y_vl, False)

    model     = build_model(22, DEVICE)
    cw        = _class_weights(y_tr).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1, best_state = -1.0, None
    log.info(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrF1':>6}  {'VaLoss':>8}  {'VaF1':>6}")
    for ep in range(1, epochs + 1):
        tl, tf = _run_epoch(model, train_loader, criterion, optimizer, True)
        vl, vf = _run_epoch(model, val_loader,   criterion, None,      False)
        scheduler.step()
        mark = " ★" if vf > best_val_f1 else ""
        log.info(f"  {ep:4d}  {tl:8.4f}  {tf:6.3f}  {vl:8.4f}  {vf:6.3f}{mark}")
        if vf > best_val_f1:
            best_val_f1 = vf
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt = os.path.join(results_dir, "cnn_lstm_backbone.pt")
    torch.save(model.state_dict(), ckpt)
    log.info(f"  Best val F1={best_val_f1:.4f}  backbone → {ckpt}")
    return model


# ── Step 3 ────────────────────────────────────────────────────────────
def step_phase2(backbone, tensors, results_dir, symbol_filter):
    log.info("─" * 55)
    log.info("STEP 3  Phase 2: per-symbol fine-tune")

    registry = {}
    for sym in ([symbol_filter] if symbol_filter else SYMBOLS):
        X, y, dates = tensors[sym]
        vl          = (dates >= VAL_START) & (dates <= VAL_END)
        X_vl, y_vl  = X[vl], y[vl]
        if len(X_vl) < 10:
            log.warning(f"  {sym}: too few val rows — skipping"); continue

        ev_loader = _to_loader(X_vl, y_vl, False)
        ft_loader = _to_loader(X_vl, y_vl, True)
        cw        = _class_weights(y_vl).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw)

        model = copy.deepcopy(backbone).to(DEVICE)
        for p in model.parameters(): p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)

        best_f1, best_state, no_improve = -1.0, None, 0
        for ep in range(1, PHASE2_EPOCHS + 1):
            _run_epoch(model, ft_loader, criterion, optimizer, True)
            _, vf = _run_epoch(model, ev_loader, criterion, None, False)
            if vf > best_f1:
                best_f1 = vf
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE: break

        model.load_state_dict(best_state)
        _, final_f1 = _run_epoch(model, ev_loader, criterion, None, False)
        sym_path = os.path.join(results_dir, f"cnn_lstm_{sym}.pt")
        torch.save(model.state_dict(), sym_path)
        registry[sym] = {"symbol": sym, "cap": CAP_MAP[sym],
                         "val_f1_oos": round(final_f1, 4), "model_path": sym_path}
        log.info(f"  {sym:12s}  val_F1_OOS={final_f1:.4f}  → {sym_path}")

    reg_path = os.path.join(results_dir, "cnn_lstm_registry.pkl")
    with open(reg_path, "wb") as f: pickle.dump(registry, f)
    log.info(f"  Registry → {reg_path}")
    return registry


# ── Step 4 ────────────────────────────────────────────────────────────
def step_manifest(registry, results_dir):
    log.info("─" * 55)
    log.info("STEP 4  Writing manifest")
    manifest = {
        "pipeline": "QBEAST-AI.N", "week": "6-7", "model": "CNN-LSTM",
        "device": DEVICE,
        "train_start": str(TRAIN_START.date()), "train_end": str(TRAIN_END.date()),
        "val_start": str(VAL_START.date()),     "val_end":   str(VAL_END.date()),
        "architecture": {"input_channels": 22, "lookback": 30,
                         "conv_filters": [64, 128], "lstm_hidden": 128},
        "symbols": registry,
        "backbone_path": os.path.join(results_dir, "cnn_lstm_backbone.pt"),
        "registry_path": os.path.join(results_dir, "cnn_lstm_registry.pkl"),
    }
    mpath = os.path.join(results_dir, "week6_7_manifest.json")
    with open(mpath, "w") as f: json.dump(manifest, f, indent=2)
    log.info(f"  Manifest → {mpath}")
    log.info("─" * 55)
    log.info("FINAL RESULTS")
    log.info(f"  {'Symbol':12s}  {'Cap':3s}  {'Val F1 OOS':>10}")
    for sym, rec in registry.items():
        log.info(f"  {sym:12s}  {rec['cap']:3s}  {rec['val_f1_oos']:>10.4f}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QBEAST-AI.N Week 6-7: CNN-LSTM")
    parser.add_argument("--phase",       type=int, default=0)
    parser.add_argument("--symbol",      type=str, default=None)
    parser.add_argument("--epochs",      type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--data-dir",    type=str,
                        default=os.path.join(_REPO_ROOT, "data", "raw"))
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(_REPO_ROOT, "results", "cnn_lstm"))
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    log.info("=" * 55)
    log.info("QBEAST-AI.N  ·  Week 6-7  ·  CNN-LSTM")
    log.info(f"  device={DEVICE}  data={args.data_dir}  results={args.results_dir}")
    log.info("=" * 55)

    t0          = time.time()
    regime_data = _load_regime(args.data_dir)
    tensors     = step_build_tensors(args.data_dir, regime_data)

    backbone_path = os.path.join(args.results_dir, "cnn_lstm_backbone.pt")
    if args.phase in (0, 1):
        backbone = step_phase1(tensors, args.results_dir, args.epochs)
    else:
        if not os.path.exists(backbone_path):
            log.error(f"Backbone not found at {backbone_path}. Run phase 1 first.")
            sys.exit(1)
        backbone = build_model(22, DEVICE)
        backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
        log.info(f"Loaded backbone from {backbone_path}")

    registry = {}
    if args.phase in (0, 2):
        registry = step_phase2(backbone, tensors, args.results_dir, args.symbol)

    if registry:
        step_manifest(registry, args.results_dir)

    log.info(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()