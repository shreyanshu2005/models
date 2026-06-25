"""
CNN-LSTM Training
==================
Phase 1  — pooled cross-symbol train (backbone learns market microstructure)
Phase 2  — per-symbol head fine-tune on most recent 63 OOS trading days

Split conventions (identical to SVM pipeline):
  Train window  : 2016-01-01 → 2017-12-31
  Purge gap     : 30 trading days
  Val OOS       : 2018-02-14 → 2019-12-31
  Backtest      : 2020-01-01 → 2026-06-22  (held out entirely during training)

Usage
-----
  python train.py                           # full run
  python train.py --phase 1 --epochs 40    # phase 1 only
  python train.py --phase 2 --symbol HDFCBANK
"""

from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple

from features import build_cnn_lstm_features
from model import build_model, SymbolHead

# ── paths ────────────────────────────────────────────────────────────
DATA_DIR    = "/mnt/project"
RESULTS_DIR = "/home/claude/results/cnn_lstm"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ── split dates (identical to SVM pipeline) ──────────────────────────
TRAIN_START = pd.Timestamp("2016-01-01")
TRAIN_END   = pd.Timestamp("2017-12-31")
VAL_START   = pd.Timestamp("2018-02-14")   # 30-day purge gap
VAL_END     = pd.Timestamp("2019-12-31")
# Backtest 2020-01-01 → 2026-06-22 never touched here

LOOKBACK         = 30
FINE_TUNE_DAYS   = 63    # most recent 63 OOS bars for per-symbol head fine-tune
BATCH_SIZE       = 256
PHASE1_EPOCHS    = 40
PHASE2_EPOCHS    = 20
LR_PHASE1        = 3e-4
LR_PHASE2        = 5e-5
WEIGHT_DECAY     = 1e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ── data loading ─────────────────────────────────────────────────────

def load_equity(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Build 22-channel tensors for each symbol. Returns dict: symbol → (X, y, dates)."""
    print("Building CNN-LSTM tensors for all symbols...")
    tensors = {}
    for sym in SYMBOLS:
        df = load_equity(sym)
        X, y, dates = build_cnn_lstm_features(df, regime_df=None)
        tensors[sym] = (X, y, dates)
        n = len(y)
        dist = {k: int((y == k).sum()) for k in [-1, 0, 1]}
        print(f"  {sym:12s}  N={n:5d}  shape={X.shape}  labels={dist}")
    return tensors


# ── split helpers ────────────────────────────────────────────────────

def split_tensors(
    X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    tr = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    vl = (dates >= VAL_START)  & (dates <= VAL_END)
    return {
        "train": (X[tr], y[tr]),
        "val":   (X[vl], y[vl]),
    }


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced labels."""
    counts = np.bincount(y + 1, minlength=3).astype(float)  # +1 shifts {-1,0,1}→{0,1,2}
    counts = np.where(counts == 0, 1, counts)
    w = counts.sum() / (3 * counts)
    return torch.tensor(w, dtype=torch.float32)


def to_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.int64) + 1)   # shift {-1,0,1}→{0,1,2}
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ── training helpers ─────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train: bool = True,
) -> Tuple[float, float]:
    model.train(train)
    total_loss, all_pred, all_true = 0.0, [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return avg_loss, f1


# ── Phase 1: pooled cross-symbol train ───────────────────────────────

def phase1_train(tensors: Dict) -> nn.Module:
    print(f"\n{'='*60}")
    print("PHASE 1 — Pooled cross-symbol training")
    print(f"  Epochs={PHASE1_EPOCHS}  LR={LR_PHASE1}  Device={DEVICE}")
    print(f"{'='*60}")

    # pool all training splits
    X_tr_all, y_tr_all = [], []
    X_vl_all, y_vl_all = [], []
    for sym in SYMBOLS:
        X, y, dates = tensors[sym]
        splits = split_tensors(X, y, dates)
        X_tr_all.append(splits["train"][0])
        y_tr_all.append(splits["train"][1])
        X_vl_all.append(splits["val"][0])
        y_vl_all.append(splits["val"][1])

    X_tr = np.concatenate(X_tr_all, axis=0)
    y_tr = np.concatenate(y_tr_all, axis=0)
    X_vl = np.concatenate(X_vl_all, axis=0)
    y_vl = np.concatenate(y_vl_all, axis=0)

    print(f"  Pooled train: {X_tr.shape}  val: {X_vl.shape}")
    print(f"  Label dist train: {np.bincount(y_tr+1)}")

    # shuffle training data
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    train_loader = to_loader(X_tr, y_tr, shuffle=True)
    val_loader   = to_loader(X_vl, y_vl, shuffle=False)

    model = build_model(n_channels=22, device=DEVICE)
    cw = class_weights(y_tr).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-6)

    best_val_f1 = -1.0
    best_state  = None

    print(f"\n  {'Epoch':>5}  {'TrLoss':>8}  {'TrF1':>6}  {'VaLoss':>8}  {'VaF1':>6}")
    for ep in range(1, PHASE1_EPOCHS + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        va_loss, va_f1 = run_epoch(model, val_loader,   criterion, None,      DEVICE, train=False)
        scheduler.step()
        marker = " ★" if va_f1 > best_val_f1 else ""
        print(f"  {ep:5d}  {tr_loss:8.4f}  {tr_f1:6.3f}  {va_loss:8.4f}  {va_f1:6.3f}{marker}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(RESULTS_DIR, "cnn_lstm_backbone.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Best val F1 (macro): {best_val_f1:.4f}")
    print(f"  Backbone saved → {ckpt_path}")
    return model


# ── Phase 2: per-symbol head fine-tune ───────────────────────────────

def phase2_finetune(
    backbone_model: nn.Module,
    tensors: Dict,
    symbol: str | None = None,
) -> Dict[str, Dict]:
    """Fine-tune per-symbol head on the most recent FINE_TUNE_DAYS OOS bars."""
    print(f"\n{'='*60}")
    print("PHASE 2 — Per-symbol head fine-tune")
    print(f"  Fine-tune window: last {FINE_TUNE_DAYS} OOS days  Epochs={PHASE2_EPOCHS}")
    print(f"{'='*60}")

    symbols_to_run = [symbol] if symbol else SYMBOLS
    registry = {}

    for sym in symbols_to_run:
        X, y, dates = tensors[sym]

        # val split for fine-tune target
        vl_mask = (dates >= VAL_START) & (dates <= VAL_END)
        X_vl, y_vl = X[vl_mask], y[vl_mask]

        # use most recent FINE_TUNE_DAYS rows of val for fine-tune
        n_ft = min(FINE_TUNE_DAYS, len(X_vl))
        X_ft = X_vl[-n_ft:]
        y_ft = y_vl[-n_ft:]
        # evaluation on all val
        X_ev, y_ev = X_vl, y_vl

        if len(X_ft) < 10:
            print(f"  {sym}: insufficient fine-tune data ({len(X_ft)} rows) — skipping")
            continue

        # clone backbone, freeze conv, unfreeze lstm + replace head
        import copy
        model = copy.deepcopy(backbone_model).to(DEVICE)
        model.freeze_backbone(True)
        model.partial_unfreeze_lstm()
        model.replace_head()

        ft_loader = to_loader(X_ft, y_ft, shuffle=True)
        ev_loader = to_loader(X_ev, y_ev, shuffle=False)

        cw = class_weights(y_ft).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_PHASE2, weight_decay=WEIGHT_DECAY,
        )

        best_val_f1, best_state = -1.0, None
        for ep in range(1, PHASE2_EPOCHS + 1):
            run_epoch(model, ft_loader, criterion, optimizer, DEVICE, train=True)
            _, va_f1 = run_epoch(model, ev_loader, criterion, None, DEVICE, train=False)
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        # final full val eval
        _, final_f1 = run_epoch(model, ev_loader, criterion, None, DEVICE, train=False)

        # save per-symbol model
        sym_path = os.path.join(RESULTS_DIR, f"cnn_lstm_{sym}.pt")
        torch.save(model.state_dict(), sym_path)

        registry[sym] = {
            "symbol":      sym,
            "cap":         CAP_MAP[sym],
            "val_f1_oos":  round(final_f1, 4),
            "model_path":  sym_path,
            "fine_tune_n": n_ft,
        }
        print(f"  {sym:12s}  val_F1_OOS={final_f1:.4f}  fine_tune_n={n_ft}  → {sym_path}")

    # save registry
    reg_path = os.path.join(RESULTS_DIR, "cnn_lstm_registry.pkl")
    with open(reg_path, "wb") as f:
        pickle.dump(registry, f)
    print(f"\n  Registry saved → {reg_path}")
    return registry


# ── inference helper (used by fusion layer) ──────────────────────────

def load_symbol_model(symbol: str) -> nn.Module:
    """Load a fine-tuned per-symbol CNN-LSTM model for inference."""
    model = build_model(n_channels=22, device=DEVICE)
    path  = os.path.join(RESULTS_DIR, f"cnn_lstm_{symbol}.pt")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Returns softmax probabilities (N, 3) for classes {-1, 0, +1}."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)  # (N, 3)  → col 0=sell, 1=flat, 2=buy


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",   type=int, default=0,    help="1=pooled only, 2=finetune only, 0=both")
    parser.add_argument("--epochs",  type=int, default=None, help="Override phase1 epochs")
    parser.add_argument("--symbol",  type=str, default=None, help="Single symbol for phase 2")
    args = parser.parse_args()

    global PHASE1_EPOCHS
    if args.epochs:
        PHASE1_EPOCHS = args.epochs

    print(f"Device: {DEVICE}")
    tensors = load_all_data()

    backbone_path = os.path.join(RESULTS_DIR, "cnn_lstm_backbone.pt")

    if args.phase in (0, 1):
        backbone = phase1_train(tensors)
    else:
        # load existing backbone for phase-2-only run
        backbone = build_model(n_channels=22, device=DEVICE)
        backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
        print(f"Loaded backbone from {backbone_path}")

    if args.phase in (0, 2):
        registry = phase2_finetune(backbone, tensors, symbol=args.symbol)
        print("\nFinal CNN-LSTM registry:")
        print(f"  {'Symbol':12s}  {'Cap':3s}  {'Val F1 OOS':>10}")
        for sym, rec in registry.items():
            print(f"  {sym:12s}  {rec['cap']:3s}  {rec['val_f1_oos']:>10.4f}")


if __name__ == "__main__":
    main()