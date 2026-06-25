"""
CNN-LSTM Architecture
======================
Input  : (batch, lookback=30, n_channels=22)
         ↓
Conv1D(filters=64, kernel=3, padding='same') + BatchNorm + ReLU
Conv1D(filters=64, kernel=3, padding='same') + BatchNorm + ReLU
MaxPool1D(pool_size=2)                        → (batch, 15, 64)
         ↓
Conv1D(filters=128, kernel=3, padding='same') + BatchNorm + ReLU
MaxPool1D(pool_size=2)                        → (batch, 7, 128)
         ↓
LSTM(hidden=128, dropout=0.3)                 → (batch, 128)
         ↓
Shared backbone output
         ↓ (per-symbol head, fine-tuned on last 63 days)
Linear(128 → 64) + ReLU + Dropout(0.3)
Linear(64 → 3)   — logits for {-1, 0, +1}

Training strategy
-----------------
Phase 1 (cross-symbol pooled):
  - Pool all 10 symbols' training tensors together.
  - Train full model for 40 epochs, cosine LR schedule.
  - Save backbone weights: cnn_lstm_backbone.pt

Phase 2 (per-symbol fine-tune):
  - Freeze backbone (optionally partially freeze).
  - Replace head with fresh symbol-specific head.
  - Fine-tune on most recent 63 trading days of OOS data.
  - Save per-symbol model: cnn_lstm_{symbol}.pt
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class CnnLstmBackbone(nn.Module):
    """
    Shared backbone: Conv layers → LSTM → 128-dim embedding.
    Input shape:  (batch, seq_len, n_channels)  — time-last for Conv1d is flipped internally.
    Output shape: (batch, lstm_hidden)
    """
    def __init__(
        self,
        n_channels: int = 22,
        conv_filters_1: int = 64,
        conv_filters_2: int = 128,
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1a = ConvBlock(n_channels, conv_filters_1, kernel=3)
        self.conv1b = ConvBlock(conv_filters_1, conv_filters_1, kernel=3)
        self.pool1  = nn.MaxPool1d(kernel_size=2)

        self.conv2  = ConvBlock(conv_filters_1, conv_filters_2, kernel=3)
        self.pool2  = nn.MaxPool1d(kernel_size=2)

        self.lstm   = nn.LSTM(
            input_size=conv_filters_2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,     # single layer, dropout handled externally
        )
        self.lstm_drop = nn.Dropout(lstm_dropout)
        self.lstm_hidden = lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_channels) → transpose for Conv1d: (batch, channels, seq)
        x = x.transpose(1, 2)               # (B, C, L)
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.transpose(1, 2)               # back to (B, L', C')
        _, (h, _) = self.lstm(x)            # h: (1, B, hidden)
        x = self.lstm_drop(h.squeeze(0))    # (B, hidden)
        return x


class SymbolHead(nn.Module):
    """Per-symbol classification head: 128 → 64 → 3"""
    def __init__(self, lstm_hidden: int = 128, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(lstm_hidden, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class CnnLstmModel(nn.Module):
    """Full model: backbone + symbol head"""
    def __init__(
        self,
        n_channels: int = 22,
        conv_filters_1: int = 64,
        conv_filters_2: int = 128,
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.3,
        n_classes: int = 3,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = CnnLstmBackbone(
            n_channels=n_channels,
            conv_filters_1=conv_filters_1,
            conv_filters_2=conv_filters_2,
            lstm_hidden=lstm_hidden,
            lstm_dropout=lstm_dropout,
        )
        self.head = SymbolHead(lstm_hidden, n_classes, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)

    def replace_head(self, new_head: Optional[SymbolHead] = None):
        """Swap in a fresh head for per-symbol fine-tuning."""
        if new_head is None:
            new_head = SymbolHead(
                self.backbone.lstm_hidden,
                n_classes=3,
                dropout=0.3,
            )
        self.head = new_head

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def partial_unfreeze_lstm(self):
        """Unfreeze LSTM only (keep Conv frozen) — good for fine-tune."""
        for p in self.backbone.lstm.parameters():
            p.requires_grad = True


def build_model(
    n_channels: int = 22,
    device: str = "cpu",
) -> CnnLstmModel:
    model = CnnLstmModel(n_channels=n_channels)
    return model.to(device)