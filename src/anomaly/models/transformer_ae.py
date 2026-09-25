"""Transformer autoencoder detector.

linear input projection -> sinusoidal positional encoding ->
nn.TransformerEncoder -> linear output, reconstructing the window.
Attention within a window is allowed to be BIDIRECTIONAL (no causal
mask) because the whole window lies in the past by the time it's
scored -- causality is enforced at the window-boundary level (see
data/windows.py), not inside the window itself.

Reuses trainer.train_autoencoder (same shared loop as lstm_ae.py) and
the same Detector contract: fit/score/window/name, save/load, explain.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anomaly.data.windows import make_windows, window_scores_to_points
from anomaly.models.trainer import get_device, set_seed, train_autoencoder

EPS = 1e-8


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _TransformerAEModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        window: int,
        d_model: int,
        nhead: int,
        layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.register_buffer("pos_encoding", _sinusoidal_positional_encoding(window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features). No attention mask -- bidirectional
        # within the window is fine, the whole window is already in the past.
        h = self.input_proj(x) + self.pos_encoding[: x.shape[1]]
        h = self.encoder(h)
        return self.output_proj(h)


class TransformerAEDetector:
    name = "transformer_ae"

    def __init__(
        self,
        window: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_epochs: int = 30,
        patience: int = 5,
        score_mode: str = "window_mean",
        seed: int = 0,
    ):
        if score_mode not in ("window_mean", "last_step"):
            raise ValueError(f"score_mode must be 'window_mean' or 'last_step', got {score_mode!r}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.window = window
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.score_mode = score_mode
        self.seed = seed

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.model: _TransformerAEModule | None = None
        self.history: dict | None = None
        self.device = get_device()

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit(self, train: np.ndarray) -> TransformerAEDetector:
        set_seed(self.seed)
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0) + EPS
        scaled = self._scale(train)
        windows = make_windows(scaled, self.window)

        n_features = train.shape[1]
        self.model = _TransformerAEModule(
            n_features, self.window, self.d_model, self.nhead, self.layers,
            self.dim_feedforward, self.dropout,
        )
        self.history = train_autoencoder(
            self.model,
            windows,
            lr=self.lr,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
            device=self.device,
        )
        return self

    def score(self, x: np.ndarray, chunk_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before score()")
        scaled = self._scale(x)
        windows = make_windows(scaled, self.window)

        self.model.eval()
        errors = np.empty(len(windows), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(windows), chunk_size):
                batch = torch.as_tensor(windows[start : start + chunk_size], dtype=torch.float32)
                batch = batch.to(self.device)
                recon = self.model(batch)
                sq_err = (recon - batch) ** 2
                if self.score_mode == "window_mean":
                    batch_scores = sq_err.mean(dim=(1, 2))
                else:
                    batch_scores = sq_err[:, -1, :].mean(dim=1)
                errors[start : start + chunk_size] = batch_scores.cpu().numpy()

        return window_scores_to_points(errors, self.window, len(x))

    def explain(self, x_window: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        if self.model is None:
            raise RuntimeError("call fit() before explain()")
        scaled = self._scale(x_window)
        batch = torch.as_tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            recon = self.model(batch)
            per_feature = ((recon - batch) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
        order = np.argsort(-per_feature)[:top_k]
        return [(int(i), float(per_feature[i])) for i in order]

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        config = {
            "window": self.window,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "layers": self.layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "score_mode": self.score_mode,
            "seed": self.seed,
            "n_features": len(self.mean_),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        np.savez(path / "scaler.npz", mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str | Path) -> TransformerAEDetector:
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        n_features = config.pop("n_features")
        config.pop("detector_type", None)  # added by export.py for serve/state.py; not a ctor arg
        det = cls(**config)
        det.model = _TransformerAEModule(
            n_features, det.window, det.d_model, det.nhead, det.layers,
            det.dim_feedforward, det.dropout,
        )
        det.model.load_state_dict(torch.load(path / "model.pt", map_location=det.device))
        det.model.to(det.device)
        scaler = np.load(path / "scaler.npz")
        det.mean_ = scaler["mean"]
        det.std_ = scaler["std"]
        return det
