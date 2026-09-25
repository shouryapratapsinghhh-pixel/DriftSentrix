"""LSTM autoencoder detector.

encoder LSTM -> linear -> latent vector -> decoder LSTM (latent repeated
across the window) -> linear -> reconstructs the input window. MSE loss.

Same Detector contract as baselines (fit/score/window/name), plus
save/load (weights + config + scaler stats) and explain() for
per-feature reconstruction-error attribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anomaly.data.windows import make_windows, window_scores_to_points
from anomaly.models.trainer import get_device, set_seed, train_autoencoder

EPS = 1e-8


class _LSTMAEModule(nn.Module):
    """The actual PyTorch network. Kept separate from the sklearn-style
    detector wrapper below so trainer.py can operate on a plain nn.Module.
    """

    def __init__(self, n_features: int, hidden: int, layers: int, latent: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden, latent)
        self.decoder_lstm = nn.LSTM(latent, hidden, num_layers=layers, batch_first=True)
        self.output_fc = nn.Linear(hidden, n_features)
        self.latent = latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        window = x.shape[1]
        _, (h_n, _) = self.encoder_lstm(x)
        z = self.encoder_fc(h_n[-1])  # (batch, latent) -- last layer's final hidden state
        z_repeated = z.unsqueeze(1).expand(-1, window, -1)  # repeat latent across the window
        dec_out, _ = self.decoder_lstm(z_repeated)
        return self.output_fc(dec_out)  # (batch, window, n_features)


class LSTMAEDetector:
    name = "lstm_ae"

    def __init__(
        self,
        window: int = 32,
        hidden: int = 64,
        layers: int = 1,
        latent: int = 16,
        batch_size: int = 128,
        lr: float = 1e-3,
        max_epochs: int = 30,
        patience: int = 5,
        score_mode: str = "window_mean",
        seed: int = 0,
    ):
        if score_mode not in ("window_mean", "last_step"):
            raise ValueError(f"score_mode must be 'window_mean' or 'last_step', got {score_mode!r}")
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.latent = latent
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.score_mode = score_mode
        self.seed = seed

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.model: _LSTMAEModule | None = None
        self.history: dict | None = None
        self.device = get_device()

    # -- scaling -----------------------------------------------------

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    # -- Detector contract --------------------------------------------

    def fit(self, train: np.ndarray) -> LSTMAEDetector:
        set_seed(self.seed)
        self.mean_ = train.mean(axis=0)
        self.std_ = train.std(axis=0) + EPS
        scaled = self._scale(train)
        windows = make_windows(scaled, self.window)

        n_features = train.shape[1]
        self.model = _LSTMAEModule(n_features, self.hidden, self.layers, self.latent)
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
        windows = make_windows(scaled, self.window)  # (n_windows, window, d)

        self.model.eval()
        errors = np.empty(len(windows), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(windows), chunk_size):
                batch = torch.as_tensor(windows[start : start + chunk_size], dtype=torch.float32)
                batch = batch.to(self.device)
                recon = self.model(batch)
                sq_err = (recon - batch) ** 2  # (b, window, d)
                if self.score_mode == "window_mean":
                    batch_scores = sq_err.mean(dim=(1, 2))
                else:  # last_step
                    batch_scores = sq_err[:, -1, :].mean(dim=1)
                errors[start : start + chunk_size] = batch_scores.cpu().numpy()

        return window_scores_to_points(errors, self.window, len(x))

    # -- explainability --------------------------------------------------

    def explain(self, x_window: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """Per-feature reconstruction error for one window (shape
        (window, n_features)). Returns [(feature_idx, error), ...] for
        the top_k most-contributing features, descending.
        """
        if self.model is None:
            raise RuntimeError("call fit() before explain()")
        scaled = self._scale(x_window)
        batch = torch.as_tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            recon = self.model(batch)
            per_feature = ((recon - batch) ** 2).mean(dim=1).squeeze(0).cpu().numpy()  # (d,)
        order = np.argsort(-per_feature)[:top_k]
        return [(int(i), float(per_feature[i])) for i in order]

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        config = {
            "window": self.window,
            "hidden": self.hidden,
            "layers": self.layers,
            "latent": self.latent,
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
    def load(cls, path: str | Path) -> LSTMAEDetector:
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        n_features = config.pop("n_features")
        config.pop("detector_type", None)  # added by export.py for serve/state.py; not a ctor arg
        det = cls(**config)
        det.model = _LSTMAEModule(n_features, det.hidden, det.layers, det.latent)
        det.model.load_state_dict(torch.load(path / "model.pt", map_location=det.device))
        det.model.to(det.device)
        scaler = np.load(path / "scaler.npz")
        det.mean_ = scaler["mean"]
        det.std_ = scaler["std"]
        return det
