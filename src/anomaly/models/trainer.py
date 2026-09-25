"""Shared training loop for autoencoder-style deep detectors (LSTM-AE,
Transformer-AE both use this -- one loop, reused, not duplicated per model).

Enforces the non-negotiable rules from AGENTS.md: chronological
validation split (never shuffle across time between train/val), full
seeding, early stopping with best-weights restore, gradient clipping.
"""

from __future__ import annotations

import copy
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: torch.nn.Module,
    train_windows: np.ndarray,
    *,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 30,
    patience: int = 5,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> dict:
    """Train a reconstruction autoencoder on (n_windows, window, n_features)
    data. Validation split is the LAST 20% of windows, chronological (no
    shuffling across the train/val boundary; shuffling batches WITHIN
    training is fine and expected).

    Returns a history dict: train_loss, val_loss (one entry per epoch
    actually run), best_val_loss, epochs_trained. Restores the model's
    best-validation-loss weights in place before returning.
    """
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    if device.type != "cuda":
        use_amp = False  # spec: mixed precision only on CUDA

    n = len(train_windows)
    val_size = max(1, int(0.2 * n))
    train_w = train_windows[:-val_size]
    val_w = train_windows[-val_size:]
    if len(train_w) == 0:
        raise ValueError("not enough windows to leave a non-empty training split")

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_w, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,  # shuffling windows WITHIN an epoch is fine; splits stay chronological
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_w, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                recon = model(batch)
                loss = torch.nn.functional.mse_loss(recon, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * len(batch)
        train_loss = train_loss_sum / len(train_w)

        model.eval()
        val_loss_sum = 0.0
        with torch.inference_mode():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = torch.nn.functional.mse_loss(recon, batch)
                val_loss_sum += loss.item() * len(batch)
        val_loss = val_loss_sum / len(val_w)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info("epoch %d: train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_trained"] = len(history["train_loss"])
    return history
