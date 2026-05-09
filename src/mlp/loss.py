from __future__ import annotations
import numpy as np

def mse_loss(label: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - label) ** 2))

def mse_loss_grad(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 2 * (pred - label) / pred.size
