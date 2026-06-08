from __future__ import annotations
import numpy as np

# compute the mean squared error loss
def mse_loss(label: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - label) ** 2))

# compute the gradient of the mean squared error loss with respect to the predictions
def mse_loss_grad(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 2 * (pred - label) / pred.size