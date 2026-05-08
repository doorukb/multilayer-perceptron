"""
Section 5.1 — Mean squared error loss.
"""

from __future__ import annotations
import numpy as np


def mse_loss(label: np.ndarray, pred: np.ndarray) -> float:
    """
    §5.1 — Mean squared error between ``label`` and ``pred``.
    """
    # TODO §5.1
    pass


def mse_loss_grad(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. ``pred`` (used as the seed of backprop in §5.2).

    For loss = mean((pred - label)^2),
        d_loss / d_pred = 2 * (pred - label) / N
    where N is the number of samples.
    """
    # TODO (used by backward.py)
    pass
