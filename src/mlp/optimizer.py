"""
Section 5.3 — Vanilla gradient descent training loop.
"""

from __future__ import annotations
import numpy as np


def grad_descent(
    data: np.ndarray,
    my_mlp: dict[str, np.ndarray],
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], dict[str, np.ndarray]]:
    """
    §5.3 — Train ``my_mlp`` on ``data`` for ``iterations`` full-batch steps.

    Each iteration:
      1. forward pass
      2. compute loss and append to ``losses``
      3. backprop
      4. W_l ← W_l - lr * dW_l   for every layer

    The loss BEFORE any update should also be in ``losses`` (so the returned
    list has length ``iterations + 1``).

    Parameters
    ----------
    data : ndarray of shape (n, 3)
        Columns are (X, Y, Z). The model is trained to predict Z from (X, Y).
    """
    # TODO §5.3
    losses: list[float] = []
    return losses, my_mlp
