"""
Section 2 — Synthetic dataset.

Sampling from:
    X ~ U[-1, 1]
    Y ~ U[-1, 1]
    Err ~ N(0, 0.5)
    Z = X^2 - Y^2 + 1.2 + Err
"""

from __future__ import annotations
import numpy as np


def sample_points(n: int) -> np.ndarray:
    """
    Sample ``n`` points from the target distribution.

    Returns
    -------
    np.ndarray of shape (n, 3)
        Columns are (X, Y, Z) in that order, so ``points[:, 2]`` is Z.
    """
    # TODO §2.1
    points = np.ones((n, 3))
    return points


def create_train_and_test(
    train_size: int = 100, test_size: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a train and test set by sampling independently.

    Returns
    -------
    (train_data, test_data) : each an ndarray of shape (size, 3)
    """
    # TODO §2.3
    raise NotImplementedError
