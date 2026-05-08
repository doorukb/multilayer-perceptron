"""
Section 6 — Hyperparameter tuning.

Hard rule from the problem statement: the test set must not influence any
hyperparameter choice. Use a *validation* split carved out of the training
data for all model selection.
"""

from __future__ import annotations
import numpy as np


def split_train_validation(
    train_data: np.ndarray, val_fraction: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    §6.2 — Reserve ``val_fraction`` of the *training* data as validation.
    """
    # TODO §6.2
    raise NotImplementedError


def grad_descent_with_validation(
    train_data: np.ndarray,
    val_data: np.ndarray,
    my_mlp: dict[str, np.ndarray],
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], list[float], dict[str, np.ndarray]]:
    """
    §6.2 — Same as `optimizer.grad_descent`, but also evaluates loss on
    ``val_data`` every iteration WITHOUT updating weights from it.

    Returns
    -------
    train_losses, val_losses, final_model
    """
    # TODO §6.2
    raise NotImplementedError


def hyperparameter_search(train_data: np.ndarray, search_space: dict) -> dict:
    """
    §6.3 — Search over architectures and learning rates using only train+val
    data. Return the best configuration and the trained model.

    `search_space` shape is up to you — typical keys: 'output_sizes',
    'learning_rate', 'iterations'.
    """
    # TODO §6.3
    raise NotImplementedError
