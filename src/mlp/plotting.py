"""
Plotting helpers (Sections 2, 5, 6).

Kept separate from `data.py` and the model code so the math modules have
zero matplotlib dependency — useful for headless testing.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401  (used by callers below)


def plot_3d_scatter(points: np.ndarray) -> None:
    """
    §2.2 — Plot points as a 3D scatter colored by Z (blue=low, red=high),
    with opacity 0.5.
    """
    # TODO §2.2
    plt.show()


def plot_train_and_test(train_data: np.ndarray, test_data: np.ndarray) -> None:
    """
    §2.3 — Same axes as `plot_3d_scatter` but plot train and test with
    different markers so they're distinguishable.
    """
    # TODO §2.3
    plt.show()


def plot_data_and_pred(my_mlp, train_points: np.ndarray, test_points: np.ndarray) -> None:
    """
    §5.4 — 3D scatter of train/test points + a surface plot of the model's
    prediction across the (x, y) input domain.
    """
    # TODO §5.4
    plt.show()


def plot_learning_curves(curves: dict[str, list[float]]) -> None:
    """
    §6 — Plot one or more learning curves on the same axes.

    Convention: pairs of curves for the same model share a color, with the
    validation curve drawn dashed.
    """
    # TODO §6.2
    plt.show()
