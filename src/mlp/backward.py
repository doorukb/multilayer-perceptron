"""
Section 5.2 — Backpropagation.

Uses the cache from `forward.mlp_forward` and the analytical results from
§1.1–1.3 to compute ∂L/∂W_l for every weight matrix.

The returned dict mirrors the model dict: for each ``W{l}`` in the model,
``dcache`` contains a ``dW{l}`` of the SAME shape.
"""

from __future__ import annotations
import numpy as np


def backprop(
    my_mlp: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    label: np.ndarray,
    pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    §5.2 — Compute gradients of MSE loss w.r.t. every weight matrix.

    Parameters
    ----------
    my_mlp : the model dict from `init.init_mlp`
    cache  : the cache from `forward.mlp_forward` (contains A0, A1, ...)
    label  : ground-truth targets
    pred   : the network's output for the same inputs

    Returns
    -------
    dcache : dict
        Keys 'dW0', 'dW1', ... each with the same shape as the matching W.
    """
    # TODO §5.2
    dcache: dict[str, np.ndarray] = {}
    return dcache
