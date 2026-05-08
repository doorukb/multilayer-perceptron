"""
Sections 1.4 and 4.2 — Implicit-bias trick + full forward pass.

The forward pass stores each layer's post-activation output in a cache as
``A{l}`` (with ``A0 = X``), so that `backward.py` can reuse them without
recomputing.
"""

from __future__ import annotations
import numpy as np


def modify_x_w(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    §1.4 — Return (X', W') such that X · W + b == X' · W'.

    Parameters
    ----------
    x : (..., d_in) array
    w : (d_in, d_out) array
    b : (d_out,) or (1, d_out) array
    """
    # TODO §1.4
    X_new = None
    W_new = None
    return X_new, W_new


def mlp_forward(my_mlp: dict[str, np.ndarray], x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    §4.2 — Forward pass through the MLP.

    Steps:
      1. Augment ``x`` with the implicit-bias column (from §1.4).
      2. For each weight matrix in order, compute  A_next = σ(A · W),
         applying the sigmoid after every layer EXCEPT the final one.
      3. Cache each layer's output as ``A{l}``, with ``A0 = x`` (the
         original, un-augmented input).

    Returns
    -------
    cache : dict   keys 'A0', 'A1', ...
    output : ndarray   model prediction (same as the last A in the cache)
    """
    # TODO §4.2
    cache = {"A0": x}
    output = None
    return cache, output
