from __future__ import annotations
import numpy as np

def modify_x_w(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(x)
    ones_column = np.ones((x.shape[0], 1))
    x_new = np.hstack([x, ones_column])
    b_row = np.atleast_2d(b)
    w_new = np.vstack([w, b_row])
    return x_new, w_new

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
