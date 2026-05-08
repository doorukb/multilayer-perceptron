"""
Section 4.1 — Sigmoid activation.

The backward function uses the closed-form result from §1.1
(see `math_derivations.md`).
"""

from __future__ import annotations
import numpy as np


def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """
    Elementwise sigmoid: σ(x) = 1 / (1 + exp(-x)).
    """
    # TODO §4.1
    pass


def sigmoid_backward(x: np.ndarray) -> np.ndarray:
    """
    Elementwise derivative of sigmoid w.r.t. its input, evaluated at ``x``.

    Uses the simplified form from §1.1.
    """
    # TODO (used by backward.py once §5.2 is implemented)
    pass
