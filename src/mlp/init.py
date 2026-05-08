"""
Section 3 — Model initialization.

Weights are sampled from N(mean=1, std=0.25). The MLP is stored as a dict
``{'W0': ndarray, 'W1': ndarray, ...}`` to match the notebook's convention.
"""

from __future__ import annotations
import numpy as np


def init_weight_matrix(In: int = 2, Out: int = 1) -> np.ndarray:
    """
    §3.1 — Return an (In x Out) matrix sampled from N(1, 0.25).
    """
    # TODO §3.1
    weight_matrix = np.ones((In, Out))
    return weight_matrix


def init_mlp(output_sizes: list[int] = [5, 1]) -> dict[str, np.ndarray]:
    """
    §3.2 — Build an MLP whose layer output sizes follow ``output_sizes``.

    The first element of ``output_sizes`` is treated as the input dim; each
    subsequent element is the output dim of that layer (and the input dim of
    the next).

    Example: ``[2, 5, 10, 1]`` → matrices of shape (2, 5), (5, 10), (10, 1).

    Note: when you wire in the implicit-bias trick from §1.4, the input dim
    of each weight matrix should be incremented by 1 to absorb the bias row.
    Decide whether to handle that here or inside `mlp_forward` and document
    your choice.
    """
    # TODO §3.2
    model: dict[str, np.ndarray] = {}
    return model
