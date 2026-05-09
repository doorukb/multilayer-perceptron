from __future__ import annotations
import numpy as np
from mlp.activations import sigmoid_forward

def modify_x_w(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(x)
    ones_column = np.ones((x.shape[0], 1))
    x_new = np.hstack([x, ones_column])
    w_new = np.vstack([w, np.atleast_2d(b)])
    return x_new, w_new


def mlp_forward(my_mlp: dict[str, np.ndarray], x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = np.atleast_2d(x)
    cache = {"A0": x}
    n_layers = len(my_mlp)
    A = x

    for l in range(n_layers):
        W = my_mlp[f"W{l}"]
        A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
        Z = A_aug @ W
        A = sigmoid_forward(Z) if l < n_layers - 1 else Z
        cache[f"A{l + 1}"] = A

    return cache, A
