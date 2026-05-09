from __future__ import annotations
import numpy as np
from mlp.loss import mse_loss_grad

def backprop(my_mlp: dict[str, np.ndarray], cache: dict[str, np.ndarray], label: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    total_layers = len(my_mlp)
    dcache: dict[str, np.ndarray] = {}
    delta = mse_loss_grad(label, pred)

    for layer in range(total_layers - 1, -1, -1):
        A = cache[f"A{layer}"]
        ones_column = np.ones((A.shape[0], 1))
        A_augmented = np.hstack([A, ones_column])
        dcache[f"dW{layer}"] = A_augmented.T @ delta
        if layer > 0:
            W = my_mlp[f"W{layer}"]
            W_without_bias = W[:-1, :]
            sigmoid_gradient = A * (1 - A)
            delta = (delta @ W_without_bias.T) * sigmoid_gradient
    return dcache