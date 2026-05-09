from __future__ import annotations
import numpy as np

from mlp.forward import mlp_forward
from mlp.backward import backprop
from mlp.loss import mse_loss


def grad_descent(data: np.ndarray, my_mlp: dict[str, np.ndarray], iterations: int, learning_rate: float,) -> tuple[list[float], dict[str, np.ndarray]]:
    # Each iteration makes a forward pass, computes MSE loss, backpropagates, and then updates each weight matrices : W_layer -= learning_rate * dW_layer

    # Split into features (X, Y) and targets (Z); keep Z 2D for safe broadcasting
    inputs = data[:, :2]
    targets = data[:, 2:3]

    _, predictions = mlp_forward(my_mlp, inputs)
    losses = [mse_loss(targets, predictions)]

    for _ in range(iterations):
        cache, predictions = mlp_forward(my_mlp, inputs)
        gradients = backprop(my_mlp, cache, targets, predictions)
        for layer in my_mlp:
            my_mlp[layer] -= learning_rate * gradients[f"d{layer}"]

        _, predictions = mlp_forward(my_mlp, inputs)
        losses.append(mse_loss(targets, predictions))

    return losses, my_mlp
