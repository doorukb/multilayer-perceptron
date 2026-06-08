from __future__ import annotations
import numpy as np
from mlp.forward import mlp_forward
from mlp.backward import backprop
from mlp.loss import mse_loss

# gradient descent optimizer
def grad_descent(data: np.ndarray, my_mlp: dict[str, np.ndarray], epochs: int, learning_rate: float, batch_size: int | None = None, seed: int | None = None) -> tuple[list[float], dict[str, np.ndarray]]:
    inputs = data[:, :2]
    targets = data[:, 2:3]
    n = data.shape[0]
    if batch_size is None:
        batch_size = n
    # random number to shuffle the data
    rng = np.random.default_rng(seed)

    _, predictions = mlp_forward(my_mlp, inputs)
    # compute the loss
    losses = [mse_loss(targets, predictions)]

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch = inputs[idx]
            y_batch = targets[idx]
            cache, pred = mlp_forward(my_mlp, x_batch)
            grads = backprop(my_mlp, cache, y_batch, pred)
            for layer in my_mlp:
                my_mlp[layer] -= learning_rate * grads[f"d{layer}"]

        _, predictions = mlp_forward(my_mlp, inputs)
        losses.append(mse_loss(targets, predictions))

    return losses, my_mlp