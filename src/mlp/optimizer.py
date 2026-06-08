from __future__ import annotations
import numpy as np
from mlp.backward import backprop
from mlp.forward import mlp_forward
from mlp.loss import mse_loss

# resolve the batch size to use for the gradient descent
def _resolve_batch_size(batch_size: int | None, n: int) -> int:
    # Full-batch GD is the same algorithm with batch_size=n; None selects that default.
    if batch_size is None:
        return n
    else:
        return batch_size

# run the gradient descent for one epoch using mini-batch gradient descent
def _run_epoch_batches(my_mlp: dict[str, np.ndarray], inputs: np.ndarray, targets: np.ndarray, n: int, effective_batch_size: int, learning_rate: float, rng: np.random.Generator) -> None:
    perm = rng.permutation(n)
    for start in range(0, n, effective_batch_size):
        idx = perm[start : start + effective_batch_size]
        x_batch = inputs[idx]
        y_batch = targets[idx]
        cache, pred = mlp_forward(my_mlp, x_batch)
        grads = backprop(my_mlp, cache, y_batch, pred)
        for layer in my_mlp:
            my_mlp[layer] -= learning_rate * grads[f"d{layer}"]

# train via shuffled mini-batch gradient descent
# full-batch GD, mini-batch GD, and SGD are the same algorithm with different batch sizes
def grad_descent(data: np.ndarray, my_mlp: dict[str, np.ndarray], epochs: int, learning_rate: float, batch_size: int | None = None, seed: int | None = None) -> tuple[list[float], dict[str, np.ndarray]]:
    inputs = data[:, :2]
    targets = data[:, 2:3]
    n = data.shape[0]
    effective_batch_size = _resolve_batch_size(batch_size, n)
    rng = np.random.default_rng(seed)

    _, predictions = mlp_forward(my_mlp, inputs)
    losses = [mse_loss(targets, predictions)]

    for _ in range(epochs):
        _run_epoch_batches(my_mlp, inputs, targets, n, effective_batch_size, learning_rate, rng)
        _, predictions = mlp_forward(my_mlp, inputs)
        losses.append(mse_loss(targets, predictions))

    return losses, my_mlp