from __future__ import annotations
from collections.abc import Callable
import numpy as np
from mlp.activations import sigmoid_backward, sigmoid_forward
from mlp.backward import backprop
from mlp.forward import mlp_forward
from mlp.loss import mse_loss

# resolve the batch size to use for the gradient descent
def _resolve_batch_size(batch_size: int | None, n: int) -> int:
    # Full-batch GD is the same algorithm with batch_size=n; None selects that default.
    return n if batch_size is None else batch_size

# run the gradient descent for one epoch using mini-batch gradient descent
def _run_epoch_batches(
    my_mlp: dict[str, np.ndarray], # the model weights
    inputs: np.ndarray, # the input data
    targets: np.ndarray, # the target labels
    n: int, # the number of data points
    effective_batch_size: int, # the effective batch size
    learning_rate: float, # the learning rate
    rng: np.random.Generator, 
    activation: Callable[[np.ndarray], np.ndarray], # the activation function
    activation_backward: Callable[[np.ndarray], np.ndarray], # the backward pass of the activation function
) -> None:
    perm = rng.permutation(n)
    for start in range(0, n, effective_batch_size):
        idx = perm[start : start + effective_batch_size]
        x_batch = inputs[idx]
        y_batch = targets[idx]
        cache, pred = mlp_forward(my_mlp, x_batch, activation=activation)
        grads = backprop(my_mlp, cache, y_batch, pred, activation_backward=activation_backward)
        for layer in my_mlp:
            my_mlp[layer] -= learning_rate * grads[f"d{layer}"]

# train the model using mini-batch gradient descent
def grad_descent(
    data: np.ndarray, # the input data
    my_mlp: dict[str, np.ndarray], # the model weights
    epochs: int, # the number of epochs
    learning_rate: float, # the learning rate
    batch_size: int | None = None, # the batch size
    seed: int | None = None, # the random seed
    activation: Callable[[np.ndarray], np.ndarray] = sigmoid_forward, # the activation function
    activation_backward: Callable[[np.ndarray], np.ndarray] = sigmoid_backward, # the backward pass of the activation function
) -> tuple[list[float], dict[str, np.ndarray]]:
    inputs = data[:, :2]
    targets = data[:, 2:3]
    n = data.shape[0]
    effective_batch_size = _resolve_batch_size(batch_size, n)
    rng = np.random.default_rng(seed)

    _, predictions = mlp_forward(my_mlp, inputs, activation=activation)
    losses = [mse_loss(targets, predictions)]

    for _ in range(epochs):
        _run_epoch_batches(my_mlp, inputs, targets, n, effective_batch_size, learning_rate, rng, activation, activation_backward)
        _, predictions = mlp_forward(my_mlp, inputs, activation=activation)
        losses.append(mse_loss(targets, predictions))

    return losses, my_mlp