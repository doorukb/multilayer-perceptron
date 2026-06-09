from __future__ import annotations
import numpy as np

# calculate the l2 penalty for the model
# a Gaussian prior favoring small weights penalizes complexity without shrinking the bias offset
# each weight matrix W{l} has shape (fan_in + 1, out_dim) with the bias in the last row (see init.py)
# only W[:-1, :] is penalized: biases control the output baseline / decision-boundary offset, not input sensitivity, so regularizing them would unfairly shrink the model's ability to fit the data's mean level.
def l2_penalty(my_mlp: dict[str, np.ndarray], lmbda: float) -> float:
    if lmbda == 0.0:
        return 0.0
    sq_sum = sum(np.sum(W[:-1, :] ** 2) for W in my_mlp.values())
    return 0.5 * lmbda * float(sq_sum)

# per layer gradient of the l2 penalty with respect to the weights
def l2_penalty_grad(my_mlp: dict[str, np.ndarray], lmbda: float) -> dict[str, np.ndarray]:
    if lmbda == 0.0:
        return {f"d{key}": np.zeros_like(W) for key, W in my_mlp.items()}

    # cache the gradients
    dcache : dict[str, np.ndarray] = {}
    # calculate the gradient for each layer
    for key, W in my_mlp.items():
        dW = np.vstack([lmbda * W[:-1, :], np.zeros((1, W.shape[1]))])
        dcache[f"d{key}"] = dW
    return dcache