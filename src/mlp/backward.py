from __future__ import annotations
from collections.abc import Callable
import numpy as np
from mlp.activations import sigmoid_backward
from mlp.loss import mse_loss_grad

# compute the gradients of the loss function with respect to the weights of the network
def backprop(
    my_mlp: dict[str, np.ndarray], # the model weights
    cache: dict[str, np.ndarray], # the cache of the forward pass
    label: np.ndarray, # the target labels
    pred: np.ndarray, # the predicted labels
    activation_backward: Callable[[np.ndarray], np.ndarray] = sigmoid_backward, # the backward pass of the activation function
) -> dict[str, np.ndarray]:
    total_layers = len(my_mlp)
    dcache: dict[str, np.ndarray] = {}
    # get the backward pass with dL/d(pred) from the loss function
    delta = mse_loss_grad(label, pred)

    for layer in range(total_layers - 1, -1, -1):
        # activations are entering this layer, shape (n, in_dim)
        A = cache[f"A{layer}"]
        # the weight matrix includes a bias row, so augment A with a column of ones before computing the gradient. This keeps bias and weight gradients in a single matrix operation
        ones_column = np.ones((A.shape[0], 1))
        A_augmented = np.hstack([A, ones_column])
        dcache[f"dW{layer}"] = A_augmented.T @ delta
        if layer > 0:
            W = my_mlp[f"W{layer}"]
            # strip the bias row before propagating delta to the previous layer. The bias has no upstream connection, so its row does not participate in the gradient flowing backward.
            W_without_bias = W[:-1, :]
            Z = cache[f"Z{layer}"]
            act_grad = activation_backward(Z)
            delta = (delta @ W_without_bias.T) * act_grad
    return dcache