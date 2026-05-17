from __future__ import annotations
import numpy as np
from mlp.loss import mse_loss_grad

def backprop(my_mlp: dict[str, np.ndarray], cache: dict[str, np.ndarray], label: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute gradients for every weight matrix via backpropagation by traversing the network in reverse layer order, applying the chain rule at each step. The bias weights are folded into each weight matrix as an extra row, so augmentation is required at every layer boundary.
    Takes the network paramters, cache, labels, and predictions as input, and outputs a dictionary/map of gradients that can be directly applied to the weights by the optimizer.
    """
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
            sigmoid_gradient = A * (1 - A)
            delta = (delta @ W_without_bias.T) * sigmoid_gradient
    return dcache