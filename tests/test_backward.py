import numpy as np
import pytest
from mlp.activations import (
    relu_backward,
    relu_forward,
    sigmoid_backward,
    sigmoid_forward,
    tanh_backward,
    tanh_forward,
)

# assert that the backprop function matches the numerical gradient for a given activation function

def _assert_backprop_matches_numerical_gradient(activation_forward, activation_backward, *, seed: int = 42, lmbda: float = 0.0):
    from mlp.backward import backprop
    from mlp.forward import mlp_forward
    from mlp.init import init_mlp
    from mlp.loss import mse_loss
    from mlp.regularization import l2_penalty

    rng = np.random.default_rng(seed)
    n, in_dim = 8, 2
    x = rng.normal(size=(n, in_dim))
    y = rng.normal(size=(n, 1))

    np.random.seed(seed)
    model = init_mlp([in_dim, 4, 1])
    cache, pred = mlp_forward(model, x, activation=activation_forward)
    grads = backprop(model, cache, y, pred, activation_backward=activation_backward, lmbda=lmbda)

    eps = 1e-5
    atol = 1e-4
    for key, W in model.items():
        dW = grads[f"d{key}"]
        assert dW.shape == W.shape, f"shape mismatch for {key}"

        for idx in np.ndindex(W.shape):
            W[idx] += eps
            _, p_hi = mlp_forward(model, x, activation=activation_forward)
            loss_hi = mse_loss(y, p_hi) + l2_penalty(model, lmbda)

            W[idx] -= 2 * eps
            _, p_lo = mlp_forward(model, x, activation=activation_forward)
            loss_lo = mse_loss(y, p_lo) + l2_penalty(model, lmbda)

            W[idx] += eps  # restore

            numerical = (loss_hi - loss_lo) / (2 * eps)
            analytical = dW[idx]
            assert np.isclose(numerical, analytical, atol=atol), f"gradient mismatch at {key}{idx}: numerical={numerical:.6f}, analytical={analytical:.6f}"


@pytest.mark.parametrize(
    "activation_forward,activation_backward",
    [(sigmoid_forward, sigmoid_backward), (tanh_forward, tanh_backward), (relu_forward, relu_backward)],
)
@pytest.mark.parametrize("lmbda", [0.0, 0.1])
def test_backprop_matches_numerical_gradient(activation_forward, activation_backward, lmbda):
    _assert_backprop_matches_numerical_gradient(activation_forward, activation_backward, lmbda=lmbda)