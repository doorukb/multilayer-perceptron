import numpy as np
import pytest

def test_loss_decreases_on_simple_problem():
    # On the synthetic dataset, gradient descent should reduce the loss over the first handful of iterations. This catches sign errors in backprop
    from mlp.data import sample_points
    from mlp.init import init_mlp
    from mlp.optimizer import grad_descent

    rng = np.random.default_rng(0)
    np.random.seed(0)
    data = sample_points(100)
    model = init_mlp([2, 5, 1])

    losses, _ = grad_descent(data, model, iterations=20, learning_rate=1e-3)

    assert len(losses) == 21  # initial + 20 updates
    assert losses[-1] < losses[0]
