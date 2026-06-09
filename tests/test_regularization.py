import numpy as np
from mlp.backward import backprop
from mlp.forward import mlp_forward
from mlp.init import init_mlp
from mlp.regularization import l2_penalty

def test_l2_penalty_zero_lambda():
    model = {"W0": np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])}
    assert l2_penalty(model, lmbda=0.0) == 0.0

def test_l2_penalty_known_value():
    model = {"W0": np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])}
    lmbda = 2.0
    expected = 0.5 * lmbda * (1 + 4 + 9 + 16)
    assert l2_penalty(model, lmbda) == expected

# bias lives in the last row; changing it must not change the penalty
def test_l2_penalty_excludes_bias_row():
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    model_a = {"W0": np.vstack([weights, [100.0, 100.0]])}
    model_b = {"W0": np.vstack([weights, [0.0, 0.0]])}
    lmbda = 0.5
    assert l2_penalty(model_a, lmbda) == l2_penalty(model_b, lmbda)

def test_l2_penalty_sums_all_layers():
    model = {"W0": np.array([[1.0, 0.0], [0.0, 0.0]]), "W1": np.array([[2.0], [0.0]])}
    lmbda = 1.0
    expected = 0.5 * lmbda * (1 + 4)
    assert l2_penalty(model, lmbda) == expected

def test_l2_grad_diff_equals_lambda_times_weights():
    seed = 42
    lmbda = 0.3
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 2))
    y = rng.normal(size=(8, 1))

    np.random.seed(seed)
    model = init_mlp([2, 4, 1])
    cache, pred = mlp_forward(model, x)

    grads_0 = backprop(model, cache, y, pred, lmbda=0.0)
    grads_l = backprop(model, cache, y, pred, lmbda=lmbda)

    for key, W in model.items():
        diff = grads_l[f"d{key}"] - grads_0[f"d{key}"]
        expected = np.zeros_like(W)
        expected[:-1, :] = lmbda * W[:-1, :]
        np.testing.assert_allclose(diff, expected)