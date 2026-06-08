import numpy as np
import pytest
from mlp.init import init_weight_matrix, init_mlp

def test_init_weight_matrix_shape_and_xavier_stats():
    fan_in, out_dim = 100, 50
    W = init_weight_matrix(fan_in, out_dim)
    assert W.shape == (fan_in + 1, out_dim)
    np.testing.assert_allclose(W[-1, :], 0.0)
    expected_std = np.sqrt(1.0 / fan_in)
    assert 0.8 * expected_std < W[:-1, :].std() < 1.2 * expected_std
    assert abs(W[:-1, :].mean()) < 0.05

def test_init_mlp_layer_shapes():
    model = init_mlp([2, 5, 10, 1])
    expected = [(2, 5), (5, 10), (10, 1)]
    for i, exp_shape in enumerate(expected):
        W = model[f"W{i}"]
        assert W.shape == (exp_shape[0] + 1, exp_shape[1])

def test_untrained_activations_avoid_sigmoid_saturation():
    from mlp.forward import mlp_forward
    from mlp.data import create_train_and_test

    np.random.seed(0)
    train_data, _ = create_train_and_test(train_size=200, test_size=20)
    x = train_data[:, :2]
    model = init_mlp([2, 20, 20, 1])
    cache, _ = mlp_forward(model, x)

    for key in ("A1", "A2"):
        activations = cache[key]
        mean_abs_deviation_from_half = np.abs(np.abs(activations) - 0.5).mean()
        assert mean_abs_deviation_from_half < 0.35, (
            f"{key} activations are saturated at init "
            f"(mean |sigmoid| deviation from 0.5 = {mean_abs_deviation_from_half:.3f})"
        )
