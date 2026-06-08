import numpy as np
import pytest
from mlp.forward import modify_x_w, mlp_forward

def test_modify_x_w_vector():
    X = np.array([1, 2, 3, 4])
    W = np.ones((4, 2))
    b = np.ones(2)
    X_new, W_new = modify_x_w(X, W, b)
    np.testing.assert_allclose(np.squeeze(X_new @ W_new), X @ W + b)

def test_modify_x_w_matrix():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    W = np.array([[1, 2, 3, 10], [4, 5, 6, 10], [7, 8, 9, 10]])
    b = np.array([[5, 4, 6, 9]])
    X_new, W_new = modify_x_w(X, W, b)
    np.testing.assert_allclose(X_new @ W_new, X @ W + b)

def test_mlp_forward_single_layer_affine():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    w = rng.normal(size=(3, 2))
    b = rng.normal(size=(1, 2))
    model = {"W0": np.vstack([w, b])}
    _, out = mlp_forward(model, x)
    np.testing.assert_allclose(out, x @ w + b)

def test_mlp_forward_cache_keys():
    from mlp.init import init_mlp
    model = init_mlp([2, 5, 1])
    x = np.random.randn(10, 2)
    cache, output = mlp_forward(model, x)
    # A0 plus one A per layer
    n_layers = sum(1 for k in model if k.startswith("W"))
    expected_keys = {f"A{i}" for i in range(n_layers + 1)}
    assert expected_keys.issubset(cache.keys())
    assert output.shape[0] == 10

# test that the mlp_forward function can use a different activation function for the hidden layers
def test_mlp_forward_swappable_activation():
    from mlp.activations import relu_forward, tanh_forward
    from mlp.init import init_mlp

    np.random.seed(0)
    x = np.random.randn(8, 2)
    model_sigmoid = init_mlp([2, 6, 1])
    _, out_sigmoid = mlp_forward(model_sigmoid, x)
    np.random.seed(0)
    model_relu = init_mlp([2, 6, 1])
    cache_relu, out_relu = mlp_forward(model_relu, x, activation=relu_forward)
    np.random.seed(0)
    model_tanh = init_mlp([2, 6, 1])
    cache_tanh, out_tanh = mlp_forward(model_tanh, x, activation=tanh_forward)

    assert np.all(cache_relu["A1"] >= 0)
    assert np.all(cache_tanh["A1"] <= 1) and np.all(cache_tanh["A1"] >= -1)
    assert not np.allclose(out_sigmoid, out_relu)
    assert not np.allclose(out_sigmoid, out_tanh)

# test that the output layer stays linear for scalar regression
def test_mlp_forward_output_layer_stays_linear():
    from mlp.activations import relu_forward
    from mlp.init import init_mlp

    np.random.seed(1)
    model = init_mlp([2, 4, 1])
    x = np.random.randn(5, 2)
    cache, out = mlp_forward(model, x, activation=relu_forward)

    np.testing.assert_allclose(out, cache["A2"])
    assert np.all(cache["A1"] >= 0)
    assert not np.allclose(out, relu_forward(out))