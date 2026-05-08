"""Tests for src/mlp/forward.py (§1.4, §4.2)."""
import numpy as np
import pytest
from mlp.forward import modify_x_w, mlp_forward


@pytest.mark.skip(reason="Enable once §1.4 modify_x_w is implemented")
def test_modify_x_w_vector():
    """Mirrors the notebook's vector test case."""
    X = np.array([1, 2, 3, 4])
    W = np.ones((4, 2))
    b = np.ones(2)
    X_new, W_new = modify_x_w(X, W, b)
    np.testing.assert_allclose(X_new @ W_new, X @ W + b)


@pytest.mark.skip(reason="Enable once §1.4 modify_x_w is implemented")
def test_modify_x_w_matrix():
    """Mirrors the notebook's matrix test case."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    W = np.array([[1, 2, 3, 10], [4, 5, 6, 10], [7, 8, 9, 10]])
    b = np.array([[5, 4, 6, 9]])
    X_new, W_new = modify_x_w(X, W, b)
    np.testing.assert_allclose(X_new @ W_new, X @ W + b)


@pytest.mark.skip(reason="Enable once §4.2 mlp_forward is implemented")
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
