"""Tests for src/mlp/activations.py (§4.1, §1.1)."""
import numpy as np
import pytest
from mlp.activations import sigmoid_forward, sigmoid_backward


@pytest.mark.skip(reason="Enable once sigmoid_forward is implemented")
def test_sigmoid_forward_known_values():
    assert np.isclose(sigmoid_forward(np.array([0.0])), 0.5)
    assert sigmoid_forward(np.array([100.0]))[0] > 0.99
    assert sigmoid_forward(np.array([-100.0]))[0] < 0.01


@pytest.mark.skip(reason="Enable once sigmoid_backward is implemented")
def test_sigmoid_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    eps = 1e-6
    numerical = (sigmoid_forward(x + eps) - sigmoid_forward(x - eps)) / (2 * eps)
    analytical = sigmoid_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)
