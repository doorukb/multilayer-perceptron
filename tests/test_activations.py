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

def test_sigmoid_forward_known_values():
    assert np.isclose(sigmoid_forward(np.array([0.0])), 0.5)
    assert sigmoid_forward(np.array([100.0]))[0] > 0.99
    assert sigmoid_forward(np.array([-100.0]))[0] < 0.01

def test_sigmoid_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    eps = 1e-6
    numerical = (sigmoid_forward(x + eps) - sigmoid_forward(x - eps)) / (2 * eps)
    analytical = sigmoid_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)

def test_tanh_forward_known_values():
    assert np.isclose(tanh_forward(np.array([0.0])), 0.0)
    assert tanh_forward(np.array([100.0]))[0] > 0.99
    assert tanh_forward(np.array([-100.0]))[0] < -0.99

def test_tanh_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    eps = 1e-6
    numerical = (tanh_forward(x + eps) - tanh_forward(x - eps)) / (2 * eps)
    analytical = tanh_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)

def test_relu_forward_known_values():
    assert relu_forward(np.array([-2.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 3.0]

def test_relu_backward_zero_convention():
    assert relu_backward(np.array([0.0]))[0] == 0.0
    assert relu_backward(np.array([-1.0]))[0] == 0.0
    assert relu_backward(np.array([1.0]))[0] == 1.0

def test_relu_backward_matches_finite_difference():
    # ReLU is non-differentiable at x=0; skip the kink in finite-difference checks.
    x = np.array([-3.0, -0.5, 0.5, 2.0, 4.0])
    eps = 1e-6
    numerical = (relu_forward(x + eps) - relu_forward(x - eps)) / (2 * eps)
    analytical = relu_backward(x)
    assert np.allclose(numerical, analytical, atol=1e-5)