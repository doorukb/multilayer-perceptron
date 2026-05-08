"""Tests for src/mlp/loss.py (§5.1)."""
import numpy as np
import pytest
from mlp.loss import mse_loss, mse_loss_grad


@pytest.mark.skip(reason="Enable once mse_loss is implemented")
def test_mse_loss_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0])
    assert mse_loss(y, y) == 0.0


@pytest.mark.skip(reason="Enable once mse_loss is implemented")
def test_mse_loss_known_value():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 4.0])  # squared errors: 1, 0, 1 -> mean 2/3
    assert np.isclose(mse_loss(y, p), 2 / 3)


@pytest.mark.skip(reason="Enable once mse_loss_grad is implemented")
def test_mse_grad_matches_finite_difference():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    p = rng.normal(size=20)
    eps = 1e-6
    numerical = np.zeros_like(p)
    for i in range(p.size):
        ph, pl = p.copy(), p.copy()
        ph[i] += eps
        pl[i] -= eps
        numerical[i] = (mse_loss(y, ph) - mse_loss(y, pl)) / (2 * eps)
    analytical = mse_loss_grad(y, p)
    np.testing.assert_allclose(numerical, analytical, atol=1e-5)
