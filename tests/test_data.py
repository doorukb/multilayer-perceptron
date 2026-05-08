"""Tests for src/mlp/data.py (§2)."""
import numpy as np
import pytest
from mlp.data import sample_points, create_train_and_test


def test_sample_points_shape():
    pts = sample_points(50)
    assert pts.shape == (50, 3)


@pytest.mark.skip(reason="Enable once §2.1 is implemented")
def test_sample_points_z_relationship():
    """Z should be approximately X^2 - Y^2 + 1.2 (within noise)."""
    rng = np.random.default_rng(0)
    pts = sample_points(5000)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    residual = z - (x**2 - y**2 + 1.2)
    # noise ~ N(0, 0.5), so residual mean ~0, std ~0.5
    assert abs(residual.mean()) < 0.1
    assert 0.3 < residual.std() < 0.7


@pytest.mark.skip(reason="Enable once §2.3 is implemented")
def test_train_test_sizes():
    train, test = create_train_and_test(100, 20)
    assert train.shape == (100, 3)
    assert test.shape == (20, 3)
