"""Tests for src/mlp/init.py (§3)."""
import numpy as np
import pytest
from mlp.init import init_weight_matrix, init_mlp


@pytest.mark.skip(reason="Enable once §3.1 is implemented")
def test_init_weight_matrix_shape_and_stats():
    W = init_weight_matrix(5, 10)
    assert W.shape == (5, 10)
    # Sample is small, so use loose bounds
    assert 0.7 < W.mean() < 1.3
    assert 0.1 < W.std() < 0.4


@pytest.mark.skip(reason="Enable once §3.2 is implemented")
def test_init_mlp_layer_shapes():
    """[2, 5, 10, 1] -> shapes (2, 5), (5, 10), (10, 1).

    If you fold bias inside init_mlp, adjust the expected input dims.
    """
    model = init_mlp([2, 5, 10, 1])
    expected = [(2, 5), (5, 10), (10, 1)]
    for i, exp_shape in enumerate(expected):
        W = model[f"W{i}"]
        # Allow +1 on the input dim for implicit bias
        assert W.shape == exp_shape or W.shape == (exp_shape[0] + 1, exp_shape[1])
