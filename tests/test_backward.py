"""
Tests for src/mlp/backward.py (§5.2).

The numerical gradient check below is the single most valuable test in the
whole project. If it passes, your manual backprop derivation is almost
certainly correct.
"""
import numpy as np
import pytest


@pytest.mark.skip(reason="Enable once backprop, mlp_forward, and mse_loss are implemented")
def test_backprop_matches_numerical_gradient():
    from mlp.init import init_mlp
    from mlp.forward import mlp_forward
    from mlp.loss import mse_loss
    from mlp.backward import backprop

    rng = np.random.default_rng(42)
    n, in_dim = 8, 2
    x = rng.normal(size=(n, in_dim))
    y = rng.normal(size=(n, 1))

    model = init_mlp([in_dim, 4, 1])
    cache, pred = mlp_forward(model, x)
    grads = backprop(model, cache, y, pred)

    eps = 1e-5
    for key, W in model.items():
        dW = grads[f"d{key}"]
        assert dW.shape == W.shape, f"shape mismatch for {key}"

        # Spot-check a handful of entries (full check would be slow)
        idxs = [tuple(rng.integers(0, s) for s in W.shape) for _ in range(5)]
        for idx in idxs:
            W[idx] += eps
            _, p_hi = mlp_forward(model, x)
            loss_hi = mse_loss(y, p_hi)

            W[idx] -= 2 * eps
            _, p_lo = mlp_forward(model, x)
            loss_lo = mse_loss(y, p_lo)

            W[idx] += eps  # restore

            numerical = (loss_hi - loss_lo) / (2 * eps)
            analytical = dW[idx]
            assert np.isclose(numerical, analytical, atol=1e-4), (
                f"gradient mismatch at {key}{idx}: "
                f"numerical={numerical:.6f}, analytical={analytical:.6f}"
            )
