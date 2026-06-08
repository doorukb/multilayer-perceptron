import numpy as np
import pytest

# test that the train and validation subsets have the correct shapes and no overlap
def test_split_train_validation_shapes_and_no_overlap():
    from mlp.tuning import split_train_validation

    rng = np.random.default_rng(0)
    train = rng.normal(size=(100, 3))
    train_sub, val_sub = split_train_validation(train, val_fraction=0.2, seed=7)

    assert train_sub.shape == (80, 3)
    assert val_sub.shape == (20, 3)
    # The two subsets must partition the original (every original row appears
    # in exactly one of them).
    combined = np.vstack([train_sub, val_sub])
    assert combined.shape == train.shape
    # Round-trip via sorted views: same rows must be present.
    a = np.sort(train.view([("", train.dtype)] * 3), axis=0)
    b = np.sort(combined.view([("", combined.dtype)] * 3), axis=0)
    assert np.array_equal(a, b)

# test that the split is reproducible with the same seed
def test_split_is_reproducible_with_same_seed():
    from mlp.tuning import split_train_validation

    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 3))
    s1 = split_train_validation(train, val_fraction=0.25, seed=42)
    s2 = split_train_validation(train, val_fraction=0.25, seed=42)
    np.testing.assert_array_equal(s1[0], s2[0])
    np.testing.assert_array_equal(s1[1], s2[1])


def test_grad_descent_with_validation_loss_lengths():
    from mlp.data import sample_points
    from mlp.init import init_mlp
    from mlp.tuning import split_train_validation, grad_descent_with_validation

    np.random.seed(0)
    train = sample_points(60)
    train_sub, val_sub = split_train_validation(train, val_fraction=0.25, seed=0)
    model = init_mlp([2, 5, 1])

    epochs = 10
    train_losses, val_losses, _ = grad_descent_with_validation(
        train_sub, val_sub, model, epochs=epochs, learning_rate=0.05
    )
    assert len(train_losses) == epochs + 1
    assert len(val_losses) == epochs + 1
    # Validation loss never blows up (basic sanity).
    assert all(np.isfinite(v) for v in val_losses)

# test that the hyperparameter search returns the correct results
def test_hyperparameter_search_smoke():
    from mlp.data import sample_points
    from mlp.tuning import split_train_validation, hyperparameter_search

    np.random.seed(0)
    train = sample_points(60)
    train_sub, val_sub = split_train_validation(train, val_fraction=0.25, seed=0)

    architectures = [[2, 4, 1], [2, 4, 4, 1]]
    learning_rates = [0.05, 0.1]
    results = hyperparameter_search(
        train_sub, val_sub,
        architectures=architectures,
        learning_rates=learning_rates,
        epochs=20,
    )

    assert len(results) == len(architectures) * len(learning_rates)
    for r in results:
        assert {"arch", "lr", "final_train_loss", "final_val_loss",
                "best_val_loss", "best_val_iter", "model",
                "train_curve", "val_curve"} <= r.keys()
        assert len(r["train_curve"]) == 21
        assert len(r["val_curve"]) == 21
        # Best val should be ≤ final val by definition.
        assert r["best_val_loss"] <= r["final_val_loss"] + 1e-12