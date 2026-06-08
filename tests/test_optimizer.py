import numpy as np

# train the model and return the losses
def _train_copy(data, arch, epochs, learning_rate, batch_size=None, seed=None):
    from mlp.init import init_mlp
    from mlp.optimizer import grad_descent

    np.random.seed(0)
    model = init_mlp(arch)
    return grad_descent(
        data, model, epochs=epochs, learning_rate=learning_rate,
        batch_size=batch_size, seed=seed,
    )

# test that the loss decreases on a simple problem
def test_loss_decreases_on_simple_problem():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)
    losses, _ = _train_copy(data, [2, 5, 1], epochs=20, learning_rate=1e-3)

    assert len(losses) == 21
    assert losses[-1] < losses[0]

# test that the same seed is reproducible
def test_same_seed_is_reproducible():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)

    losses_a, _ = _train_copy(
        data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=4, seed=7,
    )
    losses_b, _ = _train_copy(
        data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=4, seed=7,
    )
    np.testing.assert_allclose(losses_a, losses_b)

# test that the different seeds differ
def test_different_seeds_differ():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)

    losses_a, _ = _train_copy(
        data, [2, 5, 1], epochs=10, learning_rate=1e-2, batch_size=1, seed=1,
    )
    losses_b, _ = _train_copy(
        data, [2, 5, 1], epochs=10, learning_rate=1e-2, batch_size=1, seed=2,
    )
    assert losses_a[-1] != losses_b[-1]

# test that the none batch size equals the full batch size
def test_none_batch_size_equals_full_batch():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)
    n = data.shape[0]

    losses_none, _ = _train_copy(data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=None, seed=3)
    losses_full, _ = _train_copy(data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=n, seed=3)
    np.testing.assert_allclose(losses_none, losses_full)

# test that the full batch shuffle is invariant
def test_full_batch_shuffle_is_invariant():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)
    n = data.shape[0]

    losses_a, _ = _train_copy(data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=n, seed=1)
    losses_b, _ = _train_copy(data, [2, 5, 1], epochs=10, learning_rate=1e-3, batch_size=n, seed=99)
    np.testing.assert_allclose(losses_a, losses_b)

# test that the SGD batch size one
def test_sgd_batch_size_one():
    from mlp.data import sample_points

    np.random.seed(0)
    data = sample_points(100)
    losses, _ = _train_copy(data, [2, 5, 1], epochs=30, learning_rate=1e-3, batch_size=1, seed=0)

    assert len(losses) == 31
    assert losses[-1] < losses[0]