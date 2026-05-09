from __future__ import annotations
import numpy as np

def split_train_validation(
    train_data: np.ndarray, val_fraction: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    n_total = train_data.shape[0]
    n_val = int(round(n_total * val_fraction))
    indices = rng.permutation(n_total)
    return train_data[indices[n_val:]], train_data[indices[:n_val]]


def grad_descent_with_validation(
    train_data: np.ndarray,
    val_data: np.ndarray,
    my_mlp: dict[str, np.ndarray],
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], list[float], dict[str, np.ndarray]]:
    
    x_tr, y_tr = train_data[:, :2], train_data[:, 2:3]
    x_va, y_va = val_data[:, :2], val_data[:, 2:3]

    _, p_tr = mlp_forward(my_mlp, x_tr)
    _, p_va = mlp_forward(my_mlp, x_va)
    train_losses = [mse_loss(y_tr, p_tr)]
    val_losses = [mse_loss(y_va, p_va)]

    for _ in range(iterations):
        cache, p_tr = mlp_forward(my_mlp, x_tr)
        grads = backprop(my_mlp, cache, y_tr, p_tr)
        for key in my_mlp:
            my_mlp[key] -= learning_rate * grads[f"d{key}"]

        _, p_tr = mlp_forward(my_mlp, x_tr)
        _, p_va = mlp_forward(my_mlp, x_va)
        train_losses.append(mse_loss(y_tr, p_tr))
        val_losses.append(mse_loss(y_va, p_va))

    return train_losses, val_losses, my_mlp

def hyperparameter_search(train_data: np.ndarray, search_space: dict) -> dict:
    """
    §6.3 — Search over architectures and learning rates using only train+val
    data. Return the best configuration and the trained model.

    `search_space` shape is up to you — typical keys: 'output_sizes',
    'learning_rate', 'iterations'.
    """
    # TODO §6.3
    raise NotImplementedError
