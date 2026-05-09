from __future__ import annotations
import numpy as np

from mlp.init import init_mlp
from mlp.forward import mlp_forward
from mlp.backward import backprop
from mlp.loss import mse_loss

def split_train_validation(train_data: np.ndarray, val_fraction: float = 0.2, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
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

def hyperparameter_search(
    train_subset: np.ndarray,
    val_subset: np.ndarray,
    architectures: list[list[int]],
    learning_rates: list[float],
    iterations: int = 2000,
    init_seed: int = 42,
) -> list[dict]:
    results = []
    for arch in architectures:
        for lr in learning_rates:
            # Re-seed before each init so all configs start from comparable
            # random states — differences come from arch/lr, not luck of init.
            np.random.seed(init_seed)
            model = init_mlp(arch)

            train_losses, val_losses, model = grad_descent_with_validation(
                train_subset, val_subset, model,
                iterations=iterations, learning_rate=lr,
            )

            results.append({
                "arch": arch,
                "lr": lr,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "best_val_loss": min(val_losses),
                "best_val_iter": int(np.argmin(val_losses)),
                "model": model,
                "train_curve": train_losses,
                "val_curve": val_losses,
            })
    return results