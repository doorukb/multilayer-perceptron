from __future__ import annotations
import _path_setup
import matplotlib.pyplot as plt
import numpy as np
from mlp.data import create_train_and_test
from mlp.init import init_mlp
from mlp.tuning import grad_descent_with_validation, split_train_validation

# sweep L2 strength and plot the classic regularization curve
# train MSE rises with lambda; val MSE typically dips then rises
# reported metrics are pure MSE (penalty affects gradients only)

ARCH = [2, 10, 10, 1]
EPOCHS = 2000
LEARNING_RATE = 0.05
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
INIT_SEED = 42
SPLIT_SEED = 0
SHUFFLE_SEED = 0
OUTPUT_PATH = "06_regularization_sweep.png"

def _format_lambda(lmbda: float) -> str:
    if lmbda == 0.0:
        return "0"
    return f"{lmbda:.0e}"

def _plot_x(lmbda: float) -> float:
    return 1e-5 if lmbda == 0.0 else lmbda

def main() -> None:
    np.random.seed(0)
    train_data, _ = create_train_and_test()
    train_sub, val_sub = split_train_validation(train_data, val_fraction=0.2, seed=SPLIT_SEED)

    results: list[dict] = []

    print(f"Architecture {ARCH}  |  epochs={EPOCHS}  lr={LEARNING_RATE}")
    print(f"{'lambda':>10} {'final_train':>12} {'final_val':>12} {'best_val':>12} {'best_epoch':>10}")
    print("-" * 60)

    for lmbda in LAMBDAS:
        np.random.seed(INIT_SEED)
        model = init_mlp(ARCH)
        train_losses, val_losses, _ = grad_descent_with_validation(train_sub, val_sub, model, EPOCHS, LEARNING_RATE, seed=SHUFFLE_SEED, lmbda=lmbda)
        best_val_epoch = int(np.argmin(val_losses))
        row = {"lmbda": lmbda, "final_train": train_losses[-1], "final_val": val_losses[-1], "best_val": val_losses[best_val_epoch], "best_val_epoch": best_val_epoch}
        results.append(row)
        print(
            f"{_format_lambda(lmbda):>10} "
            f"{row['final_train']:>12.4f} "
            f"{row['final_val']:>12.4f} "
            f"{row['best_val']:>12.4f} "
            f"{row['best_val_epoch']:>10}"
        )

    best_row = min(results, key=lambda r: r["best_val"])
    print()
    print(f"Sweet spot (lowest best val MSE): lambda={_format_lambda(best_row['lmbda'])}  best_val={best_row['best_val']:.4f}")
    print()
    print(
        "Note: on this dataset the regularization effect is modest. With ~80 train points and a "
        "relatively small network, the model does not overfit aggressively - train and val MSE stay "
        "near the noise floor (~0.25). The U-shape in val loss may be shallow rather than dramatic; "
        "that is itself an important observation about when L2 matters."
    )

    plot_x = [_plot_x(r["lmbda"]) for r in results]
    final_train = [r["final_train"] for r in results]
    final_val = [r["final_val"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(plot_x, final_train, "o-", label="final train MSE")
    plt.plot(plot_x, final_val, "s-", label="final val MSE")
    plt.xscale("log")
    plt.xlabel("lambda (L2 strength; 0 plotted at 1e-5)")
    plt.ylabel("MSE")
    plt.title(f"L2 regularization sweep on {ARCH} - bias rows excluded from penalty; metrics are pure MSE")
    plt.xticks(plot_x, [_format_lambda(r["lmbda"]) for r in results])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"\nSaved plot to {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    main()