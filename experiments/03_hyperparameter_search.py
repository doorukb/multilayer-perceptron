from __future__ import annotations
import _path_setup
import numpy as np
from mlp.data import create_train_and_test
from mlp.forward import mlp_forward
from mlp.loss import mse_loss
from mlp.tuning import split_train_validation, hyperparameter_search

# The test set must be touched only once, after the final architecture and learning rate have been chosen using train + validation only

ARCHITECTURES = [
    [2, 5, 1],
    [2, 10, 1],
    [2, 5, 5, 1],
    [2, 10, 10, 1],
]
LEARNING_RATES = [0.01, 0.05, 0.1]
EPOCHS = 2000

def main() -> None:
    np.random.seed(0)
    train_data, test_data = create_train_and_test()
    train_sub, val_sub = split_train_validation(train_data, val_fraction=0.2, seed=0)

    print(f"train_sub: {train_sub.shape}    (used for gradient updates)")
    print(f"val_sub:   {val_sub.shape}    (used for model selection)")
    print(f"test:      {test_data.shape}    (untouched until final reveal)")
    print()

    print(f"Searching {len(ARCHITECTURES)} architectures x "
          f"{len(LEARNING_RATES)} learning rates = "
          f"{len(ARCHITECTURES) * len(LEARNING_RATES)} configurations...")
    results = hyperparameter_search(
        train_sub, val_sub,
        architectures=ARCHITECTURES,
        learning_rates=LEARNING_RATES,
        epochs=EPOCHS,
    )

    header = f"{'arch':<22} {'lr':<6} {'train':<10} {'val':<10} {'best_val':<10}"
    print()
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["final_val_loss"]):
        print(
            f"{str(r['arch']):<22} {r['lr']:<6.2f} "
            f"{r['final_train_loss']:<10.4f} {r['final_val_loss']:<10.4f} "
            f"{r['best_val_loss']:<10.4f}"
        )

    best = min(results, key=lambda r: r["final_val_loss"])
    print()
    print("=" * 60)
    print("Selected configuration (val-only):")
    print(f"  arch = {best['arch']}")
    print(f"  lr   = {best['lr']}")
    print(f"  train MSE = {best['final_train_loss']:.4f}")
    print(f"  val   MSE = {best['final_val_loss']:.4f}")
    print("=" * 60)

    # the single, final set evaluation
    x_test = test_data[:, :2]
    y_test = test_data[:, 2:3]
    _, pred_test = mlp_forward(best["model"], x_test)
    test_loss = mse_loss(y_test, pred_test)

    print()
    print(f"  test  MSE = {test_loss:.4f}    <-- the headline number")
    print()

if __name__ == "__main__":
    main()