from __future__ import annotations
import _path_setup 
import numpy as np
import matplotlib.pyplot as plt
from mlp.data import create_train_and_test
from mlp.init import init_mlp
from mlp.tuning import split_train_validation, grad_descent_with_validation

# adding a validation split and plotting validation curves

ARCH_1 = [2, 5, 1]
ARCH_2 = [2, 5, 5, 1]
EPOCHS = 2000
LEARNING_RATE = 0.05

def main() -> None:
    np.random.seed(0)
    train_data, _ = create_train_and_test()
    train_sub, val_sub = split_train_validation(train_data, val_fraction=0.2, seed=0)

    np.random.seed(42)
    model_1 = init_mlp(ARCH_1)
    train_1, val_1, _ = grad_descent_with_validation(train_sub, val_sub, model_1, EPOCHS, LEARNING_RATE)

    np.random.seed(42)
    model_2 = init_mlp(ARCH_2)
    train_2, val_2, _ = grad_descent_with_validation(train_sub, val_sub, model_2, EPOCHS, LEARNING_RATE)

    print(f"1 layer  final  train: {train_1[-1]:.4f}   val: {val_1[-1]:.4f}")
    print(f"2 layer  final  train: {train_2[-1]:.4f}   val: {val_2[-1]:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(train_1, color="C0", label=f"{ARCH_1} — train")
    plt.plot(val_1, color="C0", linestyle="--", label=f"{ARCH_1} — val")
    plt.plot(train_2, color="C1", label=f"{ARCH_2} — train")
    plt.plot(val_2, color="C1", linestyle="--", label=f"{ARCH_2} — val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Train (solid) vs validation (dashed) loss")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()