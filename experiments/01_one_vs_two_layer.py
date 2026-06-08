from __future__ import annotations
import _path_setup
import numpy as np
import matplotlib.pyplot as plt
from mlp.data import create_train_and_test
from mlp.init import init_mlp
from mlp.optimizer import grad_descent

# trains a [2, 5, 1] and a [2, 5, 5, 1] MLP with the same learning rate and iteration count, then plots both training curves on shared axes and prints the final losses.

ARCH_1 = [2, 5, 1]
ARCH_2 = [2, 5, 5, 1]
EPOCHS = 2000
LEARNING_RATE = 0.05

def main() -> None:
    np.random.seed(0)
    train_data, _ = create_train_and_test()

    np.random.seed(42)
    model_1 = init_mlp(ARCH_1)
    losses_1, model_1 = grad_descent(train_data, model_1, EPOCHS, LEARNING_RATE)

    np.random.seed(42)
    model_2 = init_mlp(ARCH_2)
    losses_2, model_2 = grad_descent(train_data, model_2, EPOCHS, LEARNING_RATE)

    print(f"1-layer {ARCH_1} final train MSE: {losses_1[-1]:.4f}")
    print(f"2-layer {ARCH_2} final train MSE: {losses_2[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(losses_1, label=f"1 hidden layer {ARCH_1}", color="C0")
    plt.plot(losses_2, label=f"2 hidden layers {ARCH_2}", color="C1")
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("Depth comparison — same width, same learning rate")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()