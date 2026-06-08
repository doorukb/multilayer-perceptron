from __future__ import annotations
import _path_setup  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from mlp.activations import (
    relu_backward,
    relu_forward,
    sigmoid_backward,
    sigmoid_forward,
    tanh_backward,
    tanh_forward,
)
from mlp.data import create_train_and_test
from mlp.forward import mlp_forward
from mlp.init import init_mlp
from mlp.optimizer import grad_descent

# compare sigmoid, tanh, and ReLU on a deep network
# same init, lr, and epochs
# saturation / vanishing gradients show up in the learning curves

ARCH = [2, 10, 10, 10, 1]
EPOCHS = 2000
LEARNING_RATE = 0.05
INIT_SEED = 42
SHUFFLE_SEED = 0

ACTIVATIONS = [
    ("sigmoid", sigmoid_forward, sigmoid_backward),
    ("tanh", tanh_forward, tanh_backward),
    ("ReLU", relu_forward, relu_backward),
]

# print the mean sigmoid derivative sigma'(z) per hidden layer at initialization
def _print_sigmoid_gradient_diagnostics(train_data: np.ndarray, model: dict[str, np.ndarray]) -> None:
    x = train_data[:, :2]
    cache, _ = mlp_forward(model, x, activation=sigmoid_forward)
    print("  sigmoid sigma'(z) mean per hidden layer at init (vanishing signal when small):")
    for layer in (1, 2, 3):
        Z = cache[f"Z{layer}"]
        sigmoid_grad = sigmoid_backward(Z)
        print(f"    Z{layer}: {sigmoid_grad.mean():.4f}")

def main() -> None:
    np.random.seed(0)
    train_data, _ = create_train_and_test()

    results: list[tuple[str, list[float]]] = []

    print(f"Architecture {ARCH}  |  epochs={EPOCHS}  lr={LEARNING_RATE}")
    print(f"{'activation':<10} {'initial':>10} {'final':>10}")
    print("-" * 32)

    for name, act_fwd, act_bwd in ACTIVATIONS:
        np.random.seed(INIT_SEED)
        model = init_mlp(ARCH)
        if name == "sigmoid":
            _print_sigmoid_gradient_diagnostics(train_data, model)

        losses, _ = grad_descent(train_data, model, EPOCHS, LEARNING_RATE, activation=act_fwd, activation_backward=act_bwd, seed=SHUFFLE_SEED)
        results.append((name, losses))
        print(f"{name:<10} {losses[0]:>10.4f} {losses[-1]:>10.4f}")

    plt.figure(figsize=(9, 5))
    for name, losses in results:
        plt.plot(losses, label=name)
    plt.xlabel("epoch")
    plt.ylabel("train MSE")
    plt.title("Activation comparison on [2, 10, 10, 10, 1] deep network- sigmoid stalls as vanishing gradients compound; ReLU descends")
    plt.axhline(0.25, color="gray", linestyle=":", linewidth=1, label="noise floor ~0.25")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()