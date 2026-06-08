from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from mlp.forward import mlp_forward

# plot the 3d scatter of the dataset
def plot_3d_scatter(points: np.ndarray) -> None:
    # 3d scatter of the dataset
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c=z, cmap="coolwarm", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Synthetic dataset")
    plt.show()

# plot the train and test data
def plot_train_and_test(train_data: np.ndarray, test_data: np.ndarray) -> None:
    # train (circles) vs test (triangles) 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        train_data[:, 0], train_data[:, 1], train_data[:, 2],
        c=train_data[:, 2], cmap="coolwarm",
        alpha=0.5, marker="o", label="train",
    )
    ax.scatter(
        test_data[:, 0], test_data[:, 1], test_data[:, 2],
        c=test_data[:, 2], cmap="coolwarm",
        alpha=0.9, marker="^", label="test",
        edgecolors="black", linewidths=0.5,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("train (circles) vs test (triangles)")
    ax.legend()
    plt.show()

# plot the data and the prediction surface
def plot_data_and_pred(my_mlp, train_points: np.ndarray, test_points: np.ndarray) -> None:
    # rained model's prediction surface over [-1, 1]^2 
    grid_size = 30
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    grid_inputs = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    _, Z_pred = mlp_forward(my_mlp, grid_inputs)
    Z_grid = Z_pred.reshape(grid_size, grid_size)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X_grid, Y_grid, Z_grid,
        cmap="coolwarm", alpha=0.6, edgecolor="none",
    )
    ax.scatter(
        train_points[:, 0], train_points[:, 1], train_points[:, 2],
        c=train_points[:, 2], cmap="coolwarm",
        alpha=0.5, marker="o", label="train",
    )
    ax.scatter(
        test_points[:, 0], test_points[:, 1], test_points[:, 2],
        c=test_points[:, 2], cmap="coolwarm",
        alpha=1.0, marker="^", label="test",
        edgecolors="black", linewidths=0.5,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("trained MLP : prediction surface vs. data")
    ax.legend()
    plt.show()

# plot the learning curves
def plot_learning_curves(curves: dict[str, dict]) -> None:
    plt.figure(figsize=(9, 5))
    for i, (label, pair) in enumerate(curves.items()):
        color = f"C{i}"
        plt.plot(pair["train"], color=color, label=f"{label} — train")
        if "val" in pair:
            plt.plot(pair["val"], color=color, linestyle="--", label=f"{label} — val")
    plt.xlabel("iteration")
    plt.ylabel("MSE")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(True)
    plt.show()