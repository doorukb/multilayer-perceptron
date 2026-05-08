from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mlp.forward import mlp_forward

def plot_3d_scatter(points: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c=z, cmap='coolwarm', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('synthetic dataset')
    plt.show()

def plot_train_and_test(train_data: np.ndarray, test_data: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c=train_data[:, 2], cmap='coolwarm', alpha=0.5, marker='o', label='train')
    ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=test_data[:, 2], cmap='coolwarm', alpha=0.9, marker='^', label='test', edgecolors='black', linewidths=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('training (circles) vs testing (triangles)')
    ax.legend()
    plt.show()

def plot_data_and_pred(my_mlp, train_points: np.ndarray, test_points: np.ndarray) -> None:
    grid_size = 30 # input grid
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    grid_inputs = np.column_stack([X_grid.ravel(), Y_grid.ravel()]) # flatten the grid
    _, Z_pred = mlp_forward(my_mlp, grid_inputs)
    Z_grid = Z_pred.reshape(grid_size, grid_size) # reshape predictions
    fig = plt.figure(figsize=(9, 7)) # set up the figure
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm',
                    alpha=0.6, edgecolor='none')
    ax.scatter(train_points[:, 0], train_points[:, 1], train_points[:, 2], c=train_points[:, 2], cmap='coolwarm', alpha=0.5, marker='o', label='train') # training points : circles
    ax.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c=test_points[:, 2], cmap='coolwarm', alpha=1.0, marker='^', edgecolors='black', linewidths=0.5, label='test') # test points : triangles

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('mlp trained : predictions vs true data')
    ax.legend()
    plt.show()


def plot_learning_curves(curves: dict[str, list[float]]) -> None:
    """
    §6 — Plot one or more learning curves on the same axes.

    Convention: pairs of curves for the same model share a color, with the
    validation curve drawn dashed.
    """
    # TODO §6.2
    plt.show()
