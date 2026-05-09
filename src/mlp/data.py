from __future__ import annotations
import numpy as np

def sample_points(n: int) -> np.ndarray:
    x = np.random.uniform(-1, 1, size=n)
    y = np.random.uniform(-1, 1, size=n)
    err = np.random.normal(loc=0.0, scale=0.5, size=n)
    z = x ** 2 - y ** 2 + 1.2 + err
    return np.column_stack([x, y, z])


def create_train_and_test(train_size: int = 100, test_size: int = 20) -> tuple[np.ndarray, np.ndarray]:
    return sample_points(train_size), sample_points(test_size)