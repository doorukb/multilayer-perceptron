from __future__ import annotations
import numpy as np

def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x: np.ndarray) -> np.ndarray:
    s = sigmoid_forward(x)
    return s * (1 - s)
