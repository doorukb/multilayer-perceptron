from __future__ import annotations
import numpy as np

def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x: np.ndarray) -> np.ndarray:
    sigmoid = sigmoid_forward(x)
    return sigmoid * (1 - sigmoid)
