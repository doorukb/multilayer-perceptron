from __future__ import annotations
import numpy as np

def init_weight_matrix(fan_in: int = 2, out_dim: int = 1) -> np.ndarray:
    # Xavier: std = sqrt(1 / fan_in) keeps pre-activation variance ~1 across layers.
    std = np.sqrt(1.0 / fan_in)
    W = np.zeros((fan_in + 1, out_dim))
    W[:-1, :] = np.random.normal(loc=0.0, scale=std, size=(fan_in, out_dim))
    return W

def init_mlp(output_sizes: list[int] = [2, 5, 1]) -> dict[str, np.ndarray]:
    model: dict[str, np.ndarray] = {}
    n_layers = len(output_sizes) - 1
    for i in range(n_layers):
        fan_in = output_sizes[i]
        out_dim = output_sizes[i + 1]
        model[f"W{i}"] = init_weight_matrix(fan_in, out_dim)
    return model
