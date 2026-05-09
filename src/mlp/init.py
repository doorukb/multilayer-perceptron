from __future__ import annotations
import numpy as np

def init_weight_matrix(In: int = 2, Out: int = 1) -> np.ndarray:
    return np.random.normal(loc=1.0, scale=0.25, size=(In, Out))

def init_mlp(output_sizes: list[int] = [2, 5, 1]) -> dict[str, np.ndarray]:
    model: dict[str, np.ndarray] = {}
    n_layers = len(output_sizes) - 1
    for i in range(n_layers):
        in_dim = output_sizes[i] + 1   # +1 for the implicit bias row
        out_dim = output_sizes[i + 1]
        model[f"W{i}"] = init_weight_matrix(in_dim, out_dim)
    return model
