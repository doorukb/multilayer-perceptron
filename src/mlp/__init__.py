"""
mlp — From-scratch Multi-Layer Perceptron in NumPy.

Modules:
    data         §2  synthetic dataset + train/test split
    plotting     §2 & §5  3D scatter, surface, learning curves
    init         §3  weight + MLP initialization
    activations  §4.1  sigmoid forward & backward
    forward      §1.4 + §4.2  implicit-bias trick + full forward pass
    loss         §5.1  MSE loss + its gradient
    backward     §5.2  backprop using the forward cache
    optimizer    §5.3  gradient descent training loop
    tuning       §6  validation split + hyperparameter search
"""
