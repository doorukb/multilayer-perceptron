# Gradient Descent MLP

- Python 3.10+
- NumPy
- Matplotlib
- pytest

A multilayer perceptron trained with gradient descent, implemented entirely in NumPy with no machine learning frameworks. The project covers forward pass, backpropagation, MSE loss, L2 weight regularization (bias rows excluded from the penalty), and a gradient descent optimizer, all written from scratch. Hidden activations (sigmoid, tanh, ReLU) are swappable via the strategy pattern; backprop uses cached pre-activation values Z with a swappable activation_backward, not post-activation A. Analytical gradients are validated against numerical finite-difference checks. A hyperparameter search is included that follows strict train/validation/test discipline. The codebase is structured as a modular Python package with a full pytest suite.

## What it does 
The network learns to approximate the surface Z = X^2 - Y^2 + 1.2 + noise, where X and Y are drawn uniformly from [-1, 1] and noise is Gaussian with mean 0 and standard deviation 0.5. Inputs are 2D (X, Y) and the target is a scalar Z. Because the irreducible noise has standard deviation 0.5, the theoretical minimum MSE is around 0.25, which is treated as the practical lower bound.



## RESULTS 
Depth comparison (experiment 01, 2000 iterations, lr=0.05, architecture width=5):

    1-layer  [2, 5, 1]    initial MSE: 4.9805   final MSE: 0.3862
    2-layer  [2, 5, 5, 1]  initial MSE: 4.3793  final MSE: 0.3926

<p align="center">
<img width="420" height="310" alt="depth_comparison" src="https://github.com/user-attachments/assets/951b7bec-49da-496b-b90a-556ee83de55c" /> <img width="420" height="310" alt="learning_curve" src="https://github.com/user-attachments/assets/96b43f13-b540-4b37-b352-a5b84d4f8ac1" />
</p>
<p align="center">
<img width="900" height="500" alt="05_batch_size_comparison" src="https://github.com/user-attachments/assets/266ff90b-974d-495d-a89f-aeed0d6088c5" /> <img width="900" height="500" alt="06_regularization_sweep" src="https://github.com/user-attachments/assets/c4910397-feeb-42b1-b80e-21c99613e720" />
</p>

Both models start at similar initial MSE under Xavier initialization (small weights, sigmoid outputs near 0.5 rather than saturation) and converge to roughly the same final training loss near the irreducible noise floor (~0.25-0.40). The additional depth does not yield a measurable improvement on this dataset at this scale.

<p align="center">
<img width="420" height="310" alt="heat_map" src="https://github.com/user-attachments/assets/27131084-2c87-4adb-b7af-4d1693b251ae" /> <img width="420" height="310" alt="trained_predictions_vs_true_data" src="https://github.com/user-attachments/assets/4d8058c8-ff55-4c93-8266-8e1fe3858f1a" />
</p>

Activation comparison (experiment 04, 2000 epochs, lr=0.05, architecture [2, 10, 10, 10, 1]):

This is the payoff experiment for the activations phase. All three networks share identical Xavier initialization, learning rate, and data; only the hidden activation changes.

    activation    initial MSE    final MSE
    sigmoid          2.2047        0.4041
    tanh             1.9108        0.1898
    ReLU             1.8162        0.1697

At initialization the mean sigmoid derivative sigma'(z) = sigma(1-sigma) is already below 0.25 on every hidden layer (~0.23-0.24). Backprop multiplies by this factor at each layer, so gradient signal shrinks exponentially with depth. ReLU keeps a unit derivative on active units and reaches the noise floor (~0.25 MSE). tanh is symmetric around zero and lands between the two. sigmoid converges, but much more slowly and to a higher plateau — the curve looks nearly flat for long stretches while ReLU descends steadily.

Run locally: `cd experiments && python 04_activation_comparison.py`


Batch size comparison (experiment 05, 2000 epochs, lr=0.05, architecture [2, 10, 10, 1]):

Same Xavier initialization, learning rate, and data for each run; only batch size changes. Metrics are full-dataset train MSE logged at each epoch boundary.

    batch_size    initial MSE    final MSE
    1 (SGD)          1.7279        0.1735
    8                1.7279        0.1885
    32               1.7279        0.2485
    full             1.7279        0.3865

SGD (batch=1) is noisiest but reaches the lowest final loss — many small updates per epoch explore the loss surface aggressively. Full-batch GD is smooth and deterministic but makes only one gradient step per epoch, so it converges more slowly on this budget. Mini-batch sizes land between the two extremes.

Run locally: `cd experiments && python 05_batch_size_comparison.py`


Hyperparameter search (experiment 03, 4 architectures x 3 learning rates, 2000 iterations):

Data split: 80 points for training, 20 for validation, 20 for test. The test set was not accessed until after the best configuration was selected.

    arch                   lr     train      val        best_val
    [2, 5, 1]              0.10   0.3478     0.4087     0.4087
    [2, 10, 10, 1]         0.10   0.3682     0.4286     0.4286
    [2, 10, 1]             0.10   0.2689     0.4294     0.4294
    [2, 5, 1]              0.05   0.3773     0.4442     0.4442
    [2, 5, 5, 1]           0.10   0.3762     0.4461     0.4461
    [2, 10, 10, 1]         0.05   0.3815     0.4470     0.4470
    [2, 10, 1]             0.01   0.3836     0.4570     0.4570
    [2, 5, 5, 1]           0.05   0.3817     0.4571     0.4571
    [2, 10, 1]             0.05   0.3763     0.4584     0.4556
    [2, 10, 10, 1]         0.01   0.3874     0.4616     0.4616
    [2, 5, 1]              0.01   0.3864     0.4723     0.4723
    [2, 5, 5, 1]           0.01   0.3880     0.4732     0.4732

Selected configuration (lowest validation MSE):
    architecture:  [2, 5, 1]
    learning rate: 0.1
    train MSE:     0.3478
    val MSE:       0.4087
    test MSE:      0.3135

The test MSE of 0.3135 is close to the theoretical noise floor, indicating the model learned the underlying surface well and did not overfit.


Regularization sweep (experiment 06, 2000 epochs, lr=0.05, architecture [2, 10, 10, 1]):

Same initialization and train/val split for each lambda; reported metrics are pure MSE (L2 affects gradients only; bias rows excluded from the penalty).

    lambda     final_train   final_val     best_val
    0              0.3815      0.4470       0.4470
    1e-4           0.3816      0.4474       0.4474
    1e-3           0.3825      0.4511       0.4511
    1e-2           0.3905      0.4749       0.4677
    1e-1           0.3918      0.4784       0.4713

Sweet spot (lowest best val MSE): lambda=0 at 0.4470. As lambda increases, train MSE rises monotonically while val MSE worsens — there is no pronounced U-shape here. With ~80 train points and a modest network, the model does not overfit aggressively on this noisy regression surface, so L2 has little room to help. That shallow effect is itself the takeaway: regularization matters most when capacity exceeds what the data can support.

Run locally: `cd experiments && python 06_regularization_sweep.py`

## Installation

- Python 3.10 or later is required.

- Clone the repository and install the dependencies:
    git clone <repo-url>
    cd gradient-descent-mlp
    python -m venv .venv
    source .venv/Scripts/activate   # Windows (Git Bash / PowerShell: .venv\Scripts\activate)
    pip install -r requirements.txt

- Dependencies (requirements.txt):
    numpy>=1.24
    matplotlib>=3.7
    pytest>=7.4

- Training a model from scratch:
    import numpy as np
    from mlp.data import create_train_and_test
    from mlp.init import init_mlp
    from mlp.optimizer import grad_descent
    from mlp.loss import mse_loss
    from mlp.forward import mlp_forward

    np.random.seed(0)
    train_data, test_data = create_train_and_test(train_size=100, test_size=20)

    model = init_mlp([2, 10, 1])   # input dim=2, 10 hidden units, output dim=1
    losses, model = grad_descent(train_data, model, epochs=2000, learning_rate=0.1, lmbda=1e-3)

    x_test = test_data[:, :2]
    y_test = test_data[:, 2:3]
    _, pred = mlp_forward(model, x_test)
    print(f"test MSE: {mse_loss(y_test, pred):.4f}")

- Training with a validation split:

    from mlp.tuning import split_train_validation, grad_descent_with_validation

    train_sub, val_sub = split_train_validation(train_data, val_fraction=0.2, seed=0)
    train_losses, val_losses, model = grad_descent_with_validation(
        train_sub, val_sub, model, epochs=2000, learning_rate=0.1, lmbda=1e-3
    )

- Comparing activations on a deep network (sigmoid vs tanh vs ReLU):
    cd experiments
    python 04_activation_comparison.py

- Comparing batch sizes (SGD vs mini-batch vs full-batch):
    cd experiments
    python 05_batch_size_comparison.py

- Running the hyperparameter search:
    cd experiments
    python 03_hyperparameter_search.py


## Testing

To run all the tests from the project root, run :
    pytest tests/ -v
