# Gradient Descent MLP

- Python 3.10+
- NumPy
- Matplotlib
- pytest

A multilayer perceptron trained with gradient descent, implemented entirely in NumPy with no machine learning frameworks. The project covers forward pass, backpropagation, MSE loss, L2 weight regularization (bias rows excluded from the penalty), and a gradient descent optimizer, all written from scratch. Hidden activations (sigmoid, tanh, ReLU) are swappable via the strategy pattern; backprop uses cached pre-activation values Z with a swappable activation_backward, not post-activation A. Analytical gradients are validated against numerical finite-difference checks. A hyperparameter search is included that follows strict train/validation/test discipline. The codebase is structured as a modular Python package with a full pytest suite.


THE PROBLEM
The network learns to approximate the surface Z = X^2 - Y^2 + 1.2 + noise, where X and Y are drawn uniformly from [-1, 1] and noise is Gaussian with mean 0 and standard deviation 0.5. Inputs are 2D (X, Y) and the target is a scalar Z. Because the irreducible noise has standard deviation 0.5, the theoretical minimum MSE is around 0.25, which is treated as the practical lower bound.


RESULTS 
Depth comparison (experiment 01, 2000 iterations, lr=0.05, architecture width=5):

    1-layer  [2, 5, 1]    initial MSE: 9.6689   final MSE: 0.3838
    2-layer  [2, 5, 5, 1]  initial MSE: 19.7019  final MSE: 0.4085

<img width="320" height="230" alt="depth_comparison" src="https://github.com/user-attachments/assets/951b7bec-49da-496b-b90a-556ee83de55c" /> <img width="336" height="230" alt="learning_curve" src="https://github.com/user-attachments/assets/96b43f13-b540-4b37-b352-a5b84d4f8ac1" />



Both models converge to roughly the same final training loss near the irreducible noise floor (~0.25-0.40). The additional depth does not yield a measurable improvement on this dataset at this scale, and increases the initial loss because there are more weights to initialise.

<img width="424" height="300" alt="heat_map" src="https://github.com/user-attachments/assets/27131084-2c87-4adb-b7af-4d1693b251ae" /> <img width="361" height="300" alt="trained_predictions_vs_true_data" src="https://github.com/user-attachments/assets/4d8058c8-ff55-4c93-8266-8e1fe3858f1a" />


Activation comparison (experiment 04, 2000 epochs, lr=0.05, architecture [2, 10, 10, 10, 1]):

This is the payoff experiment for the activations phase. All three networks share identical Xavier initialization, learning rate, and data; only the hidden activation changes.

    activation    initial MSE    final MSE
    sigmoid          2.2047        0.4041
    tanh             1.9108        0.1898
    ReLU             1.8162        0.1697

At initialization the mean sigmoid derivative sigma'(z) = sigma(1-sigma) is already below 0.25 on every hidden layer (~0.23-0.24). Backprop multiplies by this factor at each layer, so gradient signal shrinks exponentially with depth. ReLU keeps a unit derivative on active units and reaches the noise floor (~0.25 MSE). tanh is symmetric around zero and lands between the two. sigmoid converges, but much more slowly and to a higher plateau — the curve looks nearly flat for long stretches while ReLU descends steadily.

Run locally: `cd experiments && python 04_activation_comparison.py`


Hyperparameter search (experiment 03, 4 architectures x 3 learning rates, 2000 iterations):

Data split: 80 points for training, 20 for validation, 20 for test. The test set was not accessed until after the best configuration was selected.

    arch                   lr     train      val        best_val
    [2, 10, 1]             0.10   0.2432     0.3391     0.3391
    [2, 5, 1]              0.10   0.3014     0.4128     0.4128
    [2, 10, 10, 1]         0.10   0.3529     0.4512     0.4512
    [2, 5, 1]              0.01   0.3814     0.4571     0.4571
    [2, 10, 1]             0.01   0.3756     0.4646     0.4646
    [2, 5, 5, 1]           0.10   0.3873     0.4695     0.4695
    [2, 5, 1]              0.05   0.3634     0.4698     0.4519
    [2, 10, 1]             0.05   0.3603     0.4715     0.4639
    [2, 10, 10, 1]         0.05   0.3921     0.4786     0.4745
    [2, 10, 10, 1]         0.01   0.3921     0.4787     0.4740
    [2, 5, 5, 1]           0.05   0.3919     0.4788     0.4788
    [2, 5, 5, 1]           0.01   0.3944     0.4831     0.4810

Selected configuration (lowest validation MSE):
    architecture:  [2, 10, 1]
    learning rate: 0.1
    train MSE:     0.2432
    val MSE:       0.3391
    test MSE:      0.2536

The test MSE of 0.2536 is close to the theoretical noise floor, indicating the model learned the underlying surface well and did not overfit.


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

INSTALLATION

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

- Running the hyperparameter search:
    cd experiments
    python 03_hyperparameter_search.py


TESTING

To run all the tests from the project root, run :
    pytest tests/ -v
Feel free to add more tests you'd like.

After any change to init, forward, or backward, re-run the gradient-check tests:
    pytest tests/test_backward.py tests/test_loss.py tests/test_activations.py -v
The numerical gradient check is a regression test, not a one-time validation.

test_activations
    Checks sigmoid_forward at x=0 (must return 0.5), at large positive/negative inputs, and verifies sigmoid_backward matches the numerical finite-difference derivative to within 1e-5. tanh_forward is checked at x=0 (must return 0.0) and at saturation; tanh_backward is verified the same way. relu_forward zeros negatives; relu_backward uses grad=1 if x>0 else 0 at the x=0 kink; finite-difference check excludes x=0.

test_backward
    Verifies backprop against a numerical gradient check (regression test — re-run after init/forward/backward changes). Parametrized over sigmoid, tanh, and ReLU hidden activations and lmbda in {0, 0.1}. Hidden-layer derivatives use cached Z plus activation_backward, not post-activation A. For each entry in every weight matrix of a [2, 4, 1] Xavier-initialized network, epsilon=1e-5 central-difference estimates on total loss MSE + l2_penalty are compared to analytical gradients. Tolerance is 1e-4.

test_loss
    MSE loss returns 0 when prediction equals label, returns the correct value on a known example (2/3 for unit-step errors), and the gradient matches the finite-difference gradient to within 1e-5.

test_forward
    modify_x_w is checked on a vector and a matrix input to confirm that appending a bias column and stacking b as an extra row produces the same result as X @ W + b. mlp_forward uses modify_x_w internally and accepts a swappable hidden activation (default sigmoid); the output layer stays linear for regression. ReLU and sigmoid produce different hidden activations on the same weights and inputs.

test_init
    init_weight_matrix uses Xavier scaling (std = sqrt(1 / fan_in), zero-mean weights, zero bias row). init_mlp produces weight matrices whose shapes are consistent with the requested layer sizes (including the +1 bias row). Untrained hidden activations stay away from sigmoid saturation.

test_data
    sample_points returns shape (n, 3), the residual Z - (X^2 - Y^2 + 1.2) has mean near 0 and std near 0.5 on a large sample, and create_train_and_test returns arrays of the requested sizes.

test_optimizer
    grad_descent trains for epochs with shuffled mini-batches and accepts swappable activation pairs. Optional lmbda passes through to backprop; logged loss stays pure MSE. Full-batch GD and mini-batch GD share one loop: batch_size=None is equivalent to batch_size=len(data); batch_size=1 gives SGD. Same shuffle seed yields identical loss curves; batch_size=None matches explicit full-batch; different seeds diverge under SGD; full-batch training is invariant to shuffle order. Mini-batch convergence is checked as final full-dataset loss below initialization (not monotonic decrease every epoch — batch updates are noisy). ReLU training via grad_descent reduces loss.

test_tuning
    split_train_validation produces the correct shapes and no row appears in both splits. The split is reproducible when the same seed is used. grad_descent_with_validation returns loss lists of length epochs+1 with all finite values. hyperparameter_search returns one result dict per configuration with the expected keys and correct curve lengths.

test_regularization
    l2_penalty computes (lambda/2) * sum(W^2) over connection weights only; bias rows are excluded. l2_penalty_grad applies lmbda*W on weight rows and zero on the bias row. Known-value, zero-lambda, and multi-layer sum tests. Gradient-diff test: backprop(lmbda>0) minus backprop(lmbda=0) equals lmbda*W[:-1,:] with zero bias-row delta.


Roadmap

- Momentum / Adam optimiser
- Classification variant with cross-entropy loss and softmax output
- Interactive visualisation widgets for exploring the training surface