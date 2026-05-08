from __future__ import annotations
import numpy as np
from mlp.forward import mlp_forward
from mlp.backward import backprop
from mlp.loss import mse_loss

def grad_descent(
    data: np.ndarray,
    my_mlp: dict[str, np.ndarray],
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], dict[str, np.ndarray]]:
    '''
    This will repeatedly calculate the output of the model and then apply a step of gradient descent.
    A list of the loss after each iteration (and also the loss before any iterations) should be returned along with the final model.

    Now we train the mlp on the dataset we had earlier with gradient descent.
    At each iteration : 
    forward pass -> compute MSE loss -> backpropagation -> update weights with learning rate : each W_layer -= learning_rate * dW_layer
    '''
    label_x = data[:, :2] # split the data into two layers of input, say x and y
    label_y = data[:, 2:3]
    _, predictions = mlp_forward(my_mlp, label_x) # record the initial losses here
    losses = [mse_loss(label_y, predictions)]
    for current in range(iterations):
        cache, predictions = mlp_forward(my_mlp, label_x) # forward pass
        gradients = backprop(my_mlp, cache, label_y, predictions) # backward pass
        for layer in my_mlp: # update each weight matrix here
            my_mlp[layer] -= learning_rate * gradients[f"d{layer}"]
        _, predictions = mlp_forward(my_mlp, label_x) # record the losses
        losses.append(mse_loss(label_y, predictions))

    return losses, my_mlp