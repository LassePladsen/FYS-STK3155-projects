from typing import Callable

from jax import grad
import jax.numpy as np


def sigmoid(x):
    """Sigmoid function for activation."""
    return 1 / (1 + np.exp(-x))

def cost_mse(y_true):
    """Returns function for the mean squared error cost function."""
    def func(y_pred):
        return np.mean((y_true - y_pred)**2)

    return func


class FFNN:
    """Feed Forward Neural Network using backpropagation for training. Inspired by the FFNN class code from the
    FYS-STK3155 UiO lecture notes at https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#the-neural-network

    Parameters:
        dimensions (tuple[int]): A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        hidden_func (Callable): The activation function for the hidden layers
        output_func (Callable): The activation function for the output layer
        cost_func (Callable): Cost function for the network, it should be a function with parameter y_true (the target)
            and should return a function with parameter y_pred (the prediction).
        seed (int): Sets seed for random number generator, makes results reproducible


    """

    def __init__(
            self,
            dimensions: tuple[int],
            hidden_func: Callable = sigmoid,
            output_func: Callable = sigmoid,
            cost_function: Callable = cost_mse,
            seed: int = 2023
    ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_function = cost_function
        self.seed = seed

        self._init_weights()


