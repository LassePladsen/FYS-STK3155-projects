import sys
import autograd.numpy as np

from typing import Callable
from autograd import grad
from numpy.random import default_rng
from sklearn.metrics import accuracy_score

from cost import cost_ols, cost_logreg, cost_crossentropy
from activation import identity, sigmoid, softmax, relu, lrelu, derivate

'''def sigmoid(x):
    """Sigmoid function for activation."""
    return 1 / (1 + np.exp(-x))


def cost_mse(y_true):
    """Returns function for the mean squared error cost function."""

    def func(y_pred):
        return np.mean((y_true - y_pred)**2)

    return func'''


class FFNN:
    """Feed forward regression/classification neural network using backpropagation for training.
    Inspired by the FFNN class code from the FYS-STK3155 UiO course's lecture notes at
    https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#the-neural-network
    but implemented using our own written code.

    Attributes
    ----------
        dimensions (tuple[int]) : A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        hidden_func (Callable) : The activation function for the hidden layers
        output_func (Callable) : The activation function for the output layer
        cost_func (Callable) : Cost function for the network, it should be a function with parameter y_true (the target)
            and should return a function with parameter y_pred (the prediction).
        seed (int) : Sets seed for random number generator, makes results reproducible
    """

    def __init__(
            self,
            dimensions: tuple[int, ...],
            hidden_func: Callable = sigmoid,
            output_func: Callable = sigmoid,
            cost_func: Callable = cost_ols,
            seed: int = None
    ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.rng = default_rng(seed)

        # Set as classification or not (regression)
        self._set_classification()

        # Initialize weights and biases
        self.output_bias = list()
        self.output_weights = list()
        self.hidden_bias = list()
        self.hidden_weights = list()
        self.reset()

    def reset(self) -> None:
        """Resets hidden layer's and output layer's weights to random values from a normal distribution
        and biases to 0.01, in order to train the network from scratch.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        # Weights and bias in the hidden layer
        self.hidden_weights = list()
        self.hidden_bias = list()
        for i in range(1, len(self.dimensions) - 1):
            weight_array = self.rng.standard_normal(size=(self.dimensions[i], self.dimensions[i + 1]))
            bias_array = np.zeros(self.dimensions[i + 1]) + 0.01
            self.hidden_weights.append(weight_array)
            self.hidden_bias.append(bias_array)

        # Set the classification attribute
        self._set_classification()

        # Weights and bias in the output layer
        self.output_weights = self.rng.standard_normal((self.dimensions[-2], self.dimensions[-1]))
        self.output_bias = np.zeros(self.dimensions[-1]) + 0.01

    def _set_classification(self):
        """Decides if FFNN acts as classifier (True) og regressor (False), sets self.classification during init()

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        if (
                self.cost_func.__name__ == "cost_logreg"
                or self.cost_func.__name__ == "cost_cross_entropy"
        ):
            self.classification = True
        else:
            self.classification = False

    def _feedforward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feed forward algorithm, feeds the input through all the hidden layers, stores all the z_h and a_h values,
        then returns the output probabilities (a_o).

        Parameters
        ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features=self.dimensions[0])

        Returns
        -------
            z_o (np.ndarray) : Output layer weighted sum of inputs with shape (n_samples, n_outputs=self.dimensions[-1])
            probabilities (np.ndarray) : Output layer activation; the predicted probabilities with
                shape (n_samples, n_outputs=self.dimensions[-1])
        """

        # if X is just a vector, make it into a matrix (column vector)
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Store z and a matrix values for hidden layers
        self.z_matrices = list()
        self.a_matrices = list()

        # For the first hidden layer the activation is the design matrix X
        self.z_matrices.append(X)
        self.a_matrices.append(X)

        # Inputs and activation in the hidden layers
        for i in range(len(self.hidden_weights)):
            # Weighted sum of input to the hidden layer i
            z_h = self.a_matrices[i] @ self.hidden_weights[i] + self.hidden_bias[i]

            # Activation of the hidden layer i
            a_h = self.hidden_func(z_h)

            # Store matrices for hidden layer i
            self.z_matrices.append(z_h)
            self.a_matrices.append(a_h)

        # Weighted sum of inputs to the output layer
        print(a_h.shape, self.output_weights.shape, self.output_bias.shape)
        z_o = a_h @ self.output_weights + self.output_bias

        # Activation of output layer; contains the output probabilities
        probabilities = self.output_func(z_o)  # this is a_o

        return z_o, probabilities

    def _backpropagate(
            self,
            X: np.ndarray,
            target: np.ndarray,
            eta: float = 0.1,
            lmbda: float = 0.001
    ) -> None:

        """Performs the backpropagation algorithm. In other words, this method
            calculates the gradient of all the layers starting at the
            output layer, and moving from right to left accumulates the gradient until
            the input layer is reached. Each layers respective weights are updated while
            the algorithm propagates backwards from the output layer (auto-differentation in reverse mode).

        Parameters
        ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features=self.dimensions[0])
            target (np.ndarray) : Target data column-vector of size (n_samples)
            eta (float) : Learning rate for the network
            lmbda (float) : Regularization hyperparameter lambda

        Returns
        -------
            None
        """

        out_derivative = grad(self.output_func)
        hidden_derivative = grad(self.hidden_func)
        cost_func_derivative = grad(self.cost_func(target))

        z_o, probabilities = self._feedforward(X)

        # Output layer error delta^L term
        if self.output_func.__name__ == "softmax":  # multi-class classification
            delta_matrix = probabilities - target

        else:  # single class classification
            cost_func_derivative = grad(self.cost_func(target))
            delta_matrix = out_derivative(z_o) * cost_func_derivative(probabilities)

        # Gradients for the output layer
        output_weights_gradient = self.a_matrices[-1].T @ delta_matrix
        output_bias_gradient = np.sum(delta_matrix, axis=0)

        for i in range(len(self.hidden_weights) - 1, -1, -1):
            # Error delta^1 term for hidden layer i
            delta_matrix = delta_matrix @ self.hidden_weights[i + 1].T * hidden_derivative(self.z_matrices[i + 1])

            # Calculate gradients for hidden layer i
            hidden_weights_gradient = self.a_matrices[i].T @ delta_matrix
            hidden_bias_gradient = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

            # Regularization term
            hidden_weights_gradient += self.hidden_weights[i] * lmbda
            hidden_bias_gradient += self.hidden_bias[i] * lmbda

            # Update weights and biases for hidden layer i
            self.hidden_weights[i] -= eta * hidden_weights_gradient
            self.hidden_bias[i] -= eta * hidden_bias_gradient

    def _format(self, value, decimals=4):
        """Formats decimal numbers for progress bar

        Parameters
        ----------
            value (float) : Value to be formatted
            decimals (int) : Number of decimals to be printed

        Returns
        -------
            str : Formatted string
        """

        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1

        n = 1 + int(np.floor(np.log10(v)))

        if n >= decimals - 1:
            return str(round(value))

        return f"{value:.{decimals - n - 1}f}"


    def _progress_bar(self, progression, **kwargs) -> int:
        """Displays progress of training

        Parameters
        ----------
            progression (float) : Progression of training, should be between 0 and 1
            **kwargs : Metrics to be printed

        Returns
        -------
            int : Length of the printed line
        """

        print_length = 40

        num_equals = int(progression * print_length)
        num_not = print_length - num_equals

        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._format(progression * 100, decimals=5)
        line = f"  {bar} {perc_print}% "

        # Metrics
        for key in kwargs:
            if not np.isnan(kwargs[key]):
                value = self._format(kwargs[key], decimals=4)
                line += f"| {key}: {value} "

        sys.stdout.write("\r" + line)
        sys.stdout.flush()

        return len(line)

    def train(
            self,
            X: np.ndarray,
            target: np.ndarray,
            eta: float = 0.1,
            lmbda: float = 0.001,
            epochs: int = 1000,
            prnt: bool = True
    ) -> dict[str, np.ndarray]:
        """Trains/fits the neural network using back propagation a given amount of epoch iterations.

         Parameters
         ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features=self.dimensions[0])
            target (np.ndarray) : Target data column-vector of size (n_samples)
            eta (float) : Learning rate for the network
            lmbda (float) : Regularization hyperparameter lambda
            epochs (int) : Maximum number of epochs before stopping training
            prnt (bool) : If True, prints the training progress bar and metrics for each epoch

        Returns
        -------
            dict[str, np.ndarray] : Dictionary containing the training error and accuracy for each epoch. Keys are
                "error" and "accuracy".
        """

        cost_func_target = self.cost_func(target)

        # Initialize arrays for score metrics
        errors = np.empty(epochs)
        errors.fill(np.nan)
        accuracies = np.empty(epochs)
        accuracies.fill(np.nan)

        if prnt:
            print(f"{eta=}, lambda={lmbda}")

        try:
            for e in range(epochs):
                self._backpropagate(X=X, target=target, eta=eta, lmbda=lmbda)

                # Calculate performance metrics
                pred = self.predict(X)
                errors[e] = cost_func_target(pred)

                if self.classification:
                    train_acc = self._accuracy(self.predict(X), target)
                    accuracies[e] = train_acc

                # printing progress bar
                if prnt:
                    progression = e / epochs
                    print_length = self._progress_bar(
                            progression,
                            error=errors[e],
                            accuracy=accuracies[e],
                    )

        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        scores = {"error": errors, "accuracy": accuracies}
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Performs the prediction after the network has been trained. Rounds all probability values from the
        feed forward algorithm to nearest integer.

        Parameters
        ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features)

        Returns
        -------
            np.ndarray : Classicification prediction vector (row) of integers for each row in the design
                matrix (n_samples)
        """

        probabilities = self._feedforward(X)[-1]
        return np.where(probabilities >= 0.5, 1, 0)  # this rounds the probability array to nearest int


if __name__ == "__main__":
    X = np.asarray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    yOR = np.asarray([
        [0],
        [1],
        [1],
        [1]
    ])


    nn = FFNN(dimensions=(2, 2, 1))
    nn.train(X=X, target=yOR)
    pred = nn.predict(X)
    print(pred)
