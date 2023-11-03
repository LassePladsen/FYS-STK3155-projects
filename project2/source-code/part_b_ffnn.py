import sys
import autograd.numpy as np

from typing import Callable, Iterable
from autograd import grad, elementwise_grad
from numpy.random import default_rng
from sklearn.metrics import accuracy_score

from part_b_cost import cost_ols, cost_logreg, cost_crossentropy
from part_b_activation import identity, sigmoid, softmax, relu, lrelu, derivate


class FFNN:
    """Feed forward regression/classification neural network using backpropagation for training.
    Inspired by the FFNN class code from the FYS-STK3155 UiO course's lecture notes at
    https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#the-neural-network
    but implemented using our own written code.

    Parameters
    ----------
        dimensions (Iterable[int]): An iterable of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        hidden_func (Callable) : The activation function for the hidden layers
        output_func (Callable) : The activation function for the output layer
        cost_func (Callable) : Cost function for the network, it should be a function with parameter y_true (the target)
            and should return a function with parameter y_pred (the prediction).
        seed (int) : Sets seed for random number generator, makes results reproducible

    Attributes
    ----------
        dimensions (Iterable[int]): An iterable of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        hidden_func (Callable) : The activation function for the hidden layers
        output_func (Callable) : The activation function for the output layer
        cost_func (Callable) : Cost function for the network, it should be a function with parameter y_true (the target)
            and should return a function with parameter y_pred (the prediction).
        rng (np.random.Generator) : Random number generator for creating random weights and biases
    """

    def __init__(
            self,
            dimensions: Iterable[int],
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
        self._biases = list()
        self._weights = list()
        self.reset()

    def reset(self, bias_std: float = 0.01) -> None:
        """Resets hidden layer's and output layer's weights and biases to random values from a normal distribution
        , in order to train the network from scratch.

        Parameters
        ----------
            bias_std (float) : Bias standard deviation for the hidden and output layers

        Returns
        -------
            None
        """

        # Weights and bias in the layers
        self._weights = list()
        self._biases = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = self.rng.standard_normal(size=(self.dimensions[i], self.dimensions[i + 1]))
            bias_array = self.rng.standard_normal(size=self.dimensions[i + 1]) * 0.01
            # bias_array = self.rng.normal(0, bias_std, size=self.dimensions[i + 1])

            self._weights.append(weight_array)
            self._biases.append(bias_array)

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
            print(f"Eta={eta}, Lambda={lmbda}")

        try:
            for e in range(epochs):
                self._backpropagate(X=X, target=target, eta=eta, lmbda=lmbda)

                # Calculate performance metrics
                pred = self.predict(X)
                errors[e] = cost_func_target(pred)

                if self.classification:
                    accuracies[e] = accuracy_score(target, pred)

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

        if prnt:
            # Finish progress bar
            sys.stdout.write("\r" + " " * print_length)
            sys.stdout.flush()
            self._progress_bar(
                    1,
                    error=errors[e],
                    accuracy=accuracies[e],
            )
            print()

        scores = {"error": errors, "accuracy": accuracies}
        return scores

    def predict(self, X: np.ndarray, threshold: int = 0.5) -> np.ndarray:
        """Performs the prediction after the network has been trained. If regression this returns the prediction of
        floats, if classification then it rounds all probability values from the probability prediction with
        the given threshold.

        Parameters
        ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features)
            threshold (int) : Threshold for classification, only used if self.classification is True

        Returns
        -------
            np.ndarray : Pprediction vector (row) of for each row in the design matrix (n_samples.
        """

        probabilities = self._feedforward(X)

        if self.classification:
            return np.where(probabilities >= threshold, 1, 0)
        else:
            return probabilities


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
                self.cost_func.__name__ == """cost_logreg"""
                or self.cost_func.__name__ == "cost_cross_entropy"
        ):
            self.classification = True
        else:
            self.classification = False

    def _feedforward(self, X: np.ndarray) -> None:
        """Feed forward algorithm, feeds the input through all the hidden layers, stores all the z^h and a^h layer
        values, also returns the output probabilities (a^L).

        Parameters
        ----------
            X (np.ndarray) : Input design matrix with shape (n_samples, n_features=self.dimensions[0])

        Returns
        -------
            np.ndarray : Final output layer activation with shape (n_samples, n_features=self.dimensions[-1])
                which contains the probabilities.
        """

        # if X is just a vector, make it into a matrix (column vector)
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # Store z and a matrix values for layers
        self._z_matrices = list()
        self._a_matrices = list()

        # For the first hidden layer the activation is the design matrix X
        self._z_matrices.append(X)
        self._a_matrices.append(X)

        # Inputs and activation in the layers
        for i in range(len(self._weights)):
            # Weighted sum of input to the hidden layer i
            z = self._a_matrices[i] @ self._weights[i] + self._biases[i]

            # Activation of the layer i
            a = self.hidden_func(z)

            # Store matrices for layer i
            self._z_matrices.append(z)
            self._a_matrices.append(a)

        # The final activation (output layer) a^L, which contains the probabilities
        return a

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

        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)

        probabilities = self._feedforward(X)

        for i in range(len(self._weights) - 1, -1, -1):

            # Output layer error delta^L term
            if i == len(self._weights) - 1:

                # multi-class classification
                if self.output_func.__name__ == "softmax":
                    delta_matrix = probabilities - target

                # single class classification
                else:
                    cost_func_derivative = grad(self.cost_func(target))
                    delta_matrix = out_derivative(self._z_matrices[-1]) * cost_func_derivative(probabilities)

            # Error delta^1 term for hidden layer i
            else:
                delta_matrix = delta_matrix @ self._weights[i + 1].T * hidden_derivative(self._z_matrices[i + 1])

            # Calculate gradients for layer i
            weights_gradient = self._a_matrices[i].T @ delta_matrix
            bias_gradient = np.sum(delta_matrix, axis=0)

            # Regularization term
            weights_gradient += self._weights[i] * lmbda
            bias_gradient += self._biases[i] * lmbda

            # Update weights and biases for layer i
            self._weights[i] -= eta * weights_gradient
            self._biases[i] -= eta * bias_gradient

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


"""if __name__ == "__main__":
    X = np.asarray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    t = np.asarray([
        [0],
        [1],
        [1],
        [0]
    ])

    nn = FFNN(
            dimensions=(2, 2, 1),
            cost_func=cost_logreg,
            hidden_func=sigmoid,  # relu is bad for XOR gate with cost_logreg?
            output_func=identity, ### SOFTMAX OR IDENTITY GIVES BEST RESUTLS FOR XOR GATE WITH cost_logreg, (CONCLUSION: SIGMOID NOT GOOD FOR OUTPUT ACTIVATION?

    )
    scores = nn.train(X=X, target=t, epochs=1000)
    pred = nn.predict(X)
    print(pred)"""
