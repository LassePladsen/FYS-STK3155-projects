import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from jax import grad  # automatic differentitation

from part_b_activation import sigmoid
from part_b_cost import cost_logreg

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value
lmbda = 0.0001  # shrinkage hyperparameter lambda
eta = 0.001  # learning rate
n_batches = 3  # no. minibatches for sgd
degree = 2  # max polynomial degree for design matrix
n_epochs = 1000  # max no. epochs/iterations for nn training
theta_tol = 1e-7  # tolerance for convergence check
rng_seed = 2023  # seed for generating psuedo-random values, helps with debugging purposes

"""# ADAM parameters
rho1 = 0.99
rho2 = 0.999
delta = 1e-10  # small constant to avoid division by zero"""

# Grid plot parameters
eta_vals = np.logspace(-5, -1, 5)  # learning rates
lmbd_vals = np.logspace(-5, 0, 6)  # regularization rates

# Set up design matrix and target vector
data = load_breast_cancer()
X = data.data
target = data.target

# Split data into training and testing set
X_train, X_test, y_train, y_test  = train_test_split(X, target, test_size=0.2,
                                                        random_state=rng_seed)

# Set up random number generator
rng = np.random.default_rng(rng_seed)

# Set up gradient function
train_cost = cost_logreg(y_train.ravel())    
cost_grad = grad(train_cost)

def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n + 1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X


def stochastic_gradient_descent_logreg(X, y, eta, n_batches, lmbda=0,
                             max_epochs=1000, tol=1e-7):
    """Stochastic minibatch gradient descent algorithm"""
    M = int(n/n_batches)  # size of minibatches
    
    # Start with random theta guess
    theta = rng.standard_normal((2, 1))

    for epoch in max_epochs:
        # Iterate through all minibatches for each epoch iteration
        for i in range(n_batches):
            # Pick a minibatch at random from the data set 
            random_index = M * rng.integers(m)
            Xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]
            
            # Calculate logistic regression gradient
            gradient = cost_grad(Xi, yi, lmbda, theta)

            # Update theta
            theta_prev = theta.copy()
            theta -= eta * gradient

        # Convergence test
        if all(abs(theta - theta_prev) <= tol):
            return theta

    return theta


test_cost = cost_logreg(y_test.ravel())                                                   

print()

# plt.grid(True)
# plt.legend()
# plt.savefig("../results/figures/part_e_log_reg.png")