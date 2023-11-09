import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from autograd import grad
import autograd.numpy as anp

from part_b_activation import sigmoid

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value
lmbda = 1e-5  # shrinkage hyperparameter lambda
eta = 0.5  # learning rate
n_batches = 10  # no. minibatches for sgd
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

def cost_logreg_w_regularization(X, y, theta, lmbda):
    """Cost function for logistic regression/cost entropy for binary classification
    with added L2 regularization hyperparameter lambda
    """
    d = 1e-10
    z = X @ theta
    a = sigmoid(z)
    return - 1/y.shape[0] * anp.sum(y * anp.log(a+d) + (1-y) * anp.log(1-a + d)) + lmbda * anp.linalg.norm(theta+d)


def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n + 1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X


# Set up the cost functions gradient using autodiff
cost_grad = grad(cost_logreg_w_regularization, 2)

def stochastic_gradient_descent_logreg(X, y, eta, lmbda, n_batches=1,
                             max_epochs=1000, tol=1e-7, prnt=True):
    """Stochastic minibatch gradient descent algorithm"""

    # Size of minibatches
    M = int(n/n_batches)  
    
    # Start with random theta/weight guess
    theta = rng.standard_normal((X.shape[1], 1))

    for epoch in range(max_epochs):
        # Iterate through all minibatches for each epoch iteration
        for i in range(n_batches):
            # Pick a minibatch at random from the data set 
            random_index = M * rng.integers(n_batches)
            Xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]
            
            # Calculate logistic regression gradient
            gradient = cost_grad(Xi, yi, theta, lmbda)

            # Update theta
            theta_prev = theta.copy()
            theta -= eta * gradient

        # Convergence test
        if all(abs(theta - theta_prev) <= tol):
            if prnt:
                print(f"Converged after {epoch} iterations.")
            return theta
    return theta


theta = stochastic_gradient_descent_logreg(X_train, y_train, eta, lmbda,
                                            n_batches, n_epochs, theta_tol)
pred = sigmoid(X_test @ theta)    
print(y_test)                                      
print(pred.ravel())
print(accuracy_score(y_test, pred))

# plt.grid(True)
# plt.legend()
# plt.savefig("../results/figures/part_e_log_reg.png")