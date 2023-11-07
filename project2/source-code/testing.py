from part_b_ffnn import FFNN
from part_b_activation import *
from part_b_cost import cost_ols

import numpy as np

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value

lmbda = 0.0001  # shrinkage  hyperparameter lambda
eta = 0.01  # learning rate
degree = 1  # max polynomial degree for design matrix
n_epochs = 1000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Create data set
rng = np.random.default_rng(rng_seed)
x = rng.uniform(-xmax, xmax, size=(n, 1))#.reshape(-1, 1)
noise = rng.normal(0, noise_std, x.shape)
y = 2 + 3*x + 4*x**2# + noise

def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n+1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X

X = create_X_1d(x, degree)

nn = FFNN(
    dimensions=[X.shape[1], 50, 1],
    hidden_func=sigmoid,
    output_func=identity,
    cost_func=cost_ols,
    seed=rng_seed,
)

nn.train(
        X=X,
        target=y,
        eta=eta,
        lmbda=lmbda,
        epochs=n_epochs,
)

pred = nn.predict(X)
print("\nData:")
print(y.ravel())
print("\nPredictions:")
print(pred.ravel())

print(nn._z_matrices[1])
print(sigmoid(nn._z_matrices[1]))

