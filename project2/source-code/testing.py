import numpy as np

from ffnn import FFNN
from activation import *
from cost import cost_ols

# Pparameters
N = 8  # no. data points
N_epoch = 1000  # max no. epochs/iterations before stopping
noise_std = 0.1  # standard deviation of noise
lmbda = 0.001  # Ridge hyperparameter lambda
eta = 0.1  # learning rate
theta_tol = 1e-7  # theta tolerance for stopping iteration when |theta_new - theta_old| <= theta_tol
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Create data set
rng = np.random.default_rng(rng_seed)
x = rng.random((N, 1)).reshape(-1, 1)
noise = rng.normal(0, noise_std, x.shape)
y = 5 - 10 * x + 2 * x**2 + noise

def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n+1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X

X = create_X_1d(x, 2)

nn = FFNN(
    dimensions=[X.shape[1], 100, 1],
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
        epochs=1000,
)

pred = nn.predict(X)
print("\nData:")
print(y.ravel())
print("\nPredictions:")
print(pred.ravel())

