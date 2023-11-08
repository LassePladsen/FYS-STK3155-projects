import numpy as np
import matplotlib.pyplot as plt

from part_b_activation import sigmoid

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value
lmbda = 0.0001  # shrinkage  hyperparameter lambda
eta = 0.001  # learning rate
n_batches = 3  # no. minibatches for sgd
M = int(n/n_batches)
degree = 2  # max polynomial degree for design matrix
n_epochs = 1000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes


# ADAM parameters
rho1 = 0.99
rho2 = 0.999
delta = 1e-10  # small constant to avoid division by zero


# Grid plot parameters
eta_vals = np.logspace(-5, -1, 5)  # learning rates
lmbd_vals = np.logspace(-5, 0, 6)  # regularization rates

# Create data set
rng = np.random.default_rng(rng_seed)
x = rng.uniform(-xmax, xmax, size=(n, 1))  # .reshape(-1, 1)
noise = rng.normal(0, noise_std, x.shape)
y = 2 + 3 * x + 4 * x**2 + noise


def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n + 1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X


x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2,
                                                    random_state=rng_seed)

test_cost = cost_ols(y_test.ravel())                                                   

X_train = create_X_1d(x_train, degree)
X_test = create_X_1d(x_test, degree)

theta_adamsgd_ridge_prev = np.zeros(theta_adamsgd_ridge.shape)
for epoch_ridge in epoch_iters:
    theta_adamsgd_ridge_prev = theta_adamsgd_ridge.copy()
    moment1 = 0
    moment2 = 0

    # Iterate through all minibatches for each epoch iteration
    for i in range(m):
        # Pick a minibatch at random from the data set 
        random_index = M * rng.integers(m)
        Xi = X[random_index:random_index + M]
        yi = y[random_index:random_index + M]
        
        # Calculate gradient
        gradient = grad_ridge(Xi, yi, lmbda, theta_adamsgd_ridge)
        
        # Update moments
        moment1 = rho1 * moment1 + (1 - rho1) * gradient
        moment2 = rho2 * moment2 + (1 - rho2) * gradient * gradient
        
        # Correct moment bias'
        term1 = moment1/(1 - rho1**(epoch_ridge+1))
        term2 = moment2/(1 - rho2**(epoch_ridge+1))

        # Update theta
        theta_adamsgd_ridge -= learn_rate*term1 / (delta + np.sqrt(term2))  # scale learning rate with r

    # Store MSE as error
    errors_adamsgd_ridge.append(ols_cost(X, y, theta_adamsgd_ridge))

    # Convergence test
    if all(abs(theta_adamsgd_ridge - theta_adamsgd_ridge_prev) <= theta_tol):
        break

plt.plot(errors_adamsgd_ols, label="OLS")
plt.plot(errors_adamsgd_ridge, label=f"Ridge, lambda={lmbda}")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title(rf"ADAM SGD, $\eta_0={learn_rate:.2f}, \rho_1={rho1}, \rho_2={rho2}, {M=}$")
plt.grid(True)
plt.legend()