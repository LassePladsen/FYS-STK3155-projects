from ffnn import FFNN
from activation import *
from cost import *
from scheduler import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value

lmbda = 0.0001  # shrinkage  hyperparameter lambda
eta = 0.01  # learning rate
degree = 1  # max polynomial degree for design matrix
n_epochs = 1000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Grid plot parameters
eta_vals = np.asarray([0.001, 0.005, 0.01, 0.02, 0.03, 0.04])#0.05, 0.1, 0.2, 0.3, 0.4])  # learning rates
lmbd_vals = np.logspace(-5, 0, 6) # regularization rates

# Create data set
rng = np.random.default_rng(rng_seed)
x = rng.uniform(-xmax, xmax, size=(n, 1))#.reshape(-1, 1)
noise = rng.normal(0, noise_std, x.shape)
y = 2 + 3*x + 4*x**2 + noise

def create_X_1d(x, n):
    """Returns the design matrix X from coordinates x with n polynomial degrees."""
    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, n+1))

    for p in range(1, n + 1):
        X[:, p] = x**p

    return X


x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2,
                                                    random_state=rng_seed)

X_train = create_X_1d(x_train, degree)
X_test = create_X_1d(x_test, degree)

nn = FFNN(
    dimensions=[X_train.shape[1], 50, 1],
    hidden_func=sigmoid,
    output_func=identity,
    cost_func=cost_ols,
    seed=rng_seed,
)

scores = nn.fit(
        X=X_train,
        t=y_train,
        lam=lmbda,
        epochs=n_epochs,
        scheduler=Constant(eta=eta),
        # scheduler=Adam(eta=eta, rho=0.01, rho2=0.01),
        batches=1,
)

def print_pred():
    pred = nn.predict(X_test)

    # PRINT DATA AND PREDICTION
    print("\nData:")
    print(y.ravel())
    print("\nPredictions:")
    print(pred.ravel())

def plot_pred(filename: str = ""):
    pred = nn.predict(X_test)

    # PLOT DATA AND PREDICTION
    test_mse = cost_ols(y_test)(pred)
    test_r2 = r2_score(y_test, pred)
    sort_order = np.argsort(x_test.ravel())
    x_sort = x_test.ravel()[sort_order]
    plt.scatter(x_sort, y_test.ravel()[sort_order], 5, label="Test data")
    plt.plot(x_sort, pred.ravel()[sort_order], "r-", label="Prediction fit")
    plt.title(f"$p={degree}$ | $\eta={eta}$ | $\lambda={lmbda}$ | {n_epochs=} | mse={test_mse:.1f} | r2={test_r2:.2f}")
    plt.legend()

    if filename:
        plt.savefig(filename)

    else:
        plt.show()

def plot_mse_grid(filename: str = ""):
    # PLOT MSE AS FUNC OF LAMBDA AND ETA
    # Iterate through parameters -> train -> save mse to heatmap
    mse_scores = np.zeros((eta_vals.size, lmbd_vals.size))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            nn.reset_weights()
            nn.fit(
                    X=X_train,
                    t=y_train,
                    lam=lmbd,
                    epochs=n_epochs,
                    scheduler=Constant(eta=eta),
                    # scheduler=Adam(eta=eta, rho=0.01, rho2=0.01),
                    batches=1,
            )
            pred = nn.predict(X_test)
            mse_scores[i, j] = cost_ols(y_test)(pred)

    # MSE heatmap
    sns.heatmap(
            mse_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "MSE"},
            cmap="coolwarm",
    )
    plt.title(f"Prediction MSE with {n_epochs} epochs")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename:
        plt.savefig(filename)

    else:
        plt.show()

def plot_r2_grid(filename: str = ""):
    # PLOT r2 AS FUNC OF LAMBDA AND ETA

    # Iterate through parameters -> train -> save r2 to heatmap
    r2_scores = np.zeros((eta_vals.size, lmbd_vals.size))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            nn.reset_weights()
            nn.fit(
                    X=X_train,
                    t=y_train,
                    lam=lmbd,
                    epochs=n_epochs,
                    scheduler=Constant(eta=eta),
                    # scheduler=Adam(eta=eta, rho=0.01, rho2=0.01),
                    batches=1,
            )
            pred = nn.predict(X_test)
            r2_scores[i, j] = r2_score(y_test, pred)

    # MSE heatmap
    sns.heatmap(
            r2_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "MSE"},
            cmap="viridis",
    )
    plt.title(f"Prediction R2 score with {n_epochs} epochs")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename:
        plt.savefig(filename)

    else:
        plt.show()


plot_pred("../../results/figures/part_b_pred.png")
# plot_r2_grid("../../results/figures/part_b_r2_grid.png")
# plot_mse_grid("../../results/figures/part_b_mse_grid.png")