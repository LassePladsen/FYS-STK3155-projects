from code_lectures_notes.ffnn import FFNN
from code_lectures_notes.activation import *
from code_lectures_notes.cost import *
from code_lectures_notes.scheduler import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # neural network from sckikit-learn for comparision
from sklearn.metrics import accuracy_score

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value
lmbda = 1e-3  # shrinkage  hyperparameter lambda
eta = 0.001  # learning rate
n_batches = 10  # no. minibatches for sgd
degree = 2  # max polynomial degree for design matrix
n_epochs = 5000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Grid plot parameters
eta_vals = np.logspace(-5, -1, 5)  # learning rates
lmbd_vals = np.logspace(-5, 0, 6)  # regularization rates

# Set up design matrix and target vector
data = load_breast_cancer()
X = data.data
target = data.target.reshape(-1,1)

# Split data into training and testing set
X_train, X_test, y_train, y_test  = train_test_split(X, target, test_size=0.2,
                                                        random_state=rng_seed)

test_cost = cost_ols(y_test.ravel())                                                   

nn = FFNN(
        dimensions=[X_train.shape[1], 1],
        hidden_func=sigmoid,
        output_func=sigmoid,
        cost_func=cost_logreg,
        seed=rng_seed,
)
nn.fit(
        X=X_train,
        t=y_train,
        lam=lmbda,
        epochs=n_epochs,
        scheduler=Adam(eta=eta, rho=0.9, rho2=0.999),
        # scheduler=Constant(eta=eta),
        batches=n_batches,
)


def print_pred():
    pred = nn.predict(X_test)

    # PRINT DATA AND PREDICTION
    print("\nData:")
    print(y_test.ravel())
    print("\nPredictions:")
    print(pred.ravel())


def plot_pred(filename: str = ""):
    pred = nn.predict(X_test)

    # PLOT DATA AND PREDICTION
    test_mse = test_cost(pred.ravel())
    test_r2 = r2_score(y_test, pred)
    sort_order = np.argsort(x_test.ravel())
    x_sort = x_test.ravel()[sort_order]

    plt.figure()
    plt.scatter(x_sort, y_test.ravel()[sort_order], 5, label="Test data")
    plt.plot(x_sort, pred.ravel()[sort_order], "r-", label="Prediction fit")
    plt.title(f"$p={degree}$ | $\eta={eta}$ | $\lambda={lmbda}$ | {n_epochs=} "
              f"\n{n_batches=} | mse={test_mse:.1f} | $R^2$={test_r2:.2f}")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)

    else:
        plt.show()


def plot_pred_scikit(filename: str = ""):
    nn_sk = MLPRegressor(hidden_layer_sizes=(50,), activation='logistic',
                         alpha=lmbda, learning_rate_init=eta, solver="sgd",
                         learning_rate="adaptive", max_iter=n_epochs,
                         batch_size=n_batches, random_state=rng_seed, momentum=0,
                         )
    nn_sk.fit(X_train, y_train.ravel())

    pred = nn_sk.predict(X_test)

    # PLOT DATA AND PREDICTION
    test_mse = test_cost(pred.ravel())
    test_r2 = r2_score(y_test, pred)
    sort_order = np.argsort(x_test.ravel())
    x_sort = x_test.ravel()[sort_order]

    plt.figure()
    plt.scatter(x_sort, y_test.ravel()[sort_order], 5, label="Test data")
    plt.plot(x_sort, pred.ravel()[sort_order], "r-", label="Prediction fit")
    plt.title(f"$p={degree}$ | $\eta={eta}$ | $\lambda={lmbda}$ | {n_epochs=} "
              f"\n{n_batches=} | mse={test_mse:.1f} | $R^2$={test_r2:.2f}")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)

    else:
        plt.show()


def plot_accuracy_grid(filename_mse: str = "", filename_r2: str = ""):
    # PLOT ACCURACY SCORE AS FUNC OF LAMBDA AND ETA

    # Iterate through parameters -> train -> save accuracy to heatmap
    mse_scores = np.zeros((eta_vals.size, lmbd_vals.size))
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
                    batches=n_batches,
            )
            pred = nn.predict(X_test)
            mse_scores[i, j] = test_cost(pred.ravel())
            r2_scores[i, j] = r2_score(y_test, pred)

    # MSE heatmap
    plt.figure()
    sns.heatmap(
            mse_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "MSE"},
            cmap="coolwarm",
    )
    plt.title(f"MSE: $p={degree}$ | {n_epochs=} | {n_batches=}")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename_mse:
        plt.savefig(filename_mse)

    else:
        plt.show()

    # R2 heatmap
    plt.figure()
    sns.heatmap(
            r2_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "R2 score"},
            cmap="viridis",
    )
    plt.title(f"R2: $p={degree}$ | {n_epochs=} | {n_batches=}")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename_r2:
        plt.savefig(filename_r2)

    else:
        plt.show()


# print_pred()
pred = nn.predict(X_test)
print(y_test.ravel())
print(pred.ravel())
print(accuracy_score(y_test, pred))
# plot_pred("../../results/figures/part_b_pred.png")
# plot_pred_scikit("../../results/figures/part_b_pred_scikit.png")
# plot_mse_r2_grid(
#         "../../results/figures/part_b_mse_grid.png",
#         "../../results/figures/part_b_r2_grid.png"
# )