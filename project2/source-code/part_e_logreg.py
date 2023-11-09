import warnings

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
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Parameters
n = 100  # no. data points
noise_std = 1  # standard deviation of noise
xmax = 5  # max x value
lmbda = 1e-2  # shrinkage  hyperparameter lambda
eta = 0.001  # learning rate
n_batches = 5  # no. minibatches for sgd
n_epochs = 1000  # no. epochs/iterations for nn training
rng_seed = 2023  # seed for generating psuedo-random values, helps withbugging purposes

# Grid plot parameters
eta_vals = np.logspace(-6, 0, 7)  # learning rates
lmbd_vals = np.logspace(-6, 0, 7)  # regularization rates

# Set up design matrix and target vector
data = load_breast_cancer()
X = data.data
target = data.target.reshape(-1,1)

# Split data into training and testing set
X_train, X_test, y_train, y_test  = train_test_split(X, target, test_size=0.2,
                                                        random_state=rng_seed)

test_cost = cost_ols(y_test.ravel())                                                   

# FOR LOGISTIC REGRESSION WE JUST USE THE NN CODE WITH NO HIDDEN LAYERS
nn = FFNN(
        dimensions=[X_train.shape[1], 1],
        hidden_func=sigmoid,
        output_func=sigmoid,
        cost_func=cost_logreg,
        seed=rng_seed,
)

def train():
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
    train()
    pred = nn.predict(X_test)

    # PRINT DATA AND PREDICTION
    print("\nData:")
    print(y_test.ravel())
    print("\nPredictions:")
    print(pred.ravel())
    print(f"\nTest accuracy={(accuracy_score(y_test, pred))}")
    
def print_pred_scikit():
    model = LogisticRegression(
                        C=1 / lmbda,
                        penalty="l2",
                        max_iter=n_epochs,
                        tol=1e-4,
                        solver="lbfgs",
                        random_state=rng_seed,
                )
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)

    # PRINT DATA AND PREDICTION
    print("\nData:")
    print(y_test.ravel())
    print("\nPredictions:")
    print(pred.ravel())
    print(f"\nTest accuracy={(accuracy_score(y_test, pred))}")

def plot_accuracy_grid(filename: str = ""):
    # PLOT ACCURACY SCORE AS FUNC OF LAMBDA AND ETA

    # Iterate through parameters -> train -> save accuracy to heatmap
    acc_scores = np.zeros((eta_vals.size, lmbd_vals.size))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            nn.reset_weights()
            try:
                nn.fit(
                        X=X_train,
                        t=y_train,
                        lam=lmbd,
                        epochs=n_epochs,
                        scheduler=Adam(eta=eta, rho=0.9, rho2=0.999),
                        # scheduler=Constant(eta=eta),
                        batches=n_batches,
                )
                pred = nn.predict(X_test)
                acc_scores[i, j] = accuracy_score(y_test, pred)
            except Exception as e:
                print(f"Error: {type(e)} for ({eta=}, {lmbd=}), replacing accuracy with NaN")
                acc_scores[i, j] = np.nan
            
    # Accuracy heatmap
    plt.figure()
    sns.heatmap(
            acc_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "Test accuracy"},
            cmap="viridis",
    )
    plt.title(f"Accuracy scores: {n_epochs=} | {n_batches=}")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename:
        plt.savefig(filename)

    else:
        plt.show()


def plot_accuracy_grid_scikit(filename: str = ""):
    # SKLEARN LOGISTIC REGRESSION FOR COMPARISON

    # Iterate through parameters -> train -> save accuracy to heatmap
    acc_scores = np.zeros((eta_vals.size, lmbd_vals.size))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            try:
                model = LogisticRegression(
                        C=1 / lmbd,
                        penalty="l2",
                        max_iter=n_epochs,
                        tol=1e-4,
                        solver="lbfgs",
                        random_state=rng_seed,
                )
                model.fit(X_train, y_train.ravel())
                pred = model.predict(X_test)
                acc_scores[i, j] = accuracy_score(y_test, pred)
            except Exception as e:
                print(f"Error: {type(e)} for ({eta=}, {lmbd=}), replacing accuracy with NaN")
                acc_scores[i, j] = np.nan

    # Accuracy heatmap
    plt.figure()
    sns.heatmap(
            acc_scores,
            annot=True,
            fmt=".2f",
            xticklabels=[f"{lmbd:g}" for lmbd in lmbd_vals],
            yticklabels=[f"{eta:g}" for eta in eta_vals],
            cbar_kws={"label": "Test accuracy"},
            cmap="viridis",
    )
    plt.title(f"Accuracy scores: {n_epochs=}")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# print_pred()
# print_pred_scikit()
# plot_accuracy_grid(f"../results/figures/part_e_grid_batches{n_batches}.png")
# plot_accuracy_grid_scikit(f"../results/figures/part_e_grid_scikit.png")
