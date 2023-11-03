import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import seaborn as sns

n = 100
x = 2 * np.random.rand(n, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = 2 + 3*x + 4*x**2 + noise

X = np.c_[np.ones((n, 1)), x]
XT_X = X.T @ X
I = np.identity(2)
lamb = 0.001          # Ridge parameter

beta_linreg_OLS = np.linalg.inv(XT_X) @ (X.T @ y)             # Analytical beta_OLS
beta_linreg_Ridge = np.linalg.inv(XT_X - lamb*I) @ (X.T @ y)  # Analytical beta_Ridge

# Hessian matrix
H_OLS = (2.0/n) * XT_X
H_Ridge = (2.0/n) * XT_X + 2*lamb*I
EigValues_OLS, EigVectors_OLS = np.linalg.eig(H_OLS)
EigValues_Ridge, EigVectors_Ridge = np.linalg.eig(H_Ridge)
#  1 / np.max(EigValues_OLS)  [Known learning rate]


# Tasks week 41, Anal.grad. [for GD, GDM, SGD, SGDM]
N = 1000                               # Max nr. of iterations
tol = 1e-4                             # Convergence tolerance
mom = 0.3
gamma_GD = 0.2
n_epochs = 50
M = 5           # size of each minibatch
m = int(n/M)    # number of minibatches
t0, t1 = 5, 50
# AdaGrad parameter to avoid possible division by zero:
delta = 1e-8
beta_anal = np.random.randn(2, 1)


def learning_schedule(t):
    """Used to find gamma in SGD/SGDM"""
    return t0/(t+t1)


def anal_GD(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    itera = []
    error = []
    Giter = 0.0
    iter = 0.0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(N):
        itera.append(i)
        iter += 1
        error_ = np.mean((X @ Beta - y) ** 2)
        error.append(error_)

        if method == 0:
            gradients = 2.0/n * X.T @ ((X @ Beta)-y)
        else:
            gradients = 2.0/n * X.T @ ((X @ Beta)-y) + 2*lamb*Beta

        if Type == 0:
            Giter += gradients * gradients
            update = gamma_GD*gradients / (delta+np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        elif Type == 1:
            # Now also scaling Giter with rho:
            Giter += (rho * Giter + (1 - rho) * gradients * gradients)
            update = gamma_GD * gradients / (delta + np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        elif Type == 2:
            # Computing moments first
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1 ** iter)
            second_term = second_moment / (1.0 - beta2 ** iter)
            # Scaling with rho the new and the previous results
            update = gamma_GD * first_term / (np.sqrt(second_term) + delta)
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        else:
            update = gamma_GD * gradients
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
    return Beta, error, itera


def anal_GDM(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    update = 0
    itera = []
    error = []
    Giter = 0.0
    iter = 0.0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(N):
        itera.append(i)
        iter += 1
        error_ = np.mean((X @ Beta - y) ** 2)
        error.append(error_)

        if method == 0:
            gradients = 2.0/n * X.T @ ((X @ Beta)-y)
        else:
            gradients = 2.0/n * X.T @ ((X @ Beta)-y) + 2*lamb*Beta

        if Type == 0:
            Giter += gradients * gradients
            update = mom * update + gamma_GD*gradients / (delta+np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update

        elif Type == 1:
            # Now also scaling Giter with rho:
            Giter += (rho * Giter + (1 - rho) * gradients * gradients)
            update = mom * update + gamma_GD * gradients / (delta + np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update

        elif Type == 2:
            # Computing moments first
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1 ** iter)
            second_term = second_moment / (1.0 - beta2 ** iter)
            # Scaling with rho the new and the previous results
            update = mom * update + gamma_GD * first_term / (np.sqrt(second_term) + delta)
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        else:
            update = mom * update + gamma_GD * gradients
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
    return Beta, error, itera


def anal_SGD(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    k = 0
    itera = []
    error = []
    iter = 0.0
    """first_moment = 0.0
    second_moment = 0.0"""
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        Giter = 0.0
        for i in range(m):
            itera.append(k)
            k += 1
            iter += 1
            error_ = np.mean((X @ Beta - y) ** 2)
            error.append(error_)

            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            if method == 0:
                gradients = (2.0 / M) * xi.T @ ((xi @ Beta) - yi)
            else:
                gradients = (2.0/M) * xi.T @ ((xi @ Beta)-yi) + 2*lamb*Beta
            gamma = learning_schedule(epoch * m + i)

            if Type == 0:
                Giter += gradients*gradients
                update = gamma*gradients / (delta+np.sqrt(Giter))
                dif = (Beta-update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 1:
                # Now also scaling Giter with rho:
                Giter += (rho * Giter + (1 - rho) * gradients * gradients)
                update = gamma * gradients / (delta + np.sqrt(Giter))
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 2:
                # Computing moments first
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
                first_term = first_moment / (1.0 - beta1 ** iter)
                second_term = second_moment / (1.0 - beta2 ** iter)
                # Scaling with rho the new and the previous results
                update = gamma * first_term / (np.sqrt(second_term) + delta)
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            else:
                update = gamma * gradients
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
    return Beta, error, itera


def anal_SGDM(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    k = 0
    itera = []
    error = []
    update = 0
    iter = 0.0
    """first_moment = 0.0
    second_moment = 0.0"""
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        Giter = 0.0
        for i in range(m):
            iter += 1
            itera.append(k)
            k += 1
            error_ = np.mean((X @ Beta - y) ** 2)
            error.append(error_)

            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            if method == 0:
                gradients = (2.0 / M) * xi.T @ ((xi @ Beta) - yi)
            else:
                gradients = (2.0/M) * xi.T @ ((xi @ Beta)-yi) + 2*lamb*Beta
            gamma = learning_schedule(epoch * m + i)

            if Type == 0:
                Giter += gradients*gradients
                update = mom * update + gamma*gradients / (delta+np.sqrt(Giter))
                dif = (Beta-update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 1:
                # Now also scaling Giter with rho:
                Giter += (rho * Giter + (1 - rho) * gradients * gradients)
                update = mom * update + gamma * gradients / (delta + np.sqrt(Giter))
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 2:
                # Computing moments first
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
                first_term = first_moment / (1.0 - beta1 ** iter)
                second_term = second_moment / (1.0 - beta2 ** iter)
                # Scaling with rho the new and the previous results
                update = mom * update + gamma * first_term / (np.sqrt(second_term) + delta)
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            else:
                update = mom * update + gamma * gradients
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
    return Beta, error, itera


# Plain (Normal):
error_OLS_GD_plain1, iterations_OLS_GD_plain1 = anal_GD(beta_anal, 0, 3)[1:3]
error_OLS_GDM_plain1, iterations_OLS_GDM_plain1 = anal_GDM(beta_anal, 0, 3)[1:3]

error_OLS_SGD_plain1, iterations_OLS_SGD_plain1 = anal_SGD(beta_anal, 0, 3)[1:3]
error_OLS_SGDM_plain1, iterations_OLS_SGDM_plain1 = anal_SGDM(beta_anal, 0, 3)[1:3]

error_Ridge_GD_plain1, iterations_Ridge_GD_plain1 = anal_GD(beta_anal, 1, 3)[1:3]
error_Ridge_GDM_plain1, iterations_Ridge_GDM_plain1 = anal_GDM(beta_anal, 1, 3)[1:3]

error_Ridge_SGD_plain1, iterations_Ridge_SGD_plain1 = anal_SGD(beta_anal, 1, 3)[1:3]
error_Ridge_SGDM_plain1, iterations_Ridge_SGDM_plain1 = anal_SGDM(beta_anal, 1, 3)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE with Anal.grad.")
plt.plot(iterations_OLS_GD_plain1, error_OLS_GD_plain1, label="OLS_GD")
plt.plot(iterations_OLS_GDM_plain1, error_OLS_GDM_plain1, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_plain1, error_OLS_SGD_plain1, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_plain1, error_OLS_SGDM_plain1, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_plain1, error_Ridge_GD_plain1, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_plain1, error_Ridge_GDM_plain1, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_plain1, error_Ridge_SGD_plain1, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_plain1, error_Ridge_SGDM_plain1, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# AdaGrad:
error_OLS_GD_adagrad1, iterations_OLS_GD_adagrad1 = anal_GD(beta_anal, 0, 0)[1:3]
error_OLS_GDM_adagrad1, iterations_OLS_GDM_adagrad1 = anal_GDM(beta_anal, 0, 0)[1:3]

error_OLS_SGD_adagrad1, iterations_OLS_SGD_adagrad1 = anal_SGD(beta_anal, 0, 0)[1:3]
error_OLS_SGDM_adagrad1, iterations_OLS_SGDM_adagrad1 = anal_SGDM(beta_anal, 0, 0)[1:3]

error_Ridge_GD_adagrad1, iterations_Ridge_GD_adagrad1 = anal_GD(beta_anal, 1, 0)[1:3]
error_Ridge_GDM_adagrad1, iterations_Ridge_GDM_adagrad1 = anal_GDM(beta_anal, 1, 0)[1:3]

error_Ridge_SGD_adagrad1, iterations_Ridge_SGD_adagrad1 = anal_SGD(beta_anal, 1, 0)[1:3]
error_Ridge_SGDM_adagrad1, iterations_Ridge_SGDM_adagrad1 = anal_SGDM(beta_anal, 1, 0)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using AdaGrad (with Anal.grad.)")
plt.plot(iterations_OLS_GD_adagrad1, error_OLS_GD_adagrad1, label="OLS_GD")
plt.plot(iterations_OLS_GDM_adagrad1, error_OLS_GDM_adagrad1, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_adagrad1, error_OLS_SGD_adagrad1, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_adagrad1, error_OLS_SGDM_adagrad1, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_adagrad1, error_Ridge_GD_adagrad1, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_adagrad1, error_Ridge_GDM_adagrad1, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_adagrad1, error_Ridge_SGD_adagrad1, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_adagrad1, error_Ridge_SGDM_adagrad1, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# RMSprop:
# Value for parameter rho
rho = 0.99

error_OLS_GD_RMSprop1, iterations_OLS_GD_RMSprop1 = anal_GD(beta_anal, 0, 1)[1:3]
error_OLS_GDM_RMSprop1, iterations_OLS_GDM_RMSprop1 = anal_GDM(beta_anal, 0, 1)[1:3]

error_OLS_SGD_RMSprop1, iterations_OLS_SGD_RMSprop1 = anal_SGD(beta_anal, 0, 1)[1:3]
error_OLS_SGDM_RMSprop1, iterations_OLS_SGDM_RMSprop1 = anal_SGDM(beta_anal, 0, 1)[1:3]

error_Ridge_GD_RMSprop1, iterations_Ridge_GD_RMSprop1 = anal_GD(beta_anal, 1, 1)[1:3]
error_Ridge_GDM_RMSprop1, iterations_Ridge_GDM_RMSprop1 = anal_GDM(beta_anal, 1, 1)[1:3]

error_Ridge_SGD_RMSprop1, iterations_Ridge_SGD_RMSprop1 = anal_SGD(beta_anal, 1, 1)[1:3]
error_Ridge_SGDM_RMSprop1, iterations_Ridge_SGDM_RMSprop1 = anal_SGDM(beta_anal, 1, 1)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using RMSprop (with Anal.grad.)")
plt.plot(iterations_OLS_GD_RMSprop1, error_OLS_GD_RMSprop1, label="OLS_GD")
plt.plot(iterations_OLS_GDM_RMSprop1, error_OLS_GDM_RMSprop1, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_RMSprop1, error_OLS_SGD_RMSprop1, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_RMSprop1, error_OLS_SGDM_RMSprop1, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_RMSprop1, error_Ridge_GD_RMSprop1, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_RMSprop1, error_Ridge_GDM_RMSprop1, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_RMSprop1, error_Ridge_SGD_RMSprop1, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_RMSprop1, error_Ridge_SGDM_RMSprop1, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# ADAM:
# Value for parameters beta1 and beta2
beta1 = 0.9
beta2 = 0.999

error_OLS_GD_ADAM1, iterations_OLS_GD_ADAM1 = anal_GD(beta_anal, 0, 2)[1:3]
error_OLS_GDM_ADAM1, iterations_OLS_GDM_ADAM1 = anal_GDM(beta_anal, 0, 2)[1:3]

error_OLS_SGD_ADAM1, iterations_OLS_SGD_ADAM1 = anal_SGD(beta_anal, 0, 2)[1:3]
error_OLS_SGDM_ADAM1, iterations_OLS_SGDM_ADAM1 = anal_SGDM(beta_anal, 0, 2)[1:3]

error_Ridge_GD_ADAM1, iterations_Ridge_GD_ADAM1 = anal_GD(beta_anal, 1, 2)[1:3]
error_Ridge_GDM_ADAM1, iterations_Ridge_GDM_ADAM1 = anal_GDM(beta_anal, 1, 2)[1:3]

error_Ridge_SGD_ADAM1, iterations_Ridge_SGD_ADAM1 = anal_SGD(beta_anal, 1, 2)[1:3]
error_Ridge_SGDM_ADAM1, iterations_Ridge_SGDM_ADAM1 = anal_SGDM(beta_anal, 1, 2)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using ADAM (with AutoGrad)")
plt.plot(iterations_OLS_GD_ADAM1, error_OLS_GD_ADAM1, label="OLS_GD")
plt.plot(iterations_OLS_GDM_ADAM1, error_OLS_GDM_ADAM1, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_ADAM1, error_OLS_SGD_ADAM1, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_ADAM1, error_OLS_SGDM_ADAM1, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_ADAM1, error_Ridge_GD_ADAM1, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_ADAM1, error_Ridge_GDM_ADAM1, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_ADAM1, error_Ridge_SGD_ADAM1, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_ADAM1, error_Ridge_SGDM_ADAM1, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# EXTRA:
# Here we plot the relationship between the MSE of the different methods using optimal parameters
plt.figure(figsize=(10, 9))
plt.suptitle(fr"[Anal.grad.OLS]: Iterations = {N}, GD learning rate = {gamma_GD}, mini-batches = {m}, epochs = {n_epochs}")
plt.subplot(2, 2, 1)
plt.title("GD")
plt.plot(iterations_OLS_GD_plain1, error_OLS_GD_plain1, label="Normal")
plt.plot(iterations_OLS_GD_adagrad1, error_OLS_GD_adagrad1, label="AdaGrad")
plt.plot(iterations_OLS_GD_RMSprop1, error_OLS_GD_RMSprop1, label="RMSprop")
plt.plot(iterations_OLS_GD_ADAM1, error_OLS_GD_ADAM1, label="ADAM")
plt.legend()
plt.ylabel("MSE")
plt.subplot(2, 2, 2)
plt.title("GDM")
plt.plot(iterations_OLS_GDM_plain1, error_OLS_GDM_plain1, label="Normal")
plt.plot(iterations_OLS_GDM_adagrad1, error_OLS_GDM_adagrad1, label="AdaGrad")
plt.plot(iterations_OLS_GDM_RMSprop1, error_OLS_GDM_RMSprop1, label="RMSprop")
plt.plot(iterations_OLS_GDM_ADAM1, error_OLS_GDM_ADAM1, label="ADAM")
plt.legend()
plt.subplot(2, 2, 3)
plt.title("SGD")
plt.plot(iterations_OLS_SGD_plain1, error_OLS_SGD_plain1, label="Normal")
plt.plot(iterations_OLS_SGD_adagrad1, error_OLS_SGD_adagrad1, label="AdaGrad")
plt.plot(iterations_OLS_SGD_RMSprop1, error_OLS_SGD_RMSprop1, label="RMSprop")
plt.plot(iterations_OLS_SGD_ADAM1, error_OLS_SGD_ADAM1, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.subplot(2, 2, 4)
plt.title("SGDM")
plt.plot(iterations_OLS_SGDM_plain1, error_OLS_SGDM_plain1, label="Normal")
plt.plot(iterations_OLS_SGDM_adagrad1, error_OLS_SGDM_adagrad1, label="AdaGrad")
plt.plot(iterations_OLS_SGDM_RMSprop1, error_OLS_SGDM_RMSprop1, label="RMSprop")
plt.plot(iterations_OLS_SGDM_ADAM1, error_OLS_SGDM_ADAM1, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
# plt.show()

plt.figure(figsize=(10, 9))
plt.suptitle(fr"[Anal.grad.Ridge]: Iterations = {N}, $\lambda$ = {lamb}, GD Learning rate = {gamma_GD}, mini-batches = {m}, epochs = {n_epochs}")
plt.subplot(2, 2, 1)
plt.title("GD")
plt.plot(iterations_Ridge_GD_plain1, error_Ridge_GD_plain1, label="Normal")
plt.plot(iterations_Ridge_GD_adagrad1, error_Ridge_GD_adagrad1, label="AdaGrad")
plt.plot(iterations_Ridge_GD_RMSprop1, error_Ridge_GD_RMSprop1, label="RMSprop")
plt.plot(iterations_Ridge_GD_ADAM1, error_Ridge_GD_ADAM1, label="ADAM")
plt.legend()
plt.ylabel("MSE")
plt.subplot(2, 2, 2)
plt.title("GDM")
plt.plot(iterations_Ridge_GDM_plain1, error_Ridge_GDM_plain1, label="Normal")
plt.plot(iterations_Ridge_GDM_adagrad1, error_Ridge_GDM_adagrad1, label="AdaGrad")
plt.plot(iterations_Ridge_GDM_RMSprop1, error_Ridge_GDM_RMSprop1, label="RMSprop")
plt.plot(iterations_Ridge_GDM_ADAM1, error_Ridge_GDM_ADAM1, label="ADAM")
plt.legend()
plt.subplot(2, 2, 3)
plt.title("SGD")
plt.plot(iterations_Ridge_SGD_plain1, error_Ridge_SGD_plain1, label="Normal")
plt.plot(iterations_Ridge_SGD_adagrad1, error_Ridge_SGD_adagrad1, label="AdaGrad")
plt.plot(iterations_Ridge_SGD_RMSprop1, error_Ridge_SGD_RMSprop1, label="RMSprop")
plt.plot(iterations_Ridge_SGD_ADAM1, error_Ridge_SGD_ADAM1, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.subplot(2, 2, 4)
plt.title("SGDM")
plt.plot(iterations_Ridge_SGDM_plain1, error_Ridge_SGDM_plain1, label="Normal")
plt.plot(iterations_Ridge_SGDM_adagrad1, error_Ridge_SGDM_adagrad1, label="AdaGrad")
plt.plot(iterations_Ridge_SGDM_RMSprop1, error_Ridge_SGDM_RMSprop1, label="RMSprop")
plt.plot(iterations_Ridge_SGDM_ADAM1, error_Ridge_SGDM_ADAM1, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
# plt.show()




# Tasks week 42, Autograd [for GD, GDM, SGD, SGDM]
def cost_OLS(x, y, theta):
    return np.sum((y - x@theta)**2)


def cost_Ridge(x, y, theta):
    rid1 = np.sum((y - x@theta)**2)
    rid2 = lamb * np.sum(theta[1:]**2)
    return (rid1 + rid2)


beta_autograd = np.random.randn(2, 1)
training_gradient_OLS = grad(cost_OLS, 2)
training_gradient_Ridge = grad(cost_Ridge, 2)


def autograd_GD(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    itera = []
    error = []
    Giter = 0.0
    iter = 0.0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(N):
        itera.append(i)
        iter += 1
        error_ = np.mean((X @ Beta - y) ** 2)
        error.append(error_)

        if method == 0:
            gradients = (1/N)*training_gradient_OLS(X, y, Beta)
        else:
            gradients = (1/N)*training_gradient_Ridge(X, y, Beta)

        if Type == 0:
            Giter += gradients * gradients
            update = gamma_GD*gradients / (delta+np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        elif Type == 1:
            # Now also scaling Giter with rho:
            Giter += (rho * Giter + (1 - rho) * gradients * gradients)
            update = gamma_GD * gradients / (delta + np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        elif Type == 2:
            # Computing moments first
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1 ** iter)
            second_term = second_moment / (1.0 - beta2 ** iter)
            # Scaling with rho the new and the previous results
            update = gamma_GD * first_term / (np.sqrt(second_term) + delta)
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        else:
            update = gamma_GD * gradients
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
    return Beta, error, itera


def autograd_GDM(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    update = 0
    itera = []
    error = []
    Giter = 0.0
    iter = 0.0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(N):
        itera.append(i)
        iter += 1
        error_ = np.mean((X @ Beta - y) ** 2)
        error.append(error_)

        if method == 0:
            gradients = (1/N)*training_gradient_OLS(X, y, Beta)
        else:
            gradients = (1/N)*training_gradient_Ridge(X, y, Beta)

        if Type == 0:
            Giter += gradients * gradients
            update = mom * update + gamma_GD*gradients / (delta+np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update

        elif Type == 1:
            # Now also scaling Giter with rho:
            Giter += (rho * Giter + (1 - rho) * gradients * gradients)
            update = mom * update + gamma_GD * gradients / (delta + np.sqrt(Giter))
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update

        elif Type == 2:
            # Computing moments first
            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
            first_term = first_moment / (1.0 - beta1 ** iter)
            second_term = second_moment / (1.0 - beta2 ** iter)
            # Scaling with rho the new and the previous results
            update = mom * update + gamma_GD * first_term / (np.sqrt(second_term) + delta)
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
        else:
            update = mom * update + gamma_GD * gradients
            dif = (Beta - update) - Beta
            if all(abs(x) <= tol for x in dif):  # Convergence test
                break
            Beta -= update
    return Beta, error, itera


def autograd_SGD(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    k = 0
    itera = []
    error = []
    iter = 0.0
    """first_moment = 0.0
    second_moment = 0.0"""
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        Giter = 0.0
        for i in range(m):
            itera.append(k)
            k += 1
            iter += 1
            error_ = np.mean((X @ Beta - y) ** 2)
            error.append(error_)

            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            if method == 0:
                gradients = (1.0/M)*training_gradient_OLS(xi, yi, Beta)
            else:
                gradients = (1.0/M)*training_gradient_Ridge(xi, yi, Beta)
            gamma = learning_schedule(epoch * m + i)

            if Type == 0:
                Giter += gradients*gradients
                update = gamma*gradients / (delta+np.sqrt(Giter))
                dif = (Beta-update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 1:
                # Now also scaling Giter with rho:
                Giter += (rho * Giter + (1 - rho) * gradients * gradients)
                update = gamma * gradients / (delta + np.sqrt(Giter))
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 2:
                # Computing moments first
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
                first_term = first_moment / (1.0 - beta1 ** iter)
                second_term = second_moment / (1.0 - beta2 ** iter)
                # Scaling with rho the new and the previous results
                update = gamma * first_term / (np.sqrt(second_term) + delta)
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            else:
                update = gamma * gradients
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
    return Beta, error, itera


def autograd_SGDM(beta, method, Type):
    """method == 0 refer to OLS, any other number refer to Ridge"""
    """Type: 0 = AdaGrad, 1 = RMSprop, 2 = ADAM, any number = Plain"""
    Beta = beta.copy()
    k = 0
    itera = []
    error = []
    update = 0
    iter = 0.0
    """first_moment = 0.0
    second_moment = 0.0"""
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        Giter = 0.0
        for i in range(m):
            iter += 1
            itera.append(k)
            k += 1
            error_ = np.mean((X @ Beta - y) ** 2)
            error.append(error_)

            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            if method == 0:
                gradients = (1/M)*training_gradient_OLS(xi, yi, Beta)
            else:
                gradients = (1/M)*training_gradient_Ridge(xi, yi, Beta)
            gamma = learning_schedule(epoch * m + i)

            if Type == 0:
                Giter += gradients*gradients
                update = mom * update + gamma*gradients / (delta+np.sqrt(Giter))
                dif = (Beta-update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 1:
                # Now also scaling Giter with rho:
                Giter += (rho * Giter + (1 - rho) * gradients * gradients)
                update = mom * update + gamma * gradients / (delta + np.sqrt(Giter))
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            elif Type == 2:
                # Computing moments first
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
                first_term = first_moment / (1.0 - beta1 ** iter)
                second_term = second_moment / (1.0 - beta2 ** iter)
                # Scaling with rho the new and the previous results
                update = mom * update + gamma * first_term / (np.sqrt(second_term) + delta)
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
            else:
                update = mom * update + gamma * gradients
                dif = (Beta - update) - Beta
                if all(abs(x) <= tol for x in dif):  # Convergence test
                    break
                Beta -= update
    return Beta, error, itera


# Plain (Normal):
error_OLS_GD_plain, iterations_OLS_GD_plain = autograd_GD(beta_autograd, 0, 3)[1:3]
error_OLS_GDM_plain, iterations_OLS_GDM_plain = autograd_GDM(beta_autograd, 0, 3)[1:3]

error_OLS_SGD_plain, iterations_OLS_SGD_plain = autograd_SGD(beta_autograd, 0, 3)[1:3]
error_OLS_SGDM_plain, iterations_OLS_SGDM_plain = autograd_SGDM(beta_autograd, 0, 3)[1:3]

error_Ridge_GD_plain, iterations_Ridge_GD_plain = autograd_GD(beta_autograd, 1, 3)[1:3]
error_Ridge_GDM_plain, iterations_Ridge_GDM_plain = autograd_GDM(beta_autograd, 1, 3)[1:3]

error_Ridge_SGD_plain, iterations_Ridge_SGD_plain = autograd_SGD(beta_autograd, 1, 3)[1:3]
error_Ridge_SGDM_plain, iterations_Ridge_SGDM_plain = autograd_SGDM(beta_autograd, 1, 3)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE with AutoGrad")
plt.plot(iterations_OLS_GD_plain, error_OLS_GD_plain, label="OLS_GD")
plt.plot(iterations_OLS_GDM_plain, error_OLS_GDM_plain, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_plain, error_OLS_SGD_plain, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_plain, error_OLS_SGDM_plain, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_plain, error_Ridge_GD_plain, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_plain, error_Ridge_GDM_plain, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_plain, error_Ridge_SGD_plain, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_plain, error_Ridge_SGDM_plain, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# AdaGrad:
error_OLS_GD_adagrad, iterations_OLS_GD_adagrad = autograd_GD(beta_autograd, 0, 0)[1:3]
error_OLS_GDM_adagrad, iterations_OLS_GDM_adagrad = autograd_GDM(beta_autograd, 0, 0)[1:3]

error_OLS_SGD_adagrad, iterations_OLS_SGD_adagrad = autograd_SGD(beta_autograd, 0, 0)[1:3]
error_OLS_SGDM_adagrad, iterations_OLS_SGDM_adagrad = autograd_SGDM(beta_autograd, 0, 0)[1:3]

error_Ridge_GD_adagrad, iterations_Ridge_GD_adagrad = autograd_GD(beta_autograd, 1, 0)[1:3]
error_Ridge_GDM_adagrad, iterations_Ridge_GDM_adagrad = autograd_GDM(beta_autograd, 1, 0)[1:3]

error_Ridge_SGD_adagrad, iterations_Ridge_SGD_adagrad = autograd_SGD(beta_autograd, 1, 0)[1:3]
error_Ridge_SGDM_adagrad, iterations_Ridge_SGDM_adagrad = autograd_SGDM(beta_autograd, 1, 0)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using AdaGrad (with AutoGrad)")
plt.plot(iterations_OLS_GD_adagrad, error_OLS_GD_adagrad, label="OLS_GD")
plt.plot(iterations_OLS_GDM_adagrad, error_OLS_GDM_adagrad, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_adagrad, error_OLS_SGD_adagrad, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_adagrad, error_OLS_SGDM_adagrad, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_adagrad, error_Ridge_GD_adagrad, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_adagrad, error_Ridge_GDM_adagrad, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_adagrad, error_Ridge_SGD_adagrad, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_adagrad, error_Ridge_SGDM_adagrad, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# RMSprop:
error_OLS_GD_RMSprop, iterations_OLS_GD_RMSprop = autograd_GD(beta_autograd, 0, 1)[1:3]
error_OLS_GDM_RMSprop, iterations_OLS_GDM_RMSprop = autograd_GDM(beta_autograd, 0, 1)[1:3]

error_OLS_SGD_RMSprop, iterations_OLS_SGD_RMSprop = autograd_SGD(beta_autograd, 0, 1)[1:3]
error_OLS_SGDM_RMSprop, iterations_OLS_SGDM_RMSprop = autograd_SGDM(beta_autograd, 0, 1)[1:3]

error_Ridge_GD_RMSprop, iterations_Ridge_GD_RMSprop = autograd_GD(beta_autograd, 1, 1)[1:3]
error_Ridge_GDM_RMSprop, iterations_Ridge_GDM_RMSprop = autograd_GDM(beta_autograd, 1, 1)[1:3]

error_Ridge_SGD_RMSprop, iterations_Ridge_SGD_RMSprop = autograd_SGD(beta_autograd, 1, 1)[1:3]
error_Ridge_SGDM_RMSprop, iterations_Ridge_SGDM_RMSprop = autograd_SGDM(beta_autograd, 1, 1)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using RMSprop (with AutoGrad)")
plt.plot(iterations_OLS_GD_RMSprop, error_OLS_GD_RMSprop, label="OLS_GD")
plt.plot(iterations_OLS_GDM_RMSprop, error_OLS_GDM_RMSprop, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_RMSprop, error_OLS_SGD_RMSprop, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_RMSprop, error_OLS_SGDM_RMSprop, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_RMSprop, error_Ridge_GD_RMSprop, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_RMSprop, error_Ridge_GDM_RMSprop, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_RMSprop, error_Ridge_SGD_RMSprop, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_RMSprop, error_Ridge_SGDM_RMSprop, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# ADAM:
error_OLS_GD_ADAM, iterations_OLS_GD_ADAM = autograd_GD(beta_autograd, 0, 2)[1:3]
error_OLS_GDM_ADAM, iterations_OLS_GDM_ADAM = autograd_GDM(beta_autograd, 0, 2)[1:3]

error_OLS_SGD_ADAM, iterations_OLS_SGD_ADAM = autograd_SGD(beta_autograd, 0, 2)[1:3]
error_OLS_SGDM_ADAM, iterations_OLS_SGDM_ADAM = autograd_SGDM(beta_autograd, 0, 2)[1:3]

error_Ridge_GD_ADAM, iterations_Ridge_GD_ADAM = autograd_GD(beta_autograd, 1, 2)[1:3]
error_Ridge_GDM_ADAM, iterations_Ridge_GDM_ADAM = autograd_GDM(beta_autograd, 1, 2)[1:3]

error_Ridge_SGD_ADAM, iterations_Ridge_SGD_ADAM = autograd_SGD(beta_autograd, 1, 2)[1:3]
error_Ridge_SGDM_ADAM, iterations_Ridge_SGDM_ADAM = autograd_SGDM(beta_autograd, 1, 2)[1:3]

"""plt.figure(figsize=(8, 7))
plt.title("MSE using ADAM (with AutoGrad)")
plt.plot(iterations_OLS_GD_ADAM, error_OLS_GD_ADAM, label="OLS_GD")
plt.plot(iterations_OLS_GDM_ADAM, error_OLS_GDM_ADAM, label="OLS_GDM")
plt.plot(iterations_OLS_SGD_ADAM, error_OLS_SGD_ADAM, label="OLS_SGD")
plt.plot(iterations_OLS_SGDM_ADAM, error_OLS_SGDM_ADAM, label="OLS_SGDM")
plt.plot(iterations_Ridge_GD_ADAM, error_Ridge_GD_ADAM, label="Ridge_GD")
plt.plot(iterations_Ridge_GDM_ADAM, error_Ridge_GDM_ADAM, label="Ridge_GDM")
plt.plot(iterations_Ridge_SGD_ADAM, error_Ridge_SGD_ADAM, label="Ridge_SGD")
plt.plot(iterations_Ridge_SGDM_ADAM, error_Ridge_SGDM_ADAM, label="Ridge_SGDM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()"""


# EXTRA:
# Here we plot the relationship between the MSE of the different methods using optimal parameters
plt.figure(figsize=(10, 9))
plt.suptitle(fr"[AutoGrad.OLS]: Iterations = {N}, GD learning rate = {gamma_GD}, mini-batches = {m}, epochs = {n_epochs}")
plt.subplot(2, 2, 1)
plt.title("GD")
plt.plot(iterations_OLS_GD_plain, error_OLS_GD_plain, label="Normal")
plt.plot(iterations_OLS_GD_adagrad, error_OLS_GD_adagrad, label="AdaGrad")
plt.plot(iterations_OLS_GD_RMSprop, error_OLS_GD_RMSprop, label="RMSprop")
plt.plot(iterations_OLS_GD_ADAM, error_OLS_GD_ADAM, label="ADAM")
plt.legend()
plt.ylabel("MSE")
plt.subplot(2, 2, 2)
plt.title("GDM")
plt.plot(iterations_OLS_GDM_plain, error_OLS_GDM_plain, label="Normal")
plt.plot(iterations_OLS_GDM_adagrad, error_OLS_GDM_adagrad, label="AdaGrad")
plt.plot(iterations_OLS_GDM_RMSprop, error_OLS_GDM_RMSprop, label="RMSprop")
plt.plot(iterations_OLS_GDM_ADAM, error_OLS_GDM_ADAM, label="ADAM")
plt.legend()
plt.subplot(2, 2, 3)
plt.title("SGD")
plt.plot(iterations_OLS_SGD_plain, error_OLS_SGD_plain, label="Normal")
plt.plot(iterations_OLS_SGD_adagrad, error_OLS_SGD_adagrad, label="AdaGrad")
plt.plot(iterations_OLS_SGD_RMSprop, error_OLS_SGD_RMSprop, label="RMSprop")
plt.plot(iterations_OLS_SGD_ADAM, error_OLS_SGD_ADAM, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.subplot(2, 2, 4)
plt.title("SGDM")
plt.plot(iterations_OLS_SGDM_plain, error_OLS_SGDM_plain, label="Normal")
plt.plot(iterations_OLS_SGDM_adagrad, error_OLS_SGDM_adagrad, label="AdaGrad")
plt.plot(iterations_OLS_SGDM_RMSprop, error_OLS_SGDM_RMSprop, label="RMSprop")
plt.plot(iterations_OLS_SGDM_ADAM, error_OLS_SGDM_ADAM, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
# plt.show()

plt.figure(figsize=(10, 9))
plt.suptitle(fr"[AutoGrad.Ridge]: Iterations = {N}, $\lambda$ = {lamb}, GD Learning rate = {gamma_GD}, mini-batches = {m}, epochs = {n_epochs}")
plt.subplot(2, 2, 1)
plt.title("GD")
plt.plot(iterations_Ridge_GD_plain, error_Ridge_GD_plain, label="Normal")
plt.plot(iterations_Ridge_GD_adagrad, error_Ridge_GD_adagrad, label="AdaGrad")
plt.plot(iterations_Ridge_GD_RMSprop, error_Ridge_GD_RMSprop, label="RMSprop")
plt.plot(iterations_Ridge_GD_ADAM, error_Ridge_GD_ADAM, label="ADAM")
plt.legend()
plt.ylabel("MSE")
plt.subplot(2, 2, 2)
plt.title("GDM")
plt.plot(iterations_Ridge_GDM_plain, error_Ridge_GDM_plain, label="Normal")
plt.plot(iterations_Ridge_GDM_adagrad, error_Ridge_GDM_adagrad, label="AdaGrad")
plt.plot(iterations_Ridge_GDM_RMSprop, error_Ridge_GDM_RMSprop, label="RMSprop")
plt.plot(iterations_Ridge_GDM_ADAM, error_Ridge_GDM_ADAM, label="ADAM")
plt.legend()
plt.subplot(2, 2, 3)
plt.title("SGD")
plt.plot(iterations_Ridge_SGD_plain, error_Ridge_SGD_plain, label="Normal")
plt.plot(iterations_Ridge_SGD_adagrad, error_Ridge_SGD_adagrad, label="AdaGrad")
plt.plot(iterations_Ridge_SGD_RMSprop, error_Ridge_SGD_RMSprop, label="RMSprop")
plt.plot(iterations_Ridge_SGD_ADAM, error_Ridge_SGD_ADAM, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.subplot(2, 2, 4)
plt.title("SGDM")
plt.plot(iterations_Ridge_SGDM_plain, error_Ridge_SGDM_plain, label="Normal")
plt.plot(iterations_Ridge_SGDM_adagrad, error_Ridge_SGDM_adagrad, label="AdaGrad")
plt.plot(iterations_Ridge_SGDM_RMSprop, error_Ridge_SGDM_RMSprop, label="RMSprop")
plt.plot(iterations_Ridge_SGDM_ADAM, error_Ridge_SGDM_ADAM, label="ADAM")
plt.legend()
plt.xlabel("Iterations")
plt.show()
