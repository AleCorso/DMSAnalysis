import numpy as np

def train_test_split(delta, G, test_size = 0.5):
    Mtot, N = delta.shape
    idx = np.random.permutation(Mtot)      
    delta = delta[idx]
    G = G[idx]
    Mtest =  int(Mtot*test_size)
    delta_test = delta[:Mtest, :]
    delta_train = delta[Mtest:,:]
    G_test = G[:Mtest]
    G_train = G[Mtest:] 
    return delta_train, delta_test, G_train, G_test

def ridge_regression_stable(delta, G, lam):
    n, p = delta.shape

    delta_mean = np.mean(delta, axis=0)
    X = delta - delta_mean[None, :]

    G_mean = np.mean(G)
    y = G - G_mean

    # SVD of X (not X.T @ X) — avoids squaring the condition number
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Ridge solution: V @ diag(s / (s² + n*lam)) @ U.T @ y
    d = s / (s**2 + n * lam)
    g_hat = Vt.T @ (d * (U.T @ y))

    return g_hat, delta_mean, G_mean

def ridge_regression(delta, G, lam):

    n, p = delta.shape

    delta_mean = np.mean(delta, axis = 0)

    X = delta - delta_mean[None,:]

    G_mean = np.mean(G)

    y = G - G_mean
    
    g_hat = np.linalg.solve(X.T@X + n * lam * np.eye(p), X.T @ y)

    return g_hat, delta_mean, G_mean