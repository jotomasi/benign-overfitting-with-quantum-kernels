

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg


#=================================
# Kernel functions
#=================================
def local_kernel(X1, X2, c):
    diff = X1[:, None, :] - X2[None, :, :]
    return np.prod(np.cos(0.5 * c * diff) ** 2, axis=-1)

def local_global_kernel(K, rho, q):
    return K + rho * (K ** q)

def bandwidth(d, c0=1.0, alpha=0.5):
    # bandwith to get "smooth" enough local kernel , in function of d ( canatar bandwidth enable generalization)
    return c0 * d**(-alpha)


#=================================
# Input sampling utilities
#=================================

def domain_half_width(c, n_periods=3):
    # will help to find L such that [-L,L] contains  at least n_periods period 
    # ensure non unicity of r such that kappa(r) =1 where k(x,z) = kappa(x-z)
    tau = 2 * np.pi / c
    return n_periods * tau

def sample_X(n, d, c, L= None , seed=0):
    # generate data uniformly on the hypercube [-L,L]^d, with L choosen to have several periods of the local kernel function
    rng = np.random.default_rng(seed)
    if L == None:
        L = domain_half_width(c,3)
    return rng.uniform(-L, L, size=(n, d))

def add_noise(y,sigma, seed=0):
    # add gaussian noise to labels 
    rng = np.random.default_rng(seed)
    return y + sigma * rng.normal(size=len(y))


#=================================
# Regression
#=================================

def compute_alpha(K, y, rho=0):
    # compute the alpha solving  rho_ridge rgression with kernel matrix K and  labels y
    
    if rho !=0:
        n = K.shape[0]
        Krho = K + rho * np.eye(n)
    else:
        Krho = K
    
    try: # Krho invertible
        return np.linalg.solve(Krho, y)
    except np.linalg.LinAlgError: # Krho non invertible
        return np.linalg.pinv(Krho) @ y
    
def compute_alphafig1(K, y, rho=0,eps = 0):
    # compute the alpha solving  rho_ridge rgression with kernel matrix K and  labels y
    
    if rho !=0:
        n = K.shape[0]
        Krho = K + rho * np.eye(n)
    else:
        Krho = K + eps * np.eye(K.shape[0])
    
    try: # Krho invertible
        return np.linalg.solve(Krho, y)
    except np.linalg.LinAlgError: # Krho non invertible
        return np.linalg.pinv(Krho) @ y
    

def compute_predictions(K_eval,alpha_K):
     # evaluate predictor
     return K_eval @ alpha_K

def compute_mse(y_pred, y_true):
    """Compute Mean Squared Error"""
    return np.mean((y_pred - y_true) ** 2)


#===============================
# Targets functions
#===============================

def target_lgkfct(d, c=None,L=None,rho=1,q=50,nz =50, target_std=1, 
                             coef_seed=0,n_eval = 2000):
    rng = np.random.default_rng(coef_seed)
    
    # set bandwidth parameter
    if c is None:
        c = bandwidth(d=d)
    
    # set the half width of the input domain for one dimension
    if L == None:
        L = domain_half_width(c)

    # generate anchors    
    Z = rng.uniform(-L, L, size=(nz, d))

    coeffs =  rng.normal(size=nz)

    X_eval = rng.uniform(-L, L, size=(n_eval, d))

    # compute local kernel between X and Z
    Kxz_local = local_kernel(X_eval, Z, c)

    # build full LG kernel (teacher kernel)
    if rho !=0 :
        Kxz_LG = local_global_kernel(Kxz_local, rho, q)
    else :
        Kxz_LG = Kxz_local

    fxeval = Kxz_LG @ coeffs

    sigma = np.std(fxeval)
    
    mu =  np.mean(fxeval*target_std/sigma)

    coeffs = coeffs * target_std/sigma

    if rho !=0:
        def f(X):
            K_loc = local_kernel(X, Z, c)
            K_lg = local_global_kernel(K_loc, rho, q)
            return K_lg @ coeffs - mu 
    else:
        def f(X):
            K_loc = local_kernel(X, Z, c)
            return K_loc @ coeffs - mu 

    f.coeffs = coeffs
    f.mu = np.mean(f(X_eval))
    f.std = np.std(f(X_eval))
    f.var = np.var(f(X_eval))
    f.L = L
    f.c = c
    f.nz = nz
    f.rho = rho 
    f.q = q
    f.seed = coef_seed
    return f


def target_trigo_sum_general(
    d,
    c=None,
    L=None,
    freqs=None,
    target_std=1,
    coef_seed=0,
    n_periods=3,
):
    """
    Create centered trigonometric target with exact target_std.
    """

    rng = np.random.default_rng(coef_seed)

    # -----------------------------
    # Bandwidth / domain
    # -----------------------------
    if c is None:
        c = bandwidth(d=d)

    if freqs is None:
        freqs = [c]

    k = len(freqs)

    if L is None:
        L = domain_half_width(c, n_periods)

    # -----------------------------
    # Random coefficients
    # -----------------------------
    cos_coeff = rng.normal(size=(d, k))
    sin_coeff = rng.normal(size=(d, k))

    # -----------------------------
    # Variance factors
    # -----------------------------
    var_factors_cos = np.zeros(k)
    var_factors_sin = np.zeros(k)
    mean_factors_cos = np.zeros(k)

    for i, w in enumerate(freqs):

        wL = w * L

        if abs(wL) < 1e-12:
            var_factors_cos[i] = 0.0
            var_factors_sin[i] = 0.0
            mean_factors_cos[i] = 1.0
        else:
            # variance terms
            var_factors_cos[i] = (
                0.5
                + np.sin(2*wL)/(8*wL)
                - (np.sin(wL)/(wL))**2
            )

            var_factors_sin[i] = (
                0.5
                - np.sin(2*wL)/(8*wL)
            )

            # mean term
            mean_factors_cos[i] = np.sin(wL)/(wL)

    # -----------------------------
    # Exact total variance
    # -----------------------------
    total_variance = 0.0

    for i in range(k):
        cos_norm_sq = np.sum(cos_coeff[:, i]**2)
        sin_norm_sq = np.sum(sin_coeff[:, i]**2)

        total_variance += (
            cos_norm_sq * var_factors_cos[i]
            + sin_norm_sq * var_factors_sin[i]
        )

    # Scale to target_std
    s = target_std / np.sqrt(total_variance) if total_variance > 0 else 1.0

    cos_coeff *= s
    sin_coeff *= s

    # -----------------------------
    # Exact mean computation
    # -----------------------------
    mu = 0.0
    for i in range(k):
        mu += mean_factors_cos[i] * np.sum(cos_coeff[:, i])

    # -----------------------------
    # Target function
    # -----------------------------
    def f(X):
        y = np.zeros(len(X))
        for i, w in enumerate(freqs):
            y += np.cos(w * X) @ cos_coeff[:, i]
            y += np.sin(w * X) @ sin_coeff[:, i]
        return y - mu   # centered

    # Metadata
    f.coefficients = (cos_coeff, sin_coeff)
    f.frequencies = freqs
    f.L = L
    f.metadata = {
        "total_variance_unscaled": total_variance,
        "scaling_factor": s,
        "mean_removed": mu,
    }

    return f