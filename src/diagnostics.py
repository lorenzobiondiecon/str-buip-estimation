import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def luukkonen_linearity_test(y: np.ndarray, X: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """
    Implements the Luukkonen, Saikkonen, and Terasvirta (1988) linearity test.
    
    Tests H0: Linear Model vs H1: STR Model.
    Uses an auxiliary regression with Taylor expansion terms of the transition function.
    
    Auxiliary Regression:
    u_t = beta0' x_t + beta1' (x_t * z_t) + beta2' (x_t * z_t^2) + beta3' (x_t * z_t^3) + e_t
    """
    n = len(y)
    
    # 1. Estimate linear model (H0)
    model_linear = sm.OLS(y, X).fit()
    u_linear = model_linear.resid
    rss0 = np.sum(u_linear**2)
    
    # 2. Construct Auxiliary Regressors (Z2, Z3, Z4 in paper notation)
    # Interactions: x*z, x*z^2, x*z^3
    # Note: X should usually include a constant. z_t should be the transition variable.
    
    Z_terms = []
    for k in range(1, 4): # Powers 1, 2, 3
        z_pow = (z ** k).reshape(-1, 1)
        Z_terms.append(X * z_pow)
        
    X_aux = np.hstack([X] + Z_terms)
    
    # 3. Auxiliary Regression
    # Regress residuals of linear model on X and Taylor terms
    # Or commonly: Regress y on X and Taylor terms and test joint significance of Taylor terms.
    
    model_aux = sm.OLS(u_linear, X_aux).fit()
    rss1 = np.sum(model_aux.resid**2)
    
    # 4. Compute LM Statistic
    # LM = T * (RSS0 - RSS1) / RSS0
    lm_stat = n * (rss0 - rss1) / rss0
    
    # Degrees of freedom = 3 * number of vars in X (excluding constant if z is constant, but z usually isn't)
    k = X.shape[1]
    df = 3 * k
    
    p_value = 1 - stats.chi2.cdf(lm_stat, df)
    
    return {
        "lm_stat": lm_stat,
        "p_value": p_value,
        "df": df
    }

def newey_west_covariance(resid: np.ndarray, X: np.ndarray, lags: int = 4) -> np.ndarray:
    """
    Computes Newey-West HAC covariance matrix manually for NLS results.
    
    Args:
        resid: Vector of residuals
        X: Jacobian matrix (Gradient of f(x) wrt parameters)
        lags: Max lag length
    """
    n, k = X.shape
    
    # 1. S0 (White)
    S = np.zeros((k, k))
    for t in range(n):
        xt = X[t, :].reshape(-1, 1)
        S += (resid[t]**2) * (xt @ xt.T)
    S = S / n
    
    # 2. Weights (Bartlett Kernel)
    for l in range(1, lags + 1):
        weight = 1 - l / (lags + 1)
        gamma_l = np.zeros((k, k))
        for t in range(l, n):
            xt = X[t, :].reshape(-1, 1)
            xt_minus_l = X[t-l, :].reshape(-1, 1)
            
            # Term: e_t * e_{t-l} * (x_t * x_{t-l}' + x_{t-l} * x_t')
            term = resid[t] * resid[t-l] * (xt @ xt_minus_l.T + xt_minus_l @ xt.T)
            gamma_l += term
        
        S += weight * (gamma_l / n)
        
    # 3. V = (X'X)^-1 * (n*S) * (X'X)^-1 approx
    # For NLS: V = (J'J)^-1 J' Omega J (J'J)^-1
    
    Q = (X.T @ X) / n
    try:
        Q_inv = np.linalg.inv(Q)
        V = (Q_inv @ S @ Q_inv) / n
        return V
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan)
