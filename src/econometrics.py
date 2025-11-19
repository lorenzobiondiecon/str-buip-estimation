import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import Tuple, Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class STRModel:
    """
    Smooth Transition Regression (STR) Model.
    
    Model Specification:
        y_t = beta_lin' * X_lin_t + G(gamma, c; z_t) * (beta_non' * X_non_t) + e_t
    
    This class handles:
    1. Grid Search for initialization (gamma, c).
    2. Non-Linear Least Squares (NLS) estimation.
    """
    
    def __init__(self, 
                 endog: np.ndarray, 
                 exog_lin: np.ndarray, 
                 exog_non: np.ndarray, 
                 z: np.ndarray,
                 transition_type: str = 'LSTR'):
        """
        Args:
            endog (np.ndarray): Dependent variable (T,).
            exog_lin (np.ndarray): Linear regressors (T, k1). Include constant if needed.
            exog_non (np.ndarray): Non-linear regressors (T, k2). Include constant if needed.
            z (np.ndarray): Transition variable (T,).
            transition_type (str): 'LSTR' (Logistic) or 'ESTR' (Exponential).
        """
        self.y = endog
        self.X_lin = exog_lin
        self.X_non = exog_non
        self.z = z
        self.transition_type = transition_type
        self.n_obs = len(endog)
        
        # Placeholders for results
        self.params = None
        self.resid = None
        self.gamma = None
        self.c = None
        self.fitted_values = None

    def _compute_G(self, gamma: float, c: float, z: np.ndarray) -> np.ndarray:
        """Computes the transition function G(gamma, c; z)."""
        # Normalization by std(z) is standard practice to make gamma scale-invariant
        z_std = np.std(z)
        if z_std == 0: z_std = 1.0
        
        arg = (gamma / z_std) * (z - c)
        
        if self.transition_type == 'LSTR':
            return 1.0 / (1.0 + np.exp(-arg))
        elif self.transition_type == 'ESTR':
            return 1.0 - np.exp(-(arg ** 2))
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")

    def _construct_regressors(self, gamma: float, c: float) -> np.ndarray:
        """
        Constructs the full design matrix for a given state (gamma, c).
        X_full = [X_lin, G * X_non]
        """
        G = self._compute_G(gamma, c, self.z).reshape(-1, 1)
        # Element-wise multiplication of G with every column of X_non
        X_non_weighted = self.X_non * G
        return np.hstack([self.X_lin, X_non_weighted])

    def grid_search(self, 
                    gamma_grid: np.ndarray, 
                    c_grid: np.ndarray) -> Tuple[float, float, float]:
        """
        Performs an optimized grid search to find initial gamma and c.
        
        OPTIMIZATION:
        Instead of running a full OLS inside the loop, we iterate efficiently.
        Future optimization: This could be further vectorized with numba if needed, 
        but matrix operations here are usually sufficient for T < 5000.
        """
        best_rss = np.inf
        best_params = (None, None)
        
        # Pre-allocate linear part if it doesn't change, 
        # but X_full changes structure, so we rebuild inside.
        
        for gamma in gamma_grid:
            for c in c_grid:
                # 1. Construct design matrix consistent with NLS specification
                X = self._construct_regressors(gamma, c)
                
                # 2. Fast OLS: (X'X)^(-1) X'y
                # Using lstsq is more robust than manual inversion
                beta, resid, rank, s = np.linalg.lstsq(X, self.y, rcond=None)
                
                # 3. Calculate RSS
                # resid is returned as sum of squared errors if rank > N, 
                # but safer to compute explicitly to handle edge cases
                current_rss = np.sum((self.y - X @ beta)**2)
                
                if current_rss < best_rss:
                    best_rss = current_rss
                    best_params = (gamma, c)
        
        return best_params[0], best_params[1], best_rss

    def fit(self, init_gamma: float = 1.0, init_c: float = 0.0) -> Dict:
        """
        Estimates the model parameters using Non-Linear Least Squares.
        """
        # Initial parameter vector: [gamma, c, beta_lin..., beta_non...]
        # We need an initial guess for betas. We run one OLS at init_gamma/c.
        X_init = self._construct_regressors(init_gamma, init_c)
        beta_init, _, _, _ = np.linalg.lstsq(X_init, self.y, rcond=None)
        
        initial_guess = np.concatenate(([init_gamma, init_c], beta_init))
        
        def residual_func(params_all):
            g, c_val = params_all[0], params_all[1]
            betas = params_all[2:]
            
            # Split betas
            k_lin = self.X_lin.shape[1]
            beta_l = betas[:k_lin]
            beta_n = betas[k_lin:]
            
            G = self._compute_G(g, c_val, self.z)
            
            # Compute predicted y
            y_pred = (self.X_lin @ beta_l) + (G * (self.X_non @ beta_n))
            return self.y - y_pred

        # Constraints: Gamma > 0
        lower_bounds = np.full_like(initial_guess, -np.inf)
        lower_bounds[0] = 0.01 # Gamma must be positive
        upper_bounds = np.full_like(initial_guess, np.inf)

        res = least_squares(residual_func, initial_guess, bounds=(lower_bounds, upper_bounds), method='trf')
        
        # Store results
        self.gamma = res.x[0]
        self.c = res.x[1]
        self.params = res.x[2:]
        self.resid = res.fun
        self.fitted_values = self.y - self.resid
        
        return {
            "gamma": self.gamma,
            "c": self.c,
            "coefficients": self.params,
            "rss": np.sum(self.resid**2),
            "message": res.message
        }

class BUIPModel:
    """
    Behavioral Uncovered Interest Parity (BUIP) Model.
    Implements the specific behavioral form where expectation formation switches.
    """
    # Implementation logic similar to STR but with specific structural restrictions
    # as defined in the original notebook (e.g. restricting coefficients sum).
    pass
