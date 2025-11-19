import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
from scipy import stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. SMOOTH TRANSITION REGRESSION (STR) MODEL
# ==============================================================================

@dataclass
class StrResult:
    params: pd.Series
    se: pd.Series
    tvals: pd.Series
    pvals: pd.Series
    sse: float
    rmse: float
    aic: float
    bic: float
    nobs: int
    dof: int
    success: bool
    message: str
    G: pd.Series
    y_fit: pd.Series
    resid: pd.Series
    cov_hac: np.ndarray

class STRModel:
    """
    Specific STR Model for Exchange Rates.
    
    Regime 0 (G=0): y_t = alpha + beta_c * r_s,t-1
    Regime 1 (G=1): y_t = (alpha + alpha1) - beta_f * eta_t-1
    
    Mixing Equation:
    y_t = alpha + beta_c * r_s + G * (alpha1 - beta_f * eta - beta_c * r_s)
    """
    
    def __init__(self, df: pd.DataFrame, z_col: str):
        self.y = df["y"].values
        self.z = df[z_col].values
        self.rs = df["rs_lag1"].values
        self.eta = df["eta_lag1"].values
        self.index = df.index
        self.z_name = z_col
        self.nobs = len(self.y)
        
        # Robust standard deviation for scaling gamma
        self.z_std = max(float(np.std(self.z)), 1e-8)

    def _compute_G(self, gamma: float, c: float) -> np.ndarray:
        return expit((gamma / self.z_std) * (self.z - c))

    def _design_matrix(self, gamma: float, c: float) -> np.ndarray:
        G = self._compute_G(gamma, c)
        # Columns corresponding to: [const, beta_c, beta_f, const1]
        # Derived from: alpha + beta_c*rs + G*alpha1 - G*beta_f*eta - G*beta_c*rs
        # Rearranged: alpha*1 + beta_c*(1-G)*rs + beta_f*(G*-eta) + alpha1*G
        
        # Note: Your snippet used specific column ordering:
        # X = [1, (1-G)*rs, G*(-eta), G] -> [const, beta_c, beta_f, const1]
        X = np.column_stack([
            np.ones(self.nobs),
            (1 - G) * self.rs,
            G * (-self.eta),
            G
        ])
        return X, G

    def grid_search(self, gamma_grid: np.ndarray, c_grid: np.ndarray) -> Dict[str, float]:
        """
        Grid search for initial parameters using OLS on the linearized design matrix.
        """
        best_sse = np.inf
        best_params = {}

        for g in gamma_grid:
            for c in c_grid:
                X, _ = self._design_matrix(g, c)
                
                # Fast OLS
                try:
                    beta, resid_sum, rank, s = np.linalg.lstsq(X, self.y, rcond=None)
                    if len(resid_sum) > 0:
                        sse = resid_sum[0]
                    else:
                        sse = np.sum((self.y - X @ beta)**2)
                except np.linalg.LinAlgError:
                    continue

                if sse < best_sse:
                    best_sse = sse
                    # beta order: [const, beta_c, beta_f, const1]
                    best_params = {
                        "gamma": float(g),
                        "c": float(c),
                        "const": beta[0],
                        "beta_c": beta[1],
                        "beta_f": beta[2],
                        "const1": beta[3],
                        "sse": sse
                    }
        
        if best_sse == np.inf:
             raise RuntimeError(f"STR Grid search failed for {self.z_name}")
             
        return best_params

    def fit(self, start_params: Dict[str, float], hac_lags: int = 4) -> StrResult:
        """
        Non-linear Least Squares Estimation.
        """
        # Initial vector: [const, beta_c, beta_f, gamma, c, const1]
        x0 = np.array([
            start_params.get("const", 0.0),
            start_params["beta_c"],
            start_params["beta_f"],
            max(start_params["gamma"], 0.1),
            start_params["c"],
            start_params.get("const1", 0.0)
        ])
        
        # Bounds: gamma > 0
        lb = np.array([-np.inf, -np.inf, -np.inf, 1e-4, -np.inf, -np.inf])
        ub = np.array([ np.inf,  np.inf,  np.inf, np.inf,  np.inf,  np.inf])

        def resid_fun(theta):
            # Unpack
            const, bc, bf, g, c_val, const1 = theta
            G = self._compute_G(g, c_val)
            
            # Model: y = const + bc*rs + G*(const1 - bf*eta - bc*rs)
            yhat = const + (bc * self.rs) + G * (const1 - (bf * self.eta) - (bc * self.rs))
            return self.y - yhat

        res = least_squares(resid_fun, x0, bounds=(lb, ub), method='trf', loss='linear')

        # --- Post-Estimation Statistics ---
        theta = res.x
        resid = res.fun
        sse = np.sum(resid**2)
        dof = max(self.nobs - len(theta), 1)
        
        # Recompute G at optimum for results
        G_opt = self._compute_G(theta[3], theta[4])
        yhat = self.y - resid

        # Covariance
        cov_hac = _newey_west_cov(res.jac, resid, hac_lags)
        se = np.sqrt(np.diag(cov_hac))
        
        tvals = theta / se
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), dof))

        # Formatting results
        param_names = ["const", "beta_c", "beta_f", "gamma", "c", "const1"]
        
        return StrResult(
            params=pd.Series(theta, index=param_names),
            se=pd.Series(se, index=param_names),
            tvals=pd.Series(tvals, index=param_names),
            pvals=pd.Series(pvals, index=param_names),
            sse=sse,
            rmse=np.sqrt(sse / self.nobs),
            aic=self.nobs * np.log(sse / self.nobs) + 2 * len(theta),
            bic=self.nobs * np.log(sse / self.nobs) + np.log(self.nobs) * len(theta),
            nobs=self.nobs,
            dof=dof,
            success=res.success,
            message=res.message,
            G=pd.Series(G_opt, index=self.index, name="G"),
            y_fit=pd.Series(yhat, index=self.index, name="y_hat"),
            resid=pd.Series(resid, index=self.index, name="resid"),
            cov_hac=cov_hac
        )


# ==============================================================================
# 2. BEHAVIORAL UIP (BUIP) MODEL
# ==============================================================================

@dataclass
class BuipResult:
    params: pd.Series
    se: pd.Series
    tvals: pd.Series
    pvals: pd.Series
    sse: float
    rmse: float
    aic: float
    bic: float
    nobs: int
    success: bool
    message: str
    omega: pd.Series
    U_f: pd.Series
    U_c: pd.Series
    y_fit: pd.Series
    resid: pd.Series
    cov_hac: np.ndarray

class BUIPModel:
    """
    Behavioral Model where regime weights depend on the utility difference 
    between Fundamentalists and Chartists.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.y = df["y"].values  # UIP-corrected return
        self.nobs = len(self.y)
        self.index = df.index
        
        # Pre-extract numpy arrays for speed in the optimization loop
        self.eta_lag1 = df["eta_lag1"].values
        self.r_lag1 = df["r_lag1"].values
        self.s_t1 = df["s_t1"].values
        self.s_t2 = df["s_t2"].values
        self.eta_t2 = df["eta_t2"].values
        self.r_t2 = df["r_t2"].values
        self.i_for_t1 = df["i_for_t1"].values
        self.i_dom_t1 = df["i_dom_t1"].values

    def _compute_residuals_and_cache(self, theta: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Core logic for BUIP. Calculates utility dynamically.
        theta = [beta_f, beta_c, gamma, c, const, const1]
        """
        beta_f, beta_c, gamma, c, const, const1 = theta
        
        # 1. Expectations (formed at t-1 using information from t-2)
        # Fundamentalist: Expects reversion to fundamental (s - beta_f * eta)
        E_s_f = self.s_t2 - beta_f * self.eta_t2
        # Chartist: Expects trend continuation (s + beta_c * r)
        E_s_c = self.s_t2 + beta_c * self.r_t2
        
        # 2. Expected Change (for sign of profit)
        E_ds_f = -beta_f * self.eta_t2
        E_ds_c = beta_c * self.r_t2
        
        # 3. Profits (psi)
        # Realized return from carry trade: (r_lag1 + i*_t-1 - i_t-1)
        # Note: The user snippet uses 'profit_ret' which seems to represent the realized movement + differential
        profit_ret = self.r_lag1 + self.i_for_t1 - self.i_dom_t1
        
        psi_f = profit_ret * np.sign(E_ds_f)
        psi_c = profit_ret * np.sign(E_ds_c)
        
        # 4. Risk (sigma squared) - Squared prediction error
        sig2_f = (E_s_f - self.s_t1)**2
        sig2_c = (E_s_c - self.s_t1)**2
        
        # 5. Utility
        U_f = psi_f - sig2_f
        U_c = psi_c - sig2_c
        d = U_f - U_c
        
        # 6. Weighting (Omega)
        # Scale d to make gamma scale-invariant
        scale_d = np.std(d) if np.std(d) > 1e-12 else 1.0
        omega = expit((gamma / scale_d) * (d - c))
        
        # 7. Model Prediction
        # y = const + beta_c*r + omega*(const1 - beta_f*eta - beta_c*r)
        yhat = const + (beta_c * self.r_lag1) + omega * (const1 - (beta_f * self.eta_lag1) - (beta_c * self.r_lag1))
        
        return self.y - yhat, {"omega": omega, "U_f": U_f, "U_c": U_c, "yhat": yhat}

    def fit_multistart(self, n_starts: int = 20, hac_lags: int = 4) -> BuipResult:
        """
        Optimization with multiple starting points.
        """
        # Bounds: beta_f, beta_c, gamma > 0 usually expected, but let's keep generic
        # gamma must be > 0
        lb = np.array([-np.inf, -np.inf, 1e-4, -np.inf, -np.inf, -np.inf])
        ub = np.array([ np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf])
        
        best_sse = np.inf
        best_res = None
        best_cache = None
        
        # Generate random starts
        # [beta_f, beta_c, gamma, c, const, const1]
        np.random.seed(42)
        starts = []
        for _ in range(n_starts):
            s = [
                np.random.uniform(0, 2),    # beta_f
                np.random.uniform(0, 2),    # beta_c
                np.random.uniform(1, 20),   # gamma
                np.random.uniform(-1, 1),   # c (approx)
                np.random.normal(0, 0.1),   # const
                np.random.normal(0, 0.1)    # const1
            ]
            starts.append(np.array(s))
            
        for x0 in starts:
            try:
                res = least_squares(
                    lambda theta: self._compute_residuals_and_cache(theta)[0],
                    x0, bounds=(lb, ub), method='trf'
                )
                
                sse = np.sum(res.fun**2)
                if sse < best_sse:
                    best_sse = sse
                    best_res = res
                    # Re-run to get cache
                    _, best_cache = self._compute_residuals_and_cache(res.x)
            except Exception as e:
                continue
                
        if best_res is None:
            raise RuntimeError("BUIP Multistart Optimization failed.")
            
        # --- Statistics ---
        theta = best_res.x
        resid = best_res.fun
        dof = max(self.nobs - len(theta), 1)
        
        cov_hac = _newey_west_cov(best_res.jac, resid, hac_lags)
        se = np.sqrt(np.diag(cov_hac))
        tvals = theta / se
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), dof))
        
        param_names = ["beta_f", "beta_c", "gamma", "c", "const", "const1"]
        
        return BuipResult(
            params=pd.Series(theta, index=param_names),
            se=pd.Series(se, index=param_names),
            tvals=pd.Series(tvals, index=param_names),
            pvals=pd.Series(pvals, index=param_names),
            sse=best_sse,
            rmse=np.sqrt(best_sse / self.nobs),
            aic=self.nobs * np.log(best_sse / self.nobs) + 2 * len(theta),
            bic=self.nobs * np.log(best_sse / self.nobs) + np.log(self.nobs) * len(theta),
            nobs=self.nobs,
            success=best_res.success,
            message=best_res.message,
            omega=pd.Series(best_cache["omega"], index=self.index, name="omega"),
            U_f=pd.Series(best_cache["U_f"], index=self.index, name="U_f"),
            U_c=pd.Series(best_cache["U_c"], index=self.index, name="U_c"),
            y_fit=pd.Series(best_cache["yhat"], index=self.index, name="y_hat"),
            resid=pd.Series(resid, index=self.index, name="resid"),
            cov_hac=cov_hac
        )

# ==============================================================================
# 3. SHARED HELPERS
# ==============================================================================

def _newey_west_cov(J: np.ndarray, resid: np.ndarray, maxlags: int = 4) -> np.ndarray:
    """Computes Newey-West HAC covariance matrix."""
    n, k = J.shape
    S = J * resid[:, None]
    
    # Gamma 0
    Omega = (S.T @ S) / n
    
    # Gamma k
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        S_f = S[lag:, :]
        S_b = S[:-lag, :]
        Gamma_l = (S_f.T @ S_b) / n
        Omega += weight * (Gamma_l + Gamma_l.T)
        
    H = (J.T @ J) / n
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)
        
    # Cov = 1/n * H^-1 * Omega * H^-1
    return (1.0 / n) * (H_inv @ Omega @ H_inv)
