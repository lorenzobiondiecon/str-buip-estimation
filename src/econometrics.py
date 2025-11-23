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
        Uses exponential gamma grid matching legacy implementation.
        """
        best_sse = np.inf
        best_params = {}

        for g in gamma_grid:
            for c in c_grid:
                X, _ = self._design_matrix(float(g), float(c))
                
                # Fast OLS
                try:
                    beta, residuals, rank, s = np.linalg.lstsq(X, self.y, rcond=None)
                    if len(residuals) > 0:
                        sse = residuals[0]
                    else:
                        resid = self.y - X @ beta
                        sse = float(np.dot(resid, resid))
                except np.linalg.LinAlgError:
                    continue

                if sse < best_sse:
                    best_sse = sse
                    # beta order: [const, beta_c, beta_f, const1]
                    best_params = {
                        "const": float(beta[0]),
                        "beta_c": float(beta[1]),
                        "beta_f": float(beta[2]),
                        "gamma": float(g),
                        "c": float(c),
                        "const1": float(beta[3]),
                        "sse": float(sse)
                    }
        
        if not np.isfinite(best_sse):
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
        
        # Bounds: gamma > 0 (matching legacy 1e-8 lower bound)
        lb = np.array([-np.inf, -np.inf, -np.inf, 1e-8, -np.inf, -np.inf], dtype=float)
        ub = np.array([ np.inf,  np.inf,  np.inf, np.inf,  np.inf,  np.inf], dtype=float)

        def resid_fun(theta):
            # Unpack
            const, bc, bf, g, c_val, const1 = theta
            G = self._compute_G(g, c_val)
            
            # Model: y = const + bc*rs + G*(const1 - bf*eta - bc*rs)
            yhat = const + (bc * self.rs) + G * (const1 - (bf * self.eta) - (bc * self.rs))
            return self.y - yhat

        res = least_squares(
            resid_fun, x0, bounds=(lb, ub), method='trf', loss='linear',
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=20000
        )

        # --- Post-Estimation Statistics ---
        theta = res.x
        resid = res.fun
        sse = float(2.0 * res.cost)  # Match legacy: 2*cost gives SSE
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
# 1b. UNRESTRICTED STR MODEL (NO COEFFICIENT RESTRICTIONS)
# ==============================================================================

class STRModelNoRestrictions:
    """
    Unrestricted STR Model for Exchange Rates.
    
    Allows all parameters to vary freely across regimes:
    y_t = const0 + betac0*r_s,t-1 + betaf0*eta_t-1 + G*(const1 + betac1*r_s,t-1 + betaf1*eta_t-1)
    
    Where G is the logistic transition function.
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
        # Unrestricted design: [const0, betac0, betaf0, const1*G, betac1*G, betaf1*G]
        X = np.column_stack([
            np.ones(self.nobs),
            self.rs,
            self.eta,
            G,
            G * self.rs,
            G * self.eta
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
                X, _ = self._design_matrix(float(g), float(c))
                
                # Fast OLS
                try:
                    beta, residuals, rank, s = np.linalg.lstsq(X, self.y, rcond=None)
                    if len(residuals) > 0:
                        sse = residuals[0]
                    else:
                        resid = self.y - X @ beta
                        sse = float(np.dot(resid, resid))
                except np.linalg.LinAlgError:
                    continue

                if sse < best_sse:
                    best_sse = sse
                    # beta order: [const0, betac0, betaf0, const1, betac1, betaf1]
                    best_params = {
                        "const0": float(beta[0]),
                        "betac0": float(beta[1]),
                        "betaf0": float(beta[2]),
                        "gamma": float(g),
                        "c": float(c),
                        "const1": float(beta[3]),
                        "betac1": float(beta[4]),
                        "betaf1": float(beta[5]),
                        "sse": float(sse)
                    }
        
        if not np.isfinite(best_sse):
             raise RuntimeError(f"STR-Unrestricted Grid search failed for {self.z_name}")
             
        return best_params

    def fit(self, start_params: Dict[str, float], hac_lags: int = 4) -> StrResult:
        """
        Non-linear Least Squares Estimation.
        """
        # Initial vector: [const0, betac0, betaf0, gamma, c, const1, betac1, betaf1]
        x0 = np.array([
            start_params.get("const0", 0.0),
            start_params.get("betac0", 0.0),
            start_params.get("betaf0", 0.0),
            max(start_params["gamma"], 0.1),
            start_params["c"],
            start_params.get("const1", 0.0),
            start_params.get("betac1", 0.0),
            start_params.get("betaf1", 0.0)
        ])
        
        # Bounds: gamma > 0
        lb = np.array([-np.inf, -np.inf, -np.inf, 1e-8, -np.inf, -np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([ np.inf,  np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.inf], dtype=float)

        def resid_fun(theta):
            # Unpack: [const0, betac0, betaf0, gamma, c, const1, betac1, betaf1]
            const0, betac0, betaf0, g, c_val, const1, betac1, betaf1 = theta
            G = self._compute_G(g, c_val)
            
            # Model: y = const0 + betac0*rs + betaf0*eta + G*(const1 + betac1*rs + betaf1*eta)
            yhat = const0 + (betac0 * self.rs) + (betaf0 * self.eta) + G * (const1 + (betac1 * self.rs) + (betaf1 * self.eta))
            return self.y - yhat

        res = least_squares(
            resid_fun, x0, bounds=(lb, ub), method='trf', loss='linear',
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=20000
        )

        # --- Post-Estimation Statistics ---
        theta = res.x
        resid = res.fun
        sse = float(2.0 * res.cost)
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
        param_names = ["const0", "betac0", "betaf0", "gamma", "c", "const1", "betac1", "betaf1"]
        
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
        Core logic for BUIP matching legacy implementation exactly.
        theta = [beta_f, beta_c, gamma, c, const, const1]
        """
        beta_f, beta_c, gamma, c, const, const1 = theta
        
        # 1. Expectations (formed at t-1 using information from t-2)
        # Fundamentalist: Expects mean reversion (s - beta_f * eta)
        E_s_f_tm1 = self.s_t2 - beta_f * self.eta_t2
        # Chartist: Expects trend continuation (s + beta_c * r)
        E_s_c_tm1 = self.s_t2 + beta_c * self.r_t2
        
        # 2. Expected directional changes (for profit sign)
        E_ds_f_tm1 = -beta_f * self.eta_t2
        E_ds_c_tm1 = beta_c * self.r_t2
        
        # Signs for profit calculation
        sgn_f = np.sign(E_ds_f_tm1)
        sgn_c = np.sign(E_ds_c_tm1)
        
        # 3. Realized profit/return
        profit_ret = self.r_lag1 + self.i_for_t1 - self.i_dom_t1
        
        # 4. Profits weighted by directional correctness
        psi_f = profit_ret * sgn_f
        psi_c = profit_ret * sgn_c
        
        # 5. Risk (squared forecast errors)
        sig2_f = (E_s_f_tm1 - self.s_t1)**2
        sig2_c = (E_s_c_tm1 - self.s_t1)**2
        
        # 6. Utility = Profit - Risk
        U_f = psi_f - sig2_f
        U_c = psi_c - sig2_c
        d = U_f - U_c
        
        # 7. Mixing weight (Omega) with scale normalization
        scale_d = max(float(np.std(d, ddof=1)), 1e-12)
        omega = expit((gamma / scale_d) * (d - c))
        
        # 8. Model prediction matching legacy exactly
        # yhat = const + beta_c*r + omega*(const1 - beta_f*eta - beta_c*r)
        yhat = const + (beta_c * self.r_lag1) + omega * (const1 - (beta_f * self.eta_lag1) - (beta_c * self.r_lag1))
        
        return self.y - yhat, {"omega": omega, "U_f": U_f, "U_c": U_c, "yhat": yhat, "d": d, "scale_d": scale_d}

    def fit_multistart(self, n_starts: int = None, hac_lags: int = 4) -> BuipResult:
        """
        Optimization with multiple starting points matching legacy implementation.
        If n_starts is None, uses grid-based starts like legacy.
        """
        # Bounds: gamma must be > 0, order: [beta_f, beta_c, gamma, c, const, const1]
        lb = np.array([-np.inf, -np.inf, 1e-8, -np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([ np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf], dtype=float)
        
        best_sse = np.inf
        best_res = None
        best_cache = None
        
        # Generate starts matching legacy grid approach
        if n_starts is None:
            # Grid-based starts like legacy
            # First compute initial d values to get c_grid
            bf0, bc0, gm0, c0, const0, const1_0 = 0.2, 1.2, 5.0, 0.0, 0.0, 0.0
            try:
                _, cache0 = self._compute_residuals_and_cache(np.array([bf0, bc0, gm0, c0, const0, const1_0]))
                d_vals = cache0["d"]
                d_valid = d_vals[np.isfinite(d_vals)]
                if len(d_valid) > 0:
                    c_grid = np.array([
                        float(np.quantile(d_valid, 0.10)),
                        float(np.quantile(d_valid, 0.30)),
                        float(np.median(d_valid)),
                        float(np.quantile(d_valid, 0.70)),
                        float(np.quantile(d_valid, 0.90)),
                    ])
                else:
                    c_grid = np.array([0.0])
            except:
                c_grid = np.array([0.0])
            
            bf_grid = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
            bc_grid = np.array([0.30, 0.50, 0.70, 0.90, 1.20, 1.50])
            gamma_grid = np.logspace(np.log10(1.0), np.log10(50.0), num=6)
            const_grid  = np.array([0.0])
            const1_grid = np.array([0.0])
            
            starts = [
                np.array([bf, bc, gm, c_, cst, cst1], dtype=float)
                for bf in bf_grid for bc in bc_grid for gm in gamma_grid
                for c_ in c_grid for cst in const_grid for cst1 in const1_grid
            ]
        else:
            # Random starts
            np.random.seed(42)
            starts = []
            for _ in range(n_starts):
                s = np.array([
                    np.random.uniform(0.01, 0.5),    # beta_f
                    np.random.uniform(0.3, 2.0),     # beta_c
                    np.random.uniform(1, 50),        # gamma
                    np.random.uniform(-1, 1),        # c
                    np.random.normal(0, 0.1),        # const
                    np.random.normal(0, 0.1)         # const1
                ], dtype=float)
                starts.append(s)
        
        # Multistart optimization
        no_improve = 0
        early_stop_patience = 2000
        
        for i, x0 in enumerate(starts):
            # Clip to bounds
            x0 = np.minimum(np.maximum(x0, lb + 1e-10), ub - 1e-10)
            
            try:
                res = least_squares(
                    lambda theta: self._compute_residuals_and_cache(theta)[0],
                    x0, bounds=(lb, ub), method='trf',
                    max_nfev=5000, xtol=1e-6, ftol=1e-6, gtol=1e-6
                )
                
                sse = float(2.0 * res.cost)
                if sse < best_sse:
                    best_sse = sse
                    best_res = res
                    _, best_cache = self._compute_residuals_and_cache(res.x)
                    no_improve = 0
                else:
                    no_improve += 1
            except Exception:
                no_improve += 1
                continue
            
            if no_improve >= early_stop_patience:
                break
                
        if best_res is None:
            raise RuntimeError("BUIP Multistart Optimization failed - all attempts unsuccessful.")
            
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
