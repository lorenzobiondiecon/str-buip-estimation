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
    Regime 1 (G=1): y_t = (alpha + alpha1) + beta_f * eta_t-1

    Mixing Equation (eta enters with a positive sign so mean reversion implies beta_f < 0):
    y_t = alpha + beta_c * r_s + G * (alpha1 + beta_f * eta - beta_c * r_s)
    """
    
    def __init__(self, df: pd.DataFrame, z_col: str):
        self.y = df["y"].values
        self.z = df[z_col].values
        self.rs = df["rs_t1"].values
        self.eta = df["eta_t1"].values
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
        # Derived from: alpha + beta_c*rs + G*alpha1 + G*beta_f*eta - G*beta_c*rs
        # Rearranged: alpha*1 + beta_c*(1-G)*rs + beta_f*(G*eta) + alpha1*G

        # Note: Order kept as X = [1, (1-G)*rs, G*(eta), G] -> [const, beta_c, beta_f, const1]
        X = np.column_stack([
            np.ones(self.nobs),
            (1 - G) * self.rs,
            G * (self.eta),
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
            
            # Model: y = const + bc*rs + G*(const1 + bf*eta - bc*rs)
            yhat = const + (bc * self.rs) + G * (const1 + (bf * self.eta) - (bc * self.rs))
            return self.y - yhat

        res = least_squares(
            resid_fun, x0, bounds=(lb, ub), method='trf',
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
        self.rs = df["rs_t1"].values
        self.eta = df["eta_t1"].values
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

    def fit(self, start_params: Dict[str, float], hac_lags: int = None) -> StrResult:
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

# ============================================================================
# 1c. UNRESTRICTED LSTR2 MODEL (TWO THRESHOLDS)
# ============================================================================

class STRModelNoRestrictionsLSTR2:
    """
    Unrestricted LSTR2 Model for Exchange Rates.

    Allows all parameters to vary freely across regimes with LSTR2 transition:
    G = expit((gamma/z_std) * (z - c1) * (z - c2))

    Model:
      y_t = const0 + betac0*r_s,t-1 + betaf0*eta,t-1
            + G*(const1 + betac1*r_s,t-1 + betaf1*eta,t-1)
    """

    def __init__(self, df: pd.DataFrame, z_col: str):
        self.y = df["y"].values
        self.z = df[z_col].values
        self.rs = df["rs_t1"].values
        self.eta = df["eta_t1"].values
        self.index = df.index
        self.z_name = z_col
        self.nobs = len(self.y)

        # Robust standard deviation for scaling gamma
        self.z_std = max(float(np.std(self.z)), 1e-8)

    def _compute_G(self, gamma: float, c1: float, c2: float) -> np.ndarray:
        return expit((gamma / self.z_std) * (self.z - c1) * (self.z - c2))

    def _design_matrix(self, gamma: float, c1: float, c2: float) -> np.ndarray:
        G = self._compute_G(gamma, c1, c2)
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
        Searches gamma in gamma_grid and (c1,c2) over all ordered pairs from c_grid with c1 < c2.
        """
        best_sse = np.inf
        best_params: Dict[str, float] = {}

        for g in gamma_grid:
            for i, c1 in enumerate(c_grid):
                for c2 in c_grid[i+1:]:  # ensure c1 < c2
                    X, _ = self._design_matrix(float(g), float(c1), float(c2))
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
                        best_params = {
                            "const0": float(beta[0]),
                            "betac0": float(beta[1]),
                            "betaf0": float(beta[2]),
                            "gamma": float(g),
                            "c1": float(c1),
                            "c2": float(c2),
                            "const1": float(beta[3]),
                            "betac1": float(beta[4]),
                            "betaf1": float(beta[5]),
                            "sse": float(sse)
                        }

        if not np.isfinite(best_sse):
            raise RuntimeError(f"STR-Unrestricted LSTR2 Grid search failed for {self.z_name}")

        return best_params

    def fit(self, start_params: Dict[str, float], hac_lags: int = None) -> StrResult:
        """Non-linear Least Squares Estimation for LSTR2."""
        # Initial vector: [const0, betac0, betaf0, gamma, c1, c2, const1, betac1, betaf1]
        x0 = np.array([
            start_params.get("const0", 0.0),
            start_params.get("betac0", 0.0),
            start_params.get("betaf0", 0.0),
            max(start_params["gamma"], 0.1),
            start_params["c1"],
            start_params["c2"],
            start_params.get("const1", 0.0),
            start_params.get("betac1", 0.0),
            start_params.get("betaf1", 0.0)
        ])

        # Bounds: gamma > 0; c1, c2 unbounded
        lb = np.array([-np.inf, -np.inf, -np.inf, 1e-8, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([ np.inf,  np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf], dtype=float)

        def resid_fun(theta):
            # Unpack: [const0, betac0, betaf0, gamma, c1, c2, const1, betac1, betaf1]
            const0, betac0, betaf0, g, c1_val, c2_val, const1, betac1, betaf1 = theta
            G = self._compute_G(g, c1_val, c2_val)
            yhat = const0 + (betac0 * self.rs) + (betaf0 * self.eta) + G * (const1 + (betac1 * self.rs) + (betaf1 * self.eta))
            return self.y - yhat

        res = least_squares(
            resid_fun, x0, bounds=(lb, ub), method='trf', loss='linear',
            xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=20000
        )

        theta = res.x
        resid = res.fun
        sse = float(2.0 * res.cost)
        dof = max(self.nobs - len(theta), 1)

        # Recompute G and yhat at optimum
        G_opt = self._compute_G(theta[3], theta[4], theta[5])
        yhat = self.y - resid

        # HAC covariance
        cov_hac = _newey_west_cov(res.jac, resid, hac_lags)
        se = np.sqrt(np.diag(cov_hac))
        tvals = theta / se
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), dof))

        param_names = ["const0", "betac0", "betaf0", "gamma", "c1", "c2", "const1", "betac1", "betaf1"]

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
    scale_d: float

# =========================
# CLEAN DROP-IN PATCH
# Fixes applied:
# (1) Initialize scale attributes in __init__ so errors are explicit
# (2) Ensure beta_f0, beta_c0 exist in every branch (random starts bug)
# (3) Ensure _setup_scales() is always called before any call that requires scales
# NO changes to:
# - enforcing mean reversion sign on beta_f0
# - omega-variation guard in conditional OLS
# =========================

class BUIPModel:
    """
    Behavioral Model where regime weights depend on the utility difference 
    between Fundamentalists and Chartists.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.y = df["y"].values
        self.nobs = len(self.y)
        self.index = df.index
        
        self.eta_t1 = df["eta_t1"].values
        self.rs_t1  = df["rs_t1"].values
        self.s_t1   = df["s_t1"].values
        self.s_t2   = df["s_t2"].values
        self.eta_t2 = df["eta_t2"].values
        self.rs_t2  = df["rs_t2"].values
        self.i_for_t1 = df["i_for_t1"].values
        self.i_dom_t1 = df["i_dom_t1"].values

        # ---- initialize caches / scales (prevents AttributeError) ----
        self.kappa = None
        self.scale_Eds_f = None
        self.scale_Eds_c = None
        self.scale_d = None
        self.d0_quantiles = None
        self.d0_iqr = None

    def _compute_residuals_and_cache(self, theta: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        theta = [beta_f, beta_c, gamma, c, const, const1]
        """
        if self.scale_d is None or self.scale_Eds_f is None or self.scale_Eds_c is None or self.kappa is None:
            raise RuntimeError("Scales not initialized. Call _setup_scales(beta_f0, beta_c0, kappa=...) before fitting.")

        beta_f, beta_c, gamma, c, const, const1 = theta
        
        E_s_f_tm1 = self.s_t2 + beta_f * self.eta_t2
        E_s_c_tm1 = self.s_t2 + beta_c * self.rs_t2

        E_ds_f_tm1 = beta_f * self.eta_t2
        E_ds_c_tm1 = beta_c * self.rs_t2
        
        kappa = self.kappa
        sgn_f = np.tanh(kappa * (E_ds_f_tm1 / self.scale_Eds_f))
        sgn_c = np.tanh(kappa * (E_ds_c_tm1 / self.scale_Eds_c))
        
        profit_ret = self.rs_t1 + self.i_for_t1 - self.i_dom_t1
        
        psi_f = profit_ret * sgn_f
        psi_c = profit_ret * sgn_c
        
        sig2_f = (E_s_f_tm1 - self.s_t1)**2
        sig2_c = (E_s_c_tm1 - self.s_t1)**2
        
        U_f = psi_f - sig2_f
        U_c = psi_c - sig2_c
        d = U_f - U_c
        
        omega = expit((gamma / self.scale_d) * (d - c))
        scale_d = self.scale_d
        
        yhat = const + (beta_c * self.rs_t1) + omega * (const1 + (beta_f * self.eta_t1) - (beta_c * self.rs_t1))
        
        return self.y - yhat, {
            "omega": omega,
            "U_f": U_f,
            "U_c": U_c,
            "yhat": yhat,
            "d": d,
            "scale_d": scale_d
        }

    def _setup_scales(self, beta_f0: float, beta_c0: float, kappa: float = 10.0) -> Dict[str, float]:
        self.kappa = float(kappa)
        
        E_ds_f0 = beta_f0 * self.eta_t2
        E_ds_c0 = beta_c0 * self.rs_t2
        
        self.scale_Eds_f = max(float(np.std(E_ds_f0, ddof=1)), 1e-12)
        self.scale_Eds_c = max(float(np.std(E_ds_c0, ddof=1)), 1e-12)
        
        E_s_f0 = self.s_t2 + beta_f0 * self.eta_t2
        E_s_c0 = self.s_t2 + beta_c0 * self.rs_t2
        
        sgn_f0 = np.tanh(self.kappa * (E_ds_f0 / self.scale_Eds_f))
        sgn_c0 = np.tanh(self.kappa * (E_ds_c0 / self.scale_Eds_c))
        
        profit_ret = self.rs_t1 + self.i_for_t1 - self.i_dom_t1
        
        psi_f0 = profit_ret * sgn_f0
        psi_c0 = profit_ret * sgn_c0
        
        sig2_f0 = (E_s_f0 - self.s_t1)**2
        sig2_c0 = (E_s_c0 - self.s_t1)**2
        
        d0 = (psi_f0 - sig2_f0) - (psi_c0 - sig2_c0)
        d0 = d0[np.isfinite(d0)]
        
        # if len(d0) < 10:
        #     self.scale_d = 1.0
        #     self.d0_quantiles = np.array([0.0])
        #     self.d0_iqr = 1.0
        #     return {"scale_d": 1.0, "IQR": 1.0, "quantiles": self.d0_quantiles, "gamma_0": 5.0}
        
        self.scale_d = max(float(np.std(d0, ddof=1)), 1e-12)
        q25, q75 = np.quantile(d0, [0.25, 0.75])
        IQR = max(float(q75 - q25), 1e-8)
        
        gamma_0 = 4.394 * self.scale_d / IQR
        
        # Increased from 7 to 9 quantiles for better threshold coverage (15 x 9 = 135 total starts)
        quantiles = np.quantile(d0, [0.05, 0.15, 0.30, 0.45, 0.50, 0.55, 0.70, 0.85, 0.95])
        self.d0_quantiles = quantiles
        self.d0_iqr = float(IQR)
        
        return {"scale_d": self.scale_d, "IQR": IQR, "quantiles": quantiles, "gamma_0": gamma_0}

    def _conditional_ols_start(self, gamma: float, c: float, beta_f0: float, beta_c0: float) -> np.ndarray:
        # Assumes scales already set via _setup_scales()
        theta_temp = np.array([beta_f0, beta_c0, gamma, c, 0.0, 0.0])
        _, cache = self._compute_residuals_and_cache(theta_temp)
        omega = cache["omega"]
        
        X = np.column_stack([
            np.ones(self.nobs),
            (1 - omega) * self.rs_t1,
            omega * self.eta_t1,
            omega
        ])
        
        try:
            beta = np.linalg.lstsq(X, self.y, rcond=None)[0]
            const_ols, beta_c_ols, beta_f_ols, const1_ols = beta
        except np.linalg.LinAlgError:
            const_ols, beta_c_ols, beta_f_ols, const1_ols = 0.0, beta_c0, beta_f0, 0.0
        
        return np.array([beta_f_ols, beta_c_ols, gamma, c, const_ols, const1_ols], dtype=float)

    def fit_multistart(
        self,
        n_starts: int = None,
        hac_lags: int = None,
        random_seed: int = 42,
        method: str = "conditional_ols",
        kappa: float = 10.0,
        top_K: int = 50
    ) -> BuipResult:
        np.random.seed(random_seed)
        
        lb = np.array([-np.inf, -np.inf, 1e-8, -np.inf, -np.inf, -np.inf], dtype=float)
        ub = np.array([ np.inf,  np.inf, np.inf,  np.inf,  np.inf,  np.inf], dtype=float)
        
        best_sse = np.inf
        best_res = None
        best_cache = None

        # ---- Always compute baseline slopes once (fixes undefined beta_f0/beta_c0) ----
        X_simple = np.column_stack([np.ones(self.nobs), self.rs_t1, self.eta_t1])
        try:
            beta_simple = np.linalg.lstsq(X_simple, self.y, rcond=None)[0]
            beta_c0 = float(beta_simple[1])
            beta_f0 = float(beta_simple[2])
            beta_f0 = np.clip(beta_f0, -1.0, -0.01)
            beta_c0 = np.clip(beta_c0, 0.1, 2.0)
        except Exception:
            beta_f0, beta_c0 = -0.2, 0.8

        # ---- Always setup scales once before generating starts ----
        scales = self._setup_scales(beta_f0, beta_c0, kappa=kappa)

        if method == "conditional_ols" and n_starts is None:
            c_grid = np.array(scales["quantiles"], dtype=float)
            gamma_0 = float(scales["gamma_0"])
            # Increased from 13 to 15 gamma points for better coverage (15 x 7 = 105 starts)
            gamma_grid = gamma_0 * np.logspace(-2, 2, 15)

            starts = [self._conditional_ols_start(gm, c_val, beta_f0, beta_c0)
                      for gm in gamma_grid for c_val in c_grid]

        elif n_starts is None:
            # legacy grid (kept)
            bf_grid = np.array([-1.0, -0.75, -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, -0.01])
            bc_grid = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.20, 1.50])
            gamma_grid = np.logspace(np.log10(1.0), np.log10(50.0), num=6)
            c_grid = np.array(np.quantile((self._compute_residuals_and_cache(np.array([beta_f0, beta_c0, 5.0, 0.0, 0.0, 0.0]))[1]["d"])[np.isfinite(
                self._compute_residuals_and_cache(np.array([beta_f0, beta_c0, 5.0, 0.0, 0.0, 0.0]))[1]["d"]
            )], [0.10, 0.30, 0.50, 0.70, 0.90])) if self.nobs > 0 else np.array([0.0])

            starts = [np.array([bf, bc, gm, c_, 0.0, 0.0], dtype=float)
                      for bf in bf_grid for bc in bc_grid for gm in gamma_grid for c_ in c_grid]

        else:
            # random starts: now beta_f0/beta_c0 are defined and scales are set
            starts = []
            for _ in range(n_starts):
                starts.append(np.array([
                    np.random.uniform(-0.5, -0.01),
                    np.random.uniform(0.3, 2.0),
                    np.random.uniform(1, 50),
                    np.random.uniform(-1, 1),
                    np.random.normal(0, 0.1),
                    np.random.normal(0, 0.1)
                ], dtype=float))

        # ---- Two-pass optimization (unchanged logic) ----
        quick_results = []
        for x0 in starts:
            x0 = np.minimum(np.maximum(x0, lb + 1e-10), ub - 1e-10)
            try:
                res = least_squares(
                    lambda th: self._compute_residuals_and_cache(th)[0],
                    x0, bounds=(lb, ub), method="trf",
                    max_nfev=2000, xtol=1e-6, ftol=1e-6, gtol=1e-6
                )
                quick_results.append((float(2.0 * res.cost), res))
            except Exception:
                continue

        if not quick_results:
            raise RuntimeError("BUIP Multistart: All starts failed in quick pass.")

        quick_results.sort(key=lambda t: t[0])
        for _, res_quick in quick_results[:min(int(top_K), len(quick_results))]:
            try:
                res = least_squares(
                    lambda th: self._compute_residuals_and_cache(th)[0],
                    res_quick.x, bounds=(lb, ub), method="trf",
                    max_nfev=8000, xtol=1e-8, ftol=1e-8, gtol=1e-8
                )
                sse = float(2.0 * res.cost)
                if sse < best_sse:
                    best_sse = sse
                    best_res = res
                    _, best_cache = self._compute_residuals_and_cache(res.x)
            except Exception:
                continue

        if best_res is None:
            raise RuntimeError("BUIP Multistart: refinement produced no successful result.")

        # stats (unchanged)
        theta = best_res.x
        resid = best_res.fun
        dof = max(self.nobs - len(theta), 1)

        cov_hac = _newey_west_cov(best_res.jac, resid, hac_lags)
        se = np.sqrt(np.diag(cov_hac))
        tvals = theta / se
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), dof))

        param_names = ["beta_f", "beta_c", "gamma", "c", "const", "const1"]

        # scales_out = {
        #     "scale_d": float(self.scale_d),
        #     "scale_Eds_f": float(self.scale_Eds_f),
        #     "scale_Eds_c": float(self.scale_Eds_c),
        #     "kappa": float(self.kappa),
        #     "d0_iqr": None if self.d0_iqr is None else float(self.d0_iqr),
        #     "d0_quantiles": None if self.d0_quantiles is None else self.d0_quantiles.copy(),
        #     }

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
            cov_hac=cov_hac,
            scale_d=float(self.scale_d)
        )

# ==============================================================================
# 3. SHARED HELPERS
# ==============================================================================

def _newey_west_cov(J: np.ndarray, resid: np.ndarray, maxlags: int = None) -> np.ndarray:
    """
    Computes Newey-West HAC covariance matrix.
    
    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix (n x k)
    resid : np.ndarray
        Residuals (n,)
    maxlags : int, optional
        Maximum number of lags. If None, uses Newey-West (1994) rule:
        L = floor(4 * (T/100)^(2/9))
    
    Returns
    -------
    np.ndarray
        HAC covariance matrix (k x k)
    """
    n, k = J.shape
    
    # Automatic bandwidth selection if not specified (Newey-West 1994)
    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100) ** (2/9)))
    
    S = J * resid[:, None]
    
    # Gamma 0
    Omega = (S.T @ S) / n
    
    # Gamma k (Bartlett kernel)
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

# ==============================================================================
# 4. LINEAR ARX BENCHMARK (FOR COMPARISON)
# ==============================================================================

def run_linear_arx_benchmark(data: pd.DataFrame) -> dict:
    """Estimate a linear ARX benchmark: y ~ const + rs_t1 + (-eta_t1).

    Returns parameter table and fit metrics including RMSE, AIC, BIC, pseudo-R2.
    """
    import statsmodels.api as sm

    df_lin = pd.DataFrame({
        "y": data["r_s"] - (data["i_for"] - data["i_dom"]),
        "rs_t1": data["r_s"].shift(1),
        "eta_t1": data["q"].shift(1)
    }).dropna()
    y = df_lin["y"].to_numpy()
    X = sm.add_constant(df_lin[["rs_t1","eta_t1"]])
    olin = sm.OLS(y, X).fit()
    olin_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    SSR = float(np.sum(olin.resid**2))
    TSS = float(np.sum(y**2))
    n, k = X.shape
    rmse = np.sqrt(SSR / n)
    aic = n * np.log(SSR / n) + 2 * k
    bic = n * np.log(SSR / n) + np.log(n) * k
    pseudo_r2 = 1.0 - SSR / TSS if TSS > 0 else np.nan

    arx_tbl = pd.DataFrame({
        "Parameter": ["mu(const)", "phi(rs_t1)", "theta(eta_t1)"],
        "Estimate_IID": [olin.params["const"], olin.params["rs_t1"], olin.params["eta_t1"]],
        "SE_IID":       [olin.bse["const"],    olin.bse["rs_t1"],    olin.bse["eta_t1"]],
        "t_IID":        [olin.tvalues["const"],olin.tvalues["rs_t1"],olin.tvalues["eta_t1"]],
        "p_IID":        [olin.pvalues["const"],olin.pvalues["rs_t1"],olin.pvalues["eta_t1"]],
        "SE_HAC(L=6)":  [olin_hac.bse["const"],olin_hac.bse["rs_t1"],olin_hac.bse["eta_t1"]],
    })
    fit_tbl = pd.DataFrame([{"RMSE": rmse, "AIC": aic, "BIC": bic, "pseudo_R2": pseudo_r2, "nobs": n}])
    return {"params": arx_tbl, "fit": fit_tbl}
