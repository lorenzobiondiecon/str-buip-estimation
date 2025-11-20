import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Optional, List, Tuple

# ==============================================================================
# 1. Z-CANDIDATE GENERATION
# ==============================================================================

def build_z_candidates(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Generates 35 z-candidate variables for STR transition variable selection.
    
    Legacy implementation: 7 base variables × 5 lags = 35 candidates.
    
    Base variables:
    - eta: Real exchange rate (q)
    - eta_abs: |q|
    - drs_abs: |r_s| (absolute returns)
    - ID: Interest rate differential (i_for - i_dom)
    - ppp_abs: |f_ppp| (absolute PPP deviation)
    - drf_abs: |f_ppp_rel| (absolute forward premium)
    - rel_misalignment_abs: |f_ppp_rel - r_s| (relative misalignment)
    
    Args:
        data: DataFrame with columns [r_s, q, i_for, i_dom, f_ppp, f_ppp_rel]
    
    Returns:
        Dictionary mapping candidate names to lagged Series (35 total)
    """
    z_base = {
        "eta": data["q"],
        "eta_abs": np.abs(data["q"]),
        "drs_abs": np.abs(data["r_s"]),
        "ID": (data["i_for"] - data["i_dom"]),
        "ppp_abs": np.abs(data["f_ppp"]),
        "drf_abs": np.abs(data["f_ppp_rel"]),
        "rel_misalignment_abs": np.abs(data["f_ppp_rel"] - data["r_s"]),
    }
    
    out: Dict[str, pd.Series] = {}
    for nm, s in z_base.items():
        for lag in (1, 2, 3, 4, 5):
            out[f"{nm}_lag{lag}"] = s.shift(lag)
    
    return out


# ==============================================================================
# 2. LM LINEARITY TESTS (LEGACY IMPLEMENTATION)
# ==============================================================================

def build_linear_base_for_lm_test(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares base DataFrame for LM linearity tests.
    
    Creates dependent variable (y) and linear model residuals (u) for testing.
    
    Args:
        data: DataFrame with [r_s, i_for, i_dom, q]
    
    Returns:
        DataFrame with columns [y, rs_lag1, eta_lag1, u]
    """
    df = pd.DataFrame({
        "y": data["r_s"] - (data["i_for"] - data["i_dom"]),
        "rs_lag1": data["r_s"].shift(1),
        "eta_lag1": data["q"].shift(1)
    }).dropna()
    
    if len(df) < 20:
        raise ValueError("Insufficient observations for LM test after lagging")
    
    # Estimate linear model to get residuals
    y = df["y"].to_numpy()
    X = df[["rs_lag1", "eta_lag1"]].to_numpy()
    
    model = sm.OLS(y, X).fit()
    df["u"] = model.resid
    
    return df


def lm_linearity_for_z(base: pd.DataFrame, z: pd.Series, maxlags_hac: int = 4) -> Optional[Dict]:
    """
    Performs LM linearity test for a specific z-candidate variable.
    
    Tests H0: Linear vs H1: STR with transition variable z.
    Uses auxiliary regression with Taylor expansion: x*z, x*z^2, x*z^3.
    
    Args:
        base: Linear model DataFrame with [y, rs_lag1, eta_lag1, u]
        z: Candidate transition variable
        maxlags_hac: Lags for HAC covariance
    
    Returns:
        Dict with keys [LM, p_value, LM_HAC, p_value_HAC, df, nobs]
        Returns None if insufficient data
    """
    df = base.join(z.rename("z"), how="inner").dropna()
    
    if len(df) < 20:
        return None
    
    T = len(df)
    y = df["y"].to_numpy()
    u = df["u"].to_numpy()
    X = df[["rs_lag1", "eta_lag1"]].to_numpy()
    
    zc = df["z"].to_numpy()
    z1, z2, z3 = zc, zc**2, zc**3
    
    # Construct interaction terms: X * z^k for k=1,2,3
    W_rs = np.column_stack([
        df["rs_lag1"].to_numpy() * z1,
        df["rs_lag1"].to_numpy() * z2,
        df["rs_lag1"].to_numpy() * z3
    ])
    W_eta = np.column_stack([
        df["eta_lag1"].to_numpy() * z1,
        df["eta_lag1"].to_numpy() * z2,
        df["eta_lag1"].to_numpy() * z3
    ])
    W = np.column_stack([W_rs, W_eta])
    k = W.shape[1]  # Should be 6 (2 vars × 3 powers)
    
    # Project W onto M_X (residual maker for X)
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    P_X = X @ XTX_inv @ X.T
    M_X = np.eye(T) - P_X
    W_star = M_X @ W
    
    # Standard LM test
    aux = sm.OLS(u, W_star).fit()
    R2 = aux.rsquared
    LM = T * R2
    pval = 1.0 - stats.chi2.cdf(LM, df=k)
    
    # HAC-robust LM test
    aux_hac = sm.OLS(u, W_star).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags_hac})
    beta = np.asarray(aux_hac.params)
    cov = np.asarray(aux_hac.cov_params())
    cov_inv = np.linalg.pinv(cov)
    LM_HAC = float(beta.T @ cov_inv @ beta)
    pval_HAC = 1.0 - stats.chi2.cdf(LM_HAC, df=k)
    
    return {
        "LM": float(LM),
        "p_value": float(pval),
        "LM_HAC": float(LM_HAC),
        "p_value_HAC": float(pval_HAC),
        "df": int(k),
        "nobs": int(T)
    }


def run_lm_linearity_tests(data: pd.DataFrame, maxlags_hac: int = 4) -> pd.DataFrame:
    """
    Runs LM linearity tests for all 35 z-candidate variables.
    
    Args:
        data: Country-specific DataFrame
        maxlags_hac: Lags for HAC covariance
    
    Returns:
        DataFrame sorted by p_value with columns:
        [z_var, LM, p_value, LM_HAC, p_value_HAC, df, nobs]
    """
    base = build_linear_base_for_lm_test(data)
    candidates = build_z_candidates(data)
    
    rows = []
    for name, z in candidates.items():
        res = lm_linearity_for_z(base, z, maxlags_hac=maxlags_hac)
        if res is None:
            continue
        row = {"z_var": name}
        row.update(res)
        rows.append(row)
    
    lm_tbl = pd.DataFrame(rows)
    if not lm_tbl.empty:
        lm_tbl = lm_tbl.sort_values("p_value").reset_index(drop=True)
    
    return lm_tbl


# ==============================================================================
# 3. SIMPLE LINEARITY TEST (BACKWARD COMPATIBILITY)
# ==============================================================================

def luukkonen_linearity_test(y, X, z) -> float:
    """
    Implements the Luukkonen, Saikkonen, and Terasvirta (1988) linearity test.
    
    Tests H0: Linear Model vs H1: STR Model.
    Uses an auxiliary regression with Taylor expansion terms of the transition function.
    
    Args:
        y: Dependent variable (pandas Series or numpy array)
        X: Independent variables (pandas Series/DataFrame or numpy array)
        z: Transition variable (pandas Series or numpy array)
    
    Returns:
        p_value: P-value for the linearity test
    
    Auxiliary Regression:
    u_t = beta0' x_t + beta1' (x_t * z_t) + beta2' (x_t * z_t^2) + beta3' (x_t * z_t^3) + e_t
    """
    # Convert to numpy arrays
    y_arr = np.asarray(y).ravel()
    X_arr = np.asarray(X).reshape(-1, 1) if np.asarray(X).ndim == 1 else np.asarray(X)
    z_arr = np.asarray(z).ravel()
    
    n = len(y_arr)
    
    # 1. Estimate linear model (H0)
    model_linear = sm.OLS(y_arr, X_arr).fit()
    u_linear = model_linear.resid
    rss0 = np.sum(u_linear**2)
    
    # 2. Construct Auxiliary Regressors (Z2, Z3, Z4 in paper notation)
    # Interactions: x*z, x*z^2, x*z^3
    Z_terms = []
    for k in range(1, 4): # Powers 1, 2, 3
        z_pow = (z_arr ** k).reshape(-1, 1)
        Z_terms.append(X_arr * z_pow)
        
    X_aux = np.hstack([X_arr] + Z_terms)
    
    # 3. Auxiliary Regression
    model_aux = sm.OLS(u_linear, X_aux).fit()
    rss1 = np.sum(model_aux.resid**2)
    
    # 4. Compute LM Statistic
    # LM = T * (RSS0 - RSS1) / RSS0
    lm_stat = n * (rss0 - rss1) / rss0
    
    # Degrees of freedom = 3 * number of vars in X
    k = X_arr.shape[1]
    df = 3 * k
    
    p_value = 1 - stats.chi2.cdf(lm_stat, df)
    
    return p_value


# ==============================================================================
# 4. DIAGNOSTIC TESTS
# ==============================================================================

def ljung_box_test(resid: pd.Series, lags: int = 10) -> Dict[str, float]:
    """
    Ljung-Box Q-test for autocorrelation in residuals.
    
    H0: No autocorrelation up to lag h
    
    Args:
        resid: Residual series
        lags: Number of lags to test
    
    Returns:
        Dict with keys [Q_stat, p_value, lags]
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    result = acorr_ljungbox(resid, lags=lags, return_df=True)
    
    # Get last row (max lag)
    Q_stat = float(result.iloc[-1]['lb_stat'])
    p_value = float(result.iloc[-1]['lb_pvalue'])
    
    return {
        "Q_stat": Q_stat,
        "p_value": p_value,
        "lags": int(lags)
    }


def arch_test(resid: pd.Series, lags: int = 4) -> Dict[str, float]:
    """
    ARCH test for heteroskedasticity (ARCH effects).
    
    H0: No ARCH effects up to lag q
    
    Args:
        resid: Residual series
        lags: Number of lags to test
    
    Returns:
        Dict with keys [LM_stat, p_value, lags]
    """
    from statsmodels.stats.diagnostic import het_arch
    
    lm_stat, p_value, f_stat, fp_value = het_arch(resid, nlags=lags)
    
    return {
        "LM_stat": float(lm_stat),
        "p_value": float(p_value),
        "lags": int(lags)
    }


def jarque_bera_test(resid: pd.Series) -> Dict[str, float]:
    """
    Jarque-Bera test for normality of residuals.
    
    H0: Residuals are normally distributed
    
    Args:
        resid: Residual series
    
    Returns:
        Dict with keys [JB_stat, p_value, skewness, kurtosis]
    """
    from scipy.stats import jarque_bera, skew, kurtosis
    
    jb_stat, p_value = jarque_bera(resid)
    skewness = skew(resid)
    kurt = kurtosis(resid)
    
    return {
        "JB_stat": float(jb_stat),
        "p_value": float(p_value),
        "skewness": float(skewness),
        "kurtosis": float(kurt)
    }


def compute_regime_occupancy(G: pd.Series, z: pd.Series, c: float, gamma: float) -> Dict[str, float]:
    """
    Computes regime occupancy statistics for STR model.
    
    Regime classification:
    - Low (Chartist): G < 0.2
    - Middle: 0.2 <= G <= 0.8
    - High (Fundamentalist): G > 0.8
    
    Args:
        G: Transition function values
        z: Transition variable
        c: Threshold parameter
        gamma: Smoothness parameter
    
    Returns:
        Dict with keys:
        - p_lo: Proportion in low regime
        - p_mid: Proportion in middle regime
        - p_hi: Proportion in high regime
        - avg_spell_C: Average spell length in chartist regime
        - avg_spell_F: Average spell length in fundamentalist regime
        - gamma_norm: Gamma normalized by sd(z)
    """
    if len(G) == 0:
        return {}
    
    # Regime probabilities
    p_lo = float((G < 0.2).mean())
    p_mid = float(((G >= 0.2) & (G <= 0.8)).mean())
    p_hi = float((G > 0.8).mean())
    
    # Spell lengths (runs analysis)
    states = (G >= 0.5).astype(int).to_numpy()
    runs = []
    cur = states[0]
    L = 1
    for s in states[1:]:
        if s == cur:
            L += 1
        else:
            runs.append((cur, L))
            cur = s
            L = 1
    runs.append((cur, L))
    
    low_runs = [l for st, l in runs if st == 0]
    high_runs = [l for st, l in runs if st == 1]
    
    avg_c = float(np.mean(low_runs)) if len(low_runs) > 0 else np.nan
    avg_f = float(np.mean(high_runs)) if len(high_runs) > 0 else np.nan
    
    # Normalized gamma
    z_arr = pd.to_numeric(z, errors="coerce").dropna()
    sdz = float(z_arr.std(ddof=1))
    gamma_norm = gamma / sdz if np.isfinite(sdz) and sdz > 0 else np.nan
    
    return {
        "p_lo": p_lo,
        "p_mid": p_mid,
        "p_hi": p_hi,
        "avg_spell_C": avg_c,
        "avg_spell_F": avg_f,
        "gamma_norm": gamma_norm
    }


def run_full_diagnostics(resid: pd.Series, G: Optional[pd.Series] = None, 
                        z: Optional[pd.Series] = None, c: Optional[float] = None, 
                        gamma: Optional[float] = None) -> Dict[str, any]:
    """
    Runs complete diagnostic test suite on model residuals.
    
    Args:
        resid: Model residuals
        G: Transition function (optional, for STR regime analysis)
        z: Transition variable (optional)
        c: Threshold parameter (optional)
        gamma: Smoothness parameter (optional)
    
    Returns:
        Dict containing all diagnostic test results
    """
    diagnostics = {}
    
    # Autocorrelation
    try:
        diagnostics["ljung_box"] = ljung_box_test(resid, lags=10)
    except Exception as e:
        diagnostics["ljung_box"] = {"error": str(e)}
    
    # ARCH effects
    try:
        diagnostics["arch"] = arch_test(resid, lags=4)
    except Exception as e:
        diagnostics["arch"] = {"error": str(e)}
    
    # Normality
    try:
        diagnostics["jarque_bera"] = jarque_bera_test(resid)
    except Exception as e:
        diagnostics["jarque_bera"] = {"error": str(e)}
    
    # Regime occupancy (STR only)
    if G is not None and z is not None and c is not None and gamma is not None:
        try:
            diagnostics["regime_occupancy"] = compute_regime_occupancy(G, z, c, gamma)
        except Exception as e:
            diagnostics["regime_occupancy"] = {"error": str(e)}
    
    return diagnostics


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
