"""
Analysis functions for STR and BUIP estimation pipeline.
Contains helper functions for z-candidate selection, linearity testing, model estimation,
and diagnostic tests.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Optional, List, Tuple
from .econometrics import STRModel, BUIPModel
from .utils import build_beh_sample


def build_z_candidates(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Generates 35 transition variable candidates for STR models.
    
    Base variables:
    - eta_abs: |q|
    - drs_abs: |r_s| (absolute returns)
    - ID_abs: |i_for - i_dom| (absolute interest rate differential)
    - ppp_abs: |f_ppp| (absolute PPP deviation)
    - drf_abs: |f_ppp_rel| (absolute forward premium)
    - rel_misalignment_abs: |f_ppp_rel - r_s| (relative misalignment)
    - excess_returns_abs: |r_s - (i_for - i_dom)| (absolute excess returns)
    
    Args:
        data: DataFrame with columns [r_s, q, i_for, i_dom, f_ppp, f_ppp_rel]
    
    Returns:
        Dictionary mapping candidate names to lagged Series (35 total)
    """
    z_base = {
        # "eta": data["q"],
        "eta_abs": np.abs(data["q"]),
        "drs_abs": np.abs(data["r_s"]),
        "ID_abs": np.abs(data["i_for"] - data["i_dom"]),
        "ppp_abs": np.abs(data["f_ppp"]),
        "drf_abs": np.abs(data["f_ppp_rel"]),
        "rel_misalignment_abs": np.abs(data["f_ppp_rel"] - data["r_s"]),
        "excess_returns_abs": np.abs(data["r_s"] - (data["i_for"] - data["i_dom"]))
    }
    
    candidates = {}
    for name, series in z_base.items():
        for lag in range(1, 6):  # lags 1-5
            candidates[f"{name}_lag{lag}"] = series.shift(lag)
    
    return candidates


# def lm_linearity_test_for_z(data: pd.DataFrame, z: pd.Series, maxlags_hac: int = 4) -> Optional[dict]:
#     """
#     Performs LM linearity test for a specific transition variable.
    
#     Tests H0: Linear model vs H1: STR model using Taylor expansion approximation.
#     Uses HAC-robust standard errors for the test statistic.
    
#     Parameters
#     ----------
#     data : pd.DataFrame
#         Country-specific panel data
#     z : pd.Series
#         Candidate transition variable
#     maxlags_hac : int, default=4
#         Number of lags for HAC covariance estimation
    
#     Returns
#     -------
#     dict or None
#         Test results with keys: LM, p_value, LM_HAC, p_value_HAC, df, nobs
#         Returns None if insufficient data
#     """
#     # Build base linear model
#     df = pd.DataFrame({
#         "y": data["r_s"] - (data["i_for"] - data["i_dom"]),
#         "rs_t1": data["r_s"].shift(1),
#         "eta_t1": data["q"].shift(1),
#     }).dropna()
    
#     # Join with z
#     df = df.join(z.rename("z"), how="inner").dropna()
    
#     if len(df) < 20:
#         return None
    
#     T = len(df)
#     y = df["y"].to_numpy()
#     X = df[["rs_t1", "eta_t1"]].to_numpy()
    
#     # Linear model residuals
#     lin_res = sm.OLS(y, X).fit()
#     u = lin_res.resid
    
#     # Taylor expansion terms (3rd order)
#     zc = df["z"].to_numpy()
#     z1, z2, z3 = zc, zc**2, zc**3
    
#     W_rs = np.column_stack([
#         df["rs_t1"].to_numpy() * z1,
#         df["rs_t1"].to_numpy() * z2,
#         df["rs_t1"].to_numpy() * z3
#     ])
#     W_eta = np.column_stack([
#         df["eta_t1"].to_numpy() * z1,
#         df["eta_t1"].to_numpy() * z2,
#         df["eta_t1"].to_numpy() * z3
#     ])
#     W = np.column_stack([W_rs, W_eta])
#     k = W.shape[1]
    
#     # Project out X
#     XTX_inv = np.linalg.pinv(X.T @ X)
#     P_X = X @ XTX_inv @ X.T
#     M_X = np.eye(T) - P_X
#     W_star = M_X @ W
    
#     # Auxiliary regression
#     aux = sm.OLS(u, W_star).fit()
#     R2 = aux.rsquared
#     LM = T * R2
#     pval = 1.0 - stats.chi2.cdf(LM, df=k)
    
#     # HAC version (robust to heteroskedasticity and autocorrelation)
#     aux_hac = sm.OLS(u, W_star).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags_hac})
#     beta = np.asarray(aux_hac.params)
#     cov = np.asarray(aux_hac.cov_params())
#     cov_inv = np.linalg.pinv(cov)
#     LM_HAC = float(beta.T @ cov_inv @ beta)
#     pval_HAC = 1.0 - stats.chi2.cdf(LM_HAC, df=k)
    
#     return {
#         "LM": LM,
#         "p_value": pval,
#         "LM_HAC": LM_HAC,
#         "p_value_HAC": pval_HAC,
#         "df": k,
#         "nobs": T
#     }
def _nw_auto_lags(T: int) -> int:
    # Newey–West (1994) rule-of-thumb
    return int(np.floor(4 * (T / 100.0) ** (2/9)))

def lm_linearity_test_for_z(data, z):

    df = pd.DataFrame({
        "y": data["r_s"] - (data["i_for"] - data["i_dom"]),
        "rs": data["r_s"].shift(1),
        "eta": data["q"].shift(1),
        "z": z
    }).dropna()

    T = len(df)
    if T < 25:
        return None

    y = df["y"].values
    rs = df["rs"].values
    eta = df["eta"].values
    s = df["z"].values

    # Null model
    X0 = np.column_stack([np.ones(T), rs, eta])
    res0 = sm.OLS(y, X0).fit()
    SSR0 = np.sum(res0.resid**2)

    # Taylor expansion regressors
    Z1 = np.column_stack([rs*s, eta*s])
    Z2 = np.column_stack([rs*s**2, eta*s**2])
    Z3 = np.column_stack([rs*s**3, eta*s**3])

    X1 = np.column_stack([X0, Z1, Z2, Z3])
    res1 = sm.OLS(y, X1).fit()
    SSR1 = np.sum(res1.resid**2)

    q = X1.shape[1] - X0.shape[1]   # number of restrictions

    LM = T * (SSR0 - SSR1) / SSR0
    pval = 1 - stats.chi2.cdf(LM, q)

    # HAC F-version
    L = _nw_auto_lags(T)
    res1_hac = sm.OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    R = np.zeros((q, X1.shape[1]))
    R[:, X0.shape[1]:] = np.eye(q)
    wald = res1_hac.wald_test(R)

    return {
        "LM": float(LM),
        "p_value": float(pval),
        "Wald_HAC": float(wald.statistic),
        "p_value_HAC": float(wald.pvalue),
        "df": int(q),
        "nobs": int(T)
    }


def estimate_str_model(country_df: pd.DataFrame, country_name: str, verbose: bool = True) -> Optional[dict]:
    """
    Complete STR model estimation pipeline for a single country.
    
    Steps:
    1. Generate transition variable candidates
    2. Run linearity tests for all candidates
    3. Select best candidate (lowest p-value)
    4. Perform grid search for gamma and c
    5. Estimate final model with NLS
    
    Parameters
    ----------
    country_df : pd.DataFrame
        Country-specific panel data (indexed by date)
    country_name : str
        Country name for logging
    verbose : bool, default=True
        Print estimation progress
    
    Returns
    -------
    dict or None
        Dictionary with keys: 'result', 'z_name', 'lm_pval'
        Returns None if estimation fails
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {country_name}")
        print(f"{'='*60}")
    
    try:
        if len(country_df) < 50:
            if verbose:
                print(f"Insufficient data for {country_name}")
            return None
        
        # Generate z-candidates
        z_candidates = build_z_candidates(country_df)
        
        # Run linearity tests for all candidates (sorted for reproducibility)
        if verbose:
            print(f"Running linearity tests for {len(z_candidates)} candidates...")
        
        lm_results = []
        for z_name in sorted(z_candidates.keys()):  # Sort for deterministic order
            z_series = z_candidates[z_name]
            result = lm_linearity_test_for_z(country_df, z_series)
            if result is not None:
                result['z_var'] = z_name
                lm_results.append(result)
        
        if len(lm_results) == 0:
            if verbose:
                print(f"No valid linearity tests for {country_name}")
            return None
        
        # Select best z (lowest p-value indicates strongest nonlinearity)
        lm_df = pd.DataFrame(lm_results).sort_values('p_value_HAC')
        best_z_name = lm_df.iloc[0]['z_var']
        best_z = z_candidates[best_z_name]
        
        if verbose:
            print(f"Best transition variable: {best_z_name} (p={lm_df.iloc[0]['p_value_HAC']:.4f})")
        
        # Build STR estimation sample
        str_df = pd.DataFrame({
            "y": country_df["r_s"] - (country_df["i_for"] - country_df["i_dom"]),
            "rs_t1": country_df["r_s"].shift(1),
            "eta_t1": country_df["q"].shift(1),
            "z": best_z
        }).dropna()
        
        # Grid search with exponential gamma grid
        if verbose:
            print(f"Running grid search...")
        
        gamma_grid = np.logspace(np.log10(0.125), np.log10(256.0), num=120)
        zc = pd.to_numeric(str_df["z"], errors="coerce").dropna().to_numpy()
        c_grid = np.quantile(zc, np.linspace(0.05, 0.95, 120))
        
        # Initialize and estimate
        str_model = STRModel(str_df, z_col='z')
        start_params = str_model.grid_search(gamma_grid, c_grid)
        
        if verbose:
            print(f"Grid search complete. Starting NLS...")
        
        # HAC bandwidth: Set to None for automatic Newey-West (1994), or integer for manual
        str_result = str_model.fit(start_params, hac_lags=None)  # Automatic Newey-West (1994) bandwidth
        
        if verbose:
            print(f"\nSTR Results for {country_name}:")
            print(str_result.params)
            print(f"AIC: {str_result.aic:.2f}, BIC: {str_result.bic:.2f}")
        
        return {
            'result': str_result,
            'z_name': best_z_name,
            'lm_pval': lm_df.iloc[0]['p_value']
        }
        
    except Exception as e:
        if verbose:
            print(f"ERROR processing {country_name}: {e}")
            import traceback
            traceback.print_exc()
        return None


def estimate_buip_model(country_df: pd.DataFrame, country_name: str, verbose: bool = True) -> Optional[object]:
    """
    Complete BUIP model estimation pipeline for a single country.
    
    Uses multistart optimization to avoid local minima.
    
    Parameters
    ----------
    country_df : pd.DataFrame
        Country-specific panel data (indexed by date)
    country_name : str
        Country name for logging
    verbose : bool, default=True
        Print estimation progress
    
    Returns
    -------
    BuipResult or None
        Fitted BUIP model result object
        Returns None if estimation fails
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"BUIP: {country_name}")
        print(f"{'='*60}")
    
    try:
        if len(country_df) < 50:
            if verbose:
                print(f"Insufficient data for {country_name}")
            return None
        
        # Build BUIP sample
        buip_df = build_beh_sample(country_df)
        
        if verbose:
            print(f"Sample size: {len(buip_df)}")
            print(f"Running scientifically-grounded multistart optimization...")
            print(f"  Strategy: Two-stage conditional OLS + refinement")
        
        # Estimate with scientifically defensible multistart
        # HAC bandwidth: Set to None for automatic Newey-West (1994), or integer for manual
        buip_model = BUIPModel(buip_df)
        buip_result = buip_model.fit_multistart(
            n_starts=None, 
            hac_lags=None,  # Automatic Newey-West (1994) bandwidth
            method='conditional_ols'  # Use scientifically defensible approach
        )
        
        if verbose:
            print(f"\nBUIP Results for {country_name}:")
            print(buip_result.params)
            print(f"AIC: {buip_result.aic:.2f}, BIC: {buip_result.bic:.2f}")
        
        return buip_result
        
    except Exception as e:
        if verbose:
            print(f"ERROR processing {country_name}: {e}")
            import traceback
            traceback.print_exc()
        return None


def estimate_all_countries(df_panel: pd.DataFrame, countries: Dict[str, str], 
                          model_type: str = 'both') -> tuple:
    """
    Estimate STR and/or BUIP models for all countries.
    
    Parameters
    ----------
    df_panel : pd.DataFrame
        Full panel dataset with all countries
    countries : Dict[str, str]
        Dictionary mapping country names to codes
    model_type : str, default='both'
        Which models to estimate: 'str', 'buip', or 'both'
    
    Returns
    -------
    tuple
        (str_results, buip_results) dictionaries
        Empty dict if model_type doesn't include that model
    """
    str_results = {}
    buip_results = {}
    
    # Sort countries for deterministic iteration order
    for country_name in sorted(countries.keys()):
        country_df = df_panel[df_panel['country'] == country_name].copy().set_index('date').sort_index()
        
        # STR estimation
        if model_type in ['str', 'both']:
            result = estimate_str_model(country_df, country_name, verbose=True)
            if result is not None:
                str_results[country_name] = result
        
        # BUIP estimation
        if model_type in ['buip', 'both']:
            result = estimate_buip_model(country_df, country_name, verbose=True)
            if result is not None:
                buip_results[country_name] = result
    
    print(f"\n\n{'='*60}")
    print("ESTIMATION COMPLETE")
    print(f"{'='*60}")
    if str_results:
        print(f"STR estimation complete for {len(str_results)} countries")
    if buip_results:
        print(f"BUIP estimation complete for {len(buip_results)} countries")
    
    return str_results, buip_results


# ==============================================================================
# DIAGNOSTIC TESTS
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
