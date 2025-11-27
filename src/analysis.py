"""
Analysis functions for STR and BUIP estimation pipeline.
Contains helper functions for z-candidate selection, linearity testing, and model estimation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Optional
from .econometrics import STRModel, BUIPModel
from .utils import build_beh_sample


def build_z_candidates(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Generates 35 transition variable candidates for STR models.
    
    Creates candidates from 7 base variables × 5 lags each:
    - eta: Real exchange rate (q)
    - eta_abs: Absolute value of real exchange rate
    - drs_abs: Absolute value of exchange rate returns
    - ID: Interest rate differential
    - ppp_abs: Absolute value of PPP deviation
    - drf_abs: Absolute value of fundamental returns
    - rel_misalignment_abs: Absolute relative misalignment
    
    Parameters
    ----------
    data : pd.DataFrame
        Country-specific panel data with columns: q, r_s, i_for, i_dom, f_ppp, f_ppp_rel
    
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary mapping candidate names (e.g., 'eta_lag1') to their Series
    """
    z_base = {
        "eta": data["q"],
        "eta_abs": np.abs(data["q"]),
        "drs_abs": np.abs(data["r_s"]),
        "ID": (data["i_for"] - data["i_dom"]),
        "ppp_abs": np.abs(data["f_ppp"]),
        "drf_abs": np.abs(data["f_ppp_rel"]),
        "rel_misalignment_abs": np.abs(data["f_ppp_rel"] - data["r_s"]),
        "excess_returns": data["r_s"] - (data["i_for"] - data["i_dom"])
    }
    
    candidates = {}
    for name, series in z_base.items():
        for lag in range(1, 6):  # lags 1-5
            candidates[f"{name}_lag{lag}"] = series.shift(lag)
    
    return candidates


def lm_linearity_test_for_z(data: pd.DataFrame, z: pd.Series, maxlags_hac: int = 4) -> Optional[dict]:
    """
    Performs LM linearity test for a specific transition variable.
    
    Tests H0: Linear model vs H1: STR model using Taylor expansion approximation.
    Uses HAC-robust standard errors for the test statistic.
    
    Parameters
    ----------
    data : pd.DataFrame
        Country-specific panel data
    z : pd.Series
        Candidate transition variable
    maxlags_hac : int, default=4
        Number of lags for HAC covariance estimation
    
    Returns
    -------
    dict or None
        Test results with keys: LM, p_value, LM_HAC, p_value_HAC, df, nobs
        Returns None if insufficient data
    """
    # Build base linear model
    df = pd.DataFrame({
        "y": data["r_s"] - (data["i_for"] - data["i_dom"]),
        "rs_lag1": data["r_s"].shift(1),
        "eta_lag1": data["q"].shift(1),
    }).dropna()
    
    # Join with z
    df = df.join(z.rename("z"), how="inner").dropna()
    
    if len(df) < 20:
        return None
    
    T = len(df)
    y = df["y"].to_numpy()
    X = df[["rs_lag1", "eta_lag1"]].to_numpy()
    
    # Linear model residuals
    lin_res = sm.OLS(y, X).fit()
    u = lin_res.resid
    
    # Taylor expansion terms (3rd order)
    zc = df["z"].to_numpy()
    z1, z2, z3 = zc, zc**2, zc**3
    
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
    k = W.shape[1]
    
    # Project out X
    XTX_inv = np.linalg.pinv(X.T @ X)
    P_X = X @ XTX_inv @ X.T
    M_X = np.eye(T) - P_X
    W_star = M_X @ W
    
    # Auxiliary regression
    aux = sm.OLS(u, W_star).fit()
    R2 = aux.rsquared
    LM = T * R2
    pval = 1.0 - stats.chi2.cdf(LM, df=k)
    
    # HAC version (robust to heteroskedasticity and autocorrelation)
    aux_hac = sm.OLS(u, W_star).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags_hac})
    beta = np.asarray(aux_hac.params)
    cov = np.asarray(aux_hac.cov_params())
    cov_inv = np.linalg.pinv(cov)
    LM_HAC = float(beta.T @ cov_inv @ beta)
    pval_HAC = 1.0 - stats.chi2.cdf(LM_HAC, df=k)
    
    return {
        "LM": LM,
        "p_value": pval,
        "LM_HAC": LM_HAC,
        "p_value_HAC": pval_HAC,
        "df": k,
        "nobs": T
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
        
        # Run linearity tests for all candidates
        if verbose:
            print(f"Running linearity tests for {len(z_candidates)} candidates...")
        
        lm_results = []
        for z_name, z_series in z_candidates.items():
            result = lm_linearity_test_for_z(country_df, z_series, maxlags_hac=4)
            if result is not None:
                result['z_var'] = z_name
                lm_results.append(result)
        
        if len(lm_results) == 0:
            if verbose:
                print(f"No valid linearity tests for {country_name}")
            return None
        
        # Select best z (lowest p-value indicates strongest nonlinearity)
        lm_df = pd.DataFrame(lm_results).sort_values('p_value')
        best_z_name = lm_df.iloc[0]['z_var']
        best_z = z_candidates[best_z_name]
        
        if verbose:
            print(f"Best transition variable: {best_z_name} (p={lm_df.iloc[0]['p_value']:.4f})")
        
        # Build STR estimation sample
        str_df = pd.DataFrame({
            "y": country_df["r_s"] - (country_df["i_for"] - country_df["i_dom"]),
            "rs_lag1": country_df["r_s"].shift(1),
            "eta_lag1": country_df["q"].shift(1),
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
        
        str_result = str_model.fit(start_params, hac_lags=4)
        
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
            print(f"Running multistart optimization...")
        
        # Estimate with multistart
        buip_model = BUIPModel(buip_df)
        buip_result = buip_model.fit_multistart(n_starts=None, hac_lags=4)
        
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
    
    for country_name in countries.keys():
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
