import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def load_panel_data(filepath: str, date_col: str = 'Date', cross_section_col: str = 'Country') -> pd.DataFrame:
    """Loads and sorts panel data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file at {filepath}")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=[cross_section_col, date_col])
    
    return df

def build_str_data(df: pd.DataFrame, y_col: str, z_col: str) -> pd.DataFrame:
    """
    Prepares data specifically for the STR model logic:
    Needs: y, z, rs_lag1, eta_lag1
    """
    # Assumes the dataframe already has the necessary lags. 
    # If not, you would create them here.
    req_cols = [y_col, z_col, "rs_lag1", "eta_lag1"]
    
    # Check existence
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"STR Data missing columns: {missing}")
        
    data = df[req_cols].copy()
    data.columns = ["y", "z", "rs_lag1", "eta_lag1"]
    data = data.dropna()
    
    if data.empty:
        raise ValueError("STR dataset empty after dropping NaNs")
        
    return data

def build_beh_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the sample for BUIP with specific utility calculation lags.
    Matches the user logic for 'y' calculation and t-2, t-3 lags.
    """
    req_cols = ["r_s", "i_for", "i_dom", "q", "S", "s"]
    for col in req_cols:
        if col not in df.columns:
            raise KeyError(f"BUIP Missing required column: {col}")

    x = pd.DataFrame(index=df.index)

    # 1. Dependent Variable: r_s - (i* - i)
    x["y"] = df["r_s"] - (df["i_for"] - df["i_dom"])

    # 2. Lags for Utility Calculation (Expectations at t-2, etc)
    x["eta_lag1"] = df["q"].shift(1)
    x["r_lag1"]   = df["r_s"].shift(1)
    x["s_t1"]     = df["s"].shift(1)
    x["s_t2"]     = df["s"].shift(2) # Used for expectations
    x["eta_t2"]   = df["q"].shift(2)
    x["r_t2"]     = df["r_s"].shift(2)
    
    # 3. Lags for Profit Calculation
    x["i_dom_t1"] = df["i_dom"].shift(1)
    x["i_for_t1"] = df["i_for"].shift(1)

    # Clean
    x = x.dropna()
    if x.empty:
        raise ValueError("BUIP dataset empty after lagging/dropping NaNs.")
        
    return x
