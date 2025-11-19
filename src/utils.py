import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def load_panel_data(filepath: str, date_col: str = 'Date', cross_section_col: str = 'Country') -> pd.DataFrame:
    """
    Loads panel data and performs basic cleaning.
    """
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[cross_section_col, date_col])
    return df

def prepare_regression_matrices(df: pd.DataFrame, 
                                y_col: str, 
                                linear_vars: List[str], 
                                nonlinear_vars: List[str], 
                                z_col: str) -> dict:
    """
    Extracts numpy arrays for regression from DataFrame.
    Handles missing values by dropping rows.
    """
    cols_needed = [y_col] + list(set(linear_vars + nonlinear_vars + [z_col]))
    
    data = df[cols_needed].dropna()
    
    y = data[y_col].values
    X_lin = sm.add_constant(data[linear_vars]).values
    X_non = sm.add_constant(data[nonlinear_vars]).values
    z = data[z_col].values
    
    return {
        "y": y,
        "X_lin": X_lin,
        "X_non": X_non,
        "z": z,
        "index": data.index
    }

import statsmodels.api as sm
