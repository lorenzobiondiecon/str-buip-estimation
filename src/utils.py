import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
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

def build_str_data(df: pd.DataFrame, y_col: str = 'r_s', z_col: str = 'q') -> pd.DataFrame:
    """
    Prepares data specifically for the STR model logic:
    Needs: y, z, rs_t1, eta_t1
    Creates lags if they don't exist.
    """
    # Create lags if not present
    if 'rs_t1' not in df.columns and 'r_s' in df.columns:
        df = df.copy()
        df['rs_t1'] = df['r_s'].shift(1)
    
    if 'eta_t1' not in df.columns and 'q' in df.columns:
        df = df.copy()
        df['eta_t1'] = df['q'].shift(1)
    
    req_cols = [y_col, z_col, "rs_t1", "eta_t1"]
    
    # Check existence
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"STR Data missing columns: {missing}")
        
    data = df[req_cols].copy()
    data.columns = ["y", "z", "rs_t1", "eta_t1"]
    data = data.dropna()
    
    if data.empty:
        raise ValueError("STR dataset empty after dropping NaNs")
        
    return data

def build_beh_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the sample for BUIP with specific utility calculation lags.
    Matches the legacy implementation exactly with S_t1, S_t2, S_t3 construction.
    """
    req_cols = ["r_s", "i_for", "i_dom", "q", "S", "s"]
    for col in req_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column for BUIP: {col}")
    
    x = pd.DataFrame(index=df.index)
    x["y"]        = df["r_s"] - (df["i_for"] - df["i_dom"])
    x["eta_t1"] = df["q"].shift(1)
    x["rs_t1"]   = df["r_s"].shift(1)
    x["S_t1"]     = df["S"].shift(1)
    x["S_t2"]     = df["S"].shift(2)
    x["S_t3"]     = df["S"].shift(3)
    x["s_t1"]     = df["s"].shift(1)
    x["s_t2"]     = df["s"].shift(2)
    x["s_t3"]     = df["s"].shift(3)
    x["rs_t2"]     = df["r_s"].shift(2)
    x["rs_t3"]     = df["r_s"].shift(3)
    x["eta_t2"]   = df["q"].shift(2)
    x["eta_t3"]   = df["q"].shift(3)
    x["i_dom_t1"] = df["i_dom"].shift(1)
    x["i_for_t1"] = df["i_for"].shift(1)
    x = x.dropna()
    
    if len(x) == 0:
        raise ValueError("After lagging, BUIP sample is empty.")
    
    return x


def load_dataset(use_max: bool = False, label: str = None, exclude_post_covid: bool = None) -> Tuple[pd.DataFrame, Path, Path, Path, Path]:
    """
    Load or build dataset with automatic directory setup.

    Consolidates the dataset loading logic used across notebooks.

    Parameters
    ----------
    use_max : bool, default=False
        If True, load/build maximum-length dataset.
        If False, load/build restricted dataset (2000-2024).
    label : str, optional
        Override the output subdirectory name. Useful for robustness checks
        that need their own results folder (e.g. 'truncated'). Defaults to
        the dataset mode ('max' or 'restricted').
    exclude_post_covid : bool, optional
        If True, drop observations after config.COVID_CUTOFF and write results
        to results/no-covid/. If None (default), use config.EXCLUDE_POST_COVID.

    Returns
    -------
    tuple
        (df_panel, output_dir, tables_dir, agg_fig_dir, country_fig_dir)
        - df_panel: Panel dataset
        - output_dir: Base output directory for this mode
        - tables_dir: Directory for tables
        - agg_fig_dir: Directory for aggregate figures
        - country_fig_dir: Directory for country-specific figures
    """
    from .config import config
    from .data_builder import DataBuilder

    if exclude_post_covid is None:
        exclude_post_covid = config.EXCLUDE_POST_COVID

    # Determine dataset path
    if use_max:
        data_path = config.DATA_DIR / "df_panel_max.csv"
        mode = "max"
    else:
        data_path = config.DATA_DIR / "df_panel_final.csv"
        mode = "restricted"

    # Load or build dataset
    if not data_path.exists():
        logger.info(f"Building {mode} dataset from DBnomics...")
        builder = DataBuilder()
        df_panel = builder.run(build_mode=mode)
    else:
        logger.info(f"Loading existing dataset from {data_path}")
        df_panel = pd.read_csv(data_path, parse_dates=["date"])

    # Optionally drop post-covid observations
    if exclude_post_covid:
        cutoff = pd.Timestamp(config.COVID_CUTOFF)
        df_panel = df_panel[df_panel["date"] <= cutoff]
        logger.info(f"Post-covid exclusion active: sample truncated at {config.COVID_CUTOFF}")

    # Configure output directories
    subdir = label if label else mode
    if exclude_post_covid:
        output_dir = config.OUTPUT_DIR / "no-covid" / subdir
    else:
        output_dir = config.OUTPUT_DIR / subdir
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'
    agg_fig_dir = figures_dir / 'aggregate'
    country_fig_dir = figures_dir / 'country'
    
    # Create directories
    for d in [tables_dir, agg_fig_dir, country_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directories configured:")
    logger.info(f"  Tables: {tables_dir}")
    logger.info(f"  Aggregate Figures: {agg_fig_dir}")
    logger.info(f"  Country Figures: {country_fig_dir}")
    
    return df_panel, output_dir, tables_dir, agg_fig_dir, country_fig_dir

