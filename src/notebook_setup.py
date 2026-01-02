"""
Common setup utilities for Jupyter notebooks.
Consolidates imports, plotting styles, and configuration for reproducibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import local modules
from .config import config, COUNTRIES, DATA_SOURCES
from .data_builder import DataBuilder
from .analysis import build_z_candidates, estimate_all_countries
from .utils import load_dataset

# Set publication-quality plotting style (black & white)
PLOT_STYLE = {
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': False,
}


def setup_notebook(title: str = "STR-BUIP Analysis") -> None:
    """
    Initialize notebook environment with standard imports and styling.
    
    Parameters
    ----------
    title : str
        Title to display at notebook startup
    """
    # Apply plotting style
    plt.rcParams.update(PLOT_STYLE)
    
    # Print welcome message
    print("=" * 60)
    print(title)
    print("=" * 60)
    print("\n✓ Environment configured")
    print(f"✓ {len(COUNTRIES)} countries available")
    print(f"✓ Output directory: {config.OUTPUT_DIR}")
    print("\nReady to run analysis!")


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print formatted summary of the panel dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel dataset with 'date' and 'country' columns
    """
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total observations: {len(df):,}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nObservations per country:")
    print(df['country'].value_counts().sort_index())


# Export all commonly used items
__all__ = [
    'np', 'pd', 'Path', 'plt',
    'config', 'COUNTRIES', 'DATA_SOURCES',
    'DataBuilder', 'build_z_candidates', 'estimate_all_countries',
    'load_dataset', 'setup_notebook', 'print_dataset_summary'
]
