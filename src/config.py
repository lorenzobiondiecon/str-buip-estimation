import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Global configuration for the econometric pipeline."""
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "results"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Econometric Settings
    P_MAX_HAC = 4          # Lag length for Newey-West errors
    GRID_POINTS = 30       # Density of grid for gamma/c search
    TRIM_PERCENT = 0.15    # Trimming for grid search (15% - 85% quantiles)
    
    # Optimization
    MAX_N_EVALS = 1000     # Max iterations for NLS
    TOLERANCE = 1e-8       # Convergence tolerance
    
    # Model Toggles
    ENABLE_PLOTS = True
    SAVE_LATEX = True

config = Config()