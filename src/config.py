import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

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

    # Covid sample exclusion
    # Set True to drop all observations after COVID_CUTOFF; results go to results/no-covid/
    EXCLUDE_POST_COVID = False
    COVID_CUTOFF = "2019-12-31"  # Last date kept when EXCLUDE_POST_COVID = True

    # Per-country start date truncation for robustness checks.
    # Format: {"Country Name": "YYYY-MM-DD"}. None means no truncation.
    COUNTRY_START_DATES: Dict[str, str] = None

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

# Country mappings (name -> ISO code)
COUNTRIES: Dict[str, str] = {
    "Australia": "AUS",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Euro Area": "EA",
    "Indonesia": "IND",
    "Japan": "JPN",
    "Korea": "KOR",
    "Mexico": "MEX",
    "New Zealand": "NZL",
    "Philippines": "PHL",
    "Switzerland": "CHE",
    "Thailand": "THA",
    "Türkiye": "TUR",
    "United Kingdom": "GBR"
}

# Data sources by country
DATA_SOURCES: Dict[str, str] = {
    'Australia': 'IMF/IFS (S, CPI interp., i)',
    'Brazil': 'IMF/IFS (S, CPI, MM rate)',
    'Canada': 'IMF/IFS (S, CPI), OECD (i)',
    'Euro Area': 'IMF/IFS (S, CPI), OECD (i)',
    'Indonesia': 'IMF/IFS (S, CPI, i combined)',
    'Japan': 'IMF/IFS (S, CPI), OECD (i)',
    'Korea': 'IMF/IFS (S, CPI, MM rate)',
    'Mexico': 'IMF/IFS (S, CPI, MM rate)',
    'New Zealand': 'IMF/IFS (S, CPI interp., MM rate)',
    'Philippines': 'IMF/IFS (S, CPI, MM rate)',
    'Switzerland': 'IMF/IFS (S, CPI), OECD (i)',
    'Thailand': 'IMF/IFS (S, CPI, MM rate)',
    'Türkiye': 'IMF/IFS (S, CPI, i)',
    'United Kingdom': 'IMF/IFS (S, CPI), OECD (i)'
}