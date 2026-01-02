# STR-BUIP Exchange Rate Estimation

This repository contains the code for estimating **Smooth Transition Regression (STR)** and **Behavioral Uncovered Interest Parity (BUIP)** models to investigate exchange rate dynamics across 14 countries.

## Overview

The project automates the complete econometric pipeline:

1.  **Data Collection**: Fetches macroeconomic time-series (Exchange Rates, Interest Rates, CPI) from DBnomics (sources: IMF, OECD).
2.  **Data Processing**: Constructs a clean panel dataset with necessary transformations (log-differences, real exchange rates, PPP deviations).
3.  **Econometric Analysis**:
    -   **STR Models**: Tests for non-linearity via LM tests, selects optimal transition variables, estimates regime-switching models
    -   **BUIP Models**: Estimates behavioral models with utility-based regime switching between chartist and fundamentalist strategies
4.  **Results Generation**: Produces publication-ready tables (LaTeX + CSV) and figures (PDF + PNG)

## Project Structure

```
├── main_reproduction.ipynb    # Main notebook - run this to reproduce all results
├── Requirements.txt           # Python dependencies
├── src/
│   ├── notebook_setup.py     # Consolidated imports and notebook initialization
│   ├── config.py             # Configuration settings and country definitions
│   ├── analysis.py           # Estimation functions and diagnostic tests
│   ├── data_builder.py       # Data fetching and processing from DBnomics
│   ├── econometrics.py       # STR and BUIP model classes
│   ├── output_figures.py     # Figure generation functions
│   ├── output_tables.py      # Table generation utilities
│   └── utils.py              # Helper functions (data loading, etc.)
├── data/
│   ├── df_panel_final.csv    # Restricted dataset (2000-2024)
│   └── df_panel_max.csv      # Maximum-length dataset (country-specific periods)
└── results/
    ├── restricted/           # Results for restricted dataset
    │   ├── tables/          # CSV + LaTeX tables
    │   └── figures/         # PDF + PNG figures
    └── max/                  # Results for maximum-length dataset
        ├── tables/
        └── figures/
```

## Key Features

- **Clean Notebook Interface**: Single notebook (`main_reproduction.ipynb`) with all complex functions modularized in `src/`
- **Automatic Setup**: `notebook_setup.py` handles all imports and configuration in one line
- **Dataset Flexibility**: Choose between restricted (2000-2024) or maximum-length (country-specific) datasets
- **Automatic Transition Variable Selection**: Tests 35 candidates per country, selects best via LM linearity test
- **Publication-Ready Outputs**: LaTeX tables with bold significance markers, high-quality PDF/PNG figures
- **Two Model Specifications**: STR (restricted) and BUIP (behavioral)
- **Comprehensive Diagnostics**: Ljung-Box, ARCH, and Jarque-Bera tests for all models

## Quick Start

### Interactive Notebook (Recommended)

The easiest way to reproduce all results:

```bash
# 1. Install dependencies
pip install -r Requirements.txt

# 2. Open the main notebook
jupyter notebook main_reproduction.ipynb

# 3. Run all cells sequentially (or "Run All")
```

**What happens when you run the notebook:**
- Automatically loads or builds the dataset from DBnomics
- Estimates STR and BUIP models for all 14 countries (~15-30 minutes)
- Generates all tables (CSV + LaTeX) and figures (PDF + PNG)
- Provides progress updates and diagnostics
- Saves results to `results/max/` or `results/restricted/`

**Configuration options:**
- Set `USE_MAX_DATASET = True/False` to choose dataset mode
- Modify `model_type='both'` to run only `'str'` or `'buip'` models
- Test mode available: Run with 2 countries first (~3 minutes)

## Understanding the Code Structure

### For Users (Replication)

**Just open and run `main_reproduction.ipynb`** - it's self-contained and well-documented. All technical details are handled behind the scenes.

The notebook uses a streamlined setup:
```python
from src.notebook_setup import (
    np, pd, plt, config, COUNTRIES,
    load_dataset, estimate_all_countries
)
```

### For Developers (Modification)

**Module Organization:**

- **`src/notebook_setup.py`**: One-line setup for notebooks
  - Consolidates all imports (numpy, pandas, matplotlib, etc.)
  - Provides `setup_notebook()` and helper functions
  - Exports `COUNTRIES` and `DATA_SOURCES` from config

- **`src/config.py`**: Central configuration
  - Paths (data, results directories)
  - Econometric settings (HAC lags, grid points, tolerance)
  - Country definitions and data sources

- **`src/analysis.py`**: Core estimation and diagnostics
  - `build_z_candidates()`: Generates 35 transition variable candidates
  - `lm_linearity_test_for_z()`: LM linearity test with HAC standard errors
  - `estimate_str_model()`: Complete STR estimation pipeline
  - `estimate_buip_model()`: Complete BUIP estimation pipeline
  - `estimate_all_countries()`: Batch estimation wrapper
  - Diagnostic tests: `ljung_box_test()`, `arch_test()`, `jarque_bera_test()`

- **`src/econometrics.py`**: Model implementations
  - `STRModel`: Smooth Transition Regression (LSTR1)
  - `BUIPModel`: Behavioral UIP with utility-based switching

- **`src/utils.py`**: Helper functions
  - `load_dataset()`: Unified dataset loading with auto-build
  - `build_str_data()`, `build_beh_sample()`: Data preparation

- **`src/data_builder.py`**: Data fetching from DBnomics
  - Fetches from IMF/IFS and OECD APIs
  - Handles interpolation and data gaps
  - Builds both restricted and maximum-length datasets

- **`src/output_figures.py`**: Publication-quality plotting
- **`src/output_tables.py`**: LaTeX table generation

## Output Files

After running the notebook, results are saved to `results/max/` or `results/restricted/` depending on dataset choice.

**Tables** (in `{mode}/tables/`):
- `data_availability_sources`: Data coverage and sources by country
- `descriptive_statistics`: Summary statistics for all variables
- `transition_variable_selection`: Selected transition variables with LM test p-values
- `str_results_table`: STR parameter estimates with significance stars
- `buip_results_table`: BUIP parameter estimates with significance stars
- `str_diagnostics_table`: STR residual diagnostics (Ljung-Box, ARCH, Jarque-Bera)
- `buip_diagnostics_table`: BUIP residual diagnostics
- `str_threshold_validation`: Gamma and threshold parameter validation
- `buip_threshold_validation`: Cost parameter validation
- `comprehensive_model_comparison`: AIC, BIC, RMSE comparison across models

All tables available in both `.csv` and `.tex` formats.

**Aggregate Figures** (in `{mode}/figures/aggregate/`):
- `str_regime_distribution`: Histogram of STR regime probabilities
- `buip_regime_distribution`: Histogram of BUIP regime probabilities
- `regime_prevalence_comparison`: Chartist vs Fundamentalist prevalence
- `str_gamma_comparison`: Transition smoothness by country
- `buip_gamma_comparison`: Behavioral switching speed by country
- `str_buip_rmse_comparison`: Model fit comparison
- `log_nominal_vs_ppp_6panel`: Exchange rate vs PPP comparison
14 countries spanning advanced and emerging economies:

**Advanced:** Australia, Canada, Euro Area, Japan, New Zealand, Switzerland, United Kingdom

**Emerging:** Brazil, Indonesia, Korea, Mexico, Philippines, Thailand, Türkiye

Country definitions and ISO codes are centrally managed in `src/config.py`.

## Dataset Modes

**Restricted Dataset** (`df_panel_final.csv`):
- Common period: 2000-01 to 2024-12
- Consistent comparison across all countries
- Recommended for cross-country analysis

**Maximum-Length Dataset** (`df_panel_max.csv`):
- Country-specific periods based on data availability
- Maximizes sample size per country
- Custom handling for Indonesia, Korea, Philippines (data gaps)
- Useful for within-country analysis

Switch between modes by setting `USE_MAX_DATASET = True/False` in the notebook.
- Korea
- MTechnical Details

**Estimation Methods:**
- **Grid Search**: Initial parameter search over gamma (0.125-256) and c (5th-95th percentiles)
- **Nonlinear Least Squares**: Refinement with HAC-robust standard errors
- **Multistart Optimization**: BUIP models use multiple random starts to avoid local minima
- **HAC Standard Errors**: Newey-West (1994) automatic bandwidth selection

**Model Diagnostics:**
- Ljung-Box Q-test for autocorrelation
- ARCH test for conditional heteroskedasticity
- Jarque-Bera test for normality
- Regime occupancy and persistence metrics

**Computational Performance:**
- ~1-2 minutes per country (STR + BUIP)
- Total runtime: 15-30 minutes for 14 countries
- Parallel processing not currently implemented but feasible

## Troubleshooting

**ImportError after code changes:**
- Restart Jupyter kernel (Kernel → Restart)
- The notebook includes `%autoreload` to minimize this issue

**Missing data or build errors:**
- Delete files in `data/` folder to trigger fresh download from DBnomics
- Check internet connection (requires API access)

**Estimation failures:**
- Check cell output for country-specific error messages
- Some countries may fail due to insufficient data or convergence issues
- Results are saved for successful estimations

**Module not found:**
- Ensure you're in the project root directory
- Verify all files in `src/` are present
- Run `pip install -r Requirements.txt` again


## References

**Methodology:**
-   Teräsvirta, T. (1994). *Specification, estimation, and evaluation of smooth transition autoregressive models*. Journal of the American Statistical Association, 89(425), 208-218.
-   Granger, C. W., & Teräsvirta, T. (1993). *Modelling non-linear economic relationships*. Oxford University Press.
- Teräsvirta, T., Tjøstheim, D., and Granger, C. (2010). *Modelling nonlinear economic time series*. Oxford University Press.

**Economic Theory (BUIP):**
- Proaño, C. R. (2013). *Monetary policy rules and macroeconomic stabilization in small open economies under behavioral fx trading: Insights from numerical simulations*. The Manchester School, 81(6):992–1011.

## License

See LICENSE file for details.