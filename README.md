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
├── main_reproduction.ipynb    # Main notebook for reproducing all results (CLEAN & USER-FRIENDLY)
├── build_dataset.py           # Script to fetch and build panel dataset
├── run_estimation.py          # Command-line script for batch estimation
├── src/
│   ├── analysis.py           # Analysis functions (z-candidates, LM tests, estimation)
│   ├── data_builder.py       # Data fetching and processing
│   ├── econometrics.py       # STR and BUIP model classes
│   ├── output_figures.py     # Figure generation functions
│   ├── output_tables.py      # Table generation functions
│   └── utils.py              # Helper utilities
├── data/
│   └── df_panel_final.csv    # Panel dataset (generated)
└── results/
    ├── tables/               # Results tables (CSV + LaTeX)
    └── figures/
        ├── aggregate/        # Cross-country summary figures
        └── country/          # Country-specific detailed plots
```

## Key Features

- **Clean Notebook Interface**: `main_reproduction.ipynb` is streamlined with all complex functions moved to `src/analysis.py`
- **Automatic Transition Variable Selection**: Tests 35 candidates per country, selects best via LM linearity test
- **Publication-Ready Outputs**: LaTeX tables with bold significance markers, high-quality PDF/PNG figures
- **Two Model Specifications**: STR (restricted) and BUIP (behavioral)

## Quick Start

### Option 1: Interactive Notebook (Recommended)

The easiest way to reproduce all results:

```bash
# 1. Install dependencies
pip install -r Requirements.txt

# 2. Open the main notebook
jupyter notebook main_reproduction.ipynb

# 3. Run all cells sequentially
```

The notebook will:
- Build the dataset (if not already present)
- Estimate STR and BUIP models for all countries
- Generate all tables and figures
- Print a summary of outputs

### Option 2: Command Line

For batch processing without Jupyter:

```bash
# Build dataset
python build_dataset.py

# Run estimations
python run_estimation.py
```

## Understanding the Code Structure

### For Users (Replication)

**Just use `main_reproduction.ipynb`** - it's clean and well-commented. You don't need to look at other files.

### For Developers (Modification)

If you want to modify the estimation procedures or add new features:

- **`src/analysis.py`**: Contains all estimation functions
  - `build_z_candidates()`: Generates 35 transition variable candidates
  - `lm_linearity_test_for_z()`: Tests for non-linearity
  - `estimate_str_model()`: Complete STR estimation pipeline
  - `estimate_buip_model()`: Complete BUIP estimation pipeline
  - `estimate_all_countries()`: Batch estimation wrapper

- **`src/econometrics.py`**: Core model classes
  - `STRModel`: Smooth Transition Regression
  - `BUIPModel`: Behavioral UIP

- **`src/output_figures.py`**: All plotting functions
- **`src/output_tables.py`**: Table generation utilities

## Output Files

After running the notebook, you'll find:

**Tables** (in `results/tables/`):
- `descriptive_statistics.csv/.tex`: Summary statistics
- `transition_variable_selection.csv/.tex`: Selected transition variables and LM test p-values
- `str_results_table.csv/.tex`: STR parameter estimates
- `buip_results_table.csv/.tex`: BUIP parameter estimates

**Aggregate Figures** (in `results/figures/aggregate/`):
- `str_regime_distribution.pdf/.png`: STR regime distribution
- `buip_regime_distribution.pdf/.png`: BUIP regime distribution  
- `regime_prevalence_comparison.pdf/.png`: Chartist vs Fundamentalist prevalence
- `str_gamma_comparison.pdf/.png`: Gamma parameters by country (STR)
- `buip_gamma_comparison.pdf/.png`: Gamma parameters by country (BUIP)
- `str_buip_rmse_comparison.pdf/.png`: Model fit comparison

**Country Plots** (in `results/figures/country/`):
- `{Country}_str_detailed.pdf`: STR transition function and regime evolution
- `{Country}_buip_detailed.pdf`: BUIP utility differential and mixing weights

## Countries Included

- Australia
- Brazil  
- Canada
- Euro Area
- Indonesia
- Japan
- Korea
- Mexico
- New Zealand
- Philippines
- Switzerland
- Thailand
- Türkiye
- United Kingdom

## References

**Methodology:**
-   Teräsvirta, T. (1994). *Specification, estimation, and evaluation of smooth transition autoregressive models*. Journal of the American Statistical Association, 89(425), 208-218.
-   Granger, C. W., & Teräsvirta, T. (1993). *Modelling non-linear economic relationships*. Oxford University Press.
- Teräsvirta, T., Tjøstheim, D., and Granger, C. (2010). *Modelling nonlinear economic time series*. Oxford University Press.

**Economic Theory (BUIP):**
- Proaño, C. R. (2013). *Monetary policy rules and macroeconomic stabilization in small open economies under behavioral fx trading: Insights from numerical simulations*. The Manchester School, 81(6):992–1011.

## License

See LICENSE file for details.