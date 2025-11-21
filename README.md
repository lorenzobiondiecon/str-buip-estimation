# STR-BUIP Estimation

This repository contains the code for estimating Smooth Transition Regression (STR) models and investigating the exchange rate dynamics and their departures from the PPP theory across countries.

## Overview

The project automates the process of:
1.  **Data Collection**: Fetching macroeconomic time-series (Exchange Rates, Interest Rates, CPI) from DBnomics client (sources: IMF, OECD).
2.  **Data Processing**: Constructing a clean panel dataset with necessary transformations (log-differences, real exchange rates, etc.).
3.  **Econometric Analysis**: Following the methodology of Teräsvirta (1994) and Granger & Teräsvirta (1993):
    -   **Specification**: Testing for non-linearity and selecting the appropriate transition variable.
    -   **Estimation**: Estimating Smooth Transition Regression models to capture regime-switching behavior in exchange rates. Extending the analysis to include utilities computation of the tradeing rules as in Proaño (2013).
    -   **Evaluation**: Testing residuals for ARCH effects, autocorrelation and non-normality.

## Project Structure

-   `build_dataset.py`: Script to fetch raw data and build the final panel dataset (`data/df_panel_final.csv`).
-   `run_estimation.py`: Main script to run the econometric estimation pipeline and generate results.
-   `main_reproduction.ipynb`: Jupyter notebook for interactive reproduction of the results.
-   `src/`: Source code for data building, econometrics models, and utilities.
-   `results/`: Directory where output tables and figures are saved.

## Installation

Ensure you have Python installed. It is recommended to use a virtual environment.

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r Requirements.txt
```

## Usage

### 1. Build the Dataset

To download the latest data and construct the panel:

```bash
python build_dataset.py
```

This will generate `data/df_panel_final.csv`.

### 2. Run Estimations

To run the full estimation pipeline (STR models, linearity tests, etc.):

```bash
python run_estimation.py
```

Results will be saved in the `results/` directory.

### 3. Interactive Reproduction

Open `main_reproduction.ipynb` in Jupyter Lab or VS Code to run the analysis step-by-step and visualize the results.

## References

**Methodology:**
-   Teräsvirta, T. (1994). *Specification, estimation, and evaluation of smooth transition autoregressive models*. Journal of the American Statistical Association, 89(425), 208-218.
-   Granger, C. W., & Teräsvirta, T. (1993). *Modelling non-linear economic relationships*. Oxford University Press.
- Teräsvirta, T., Tjøstheim, D., and Granger, C. (2010). *Modelling nonlinear economic time series*. Oxford University Press.

**Economic Theory (BUIP):**
- Proaño, C. R. (2013). *Monetary policy rules and macroeconomic stabilization in small open economies under behavioral fx trading: Insights from numerical simulations*. The Manchester School, 81(6):992–1011.