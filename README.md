# STR and BUIP Estimation: Exchange Rate Regime Dynamics

A production-ready Python package for estimating **Smooth Transition Regression (STR)** and **Behavioral Uncovered Interest Parity (BUIP)** models on exchange rate panel data. This repository implements the complete econometric pipeline for analyzing regime-switching behavior in foreign exchange markets across 14 countries.

## Overview

This research examines exchange rate dynamics through two complementary non-linear modeling approaches:

- **STR Model**: Smooth transitions between chartist and fundamentalist regimes using logistic transition functions
- **BUIP Model**: Behavioral UIP with utility-based mixing weights between fundamental and chartist trading rules

The package provides a complete replication pipeline from raw data construction to publication-ready tables and figures.

---

## Features

### Core Econometrics
- **Non-Linear Least Squares (NLS)** with HAC-robust standard errors (Newey-West, 4 lags)
- **Grid-based multistart optimization** (1,080+ starting values) to avoid local minima
- **Linearity testing** with 35 transition variable candidates per country
- **LM tests** (Luukkonen et al., 1988) with HAC-robust inference

### Data & Coverage
- **14 countries**: Australia, Brazil, Canada, Euro Area, Indonesia, Japan, Korea, Mexico, New Zealand, Philippines, Switzerland, Thailand, Türkiye, United Kingdom
- **4,186 observations** (2000-2025)
- **DBnomics API integration** for automated data retrieval (exchange rates, interest rates, PPP)

### Publication Outputs
- **LaTeX-ready tables** with significance stars (*, **, ***)
- **Publication-quality figures** (300 DPI, Times New Roman fonts)
- **Regime diagnostics**: Ljung-Box, ARCH, Jarque-Bera tests
- **Regime occupancy metrics**: Spell lengths, transition smoothness

---

## Installation

```bash
# Clone the repository
git clone https://github.com/lorenzobiondiecon/str-buip-estimation.git
cd str-buip-estimation

# Install dependencies
pip install -r Requirements.txt
```

**Dependencies**: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `dbnomics`

---

## Usage

### Quick Start: Run the Complete Pipeline

The `main_reproduction.ipynb` notebook contains the full analysis pipeline:

1. **Load or build the dataset** (automatically fetches from DBnomics if needed)
2. **Generate descriptive statistics** for all 14 countries
3. **Run LM linearity tests** for 35 transition variable candidates
4. **Estimate STR models** for countries showing nonlinearity
5. **Estimate BUIP models** for all countries with grid-based multistart
6. **Generate publication tables** with formatted parameters and significance stars
7. **Create publication figures** (regime distributions, time series, cross-country comparisons)

```bash
# Open the notebook
jupyter notebook main_reproduction.ipynb

# Or run cells sequentially in VS Code
```

### Alternative: Command-Line Scripts

```bash
# Step 1: Build the dataset
python build_dataset.py

# Step 2: Run estimation for all countries
python run_estimation.py
```

### Output Structure

All results are saved to `results/`:
```
results/
├── descriptive_statistics.csv
├── str_results_table.csv
├── buip_results_table.csv
├── str_regime_distribution.png/pdf
├── buip_regime_distribution.png/pdf
└── regime_prevalence_comparison.png/pdf
```

---

## Methodology

### STR Model Specification

The Smooth Transition Regression model follows Teräsvirta (1994):

$$r_t - (i_t^* - i_t) = \beta_c \cdot \eta_{t-1} + \beta_f \cdot rs_{t-1} + G(z_t; \gamma, c) \cdot [\alpha_1 + \beta_c^{(1)} \cdot \eta_{t-1} + \beta_f^{(1)} \cdot rs_{t-1}] + u_t$$

Where:
- $r_t$: Exchange rate return
- $i_t^*, i_t$: Foreign and domestic interest rates
- $\eta_t$: Real exchange rate deviation (PPP)
- $G(z_t; \gamma, c) = (1 + \exp(-\gamma(z_t - c)))^{-1}$: Logistic transition function
- $z_t$: Transition variable (endogenously selected from 35 candidates)
- $\gamma$: Smoothness parameter (larger = sharper transition)
- $c$: Threshold parameter

### BUIP Model Specification

The Behavioral UIP model with utility-based regime mixing:

$$r_t = \omega_t \cdot \beta_f \cdot (i_t^* - i_t) + (1 - \omega_t) \cdot \beta_c \cdot \text{lags} + \varepsilon_t$$

Where:
- $\omega_t$: Time-varying mixing weight (fundamentalist vs chartist)
- $\omega_t = G(d_t; \gamma, c)$ with $d_t$ as utility difference
- $\beta_f$: Fundamentalist coefficient (UIP parameter)
- $\beta_c$: Chartist coefficient (trend-following)

**Key Innovation**: The mixing weight $\omega_t$ is endogenously determined by the utility differential between strategies, creating smooth transitions between fundamentalist and chartist regimes.

### Estimation Strategy

1. **Z-candidate generation**: 7 base variables × 5 lags = 35 candidates
   - Variables: `eta`, `|eta|`, `|Δrs|`, `ID`, `|f_ppp|`, `|Δrf|`, `|rel_misalignment|`

2. **LM linearity tests**: HAC-robust tests for all 35 candidates, select best (lowest p-value)

3. **Grid search initialization**:
   - Gamma grid: 120 values (log-spaced 0.125 to 256)
   - Threshold grid: 120 quantiles (5% to 95%)
   - BUIP: 1,080 starting values (6×6×6×5 grid)

4. **NLS estimation**: Levenberg-Marquardt with Newey-West HAC covariance (4 lags)

5. **Diagnostics**: Ljung-Box (autocorrelation), ARCH (heteroskedasticity), Jarque-Bera (normality)

---

## Repository Structure

```
str-buip-estimation/
├── src/
│   ├── config.py           # Global configuration (paths, parameters)
│   ├── data_builder.py     # DBnomics data fetching and processing
│   ├── utils.py            # Data loading and sample preparation
│   ├── econometrics.py     # STR and BUIP model classes
│   ├── diagnostics.py      # LM tests, residual diagnostics, regime metrics
│   ├── output_tables.py    # LaTeX table generation
│   └── output_figures.py   # Publication-quality figure generation
├── data/
│   └── df_panel_final.csv  # (Generated by build_dataset.py)
├── results/                # (Generated output files)
├── build_dataset.py        # Script to fetch and build panel data
├── run_estimation.py       # Main estimation script
├── main_reproduction.ipynb # Complete interactive analysis notebook
├── Analysis.ipynb          # Additional analysis notebook
├── Requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Key Results Summary

From the 14-country estimation (2000-2025):

### BUIP Model
- **14/14 countries** successfully estimated
- **Regime heterogeneity**: 
  - Fundamentalist-dominant: Australia, Korea, Mexico, New Zealand, Thailand, United Kingdom
  - Mixed regimes: Brazil, Euro Area, Philippines, Türkiye
  - Chartist-dominant: Canada, Indonesia, Japan, Switzerland
- **Mean omega** ranges from 0.024 (Japan) to 1.000 (Australia)

### STR Model
- **Linearity tests**: 19/35 candidates significant at 5% (Australia example)
- **Transition smoothness** varies substantially across countries
- **Regime persistence**: Average spell lengths range from 2-20 periods

---

## Citation

If you use this code in your research, please cite:

```
[Your Name]. (2025). STR and BUIP Estimation: Exchange Rate Regime Dynamics.
GitHub: https://github.com/lorenzobiondiecon/str-buip-estimation
```

---

## References

- Teräsvirta, T. (1994). "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models." *Journal of the American Statistical Association*, 89(425), 208-218.
- Luukkonen, R., Saikkonen, P., & Teräsvirta, T. (1988). "Testing Linearity Against Smooth Transition Autoregressive Models." *Biometrika*, 75(3), 491-499.
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708.

---

## License

MIT License - see LICENSE file for details

---

## Contact

Lorenzo Biondi  
Email: [your-email]  
GitHub: [@lorenzobiondiecon](https://github.com/lorenzobiondiecon)

