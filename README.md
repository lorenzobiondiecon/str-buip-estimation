# Behavioral FX Dynamics: Replication Code

Replication materials for **"Behavioral FX Dynamics"** (Biondi and Proaño,
2026). The repository estimates Smooth Transition Regression (STR) and
Behavioral Uncovered Interest Parity (BUIP) models for monthly bilateral
USD exchange rates of 14 advanced and emerging economies, and reproduces
all empirical results, tables, and figures in the paper.

## What this repository contains

```
.
├── main_reproduction.ipynb     Single entry point. Runs the full pipeline.
├── Requirements.txt            Python dependencies.
├── data/
│   └── df_panel_max.csv        Panel dataset (snapshot used in the paper).
├── results/
│   └── max/                    All paper outputs.
│       ├── tables/             10 tables, both .csv and .tex.
│       └── figures/
│           ├── aggregate/      10 cross-country figures.
│           └── country/        56 country-specific figures (14 × 4).
└── src/
    ├── config.py               Paths, country list, econometric settings.
    ├── data_builder.py         DBnomics fetch and panel construction.
    ├── utils.py                Dataset loading and sample preparation.
    ├── analysis.py             STR/BUIP estimation pipeline.
    ├── econometrics.py         STR and BUIP model classes.
    ├── output_tables.py        LaTeX/CSV table generation.
    ├── output_figures.py       Publication-quality figure generation.
    └── notebook_setup.py       Consolidated imports for the notebook.
```

## Quick replication

```bash
# 1. Clone and enter the repository
git clone https://github.com/lorenzobiondiecon/str-buip-estimation.git
cd str-buip-estimation

# 2. (Optional but recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r Requirements.txt

# 4. Open the notebook and run all cells
jupyter notebook main_reproduction.ipynb
```

Total runtime: 15 to 30 minutes for the 14 countries.

The notebook will:
- Load `data/df_panel_max.csv` if it exists, or fetch the raw series from
  DBnomics and rebuild it.
- Estimate STR and BUIP models for each country.
- Write all tables and figures to `results/max/`.

## Reproducing specific outputs in the paper

The notebook has two user-facing toggles in the data-loading cell. They are
the only switches you need.

```python
USE_MAX_DATASET    = True     # See "Dataset modes" below.
EXCLUDE_POST_COVID = False    # See "Pre-COVID robustness" below.
```

### Main results (Tables 1 to 6, all 67 figures)

```python
USE_MAX_DATASET    = True
EXCLUDE_POST_COVID = False
```

Run all cells. Output lands in `results/max/`. This reproduces:

- **Table 1**: STR parameter estimates.
- **Table 2**: STR residual diagnostics.
- **Table 3**: BUIP parameter estimates.
- **Table 4**: BUIP residual diagnostics.
- **Table 5**: Linear / STR / BUIP comparison (RMSE, AIC, BIC).
- **Table 6**: Threshold and identification validation.
- All 10 cross-country figures (`results/max/figures/aggregate/`).
- All 56 country figures (`results/max/figures/country/`).

These outputs are already committed under `results/max/`, so you can also
inspect the paper's figures and tables directly without rerunning the
pipeline.

### Pre-COVID robustness (Section 5.3, Table 7)

```python
USE_MAX_DATASET    = True
EXCLUDE_POST_COVID = True
```

Run all cells. Output lands in `results/no-covid/max/`. The sample is
truncated at `2019-12-31` (configurable in `src/config.py` via
`COVID_CUTOFF`). This reproduces Table 7, which reports STR estimates for
the United Kingdom and BUIP estimates for Japan, Korea, and Türkiye on the
pre-pandemic sample.

## Dataset modes

`USE_MAX_DATASET = True` (default, used in the paper) loads
`data/df_panel_max.csv`: each country uses the longest continuous span in
which the nominal exchange rate, CPI, and short rate are jointly available,
with country-specific trims for known data gaps in Indonesia, Korea, and
the Philippines. Sample bounds run from the early 1980s to 2025.

`USE_MAX_DATASET = False` builds (or loads) `df_panel_final.csv`, which
restricts the sample to the common 2000-01 through 2024-12 window. This
mode is provided for cross-country comparisons over a uniform window and
is not the specification reported in the paper.

## How the dataset is built

`src/data_builder.py` fetches monthly series from DBnomics:

| Variable | Source | DBnomics code |
| --- | --- | --- |
| Nominal exchange rate (LCU per USD) | IMF/IFS | `M.<ISO2>.ENDE_XDC_USD_RATE` |
| Consumer price index | IMF/IFS | `M.<ISO2>.PCPI_IX` |
| Policy rate (AUS, IDN, TUR) | IMF/IFS | `M.<ISO2>.FPOLM_PA` |
| Money market rate (BRA, KOR, MEX, NZL, PHL, THA) | IMF/IFS | `M.<ISO2>.FIMM_PA` |
| Short rate (CAN, CHE, EA, GBR, JPN) | OECD | `OECD/DSD_KEI@DF_KEI/<ISO3>.M.IRSTCI.PA._Z._Z._Z` |
| US CPI and federal funds rate (benchmark) | IMF/IFS | `M.US.PCPI_IX`, `M.US.FPOLM_PA` |
| Euro Area HICP | IMF/IFS | `M.U2.PCPIHA_IX` |
| Australia and New Zealand CPI | IMF/IFS quarterly, spline-interpolated to monthly via DBnomics filter | `Q.<ISO2>.PCPI_IX` |

Country-specific patches handled in `data_builder.py`:

- **Indonesia**: policy-rate gaps filled with the money-market rate.
- **Philippines**: missing 2022-01 policy rate linearly interpolated.

After fetching, `engineer_features()` constructs the variables used in
estimation:

| Variable | Formula | Description |
| --- | --- | --- |
| `s` | `log(S)` | Log nominal exchange rate. |
| `p_dom`, `p_for` | `log(CPI)` | Log price levels. |
| `r_s` | `Δs` | Nominal exchange rate return. |
| `q` | `s + p_for − p_dom` | Real exchange rate (logs). |
| `i_dom`, `i_for` | `(1 + r/100)^(1/12) − 1` | Annual rate converted to monthly effective. |
| `f_ppp` | `p_dom − p_for` | PPP fundamental level. |
| `f_ppp_rel` | `Δp_dom − Δp_for` | Inflation differential. |
| `r_q` | `r_s − f_ppp_rel` | Excess return over PPP. |

The dependent variable in both STR and BUIP estimation is the excess
return on the long-domestic carry trade,
`y = r_s − (i_for − i_dom)`.

## Estimation details

For each country, `src/analysis.py`:

1. Builds 35 candidate transition variables (7 base series × 5 lags).
2. Selects the candidate with the lowest p-value in a HAC-robust LM
   linearity test.
3. Runs a 120 × 120 grid search over `(γ, c)` for STR and a multistart NLS
   for BUIP.
4. Refines via nonlinear least squares with Newey-West HAC standard
   errors (automatic 1994 bandwidth).
5. Computes residual diagnostics: Ljung-Box Q, ARCH LM, Jarque-Bera.

## Output files generated by the notebook

Tables (CSV and LaTeX in `results/max/tables/`):

| File | Paper |
| --- | --- |
| `data_availability_sources` | Data appendix table. |
| `descriptive_statistics` | Summary statistics. |
| `transition_variable_selection` | STR transition-variable choice per country. |
| `str_results_table`, `str_results_with_ci` | Table 1. |
| `str_diagnostics_table` | Table 2. |
| `str_threshold_validation` | Table 6 (STR rows). |
| `buip_results_table`, `buip_results_with_ci` | Table 3. |
| `buip_diagnostics_table` | Table 4. |
| `buip_threshold_validation` | Table 6 (BUIP rows). |
| `comprehensive_model_comparison` | Table 5. |

Aggregate figures (`results/max/figures/aggregate/`):

| File | Description |
| --- | --- |
| `raw_series_exchange_rate` | Log nominal exchange rates. |
| `raw_series_cpi_dom` | Domestic CPI levels. |
| `raw_series_interest_dom` | Domestic short rates. |
| `log_nominal_vs_ppp_6panel` | Six-panel comparison of nominal rate and PPP fundamental. |
| `returns_logS_vs_logPPP_6panel` | Six-panel comparison of returns and inflation differential. |
| `str_regime_distribution`, `str_regime_distribution_groups` | STR regime histogram, pooled and by group. |
| `buip_regime_distribution`, `buip_regime_distribution_groups` | BUIP regime histogram, pooled and by group. |
| `regime_prevalence_comparison` | STR vs BUIP regime prevalence by country. |

Country figures (`results/max/figures/country/`): for each of the 14
countries, four PDFs are produced.

- `<Country>_str_detailed`: STR fit, transition function, and regime path.
- `<Country>_str_scatter`: STR fitted vs actual scatter.
- `<Country>_buip_detailed`: BUIP fit, market shares, and regime path.
- `<Country>_buip_scatter`: BUIP fitted vs actual scatter.

## Countries

Advanced: Australia, Canada, Euro Area, Japan, New Zealand, Switzerland,
United Kingdom.

Emerging: Brazil, Indonesia, Korea, Mexico, Philippines, Thailand,
Türkiye.

Country definitions and ISO codes are in `src/config.py`.

## Software requirements

- Python 3.9 or newer.
- Packages listed in `Requirements.txt`: `numpy`, `pandas`, `scipy`,
  `statsmodels`, `matplotlib`, `seaborn`, `tqdm`, `dbnomics`.
- An internet connection is needed only the first time, when the dataset
  is rebuilt from DBnomics. Afterwards `data/df_panel_max.csv` is reused.

## Troubleshooting

**The notebook tries to fetch from DBnomics and fails.**
Confirm the internet connection. If DBnomics is rate-limiting or
unreachable, simply rerun later. As long as `data/df_panel_max.csv` is
present (which it is in this repository), the notebook will skip the
fetch.

**A specific country fails to converge.**
Country-level errors are logged but do not abort the run. Check the cell
output for the country name. The pre-COVID toggle often resolves
identification problems for Japan, Korea, Türkiye, and the United Kingdom
(this is documented in Section 5.3 of the paper).

**Imports fail after editing files in `src/`.**
The notebook uses `%load_ext autoreload` and `%autoreload 2`, so most
changes are picked up automatically. If they are not, restart the Jupyter
kernel.

**You want to rebuild the dataset from scratch.**
Delete `data/df_panel_max.csv` and rerun the notebook. The next run will
hit DBnomics and rebuild the file.

## References

**Methodology:**

- Granger, C. W. J., and Teräsvirta, T. (1993). *Modelling Non-Linear
  Economic Relationships*. Oxford University Press.
- Teräsvirta, T. (1994). Specification, estimation, and evaluation of
  smooth transition autoregressive models. *Journal of the American
  Statistical Association*, 89(425), 208 to 218.
- Teräsvirta, T., Tjøstheim, D., and Granger, C. W. J. (2010). *Modelling
  Nonlinear Economic Time Series*. Oxford University Press.

**Behavioral UIP:**

- Proaño, C. R. (2013). Monetary policy rules and macroeconomic
  stabilization in small open economies under behavioral FX trading:
  insights from numerical simulations. *The Manchester School*, 81(6),
  992 to 1011.

## License

See `LICENSE`.
