import numpy as np, pandas as pd
from pathlib import Path
from src.utils import load_dataset
from src.analysis import lm_linearity_test_for_z
from src.econometrics import STRModel
from src.output_figures import plot_country_detailed_analysis

def add_stars(pval):
    """Add significance stars based on p-value"""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.1:
        return "*"
    else:
        return ""

use_max = True
min_obs = 50
gamma_grid = np.logspace(np.log10(0.125), np.log10(256.0), num=120)
c_quantiles = np.linspace(0.05, 0.95, 120)

print("Loading dataset...")
df_panel, *_ = load_dataset(use_max=use_max)
countries = sorted(df_panel['country'].unique())
print(f"Countries: {len(countries)}\n")

# Step 1: For each country, test all lags and select best
print("="*60)
print("STEP 1: Linearity tests for all lags - selecting best")
print("="*60)

country_best_lag = {}
country_best_var = {}
lm_summary = []

for ctry in countries:
    g = df_panel[df_panel["country"] == ctry].copy().set_index("date")
    if len(g) < min_obs:
        continue
    
    # Build squared transition variable: squared real exchange rate returns
    z_candidates = {
        'deta_sq': (g["q"].diff()) ** 2
    }
    
    lag_results = []
    for z_name, z_base in z_candidates.items():
        for z_lag in range(1, 6):
            z = z_base.shift(z_lag)
            lm_res = lm_linearity_test_for_z(g, z)
            if lm_res:
                lag_results.append({
                    'var_name': z_name,
                    'lag': z_lag,
                    'LM': lm_res["LM"],
                    'p_value': lm_res["p_value"],
                    'Wald_HAC': lm_res["Wald_HAC"],
                    'p_value_HAC': lm_res["p_value_HAC"],
                    'nobs': lm_res["nobs"]
                })
    
    if lag_results:
        best = min(lag_results, key=lambda x: x['p_value_HAC'])
        country_best_lag[ctry] = best['lag']
        country_best_var[ctry] = best['var_name']
        lm_summary.append({
            'country': ctry,
            'selected_var': best['var_name'],
            'selected_lag': best['lag'],
            'LM': best['LM'],
            'p_value': best['p_value'],
            'Wald_HAC': best['Wald_HAC'],
            'p_value_HAC': best['p_value_HAC'],
            'nobs': best['nobs']
        })

lm_df = pd.DataFrame(lm_summary).sort_values('p_value_HAC')
print("\nLinearity test results (best lag per country, sorted by p_value_HAC):")
print(lm_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

# Step 2: Estimate STR with selected lags
print("\n" + "="*60)
print("STEP 2: STR estimation with selected lags")
print("="*60)

est_results = []
plot_dir = Path("tmp_plots_best_lag")
plot_dir.mkdir(parents=True, exist_ok=True)

for ctry in country_best_lag.keys():
    g = df_panel[df_panel["country"] == ctry].copy().set_index("date")
    
    z_lag = country_best_lag[ctry]
    z_var_name = country_best_var[ctry]
    
    # Reconstruct the selected transition variable: squared real exchange rate returns
    z_candidates = {
        'deta_sq': (g["q"].diff()) ** 2
    }
    
    z = z_candidates[z_var_name].shift(z_lag)
    str_df = pd.DataFrame({
        "y": g["r_s"] - (g["i_for"] - g["i_dom"]),
        "rs_t1": g["r_s"].shift(1),
        "eta_t1": g["q"].diff().shift(1),
        "z": z
    }).dropna()
    
    if len(str_df) < 25:
        continue
    
    print(f"\nEstimating {ctry} (var: {z_var_name}, lag {z_lag})...")
    
    c_grid = np.quantile(pd.to_numeric(str_df["z"], errors="coerce").dropna(), c_quantiles)
    model = STRModel(str_df, z_col="z")
    start = model.grid_search(gamma_grid, c_grid)
    res = model.fit(start, hac_lags=None)
    
    est_results.append({
        'country': ctry,
        'var': z_var_name,
        'lag': z_lag,
        'const': res.params["const"],
        'p_const': res.pvals["const"],
        'beta_c': res.params["beta_c"],
        'p_beta_c': res.pvals["beta_c"],
        'beta_f': res.params["beta_f"],
        'p_beta_f': res.pvals["beta_f"],
        'gamma': res.params["gamma"],
        'p_gamma': res.pvals["gamma"],
        'c': res.params["c"],
        'p_c': res.pvals["c"],
        'const1': res.params["const1"],
        'p_const1': res.pvals["const1"],
        'AIC': res.aic,
        'BIC': res.bic,
        'RMSE': res.rmse,
        'nobs': res.nobs
    })
    
    # Generate 3-panel plot
    try:
        plot_country_detailed_analysis(
            y=str_df["y"],
            transition_var=str_df["z"],
            transition_func=res.G,
            c_threshold=res.params["c"],
            country=ctry,
            var_name=f"{z_var_name} lag{z_lag}",
            model_type='STR',
            save_path=plot_dir / f"{ctry.replace(' ', '_')}.png"
        )
        print(f"  -> Plot saved")
    except Exception as e:
        print(f"  -> Plot failed: {e}")

print("\n" + "="*60)
print("STEP 3: Estimation results with significance stars")
print("="*60)
print("Stars: *** p<0.01, ** p<0.05, * p<0.1\n")

# Create formatted output with stars
for row in sorted(est_results, key=lambda x: x['AIC']):
    print(f"\n{row['country']} (var: {row['var']}, lag {row['lag']}):")
    print(f"  const   = {row['const']:>10.4f}{add_stars(row['p_const']):>3}  (p={row['p_const']:.4f})")
    print(f"  beta_c  = {row['beta_c']:>10.4f}{add_stars(row['p_beta_c']):>3}  (p={row['p_beta_c']:.4f})")
    print(f"  beta_f  = {row['beta_f']:>10.4f}{add_stars(row['p_beta_f']):>3}  (p={row['p_beta_f']:.4f})")
    print(f"  gamma   = {row['gamma']:>10.4f}{add_stars(row['p_gamma']):>3}  (p={row['p_gamma']:.4f})")
    print(f"  c       = {row['c']:>10.4f}{add_stars(row['p_c']):>3}  (p={row['p_c']:.4f})")
    print(f"  const1  = {row['const1']:>10.4f}{add_stars(row['p_const1']):>3}  (p={row['p_const1']:.4f})")
    print(f"  AIC     = {row['AIC']:>10.2f}    BIC = {row['BIC']:>10.2f}    RMSE = {row['RMSE']:.4f}    nobs = {row['nobs']}")

# Also create compact table
print("\n" + "="*60)
print("Compact summary table")
print("="*60)

est_df = pd.DataFrame(est_results).sort_values('AIC')
compact = est_df[['country', 'var', 'lag', 'beta_c', 'beta_f', 'gamma', 'AIC', 'BIC', 'RMSE', 'nobs']].copy()
print(compact.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

print(f"\n\nDone! Plots saved to {plot_dir}/")
