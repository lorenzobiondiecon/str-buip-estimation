"""
Reproduce scatter plots and 6-panel figure with updated font sizes.
Requires pre-run model estimation (str_results, buip_results).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.special import expit

from src.notebook_setup import (
    np, pd, Path,
    config, COUNTRIES,
    load_dataset, setup_notebook, print_dataset_summary,
    build_z_candidates, estimate_all_countries
)
from src.utils import build_beh_sample

setup_notebook(title="REPRODUCE SCATTER AND 6-PANEL FIGURES")

USE_MAX_DATASET = True
df_panel, output_dir, tables_dir, agg_fig_dir, country_fig_dir = load_dataset(use_max=USE_MAX_DATASET)

import time
start = time.time()
print("Estimating models...")
str_results, buip_results = estimate_all_countries(df_panel, COUNTRIES, model_type='both')
print(f"Done in {(time.time()-start)/60:.1f} min")

# ── 6-panel: log nominal vs PPP ───────────────────────────────────────────────
advanced_countries = ["United Kingdom", "Japan", "Switzerland"]
emerging_countries = ["Brazil", "Mexico", "Türkiye"]
plot_countries = advanced_countries + emerging_countries

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()

for idx, country in enumerate(plot_countries):
    ax = axes[idx]
    sub = df_panel[df_panel['country'] == country].sort_values('date').copy()
    sub = sub[['date', 'S', 'CPI_dom', 'CPI_for']].dropna()
    if sub.empty:
        ax.set_visible(False)
        continue
    log_s = np.log(sub['S'])
    log_ppp = np.log(sub['CPI_dom']) - np.log(sub['CPI_for'])
    log_s = log_s - log_s.iloc[0]
    log_ppp = log_ppp - log_ppp.iloc[0]
    ax.plot(sub['date'], log_s, color='black', linewidth=1.0, label='log S (normalized)')
    ax.plot(sub['date'], log_ppp, color='dimgray', linewidth=1.0, linestyle='--', label='log PPP (normalized)')
    ax.set_title(country, fontsize=13, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=12)
fig.suptitle('Log Nominal Exchange Rate vs Log PPP (Advanced vs Emerging)', fontsize=15)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])

fig.savefig(agg_fig_dir / 'log_nominal_vs_ppp_6panel.pdf', bbox_inches='tight')
fig.savefig(agg_fig_dir / 'log_nominal_vs_ppp_6panel.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("ok log_nominal_vs_ppp_6panel")

# ── STR scatter plots ─────────────────────────────────────────────────────────
print("\nSTR scatter plots:")
for country_name in str_results.keys():
    try:
        result_dict = str_results[country_name]
        str_result = result_dict['result']
        z_name = result_dict['z_name']
        country_df = df_panel[df_panel['country'] == country_name].copy().set_index('date').sort_index()
        z_series = build_z_candidates(country_df)[z_name]
        G_series = str_result.G
        mask = z_series.notna() & G_series.notna()
        z_scatter = z_series[mask]
        G_scatter = G_series[mask]
        if len(z_scatter) == 0:
            continue
        gamma_hat = float(str_result.params['gamma'])
        c = float(str_result.params['c'])
        sd_z = float(z_scatter.std(ddof=1)) or 1.0
        x_grid = np.linspace(float(z_scatter.min()), float(z_scatter.max()), 400)
        G_line = expit(gamma_hat * ((x_grid - c) / sd_z))
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.scatter(z_scatter, G_scatter, s=2, color='k')
        ax.plot(x_grid, G_line, lw=0.5, color='k', alpha=0.3)
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{country_name}", fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.margins(x=0, y=0)
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
        plt.tight_layout()
        fig.savefig(country_fig_dir / f"{country_name.replace(' ', '_')}_str_scatter.pdf", bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"  ok {country_name}")
    except Exception as e:
        print(f"  FAILED {country_name}: {e}")

# ── BUIP scatter plots ────────────────────────────────────────────────────────
print("\nBUIP scatter plots:")
for country_name in buip_results.keys():
    try:
        buip_result = buip_results[country_name]
        country_df = df_panel[df_panel['country'] == country_name].copy().set_index('date').sort_index()
        buip_df = build_beh_sample(country_df)
        u_diff = buip_result.U_f - buip_result.U_c
        omega_series = buip_result.omega
        w_aln = omega_series.reindex(u_diff.index)
        mask = u_diff.notna() & w_aln.notna()
        u_scatter = u_diff[mask]
        w_scatter = w_aln[mask]
        if len(u_scatter) == 0:
            continue
        gamma_hat = float(buip_result.params['gamma'])
        c = float(buip_result.params['c'])
        scale_d = float(buip_result.scale_d)
        x_grid = np.linspace(float(u_scatter.min()), float(u_scatter.max()), 400)
        w_line = expit((gamma_hat / scale_d) * (x_grid - c))
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.scatter(u_scatter, w_scatter, s=2, color='k')
        ax.plot(x_grid, w_line, lw=0.5, color='k', alpha=0.3)
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{country_name}", fontsize=11)
        ax.tick_params(axis='both', labelsize=10)
        ax.margins(x=0, y=0)
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
        plt.tight_layout()
        fig.savefig(country_fig_dir / f"{country_name.replace(' ', '_')}_buip_scatter.pdf", bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"  ok {country_name}")
    except Exception as e:
        print(f"  FAILED {country_name}: {e}")

print("\nDone.")
