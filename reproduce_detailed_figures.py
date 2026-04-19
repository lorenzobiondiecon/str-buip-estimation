"""
Standalone script to reproduce COUNTRY_str_detailed.pdf and COUNTRY_buip_detailed.pdf
with increased tick label font sizes.
"""
from src.notebook_setup import (
    np, pd, plt, Path,
    config, COUNTRIES, DATA_SOURCES,
    load_dataset, setup_notebook, print_dataset_summary,
    build_z_candidates, estimate_all_countries
)
from src.utils import build_beh_sample
from src.output_figures import plot_country_detailed_analysis

setup_notebook(title="REPRODUCE DETAILED FIGURES")

USE_MAX_DATASET = True
df_panel, output_dir, tables_dir, agg_fig_dir, country_fig_dir = load_dataset(use_max=USE_MAX_DATASET)
print_dataset_summary(df_panel)

import time
start = time.time()
print("Starting model estimation (15-30 min)...")
str_results, buip_results = estimate_all_countries(df_panel, COUNTRIES, model_type='both')
print(f"Estimation done in {(time.time()-start)/60:.1f} min")

print("\nGenerating STR detailed plots...")
for country_name in str_results.keys():
    try:
        result_dict = str_results[country_name]
        str_result = result_dict['result']
        z_name = result_dict['z_name']
        country_df = df_panel[df_panel['country'] == country_name].copy().set_index('date').sort_index()
        z_candidates = build_z_candidates(country_df)
        z_series = z_candidates[z_name]
        y = country_df['r_s'] - (country_df['i_for'] - country_df['i_dom'])
        save_path = country_fig_dir / f"{country_name.replace(' ', '_')}_str_detailed.pdf"
        plot_country_detailed_analysis(
            y=y, transition_var=z_series, transition_func=str_result.G,
            c_threshold=str_result.params['c'], country=country_name,
            var_name=z_name, model_type='STR', save_path=save_path
        )
        print(f"  ok {country_name}")
    except Exception as e:
        print(f"  FAILED {country_name}: {e}")

print("\nGenerating BUIP detailed plots...")
for country_name in buip_results.keys():
    try:
        buip_result = buip_results[country_name]
        country_df = df_panel[df_panel['country'] == country_name].copy().set_index('date').sort_index()
        buip_df = build_beh_sample(country_df)
        y = buip_df['y']
        utility_diff = buip_result.U_f - buip_result.U_c
        save_path = country_fig_dir / f"{country_name.replace(' ', '_')}_buip_detailed.pdf"
        plot_country_detailed_analysis(
            y=y, transition_var=utility_diff, transition_func=buip_result.omega,
            c_threshold=buip_result.params['c'], country=country_name,
            var_name='Utility Differential', model_type='BUIP', save_path=save_path
        )
        print(f"  ok {country_name}")
    except Exception as e:
        print(f"  FAILED {country_name}: {e}")

print("\nDone. Figures saved to:", country_fig_dir)
