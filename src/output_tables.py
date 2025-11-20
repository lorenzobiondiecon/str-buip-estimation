"""
Publication-ready table generation for STR and BUIP models.

Generates LaTeX-formatted tables with:
- Parameter estimates with standard errors
- Significance stars (*, **, ***)
- Model diagnostics (N, RMSE, AIC, BIC)
- Regime statistics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def format_param_with_stars(value: float, pval: float, decimals: int = 4) -> str:
    """
    Formats parameter value with significance stars.
    
    Stars: *** p<0.01, ** p<0.05, * p<0.10
    
    Args:
        value: Parameter estimate
        pval: P-value
        decimals: Decimal places
    
    Returns:
        Formatted string with stars
    """
    stars = ''
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.10:
        stars = '*'
    
    fmt_str = f"{{:.{decimals}f}}"
    return fmt_str.format(value) + stars


def format_se(value: float, decimals: int = 4) -> str:
    """Formats standard error in parentheses."""
    fmt_str = f"({{:.{decimals}f}})"
    return fmt_str.format(value)


def create_str_table(results: Dict[str, any], save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Creates publication-ready table for STR model results across countries.
    
    Args:
        results: Dict mapping country names to STR result objects
        save_path: Optional path to save LaTeX table
    
    Returns:
        DataFrame with formatted results
    """
    rows = []
    
    for country, result in results.items():
        if result is None or result.get('status') != 'success':
            continue
        
        res = result['result']
        
        # Parameter estimates with stars
        row = {
            'Country': country,
            'const': format_param_with_stars(res.params['const'], res.pvals['const']),
            'const_se': format_se(res.se['const']),
            'beta_c': format_param_with_stars(res.params['beta_c'], res.pvals['beta_c']),
            'beta_c_se': format_se(res.se['beta_c']),
            'beta_f': format_param_with_stars(res.params['beta_f'], res.pvals['beta_f']),
            'beta_f_se': format_se(res.se['beta_f']),
            'gamma': format_param_with_stars(res.params['gamma'], res.pvals['gamma']),
            'gamma_se': format_se(res.se['gamma']),
            'c': format_param_with_stars(res.params['c'], res.pvals['c']),
            'c_se': format_se(res.se['c']),
            'const1': format_param_with_stars(res.params['const1'], res.pvals['const1']),
            'const1_se': format_se(res.se['const1']),
            'N': int(res.nobs),
            'RMSE': f"{res.rmse:.6f}",
            'AIC': f"{res.aic:.2f}",
            'BIC': f"{res.bic:.2f}"
        }
        
        # Add regime statistics if available
        if 'regime_stats' in result:
            rs = result['regime_stats']
            row['p_low'] = f"{rs.get('p_lo', np.nan):.3f}"
            row['p_high'] = f"{rs.get('p_hi', np.nan):.3f}"
            row['spell_C'] = f"{rs.get('avg_spell_C', np.nan):.2f}"
            row['spell_F'] = f"{rs.get('avg_spell_F', np.nan):.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        # Generate LaTeX table
        latex = generate_latex_table(df, model_type='STR')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex)
    
    return df


def create_buip_table(results: Dict[str, any], save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Creates publication-ready table for BUIP model results across countries.
    
    Args:
        results: Dict mapping country names to BUIP result objects
        save_path: Optional path to save LaTeX table
    
    Returns:
        DataFrame with formatted results
    """
    rows = []
    
    for country, result in results.items():
        if result is None or result.get('status') != 'success':
            continue
        
        res = result['result']
        
        # Parameter estimates with stars
        row = {
            'Country': country,
            'beta_f': format_param_with_stars(res.params['beta_f'], res.pvals['beta_f']),
            'beta_f_se': format_se(res.se['beta_f']),
            'beta_c': format_param_with_stars(res.params['beta_c'], res.pvals['beta_c']),
            'beta_c_se': format_se(res.se['beta_c']),
            'gamma': format_param_with_stars(res.params['gamma'], res.pvals['gamma']),
            'gamma_se': format_se(res.se['gamma']),
            'c': format_param_with_stars(res.params['c'], res.pvals['c']),
            'c_se': format_se(res.se['c']),
            'const': format_param_with_stars(res.params['const'], res.pvals['const']),
            'const_se': format_se(res.se['const']),
            'const1': format_param_with_stars(res.params['const1'], res.pvals['const1']),
            'const1_se': format_se(res.se['const1']),
            'N': int(res.nobs),
            'RMSE': f"{res.rmse:.6f}",
            'AIC': f"{res.aic:.2f}",
            'BIC': f"{res.bic:.2f}"
        }
        
        # Add regime statistics
        omega = res.omega.values
        row['mean_omega'] = f"{np.mean(omega):.3f}"
        row['median_omega'] = f"{np.median(omega):.3f}"
        row['std_omega'] = f"{np.std(omega):.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        # Generate LaTeX table
        latex = generate_latex_table(df, model_type='BUIP')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex)
    
    return df


def generate_latex_table(df: pd.DataFrame, model_type: str = 'STR') -> str:
    """
    Generates LaTeX code for publication-ready table.
    
    Args:
        df: DataFrame with formatted results
        model_type: 'STR' or 'BUIP'
    
    Returns:
        LaTeX table code
    """
    latex = []
    
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{model_type} Model Estimation Results}}")
    latex.append(f"\\label{{tab:{model_type.lower()}_results}}")
    
    if model_type == 'STR':
        latex.append("\\begin{tabular}{lcccccccccc}")
        latex.append("\\hline\\hline")
        latex.append("Country & $\\alpha$ & $\\beta_c$ & $\\beta_f$ & $\\gamma$ & $c$ & $\\alpha_1$ & N & RMSE & AIC & BIC \\\\")
        latex.append("\\hline")
        
        for _, row in df.iterrows():
            line = f"{row['Country']} & "
            line += f"{row['const']} & {row['beta_c']} & {row['beta_f']} & "
            line += f"{row['gamma']} & {row['c']} & {row['const1']} & "
            line += f"{row['N']} & {row['RMSE']} & {row['AIC']} & {row['BIC']} \\\\"
            latex.append(line)
            
            # Add SE row
            se_line = f" & {row['const_se']} & {row['beta_c_se']} & {row['beta_f_se']} & "
            se_line += f"{row['gamma_se']} & {row['c_se']} & {row['const1_se']} & & & & \\\\"
            latex.append(se_line)
            latex.append("\\\\[-1.5ex]")  # Reduce spacing
    
    elif model_type == 'BUIP':
        latex.append("\\begin{tabular}{lcccccccccc}")
        latex.append("\\hline\\hline")
        latex.append("Country & $\\beta_f$ & $\\beta_c$ & $\\gamma$ & $c$ & const & const1 & N & RMSE & $\\bar{\\omega}$ & $\\tilde{\\omega}$ \\\\")
        latex.append("\\hline")
        
        for _, row in df.iterrows():
            line = f"{row['Country']} & "
            line += f"{row['beta_f']} & {row['beta_c']} & {row['gamma']} & "
            line += f"{row['c']} & {row['const']} & {row['const1']} & "
            line += f"{row['N']} & {row['RMSE']} & {row['mean_omega']} & {row['median_omega']} \\\\"
            latex.append(line)
            
            # Add SE row
            se_line = f" & {row['beta_f_se']} & {row['beta_c_se']} & {row['gamma_se']} & "
            se_line += f"{row['c_se']} & {row['const_se']} & {row['const1_se']} & & & & \\\\"
            latex.append(se_line)
            latex.append("\\\\[-1.5ex]")
    
    latex.append("\\hline\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\\\[0.5em]")
    latex.append("\\begin{minipage}{\\textwidth}")
    latex.append("\\small")
    latex.append("\\textit{Notes:} Standard errors in parentheses. ")
    latex.append("Significance levels: $^{***}$ $p<0.01$, $^{**}$ $p<0.05$, $^{*}$ $p<0.10$. ")
    latex.append("HAC standard errors (Newey-West) with 4 lags. ")
    if model_type == 'BUIP':
        latex.append("$\\bar{\\omega}$ is mean mixing weight, $\\tilde{\\omega}$ is median.")
    latex.append("\\end{minipage}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def create_descriptive_stats_table(data: pd.DataFrame, save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Creates descriptive statistics table for all countries.
    
    Args:
        data: Panel DataFrame with all countries
        save_path: Optional path to save LaTeX table
    
    Returns:
        DataFrame with descriptive statistics
    """
    countries = data['country'].unique()
    
    rows = []
    for country in countries:
        df_country = data[data['country'] == country]
        
        row = {
            'Country': country,
            'N': len(df_country),
            'r_s_mean': df_country['r_s'].mean(),
            'r_s_std': df_country['r_s'].std(),
            'q_mean': df_country['q'].mean(),
            'q_std': df_country['q'].std(),
            'i_dom_mean': df_country['i_dom'].mean(),
            'i_for_mean': df_country['i_for'].mean(),
            'ID_mean': (df_country['i_for'] - df_country['i_dom']).mean(),
        }
        rows.append(row)
    
    df_stats = pd.DataFrame(rows)
    
    if save_path:
        # Generate LaTeX
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Descriptive Statistics}")
        latex.append("\\label{tab:descriptive_stats}")
        latex.append("\\begin{tabular}{lcccccccc}")
        latex.append("\\hline\\hline")
        latex.append("Country & N & $\\bar{r}_s$ & $\\sigma_{r_s}$ & $\\bar{\\eta}$ & $\\sigma_{\\eta}$ & $\\bar{i}$ & $\\bar{i}^*$ & $\\bar{i}^*-\\bar{i}$ \\\\")
        latex.append("\\hline")
        
        for _, row in df_stats.iterrows():
            line = f"{row['Country']} & {row['N']} & "
            line += f"{row['r_s_mean']:.4f} & {row['r_s_std']:.4f} & "
            line += f"{row['q_mean']:.4f} & {row['q_std']:.4f} & "
            line += f"{row['i_dom_mean']:.4f} & {row['i_for_mean']:.4f} & "
            line += f"{row['ID_mean']:.4f} \\\\"
            latex.append(line)
        
        latex.append("\\hline\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\\\[0.5em]")
        latex.append("\\begin{minipage}{\\textwidth}")
        latex.append("\\small")
        latex.append("\\textit{Notes:} $r_s$ is exchange rate return, $\\eta$ is real exchange rate (PPP deviation), ")
        latex.append("$i$ is domestic interest rate, $i^*$ is foreign interest rate.")
        latex.append("\\end{minipage}")
        latex.append("\\end{table}")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("\n".join(latex))
    
    return df_stats


def create_lm_test_summary_table(lm_results: Dict[str, pd.DataFrame], 
                                  save_path: Optional[Path] = None,
                                  top_n: int = 5) -> pd.DataFrame:
    """
    Creates summary table of LM linearity test results across countries.
    
    Args:
        lm_results: Dict mapping country names to LM test DataFrames
        save_path: Optional path to save LaTeX table
        top_n: Number of top candidates to show per country
    
    Returns:
        DataFrame with best z-candidates per country
    """
    rows = []
    
    for country, lm_df in lm_results.items():
        if lm_df is None or len(lm_df) == 0:
            continue
        
        # Get best candidate
        best = lm_df.iloc[0]
        
        # Count significant at different levels
        sig_1pct = (lm_df['p_value_HAC'] < 0.01).sum()
        sig_5pct = (lm_df['p_value_HAC'] < 0.05).sum()
        sig_10pct = (lm_df['p_value_HAC'] < 0.10).sum()
        
        row = {
            'Country': country,
            'Best_z': best['z_var'],
            'LM_HAC': f"{best['LM_HAC']:.2f}",
            'p_value': f"{best['p_value_HAC']:.4f}",
            'Sig_1pct': sig_1pct,
            'Sig_5pct': sig_5pct,
            'Sig_10pct': sig_10pct,
            'Total': len(lm_df)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{LM Linearity Test Results Summary}")
        latex.append("\\label{tab:lm_tests}")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\hline\\hline")
        latex.append("Country & Best $z$ & LM$_{HAC}$ & $p$-value & Sig@1\\% & Sig@5\\% & Sig@10\\% \\\\")
        latex.append("\\hline")
        
        for _, row in df.iterrows():
            z_short = row['Best_z'].replace('_', '\\_')
            line = f"{row['Country']} & {z_short} & {row['LM_HAC']} & "
            line += f"{row['p_value']} & {row['Sig_1pct']} & {row['Sig_5pct']} & {row['Sig_10pct']} \\\\"
            latex.append(line)
        
        latex.append("\\hline\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\\\[0.5em]")
        latex.append("\\begin{minipage}{\\textwidth}")
        latex.append("\\small")
        latex.append("\\textit{Notes:} Best $z$ is the transition variable with lowest HAC-robust $p$-value. ")
        latex.append("Sig@$x$\\% shows number of candidates significant at $x$\\% level out of 35 total. ")
        latex.append("LM$_{HAC}$ is HAC-robust Lagrange Multiplier test statistic.")
        latex.append("\\end{minipage}")
        latex.append("\\end{table}")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("\n".join(latex))
    
    return df
