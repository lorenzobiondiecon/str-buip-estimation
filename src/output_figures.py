"""
Publication-quality figure generation for STR and BUIP models.

Creates academic-style figures with:
- White background
- High contrast
- Times New Roman / Arial fonts
- Proper axis labels and titles
- Legend placement
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.5


def setup_figure(figsize: Tuple[float, float] = (10, 6), dpi: int = 300):
    """
    Sets up publication-quality figure.
    
    Args:
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saving
    
    Returns:
        fig, ax objects
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    return fig, ax


def plot_regime_distribution(omega: pd.Series, country: str, 
                            save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plots distribution of mixing weights (omega) for BUIP model.
    
    Args:
        omega: Mixing weight series
        country: Country name
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = setup_figure(figsize=(8, 5))
    
    # Histogram
    ax.hist(omega.values, bins=30, density=True, alpha=0.7, 
            color='steelblue', edgecolor='black', linewidth=0.8)
    
    # Add vertical lines for mean and median
    mean_omega = omega.mean()
    median_omega = omega.median()
    
    ax.axvline(mean_omega, color='red', linestyle='--', linewidth=2, 
              label=f'Mean = {mean_omega:.3f}')
    ax.axvline(median_omega, color='darkgreen', linestyle='-.', linewidth=2,
              label=f'Median = {median_omega:.3f}')
    
    # Labels and title
    ax.set_xlabel(r'Mixing Weight ($\omega$)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Distribution of Mixing Weights - {country}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # Regime interpretation text
    if mean_omega > 0.7:
        regime_text = "Predominantly Fundamentalist"
    elif mean_omega < 0.3:
        regime_text = "Predominantly Chartist"
    else:
        regime_text = "Mixed Regime"
    
    ax.text(0.02, 0.98, regime_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_regime_timeseries(omega: pd.Series, country: str,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plots time series of mixing weights with regime shading.
    
    Args:
        omega: Mixing weight series with datetime index
        country: Country name
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = setup_figure(figsize=(12, 5))
    
    # Plot omega over time
    ax.plot(omega.index, omega.values, color='steelblue', linewidth=1.5, alpha=0.8)
    
    # Shade regimes
    ax.axhspan(0.8, 1.0, alpha=0.15, color='green', label='Fundamentalist Regime')
    ax.axhspan(0.0, 0.2, alpha=0.15, color='red', label='Chartist Regime')
    ax.axhspan(0.2, 0.8, alpha=0.10, color='yellow', label='Transition')
    
    # Reference lines
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(r'Mixing Weight ($\omega$)', fontsize=11)
    ax.set_title(f'Time-Varying Mixing Weights - {country}', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_aggregated_regime_prevalence(results: Dict[str, any],
                                      save_path: Optional[Path] = None) -> plt.Figure:
    """
    Creates bar chart comparing regime prevalence across countries.
    
    Args:
        results: Dict mapping country names to BUIP results
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    # Extract mean omega for each country
    data = []
    for country, result in results.items():
        if result is None or result.get('status') != 'success':
            continue
        
        res = result['result']
        mean_omega = res.omega.mean()
        data.append({'Country': country, 'Mean_Omega': mean_omega})
    
    df = pd.DataFrame(data)
    df = df.sort_values('Mean_Omega', ascending=True)
    
    fig, ax = setup_figure(figsize=(10, 6))
    
    # Color bars by regime
    colors = []
    for val in df['Mean_Omega']:
        if val > 0.7:
            colors.append('forestgreen')
        elif val < 0.3:
            colors.append('crimson')
        else:
            colors.append('orange')
    
    bars = ax.barh(df['Country'], df['Mean_Omega'], color=colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Add regime boundaries
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Neutral')
    ax.axvline(0.7, color='green', linestyle=':', linewidth=1.2, alpha=0.6, label='Fund. Threshold')
    ax.axvline(0.3, color='red', linestyle=':', linewidth=1.2, alpha=0.6, label='Chart. Threshold')
    
    # Labels
    ax.set_xlabel(r'Mean Mixing Weight ($\bar{\omega}$)', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)
    ax.set_title('Regime Prevalence Across Countries', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Add value labels on bars
    for i, (country, val) in enumerate(zip(df['Country'], df['Mean_Omega'])):
        ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_str_transition_function(G: pd.Series, z: pd.Series, 
                                 country: str, z_name: str,
                                 save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plots STR transition function G against transition variable z.
    
    Args:
        G: Transition function values
        z: Transition variable values
        country: Country name
        z_name: Name of transition variable
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = setup_figure(figsize=(8, 5))
    
    # Scatter plot
    ax.scatter(z.values, G.values, alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.3)
    
    # Fitted logistic curve
    z_sorted = np.sort(z.values)
    G_sorted = G[z.argsort()].values
    ax.plot(z_sorted, G_sorted, color='red', linewidth=2, label='Logistic Transition')
    
    # Regime boundaries
    ax.axhline(0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chartist Regime')
    ax.axhline(0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Fundamentalist Regime')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_xlabel(f'Transition Variable ({z_name})', fontsize=11)
    ax.set_ylabel(r'Transition Function $G(z)$', fontsize=11)
    ax.set_title(f'STR Transition Function - {country}', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_gamma_comparison(results: Dict[str, any], model_type: str = 'BUIP',
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compares smoothness parameter (gamma) across countries.
    
    Args:
        results: Dict mapping country names to model results
        model_type: 'BUIP' or 'STR'
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    data = []
    for country, result in results.items():
        if result is None or result.get('status') != 'success':
            continue
        
        res = result['result']
        gamma = res.params['gamma']
        se = res.se['gamma']
        pval = res.pvals['gamma']
        
        data.append({
            'Country': country,
            'Gamma': gamma,
            'SE': se,
            'Significant': pval < 0.05
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Gamma', ascending=True)
    
    fig, ax = setup_figure(figsize=(10, 6))
    
    # Color by significance
    colors = ['forestgreen' if sig else 'lightgray' for sig in df['Significant']]
    
    # Bar plot with error bars
    bars = ax.barh(df['Country'], df['Gamma'], xerr=1.96*df['SE'],
                   color=colors, edgecolor='black', linewidth=0.8, 
                   alpha=0.8, capsize=3)
    
    # Labels
    ax.set_xlabel(r'Smoothness Parameter ($\gamma$)', fontsize=11)
    ax.set_ylabel('Country', fontsize=11)
    ax.set_title(f'{model_type} Model: Smoothness Parameter Comparison', 
                fontsize=12, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='forestgreen', edgecolor='black', label='Significant (p<0.05)'),
        Patch(facecolor='lightgray', edgecolor='black', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_residual_diagnostics(resid: pd.Series, country: str,
                              save_path: Optional[Path] = None) -> plt.Figure:
    """
    Creates diagnostic plots for residuals (QQ-plot and ACF).
    
    Args:
        resid: Residual series
        country: Country name
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor('white')
    
    # QQ-plot
    ax1 = axes[0]
    from scipy import stats as sp_stats
    sp_stats.probplot(resid.values, dist="norm", plot=ax1)
    ax1.set_title(f'Q-Q Plot - {country}', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ACF plot
    ax2 = axes[1]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid.values, lags=20, ax=ax2, alpha=0.05)
    ax2.set_title(f'Autocorrelation Function - {country}', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Lag', fontsize=10)
    ax2.set_ylabel('ACF', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_model_comparison(str_results: Dict[str, any], 
                         buip_results: Dict[str, any],
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compares RMSE between STR and BUIP models across countries.
    
    Args:
        str_results: Dict of STR results
        buip_results: Dict of BUIP results
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    data = []
    
    # Collect countries present in both
    common_countries = set(str_results.keys()) & set(buip_results.keys())
    
    for country in common_countries:
        str_res = str_results.get(country)
        buip_res = buip_results.get(country)
        
        if (str_res is None or str_res.get('status') != 'success' or
            buip_res is None or buip_res.get('status') != 'success'):
            continue
        
        data.append({
            'Country': country,
            'STR_RMSE': str_res['result'].rmse,
            'BUIP_RMSE': buip_res['result'].rmse
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('BUIP_RMSE', ascending=True)
    
    fig, ax = setup_figure(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['STR_RMSE'], width, label='STR',
                   color='steelblue', edgecolor='black', linewidth=0.8, alpha=0.8)
    bars2 = ax.bar(x + width/2, df['BUIP_RMSE'], width, label='BUIP',
                   color='coral', edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax.set_xlabel('Country', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Model Comparison: STR vs BUIP', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Country'], rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_country_detailed_analysis(y: pd.Series, 
                                   transition_var: pd.Series,
                                   transition_func: pd.Series,
                                   c_threshold: float,
                                   country: str,
                                   var_name: str,
                                   model_type: str = 'STR',
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Creates a detailed 3-panel stacked vertical plot for a country showing:
    - Top: Dependent variable over time
    - Middle: Transition variable with threshold c (dashed line)
    - Bottom: Transition function over time
    
    Args:
        y: Dependent variable series (indexed by date)
        transition_var: Transition variable series (z for STR, utility diff for BUIP)
        transition_func: Transition function values (G for STR, omega for BUIP)
        c_threshold: Threshold parameter c
        country: Country name
        var_name: Name of transition variable for labeling
        model_type: 'STR' or 'BUIP'
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    # Create figure with 3 subplots, equal height ratios
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), 
                            gridspec_kw={'height_ratios': [1, 1, 1]}, dpi=300)
    fig.patch.set_facecolor('white')
    
    # Ensure all series are aligned and have same index
    common_idx = y.index.intersection(transition_var.index).intersection(transition_func.index)
    y = y.loc[common_idx]
    transition_var = transition_var.loc[common_idx]
    transition_func = transition_func.loc[common_idx]
    
    # ==========================================
    # TOP PANEL: Dependent Variable
    # ==========================================
    ax_top = axes[0]
    ax_top.plot(y.index, y.values, color='black', linewidth=0.7)
    ax_top.spines['top'].set_visible(True)
    ax_top.spines['right'].set_visible(True)
    ax_top.spines['bottom'].set_visible(True)
    ax_top.spines['left'].set_visible(True)
    ax_top.set_ylabel('')
    ax_top.set_xlabel('')
    ax_top.grid(False)
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # ==========================================
    # MIDDLE PANEL: Transition Variable + Threshold
    # ==========================================
    ax_mid = axes[1]
    ax_mid.plot(transition_var.index, transition_var.values, 
                color='black', linewidth=0.7)
    ax_mid.axhline(c_threshold, color='black', linestyle='--', linewidth=0.7)
    ax_mid.spines['top'].set_visible(True)
    ax_mid.spines['right'].set_visible(True)
    ax_mid.spines['bottom'].set_visible(True)
    ax_mid.spines['left'].set_visible(True)
    ax_mid.set_ylabel('')
    ax_mid.set_xlabel('')
    ax_mid.grid(False)
    plt.setp(ax_mid.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # ==========================================
    # BOTTOM PANEL: Transition Function
    # ==========================================
    ax_bot = axes[2]
    ax_bot.plot(transition_func.index, transition_func.values, 
                color='black', linewidth=0.7)
    ax_bot.set_ylim(0.0, 1.0)
    ax_bot.spines['top'].set_visible(True)
    ax_bot.spines['right'].set_visible(True)
    ax_bot.spines['bottom'].set_visible(True)
    ax_bot.spines['left'].set_visible(True)
    ax_bot.set_ylabel('')
    ax_bot.set_xlabel('')
    ax_bot.grid(False)
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig
