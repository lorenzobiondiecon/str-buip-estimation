"""
Visualization code for df_filtered dataset
Creates 3 figures (one per variable) with 7x2 subplots showing all countries
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_variables(df_filtered, save_plots=True, plot_path=None):
    """
    Create 3 figures (one per variable) with 7x2 subplots for all countries.
    
    Parameters:
    -----------
    df_filtered : pd.DataFrame
        Filtered dataset with columns: country, date, variable, value
    save_plots : bool
        Whether to save plots to disk
    plot_path : str
        Path to save plots (if save_plots=True)
    """
    
    # Get unique variables and countries
    variables = sorted(df_filtered['variable'].unique())
    countries = sorted(df_filtered['country'].unique())
    
    print(f"Creating visualizations for {len(variables)} variables and {len(countries)} countries...")
    
    # Variable display names and units
    var_info = {
        'cpi': {'title': 'Consumer Price Index (CPI)', 'ylabel': 'Index'},
        'exchange_rate': {'title': 'Exchange Rate (Local Currency per USD)', 'ylabel': 'Exchange Rate'},
        'policy_rate': {'title': 'Policy Interest Rate', 'ylabel': 'Rate (%)'}
    }
    
    # Create a figure for each variable
    for variable in variables:
        # Create figure with 7x2 subplots
        fig, axes = plt.subplots(7, 2, figsize=(16, 20))
        fig.suptitle(var_info[variable]['title'], fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        # Plot each country
        for idx, country in enumerate(countries):
            ax = axes_flat[idx]
            
            # Filter data for this country and variable
            data = df_filtered[
                (df_filtered['country'] == country) & 
                (df_filtered['variable'] == variable)
            ].sort_values('date')
            
            if len(data) > 0:
                # Plot the time series
                ax.plot(data['date'], data['value'], linewidth=1.5, color='#2E86AB')
                
                # Formatting
                ax.set_title(country, fontsize=11, fontweight='bold')
                ax.set_xlabel('Date', fontsize=9)
                ax.set_ylabel(var_info[variable]['ylabel'], fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.tick_params(axis='both', labelsize=8)
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # Add data range info
                date_range = f"{data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}"
                ax.text(0.02, 0.98, date_range, transform=ax.transAxes, 
                       fontsize=7, verticalalignment='top', alpha=0.7,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
                ax.set_title(country, fontsize=11, fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        # Save plot if requested
        if save_plots and plot_path:
            filename = f"{plot_path}/{variable}_all_countries.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
    
    print("Visualization complete!")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# After running the filtering code and creating df_filtered, run:
# visualize_variables(df_filtered, save_plots=True, plot_path=plot_path)

# Or without saving:
# visualize_variables(df_filtered, save_plots=False)
