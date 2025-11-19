import numpy as np
import pandas as pd
from src.config import config
from src.utils import load_panel_data, prepare_regression_matrices
from src.econometrics import STRModel
from src.diagnostics import luukkonen_linearity_test
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Load Data
    logging.info("Loading data...")
    # Replace with actual filename inside your data folder
    data_path = config.DATA_DIR / "sample_panel.csv" 
    
    # Mock data creation if file doesn't exist (for demonstration)
    if not data_path.exists():
        logging.warning("Data file not found. Generating synthetic data.")
        np.random.seed(42)
        T = 200
        df = pd.DataFrame({
            'Date': pd.date_range(start='2000-01-01', periods=T, freq='M'),
            'Country': ['US'] * T,
            'dep_var': np.random.randn(T),
            'indep_var': np.random.randn(T),
            'z_var': np.random.randn(T)
        })
    else:
        df = load_panel_data(data_path)

    # 2. Prepare Data (Single Country Example)
    # In a loop, you would filter by country here
    matrices = prepare_regression_matrices(
        df,
        y_col='dep_var',
        linear_vars=['indep_var'],
        nonlinear_vars=['indep_var'], # e.g. variables that switch influence
        z_col='z_var'
    )
    
    y = matrices['y']
    X_lin = matrices['X_lin']
    X_non = matrices['X_non']
    z = matrices['z']

    # 3. Linearity Test
    logging.info("Running Luukkonen Linearity Test...")
    # Note: X for test usually combines linear and nonlinear candidates
    linearity_res = luukkonen_linearity_test(y, X_lin, z)
    logging.info(f"Linearity Test p-value: {linearity_res['p_value']:.4f}")

    if linearity_res['p_value'] < 0.05:
        logging.info("Null hypothesis of linearity rejected. Proceeding to STR estimation.")
        
        # 4. Initialize Model
        model = STRModel(y, X_lin, X_non, z, transition_type='LSTR')
        
        # 5. Grid Search
        logging.info("Performing Grid Search...")
        gamma_grid = np.linspace(0.1, 20, config.GRID_POINTS)
        
        # Trim z for c search (15% to 85% quantiles)
        z_sorted = np.sort(z)
        trim_n = int(len(z) * config.TRIM_PERCENT)
        c_grid = np.linspace(z_sorted[trim_n], z_sorted[-trim_n], config.GRID_POINTS)
        
        gamma_init, c_init, best_rss = model.grid_search(gamma_grid, c_grid)
        logging.info(f"Best Grid params: gamma={gamma_init:.2f}, c={c_init:.2f}")
        
        # 6. NLS Estimation
        logging.info("Running NLS Estimation...")
        results = model.fit(init_gamma=gamma_init, init_c=c_init)
        
        print("\n--- ESTIMATION RESULTS ---")
        print(f"Gamma: {results['gamma']:.4f}")
        print(f"Threshold (c): {results['c']:.4f}")
        print(f"RSS: {results['rss']:.4f}")
        print("Coefficients:", results['coefficients'])
        
    else:
        logging.info("Model appears linear. Skipping STR estimation.")

if __name__ == "__main__":
    main()
