import numpy as np
import pandas as pd
from src.config import config
from src.utils import load_panel_data, build_str_data, build_beh_sample
from src.econometrics import STRModel, BUIPModel
from src.diagnostics import luukkonen_linearity_test
from src.data_builder import DataBuilder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Load Data
    data_path = config.DATA_DIR / "df_panel_final.csv"
    
    if not data_path.exists():
        logging.info("Data file not found. Attempting to download and build from DBnomics...")
        try:
            builder = DataBuilder()
            builder.run()
        except Exception as e:
            logging.error(f"Failed to build data: {e}")
            return

    # Load the freshly built data
    df = load_panel_data(data_path)

    # ==========================================
    # PART A: STR ESTIMATION
    # ==========================================
    logging.info("\n=== Starting STR Estimation Pipeline ===")
    
    try:
        # Note: variables in df_panel_final.csv match the build logic: 'r_s', 'q', etc.
        # We use 'r_s' (returns) as dependent for STR typically, or 'r_q' (excess returns)
        # Adjust y_col based on your specific model theory. 
        # Here assuming y = r_s (exchange rate return)
        
        # We need to create lags if they aren't in the final CSV (DataBuilder saves raw vars)
        # Let's ensure lags exist:
        df['rs_lag1'] = df.groupby('country')['r_s'].shift(1)
        df['eta_lag1'] = df.groupby('country')['q'].shift(1)
        
        # Filter for a specific country for demonstration (e.g., Australia)
        # The code runs on the whole panel, but STR is usually timeseries specific.
        country_iso = "Australia"
        country_df = df[df.country == country_iso].copy()
        
        if country_df.empty:
            logging.warning(f"No data for {country_iso}, skipping STR demo.")
        else:
            str_df = build_str_data(country_df, y_col='r_s', z_col='q')
            
            # Linearity Test
            import statsmodels.api as sm
            X_lin_test = sm.add_constant(str_df[['rs_lag1']].values)
            lin_test = luukkonen_linearity_test(str_df['y'].values, X_lin_test, str_df['z'].values)
            logging.info(f"STR Linearity Test p-value ({country_iso}): {lin_test['p_value']:.5f}")

            if lin_test['p_value'] < 0.05:
                logging.info("Reject Linearity. Fitting STR Model...")
                str_model = STRModel(str_df, z_col='z')
                
                g_grid = np.linspace(0.5, 20, config.GRID_POINTS)
                z_vals = str_df['z'].values
                trim = int(len(z_vals) * 0.15)
                zs = np.sort(z_vals)
                c_grid = np.linspace(zs[trim], zs[-trim], config.GRID_POINTS)
                
                best_init = str_model.grid_search(g_grid, c_grid)
                res_str = str_model.fit(best_init)
                print(f"STR Results ({country_iso}):\n{res_str.params}")
            else:
                logging.info("STR: Model is Linear.")

    except Exception as e:
        logging.error(f"STR Pipeline Failed: {e}")


    # ==========================================
    # PART B: BUIP ESTIMATION
    # ==========================================
    logging.info("\n=== Starting BUIP Estimation Pipeline ===")
    
    try:
        # DataBuilder output has all cols needed for BUIP: r_s, i_for, i_dom, q, S, s
        # But we need to make sure we have lags calculated *before* passing to build_beh_sample
        # actually build_beh_sample calculates the lags internally using .shift().
        
        country_iso = "Australia"
        buip_data_raw = df[df.country == country_iso].copy()
        
        buip_df = build_beh_sample(buip_data_raw)
        buip_model = BUIPModel(buip_df)
        
        res_buip = buip_model.fit_multistart(n_starts=10)
        
        print(f"\nBUIP Results ({country_iso}):")
        print(res_buip.params)
        
    except Exception as e:
        logging.error(f"BUIP Pipeline Failed: {e}")

if __name__ == "__main__":
    main()
