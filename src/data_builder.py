import pandas as pd
import numpy as np
from dbnomics import fetch_series
import logging
from typing import List, Dict, Optional, Tuple
from .config import config

logger = logging.getLogger(__name__)


def find_max_complete_period(
    df_wide: pd.DataFrame,
    country: str,
    start_constraint: Optional[pd.Timestamp] = None,
    end_constraint: Optional[pd.Timestamp] = None
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Find the maximum continuous period with all three variables for a country.
    
    Args:
        df_wide: Wide-format dataframe with columns: country, date, exchange_rate, cpi, policy_rate
        country: Country name to filter for
        start_constraint: Optional earliest allowed start date
        end_constraint: Optional latest allowed end date
        
    Returns:
        Tuple of (start_date, end_date) for the longest continuous complete period,
        or (None, None) if no complete data exists.
    """
    # Filter for this country
    country_data = df_wide[df_wide['country'] == country].copy().sort_values('date')
    
    # Apply date constraints if provided
    if start_constraint is not None:
        country_data = country_data[country_data['date'] >= start_constraint]
    if end_constraint is not None:
        country_data = country_data[country_data['date'] <= end_constraint]
    
    if len(country_data) == 0:
        return None, None
    
    # Identify rows with all three variables present
    country_data['complete'] = (
        country_data['exchange_rate'].notna() & 
        country_data['cpi'].notna() & 
        country_data['policy_rate'].notna()
    )
    
    # Create groups of consecutive complete observations
    country_data['group'] = (country_data['complete'] != country_data['complete'].shift()).cumsum()
    
    # Filter to only complete rows
    complete_rows = country_data[country_data['complete']].copy()
    
    if len(complete_rows) == 0:
        return None, None
    
    # Find the group with the most observations
    group_sizes = complete_rows.groupby('group').size()
    max_group = group_sizes.idxmax()
    
    # Get the date range for this group
    max_period = complete_rows[complete_rows['group'] == max_group]
    start_date = max_period['date'].min()
    end_date = max_period['date'].max()
    
    return start_date, end_date


class DataBuilder:
    """
    Fetches macroeconomic data from DBnomics (IMF/IFS, OECD) and constructs 
    the panel dataset for STR/BUIP estimation.
    """
    
    # ------------------------------
    # Configuration: Country Groups
    # ------------------------------
    MP_COUNTRIES = {
        "AU": "Australia", "ID": "Indonesia", "TR": "Türkiye"
    }

    MM_COUNTRIES = {
        "BR": "Brazil", "KR": "Korea", "MX": "Mexico", 
        "NZ": "New Zealand", "PH": "Philippines", "TH": "Thailand"
    }

    OTHER_COUNTRIES = {
        "JP": "Japan", "GB": "United Kingdom", "CH": "Switzerland", 
        "CA": "Canada", "U2": "Euro Area"
    }

    OECD_REPLACEMENTS = {
        "Canada":          "OECD/DSD_KEI@DF_KEI/CAN.M.IRSTCI.PA._Z._Z._Z",
        "Switzerland":     "OECD/DSD_KEI@DF_KEI/CHE.M.IRSTCI.PA._Z._Z._Z",
        "Euro Area":       "OECD/DSD_KEI@DF_KEI/EA19.M.IRSTCI.PA._Z._Z._Z",
        "United Kingdom":  "OECD/DSD_KEI@DF_KEI/GBR.M.IRSTCI.PA._Z._Z._Z",
        "Japan":           "OECD/DSD_KEI@DF_KEI/JPN.M.IRSTCI.PA._Z._Z._Z",
    }

    def __init__(self):
        self.all_countries = {**self.MP_COUNTRIES, **self.MM_COUNTRIES, **self.OTHER_COUNTRIES}
        self.raw_dfs: List[pd.DataFrame] = []

    def run(self, build_mode: str = "restricted") -> pd.DataFrame:
        """Main execution method.
        
        Args:
            build_mode: 'restricted' (default) limits sample to 2000-01..2024-12.
                        'max' builds a maximum-length dataset per country where
                        exchange rate, CPI and policy rate are jointly available,
                        with country-specific trims for known gaps.
        """
        logger.info("Starting Data Build Process...")
        
        self.fetch_imf_data()
        self.fetch_us_data()
        self.fetch_euro_cpi()
        self.fetch_interpolated_quarterly()
        
        # Combine initial batch
        if not self.raw_dfs:
            raise ValueError("No data fetched. Check internet connection or DBnomics API status.")
            
        df_all = pd.concat(self.raw_dfs, ignore_index=True)
        df_all["date"] = pd.to_datetime(df_all["period"]).dt.to_period("M").dt.to_timestamp("M")
        
        # Drop potential duplicates (e.g., from interpolations for AU/NZ)
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=["country", "variable", "date"], keep="last")
        after = len(df_all)
        if after < before:
            logger.info(f"Dropped {before - after} duplicate (country,variable,date) rows")
        
        # Advanced Processing
        df_all = self.process_indonesia(df_all)
        df_all = self.process_oecd_replacements(df_all)
        
        # Feature Engineering
        final_df = self.engineer_features(df_all, mode=build_mode)
        
        # Save
        if build_mode == "max":
            output_path = config.DATA_DIR / "df_panel_max.csv"
        else:
            output_path = config.DATA_DIR / "df_panel_final.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        return final_df

    def _standardize_df(self, df: pd.DataFrame, country: str, variable: str, sid: str, interpolated: bool = False) -> pd.DataFrame:
        """Helper to standardize DBnomics response columns."""
        df["country"] = country
        df["variable"] = variable
        df["series_id"] = sid
        df["source"] = "IMF/IFS"
        df["freq"] = "M"
        df["interpolated"] = interpolated
        return df

    def fetch_imf_data(self):
        """Fetches basic Exchange Rate, CPI, and Policy Rates from IMF."""
        logger.info("Fetching IMF Series...")
        
        # 1. Exchange Rate & CPI
        for iso2, country in self.all_countries.items():
            for var, code in [("exchange_rate", "ENDE_XDC_USD_RATE"), ("cpi", "PCPI_IX")]:
                sid = f"IMF/IFS/M.{iso2}.{code}"
                try:
                    df = fetch_series(sid)
                    self.raw_dfs.append(self._standardize_df(df, country, var, sid))
                except Exception:
                    continue # Silent fail is expected for some missing series

        # 2. Monetary Policy Rate (MP Countries)
        for iso2, country in self.MP_COUNTRIES.items():
            sid = f"IMF/IFS/M.{iso2}.FPOLM_PA"
            try:
                df = fetch_series(sid)
                self.raw_dfs.append(self._standardize_df(df, country, "policy_rate", sid))
            except Exception:
                continue

        # 3. Money Market Rate (MM Countries)
        for iso2, country in self.MM_COUNTRIES.items():
            sid = f"IMF/IFS/M.{iso2}.FIMM_PA"
            try:
                df = fetch_series(sid)
                self.raw_dfs.append(self._standardize_df(df, country, "policy_rate", sid))
            except Exception:
                logger.warning(f"Failed to fetch MM rate for {country}")

    def fetch_us_data(self):
        """Fetches US benchmark data."""
        logger.info("Fetching US Data...")
        us_map = {"cpi": "PCPI_IX", "policy_rate": "FPOLM_PA"}
        for var, code in us_map.items():
            sid = f"IMF/IFS/M.US.{code}"
            try:
                df = fetch_series(sid)
                self.raw_dfs.append(self._standardize_df(df, "United States", var, sid))
            except Exception as e:
                logger.error(f"Error fetching US {var}: {e}")

    def fetch_euro_cpi(self):
        sid = "IMF/IFS/M.U2.PCPIHA_IX"
        try:
            df = fetch_series(sid)
            self.raw_dfs.append(self._standardize_df(df, "Euro Area", "cpi", sid))
        except Exception:
            logger.warning("Euro Area CPI fetch failed.")

    def fetch_interpolated_quarterly(self):
        """Fetches quarterly CPI for AU/NZ and interpolates to monthly via DBnomics API."""
        logger.info("Fetching and Interpolating Quarterly CPI (AU/NZ)...")
        filters = [{"code": "interpolate", "parameters": {"frequency": "monthly", "method": "spline"}}]
        try:
            df = fetch_series(["IMF/IFS/Q.AU.PCPI_IX", "IMF/IFS/Q.NZ.PCPI_IX"], filters=filters)
            cc_map = {"AU": "Australia", "NZ": "New Zealand"}
            df["country"] = df["series_code"].str.split(".").str[1].map(cc_map)
            df["variable"] = "cpi"
            df["series_id"] = "Interp_Quarterly"
            df["source"] = "IMF/IFS"
            df["freq"] = "M"
            df["interpolated"] = True
            self.raw_dfs.append(df)
        except Exception as e:
            logger.error(f"Interpolation fetch failed: {e}")

    def process_indonesia(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Fills gaps in Indonesia's policy rate using alternative series."""
        logger.info("Processing Indonesia Gaps...")
        try:
            sid_alt = "IMF/IFS/M.ID.FIMM_PA"
            alt = fetch_series(sid_alt)
            alt["date"] = pd.to_datetime(alt["period"]).dt.to_period("M").dt.to_timestamp("M")
            
            # Extract Primary and Alt
            mask_id = (df_all.country == "Indonesia") & (df_all.variable == "policy_rate")
            prim = df_all.loc[mask_id, ["date", "value"]].rename(columns={"value": "value_prim"})
            alt_data = alt[["date", "value"]].rename(columns={"value": "value_alt"})
            
            merged = prim.merge(alt_data, on="date", how="outer").sort_values("date")
            merged["value"] = merged["value_prim"].combine_first(merged["value_alt"])
            
            # Create new rows
            new_rows = df_all[mask_id].drop(columns=["value"]).drop_duplicates(subset=["country", "variable"])
            # If new_rows is empty (no ID data initially), we construct it
            if new_rows.empty:
                # Construct generic row
                pass 
            
            # Merge back structure
            # Simplified: just delete old ID rows and append new filled ones
            df_clean = df_all[~mask_id].copy()
            
            # Reconstruct full ID dataframe
            id_filled = pd.DataFrame({
                "country": "Indonesia",
                "variable": "policy_rate",
                "date": merged["date"],
                "value": merged["value"],
                "source": "IMF/IFS (Combined)",
                "freq": "M",
                "interpolated": False,
                "series_id": "COMBINED"
            })
            
            return pd.concat([df_clean, id_filled], ignore_index=True)
        except Exception as e:
            logger.error(f"Indonesia processing failed: {e}")
            return df_all

    def process_oecd_replacements(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Replaces policy rates with OECD data for specific countries."""
        logger.info("Fetching OECD Replacements...")
        dfs_oecd = []
        for country, sid in self.OECD_REPLACEMENTS.items():
            try:
                dfo = fetch_series(sid)
                dfo["country"] = country
                dfo["variable"] = "policy_rate"
                dfo["series_id"] = sid
                dfo["source"] = "OECD"
                dfo["freq"] = "M"
                dfo["interpolated"] = False
                dfo["date"] = pd.to_datetime(dfo["period"]).dt.to_period("M").dt.to_timestamp("M")
                dfs_oecd.append(dfo)
            except Exception as e:
                logger.warning(f"OECD fetch failed for {country}: {e}")

        if dfs_oecd:
            df_oecd = pd.concat(dfs_oecd, ignore_index=True)
            # Remove existing policy rates for these countries
            mask_replace = (df_all["country"].isin(self.OECD_REPLACEMENTS.keys())) & (df_all["variable"] == "policy_rate")
            df_all = df_all[~mask_replace]
            # Append OECD
            cols = ["country", "variable", "date", "value", "source", "freq", "interpolated", "series_id"]
            df_all = pd.concat([df_all, df_oecd[cols]], ignore_index=True)
            
        return df_all

    def engineer_features(self, df_all: pd.DataFrame, mode: str = "restricted") -> pd.DataFrame:
        """Pivots data and calculates derived econometric variables.
        
        If mode == 'restricted', filter to 2000-01..2024-12 (legacy behavior).
        If mode == 'max', do not hard-cut dates; instead, later trim per-country
        based on availability and known gap rules.
        """
        logger.info(f"Engineering Features (mode={mode})...")
        
        # 1. Filter Date Range (restricted mode)
        if mode == "restricted":
            df_sample = df_all[(df_all["date"] >= "2000-01-01") & (df_all["date"] <= "2024-12-31")].copy()
        else:
            df_sample = df_all.copy()
        
        # 2. Pivot
        df_wide = df_sample.pivot_table(
            index=['country', 'date'],
            columns='variable',
            values='value'
        ).reset_index()

        # 2b. Build US benchmark BEFORE any country-specific filtering that may drop US rows
        us_source = df_wide[df_wide.country == 'United States'][["date", "cpi", "policy_rate"]] \
            .rename(columns={"cpi": "CPI_for", "policy_rate": "i_for"})
        # Ensure continuous monthly coverage for US: sort and forward-fill gaps
        us_source = us_source.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        us_source[["CPI_for", "i_for"]] = us_source[["CPI_for", "i_for"]].ffill()
        
        # 3. Patch Philippines missing datum (Specific fix)
        date0 = pd.Timestamp('2022-01-31')
        mask_ph = (df_wide['country'] == 'Philippines') & (df_wide['date'] == date0)
        if mask_ph.any() and df_wide.loc[mask_ph, 'policy_rate'].isna().all():
            # Interpolate
            lower = df_wide.loc[(df_wide['country'] == 'Philippines') & (df_wide['date'] == '2021-12-31'), 'policy_rate'].values
            upper = df_wide.loc[(df_wide['country'] == 'Philippines') & (df_wide['date'] == '2022-02-28'), 'policy_rate'].values
            if len(lower) > 0 and len(upper) > 0:
                df_wide.loc[mask_ph, 'policy_rate'] = (lower[0] + upper[0]) / 2.0

        # 4. In 'max' mode, find maximum continuous complete period per country
        if mode == "max":
            logger.info("Finding maximum complete period for each country...")
            # Ensure date is Timestamp month-end
            df_wide['date'] = pd.to_datetime(df_wide['date'])
            
            # Define date constraints for specific countries with known gaps
            date_constraints = {
                'Indonesia': {'start': pd.Timestamp('1984-09-30'), 'end': None},
                'Korea': {'start': pd.Timestamp('1976-08-31'), 'end': None},
                'Philippines': {'start': None, 'end': pd.Timestamp('2021-12-31')}
            }
            
            # Find maximum complete periods for each country (excluding US)
            country_periods = {}
            non_us_countries = [c for c in df_wide['country'].unique() if c != 'United States']
            
            for country in non_us_countries:
                constraints = date_constraints.get(country, {'start': None, 'end': None})
                start_date, end_date = find_max_complete_period(
                    df_wide,
                    country,
                    start_constraint=constraints.get('start'),
                    end_constraint=constraints.get('end')
                )
                if start_date and end_date:
                    country_periods[country] = {'start': start_date, 'end': end_date}
                    logger.info(f"  {country}: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
                else:
                    logger.warning(f"  {country}: No complete data period found")
            
            # Filter df_wide to keep only the complete periods per country (and keep all US data)
            filtered_rows = []
            
            # Keep US data for benchmark
            us_data = df_wide[df_wide['country'] == 'United States']
            filtered_rows.append(us_data)
            
            # Filter each country to its maximum complete period
            for country, period in country_periods.items():
                country_mask = (
                    (df_wide['country'] == country) &
                    (df_wide['date'] >= period['start']) &
                    (df_wide['date'] <= period['end'])
                )
                filtered_rows.append(df_wide[country_mask])
            
            before_rows = len(df_wide)
            df_wide = pd.concat(filtered_rows, ignore_index=True)
            df_wide = df_wide.sort_values(['country', 'date']).reset_index(drop=True)
            logger.info(f"Max mode: kept {len(df_wide)} / {before_rows} rows after period filtering")
        
        # 5. Merge US Data (Benchmark)
        # Use the previously built US source to avoid accidental drops
        us_ref = us_source.copy()
            
        df_final = df_wide[df_wide.country != 'United States'].merge(us_ref, on="date", how="left")
        # After merge, forward-fill any remaining US gaps by date
        df_final = df_final.sort_values(["country", "date"]).groupby("country", as_index=False).apply(
            lambda g: g.assign(
                CPI_for=g["CPI_for"].ffill(),
                i_for=g["i_for"].ffill()
            )
        ).reset_index(drop=True)
        
        # 6. Rename
        df_final = df_final.rename(columns={
            "exchange_rate": "S",
            "cpi": "CPI_dom",
            "policy_rate": "i_dom",
        })
        
        # 7. Derived Variables
        
        df_final["s"] = np.log(df_final["S"])
        df_final["p_dom"] = np.log(df_final["CPI_dom"])
        df_final["p_for"] = np.log(df_final["CPI_for"])
        

        # Changes
        df_final["r_s"] = df_final.groupby("country")["s"].diff() # Returns
        df_final["q"] = df_final["s"] + df_final["p_for"] - df_final["p_dom"] # Real Exchange Rate
        
        # Effective Monthly Interest Rates
        # Input is annual percentage (e.g., 5.0 for 5%). Output is monthly decimal.
        def ann_to_month_eff(r_pa_pct):
            return (1.0 + r_pa_pct / 100.0) ** (1.0 / 12.0) - 1.0

        df_final["i_dom"] = ann_to_month_eff(df_final["i_dom"])
        df_final["i_for"] = ann_to_month_eff(df_final["i_for"])
        df_final["i_diff"] = df_final["i_for"] - df_final["i_dom"]
        
        # PPP and Excess Returns
        df_final["f_ppp"] = df_final["p_dom"] - df_final["p_for"]
        df_final["f_ppp_rel"] = df_final.groupby("country")["p_dom"].diff() - df_final.groupby("country")["p_for"].diff()
        df_final["r_q"] = df_final["r_s"] - df_final["f_ppp_rel"]
        
        # Final Selection
        keep_cols = [
            "country", "date", "S", "CPI_dom", "CPI_for",
            "s", "p_dom", "p_for", "r_s", "q",
            "i_dom", "i_for", "i_diff", "f_ppp", "f_ppp_rel", "r_q"
        ]
        
        # Drop duplicates in final set just in case
        df_out = df_final[keep_cols].drop_duplicates(subset=["country", "date"], keep="last")
        df_out = df_out.sort_values(["country", "date"]).reset_index(drop=True)

        # In max mode or general case, avoid over-aggressive dropna: require essential columns
        essential = ["S", "CPI_dom", "CPI_for", "i_dom", "i_for"]
        missing_essential = df_out[essential].isna().all(axis=None)
        df_out = df_out[df_out[essential].notna().all(axis=1)]

        # Some derived columns may still have NaNs in first rows due to differencing; trim leading NaNs when necessary
        def trim_leading_nans(group: pd.DataFrame) -> pd.DataFrame:
            idx = group.index
            # Find first index where r_s and q are not NaN
            valid = group["r_s"].notna() & group["q"].notna()
            if not valid.any():
                return group.iloc[0:0]
            first = valid.idxmax()
            return group.loc[first:]

        df_out = df_out.groupby("country", as_index=False).apply(trim_leading_nans).reset_index(drop=True)

        # Final safety: do not return an empty dataset silently
        if df_out.empty:
            logger.error("Engineer features produced an empty dataset. Likely missing US benchmarks or key series after merges.")
            raise ValueError("Engineered dataset is empty. Check DBnomics availability for US CPI/policy rate and country series.")
        return df_out
