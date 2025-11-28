"""
Analysis script to check for missing monthly observations in df_sample dataset
"""
import pandas as pd
import numpy as np

def analyze_missing_observations(df_sample):
    """
    Comprehensive analysis of missing monthly observations in the dataset.
    
    Parameters:
    -----------
    df_sample : pd.DataFrame
        The dataset with columns: country, date, variable, value, source, freq, series_id, interpolated
    
    Returns:
    --------
    dict : Dictionary containing various analysis results
    """
    
    results = {}
    
    # 1. Basic info
    print("="*80)
    print("BASIC DATASET INFORMATION")
    print("="*80)
    print(f"Total rows: {len(df_sample):,}")
    print(f"Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")
    print(f"Countries: {df_sample['country'].nunique()}")
    print(f"Variables: {df_sample['variable'].nunique()}")
    print(f"\nCountries in dataset:\n{sorted(df_sample['country'].unique())}")
    print(f"\nVariables in dataset:\n{sorted(df_sample['variable'].unique())}")
    
    # 2. Check for missing values in the 'value' column
    print("\n" + "="*80)
    print("MISSING VALUES IN 'VALUE' COLUMN")
    print("="*80)
    missing_values = df_sample['value'].isna().sum()
    print(f"Total missing values: {missing_values:,} ({missing_values/len(df_sample)*100:.2f}%)")
    
    if missing_values > 0:
        print("\nMissing values by country and variable:")
        missing_by_group = df_sample[df_sample['value'].isna()].groupby(['country', 'variable']).size()
        print(missing_by_group.to_string())
    
    # 3. Check for complete monthly time series for each country-variable combination
    print("\n" + "="*80)
    print("MISSING MONTHLY OBSERVATIONS (GAPS IN TIME SERIES)")
    print("="*80)
    
    gaps_found = []
    
    for (country, variable), group in df_sample.groupby(['country', 'variable']):
        # Sort by date
        group = group.sort_values('date')
        
        # Get date range
        min_date = group['date'].min()
        max_date = group['date'].max()
        
        # Create complete monthly range
        complete_range = pd.date_range(start=min_date, end=max_date, freq='M')
        
        # Find missing dates
        existing_dates = set(group['date'])
        missing_dates = [d for d in complete_range if d not in existing_dates]
        
        if missing_dates:
            gaps_found.append({
                'country': country,
                'variable': variable,
                'min_date': min_date,
                'max_date': max_date,
                'expected_obs': len(complete_range),
                'actual_obs': len(group),
                'missing_obs': len(missing_dates),
                'missing_dates': missing_dates[:10]  # Show first 10 missing dates
            })
    
    if gaps_found:
        print(f"Found {len(gaps_found)} country-variable combinations with missing monthly observations:\n")
        for gap in gaps_found:
            print(f"\n{gap['country']} - {gap['variable']}:")
            print(f"  Date range: {gap['min_date'].strftime('%Y-%m')} to {gap['max_date'].strftime('%Y-%m')}")
            print(f"  Expected observations: {gap['expected_obs']}")
            print(f"  Actual observations: {gap['actual_obs']}")
            print(f"  Missing observations: {gap['missing_obs']}")
            if gap['missing_dates']:
                print(f"  First missing dates: {', '.join([d.strftime('%Y-%m') for d in gap['missing_dates']])}")
    else:
        print("✓ No gaps found! All country-variable combinations have complete monthly time series.")
    
    results['gaps'] = gaps_found
    
    # 4. Summary table: observations per country-variable
    print("\n" + "="*80)
    print("OBSERVATION COUNT BY COUNTRY AND VARIABLE")
    print("="*80)
    
    obs_summary = df_sample.groupby(['country', 'variable']).agg({
        'date': ['min', 'max', 'count'],
        'value': lambda x: x.notna().sum()  # Count non-null values
    }).round(0)
    
    obs_summary.columns = ['Start Date', 'End Date', 'Total Obs', 'Non-Null Obs']
    obs_summary['Missing Values'] = obs_summary['Total Obs'] - obs_summary['Non-Null Obs']
    
    print(obs_summary.to_string())
    results['summary'] = obs_summary
    
    # 5. Check for duplicates
    print("\n" + "="*80)
    print("DUPLICATE OBSERVATIONS CHECK")
    print("="*80)
    
    duplicates = df_sample.duplicated(subset=['country', 'date', 'variable'], keep=False)
    dup_count = duplicates.sum()
    
    if dup_count > 0:
        print(f"⚠ Found {dup_count} duplicate observations!")
        print("\nDuplicate entries:")
        print(df_sample[duplicates].sort_values(['country', 'variable', 'date']))
    else:
        print("✓ No duplicate observations found.")
    
    results['duplicates'] = dup_count
    
    # 6. Wide format check (after pivot)
    print("\n" + "="*80)
    print("WIDE FORMAT ANALYSIS (PIVOT TABLE)")
    print("="*80)
    
    try:
        df_wide = df_sample.pivot_table(
            index=['country', 'date'],
            columns='variable',
            values='value'
        ).reset_index()
        
        print(f"Wide format shape: {df_wide.shape}")
        print(f"Columns: {list(df_wide.columns)}")
        
        # Check missing values in wide format
        print("\nMissing values per variable (in wide format):")
        for col in df_wide.columns:
            if col not in ['country', 'date']:
                missing = df_wide[col].isna().sum()
                total = len(df_wide)
                print(f"  {col}: {missing:,} / {total:,} ({missing/total*100:.2f}%)")
        
        # Check which country-date combinations have complete data
        complete_rows = df_wide.notna().all(axis=1).sum()
        print(f"\nComplete observations (all variables present): {complete_rows:,} / {len(df_wide):,}")
        
        results['wide_format'] = df_wide
        
    except Exception as e:
        print(f"Error creating wide format: {e}")
    
    return results


# Example usage (add this to your notebook):
# results = analyze_missing_observations(df_sample)
