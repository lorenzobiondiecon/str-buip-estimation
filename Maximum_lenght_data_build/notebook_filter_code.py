"""
NOTEBOOK CELL CODE - Copy this into your Jupyter notebook after df_sample is created
This will filter df_sample to retain only complete observation periods for each country
"""

import pandas as pd
import numpy as np

# ============================================================================
# FILTER FOR COMPLETE OBSERVATIONS
# ============================================================================

def find_max_complete_period(df_wide, country, start_constraint=None, end_constraint=None):
    """Find the maximum continuous period with all three variables for a country."""
    
    # Filter for this country
    country_data = df_wide[df_wide['country'] == country].copy().sort_values('date')
    
    # Apply date constraints if provided
    if start_constraint:
        country_data = country_data[country_data['date'] >= start_constraint]
    if end_constraint:
        country_data = country_data[country_data['date'] <= end_constraint]
    
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


# ============================================================================
# REMOVE DUPLICATES FIRST
# ============================================================================
# Some countries (AU, NZ) may have duplicate observations due to interpolation
# Keep non-interpolated data when duplicates exist, otherwise keep first occurrence

print("Checking for duplicates...")
initial_count = len(df_sample)

# Sort by interpolated flag (False first) so we keep non-interpolated data preferentially
df_sample_dedup = df_sample.sort_values('interpolated').drop_duplicates(
    subset=['country', 'date', 'variable'], 
    keep='first'
).reset_index(drop=True)

removed = initial_count - len(df_sample_dedup)
if removed > 0:
    print(f"Removed {removed} duplicate observations")
else:
    print("No duplicates found")

# Convert to wide format
df_wide = df_sample_dedup.pivot_table(
    index=['country', 'date'],
    columns='variable',
    values='value'
).reset_index()

# Define date constraints for specific countries
date_constraints = {
    'Indonesia': {'start': pd.Timestamp('1984-09-30'), 'end': None},
    'Korea': {'start': pd.Timestamp('1976-08-31'), 'end': None},
    'Philippines': {'start': None, 'end': pd.Timestamp('2021-12-31')}
}

# Find complete periods for each country
country_periods = {}

for country in df_wide['country'].unique():
    constraints = date_constraints.get(country, {'start': None, 'end': None})
    
    start_date, end_date = find_max_complete_period(
        df_wide, 
        country,
        start_constraint=constraints['start'],
        end_constraint=constraints['end']
    )
    
    if start_date and end_date:
        country_periods[country] = {'start': start_date, 'end': end_date}

# Filter the original df_sample based on these periods
filtered_rows = []

for country, period in country_periods.items():
    country_mask = (
        (df_sample_dedup['country'] == country) &
        (df_sample_dedup['date'] >= period['start']) &
        (df_sample_dedup['date'] <= period['end'])
    )
    filtered_rows.append(df_sample_dedup[country_mask])

df_filtered = pd.concat(filtered_rows, ignore_index=True)
df_filtered = df_filtered.sort_values(['country', 'variable', 'date']).reset_index(drop=True)

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("="*80)
print("FILTERED DATASET SUMMARY")
print("="*80)

summary_data = []

for country, period in sorted(country_periods.items()):
    start = period['start']
    end = period['end']
    
    # Calculate number of months
    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    
    # Count observations
    obs_count = len(df_filtered[df_filtered['country'] == country])
    
    summary_data.append({
        'Country': country,
        'Start Date': start.strftime('%Y-%m'),
        'End Date': end.strftime('%Y-%m'),
        'Months': months,
        'Observations': obs_count,
        'Expected Obs': months * 3
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Verify completeness
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

missing_values = df_filtered['value'].isna().sum()
print(f"Missing values in filtered dataset: {missing_values}")

# Check for gaps in monthly sequences
print("\nChecking for gaps in monthly sequences...")
gaps_found = False

for country in df_filtered['country'].unique():
    for variable in df_filtered['variable'].unique():
        subset = df_filtered[
            (df_filtered['country'] == country) & 
            (df_filtered['variable'] == variable)
        ].sort_values('date')
        
        if len(subset) > 0:
            min_date = subset['date'].min()
            max_date = subset['date'].max()
            expected_range = pd.date_range(start=min_date, end=max_date, freq='M')
            
            if len(subset) != len(expected_range):
                print(f"  ⚠ Gap found in {country} - {variable}")
                gaps_found = True

if not gaps_found:
    print("  ✓ No gaps found in any series")

print("\n" + "="*80)
print(f"Original df_sample: {initial_count:,} observations")
print(f"After deduplication: {len(df_sample_dedup):,} observations")
print(f"Filtered df_filtered: {len(df_filtered):,} observations")
print("="*80)
