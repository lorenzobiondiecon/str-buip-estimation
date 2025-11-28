"""
Quick analysis code to paste into your Jupyter notebook
Copy and paste this into a new cell after df_sample is created
"""

# Quick Missing Observations Analysis for df_sample
print("="*80)
print("MISSING OBSERVATIONS ANALYSIS FOR df_sample")
print("="*80)

# 1. Basic info
print(f"\nTotal rows: {len(df_sample):,}")
print(f"Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")
print(f"Countries: {df_sample['country'].nunique()}")
print(f"Variables: {df_sample['variable'].nunique()}")

# 2. Missing values in 'value' column
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing_count = df_sample['value'].isna().sum()
print(f"Total missing values: {missing_count:,} ({missing_count/len(df_sample)*100:.2f}%)")

if missing_count > 0:
    print("\nMissing values by country and variable:")
    print(df_sample[df_sample['value'].isna()].groupby(['country', 'variable']).size())

# 3. Check for gaps in monthly time series
print("\n" + "="*80)
print("GAPS IN MONTHLY TIME SERIES")
print("="*80)

gaps_summary = []

for (country, variable), group in df_sample.groupby(['country', 'variable']):
    group = group.sort_values('date')
    min_date = group['date'].min()
    max_date = group['date'].max()
    
    # Create complete monthly range
    complete_range = pd.date_range(start=min_date, end=max_date, freq='M')
    
    # Find missing dates
    existing_dates = set(group['date'])
    missing_dates = [d for d in complete_range if d not in existing_dates]
    
    if missing_dates:
        gaps_summary.append({
            'Country': country,
            'Variable': variable,
            'Start': min_date.strftime('%Y-%m'),
            'End': max_date.strftime('%Y-%m'),
            'Expected': len(complete_range),
            'Actual': len(group),
            'Missing': len(missing_dates),
            'First_Missing_Dates': ', '.join([d.strftime('%Y-%m') for d in missing_dates[:5]])
        })

if gaps_summary:
    gaps_df = pd.DataFrame(gaps_summary)
    print(f"\nFound {len(gaps_summary)} series with gaps:\n")
    print(gaps_df.to_string(index=False))
else:
    print("\n✓ No gaps found! All series have complete monthly observations.")

# 4. Observation count summary
print("\n" + "="*80)
print("OBSERVATION COUNT BY COUNTRY-VARIABLE")
print("="*80)

summary = df_sample.groupby(['country', 'variable']).agg({
    'date': ['min', 'max', 'count'],
    'value': lambda x: x.notna().sum()
})
summary.columns = ['Start', 'End', 'Total_Obs', 'Non_Null']
summary['Missing_Values'] = summary['Total_Obs'] - summary['Non_Null']
print(summary)

# 5. Check duplicates
print("\n" + "="*80)
print("DUPLICATE CHECK")
print("="*80)
dup_count = df_sample.duplicated(subset=['country', 'date', 'variable']).sum()
if dup_count > 0:
    print(f"⚠ Found {dup_count} duplicates!")
    dups = df_sample[df_sample.duplicated(subset=['country', 'date', 'variable'], keep=False)]
    print(dups.sort_values(['country', 'variable', 'date']))
else:
    print("✓ No duplicates found.")

# 6. Wide format check
print("\n" + "="*80)
print("WIDE FORMAT MISSING DATA")
print("="*80)

df_wide_check = df_sample.pivot_table(
    index=['country', 'date'],
    columns='variable',
    values='value'
).reset_index()

print(f"Wide format shape: {df_wide_check.shape}")
print("\nMissing values per variable:")
for col in df_wide_check.columns:
    if col not in ['country', 'date']:
        missing = df_wide_check[col].isna().sum()
        print(f"  {col}: {missing:,} ({missing/len(df_wide_check)*100:.2f}%)")

complete_rows = df_wide_check.notna().all(axis=1).sum()
print(f"\nComplete rows (all variables): {complete_rows:,} / {len(df_wide_check):,} ({complete_rows/len(df_wide_check)*100:.2f}%)")

# Show some examples of incomplete rows
incomplete = df_wide_check[df_wide_check.isna().any(axis=1)]
if len(incomplete) > 0:
    print(f"\nFirst 10 incomplete rows:")
    print(incomplete.head(10))
