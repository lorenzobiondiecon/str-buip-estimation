# Quick code to add to your Jupyter notebook
# This will filter df_sample to retain only complete observation periods

# Import the filtering function
from complete_observations_filter import filter_complete_observations, print_summary

# Apply the filter
df_filtered, country_periods = filter_complete_observations(df_sample)

# Print summary
print_summary(country_periods, df_filtered)

# Now you can use df_filtered instead of df_sample for your analysis
# df_filtered contains only the maximum complete observation periods for each country
