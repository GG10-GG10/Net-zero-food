import pandas as pd
import numpy as np

# Define file paths
dict_v3_path = '../../src/dict_v3.xlsx'
luh2_summary_path = '../../input/Land/LUH2_data_summary.xlsx'
fao_landuse_path = '../../input/Land/Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv'
output_path = '../../input/Land/Land_cover_base_refill.xlsx'

# 1. Read region data and filter M49 country codes
df_region = pd.read_excel(dict_v3_path, sheet_name='region')
m49_codes = df_region[df_region['Region_label_new'] != 'no']['M49_Country_Code'].unique()

# Define land cover types
land_covers = ['cropland', 'forest', 'grassland']

# Create the initial DataFrame structure
data = []
for code in m49_codes:
    for cover in land_covers:
        data.append({'M49_Country_Code': code, 'Land cover': cover})

df_result = pd.DataFrame(data)

# Add year columns and Unit column
year_columns = [f'Y{year}' for year in range(2010, 2021)]
for col in year_columns:
    df_result[col] = np.nan

df_result['Unit'] = 'ha'

# Set index for easier data filling
df_result.set_index(['M49_Country_Code', 'Land cover'], inplace=True)

# 2. Fill data from LUH2_data_summary.xlsx
print("Attempting to fill data from LUH2_data_summary.xlsx...")
df_luh2 = pd.read_excel(luh2_summary_path, sheet_name='Land cover')
df_luh2.rename(columns={'M49': 'M49_Country_Code', 'variable': 'Land cover'}, inplace=True)
df_luh2 = df_luh2.melt(id_vars=['M49_Country_Code', 'Land cover', 'Unit'], var_name='Year', value_name='Value')
df_luh2['Year'] = 'Y' + df_luh2['Year'].astype(str)

# FIX: Ensure 'Value' column is numeric, converting errors to NaN
df_luh2['Value'] = pd.to_numeric(df_luh2['Value'], errors='coerce')

# Pivot LUH2 data to match the result format
df_luh2_pivot = df_luh2.pivot_table(index=['M49_Country_Code', 'Land cover'], columns='Year', values='Value').reset_index()
df_luh2_pivot.set_index(['M49_Country_Code', 'Land cover'], inplace=True)


# Update the result DataFrame with LUH2 data
# Only update where the index matches
common_index = df_result.index.intersection(df_luh2_pivot.index)
df_result.update(df_luh2_pivot)
print(f"Filled {len(common_index)} rows from LUH2 data.")


# 3. Fill missing data from Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv
print("\nAttempting to fill remaining missing data from FAOSTAT csv...")
df_fao = pd.read_csv(fao_landuse_path, encoding='latin1')

# Filter for 'Area' element
df_fao = df_fao[df_fao['Element'] == 'Area']

# Map FAO 'Item' to our 'Land cover'
item_to_landcover = {
    'Cropland': 'cropland',
    'Pasture land': 'grassland',
    'Forest land': 'forest'
}
df_fao['Land cover'] = df_fao['Item'].map(item_to_landcover)

# Drop rows where 'Land cover' is NaN (i.e., items we don't need)
df_fao.dropna(subset=['Land cover'], inplace=True)

# Rename the area code column to the standardized name.
# The KeyError confirms 'Area Code (M49)' is wrong. We assume 'Area Code' is the correct original name.
df_fao.rename(columns={'Area Code': 'M49_Country_Code'}, inplace=True)

# Filter for the years we need and pivot
fao_year_cols = [f'Y{year}' for year in range(2010, 2021)]
df_fao_filtered = df_fao[['M49_Country_Code', 'Land cover'] + fao_year_cols]

# Convert units from 1000 ha to ha
for col in fao_year_cols:
    # Use pd.to_numeric to handle potential non-numeric data gracefully
    df_fao_filtered[col] = pd.to_numeric(df_fao_filtered[col], errors='coerce') * 1000

# Group by country and land cover in case of duplicates, taking the mean
df_fao_pivot = df_fao_filtered.groupby(['M49_Country_Code', 'Land cover']).mean().reset_index()
df_fao_pivot.set_index(['M49_Country_Code', 'Land cover'], inplace=True)


# Identify rows that are still completely null
rows_to_fill = df_result[df_result[year_columns].isnull().all(axis=1)]
print(f"Found {len(rows_to_fill)} rows with missing data to fill from FAOSTAT.")

# Update only the rows that are missing data
df_result.update(df_fao_pivot, overwrite=False) # `overwrite=False` ensures we only fill NaNs

# Final check
missing_after_fao = df_result[df_result[year_columns].isnull().all(axis=1)]
print(f"{len(missing_after_fao)} rows are still missing all year data after all steps.")
if not missing_after_fao.empty:
    print("Missing combinations (M49_Country_Code, Land cover):")
    print(missing_after_fao.index.tolist())

# Reset index to turn multi-index back into columns
df_result.reset_index(inplace=True)

# Merge with region data to add Region_label_new
df_region_info = df_region[['M49_Country_Code', 'Region_label_new']].drop_duplicates()
df_result = pd.merge(df_result, df_region_info, on='M49_Country_Code', how='left')

# Reorder columns to have Region_label_new after M49_Country_Code
cols = df_result.columns.tolist()
if 'Region_label_new' in cols:
    cols.insert(1, cols.pop(cols.index('Region_label_new')))
    df_result = df_result[cols]

# Save the final DataFrame to Excel
df_result.to_excel(output_path, index=False)

print(f"\nProcessing complete. Output saved to {output_path}")
