import pandas as pd
import numpy as np

# Define file paths
dict_v3_path = '../../src/dict_v3.xlsx'
emission_path = '../../input/Emission/Emission_LULUCF_Historical_updated.xlsx'
land_cover_path = '../../input/Land/Land_cover_base_refill.xlsx'
output_path = '../../src/Forest_EF_recal.csv'

def recal_forest_ef():
    """
    Recalculates the forest carbon absorption factor.
    """
    # 1. Read region data and get list of countries
    region_df = pd.read_excel(dict_v3_path, sheet_name='region')
    country_codes = region_df[region_df['Region_label_new'] != 'no']['M49_Country_Code'].unique()
    
    # 2. Read and process historical emissions data
    emis_df = pd.read_excel(emission_path)
    emis_df = emis_df[(emis_df['Select'] == 1) & (emis_df['Land Category'] == 'Forest')]
    
    # Detect year columns (handles integers like 2010 or strings like 'Y2010')
    year_cols_int = [col for col in emis_df.columns if isinstance(col, int) and 2010 <= col <= 2020]
    year_cols_str = [col for col in emis_df.columns if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit() and 2010 <= int(col[1:]) <= 2020]
    
    emis_long = None
    if year_cols_int:
        year_cols = year_cols_int
        is_year_prefixed = False
    elif year_cols_str:
        year_cols = year_cols_str
        is_year_prefixed = True
    elif 'Year' in emis_df.columns:
        emis_df = emis_df[emis_df['Year'].between(2010, 2020)]
        if 'Value' not in emis_df.columns:
            raise ValueError("Long format data detected, but 'Value' column is missing in emission file.")
        emis_long = emis_df[['M49_Country_Code', 'Year', 'Value']].copy()
        emis_long.rename(columns={'Value': 'Emission'}, inplace=True)
        year_cols = [] # Flag to skip melting
    else:
        raise ValueError("Could not determine year columns in the emission data file. Looked for integers (2010), strings ('Y2010'), or a 'Year' column.")

    if year_cols: # If data was in wide format
        emis_long = emis_df.melt(id_vars=['M49_Country_Code'], value_vars=year_cols, var_name='Year', value_name='Emission')
        if is_year_prefixed:
            emis_long['Year'] = emis_long['Year'].str.replace('Y', '').astype(int)
        else:
            emis_long['Year'] = emis_long['Year'].astype(int)

    # Unit of emission is MtCO2/yr. We convert it to tCO2/yr
    emis_long['Emission'] = emis_long['Emission'] * 1_000_000

    # 3. Read and process forest area data
    land_df = pd.read_excel(land_cover_path)
    land_df = land_df[land_df['Land cover'] == 'forest']
    
    # Repeat the same logic for land data
    land_year_cols_int = [col for col in land_df.columns if isinstance(col, int) and 2010 <= col <= 2020]
    land_year_cols_str = [col for col in land_df.columns if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit() and 2010 <= int(col[1:]) <= 2020]
    
    land_long = None
    if land_year_cols_int:
        land_year_cols = land_year_cols_int
        is_land_year_prefixed = False
    elif land_year_cols_str:
        land_year_cols = land_year_cols_str
        is_land_year_prefixed = True
    elif 'Year' in land_df.columns:
        land_df = land_df[land_df['Year'].between(2010, 2020)]
        if 'Value' not in land_df.columns:
            raise ValueError("Long format data detected, but 'Value' column is missing in land cover file.")
        land_long = land_df[['M49_Country_Code', 'Year', 'Value']].copy()
        land_long.rename(columns={'Value': 'Area'}, inplace=True)
        land_year_cols = [] # Flag to skip melting
    else:
        raise ValueError("Could not determine year columns in the land cover file. Looked for integers (2010), strings ('Y2010'), or a 'Year' column.")

    if land_year_cols: # If data was in wide format
        land_long = land_df.melt(id_vars=['M49_Country_Code'], value_vars=land_year_cols, var_name='Year', value_name='Area')
        if is_land_year_prefixed:
            land_long['Year'] = land_long['Year'].str.replace('Y', '').astype(int)
        else:
            land_long['Year'] = land_long['Year'].astype(int)
    
    # ✅ 修复：Land_cover_base_refill.xlsx的Unit列已经是'ha'，不需要再乘1000！
    # 原代码错误地假设单位是kha，导致面积被放大1000倍，EF被缩小1000倍
    # Area is already in ha (as per the 'Unit' column in the file), no conversion needed.
    # land_long['Area'] = land_long['Area'] * 1000  # ❌ 这是错误的！

    # 4. Merge and calculate EF
    df = pd.merge(emis_long, land_long, on=['M49_Country_Code', 'Year'], how='outer')
    
    # Create a full template of all countries and years
    years = range(2010, 2021)
    full_template = pd.MultiIndex.from_product([country_codes, years], names=['M49_Country_Code', 'Year']).to_frame(index=False)
    
    df = pd.merge(full_template, df, on=['M49_Country_Code', 'Year'], how='left')

    # Calculate EF (Emission Factor) in tCO2/ha/yr
    # Use np.divide to handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        df['EF_recal'] = np.divide(df['Emission'], df['Area'])

    # 5. Handle missing/invalid data
    # Replace inf, -inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # also replace 0 with NaN for filling purposes, as requested
    df['EF_recal'] = df['EF_recal'].replace(0, np.nan)

    # a. Country-level fill (forward and backward)
    df['EF_recal'] = df.groupby('M49_Country_Code')['EF_recal'].transform(lambda x: x.ffill().bfill())

    # b. Regional-level fill
    region_map = pd.read_excel(dict_v3_path, sheet_name='region')[['M49_Country_Code', 'Region_agg2']].drop_duplicates()
    df = pd.merge(df, region_map, on='M49_Country_Code', how='left')
    
    regional_avg = df.groupby(['Region_agg2', 'Year'])['EF_recal'].transform('mean')
    df['EF_recal'].fillna(regional_avg, inplace=True)

    # c. World-level fill
    world_avg = df.groupby('Year')['EF_recal'].transform('mean')
    df['EF_recal'].fillna(world_avg, inplace=True)
    
    # Fill any remaining NaNs (e.g. if a whole year is missing) with an overall mean
    df['EF_recal'].fillna(df['EF_recal'].mean(), inplace=True)

    # 6. Finalize and save
    df['Unit'] = 'tCO2/ha/yr'
    
    # Pivot to wide format
    ef_final = df.pivot_table(index='M49_Country_Code', columns='Year', values='EF_recal').reset_index()
    unit_df = df[['M49_Country_Code', 'Unit']].drop_duplicates()
    ef_final = pd.merge(unit_df, ef_final, on='M49_Country_Code')
    
    # Add Item column
    ef_final.insert(0, 'Item', 'Forest')

    ef_final.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    recal_forest_ef()
