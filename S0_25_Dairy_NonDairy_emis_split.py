
import pandas as pd
import numpy as np
import os
from config_paths import get_input_base, get_results_base

def split_emissions():
    """
    Splits livestock emissions for Camels, Buffalo, Sheep, and Goats into
    dairy and non-dairy categories based on stock populations.
    """
    # Define input and output paths using config
    input_base = get_input_base()
    output_base = get_results_base()
    
    manure_stock_path = os.path.join(input_base, 'Manure_Stock', 'Environment_LivestockManure_with_ratio.csv')
    emissions_path = os.path.join(input_base, 'Emission', 'Emissions_livestock_E_All_Data_NOFLAG.csv')
    
    # Create an intermediate directory for the output if it doesn't exist
    intermediate_dir = os.path.join(output_base, 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    output_path = os.path.join(intermediate_dir, 'Emissions_livestock_dairy_split.csv')

    # --- 1. Load and process stock data ---
    print("Loading and processing stock data...")
    stock_df = pd.read_csv(manure_stock_path, encoding='utf-8')
    stock_df = stock_df[stock_df['Element'] == 'Stocks']
    
    years = [f'Y{i}' for i in range(2000, 2023)]
    id_vars = ['M49_Country_Code', 'Area', 'Item']
    stock_df = stock_df[id_vars + years]

    # --- 2. Calculate stock ratios ---
    print("Calculating stock ratios...")
    # Define animal pairs for splitting
    animal_map = {
        'Buffaloes': ('Buffalo, dairy', 'Buffalo, non-dairy'),
        'Camels': ('Camel, dairy', 'Camel, non-dairy'),
        'Goats': ('Goats, dairy', 'Goats, non-dairy'),
        'Sheep': ('Sheep, dairy', 'Sheep, non-dairy')
    }
    
    ratio_dfs = []

    for _, (dairy_item, non_dairy_item) in animal_map.items():
        dairy_stock = stock_df[stock_df['Item'] == dairy_item].set_index('M49_Country_Code')[years]
        non_dairy_stock = stock_df[stock_df['Item'] == non_dairy_item].set_index('M49_Country_Code')[years]

        # Align indices to ensure correct addition
        dairy_stock, non_dairy_stock = dairy_stock.align(non_dairy_stock, fill_value=0)
        
        total_stock = dairy_stock + non_dairy_stock

        # Calculate ratios, handle division by zero
        dairy_ratio = (dairy_stock.div(total_stock)).replace([np.inf, -np.inf], np.nan)
        non_dairy_ratio = (non_dairy_stock.div(total_stock)).replace([np.inf, -np.inf], np.nan)
        
        # Rule 3: If ratio is nan, it means total_stock was 0 or data was missing.
        # When we apply this ratio later, the emission will be split into NaN.
        # We need a rule to handle this: put all emissions into non-dairy.
        # We achieve this by setting non_dairy_ratio to 1 and dairy_ratio to 0 where nans exist.
        dairy_ratio_filled = dairy_ratio.fillna(0)
        non_dairy_ratio_filled = non_dairy_ratio.fillna(0)
        
        # Where total stock is zero, both ratios are NaN. Make non-dairy 1.
        non_dairy_ratio_filled[dairy_ratio.isnull() & non_dairy_ratio.isnull()] = 1

        dairy_ratio_filled = dairy_ratio_filled.reset_index()
        non_dairy_ratio_filled = non_dairy_ratio_filled.reset_index()

        dairy_ratio_filled['Item'] = dairy_item
        non_dairy_ratio_filled['Item'] = non_dairy_item
        
        ratio_dfs.extend([dairy_ratio_filled, non_dairy_ratio_filled])

    ratios = pd.concat(ratio_dfs).set_index(['M49_Country_Code', 'Item'])[years]

    # --- 3. Load and split emissions data ---
    print("Loading and splitting emissions data...")
    emis_df = pd.read_csv(emissions_path, encoding='utf-8')
    
    elements_to_split = [
        'Enteric fermentation (Emissions CH4)',
        'Manure management (Emissions CH4)',
        'Manure management (Emissions N2O)',
        'Manure left on pasture (Emissions N2O)',
        'Manure applied to soils (Emissions N2O)'
    ]
    
    items_to_split = list(animal_map.keys())

    # Separate data into parts to be split and parts to be kept as is
    emis_to_split = emis_df[emis_df['Element'].isin(elements_to_split) & emis_df['Item'].isin(items_to_split)]
    emis_to_keep = emis_df[~(emis_df['Element'].isin(elements_to_split) & emis_df['Item'].isin(items_to_split))]

    new_rows = []

    for _, row in emis_to_split.iterrows():
        country_code = row['M49_Country_Code']
        original_item = row['Item']
        
        dairy_item, non_dairy_item = animal_map[original_item]
        
        try:
            dairy_ratio_vals = ratios.loc[(country_code, dairy_item)]
            non_dairy_ratio_vals = ratios.loc[(country_code, non_dairy_item)]
            
            # Create dairy row
            dairy_row = row.to_dict()
            dairy_row['Item'] = dairy_item
            for year in years:
                dairy_row[year] = row[year] * dairy_ratio_vals[year]
            new_rows.append(dairy_row)

            # Create non-dairy row
            non_dairy_row = row.to_dict()
            non_dairy_row['Item'] = non_dairy_item
            for year in years:
                non_dairy_row[year] = row[year] * non_dairy_ratio_vals[year]
            new_rows.append(non_dairy_row)

        except KeyError:
            # Rule 3 (fallback): Stock data for this country is missing. Assign all to non-dairy.
            non_dairy_row = row.to_dict()
            non_dairy_row['Item'] = non_dairy_item
            # Emissions for non-dairy are kept as 100%, dairy is 0.
            new_rows.append(non_dairy_row) 
            
            dairy_row = row.to_dict()
            dairy_row['Item'] = dairy_item
            for year in years:
                dairy_row[year] = 0.0 # Assign 0 emissions to dairy
            new_rows.append(dairy_row)


    # --- 4. Combine and finalize ---
    print("Combining and finalizing data...")
    split_emis_df = pd.DataFrame(new_rows)
    
    final_df = pd.concat([emis_to_keep, split_emis_df], ignore_index=True)
    
    # Sort the dataframe
    final_df = final_df.sort_values(by=['M49_Country_Code', 'Item', 'Element']).reset_index(drop=True)
    
    # Save to file
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == '__main__':
    split_emissions()
