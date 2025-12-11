import pandas as pd
import numpy as np
import io
import os

# ================= 配置路径 =================
base_dict_path = "../../src/dict_v3.xlsx"
input_file_path = "../../input/Land/Feed_pasture/Feed_need_per_head_by_country_livestcok.xlsx"
output_file_path = "../../input/Land/Feed_pasture/Feed_need_per_head_by_country_livestcok_refilled.xlsx"

# ================= 辅助函数定义 =================

def clean_m49_code(series):
    """清洗 M49 代码：去除可能存在的单引号，并补全为3位字符串"""
    s = series.astype(str).str.replace("'", "", regex=False).str.strip()
    return s.str.zfill(3)

def fill_hierarchy(df, year_cols, region_map_df, level_cols):
    """
    通用填充逻辑：前后年份 -> 区域均值 -> 世界均值
    """
    df['M49_Country_Code'] = clean_m49_code(df['M49_Country_Code'])
    region_map_df['M49_Country_Code'] = clean_m49_code(region_map_df['M49_Country_Code'])

    # 1. 前后年份填充
    df[year_cols] = df[year_cols].replace(0, np.nan)
    df[year_cols] = df[year_cols].ffill(axis=1).bfill(axis=1)
    
    # 合并区域信息
    region_map_clean = region_map_df.drop_duplicates(subset=['M49_Country_Code'])
    df = df.merge(region_map_clean, on='M49_Country_Code', how='left')
    
    # 2. 区域均值填充
    region_group_cols = level_cols + ['Region_agg2']
    region_means = df.groupby(region_group_cols)[year_cols].transform('mean')
    df[year_cols] = df[year_cols].fillna(region_means)
    
    # 3. 世界均值填充
    global_means = df.groupby(level_cols)[year_cols].transform('mean')
    df[year_cols] = df[year_cols].fillna(global_means)
    
    if 'Region_agg2' in df.columns:
        df = df.drop(columns=['Region_agg2'])
        
    return df

def apply_proxies(df, year_cols, proxy_rules, level_col='Species', multiply_factor=True):
    """
    应用替补规则。
    """
    match_cols = ['M49_Country_Code']
    has_crop = 'Crop' in df.columns
    if has_crop:
        match_cols.append('Crop')
    
    lookup_cols = [level_col] + match_cols
    df_lookup = df.set_index(lookup_cols)[year_cols]
    df_lookup = df_lookup[~df_lookup.index.duplicated(keep='first')]

    print(f"  正在应用替补规则 (匹配列: {lookup_cols})...")

    for target, (source, factor) in proxy_rules.items():
        target_mask = (df[level_col] == target)
        data_matrix = df.loc[target_mask, year_cols]
        is_missing = (data_matrix.isna().all(axis=1)) | (data_matrix.sum(axis=1) == 0)
        
        target_indices = data_matrix.index[is_missing]
        
        if len(target_indices) > 0:
            # 明确打印是否应用了系数
            status_msg = f"系数 {factor}" if multiply_factor else "系数忽略(取结构)"
            print(f"    规则: {target} <- {source} ({status_msg}), 处理 {len(target_indices)} 行缺失数据")

        for idx in target_indices:
            current_row = df.loc[idx]
            country = current_row['M49_Country_Code']
            
            if has_crop:
                crop = current_row['Crop']
                key = (source, country, crop)
            else:
                key = (source, country)
            
            if key in df_lookup.index:
                source_values = df_lookup.loc[key].values
                if source_values.ndim > 1:
                    source_values = source_values[0]

                if multiply_factor:
                    source_values = source_values * factor
                
                df.loc[idx, year_cols] = source_values

    return df

# ================= 主程序 =================

print("开始处理...")

# 1. 读取 dict_v3 并筛选国家
print("步骤 1: 读取字典并筛选有效国家...")
df_region_info = pd.read_excel(base_dict_path, sheet_name='region')
df_region_info['M49_Country_Code'] = clean_m49_code(df_region_info['M49_Country_Code'])

valid_mask = (df_region_info['Region_label_new'].astype(str).str.lower() != 'no') & \
             (df_region_info['Region_label_new'].notna())
valid_countries = df_region_info.loc[valid_mask, 'M49_Country_Code'].unique()

print(f"筛选出 {len(valid_countries)} 个有效国家代码。")
region_map_clean = df_region_info[['M49_Country_Code', 'Region_agg2']].drop_duplicates().dropna(subset=['Region_agg2'])

temp_df = pd.read_excel(input_file_path, sheet_name='total_kgDM_per_head', nrows=1)
year_cols = [c for c in temp_df.columns if str(c).startswith('Y')]

# ================= 步骤 2: 处理 total_kgDM_per_head =================
print("步骤 2: 处理 total_kgDM_per_head (总需求)...")
df_total = pd.read_excel(input_file_path, sheet_name='total_kgDM_per_head')
df_total['M49_Country_Code'] = clean_m49_code(df_total['M49_Country_Code'])
df_total = df_total[df_total['M49_Country_Code'].isin(valid_countries)]

# 正交化
all_species = df_total['Species'].unique()
idx_product = pd.MultiIndex.from_product([all_species, valid_countries], names=['Species', 'M49_Country_Code'])
df_total_indexed = df_total.set_index(['Species', 'M49_Country_Code'])
df_total_indexed = df_total_indexed[~df_total_indexed.index.duplicated(keep='first')]
df_total_full = df_total_indexed.reindex(idx_product).reset_index()

# 填充
df_total_filled = fill_hierarchy(df_total_full, year_cols, region_map_clean, level_cols=['Species'])

# 替补规则 (Total 需要乘系数)
proxy_rules_total = {
    'geese_guinea': ('turkeys', 1.0),
    'dairy_buffalo': ('dairy_cattle', 1.0),
    'meat_buffalo': ('meat_cattle', 1.0),
    'llamas': ('horse', 1.0),
    'dairy_sheep': ('dairy_cattle', 0.7),
    'meat_sheep': ('meat_cattle', 0.7),
    'dairy_goat': ('dairy_cattle', 0.7),
    'meat_goat': ('meat_cattle', 0.7)
}
# multiply_factor=True, 这里的 0.7 会生效
df_total_final = apply_proxies(df_total_filled, year_cols, proxy_rules_total, level_col='Species', multiply_factor=True)
df_total_final[year_cols] = df_total_final[year_cols].fillna(0)

# ================= 步骤 3: 处理 kgDM_per_head_crop =================
print("步骤 3: 处理 kgDM_per_head_crop (归一化与填充)...")
df_crop = pd.read_excel(input_file_path, sheet_name='kgDM_per_head_crop')
df_crop['M49_Country_Code'] = clean_m49_code(df_crop['M49_Country_Code'])
df_crop = df_crop[df_crop['M49_Country_Code'].isin(valid_countries)]

all_crops = df_crop['Crop'].unique()
idx_product_crop = pd.MultiIndex.from_product([all_species, valid_countries, all_crops], 
                                              names=['Species', 'M49_Country_Code', 'Crop'])

df_crop_indexed = df_crop.set_index(['Species', 'M49_Country_Code', 'Crop'])
df_crop_indexed = df_crop_indexed[~df_crop_indexed.index.duplicated(keep='first')]
df_crop_full = df_crop_indexed.reindex(idx_product_crop).fillna(0).reset_index()

# 3.1 归一化
group_sums = df_crop_full.groupby(['Species', 'M49_Country_Code'])[year_cols].transform('sum')
df_crop_shares = df_crop_full.copy()
df_crop_shares[year_cols] = df_crop_full[year_cols] / group_sums.replace(0, np.nan)

# 3.2 填充
df_shares_filled_list = []
for crop in all_crops:
    subset = df_crop_shares[df_crop_shares['Crop'] == crop].copy()
    subset_filled = fill_hierarchy(subset, year_cols, region_map_clean, level_cols=['Species'])
    df_shares_filled_list.append(subset_filled)

df_crop_shares_filled = pd.concat(df_shares_filled_list, ignore_index=True)

# 3.3 再次归一化
group_sums_2 = df_crop_shares_filled.groupby(['Species', 'M49_Country_Code'])[year_cols].transform('sum')
df_crop_shares_final = df_crop_shares_filled.copy()
df_crop_shares_final[year_cols] = df_crop_shares_filled[year_cols] / group_sums_2.replace(0, np.nan)

# 3.4 Proxy 填充 (结构替补)
# 关键修改：显式将系数设为 1.0，确保归一化结构不做缩放
proxy_rules_crop = {k: (v[0], 1.0) for k, v in proxy_rules_total.items()}

# multiply_factor=False 其实已经是双重保险了，但改字典更直观
df_crop_shares_final = apply_proxies(df_crop_shares_final, year_cols, proxy_rules_crop, level_col='Species', multiply_factor=False)

# 3.5 Maize 兜底
df_crop_shares_final[year_cols] = df_crop_shares_final[year_cols].fillna(0)
sums_final = df_crop_shares_final.groupby(['Species', 'M49_Country_Code'])[year_cols].transform('sum')
zero_mask = (sums_final.sum(axis=1) == 0)
maize_mask = (df_crop_shares_final['Crop'] == 'Maize (corn)')
df_crop_shares_final.loc[zero_mask & maize_mask, year_cols] = 1.0
df_crop_shares_final[year_cols] = df_crop_shares_final[year_cols].fillna(0)

# ================= 步骤 4: 处理 dm_conversion_coefficients =================
print("步骤 4: 处理 dm_conversion_coefficients...")
df_coeff = pd.read_excel(input_file_path, sheet_name='dm_conversion_coefficients')
df_coeff['M49_Country_Code'] = clean_m49_code(df_coeff['M49_Country_Code'])

idx_product_coeff = pd.MultiIndex.from_product([all_crops, valid_countries], names=['Crop', 'M49_Country_Code'])
df_coeff_indexed = df_coeff.set_index(['Crop', 'M49_Country_Code'])
df_coeff_indexed = df_coeff_indexed[~df_coeff_indexed.index.duplicated(keep='first')]
df_coeff_full = df_coeff_indexed.reindex(idx_product_coeff).reset_index()

df_coeff_filled = fill_hierarchy(df_coeff_full, year_cols, region_map_clean, level_cols=['Crop'])
df_coeff_filled[year_cols] = df_coeff_filled[year_cols].fillna(0)

# ================= 步骤 5: 保存 =================
print(f"步骤 5: 保存结果到 {output_file_path}")
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with pd.ExcelWriter(output_file_path) as writer:
    df_total_final.to_excel(writer, sheet_name='total_kgDM_per_head', index=False)
    df_crop_shares_final.to_excel(writer, sheet_name='kgDM_per_head_crop_shares', index=False)
    df_coeff_filled.to_excel(writer, sheet_name='dm_conversion_coefficients', index=False)

print("全部完成！")