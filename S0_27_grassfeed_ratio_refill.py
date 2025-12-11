import pandas as pd
import numpy as np
import os

# ================= 配置路径 =================
base_dict_path = "../../src/dict_v3.xlsx"
input_file_path = "../../input/Land/Feed_pasture/Grass_feed_ratio_by_country_livestock.xlsx"
output_file_path = "../../input/Land/Feed_pasture/Grass_feed_ratio_by_country_livestock_refilled.xlsx"

# ================= 辅助函数定义 =================

def clean_m49_code(series):
    """清洗 M49 代码：去除可能存在的单引号，并补全为3位字符串"""
    # 移除单引号，转换为字符串，去除空格，补全前导零
    s = series.astype(str).str.replace("'", "", regex=False).str.strip()
    return s.str.zfill(3)

def fill_crop_with_means(df, region_map_df):
    """
    针对 Species-Country 组合，使用 区域均值 -> 世界均值 填充 Crop 列
    """
    # 1. 准备区域映射数据
    region_map_clean = region_map_df[['M49_Country_Code', 'Region_agg2']].drop_duplicates().dropna(subset=['Region_agg2'])
    
    # 合并 Region_agg2 到主表
    df = df.merge(region_map_clean, on='M49_Country_Code', how='left')
    
    # 2. 计算均值 (仅基于现有数据)
    # 区域均值: 按 Species 和 Region_agg2 分组
    region_means = df.groupby(['Species', 'Region_agg2'])['Crop'].transform('mean')
    
    # 世界均值: 按 Species 分组
    global_means = df.groupby(['Species'])['Crop'].transform('mean')
    
    # 3. 执行填充
    # 记录填充前的缺失情况
    missing_before = df['Crop'].isna().sum()
    
    # 第一步：用区域均值填充
    df['Crop'] = df['Crop'].fillna(region_means)
    missing_after_region = df['Crop'].isna().sum()
    print(f"  - 使用区域均值填充了 {missing_before - missing_after_region} 条缺失数据")
    
    # 第二步：用世界均值填充剩余的
    df['Crop'] = df['Crop'].fillna(global_means)
    missing_after_world = df['Crop'].isna().sum()
    print(f"  - 使用世界均值填充了 {missing_after_region - missing_after_world} 条缺失数据")
    
    # 移除辅助列
    df = df.drop(columns=['Region_agg2'])
    
    return df

# ================= 主程序 =================

def main():
    print("开始执行 S0_27_grassfeed_ratio_refill.py ...")

    # ---------------------------------------------------------
    # 1. 读取字典并获取有效国家列表
    # ---------------------------------------------------------
    print(f"步骤 1: 读取字典文件 {base_dict_path} ...")
    if not os.path.exists(base_dict_path):
        print(f"错误: 找不到文件 {base_dict_path}")
        return

    df_region_info = pd.read_excel(base_dict_path, sheet_name='region')
    df_region_info['M49_Country_Code'] = clean_m49_code(df_region_info['M49_Country_Code'])

    # 筛选有效国家 (Region_label_new != 'no')
    valid_mask = (df_region_info['Region_label_new'].astype(str).str.lower() != 'no') & \
                 (df_region_info['Region_label_new'].notna())
    valid_countries = df_region_info.loc[valid_mask, 'M49_Country_Code'].unique()
    
    print(f"  - 筛选出 {len(valid_countries)} 个有效国家代码")

    # ---------------------------------------------------------
    # 2. 读取输入文件 (Grass feed ratio)
    # ---------------------------------------------------------
    print(f"步骤 2: 读取输入文件 {input_file_path} ...")
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到输入文件 {input_file_path}")
        return

    # 读取 country_level_weighted sheet
    df_input = pd.read_excel(input_file_path, sheet_name='country_level_weighted')
    df_input['M49_Country_Code'] = clean_m49_code(df_input['M49_Country_Code'])
    
    # 确保只保留有效国家 (可选，根据你的逻辑是否要剔除无效国家)
    df_input = df_input[df_input['M49_Country_Code'].isin(valid_countries)]

    # ---------------------------------------------------------
    # 3. 建立正交组合 (Species x Valid Countries)
    # ---------------------------------------------------------
    print("步骤 3: 建立 Species 与 所有有效国家的正交组合...")
    
    all_species = df_input['Species'].dropna().unique()
    print(f"  - 检测到 {len(all_species)} 种物种: {all_species}")

    # 创建 MultiIndex (Species x Country)
    idx_product = pd.MultiIndex.from_product([all_species, valid_countries], names=['Species', 'M49_Country_Code'])
    
    # 设置索引，去重（防止原数据有重复行），然后 Reindex
    # 注意：这里我们主要关心 'Crop' 列，其他列如果不需要可以不保留，或者保留后会被置为 NaN
    # 根据要求，我们最后要生成文件，所以保留原数据的所有列比较好
    df_indexed = df_input.set_index(['Species', 'M49_Country_Code'])
    
    # 去重：如果原始数据里同一个国家同一个物种有多行，保留第一行
    df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')]
    
    # Reindex: 这会引入那些 字典里有 但 输入文件里没有 的国家，其 Crop 列将为 NaN
    df_full = df_indexed.reindex(idx_product).reset_index()
    
    print(f"  - 扩展后的行数: {len(df_full)} (应为 {len(all_species)} * {len(valid_countries)})")

    # ---------------------------------------------------------
    # 4. 填充 Crop 缺失值
    # ---------------------------------------------------------
    print("步骤 4: 填充 Crop 列缺失值 (区域均值 -> 世界均值)...")
    
    # 调用填充函数
    df_filled = fill_crop_with_means(df_full, df_region_info)
    
    # 检查是否还有 NaN (理论上如果某物种全世界都没数据，这里还会是 NaN)
    final_nan_count = df_filled['Crop'].isna().sum()
    if final_nan_count > 0:
        print(f"  - 警告: 仍有 {final_nan_count} 行 Crop 数据无法填充 (可能该物种无任何基础数据)，将填充为 0")
        df_filled['Crop'] = df_filled['Crop'].fillna(0)

    # ---------------------------------------------------------
    # 5. 计算 Grass 列
    # ---------------------------------------------------------
    print("步骤 5: 计算 Grass 列 (Grass = 1 - Crop)...")
    
    df_filled['Grass'] = 1.0 - df_filled['Crop']
    
    # 简单校验：防止精度问题导致略微小于0或大于1，clip一下比较安全，或者保持原样
    # df_filled['Grass'] = df_filled['Grass'].clip(0, 1) # 可选

    # ---------------------------------------------------------
    # 6. 保存结果
    # ---------------------------------------------------------
    print(f"步骤 6: 保存结果到 {output_file_path} ...")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with pd.ExcelWriter(output_file_path) as writer:
        df_filled.to_excel(writer, sheet_name='country_level_weighted', index=False)
        
    print("处理完成！")

if __name__ == "__main__":
    main()