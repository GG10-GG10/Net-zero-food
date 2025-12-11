import pandas as pd
import numpy as np
import os
import warnings

# 忽略除以0等警告
warnings.filterwarnings('ignore')

# ================= 配置路径 =================
base_dict_path = "../../src/dict_v3.xlsx"
input_file_path = "../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
output_file_path = "../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled.csv"

# 目标 Item 列表
target_items = [
    'Raw milk of buffalo',
    'Raw milk of camel',
    'Raw milk of goats',
    'Raw milk of sheep'
]

# 处理年份
year_cols = [f'Y{year}' for year in range(2000, 2024)]

# ================= 辅助函数 =================

def clean_m49_to_string(series):
    """
    将 M49 列标准化为 3位字符串 (去除单引号，补0)，用于匹配
    """
    return series.astype(str).str.replace("'", "", regex=False).str.strip().str.zfill(3)

def fill_with_means(df, value_cols, group_cols, region_map_df):
    """
    通用填充逻辑：前后年份 -> 区域均值 -> 世界均值
    """
    # 1. 前后年份填充 (行内)
    df[value_cols] = df[value_cols].replace(0, np.nan)
    df[value_cols] = df[value_cols].ffill(axis=1).bfill(axis=1)
    
    # 准备合并区域信息
    region_map_clean = region_map_df[['M49_Country_Code', 'Region_agg2']].drop_duplicates()
    
    # 临时合并用于计算均值
    df = df.merge(region_map_clean, on='M49_Country_Code', how='left')
    
    # 2. 区域均值填充
    for col in value_cols:
        region_means = df.groupby(['Item', 'Region_agg2'])[col].transform('mean')
        df[col] = df[col].fillna(region_means)
        
    # 3. 世界均值填充
    for col in value_cols:
        global_means = df.groupby(['Item'])[col].transform('mean')
        df[col] = df[col].fillna(global_means)
        
    # 移除辅助列
    if 'Region_agg2' in df.columns:
        df = df.drop(columns=['Region_agg2'])
        
    return df

# ================= 主程序 =================

def main():
    print("开始执行 S0_27_dairy_yield_refill.py (无 Select 预筛选版) ...")

    # 1. 读取字典并筛选有效国家
    print("步骤 1: 读取字典并筛选 Region_label_new != 'no' 的国家...")
    if not os.path.exists(base_dict_path):
        print(f"错误: 找不到文件 {base_dict_path}")
        return

    df_region = pd.read_excel(base_dict_path, sheet_name='region')
    
    # 清洗字典里的 M49 用于筛选
    df_region['clean_m49'] = clean_m49_to_string(df_region['M49_Country_Code'])
    
    # 筛选有效国家
    valid_mask = (df_region['Region_label_new'].astype(str).str.lower() != 'no') & (df_region['Region_label_new'].notna())
    valid_countries = df_region.loc[valid_mask, 'clean_m49'].unique()
    
    # 准备区域映射表 (用于填充)
    region_map = df_region.loc[valid_mask, ['clean_m49', 'Region_agg2']].rename(columns={'clean_m49': 'M49_Country_Code'})
    
    print(f"  - 有效国家数量: {len(valid_countries)}")

    # 2. 读取生产数据 (不筛选 Select)
    print("步骤 2: 读取 FAO 生产数据 (保留所有 Select 状态)...")
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到文件 {input_file_path}")
        return

    df_prod = pd.read_csv(input_file_path, encoding='utf-8', low_memory=False)
    
    # 创建标准化的 M49 用于内部逻辑
    df_prod['clean_m49'] = clean_m49_to_string(df_prod['M49_Country_Code'])
    
    # 3. 计算 Yield
    print("步骤 3: 反算 Yield (kg/An) ...")
    
    # 筛选出相关的 Items
    # 这里直接使用原 Item 名，不进行映射
    df_relevant = df_prod[df_prod['Item'].isin(target_items)].copy()
    
    # 仅计算有效国家的数据 (虽然不过滤Select，但通常我们只关心有效区域的计算)
    df_relevant = df_relevant[df_relevant['clean_m49'].isin(valid_countries)]
    
    print(f"  - 相关 Item 的数据行数: {len(df_relevant)}")
    
    # 关键键
    id_vars = ['clean_m49', 'M49_Country_Code', 'Area', 'Item'] 
    
    # 提取 Production (t)
    prod_mask = df_relevant['Element'] == 'Production'
    df_p = df_relevant[prod_mask][id_vars + year_cols].set_index(id_vars)
    df_p = df_p[~df_p.index.duplicated(keep='first')] # 去重
    
    # 提取 Milk Animals (head)
    # 常见 Element 名称: 'Milk Animals' 或 'Animals milk'
    anim_mask = df_relevant['Element'].isin(['Milk Animals', 'Animals milk'])
    df_a = df_relevant[anim_mask][id_vars + year_cols].set_index(id_vars)
    df_a = df_a[~df_a.index.duplicated(keep='first')] # 去重
    
    print(f"  - Production 数据行数: {len(df_p)}")
    print(f"  - Milk Animals 数据行数: {len(df_a)}")
    
    # 对齐数据
    common_indices = df_p.index.intersection(df_a.index)
    
    if len(common_indices) == 0:
        print("  警告: 仍未找到匹配的 Production 和 Milk Animals 数据。")
        return

    df_p = df_p.loc[common_indices]
    df_a = df_a.loc[common_indices]
    
    # 执行计算: (Production * 1000) / Milk Animals
    df_a_values = df_a[year_cols].replace(0, np.nan)
    df_yield_values = (df_p[year_cols] * 1000) / df_a_values
    
    # 清理 Inf
    df_yield_values = df_yield_values.replace([np.inf, -np.inf], np.nan)
    
    # 重构 DataFrame
    df_yield = df_yield_values.reset_index()
    
    # 4. 填充缺失值
    print("步骤 4: 填充缺失数据 (Time -> Region -> World) ...")
    
    # 准备填充 (需要使用 clean_m49)
    df_yield_for_fill = df_yield.copy()
    # rename column for merge compatibility with region_map
    df_yield_for_fill = df_yield_for_fill.rename(columns={'clean_m49': 'M49_Country_Code_Temp'})
    df_yield_for_fill['M49_Country_Code'] = df_yield_for_fill['M49_Country_Code_Temp']
    
    df_filled = fill_with_means(df_yield_for_fill, year_cols, ['Item'], region_map)
    
    # 5. 格式化新行
    print("步骤 5: 格式化新生成的行 ...")
    df_final_yield = df_filled.copy()
    
    # 设置元数据
    df_final_yield['Element'] = 'Yield'
    df_final_yield['Unit'] = 'kg/An'
    df_final_yield['Select'] = 1  # 显式设为 1
    df_final_yield['Note'] = 'FAO Recalculated'
    
    # 恢复 M49_Country_Code (使用原始带引号的列)
    # 因为 reset_index 后 id_vars 里的 'M49_Country_Code' 依然存在
    # 只需要把 fill 函数引入的 Temp 列删掉，并确保使用原始列
    if 'M49_Country_Code_Temp' in df_final_yield.columns:
        # fill_with_means 里的 merge 可能会让 M49 列变得混乱，
        # 我们这里直接用 id_vars 里的 M49_Country_Code_Temp (它其实是 clean 的)，
        # 不，最好的办法是看 id_vars 里的 'M49_Country_Code' (原始 CSV 读进来的) 是否还在。
        # 在步骤 4 之前 df_yield 是 reset_index 来的，包含所有 id_vars。
        # df_filled 是基于 df_yield 修改了 year_cols。
        # 只要没有 drop 掉，它就在。
        pass

    # 此时 df_final_yield 里应该有一列 'M49_Country_Code_Temp' (clean) 和 一列 'M49_Country_Code' (原始)
    # 或者是 fill_with_means 覆盖了。
    # 为了保险，我们重新映射一次 clean -> original
    m49_map = df_prod[['clean_m49', 'M49_Country_Code']].drop_duplicates().set_index('clean_m49')['M49_Country_Code']
    
    # 如果 M49_Country_Code 列变成了 clean 格式 (即没有引号)，我们需要恢复
    # 检查第一行
    sample_m49 = str(df_final_yield['M49_Country_Code'].iloc[0])
    if "'" not in sample_m49 and 'M49_Country_Code_Temp' in df_final_yield.columns:
        # 说明可能被覆盖或混淆了，用 clean m49 映射回原始
        df_final_yield['M49_Country_Code'] = df_final_yield['M49_Country_Code_Temp'].map(m49_map)
    
    # 对齐列顺序
    original_columns = df_prod.columns.tolist()
    if 'clean_m49' in original_columns:
        original_columns.remove('clean_m49')
        
    for col in original_columns:
        if col not in df_final_yield.columns:
            df_final_yield[col] = np.nan
            
    df_final_yield = df_final_yield[original_columns]
    
    print(f"  - 新增 Yield 行数: {len(df_final_yield)}")

    # 6. 合并与保存
    print("步骤 6: 合并并保存结果 ...")
    
    if 'clean_m49' in df_prod.columns:
        df_prod = df_prod.drop(columns=['clean_m49'])
        
    # 将新行添加到原数据末尾
    df_combined = pd.concat([df_prod, df_final_yield], ignore_index=True)
    
    # 排序
    df_combined = df_combined.sort_values(by=['Area', 'Item', 'Element'])
    
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
    df_combined.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"  - 文件已生成: {output_file_path}")
    print("全部完成！")

if __name__ == "__main__":
    main()