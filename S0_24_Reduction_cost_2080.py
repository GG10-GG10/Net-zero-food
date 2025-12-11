import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置参数
# ==========================================
INPUT_FILE = '../../input/Price_Cost/Cost/macc_results_raw_AGRICULTURE.csv'
# 输出文件名改为 2080 优化版
OUTPUT_EXCEL = '../../input/Price_Cost/Cost/MACC_2080_Optimized_Analysis.xlsx'

# 【核心】2080 缺失数据填补优先级列表
PRIORITY_ORDER = [
    2080, 
    2085, 2075, 
    2090, 2070, 
    2095, 2065, 
    2060, 2055, 2050, 2045, 2040, 2035, 2030
]

# ==========================================
# 2. 定义分类与匹配逻辑 (保持不变)
# ==========================================
ag_mapping_rules = {
    'enteric': 'Enteric fermentation', '3nop': 'Enteric fermentation', 'asparagopsis': 'Enteric fermentation',
    'red algae': 'Enteric fermentation', 'propionate': 'Enteric fermentation', 'antimethanogen': 'Enteric fermentation',
    'antibiotics': 'Enteric fermentation', 'bst': 'Enteric fermentation', 'feed conversion': 'Enteric fermentation',
    'grazing': 'Enteric fermentation', 'masks': 'Enteric fermentation', 'vaccine': 'Enteric fermentation',
    'breeding': 'Enteric fermentation', 'lipid': 'Enteric fermentation',
    'digester': 'Manure management', 'lagoon': 'Manure management', 'dairy production rng': 'Manure management',
    'swine production rng': 'Manure management', 'manure management': 'Manure management', 'biodigester': 'Manure management',
    'gas': 'Manure management', 'flare': 'Manure management', 'separator': 'Manure management', 'compost': 'Manure management',
    'acidification': 'Manure management',
    'rice': 'Rice cultivation', 'midseason drainage': 'Rice cultivation', 'wetting': 'Rice cultivation',
    'flooding': 'Rice cultivation', 'dry-seeding': 'Rice cultivation', 'sulfate': 'Rice cultivation', 'hybrid': 'Rice cultivation',
    'fertilizer': 'Synthetic fertilizers', 'nitrification': 'Synthetic fertilizers', 'urea': 'Synthetic fertilizers',
    'auto fertilization': 'Synthetic fertilizers', 'synthetic': 'Synthetic fertilizers',
    'residue': 'Crop residues', 'tillage': 'Crop residues', 'no till': 'Crop residues', 'till': 'Crop residues',
    'cover crop': 'Crop residues', 'cropping': 'Crop residues', 'legume': 'Crop residues', 'biochar': 'Crop residues',
    'application': 'Manure applied to soils', 'spread': 'Manure applied to soils', 'injection': 'Manure applied to soils',
    'drained': 'Drained organic soils', 'organic soil': 'Drained organic soils', 'water table': 'Drained organic soils',
    'burning': 'Burning crop residues',
    'reforestation': 'De/Reforestation_crop', 'afforestation': 'De/Reforestation_crop', 'forest': 'De/Reforestation_crop',
    'silvopasture': 'De/Reforestation_pasture'
}

def match_ag_tech(tech_name):
    tech_lower = str(tech_name).lower()
    if 'burning' in tech_lower: return 'Burning crop residues'
    for keyword, process in ag_mapping_rules.items():
        if keyword in tech_lower: return process
    if 'manure' in tech_lower: return 'Manure management'
    return 'Other'

def get_species(row):
    tech = str(row['Technology']).lower()
    process = row['Process']
    if process == 'Rice cultivation' or 'rice' in tech: return 'Rice'
    if process in ['Enteric fermentation', 'Manure management', 'Manure applied to soils']:
        if 'dairy' in tech: return 'Dairy Cattle'
        if 'swine' in tech or 'pig' in tech or 'hog' in tech: return 'Swine'
        if 'beef' in tech or 'meat' in tech or 'steer' in tech: return 'Non-Dairy Cattle'
        if 'poultry' in tech or 'chicken' in tech or 'broiler' in tech: return 'Poultry'
        if 'sheep' in tech or 'goat' in tech: return 'Sheep/Goats'
        return 'Other Livestock/General'
    if process in ['Synthetic fertilizers', 'Crop residues', 'Burning crop residues', 'Drained organic soils']:
        return 'Crops (General)'
    if 'forest' in process.lower(): return 'Forest'
    return 'Other'

# ==========================================
# 3. 核心功能：智能填补数据
# ==========================================
def fill_data_with_priority(df, priority_list):
    """
    按 Country-Process 分组，并在每一组内按优先级列表寻找最佳年份的数据。
    """
    # 确定国家列名
    country_col = 'country' if 'country' in df.columns else 'country_code'
    
    # 结果容器
    filled_frames = []
    
    # 按 [国家, Process] 分组
    # (如果需要更细粒度，比如某个特定物种缺失也要补，可以加上 Species，
    # 但通常 MACC 数据缺失是整个 Process 级别的)
    groups = df.groupby([country_col, 'Process'])
    
    print(f"开始扫描 {len(groups)} 个 Country-Process 组合...")
    
    for (country, process), group in groups:
        # 获取该组现有的所有年份
        available_years = set(group['year'].unique())
        
        selected_year = None
        
        # 遍历优先级列表，找到第一个存在的年份
        for y in priority_list:
            if y in available_years:
                selected_year = y
                break
        
        if selected_year is not None:
            # 提取该年份的数据
            subset = group[group['year'] == selected_year].copy()
            # 标记来源年份，方便后续追踪
            subset['Year_Used'] = selected_year
            filled_frames.append(subset)
        else:
            # 极端情况：该组合在所有优先级年份里都没数据
            # print(f"警告: {country} - {process} 在所有指定年份均无数据！")
            pass
            
    if not filled_frames:
        return pd.DataFrame()
        
    return pd.concat(filled_frames, ignore_index=True)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 0. 检查路径
    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取数据
    print(f"正在读取文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("未找到输入文件！")
        
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw.rename(columns={'tech_long': 'Technology', 'q_total': 'Reduction_potential', 'p': 'Unit_Cost'}, inplace=True)
    
    # 2. 预处理
    df_raw['Process'] = df_raw['Technology'].apply(match_ag_tech)
    df_raw['Species'] = df_raw.apply(get_species, axis=1)
    
    # 确保 year 列是整数
    if 'year' in df_raw.columns:
        df_raw['year'] = df_raw['year'].fillna(0).astype(int)
    
    # 3. 执行智能填补 (构建 2080 优化版数据集)
    print(f"\n>>> 正在构建 2080 优化数据集 (优先级填补)...")
    df_2080_optimized = fill_data_with_priority(df_raw, PRIORITY_ORDER)
    
    if df_2080_optimized.empty:
        print("错误：无法生成任何数据，请检查年份列。")
    else:
        # 4. 计算加权平均成本
        country_col = 'country' if 'country' in df_raw.columns else 'country_code'
        
        def weighted_avg_agg(x):
            total_pot = x['Reduction_potential'].sum()
            if total_pot == 0: return 0
            w_avg_cost = (x['Unit_Cost'] * x['Reduction_potential']).sum() / total_pot
            return pd.Series({
                'Weighted_Avg_Unit_Cost_USD_per_tCO2e': w_avg_cost,
                'Total_Reduction_Potential_MtCO2e': total_pot,
                'Tech_Count': x['Technology'].nunique(),
                'Year_Source': int(x['Year_Used'].mode()[0]) # 记录该组主要使用哪一年的数据
            })

        # 分组聚合
        df_result = df_2080_optimized.groupby([country_col, 'Process', 'Species']).apply(weighted_avg_agg).reset_index()
        df_result = df_result.sort_values(by=[country_col, 'Process'])

        # 5. 保存结果
        print(f"\n>>> 正在保存 Excel: {OUTPUT_EXCEL}")
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
                # Sheet 1: 最终计算结果
                df_result.to_excel(writer, sheet_name='2080_Optimized', index=False)
                print("    已写入 Sheet: 2080_Optimized")
                
                # Sheet 2: 来源追踪 (统计每个 Country-Process 到底用了哪一年)
                # 这张表非常有价值，可以让你一眼看出哪些国家缺数据
                source_stats = df_2080_optimized.groupby([country_col, 'Process', 'Year_Used']).size().reset_index(name='Tech_Rows')
                source_stats = source_stats.sort_values(by=[country_col, 'Process'])
                source_stats.to_excel(writer, sheet_name='Source_Tracking', index=False)
                print("    已写入 Sheet: Source_Tracking (来源年份追踪)")
                
            print(f"\n>>> 全部完成！请查看生成的文件。")
            
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")