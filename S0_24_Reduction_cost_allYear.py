import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置参数
# ==========================================
INPUT_FILE = '../../input/Price_Cost/Cost/macc_results_raw_AGRICULTURE.csv'
OUTPUT_EXCEL = '../../input/Price_Cost/Cost/MACC_Weighted_Cost_by_Species_AllYears.xlsx'

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
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # --- 0. 检查路径 ---
    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. 读取并预处理 ---
    print(f"正在读取文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("未找到输入文件！")
        
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw.columns = df_raw.columns.str.strip()
    # 重命名关键列
    df_raw.rename(columns={'tech_long': 'Technology', 'q_total': 'Reduction_potential', 'p': 'Unit_Cost'}, inplace=True)
    
    # 预处理 Process 和 Species
    print("正在进行技术分类匹配...")
    df_raw['Process'] = df_raw['Technology'].apply(match_ag_tech)
    df_raw['Species'] = df_raw.apply(get_species, axis=1)
    
    country_col = 'country' if 'country' in df_raw.columns else 'country_code'

    # --- 2. 动态获取所有年份 (关键修改) ---
    if 'year' in df_raw.columns:
        # 获取所有唯一年份并排序
        all_years = sorted(df_raw['year'].dropna().unique().tolist())
        # 将年份转为整数（去除可能的小数点，如 2030.0 -> 2030）
        all_years = [int(y) for y in all_years]
        print(f"检测到文件包含的年份: {all_years}")
    else:
        print("警告: 文件中未找到 'year' 列，将作为单一数据集处理。")
        all_years = ['All_Data']

    # --- 3. 循环处理并写入 Excel ---
    summary_counts_list = [] # 用于存储统计结果
    
    print(f"准备写入 Excel: {OUTPUT_EXCEL}")
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
            
            for target_year in all_years:
                print(f"\n>>> 处理年份: {target_year}")
                
                # A. 筛选数据
                if target_year == 'All_Data':
                    df_year = df_raw.copy()
                else:
                    df_year = df_raw[df_raw['year'] == target_year].copy()
                
                if df_year.empty:
                    print(f"    警告: 该年份无数据，跳过。")
                    continue
                
                # B. 计算加权平均成本 (Sheet: 2020, 2030...)
                def weighted_avg_agg(x):
                    total_pot = x['Reduction_potential'].sum()
                    if total_pot == 0: return 0
                    w_avg_cost = (x['Unit_Cost'] * x['Reduction_potential']).sum() / total_pot
                    return pd.Series({
                        'Weighted_Avg_Unit_Cost_USD_per_tCO2e': w_avg_cost,
                        'Total_Reduction_Potential_MtCO2e': total_pot,
                        'Tech_Count': x['Technology'].nunique()
                    })

                # 按 [国家, Process, Species] 分组聚合
                df_detail = df_year.groupby([country_col, 'Process', 'Species']).apply(weighted_avg_agg).reset_index()
                # 排序
                df_detail = df_detail.sort_values(by=[country_col, 'Process', 'Weighted_Avg_Unit_Cost_USD_per_tCO2e'])
                
                # 写入 Sheet
                sheet_name = str(target_year)
                df_detail.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"    已生成 Sheet: {sheet_name}")

                # C. 统计数据条数
                # 按 [国家, Process] 统计
                count_stats = df_year.groupby([country_col, 'Process']).size().reset_index(name='Data_Rows_Count')
                count_stats['Year'] = target_year
                summary_counts_list.append(count_stats)
            
            # --- 4. 生成汇总统计 Sheet ---
            if summary_counts_list:
                print("\n>>> 正在生成汇总统计表 (Summary_Data_Counts)...")
                df_summary_counts = pd.concat(summary_counts_list, ignore_index=True)
                cols = ['Year', country_col, 'Process', 'Data_Rows_Count']
                df_summary_counts = df_summary_counts[cols]
                df_summary_counts = df_summary_counts.sort_values(by=['Year', country_col, 'Process'])
                
                df_summary_counts.to_excel(writer, sheet_name='Summary_Data_Counts', index=False)
                print(f"    已写入 Sheet: Summary_Data_Counts")
            
            print(f"\n>>> 全部完成！Excel 已保存至: {OUTPUT_EXCEL}")

    except Exception as e:
        print(f"\n❌ 保存失败: {e}")