import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 路径与配置
# ==========================================
# 输入文件
INPUT_FILE = '../../input/Price_Cost/Cost/macc_results_raw_AGRICULTURE.csv'

# 输出文件配置
# CSV 总表路径
OUTPUT_CSV = '../../input/Price_Cost/Cost/MACC_Final_Data.csv'
# 图片保存目录 (代码会自动提取文件夹路径)
OUTPUT_IMG_PATH_TEMPLATE = '../../input/Price_Cost/Cost/PNG/MACC_Agriculture_Final.png'

# 【关键】目标年份 (必须筛选，否则数据会重复叠加)
TARGET_YEAR = 2030 

# ==========================================
# 2. 匹配字典 (保持不变)
# ==========================================
ag_mapping_rules = {
    'enteric': 'Enteric fermentation',
    '3nop': 'Enteric fermentation',
    'asparagopsis': 'Enteric fermentation',
    'red algae': 'Enteric fermentation',
    'propionate': 'Enteric fermentation',
    'antimethanogen': 'Enteric fermentation',
    'antibiotics': 'Enteric fermentation',
    'bst': 'Enteric fermentation', 
    'feed conversion': 'Enteric fermentation',
    'grazing': 'Enteric fermentation', 
    'masks': 'Enteric fermentation',
    'vaccine': 'Enteric fermentation',
    'breeding': 'Enteric fermentation',
    'lipid': 'Enteric fermentation',
    'digester': 'Manure management', 
    'lagoon': 'Manure management',
    'dairy production rng': 'Manure management',
    'swine production rng': 'Manure management',
    'manure management': 'Manure management',
    'biodigester': 'Manure management',
    'gas': 'Manure management', 
    'flare': 'Manure management',
    'separator': 'Manure management',
    'compost': 'Manure management',
    'acidification': 'Manure management',
    'rice': 'Rice cultivation',
    'midseason drainage': 'Rice cultivation',
    'wetting': 'Rice cultivation', 
    'flooding': 'Rice cultivation',
    'dry-seeding': 'Rice cultivation',
    'sulfate': 'Rice cultivation',
    'hybrid': 'Rice cultivation',
    'fertilizer': 'Synthetic fertilizers', 
    'nitrification': 'Synthetic fertilizers',
    'urea': 'Synthetic fertilizers',
    'auto fertilization': 'Synthetic fertilizers',
    'synthetic': 'Synthetic fertilizers',
    'residue': 'Crop residues', 
    'tillage': 'Crop residues',
    'no till': 'Crop residues', 
    'till': 'Crop residues',
    'cover crop': 'Crop residues', 
    'cropping': 'Crop residues',
    'legume': 'Crop residues', 
    'biochar': 'Crop residues', 
    'application': 'Manure applied to soils',
    'spread': 'Manure applied to soils',
    'injection': 'Manure applied to soils',
    'drained': 'Drained organic soils',
    'organic soil': 'Drained organic soils',
    'water table': 'Drained organic soils',
    'burning': 'Burning crop residues',
    'reforestation': 'De/Reforestation_crop', 
    'afforestation': 'De/Reforestation_crop',
    'forest': 'De/Reforestation_crop',
    'silvopasture': 'De/Reforestation_pasture'
}

def match_ag_tech(tech_name):
    tech_lower = str(tech_name).lower()
    if 'burning' in tech_lower:
        return 'Burning crop residues'
    for keyword, process in ag_mapping_rules.items():
        if keyword in tech_lower:
            return process
    if 'manure' in tech_lower:
        return 'Manure management'
    return 'Other'

# ==========================================
# 3. 主程序逻辑
# ==========================================
if __name__ == "__main__":
    # --- 0. 路径准备 ---
    # 提取 PNG 文件夹路径: '../../input/Price_Cost/Cost/PNG'
    img_dir = os.path.dirname(OUTPUT_IMG_PATH_TEMPLATE)
    # 提取文件名模板前缀: 'MACC_Agriculture_Final'
    img_name_prefix = os.path.splitext(os.path.basename(OUTPUT_IMG_PATH_TEMPLATE))[0]
    
    if not os.path.exists(img_dir):
        print(f"创建图片文件夹: {img_dir}")
        os.makedirs(img_dir)
        
    # --- 1. 读取文件 ---
    print(f"正在读取文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"未找到输入文件: {INPUT_FILE}")
        
    df_macc = pd.read_csv(INPUT_FILE)
    
    # --- 2. 清洗与重命名 ---
    df_macc.columns = df_macc.columns.str.strip()
    rename_dict = {
        'tech_long': 'Technology',
        'q_total': 'Reduction_potential',
        'p': 'Unit_Cost'
    }
    df_macc.rename(columns=rename_dict, inplace=True)
    
    # --- 3. 筛选年份 ---
    if 'year' in df_macc.columns:
        print(f"正在筛选年份: {TARGET_YEAR}")
        df_filtered = df_macc[df_macc['year'] == TARGET_YEAR].copy()
    else:
        print("警告：未找到 'year' 列，将使用所有数据！")
        df_filtered = df_macc.copy()
        
    # --- 4. 匹配 Process ---
    df_filtered['Process'] = df_filtered['Technology'].apply(match_ag_tech)
    
    # 确定国家列名
    country_col = 'country' if 'country' in df_filtered.columns else 'country_code'
    print(f"检测到国家列: {country_col}")

    # --- 5. 生成 CSV 总表 ---
    # 聚合：按 [国家, Process, Technology]
    macc_all = df_filtered.groupby([country_col, 'Process', 'Technology']).agg({
        'Reduction_potential': 'sum',
        'Unit_Cost': 'mean'
    }).reset_index()
    
    # 全局排序（为了CSV好看，先按国家排，再按成本排）
    macc_all = macc_all.sort_values(by=[country_col, 'Unit_Cost'])
    
    # 计算 CSV 里的累积减排量（每个国家内部累积）
    macc_all['Cumulative_Reduction'] = macc_all.groupby(country_col)['Reduction_potential'].cumsum()
    
    print(f"正在保存分国家数据总表至: {OUTPUT_CSV}")
    macc_all.to_csv(OUTPUT_CSV, index=False)

    # ==========================================
    # 6. 批量绘图 (核心修改部分)
    # ==========================================
    # 获取所有国家列表
    unique_countries = macc_all[country_col].unique()
    print(f"共检测到 {len(unique_countries)} 个国家，开始批量绘图...")
    
    # 统一颜色映射 (确保所有国家的颜色一致)
    all_processes = macc_all['Process'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_processes)))
    global_color_map = dict(zip(all_processes, colors))
    
    count = 0
    for country in unique_countries:
        # 提取该国数据
        country_data = macc_all[macc_all[country_col] == country].copy()
        
        # 重新确保按成本排序 (画图必须步骤)
        country_data = country_data.sort_values(by='Unit_Cost')
        
        # 重新计算绘图用的累积量 (Width, Left)
        # 注意：这里只针对当前国家计算
        country_data['Cumulative_Reduction'] = country_data['Reduction_potential'].cumsum()
        country_data['Width'] = country_data['Reduction_potential']
        country_data['Left'] = country_data['Cumulative_Reduction'] - country_data['Width']
        
        # 如果数据量太小（比如减排量几乎为0），跳过不画
        if country_data['Reduction_potential'].sum() < 0.001:
            continue
            
        # --- 绘图 ---
        plt.figure(figsize=(12, 8))
        
        for proc in country_data['Process'].unique():
            subset = country_data[country_data['Process'] == proc]
            plt.bar(
                subset['Left'], 
                subset['Unit_Cost'], 
                width=subset['Width'], 
                align='edge', 
                color=global_color_map[proc], 
                edgecolor='black', 
                linewidth=0.3,
                label=proc,
                alpha=0.9
            )
            
        plt.axhline(0, color='black', linewidth=1.2)
        plt.xlabel('Cumulative Emission Reduction (MtCO2e)', fontsize=12, fontweight='bold')
        plt.ylabel('Marginal Abatement Cost ($/tCO2e)', fontsize=12, fontweight='bold')
        plt.title(f'Agriculture MACC: {country} ({TARGET_YEAR})', fontsize=14)
        
        # 图例去重
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Process', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # --- 保存 ---
        # 文件名格式: MACC_Agriculture_Final_China.png
        # 处理一下国家名中的特殊字符 (如 / 或空格)
        safe_country_name = str(country).replace('/', '_').replace(' ', '_')
        file_name = f"{img_name_prefix}_{safe_country_name}.png"
        save_path = os.path.join(img_dir, file_name)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # 关闭图表，释放内存
        
        count += 1
        if count % 10 == 0:
            print(f"已生成 {count} 张图片...")
            
    print(f"\n>>> 全部完成！共生成 {count} 张国家 MACC 图片。")
    print(f"图片保存路径: {img_dir}/")