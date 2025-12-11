# -*- coding: utf-8 -*-
"""
S4_1_results.py – 排放汇总与导出模块
=====================================
生成三层汇总表格，严格遵循dict_v3规范：
- Sheet1: 详细汇总（M49_Country_Code, Region_label_new, Emis Process, Emis Item, Emis GHG, Yxxxx列）
- Sheet2: Process+GHG汇总（M49_Country_Code, Region_label_new, Emis Process, Emis GHG, Yxxxx列）
- Sheet3: GHG汇总（M49_Country_Code, Region_label_new, Emis GHG, Yxxxx列）

支持所有排放类型：GCE (作物), GLE (畜牧业), GOS (土壤), GFE (渔业), LUC (土地利用变化)

年份处理策略：
- 历史时期（<=2020）: 只汇总2010-2020年的数据
- 未来时期（>2020）: 汇总模型设置的未来年份
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# AR6 GWP100
GWP100_AR6 = {'CO2': 1.0, 'CH4': 27.2, 'N2O': 273.0}

def _get_column_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return a Series for the named column even if pandas exposes it as a DataFrame because of duplicate labels.
    """
    if column not in df.columns:
        return pd.Series([], dtype=object)
    column_data = df[column]
    if isinstance(column_data, pd.DataFrame):
        if column_data.shape[1] == 0:
            return pd.Series([], dtype=object)
        column_data = column_data.iloc[:, 0]
    if isinstance(column_data, pd.Series):
        return column_data
    return pd.Series(column_data, index=df.index if hasattr(df, 'index') else None)


def _unique_column_values(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Safely get the unique values for a column, even when duplicate labels force pandas to return a DataFrame.
    """
    column_series = _get_column_series(df, column)
    return column_series.dropna().unique()


def filter_years_for_aggregation(all_years: List[int]) -> List[int]:
    """
    根据年份类型筛选用于汇总的年份
    
    规则：
    - 历史时期（<=2020）: 只汇总2010-2020年
    - 未来时期（>2020）: 汇总模型设置的所有未来年份
    
    Args:
        all_years: 所有包含的年份列表
        
    Returns:
        需要汇总的年份列表
    """
    if not all_years:
        return []
    
    hist_years = [y for y in all_years if y <= 2020]
    future_years = [y for y in all_years if y > 2020]
    
    # 历史年份只保留2010-2020
    hist_filtered = [y for y in hist_years if 2010 <= y <= 2020]
    
    # 未来年份全部保留
    return sorted(hist_filtered + future_years)


class EmissionsAggregator:
    """排放数据汇总器，负责从各模块结果生成标准化汇总表"""
    
    def __init__(self, dict_v3_path: str):
        """
        初始化汇总器
        
        Args:
            dict_v3_path: dict_v3.xlsx文件路径
        """
        self.dict_v3_path = dict_v3_path
        self._load_dict_v3()
        
    def _load_dict_v3(self):
        """加载dict_v3中的映射表"""
        # 加载region表
        self.region_df = pd.read_excel(self.dict_v3_path, sheet_name='region')
        
        # 过滤有效国家（Region_label_new != 'no'）
        valid_regions = self.region_df[self.region_df['Region_label_new'] != 'no'].copy()
        print(f"[INFO] dict_v3 中总共 {len(self.region_df)} 行，有效国家 {len(valid_regions)} 个")
        
        # 构建M49_Country_Code到Region_label_new的映射（规范化M49代码为'xxx格式）
        self.m49_to_region = {}
        for _, row in valid_regions.iterrows():
            m49_raw = str(row['M49_Country_Code']).strip("'").strip()
            # 规范化：'xxx格式（单引号+3位数字）
            try:
                m49 = f"'{int(float(m49_raw)):03d}"
            except Exception:
                m49 = f"'{m49_raw}" if not m49_raw.startswith("'") else m49_raw
            region = row['Region_label_new']
            if pd.notna(m49) and pd.notna(region):
                self.m49_to_region[m49] = region
        
        print(f"[INFO] 构建了 {len(self.m49_to_region)} 个国家的 Region 映射（仅有效国家）")
        
        # 加载Emis_item表
        self.emis_item_df = pd.read_excel(self.dict_v3_path, sheet_name='Emis_item')
        
        # 构建Process -> GHG映射（一个Process可能有多个GHG）
        self.process_ghg_map = {}
        for _, row in self.emis_item_df.iterrows():
            process = str(row['Process']).strip()
            ghg = str(row['GHG']).upper() if pd.notna(row['GHG']) else None
            
            if process not in self.process_ghg_map:
                self.process_ghg_map[process] = set()
            if ghg and ghg in ['CH4', 'N2O', 'CO2']:
                self.process_ghg_map[process].add(ghg)
        
        # 构建Process -> Item_Emis映射
        self.process_item_map = {}
        for _, row in self.emis_item_df.iterrows():
            process = str(row['Process']).strip()
            item_emis = row['Item_Emis']
            if pd.notna(item_emis):
                if process not in self.process_item_map:
                    self.process_item_map[process] = []
                if item_emis not in self.process_item_map[process]:
                    self.process_item_map[process].append(item_emis)
    
    def _normalize_emissions_data(self, 
                                  fao_results: Dict[str, Any],
                                  extra_emis: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        将所有模块的排放数据标准化为统一格式
        
        Returns:
            DataFrame with columns: M49_Country_Code, Item, Process, year, CH4_kt, N2O_kt, CO2_kt
        """
        all_records = []
        
        # 处理GCE (作物排放)
        gce = fao_results.get('GCE', {})
        if isinstance(gce, dict):
            for process_key, df in gce.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    records = self._extract_emissions(df, f'GCE:{process_key}')
                    all_records.extend(records)
        
        # 处理GLE (畜牧业排放)
        gle = fao_results.get('GLE', [])
        if isinstance(gle, list):
            for item in gle:
                if isinstance(item, pd.DataFrame) and not item.empty:
                    records = self._extract_emissions(item, 'GLE')
                    all_records.extend(records)
                elif isinstance(item, dict):
                    for process_key, df in item.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            records = self._extract_emissions(df, f'GLE:{process_key}')
                            all_records.extend(records)
        
        # 处理GOS (土壤排放)
        gos = fao_results.get('GOS', [])
        if isinstance(gos, list):
            for item in gos:
                if isinstance(item, dict):
                    for process_key, df in item.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            records = self._extract_emissions(df, f'GOS:{process_key}')
                            all_records.extend(records)
        
        # 处理GFE (渔业排放)
        gfe = fao_results.get('GFE', {})
        if isinstance(gfe, dict):
            for process_key, df in gfe.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    records = self._extract_emissions(df, f'GFE:{process_key}')
                    all_records.extend(records)
        
        # 处理LUC (土地利用变化)
        luc = fao_results.get('LUC')
        if isinstance(luc, pd.DataFrame) and not luc.empty:
            records = self._extract_emissions(luc, 'LUC')
            all_records.extend(records)
        
        # 处理额外排放数据
        if extra_emis is not None and not extra_emis.empty:
            records = self._extract_emissions(extra_emis, None)
            all_records.extend(records)
        
        if not all_records:
            return pd.DataFrame(columns=['M49_Country_Code', 'Item', 'Process', 'year', 
                                        'CH4_kt', 'N2O_kt', 'CO2_kt'])

        df = pd.DataFrame(all_records)

        # Defensive de-duplication: exact duplicate rows (same country,item,process,year,ghg values)
        try:
            dup_subset = ['M49_Country_Code', 'Item', 'Process', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
            if set(dup_subset).issubset(df.columns):
                dup_mask = df.duplicated(subset=dup_subset, keep='first')
                n_dup = int(dup_mask.sum())
                if n_dup > 0:
                    # write a small debug CSV to workspace/debug_outputs for inspection
                    try:
                        dbg_dir = Path(os.path.join(os.getcwd(), 'debug_outputs'))
                        dbg_dir.mkdir(parents=True, exist_ok=True)
                        df[dup_mask].to_csv(dbg_dir / 'duplicate_emission_rows_raw.csv', index=False)
                    except Exception:
                        pass
                    print(f"[WARN] Detected {n_dup} exact duplicate emission rows; dropping duplicates before aggregation")
                    df = df[~dup_mask].copy()
        except Exception as e:
            print(f"[ERROR] De-duplication check failed: {e}")

        return df
    
    def _extract_emissions(self, df: pd.DataFrame, default_process: Optional[str]) -> List[Dict]:
        """
        从单个DataFrame提取排放数据
        
        Returns:
            List of dicts with keys: M49_Country_Code, Item, Process, year, CH4_kt, N2O_kt, CO2_kt
        """
        records = []
        
        # 识别列名（容错处理大小写和不同命名）
        col_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # 国家代码
            if 'm49' in col_lower or 'country_code' in col_lower:
                col_mapping['m49'] = col
            elif 'country' in col_lower and 'code' not in col_lower:
                col_mapping['country'] = col
            
            # Item/Commodity
            if 'item' in col_lower or 'commodity' in col_lower:
                col_mapping['item'] = col
            
            # Process
            if 'process' in col_lower:
                col_mapping['process'] = col
            
            # Year
            if col_lower in ['year', 'time']:
                col_mapping['year'] = col
            
            # GHG排放量
            if 'ch4' in col_lower and 'kt' in col_lower:
                col_mapping['ch4'] = col
            elif 'ch4' in col_lower and '_t' not in col_lower:
                col_mapping['ch4'] = col
                
            if 'n2o' in col_lower and 'kt' in col_lower:
                col_mapping['n2o'] = col
            elif 'n2o' in col_lower and '_t' not in col_lower:
                col_mapping['n2o'] = col
                
            if 'co2' in col_lower and 'kt' in col_lower and 'eq' not in col_lower:
                col_mapping['co2'] = col
            elif 'co2' in col_lower and '_t' not in col_lower and 'eq' not in col_lower:
                col_mapping['co2'] = col
        
        # 提取数据
        for _, row in df.iterrows():
            # 获取M49代码
            m49 = None
            if 'm49' in col_mapping:
                m49_raw = str(row[col_mapping['m49']]).strip("'").strip()
                # 规范化：M49统一为'xxx格式（单引号+3位数字），与m49_to_region映射一致
                try:
                    m49_int = int(m49_raw)
                    m49 = f"'{m49_int:03d}"  # ✅ 'xxx格式，如'004
                except ValueError:
                    # 如果不是数字，添加单引号前缀
                    m49 = f"'{m49_raw}" if not m49_raw.startswith("'") else m49_raw
            elif 'country' in col_mapping:
                # 如果只有国家名，需要转换（暂时记录国家名）
                m49 = str(row[col_mapping['country']])
            
            # 获取Item
            item = row.get(col_mapping.get('item'), 'ALL')
            if pd.isna(item):
                item = 'ALL'
            
            # 获取Process
            process = row.get(col_mapping.get('process'), default_process)
            if pd.isna(process) or process is None:
                process = default_process or 'Unknown'
            
            # 清理Process名称（去掉GCE:, GLE:等前缀）
            process_clean = self._clean_process_name(str(process))
            
            # 获取year
            year = row.get(col_mapping.get('year'))
            if pd.isna(year):
                continue
            
            # 获取排放量（kt）
            ch4_kt = self._safe_float(row.get(col_mapping.get('ch4'), 0))
            n2o_kt = self._safe_float(row.get(col_mapping.get('n2o'), 0))
            co2_kt = self._safe_float(row.get(col_mapping.get('co2'), 0))
            
            # 如果三个气体都是0，跳过
            if ch4_kt == 0 and n2o_kt == 0 and co2_kt == 0:
                continue
            
            records.append({
                'M49_Country_Code': m49,
                'Item': str(item),
                'Process': process_clean,
                'year': int(year),
                'CH4_kt': ch4_kt,
                'N2O_kt': n2o_kt,
                'CO2_kt': co2_kt
            })
        
        return records
    
    def _clean_process_name(self, process: str) -> str:
        """清理Process名称，去掉模块前缀，匹配dict_v3中的标准名称"""
        # 去掉GCE:, GLE:等前缀
        if ':' in process:
            process = process.split(':', 1)[1]
        
        process = process.strip()
        
        # 尝试匹配dict_v3中的标准Process名称
        for standard_process in self.emis_item_df['Process'].unique():
            if pd.notna(standard_process):
                standard = str(standard_process).strip()
                # 模糊匹配
                if standard.lower() in process.lower() or process.lower() in standard.lower():
                    return standard
        
        return process
    
    def _safe_float(self, value, default=0.0) -> float:
        """安全转换为float"""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except:
            return default
    
    def generate_detailed_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        生成Sheet1: 最详细的汇总表
        
        Columns: M49_Country_Code, Region_label_new, Emis Process, Emis Item, Emis GHG, Y2000, Y2001, ...
        
        Args:
            emissions_df: 标准化的排放数据
            years: 需要输出的年份列表
        
        Returns:
            详细汇总DataFrame
        """
        if emissions_df.empty:
            return pd.DataFrame()
        
        
        # 规范化M49代码（确保与映射表一致）
        # 规范化：确保M49是3位数字格式，与m49_to_region映射一致
        def _normalize_m49_internal(code):
            """将M49标准化为3位数字字符串（如'004'，用于映射和输出）"""
            if pd.isna(code) or code == '':
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                # 转为整数再转回3位数字字符串，确保与m49_to_region映射一致
                m49_int = int(code_str)
                return f"{m49_int:03d}"  # ✅ 3位数字格式，如'004'
            except ValueError:
                return code_str
        
        emissions_df['M49_Country_Code'] = emissions_df['M49_Country_Code'].apply(_normalize_m49_internal)
        
        # 添加Region信息
        emissions_df['Region_label_new'] = emissions_df['M49_Country_Code'].map(self.m49_to_region)
        
        # 统计未匹配的M49代码（debug）
        unmatched = emissions_df[emissions_df['Region_label_new'].isna()]
        if not unmatched.empty:
            unmatch_countries = unmatched['M49_Country_Code'].unique()
            print(f"[WARN] {len(unmatch_countries)} 个国家未能匹配到Region: {list(unmatch_countries)[:5]}...")
        
        # 将数据从宽格式转换为长格式
        # 每个（国家，Process，Item，GHG）组合一行
        detail_records = []
        
        for (m49, region, process, item), group in emissions_df.groupby(
            ['M49_Country_Code', 'Region_label_new', 'Process', 'Item']):
            
            # 为每种GHG创建一行
            for ghg in ['CH4', 'N2O', 'CO2']:
                ghg_col = f'{ghg}_kt'  # 使用大写，因为combined_df中已经被重命名为大写
                
                # 检查该Process是否应该有这个GHG（根据dict_v3）
                process_ghgs = self.process_ghg_map.get(process, {'CH4', 'N2O', 'CO2'})
                if ghg not in process_ghgs:
                    continue
                
                # 获取该GHG在各年份的排放量
                year_values = {}
                has_data = False
                for _, row in group.iterrows():
                    year = row['year']
                    # ✅ 修复：row是Series对象，需要用loc或get()方法，不能用row.get()
                    if ghg_col in row.index:
                        value = float(row[ghg_col]) if pd.notna(row[ghg_col]) else 0
                    else:
                        value = 0
                    
                    if value != 0:
                        has_data = True
                    year_values[f'Y{int(year)}'] = value
                
                # 如果该GHG在所有年份都是0，跳过
                if not has_data:
                    continue
                
                # 创建记录
                record = {
                    'M49_Country_Code': m49,
                    'Region_label_new': region if pd.notna(region) else 'Unknown',
                    'Emis Process': process,
                    'Emis Item': item,
                    'Emis GHG': ghg
                }
                
                # 添加所有年份的数据
                for year in years:
                    year_col = f'Y{year}'
                    record[year_col] = year_values.get(year_col, 0)
                
                # 计算CO2eq
                gwp = GWP100_AR6[ghg]
                for year in years:
                    year_col = f'Y{year}'
                    co2eq_col = f'Y{year}_CO2eq'
                    record[co2eq_col] = record[year_col] * gwp
                
                detail_records.append(record)
        
        if not detail_records:
            print(f"[WARN] Sheet1 没有生成任何记录")
            return pd.DataFrame()
        
        df = pd.DataFrame(detail_records)
        
        # 统计信息
        n_countries = df['M49_Country_Code'].nunique()
        n_processes = df['Emis Process'].nunique()
        n_items = df['Emis Item'].nunique()
        print(f"[INFO] Sheet1 统计: {len(df)} 行, {n_countries} 个国家, {n_processes} 个 Process, {n_items} 个 Item")
        
        # ✅ 移除dairy/non-dairy merge逻辑
        # 保持dict_v3中定义的原始dairy/non-dairy分拆形式
        print(f"[INFO] 保留dairy/non-dairy原始分拆形式（按dict_v3定义）")
        
        # ✅ 确保M49_Country_Code格式正确（'xxx格式）
        def _normalize_m49_for_export(code):
            """将M49标准化为'xxx格式（带前导单引号）"""
            if pd.isna(code) or code == '' or code is None:
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                # 转换为整数再back to string，确保数字正确
                m49_int = int(code_str)
                return f"'{m49_int}"
            except Exception:
                # 如果不是数字，保持原样并添加引号
                if not code_str.startswith("'"):
                    return f"'{code_str}"
                return code_str
        
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49_for_export)
        
        # 重新排列列，确保M49在首位
        cols = list(df.columns)
        if 'M49_Country_Code' in cols:
            cols.remove('M49_Country_Code')
            cols = ['M49_Country_Code'] + cols
            df = df[cols]
        
        # 排序
        df = df.sort_values(['M49_Country_Code', 'Emis Process', 'Emis Item', 'Emis GHG'])
        
        return df
    
    def generate_process_ghg_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        生成Sheet2: Process+GHG级别汇总
        
        Columns: M49_Country_Code, Region_label_new, Emis Process, Emis GHG, Y2000, Y2001, ...
        
        在Sheet1基础上，汇总掉Item维度
        
        ✅ 排除dairy/non-dairy项以避免在汇总时重复计数
        """
        if emissions_df.empty:
            return pd.DataFrame()
        
        # ✅ 不排除dairy/non-dairy - 让它们正常通过汇总，后续再处理
        
        # 规范化M49代码（同generate_detailed_summary）
        def _normalize_m49_internal(code):
            """将M49标准化为数字字符串"""
            if pd.isna(code) or code == '':
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                m49_int = int(code_str)
                return str(m49_int)
            except ValueError:
                return code_str
        
        emissions_df['M49_Country_Code'] = emissions_df['M49_Country_Code'].apply(_normalize_m49_internal)
        
        emissions_df['Region_label_new'] = emissions_df['M49_Country_Code'].map(self.m49_to_region)
        
        summary_records = []
        
        for (m49, region, process), group in emissions_df.groupby(
            ['M49_Country_Code', 'Region_label_new', 'Process']):
            
            # 为每种GHG创建一行
            for ghg in ['CH4', 'N2O', 'CO2']:
                ghg_col = f'{ghg}_kt'  # 使用大写，因为combined_df中已经被重命名为大写
                
                # 检查该Process是否应该有这个GHG
                process_ghgs = self.process_ghg_map.get(process, {'CH4', 'N2O', 'CO2'})
                if ghg not in process_ghgs:
                    continue
                
                # 汇总各年份的排放量
                year_values = {}
                has_data = False
                for _, row in group.iterrows():
                    year = row['year']
                    # ✅ 修复：row是Series对象，需要用[]，不能用get()
                    if ghg_col in row.index:
                        value = float(row[ghg_col]) if pd.notna(row[ghg_col]) else 0
                    else:
                        value = 0
                    year_col = f'Y{int(year)}'
                    
                    if year_col not in year_values:
                        year_values[year_col] = 0
                    year_values[year_col] += value
                    
                    if value != 0:
                        has_data = True
                
                if not has_data:
                    continue
                
                record = {
                    'M49_Country_Code': m49,
                    'Region_label_new': region if pd.notna(region) else 'Unknown',
                    'Emis Process': process,
                    'Emis GHG': ghg
                }
                
                # 添加年份数据和CO2eq
                gwp = GWP100_AR6[ghg]
                for year in years:
                    year_col = f'Y{year}'
                    record[year_col] = year_values.get(year_col, 0)
                    co2eq_col = f'Y{year}_CO2eq'
                    record[co2eq_col] = record[year_col] * gwp
                
                summary_records.append(record)
        
        if not summary_records:
            print(f"[WARN] Sheet2 没有生成任何记录")
            return pd.DataFrame()
        
        df = pd.DataFrame(summary_records)
        n_countries = df['M49_Country_Code'].nunique()
        n_processes = df['Emis Process'].nunique()
        print(f"[INFO] Sheet2 统计: {len(df)} 行, {n_countries} 个国家, {n_processes} 个 Process")
        
        # ✅ 确保M49_Country_Code格式正确（'xxx格式）
        def _normalize_m49_for_export(code):
            """将M49标准化为'xxx格式（带前导单引号）"""
            if pd.isna(code) or code == '' or code is None:
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                m49_int = int(code_str)
                return f"'{m49_int}"
            except Exception:
                if not code_str.startswith("'"):
                    return f"'{code_str}"
                return code_str
        
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49_for_export)
        
        # 重新排列列，确保M49在首位
        cols = list(df.columns)
        if 'M49_Country_Code' in cols:
            cols.remove('M49_Country_Code')
            cols = ['M49_Country_Code'] + cols
            df = df[cols]
        
        df = df.sort_values(['M49_Country_Code', 'Emis Process', 'Emis GHG'])
        
        return df
    
    def generate_ghg_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        生成Sheet3: GHG级别汇总
        
        Columns: M49_Country_Code, Region_label_new, Emis GHG, Y2000, Y2001, ...
        
        在Sheet2基础上，汇总掉Process维度
        
        ✅ 不排除dairy/non-dairy - 让它们正常通过汇总，后续再处理
        """
        if emissions_df.empty:
            return pd.DataFrame()
        
        # ✅ 不排除dairy/non-dairy - 让它们正常通过汇总，后续再处理
        
        # 规范化M49代码（同generate_detailed_summary）
        def _normalize_m49_internal(code):
            """将M49标准化为数字字符串"""
            if pd.isna(code) or code == '':
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                m49_int = int(code_str)
                return str(m49_int)
            except ValueError:
                return code_str
        
        emissions_df['M49_Country_Code'] = emissions_df['M49_Country_Code'].apply(_normalize_m49_internal)
        
        emissions_df['Region_label_new'] = emissions_df['M49_Country_Code'].map(self.m49_to_region)
        
        summary_records = []
        
        for (m49, region), group in emissions_df.groupby(
            ['M49_Country_Code', 'Region_label_new']):
            
            # 为每种GHG创建一行
            for ghg in ['CH4', 'N2O', 'CO2']:
                ghg_col = f'{ghg}_kt'  # 使用大写，因为combined_df中已经被重命名为大写
                
                # 汇总各年份的排放量
                year_values = {}
                has_data = False
                for _, row in group.iterrows():
                    year = row['year']
                    # ✅ 修复：row是Series对象，需要用[]，不能用get()
                    if ghg_col in row.index:
                        value = float(row[ghg_col]) if pd.notna(row[ghg_col]) else 0
                    else:
                        value = 0
                    year_col = f'Y{int(year)}'
                    
                    if year_col not in year_values:
                        year_values[year_col] = 0
                    year_values[year_col] += value
                    
                    if value != 0:
                        has_data = True
                
                if not has_data:
                    continue
                
                record = {
                    'M49_Country_Code': m49,
                    'Region_label_new': region if pd.notna(region) else 'Unknown',
                    'Emis GHG': ghg
                }
                
                # 添加年份数据和CO2eq
                gwp = GWP100_AR6[ghg]
                for year in years:
                    year_col = f'Y{year}'
                    record[year_col] = year_values.get(year_col, 0)
                    co2eq_col = f'Y{year}_CO2eq'
                    record[co2eq_col] = record[year_col] * gwp
                
                summary_records.append(record)
        
        if not summary_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(summary_records)
        
        # ✅ 确保M49_Country_Code格式正确（'xxx格式）
        def _normalize_m49_for_export(code):
            """将M49标准化为'xxx格式（带前导单引号）"""
            if pd.isna(code) or code == '' or code is None:
                return ''
            code_str = str(code).strip().strip("'\"")
            try:
                m49_int = int(code_str)
                return f"'{m49_int}"
            except Exception:
                if not code_str.startswith("'"):
                    return f"'{code_str}"
                return code_str
        
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49_for_export)
        
        # 重新排列列，确保M49在首位
        cols = list(df.columns)
        if 'M49_Country_Code' in cols:
            cols.remove('M49_Country_Code')
            cols = ['M49_Country_Code'] + cols
            df = df[cols]
        
        df = df.sort_values(['M49_Country_Code', 'Emis GHG'])
        
        return df
    
    def aggregate_and_export(self, 
                            fao_results: Dict[str, Any],
                            output_path: str,
                            years: List[int],
                            extra_emis: Optional[pd.DataFrame] = None):
        """
        完整的聚合和导出流程
        
        Args:
            fao_results: FAO模块运行结果字典
            output_path: 输出Excel文件路径
            years: 需要输出的年份列表（包括历史和未来年份）
            extra_emis: 额外的排放数据（可选）
            
        说明：
            - 历史时期（<=2020）: 自动筛选2010-2020年
            - 未来时期（>2020）: 使用模型设置的所有年份
        """
        print("开始排放数据汇总...")
        
        # 0. 应用年份过滤（历史2010-2020，未来全部）
        filtered_years = filter_years_for_aggregation(years)
        print(f"  - 年份过滤: {len(years)} → {len(filtered_years)} 年")
        print(f"    原始年份范围: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")
        print(f"    筛选后范围: {min(filtered_years) if filtered_years else 'N/A'} - {max(filtered_years) if filtered_years else 'N/A'}")
        
        if not filtered_years:
            print("  WARNING: no years remain after filtering")
            return
        
        # 1. 标准化排放数据
        print("  - 标准化排放数据...")
        emissions_df = self._normalize_emissions_data(fao_results, extra_emis)
        
        if emissions_df.empty:
            print("  WARNING: no emissions data to aggregate")
            return
        
        print(f"  [诊断] 标准化后: {len(emissions_df)} 行, {emissions_df['M49_Country_Code'].nunique()} 个国家")
        if 'year' in emissions_df.columns:
            print(f"       年份范围: {emissions_df['year'].min()} - {emissions_df['year'].max()}")
        
        # 2. 按年份筛选
        print("  - 按年份筛选排放数据...")
        emissions_df_filtered = emissions_df[emissions_df['year'].isin(filtered_years)].copy()
        
        print(f"  [诊断] 年份筛选后: {len(emissions_df_filtered)} 行, {emissions_df_filtered['M49_Country_Code'].nunique()} 个国家")
        print(f"  - 共收集 {len(emissions_df_filtered)} 条排放记录")
        
        # 3. 生成三个层级的汇总表
        print("  - 生成详细汇总表（Sheet1）...")
        sheet1 = self.generate_detailed_summary(emissions_df_filtered, filtered_years)
        
        print("  - 生成Process+GHG汇总表（Sheet2）...")
        sheet2 = self.generate_process_ghg_summary(emissions_df_filtered, filtered_years)
        
        print("  - 生成GHG汇总表（Sheet3）...")
        sheet3 = self.generate_ghg_summary(emissions_df_filtered, filtered_years)
        
        # 4. 导出到Excel
        print(f"  - 导出到文件: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if not sheet1.empty:
                sheet1.to_excel(writer, sheet_name='Detail_Summary', index=False)
            if not sheet2.empty:
                sheet2.to_excel(writer, sheet_name='Process_GHG_Summary', index=False)
            if not sheet3.empty:
                sheet3.to_excel(writer, sheet_name='GHG_Summary', index=False)
        
        print(f"✓ 排放汇总完成！")
        print(f"  - 汇总年份数: {len(filtered_years)} ({min(filtered_years) if filtered_years else 'N/A'}-{max(filtered_years) if filtered_years else 'N/A'})")
        print(f"  - Sheet1 (详细): {len(sheet1)} 行")
        print(f"  - Sheet2 (Process+GHG): {len(sheet2)} 行")
        print(f"  - Sheet3 (GHG): {len(sheet3)} 行")


# 向后兼容的函数（保留原有接口）
def summarize_emissions_from_detail(emission_detail_df: pd.DataFrame,
                                    process_meta_map: Optional[dict]=None,
                                    allowed_years: Optional[List[int]]=None,
                                    dict_v3_path: Optional[str]=None,
                                    production_df: Optional[pd.DataFrame]=None) -> Dict[str, pd.DataFrame]:
    """
    从详细排放数据框生成多个粒度的汇总
    WARNING: rely on Process and Item lists defined in dict_v3 Emis_item
    
    Parameters
    ----------
    emission_detail_df : pd.DataFrame
        详细排放数据框，包含列如: M49_Country_Code, Process, Item, GHG, year, value等
    process_meta_map : dict, optional
        过程元数据映射（用于分类）
    allowed_years : List[int], optional
        允许的年份列表（用于过滤）
    dict_v3_path : str, optional
        dict_v3.xlsx路径，用于加载Region映射和Process/Item定义
    production_df : pd.DataFrame, optional
        生产数据（包含stock），用于拆分历史阶段merged livestock排放
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        多个粒度的汇总结果
    """

    # *** DEBUG ***
    print(f"[DEBUG] summarize_emissions_from_detail called:")
    print(f"  - emission_detail_df length: {len(emission_detail_df) if isinstance(emission_detail_df, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"  - allowed_years: {allowed_years}")
    print(f"  - allowed_years type: {type(allowed_years)}")
    debug_log = os.path.join(tempfile.gettempdir(), "s4_1_debug.log")
    with open(debug_log, "a", encoding="utf-8") as f:
        # Write a single debug line to the temp log
        f.write(f"summarize_emissions_from_detail called: allowed_years={allowed_years}\n")
    # *** END DEBUG ***
    
    if not isinstance(emission_detail_df, pd.DataFrame) or emission_detail_df.empty:
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': emission_detail_df.copy() if isinstance(emission_detail_df, pd.DataFrame) else pd.DataFrame()
        }
    
    df = emission_detail_df.copy()
    
    # Normalize years to integers to avoid filtering mismatches (e.g., strings vs ints)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

    allowed_years_set = {int(y) for y in allowed_years} if allowed_years else None

    # WARNING: load allowed Process/Item lists from dict_v3
    allowed_processes = set()
    allowed_items = set()
    if dict_v3_path:
        try:
            emis_item_df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
            allowed_processes = set(emis_item_df['Process'].dropna().unique())
            allowed_items = set(emis_item_df['Item_Emis'].dropna().unique())
            
            # ✅ 新逻辑：不再添加merged名称，历史排放将按stock比例拆分到dairy/non-dairy
            # dict_v3中已定义dairy/non-dairy分拆项，严格按此汇总
            
            print(f"[INFO] 从dict_v3加载: {len(allowed_processes)} 个Process, {len(allowed_items)} 个Item_Emis (含合并动物名称)")
            print(f"[INFO] 允许的Process: {sorted(allowed_processes)}")
        except Exception as e:
            print(f"WARNING: failed to load dict_v3 Emis_item: {e}")
    
    # WARNING: filter to Processes defined in dict_v3 that actually appear in data
    if allowed_processes and 'Process' in df.columns:
        process_series = _get_column_series(df, 'Process')
        process_values = _unique_column_values(df, 'Process')
        print(f"[DEBUG] Process过滤前: {len(df)} 行")
        print(f"  - 数据中的Process: {process_values}")
        print(f"  - dict_v3中的Process: {sorted(list(allowed_processes))}")
        
        # 检查是否有数据匹配 dict_v3 中的 Process
        process_mask = process_series.isin(allowed_processes)
        print(f"  - 匹配的Process行数: {process_mask.sum()}/{len(df)}")
        
        if process_mask.any():
            unmatched = df[~process_mask]
            matched = df[process_mask]
            if len(unmatched):
                print(f"[INFO] 发现 {len(unmatched)} 行 Process 未在 dict_v3 中定义，保留并继续处理")
                print(f"  - 非dict_v3 Process 样例: {_unique_column_values(unmatched, 'Process')[:5]}")
                df = pd.concat([matched, unmatched], ignore_index=True)
            else:
                rows_before = len(df)
                df = matched.copy()
                rows_after = len(df)
                if rows_before > rows_after:
                    dropped_processes = set(_unique_column_values(emission_detail_df, 'Process')) - allowed_processes
                    print(f"[INFO] 过滤掉 {rows_before - rows_after} 行非dict_v3定义的Process: {dropped_processes}")
        else:
            # 如果没有匹配的 Process，说明数据中的 Process 名称不在 dict_v3 中
            # 此时不过滤，保留所有数据
            print(f"[INFO] 数据中的Process名称与dict_v3不匹配，不进行Process过滤")
            print(f"  - 数据中的Process: {process_values}")
            print(f"  - dict_v3中的Process: {sorted(list(allowed_processes))[:5]}...")
        
        print(f"[DEBUG] Process过滤后: {len(df)} 行")
    
    # WARNING: filter to Items defined in dict_v3 that actually appear in data
    # ✅ 关键修复：只对真正的排放Item进行过滤，跳过商品名（GLE/GCE）
    # 检查是否有_is_commodity_based标记（在summarize_emissions中设置）
    has_commodity_flag = '_is_commodity_based' in df.columns
    if has_commodity_flag:
        # 分离commodity-based行（商品名）和emission-based行（排放Item名）
        commodity_rows = df[df['_is_commodity_based'] == True]
        emission_rows = df[df['_is_commodity_based'] == False]
        print(f"\n[DEBUG] 检测到_is_commodity_based标记:")
        print(f"  - Commodity-based行（商品名，不过滤）: {len(commodity_rows)}")
        print(f"  - Emission-based行（排放Item，需过滤）: {len(emission_rows)}")
        df_to_filter = emission_rows
        df_to_keep = commodity_rows
    else:
        # 没有标记，默认所有行都是排放Item
        df_to_filter = df.copy()
        df_to_keep = pd.DataFrame()
    
    if allowed_items and 'Item' in df_to_filter.columns and not df_to_filter.empty:
        item_series = _get_column_series(df_to_filter, 'Item')
        item_values = _unique_column_values(df_to_filter, 'Item')
        print(f"[DEBUG] Item过滤前（仅emission-based行）: {len(df_to_filter)} 行")
        print(f"  - 数据中的Item: {item_values}")
        print(f"  - dict_v3中的Item_Emis: {sorted(list(allowed_items))[:10]}...")
        
        # 检查是否有数据匹配 dict_v3 中的 Item
        item_mask = item_series.isin(allowed_items)
        print(f"  - 匹配的Item行数: {item_mask.sum()}/{len(df_to_filter)}")
        
        if item_mask.any():
            # 严格过滤：只保留dict_v3中定义的Item
            unmatched_items = df_to_filter[~item_mask]
            if len(unmatched_items):
                dropped_item_names = _unique_column_values(unmatched_items, 'Item')
                print(f"[INFO] 过滤掉 {len(unmatched_items)} 行非dict_v3定义的排放Item")
                print(f"  - 被过滤的排放Item: {dropped_item_names[:10]}")
            df_to_filter = df_to_filter[item_mask].copy()
        else:
            # 如果没有匹配的 Item，说明数据中的 Item 名称不在 dict_v3 中
            # 此时不过滤，保留所有数据（这种情况通常不应该发生）
            print(f"[WARNING] 数据中的Item名称与dict_v3完全不匹配，保留所有数据")
            print(f"  - 数据中的Item: {item_values[:10]}")
            print(f"  - dict_v3中的Item: {sorted(list(allowed_items))[:5]}...")
        
        # 合并过滤后的emission行和未过滤的commodity行
        if not df_to_keep.empty:
            df = pd.concat([df_to_filter, df_to_keep], ignore_index=True)
            print(f"[DEBUG] Item过滤后: {len(df_to_filter)} 排放行 + {len(df_to_keep)} 商品行 = {len(df)} 总行数")
        else:
            df = df_to_filter
            print(f"[DEBUG] Item过滤后: {len(df)} 行")
    elif has_commodity_flag and not df_to_keep.empty:
        # 没有进行Item过滤，但有commodity行需要合并
        df = pd.concat([df_to_filter, df_to_keep], ignore_index=True)
        print(f"\n[DEBUG] 跳过Item过滤（没有allowed_items），保留所有数据: {len(df)} 行")
    
    if df.empty:
        print("WARNING: result after filtering is empty")
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # ✅ 关键修复：确保M49_Country_Code列存在
    if 'M49_Country_Code' not in df.columns:
        if 'M49' in df.columns:
            df = df.rename(columns={'M49': 'M49_Country_Code'})
        else:
            print("WARNING: row missing M49_Country_Code; cannot identify country")
            return {
                'by_ctry_proc_comm': pd.DataFrame(),
                'by_ctry_proc': pd.DataFrame(),
                'by_ctry': pd.DataFrame(),
                'by_year': pd.DataFrame(),
                'long': df
            }
    
    # 加载dict_v3的region映射（M49 -> Region_label_new）
    region_mapping = {}
    valid_countries_only = set()  # 只有有效国家的M49代码
    if dict_v3_path:
        try:
            region_df = pd.read_excel(dict_v3_path, sheet_name='region')
            
            # ✅ 关键修复：规范化M49代码为3位格式并只加载有效国家
            for _, row in region_df.iterrows():
                m49_raw = str(row['M49_Country_Code']).strip().lstrip("'\"")
                # ✅ 补齐3位，与其他地方的标准化逻辑一致（51→'051'）
                m49_normalized = m49_raw.zfill(3) if m49_raw.isdigit() else m49_raw
                region = row['Region_label_new']
                
                # 只加载有效国家（Region_label_new != 'no'）
                if pd.notna(region) and region != 'no':
                    region_mapping[m49_normalized] = region
                    valid_countries_only.add(m49_normalized)
            
            print(f"[DEBUG] 从dict_v3加载了 {len(region_mapping)} 个有效国家的region映射（已规范化M49代码）")
        except Exception as e:
            print(f"WARNING: failed to map dict_v3 regions: {e}")
    
    # 调试信息：检查输入数据结构
    print(f"\n[DEBUG] summarize_emissions_from_detail 输入数据:")
    print(f"  - 形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    if 'year' in df.columns:
        years_in_data = sorted(df['year'].unique())
        print(f"  - 数据中的年份: {years_in_data} (共{len(years_in_data)}个)")
        print(f"  - 年份范围: {df['year'].min()}-{df['year'].max()}")
    print(f"  - 总行数: {len(df)}")
    if 'value' in df.columns:
        print(f"  - value非零行: {(df['value'] > 0).sum()}, 总计: {df['value'].sum():.2f}")
    if 'M49_Country_Code' in df.columns:
        m49_null_count = df['M49_Country_Code'].isna().sum()
        print(f"  - M49_Country_Code空值: {m49_null_count}/{len(df)}")
    if 'Item' in df.columns:
        item_null_count = df['Item'].isna().sum()
        print(f"  - Item空值: {item_null_count}/{len(df)}")
    
    # 1. 确保年份列为整数，再按allowed_years过滤
    year_cols = [col for col in df.columns if col.lower() in ['year', 'yr']]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col].notna()]  # 去掉无法解析年份的行
        df[col] = df[col].astype(int)
    if allowed_years_set:
        if year_cols:
            year_col = year_cols[0]
            years_in_data = sorted(df[year_col].unique())
            removed_years = [y for y in years_in_data if y not in allowed_years_set]
            before_len = len(df)
            df = df[df[year_col].isin(allowed_years_set)].reset_index(drop=True)
            after_len = len(df)
            print(f"[DEBUG] Year filter: {before_len} -> {after_len} rows (allowed_years={sorted(allowed_years_set)})")
            if removed_years:
                print(f"[DEBUG] removed years: {removed_years}")
        else:
            print(f"[DEBUG] year column not found; skip year filtering")
    
    # 1.4 ⚠️ 已禁用：历史dairy/non-dairy拆分逻辑（新输入文件已完成拆分）
    # 原因：输入文件已改为 Emissions_livestock_dairy_split.csv，该文件在源头已完成拆分
    # Buffalo/Camel/Sheep/Goats 的 dairy/non-dairy 排放已经分开，不需要在这里处理
    if False and 'Item' in df.columns and 'year' in df.columns and 'Process' in df.columns and production_df is not None:
        print(f"\n[INFO] 开始拆分历史阶段的merged livestock排放...")
        
        # 需要拆分的merged动物名称（历史排放中的Item名称）
        # ⚠️ 关键：只包含dict_v3中定义的单一动物（Buffalo, Camels, Goats, Sheep）
        # 'Sheep and Goats'等复合名称不在dict_v3中，应在数据源头被过滤
        merged_items_to_split = {
            'Buffalo', 'Buffaloes',  # Buffaloes是Buffalo的复数形式
            'Camels', 
            'Goats', 
            'Sheep'
            # ❌ 不包含'Sheep and Goats' - 不在dict_v3定义中
        }
        
        # 只有livestock相关的过程需要拆分
        livestock_processes = {
            'Enteric fermentation',
            'Manure management',
            'Manure applied to soils',
            'Manure left on pasture'
        }
        
        # ✅ 关键修复：只拆分历史年份（<=2020）的livestock过程的merged数据
        historical_threshold = 2020
        merged_mask = (
            df['Item'].isin(merged_items_to_split) &
            (df['year'] <= historical_threshold) &
            df['Process'].isin(livestock_processes)
        )
        merged_rows = df[merged_mask]
        non_merged_rows = df[~merged_mask]
        
        if len(merged_rows) > 0:
            print(f"  - 发现 {len(merged_rows)} 行merged livestock数据需要拆分")
            print(f"  - 涉及Item: {merged_rows['Item'].unique()}")
            print(f"  - 涉及年份: {sorted(merged_rows['year'].unique())}")
            
            # 从production_df获取stock数据
            # production_df结构: M49_Country_Code, Item, Item_Emis, year, stock, ...
            if 'stock' in production_df.columns and 'Item_Emis' in production_df.columns:
                # ⚠️ 关键修正：不处理'Sheep and Goats'等不在dict_v3中的项
                # 这些项应该在历史数据加载时就被过滤掉，如果出现说明数据源有问题
                # 标准化Item名称映射（仅处理dict_v3中定义的单一动物）
                item_mapping = {
                    'Buffalo': 'Buffalo',
                    'Buffaloes': 'Buffalo',  # 统一为Buffalo
                    'Camels': 'Camels',
                    'Goats': 'Goats',
                    'Sheep': 'Sheep'
                    # ❌ 不包含'Sheep and Goats' - 这个不在dict_v3中，应该被过滤
                }
                
                split_rows = []
                skipped_invalid_items = []
                
                for idx, row in merged_rows.iterrows():
                    m49 = row['M49_Country_Code']
                    item = row['Item']
                    year = row['year']
                    value = row['value']
                    
                    # ✅ 检查是否是有效的单一动物名称
                    if item not in item_mapping:
                        skipped_invalid_items.append(item)
                        continue  # 跳过不在映射中的项（如'Sheep and Goats'）
                    
                    # 统一Item名称
                    std_item = item_mapping[item]
                    
                    # 查找该国该年该动物的dairy和non-dairy stock
                    stock_data = production_df[
                        (production_df['M49_Country_Code'] == m49) &
                        (production_df['Item'] == std_item) &
                        (production_df['year'] == year)
                    ]
                    
                    # 分别获取dairy和non-dairy的stock
                    dairy_stock_rows = stock_data[stock_data['Item_Emis'] == f'{std_item}, dairy']
                    nondairy_stock_rows = stock_data[stock_data['Item_Emis'] == f'{std_item}, non-dairy']
                    
                    dairy_stock = dairy_stock_rows['stock'].sum() if not dairy_stock_rows.empty and 'stock' in dairy_stock_rows.columns else 0
                    nondairy_stock = nondairy_stock_rows['stock'].sum() if not nondairy_stock_rows.empty and 'stock' in nondairy_stock_rows.columns else 0
                    total_stock = dairy_stock + nondairy_stock
                    
                    if total_stock > 0:
                        # 按stock比例拆分排放
                        dairy_ratio = dairy_stock / total_stock
                        nondairy_ratio = nondairy_stock / total_stock
                        
                        # 创建dairy行
                        if dairy_stock > 0:
                            dairy_row = row.copy()
                            dairy_row['Item'] = f'{std_item}, dairy'
                            dairy_row['value'] = value * dairy_ratio
                            split_rows.append(dairy_row)
                        
                        # 创建non-dairy行
                        if nondairy_stock > 0:
                            nondairy_row = row.copy()
                            nondairy_row['Item'] = f'{std_item}, non-dairy'
                            nondairy_row['value'] = value * nondairy_ratio
                            split_rows.append(nondairy_row)
                    else:
                        # 没有stock数据，默认全部分配给non-dairy（肉用）
                        nondairy_row = row.copy()
                        nondairy_row['Item'] = f'{std_item}, non-dairy'
                        split_rows.append(nondairy_row)
                
                # ✅ 报告跳过的无效项
                if skipped_invalid_items:
                    unique_skipped = set(skipped_invalid_items)
                    print(f"  ⚠️ 跳过 {len(skipped_invalid_items)} 行不在dict_v3中的项: {unique_skipped}")
                    print(f"     这些项不应该出现在历史数据中，建议检查数据源过滤逻辑")
                
                if split_rows:
                    split_df = pd.DataFrame(split_rows)
                    # ✅ 删除原始merged项，只保留拆分后的dairy/non-dairy
                    df = pd.concat([non_merged_rows, split_df], ignore_index=True)
                    print(f"  ✅ 拆分完成: {len(merged_rows) - len(skipped_invalid_items)} 行有效merged数据 → {len(split_df)} 行dairy/non-dairy数据")
                    print(f"  ✅ 原始merged项已删除，仅保留dairy/non-dairy拆分结果")
                else:
                    # 如果没有成功拆分，则删除原始merged项（避免double counting）
                    df = non_merged_rows.copy()
                    print(f"  ⚠️ 未能拆分（缺少stock数据），已删除原始merged项以避免重复")
            else:
                # 缺少必要列，删除merged项以避免与未来dairy/non-dairy重复
                df = non_merged_rows.copy()
                print(f"  ⚠️ production_df缺少stock或Item_Emis列，已删除merged项以避免重复")
        else:
            print(f"  - 未发现需要拆分的merged livestock数据")
    
    # 1.4.1 ✅ 强制删除所有不在dict_v3中的merged项（无论历史还是未来）
    # dict_v3只定义了dairy/non-dairy分拆项，任何merged项（Sheep and Goats, Goats, Sheep等）都应被删除
    if 'Item' in df.columns and allowed_items:
        before_filter = len(df)
        df = df[df['Item'].isin(allowed_items)].copy()
        after_filter = len(df)
        if before_filter > after_filter:
            print(f"\n[INFO] ✅ 严格过滤：删除 {before_filter - after_filter} 行不在dict_v3中的Items")
            print(f"  保留的Items全部符合dict_v3定义（dairy/non-dairy分拆形式）")
    
    # 1.5 ✅ 关键：规范化M49代码并添加Region_label_new，然后过滤掉无效国家
    if region_mapping and 'M49_Country_Code' in df.columns:
        # 规范化M49代码：去除引号并补齐3位数字
        def normalize_m49_code(code):
            if pd.isna(code):
                return None
            s = str(code).strip().lstrip("'\"")
            if s.isdigit():
                return s.zfill(3)  # 补齐为3位（51→051）
            return s
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(normalize_m49_code)
        
        # 映射到Region
        df['Region_label_new'] = df['M49_Country_Code'].map(region_mapping)
        
        # 统计未匹配的国家
        unmatched = df[df['Region_label_new'].isna()]['M49_Country_Code'].unique()
        if len(unmatched) > 0:
            print(f"[WARNING] 未匹配的M49代码（未在dict_v3中找到）: {list(unmatched)[:10]}")
        
        # 过滤掉无效国家（Region_label_new为NaN或='no'）
        rows_before = len(df)
        df = df[df['Region_label_new'].notna() & (df['Region_label_new'] != 'no')].reset_index(drop=True)
        rows_after = len(df)
        if rows_before > rows_after:
            print(f"[INFO] 过滤掉 {rows_before - rows_after} 行无效国家数据")
        
        print(f"[INFO] 规范化和过滤后保留 {len(df)} 行，涉及 {df['M49_Country_Code'].nunique()} 个国家")
    
    # 1.9 清理辅助列
    if '_is_commodity_based' in df.columns:
        df = df.drop(columns=['_is_commodity_based'])
    
    # 2. 生成长格式结果（原始数据）
    result_long = df.copy()
    
    # ✅ 关键修复：按用户要求，Long表不包含Country和iso3列（只保留M49_Country_Code）
    # ✅ 删除没用的CH4_kt、N2O_kt、CO2_kt空列（已用value列统计排放值）
    cols_to_drop = [c for c in ['Country', 'iso3', 'country', 'CH4_kt', 'N2O_kt', 'CO2_kt'] if c in result_long.columns]
    if cols_to_drop:
        result_long = result_long.drop(columns=cols_to_drop)
    
    # 确定哪些列可用
    has_m49 = 'M49_Country_Code' in df.columns
    has_year = 'year' in df.columns
    has_process = 'Process' in df.columns
    has_item = 'Item' in df.columns
    has_ghg = 'GHG' in df.columns
    
    # ✅ 关键：确保Process列是字符串类型
    if has_process:
        df['Process'] = df['Process'].astype(str)
    
    # 2.5 ✅ 移除dairy/non-dairy merge逻辑
    # dict_v3已经定义了dairy/non-dairy分拆的Item_Emis，不应在此创建merged项
    # Sheet1直接显示dict_v3中的dairy/non-dairy项
    # Sheet2/3通过正常的groupby自动汇总（不需要特殊排除逻辑）
    print(f"[INFO] 保留dairy/non-dairy原始分拆形式（按dict_v3定义）")
    
    # 3. 生成国家-过程-商品汇总（✅ 只用M49_Country_Code，不用Country）
    groupby_cols_cpc = []
    if has_m49:
        groupby_cols_cpc.append('M49_Country_Code')
    if has_process:
        groupby_cols_cpc.append('Process')
    if has_item:
        groupby_cols_cpc.append('Item')
    if has_year:
        groupby_cols_cpc.append('year')
    if has_ghg:
        groupby_cols_cpc.append('GHG')
    
    if len(groupby_cols_cpc) >= 3 and 'value' in df.columns:
        result_by_ctry_proc_comm = df.groupby(by=groupby_cols_cpc, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        # Region_label_new已经在df中，groupby会自动保留
        if 'Region_label_new' not in result_by_ctry_proc_comm.columns and region_mapping:
            result_by_ctry_proc_comm['Region_label_new'] = result_by_ctry_proc_comm['M49_Country_Code'].map(region_mapping)
        
        # ✅ 添加Global汇总行
        if has_process and has_item and has_year and has_ghg:
            global_groupby_cols = ['Process', 'Item', 'year', 'GHG']
            global_agg = df.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = '000'
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # 合并
            result_by_ctry_proc_comm = pd.concat([result_by_ctry_proc_comm, global_agg], ignore_index=True)
        
        # ✅ 过滤：排除'All Animals'和'All Crops'
        if 'Item' in result_by_ctry_proc_comm.columns:
            result_by_ctry_proc_comm = result_by_ctry_proc_comm[
                ~result_by_ctry_proc_comm['Item'].isin(['All Animals', 'All Crops'])
            ]
        
        # ✅ 列重新排序：M49_Country_Code, Region_label_new, Process, Item, year, GHG, total_emissions
        cols_order = ['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry_proc_comm.columns]
        result_by_ctry_proc_comm = result_by_ctry_proc_comm[existing_cols]
        
        # ✅ 排序：Region_label_new, Process, Item, GHG均升序
        sort_cols = ['Region_label_new']
        if 'Process' in result_by_ctry_proc_comm.columns:
            sort_cols.append('Process')
        if 'Item' in result_by_ctry_proc_comm.columns:
            sort_cols.append('Item')
        if 'GHG' in result_by_ctry_proc_comm.columns:
            sort_cols.append('GHG')
        result_by_ctry_proc_comm = result_by_ctry_proc_comm.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry_proc_comm结果形状: {result_by_ctry_proc_comm.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry_proc_comm groupby (条件不满足: len={len(groupby_cols_cpc)}, value_exists={'value' in df.columns})")
        result_by_ctry_proc_comm = pd.DataFrame()
    
    # 4. 生成国家-过程汇总（✅ 只用M49_Country_Code）
    groupby_cols_cp = []
    if has_m49:
        groupby_cols_cp.append('M49_Country_Code')
    if has_process:
        groupby_cols_cp.append('Process')
    if has_year:
        groupby_cols_cp.append('year')
    if has_ghg:
        groupby_cols_cp.append('GHG')
    
    if len(groupby_cols_cp) >= 2 and 'value' in df.columns:
        # [DEBUG] Check input data BEFORE groupby
        import logging
        logging.info(f"\n[DEBUG-GROUPBY-INPUT] by_ctry_proc groupby INPUT:")
        logging.info(f"  - df shape: {df.shape}")
        logging.info(f"  - years in df: {sorted(df['year'].unique().tolist()) if 'year' in df.columns else 'N/A'}")
        logging.info(f"  - 2080 data rows in df: {len(df[df['year']==2080]) if 'year' in df.columns else 0}")
        if 'year' in df.columns and 'Process' in df.columns:
            livestock_procs = ['Enteric fermentation', 'Manure management', 'Manure applied to soils', 'Manure left on pasture']
            for proc in livestock_procs:
                df_2080_proc = df[(df['year']==2080) & (df['Process']==proc)]
                logging.info(f"  - {proc} 2080年: {len(df_2080_proc)}行, value非零={sum(df_2080_proc['value'].notna() & (df_2080_proc['value']!=0))}, 总计={df_2080_proc['value'].sum():.2f}")
        
        # ✅ Sheet2直接对所有数据groupby，不需要排除dairy/non-dairy
        # groupby会自动按Process汇总，不存在double counting问题
        result_by_ctry_proc = df.groupby(by=groupby_cols_cp, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        
        # [DEBUG] Check output data AFTER groupby
        logging.info(f"\n[DEBUG-GROUPBY-OUTPUT] by_ctry_proc groupby OUTPUT:")
        logging.info(f"  - result shape: {result_by_ctry_proc.shape}")
        logging.info(f"  - years in result: {sorted(result_by_ctry_proc['year'].unique().tolist()) if 'year' in result_by_ctry_proc.columns else 'N/A'}")
        logging.info(f"  - 2080 data rows in result: {len(result_by_ctry_proc[result_by_ctry_proc['year']==2080]) if 'year' in result_by_ctry_proc.columns else 0}")
        if 'year' in result_by_ctry_proc.columns and 'Process' in result_by_ctry_proc.columns:
            for proc in livestock_procs:
                res_2080_proc = result_by_ctry_proc[(result_by_ctry_proc['year']==2080) & (result_by_ctry_proc['Process']==proc)]
                logging.info(f"  - {proc} 2080年: {len(res_2080_proc)}行, total非零={sum(res_2080_proc['total_emissions'].notna() & (res_2080_proc['total_emissions']!=0))}, 总计={res_2080_proc['total_emissions'].sum():.2f}")
        
        # Region_label_new已经在df中
        if 'Region_label_new' not in result_by_ctry_proc.columns and region_mapping:
            result_by_ctry_proc['Region_label_new'] = result_by_ctry_proc['M49_Country_Code'].map(region_mapping)
        
        # ✅ 添加Global汇总行
        if has_process and has_year and has_ghg:
            global_groupby_cols = ['Process', 'year', 'GHG']
            global_agg = df.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = '000'
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # 合并
            result_by_ctry_proc = pd.concat([result_by_ctry_proc, global_agg], ignore_index=True)
        
        # ✅ 列重新排序：M49_Country_Code, Region_label_new, Process, year, GHG, total_emissions
        cols_order = ['M49_Country_Code', 'Region_label_new', 'Process', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry_proc.columns]
        result_by_ctry_proc = result_by_ctry_proc[existing_cols]
        
        # ✅ 排序：Region_label_new, Process, GHG均升序
        sort_cols = ['Region_label_new']
        if 'Process' in result_by_ctry_proc.columns:
            sort_cols.append('Process')
        if 'GHG' in result_by_ctry_proc.columns:
            sort_cols.append('GHG')
        result_by_ctry_proc = result_by_ctry_proc.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry_proc结果形状: {result_by_ctry_proc.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry_proc groupby (条件不满足)")
        result_by_ctry_proc = pd.DataFrame()
    
    # 5. 生成国家汇总（✅ 只用M49_Country_Code）
    # ✅ 使用排除dairy/non-dairy后的数据，与by_ctry_proc保持一致
    groupby_cols_c = []
    if has_m49:
        groupby_cols_c.append('M49_Country_Code')
    if has_year:
        groupby_cols_c.append('year')
    if has_ghg:
        groupby_cols_c.append('GHG')
    
    if len(groupby_cols_c) >= 2 and 'value' in df.columns:
        # ✅ Sheet3直接对所有数据groupby
        result_by_ctry = df.groupby(by=groupby_cols_c, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        # Region_label_new已经在df中
        if 'Region_label_new' not in result_by_ctry.columns and region_mapping:
            result_by_ctry['Region_label_new'] = result_by_ctry['M49_Country_Code'].map(region_mapping)
        
        # ✅ 添加Global汇总行
        if has_year and has_ghg:
            global_groupby_cols = ['year', 'GHG']
            global_agg = df.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = '000'
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # 合并
            result_by_ctry = pd.concat([result_by_ctry, global_agg], ignore_index=True)
        
        # ✅ 列重新排序：M49_Country_Code, Region_label_new, year, GHG, total_emissions
        cols_order = ['M49_Country_Code', 'Region_label_new', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry.columns]
        result_by_ctry = result_by_ctry[existing_cols]
        
        # ✅ 排序：Region_label_new, GHG均升序
        sort_cols = ['Region_label_new']
        if 'GHG' in result_by_ctry.columns:
            sort_cols.append('GHG')
        result_by_ctry = result_by_ctry.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry结果形状: {result_by_ctry.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry groupby (条件不满足)")
        result_by_ctry = pd.DataFrame()
    
    # 年份横向展开（透视表）- 直接替换原有格式
    # 为每个汇总级别生成透视表（覆盖原 long format）
    import logging
    logging.info(f"\n[DEBUG-PIVOT-LOOP] 开始pivot循环")
    logging.info(f"  - result_by_ctry_proc_comm: {type(result_by_ctry_proc_comm)}, empty={result_by_ctry_proc_comm.empty if isinstance(result_by_ctry_proc_comm, pd.DataFrame) else 'N/A'}")
    logging.info(f"  - result_by_ctry_proc: {type(result_by_ctry_proc)}, empty={result_by_ctry_proc.empty if isinstance(result_by_ctry_proc, pd.DataFrame) else 'N/A'}")
    logging.info(f"  - result_by_ctry: {type(result_by_ctry)}, empty={result_by_ctry.empty if isinstance(result_by_ctry, pd.DataFrame) else 'N/A'}")
    
    for key, df_result in [('by_ctry_proc_comm', result_by_ctry_proc_comm),
                           ('by_ctry_proc', result_by_ctry_proc),
                           ('by_ctry', result_by_ctry)]:
        logging.info(f"\n[DEBUG-PIVOT-LOOP] 检查{key}: type={type(df_result)}, empty={df_result.empty if isinstance(df_result, pd.DataFrame) else 'N/A'}, has_year={'year' in df_result.columns if isinstance(df_result, pd.DataFrame) else 'N/A'}")
        if isinstance(df_result, pd.DataFrame) and not df_result.empty and 'year' in df_result.columns:
            # *** DEBUG ***
            print(f"[DEBUG] 透视前{key}: 行={len(df_result)}, M49值={sorted(df_result['M49_Country_Code'].unique().tolist()) if 'M49_Country_Code' in df_result.columns else 'N/A'}")
            print(f"[DEBUG] 透视前{key}完整数据:\n{df_result[['M49_Country_Code', 'year', 'total_emissions']].head(10) if 'M49_Country_Code' in df_result.columns else 'N/A'}")
            # *** END DEBUG ***
            
            # 确定索引列（除了year和total_emissions）
            index_cols = [col for col in df_result.columns 
                         if col not in ['year', 'total_emissions']]
            
            print(f"[DEBUG] {key} 的index_cols: {index_cols}")
            
            if index_cols:
                try:
                    # 透视：年份横向展开
                    pivot_df = df_result.pivot_table(
                        index=index_cols,
                        columns='year',
                        values='total_emissions',
                        aggfunc='sum',
                        fill_value=0
                    ).reset_index()
                    
                    import logging
                    logging.info(f"[DEBUG] 透视后{key}: 行={len(pivot_df)}, M49值={sorted(pivot_df['M49_Country_Code'].unique().tolist()) if 'M49_Country_Code' in pivot_df.columns else 'N/A'}")
                    
                    # DEBUG: 检查透视后的年份列
                    year_cols_before_rename = [col for col in pivot_df.columns if isinstance(col, (int, float)) and not pd.isna(col)]
                    logging.info(f"[DEBUG-PIVOT] {key}透视后的年份列: {sorted(year_cols_before_rename)}")
                    if 2080 in year_cols_before_rename:
                        livestock_procs = ['Enteric fermentation', 'Manure management', 'Manure applied to soils', 'Manure left on pasture']
                        if key == 'by_ctry_proc' and 'Process' in pivot_df.columns:
                            for proc in livestock_procs:
                                proc_rows = pivot_df[pivot_df['Process'] == proc]
                                if len(proc_rows) > 0:
                                    # ✅ 修复：排除Global行（M49='000'）以避免重复计数
                                    if 'M49_Country_Code' in pivot_df.columns:
                                        proc_rows_no_global = proc_rows[proc_rows['M49_Country_Code'] != '000']
                                        y2080_nonzero = (proc_rows_no_global[2080].notna() & (proc_rows_no_global[2080] != 0)).sum()
                                        y2080_sum = proc_rows_no_global[2080].sum()
                                        y2080_global = proc_rows[proc_rows['M49_Country_Code'] == '000'][2080].sum() if '000' in proc_rows['M49_Country_Code'].values else 0
                                        logging.info(f"[DEBUG-PIVOT] {proc}: {len(proc_rows)}行(含Global), Y2080非零={y2080_nonzero}(不含Global), 总计={y2080_sum:.2f}(不含Global), Global={y2080_global:.2f}")
                                    else:
                                        y2080_nonzero = (proc_rows[2080].notna() & (proc_rows[2080] != 0)).sum()
                                        y2080_sum = proc_rows[2080].sum()
                                        logging.info(f"[DEBUG-PIVOT] {proc}: {len(proc_rows)}行, Y2080非零={y2080_nonzero}, 总计={y2080_sum:.2f}")
                    
                    # 重命名年份列为 Y2010, Y2020 等格式
                    year_columns = {col: f'Y{col}' for col in pivot_df.columns 
                                   if isinstance(col, (int, float)) and not pd.isna(col)}
                    pivot_df = pivot_df.rename(columns=year_columns)
                    
                    # ✅ 透视后保持列顺序：非年份列在前，年份列在后
                    non_year_cols = [col for col in pivot_df.columns if not col.startswith('Y')]
                    year_cols = sorted([col for col in pivot_df.columns if col.startswith('Y')])
                    pivot_df = pivot_df[non_year_cols + year_cols]
                    
                    # 覆盖原变量（用透视表替换）
                    if key == 'by_ctry_proc_comm':
                        result_by_ctry_proc_comm = pivot_df
                    elif key == 'by_ctry_proc':
                        result_by_ctry_proc = pivot_df
                    elif key == 'by_ctry':
                        result_by_ctry = pivot_df
                except Exception as e:
                    # 如果透视失败，跳过
                    print(f"[ERROR] {key} 透视失败: {e}")
                    pass
    
    # ✅ 为每个透视表添加CO2eq行（在GHG列中追加新行）
    def add_ghg_co2eq_rows(pivot_df: pd.DataFrame) -> pd.DataFrame:
        """为透视表在GHG列中追加CO2eq行（CH4_CO2eq、N2O_CO2eq、CO2_CO2eq、CO2eq总计）"""
        if pivot_df.empty or 'GHG' not in pivot_df.columns:
            return pivot_df
        
        # 找出年份列（Y开头）
        year_cols = sorted([col for col in pivot_df.columns if col.startswith('Y')])
        if not year_cols:
            return pivot_df
        
        # 找出非年份列（用于分组和保持信息）
        non_year_cols = [col for col in pivot_df.columns if not col.startswith('Y')]
        
        GWP100_AR6 = {'CO2': 1.0, 'CH4': 27.2, 'N2O': 273.0}
        
        new_rows = []
        
        # 按所有非GHG、非年份列分组
        groupby_cols = [col for col in non_year_cols if col != 'GHG']
        
        if groupby_cols:
            for group_vals, group_df in pivot_df.groupby(groupby_cols, dropna=False):
                # 为这个分组添加CO2eq行
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                
                group_dict = dict(zip(groupby_cols, group_vals))
                
                # 为每个GHG创建对应的CO2eq行
                for ghg in ['CH4', 'N2O', 'CO2']:
                    ghg_co2eq_row = group_dict.copy()
                    ghg_co2eq_row['GHG'] = f'{ghg}_CO2eq'
                    
                    # 查找原始GHG行
                    ghg_rows = group_df[group_df['GHG'] == ghg]
                    if not ghg_rows.empty:
                        ghg_row = ghg_rows.iloc[0]
                        gwp = GWP100_AR6[ghg]
                        for year_col in year_cols:
                            if year_col in ghg_row.index:
                                value = ghg_row[year_col]
                                ghg_co2eq_row[year_col] = value * gwp if pd.notna(value) and value != 0 else 0
                            else:
                                ghg_co2eq_row[year_col] = 0
                    else:
                        # 如果该GHG不存在，填0
                        for year_col in year_cols:
                            ghg_co2eq_row[year_col] = 0
                    
                    new_rows.append(ghg_co2eq_row)
                
                # 创建总CO2eq行
                total_co2eq_row = group_dict.copy()
                total_co2eq_row['GHG'] = 'CO2eq'
                
                for year_col in year_cols:
                    total_value = 0
                    for ghg, gwp in GWP100_AR6.items():
                        ghg_rows = group_df[group_df['GHG'] == ghg]
                        if not ghg_rows.empty:
                            ghg_row = ghg_rows.iloc[0]
                            if year_col in ghg_row.index:
                                value = ghg_row[year_col]
                                if pd.notna(value) and value != 0:
                                    total_value += value * gwp
                    total_co2eq_row[year_col] = total_value
                
                new_rows.append(total_co2eq_row)
        
        # 合并原数据和新的CO2eq行
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            result_df = pd.concat([pivot_df, new_rows_df], ignore_index=True)
            # 保持列顺序
            result_df = result_df[non_year_cols + year_cols]
            return result_df
        else:
            return pivot_df
    
    # 应用到三个汇总表
    if isinstance(result_by_ctry_proc_comm, pd.DataFrame) and not result_by_ctry_proc_comm.empty:
        result_by_ctry_proc_comm = add_ghg_co2eq_rows(result_by_ctry_proc_comm)
    
    if isinstance(result_by_ctry_proc, pd.DataFrame) and not result_by_ctry_proc.empty:
        result_by_ctry_proc = add_ghg_co2eq_rows(result_by_ctry_proc)
    
    if isinstance(result_by_ctry, pd.DataFrame) and not result_by_ctry.empty:
        result_by_ctry = add_ghg_co2eq_rows(result_by_ctry)
    
    # ✅ 对三个宽表进行排序：M49_Country_Code-Process-Item-GHG升序
    def sort_wide_table(df: pd.DataFrame) -> pd.DataFrame:
        """按M49_Country_Code-Process-Item-GHG升序排序"""
        if df.empty:
            return df
        
        sort_cols = []
        for col in ['M49_Country_Code', 'Process', 'Item', 'GHG']:
            if col in df.columns:
                sort_cols.append(col)
        
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        return df
    
    result_by_ctry_proc_comm = sort_wide_table(result_by_ctry_proc_comm)
    result_by_ctry_proc = sort_wide_table(result_by_ctry_proc)
    result_by_ctry = sort_wide_table(result_by_ctry)
    
    # ✅ 关键修复：将M49格式从'051'转换为"'051"（带引号前缀）
    def format_m49_with_quote(df_in: pd.DataFrame) -> pd.DataFrame:
        """将M49_Country_Code格式化为'xxx格式（带前导单引号）"""
        if df_in is None or df_in.empty or 'M49_Country_Code' not in df_in.columns:
            return df_in
        df_out = df_in.copy()
        def _fmt(x):
            if pd.isna(x) or x is None or str(x).strip() == '':
                return ''
            s = str(x).strip().lstrip("'\"")
            # 补齐3位并加引号前缀
            if s.isdigit():
                return f"'{int(s):03d}"
            return f"'{s}" if not s.startswith("'") else s
        df_out['M49_Country_Code'] = df_out['M49_Country_Code'].apply(_fmt)
        return df_out
    
    result_by_ctry_proc_comm = format_m49_with_quote(result_by_ctry_proc_comm)
    result_by_ctry_proc = format_m49_with_quote(result_by_ctry_proc)
    result_by_ctry = format_m49_with_quote(result_by_ctry)
    result_long = format_m49_with_quote(result_long)
    
    # 返回结果：不再包含 by_year（按要求删除），透视表已覆盖原变量
    return {
        'by_ctry_proc_comm': result_by_ctry_proc_comm,
        'by_ctry_proc': result_by_ctry_proc,
        'by_ctry': result_by_ctry,
        'long': result_long,  # 保留 Detail_Long
    }


def summarize_emissions(fao_results: Dict[str, Any],
                        extra_emis: Optional[pd.DataFrame]=None,
                        process_meta_map: Optional[dict]=None,
                        dict_v3_path: Optional[str]=None,
                        allowed_years: Optional[set]=None,
                        production_df: Optional[pd.DataFrame]=None) -> Dict[str, pd.DataFrame]:
    """
    从FAO模块结果生成汇总（向后兼容接口）
    
    Args:
        fao_results: FAO模块返回的结果字典，格式为 {'GCE': [df1, df2, ...], 'GLE': [...], ...}
        extra_emis: 额外的排放数据
        process_meta_map: 过程元数据映射
        dict_v3_path: dict_v3.xlsx路径，用于加载Region映射
        allowed_years: 允许的年份集合，如果提供则过滤其他年份
        production_df: 生产数据（包含stock），用于拆分历史阶段merged livestock排放
        
    Returns:
        多个粒度的汇总结果字典
    """
    # DEBUG: log allowed_years details
    import logging
    logging.info(f"\n[DEBUG-CRITICAL] summarize_emissions arguments:")
    logging.info(f"  - allowed_years type: {type(allowed_years)}")
    allowed_years_set = {int(y) for y in allowed_years} if allowed_years else None
    logging.info(f"  - allowed_years values: {sorted(list(allowed_years_set)) if allowed_years_set else None}")
    logging.info(f"  - contains 2080: {2080 in allowed_years_set if allowed_years_set else 'N/A'}")
    if not fao_results or not isinstance(fao_results, dict):
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # 收集所有排放数据
    all_emis_dfs = []

    def ensure_scalar_process(df, process_value=None):
        """确保df的Process和Item列是标量值，而不是Series或其他对象"""
        df = df.copy()
        
        # 只有当process_value被明确提供时，才覆盖Process列
        # 这样可以保留来自GLE计算的原始process列
        if process_value is not None:
            # 删除已有的process/Process列（可能包含Series）
            for col in ['process', 'Process', 'process_old']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            # 添加新的Process列作为标量值
            df['Process'] = str(process_value)
        else:
            # 如果没有提供process_value，确保现有的Process列是字符串类型
            if 'Process' in df.columns:
                # 检查是否有Series值
                if df['Process'].apply(lambda x: isinstance(x, pd.Series)).any():
                    # 如果有Series，展平它们
                    df['Process'] = df['Process'].apply(
                        lambda x: str(x.iloc[0]) if isinstance(x, pd.Series) and len(x) > 0 else str(x)
                    )
                else:
                    # 直接转换为字符串
                    df['Process'] = df['Process'].astype(str)
            elif 'process' in df.columns:
                # 重命名为Process并转换为字符串
                df = df.rename(columns={'process': 'Process'})
                df['Process'] = df['Process'].astype(str)
        
        # ✅ 新增：确保Item列也是标量值
        for item_col in ['Item', 'item', 'commodity']:
            if item_col in df.columns:
                # 检查是否有Series或列表值
                def flatten_value(x):
                    if isinstance(x, pd.Series):
                        return str(x.iloc[0]) if len(x) > 0 else ''
                    elif isinstance(x, (list, tuple)):
                        return str(x[0]) if len(x) > 0 else ''
                    else:
                        return str(x)
                
                if df[item_col].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any():
                    df[item_col] = df[item_col].apply(flatten_value)
                else:
                    df[item_col] = df[item_col].astype(str)
                
                # 如果需要重命名为Item
                if item_col != 'Item':
                    df = df.rename(columns={item_col: 'Item'})
                break
        
        return df
    
    def collect_extra_frames(obj) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        if obj is None:
            return frames
        if isinstance(obj, pd.DataFrame):
            if not obj.empty:
                frames.append(obj.copy())
            return frames
        if isinstance(obj, (list, tuple, set)):
            for entry in obj:
                frames.extend(collect_extra_frames(entry))
            return frames
        if isinstance(obj, dict):
            for entry in obj.values():
                frames.extend(collect_extra_frames(entry))
        return frames

    for module_name, module_results in fao_results.items():
        if module_results is None:
            continue
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG-S4_1] 处理模块 {module_name}, 类型: {type(module_results)}")
            
        # module_results 可能是 DataFrame, list, 或 dict
        if isinstance(module_results, pd.DataFrame):
            if not module_results.empty:
                # ✅ 关键修复：LUC/GSOIL数据已有正确的Process列，不应覆盖
                # LUC包含: De/Reforestation_crop, De/Reforestation_pasture, Forest, Wood harvest
                # GSOIL包含: Drained organic soils
                # 只有GCE/GFIRE等没有Process列的数据才需要设置process_value=module_name
                process_val = None if module_name in ['LUC', 'GSOIL'] else module_name
                df = ensure_scalar_process(module_results, process_value=process_val)
                if 'module' not in df.columns:
                    df['module'] = module_name
                all_emis_dfs.append(df)
                print(f"  添加DataFrame: {len(df)} 行")
        elif isinstance(module_results, list):
            print(f"  List包含 {len(module_results)} 个元素")
            for idx, item in enumerate(module_results):
                if isinstance(item, pd.DataFrame) and not item.empty:
                    # List中的DataFrame：保留原有的process列
                    print(f"  [List项{idx}] 原始数据 - 形状: {item.shape}, 列: {list(item.columns)}")
                    if 'process' in item.columns:
                        print(f"    process列值: {_unique_column_values(item, 'process')}")
                    
                    df = ensure_scalar_process(item, process_value=None)
                    
                    # 标准化列名：统一气体列为大写
                    rename_map = {
                        'ch4_kt': 'CH4_kt',
                        'n2o_kt': 'N2O_kt',
                        'co2_kt': 'CO2_kt',
                        'process': 'Process',
                        'country': 'Country'
                    }
                    df = df.rename(columns=rename_map)
                    
                    print(f"  [List项{idx}] ensure_scalar_process后 - 形状: {df.shape}, 列: {list(df.columns)}")
                    if 'Process' in df.columns:
                        print(f"    Process列值: {_unique_column_values(df, 'Process')}")
                    
                    if 'module' not in df.columns:
                        df['module'] = module_name
                    
                    # WARNING: debug: inspect emission values
                    print(f"  [List项{idx}] 最终形状: {df.shape}")
                    for gas_col in ['ch4_kt', 'n2o_kt', 'co2_kt', 'CH4_kt', 'N2O_kt', 'CO2_kt']:
                        if gas_col in df.columns:
                            non_zero = (df[gas_col] > 0).sum()
                            total = df[gas_col].sum()
                            print(f"    {gas_col}: {non_zero}个非零值, 总计={total:.2f}")
                    
                    print(f"  [List项{idx}] 添加到all_emis_dfs前 - 行数: {len(df)}")
                    # ✅ 关键调试：检查年份分布
                    if 'year' in df.columns:
                        years_in_df = sorted(df['year'].unique())
                        print(f"    年份: {years_in_df}")
                        print(f"    包含2080: {2080 in years_in_df}")
                    all_emis_dfs.append(df)
                    print(f"  [List项{idx}] 已添加到all_emis_dfs (总计现在有{len(all_emis_dfs)}个DataFrame)")
        elif isinstance(module_results, dict):
            print(f"  Dict包含 {len(module_results)} 个process")
            # FAO 返回的格式：{'GCE': {'Residues': df, 'Burning': df, ...}}
            for process_name, process_df in module_results.items():
                if isinstance(process_df, pd.DataFrame) and not process_df.empty:
                    print(f"  [Dict项 {process_name}] 原始数据 - 形状: {process_df.shape}, 列: {list(process_df.columns)}")
                    
                    # Dict中的DataFrame：保留原有的Process列，不用key覆盖
                    df = ensure_scalar_process(process_df, process_value=None)
                    
                    # 标准化列名：统一气体列为大写
                    rename_map = {
                        'ch4_kt': 'CH4_kt',
                        'n2o_kt': 'N2O_kt',
                        'co2_kt': 'CO2_kt',
                        'process': 'Process'
                    }
                    df = df.rename(columns=rename_map)
                    
                    if 'module' not in df.columns:
                        df['module'] = module_name
                    
                    print(f"  [Dict项 {process_name}] 标准化后 - 形状: {df.shape}, 列: {list(df.columns)}")
                    all_emis_dfs.append(df)
                    print(f"  [Dict项 {process_name}] 已添加到all_emis_dfs (总计现在有{len(all_emis_dfs)}个DataFrame)")

    # 添加额外排放数据（DataFrame/List/Dict 均可）
    extra_frames = collect_extra_frames(extra_emis)
    if extra_frames:
        print(f"[DEBUG] 额外排放输入共 {len(extra_frames)} 个DataFrame")
    for idx, extra_df in enumerate(extra_frames):
        df = ensure_scalar_process(extra_df, process_value=None)
        if 'module' not in df.columns:
            df['module'] = 'extra_emis'
        all_emis_dfs.append(df)
        print(f"  [EXTRA {idx}] 行数: {len(df)}, 列: {list(df.columns)}")
    
    if not all_emis_dfs:
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # 合并所有数据
    combined_df = pd.concat(all_emis_dfs, ignore_index=True)
    # Normalize year column to integers for reliable filtering
    if 'year' in combined_df.columns:
        combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce')
        combined_df = combined_df.dropna(subset=['year'])
        combined_df['year'] = combined_df['year'].astype(int)
    
    # ✅ 关键修复：去除重复排放记录
    # 相同的(M49, Process, Item, GHG, year)只保留一条，避免重复计数
    key_cols = ['M49_Country_Code', 'Process', 'Item', 'GHG', 'year']
    existing_key_cols = [c for c in key_cols if c in combined_df.columns]
    if len(existing_key_cols) >= 4:  # 至少需要4个关键列
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=existing_key_cols, keep='first')
        after_dedup = len(combined_df)
        if before_dedup > after_dedup:
            print(f"⚠️ [WARN] 发现并移除 {before_dedup - after_dedup} 行重复排放数据 ({before_dedup} → {after_dedup})")
            print(f"    去重键: {existing_key_cols}")
    
    
    import logging
    logging.info(f"\n[DEBUG] Combined后的DataFrame:")
    logging.info(f"  - 总行数: {len(combined_df)}")
    logging.info(f"  - 列: {list(combined_df.columns)}")
    if 'year' in combined_df.columns:
        years_in_combined = sorted(combined_df['year'].unique())
        logging.info(f"  - 年份范围: {years_in_combined}")
        logging.info(f"  - 2080是否在combined_df: {2080 in years_in_combined}")
        if 2080 in years_in_combined:
            livestock_2080 = combined_df[(combined_df['year'] == 2080) & (combined_df['module'] == 'GLE')]
            logging.info(f"  - GLE模块2080年数据: {len(livestock_2080)} 行")
    if 'module' in combined_df.columns:
        logging.info(f"  - module分布: {combined_df['module'].value_counts().to_dict()}")
    if 'Process' in combined_df.columns:
        logging.info(f"  - Process分布: {combined_df['Process'].value_counts().to_dict()}")
    
    # WARNING: filter allowed years to prevent stray years
    import logging
    if allowed_years_set and 'year' in combined_df.columns:
        years_in_data = sorted(combined_df['year'].unique())
        logging.info(f"[DEBUG-CRITICAL] Before year filter:")
        logging.info(f"  - years in combined_df: {years_in_data}")
        logging.info(f"  - allowed_years: {sorted(list(allowed_years_set)) if isinstance(allowed_years_set, set) else allowed_years_set}")
        logging.info(f"  - contains 2080 in combined_df: {2080 in years_in_data}")
        logging.info(f"  - contains 2080 in allowed_years: {2080 in allowed_years_set}")
        
        before_len = len(combined_df)
        combined_df = combined_df[combined_df['year'].isin(allowed_years_set)].copy()
        after_len = len(combined_df)
        if before_len != after_len:
            print(f"[DEBUG] Year filter: {before_len} -> {after_len} rows (removed {before_len-after_len})")
            print(f"  - removed years: {removed_years}")
    
    # ✅ 关键修复：标记哪些行是"商品基础"（应跳过dict_v3 Item_Emis过滤）
    # 规则：
    # 1. 如果有commodity列 → 肯定是商品基础（FAO direct）
    # 2. 否则，如果有Process列（排放模块）→ 也是商品基础（排放Item，需要保留）
    # 3. 其他情况 → 按严格的排放Item过滤
    if 'commodity' in combined_df.columns:
        combined_df['_is_commodity_based'] = True
        n_marked = len(combined_df)
        print(f"[DEBUG] 标记了 {n_marked} 行为commodity-based（FAO商品数据）")
    elif 'process' in combined_df.columns or 'Process' in combined_df.columns:
        # GLE/GCE/GOS等排放模块的数据有process列，这些是排放Item（如'Cattle, dairy'）
        # 这些应该被保留，因此标记为True来跳过dict_v3 Item_Emis过滤
        combined_df['_is_commodity_based'] = True
        n_marked = len(combined_df)
        print(f"[DEBUG] 标记了 {n_marked} 行为commodity-based（排放模块数据，包含Process列）")
    else:
        combined_df['_is_commodity_based'] = False
        print(f"[DEBUG] 标记了 {len(combined_df)} 行为emission-based（需要dict_v3 Item_Emis过滤）")
    
    # 标准化列名
    column_mapping = {
        'country': 'Country',
        'commodity': 'Item',  # 保持原有逻辑：commodity重命名为Item
        'process': 'Process',
        'ch4_kt': 'CH4_kt',
        'n2o_kt': 'N2O_kt',
        'co2_kt': 'CO2_kt',
        'CH4_kt': 'CH4_kt',  # ✅ 新增：GLE返回的是大写的CH4_kt
        'N2O_kt': 'N2O_kt',  # ✅ 新增：支持大写的列名
        'CO2_kt': 'CO2_kt',  # ✅ 新增：支持大写的列名
        'emissions_kt': 'value'
    }
    combined_df = combined_df.rename(columns={k: v for k, v in column_mapping.items() if k in combined_df.columns})
    
    # 确保Process列是字符串类型且是标量（不是Series）
    if 'Process' in combined_df.columns:
        # 检查是否有任何Series值
        mask = combined_df['Process'].apply(lambda x: isinstance(x, pd.Series))
        if mask.any():
            # 如果有Series，展平它们
            combined_df['Process'] = combined_df['Process'].apply(
                lambda x: str(x.iloc[0]) if isinstance(x, pd.Series) and len(x) > 0 else str(x)
            )
        else:
            # 直接转换为字符串
            combined_df['Process'] = combined_df['Process'].astype(str)
    
    # ✅ 新增：确保Item列也是标量值（不是Series或列表）
    if 'Item' in combined_df.columns:
        # 检查是否有任何Series或列表值
        def flatten_item(x):
            if isinstance(x, pd.Series):
                return str(x.iloc[0]) if len(x) > 0 else ''
            elif isinstance(x, (list, tuple)):
                return str(x[0]) if len(x) > 0 else ''
            else:
                return str(x)
        
        has_non_scalar = combined_df['Item'].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any()
        if has_non_scalar:
            combined_df['Item'] = combined_df['Item'].apply(flatten_item)
        else:
            combined_df['Item'] = combined_df['Item'].astype(str)
    
    # ✅ 新增：确保其他关键列也是标量（Country, M49_Country_Code, GHG等）
    for col in ['Country', 'M49_Country_Code', 'GHG', 'year']:
        if col in combined_df.columns:
            def flatten_col(x):
                if isinstance(x, pd.Series):
                    return x.iloc[0] if len(x) > 0 else (None if col == 'year' else '')
                elif isinstance(x, (list, tuple)):
                    return x[0] if len(x) > 0 else (None if col == 'year' else '')
                else:
                    return x
            
            has_non_scalar = combined_df[col].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any()
            if has_non_scalar:
                combined_df[col] = combined_df[col].apply(flatten_col)
            elif col != 'year':  # year通常是数值类型
                try:
                    combined_df[col] = combined_df[col].astype(str)
                except:
                    pass
    
    # 调试信息：检查combined_df结构
    print(f"\n[DEBUG] Combined DataFrame 结构:")
    print(f"  - 形状: {combined_df.shape}")
    print(f"  - 列名: {list(combined_df.columns)}")
    process_values = _unique_column_values(combined_df, 'Process')
    if not combined_df.empty:
        print(f"  - Process唯一值: {process_values[:10] if process_values.size else '无Process列'}")
        print(f"  - 年份范围: {combined_df['year'].min()}-{combined_df['year'].max() if 'year' in combined_df.columns else '无year列'}")
        # 检查气体列的值
        for gas_col in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'ch4_kt', 'n2o_kt', 'co2_kt']:
            if gas_col in combined_df.columns:
                non_zero = int((combined_df[gas_col] > 0).sum())
                total_val = float(combined_df[gas_col].sum())
                print(f"  - {gas_col}: {non_zero}个非零值, 总计={total_val:.2f}")
        # 检查Country和Item列的空值情况
        if 'Country' in combined_df.columns:
            country_null_count = combined_df['Country'].isna().sum()
            print(f"  - Country空值: {country_null_count}/{len(combined_df)}")
        if 'Item' in combined_df.columns:
            item_null_count = combined_df['Item'].isna().sum()
            print(f"  - Item空值: {item_null_count}/{len(combined_df)}")
    
    # ✅ 统一列名：将小写气体列重命名为大写格式（避免列名重复）
    col_rename_map = {
        'ch4_kt': 'CH4_kt',
        'n2o_kt': 'N2O_kt', 
        'co2_kt': 'CO2_kt',
        'country': 'Country'
    }
    combined_df = combined_df.rename(columns=col_rename_map)
    
    # 删除完全重复的列（如果存在）
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # ✅ 关键修复：处理混合格式数据（wide: CH4_kt/N2O_kt/CO2_kt + long: GHG/value）
    has_gas_cols = any(col in combined_df.columns for col in ['CH4_kt', 'N2O_kt', 'CO2_kt'])
    has_ghg_value = ('GHG' in combined_df.columns and 'value' in combined_df.columns)
    if has_gas_cols and has_ghg_value:
        # 混合格式：需要分别处理wide格式（GLE/GCE）和long格式（LUC）
        import logging
        logging.info(f"[INFO] 检测到混合格式数据（wide+long），分别处理...")
        
        # 1. 识别已经是long格式的行
        # ✅ 关键修复：使用module列辅助识别Long格式，防止LUC未来年份（value=NaN）被误判为Wide格式
        if 'module' in combined_df.columns:
            # LUC模块强制视为Long格式
            is_luc = combined_df['module'].astype(str).isin(['LUC'])
            # 其他模块如果GHG和value都有值，也视为Long格式
            is_other_long = combined_df['GHG'].notna() & combined_df['value'].notna()
            is_long = is_luc | is_other_long
            logging.info(f"  - 使用module列识别Long格式: LUC={is_luc.sum()}, Other_Long={is_other_long.sum()}, Total_Long={is_long.sum()}")
        else:
            is_long = combined_df['GHG'].notna() & combined_df['value'].notna()
            logging.info(f"  - 使用GHG/value列识别Long格式: Total_Long={is_long.sum()}")
            
        long_data = combined_df[is_long].copy()
        wide_data = combined_df[~is_long].copy()
        logging.info(f"  - Long格式（LUC等）: {len(long_data)} 行")
        logging.info(f"  - Wide格式（GLE/GCE等）: {len(wide_data)} 行")
        # ✅ 关键调试：检查long_data中的年份分布
        if 'year' in long_data.columns and not long_data.empty:
            long_years = sorted(long_data['year'].unique())
            logging.info(f"  - Long格式年份: {long_years}")
            logging.info(f"  - Long格式包含2080: {2080 in long_years}")
            if 'Process' in long_data.columns:
                long_processes = long_data['Process'].value_counts().to_dict()
                logging.info(f"  - Long格式Process分布: {long_processes}")
        # 2. 转换wide格式为long格式
        # ✅ 关键修复：保持原始单位（kt），不在此处转换为CO2eq
        # GWP转换在后续的generate_detailed_summary中统一进行，避免双重转换
        gas_dfs = []
        if len(wide_data) > 0:
            non_gas_cols = [col for col in wide_data.columns if col not in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'value', 'GHG']]
            if 'CH4_kt' in wide_data.columns:
                ch4_df = wide_data[non_gas_cols + ['CH4_kt']].copy()
                ch4_df['GHG'] = 'CH4'
                ch4_df['value'] = ch4_df['CH4_kt']  # ✅ 保持kt单位，不乘以GWP
                ch4_df = ch4_df.drop(columns=['CH4_kt'])
                gas_dfs.append(ch4_df)
            if 'N2O_kt' in wide_data.columns:
                n2o_df = wide_data[non_gas_cols + ['N2O_kt']].copy()
                n2o_df['GHG'] = 'N2O'
                n2o_df['value'] = n2o_df['N2O_kt']  # ✅ 保持kt单位，不乘以GWP
                n2o_df = n2o_df.drop(columns=['N2O_kt'])
                gas_dfs.append(n2o_df)
            if 'CO2_kt' in wide_data.columns:
                co2_df = wide_data[non_gas_cols + ['CO2_kt']].copy()
                co2_df['GHG'] = 'CO2'
                co2_df['value'] = co2_df['CO2_kt']  # ✅ 保持kt单位
                co2_df = co2_df.drop(columns=['CO2_kt'])
                gas_dfs.append(co2_df)
        # 3. 合并long格式和转换后的wide格式
        all_dfs = [long_data] + gas_dfs
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"  - 合并后总行数: {len(combined_df)}")
        if 'Process' in combined_df.columns:
            logging.info(f"  - 合并后Process分布: {combined_df['Process'].value_counts().to_dict()}")
    elif has_gas_cols:
        # 只有wide格式，按原逻辑处理
        # ✅ 关键修复：保持原始单位（kt），不在此处转换为CO2eq
        gas_dfs = []
        non_gas_cols = [col for col in combined_df.columns if col not in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'value', 'GHG']]
        if 'CH4_kt' in combined_df.columns:
            ch4_df = combined_df[non_gas_cols + ['CH4_kt']].copy()
            ch4_df['GHG'] = 'CH4'
            ch4_df['value'] = ch4_df['CH4_kt']  # ✅ 保持kt单位，不乘以GWP
            ch4_df = ch4_df.drop(columns=['CH4_kt'])
            gas_dfs.append(ch4_df)
        if 'N2O_kt' in combined_df.columns:
            n2o_df = combined_df[non_gas_cols + ['N2O_kt']].copy()
            n2o_df['GHG'] = 'N2O'
            n2o_df['value'] = n2o_df['N2O_kt']  # ✅ 保持kt单位，不乘以GWP
            n2o_df = n2o_df.drop(columns=['N2O_kt'])
            gas_dfs.append(n2o_df)
        if 'CO2_kt' in combined_df.columns:
            co2_df = combined_df[non_gas_cols + ['CO2_kt']].copy()
            co2_df['GHG'] = 'CO2'
            co2_df['value'] = co2_df['CO2_kt']  # ✅ 保持kt单位
            co2_df = co2_df.drop(columns=['CO2_kt'])
            gas_dfs.append(co2_df)
        if gas_dfs:
            combined_df = pd.concat(gas_dfs, ignore_index=True)
            print(f"\n[DEBUG] 气体拆分后:")
            print(f"  - 形状: {combined_df.shape}")
            if 'value' in combined_df.columns:
                non_zero = (combined_df['value'] > 0).sum()
                total_val = combined_df['value'].sum()
                print(f"  - value: {non_zero}个非零值, 总计={total_val:.2f}")
            if 'GHG' in combined_df.columns:
                print(f"  - GHG分布: {combined_df['GHG'].value_counts().to_dict()}")
        else:
            if 'GHG' not in combined_df.columns:
                combined_df['GHG'] = 'Mixed'
            if 'value' not in combined_df.columns:
                combined_df['value'] = 0.0
    elif 'value' not in combined_df.columns:
        if 'GHG' not in combined_df.columns:
            combined_df['GHG'] = 'CO2eq'
    if 'value' not in combined_df.columns:
        combined_df['value'] = 0.0
    
    # 调用统一的汇总函数，传入dict_v3_path和allowed_years
    # 将set转换为list（如果需要）
    allowed_years_list = sorted(allowed_years_set) if allowed_years_set else None
    return summarize_emissions_from_detail(combined_df, 
                                          process_meta_map=process_meta_map,
                                          allowed_years=allowed_years_list,
                                          dict_v3_path=dict_v3_path,
                                          production_df=production_df)


def summarize_market(model, var, universe, data=None, price_df: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    """
    从模型结果提取供需数据，包含M49_Country_Code
    """
    rows = []
    try:
        Qs = var['Qs']; Qd = var['Qd']
    except Exception:
        return pd.DataFrame(columns=['M49_Country_Code','country','year','commodity','Qs','Qd','net_import_t','net_export_t','price_global'])

    Pc = var.get('Pc', {})
    Pnet = var.get('Pnet', {})
    W = var.get('W', {})
    Cij = var.get('C', {})
    Eij = var.get('E', {})
    
    # 构建国家→M49映射
    country_to_m49 = {}
    if data and hasattr(data, 'nodes'):
        for node in data.nodes:
            if hasattr(node, 'country') and hasattr(node, 'm49'):
                country = str(node.country).strip()
                m49 = node.m49
                if m49:
                    # 标准化M49格式：去除引号和前导零，转换为"'xxx"格式
                    m49_str = str(m49).strip().strip("'\"")
                    try:
                        m49_int = int(m49_str)
                        country_to_m49[country] = f"'{m49_int}"
                    except Exception:
                        country_to_m49[country] = m49_str
    
    def _safe_float(val, default=np.nan):
        try:
            if val is None:
                return default
            out = float(val)
            if np.isnan(out):
                return default
            return out
        except Exception:
            return default

    def _get_x(v):
        try:
            return float(v.X)
        except Exception:
            return np.nan

    # 使用set追踪已添加的行，防止重复
    seen = set()
    
    for (i, j, t), svar in Qs.items():
        q_supply = _get_x(svar)
        dvar = Qd.get((i, j, t))
        q_demand = _get_x(dvar)
        imp = max(q_demand - q_supply, 0.0) if np.isfinite(q_supply) and np.isfinite(q_demand) else np.nan
        exp = max(q_supply - q_demand, 0.0) if np.isfinite(q_supply) and np.isfinite(q_demand) else np.nan
        price = Pc.get((j, t))
        price_val = _get_x(price)
        price_net = Pnet.get((i, j, t))
        price_net_val = _get_x(price_net)
        
        m49 = country_to_m49.get(i, '')
        key = (i, t, j)  # 用于去重
        if key not in seen:
            rows.append((m49, i, t, j, q_supply, q_demand, imp, exp, price_val, price_net_val))
            seen.add(key)

    import_slack = var.get('Import') or {}
    export_slack = var.get('Export') or {}
    slack_keys = set()
    if isinstance(import_slack, dict):
        slack_keys.update(import_slack.keys())
    if isinstance(export_slack, dict):
        slack_keys.update(export_slack.keys())
    for key in sorted(slack_keys):
        j, t = key
        imp_var = import_slack.get(key)
        exp_var = export_slack.get(key)
        imp_val = _get_x(imp_var)
        exp_val = _get_x(exp_var)
        if not (np.isfinite(imp_val) or np.isfinite(exp_val)):
            continue
        slack_key = ('ROW', t, j)
        if slack_key not in seen:
            rows.append(('', 'ROW', t, j, np.nan, np.nan,
                         imp_val if np.isfinite(imp_val) else np.nan,
                         exp_val if np.isfinite(exp_val) else np.nan,
                         np.nan, np.nan))
            seen.add(slack_key)

    # 注意：不再从nodes追加Q0/D0数据，因为模型已求解

    out = pd.DataFrame(rows, columns=['M49_Country_Code','country','year','commodity','Qs','Qd','net_import_t','net_export_t',
                                      'price_global','price_net'])
    if price_df is not None and len(price_df):
        out = out.merge(price_df[['country','year','commodity','price']],
                        on=['country','year','commodity'], how='left')
    if not out.empty:
        mask = out['commodity'].astype(str).str.strip().str.lower().isin({'1', '2', 'nan'})
        out = out[~mask].reset_index(drop=True)
    return out


def generate_cost_summary(var: Dict, unit_cost_data: Dict, baseline_scenario_result: Dict,
                         output_path: str, regions: List[str], commodities: List[str], 
                         years: List[int]) -> None:
    """
    生成减排成本汇总表
    
    从模型求解结果中提取减排量和成本数据，生成汇总CSV文件
    
    Args:
        var: 模型变量字典，包含Qs（供给）、abatement_vars（减排量）、abatement_costs（单位成本）
        unit_cost_data: 单位成本数据 {(region, process): cost_per_tco2eq}
        baseline_scenario_result: BASE情景结果 {'Qs': {...}}
        output_path: 输出CSV文件路径
        regions: 区域列表
        commodities: 商品列表
        years: 年份列表
    """
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("生成减排成本汇总表")
    print("=" * 80)
    
    # 提取变量
    Qs = var.get('Qs', {})
    abatement_vars = var.get('abatement_vars', {})
    abatement_costs_map = var.get('abatement_costs', {})
    
    if not abatement_vars:
        print("⚠️ 未找到减排量变量，跳过成本汇总")
        return
    
    print(f"  - 减排量变量数: {len(abatement_vars)}")
    print(f"  - 单位成本记录数: {len(abatement_costs_map)}")
    
    # 构建结果记录
    records = []
    
    for (region, commodity, year, process), abat_var in abatement_vars.items():
        try:
            # 获取减排量（tCO2eq）
            abatement = float(abat_var.X) if hasattr(abat_var, 'X') else 0.0
            
            # 获取单位成本（USD/tCO2eq）
            unit_cost = abatement_costs_map.get((region, commodity, year, process), 0.0)
            
            # 计算总成本（USD）
            total_cost = abatement * unit_cost
            
            # 只记录有减排量的记录
            if abatement > 1e-6:  # 阈值：1e-6 tCO2eq
                records.append({
                    'region': region,
                    'commodity': commodity,
                    'year': year,
                    'process': process,
                    'abatement_tco2eq': abatement,
                    'unit_cost_usd_per_tco2eq': unit_cost,
                    'total_cost_usd': total_cost
                })
        except Exception as e:
            print(f"    ⚠️ 提取减排数据失败: {region}, {commodity}, {year}, {process}: {e}")
            continue
    
    if not records:
        print("⚠️ 未找到有效的减排记录，生成空文件")
        df = pd.DataFrame(columns=['region', 'commodity', 'year', 'process',
                                  'abatement_tco2eq', 'unit_cost_usd_per_tco2eq', 'total_cost_usd'])
    else:
        df = pd.DataFrame(records)
        
        # 排序：region, year, process, commodity
        df = df.sort_values(['region', 'year', 'process', 'commodity'])
        
        # 统计信息
        total_abatement = df['abatement_tco2eq'].sum()
        total_cost = df['total_cost_usd'].sum()
        
        print(f"\n  ✓ 成本汇总统计:")
        print(f"    - 记录数: {len(df):,}")
        print(f"    - 区域数: {df['region'].nunique()}")
        print(f"    - 商品数: {df['commodity'].nunique()}")
        print(f"    - 过程数: {df['process'].nunique()}")
        print(f"    - 年份范围: {df['year'].min()}-{df['year'].max()}")
        print(f"    - 总减排量: {total_abatement:,.2f} tCO2eq")
        print(f"    - 总成本: ${total_cost:,.2f} USD")
        print(f"    - 平均单位成本: ${total_cost/total_abatement:.2f} USD/tCO2eq" if total_abatement > 0 else "")
    
    # 导出CSV
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ 成本汇总已导出: {output_path}")
    print("=" * 80 + "\n")




