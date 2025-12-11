# -*- coding: utf-8 -*-
"""
gle_emissions_complete.py
==========================
完整的畜牧业排放计算模块
实现从供需模拟的产量到排放的完整链路
包括四个排放过程：
1. Enteric fermentation (CH4)
2. Manure management (CH4, N2O)
3. Manure applied to soils (N2O)
4. Manure left on pasture (N2O)
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)


def _normalize_m49_for_comparison(m49_val) -> str:
    """
    规范化M49代码为统一格式进行比较
    将任何格式转换为 'xxx 格式（单引号+3位数字）
    """
    if pd.isna(m49_val):
        return ''
    s = str(m49_val).strip().lstrip("'\"")
    try:
        return f"'{int(s):03d}"
    except (ValueError, TypeError):
        return f"'{s}"


class LivestockEmissionsCalculator:
    """畜牧业排放计算器"""
    
    def __init__(self, 
                 gle_params_path: str,
                 hist_production_path: str,
                 hist_emissions_path: str,
                 hist_manure_stock_path: str,
                 dict_v3_path: str):
        """
        初始化计算器
        
        Args:
            gle_params_path: GLE_parameters.xlsx路径
            hist_production_path: Production_Crops_Livestock_E_All_Data_NOFLAG.csv路径
            hist_emissions_path: Emissions_livestock_dairy_split.csv路径（已拆分dairy/non-dairy）
            hist_manure_stock_path: Environment_LivestockManure_with_ratio.csv路径
            dict_v3_path: dict_v3.xlsx路径
        """
        self.gle_params_path = gle_params_path
        self.hist_production_path = hist_production_path
        self.hist_emissions_path = hist_emissions_path
        self.hist_manure_stock_path = hist_manure_stock_path
        self.dict_v3_path = dict_v3_path
        
        # 计算结果存储（用于导出到production_summary）
        self._computed_slaughter = []  # List[pd.DataFrame] 存储计算的slaughter数据
        self._computed_stock = []      # List[pd.DataFrame] 存储计算的stock数据
        self._computed_carcass_yield = []  # List[pd.DataFrame] 存储carcass_yield数据 (meat类)
        self._computed_dairy_yield = []   # List[pd.DataFrame] 存储dairy_yield数据 (milk/egg类)
        
        # 加载数据
        self._load_mappings()
        self._load_parameters()
        self._load_historical_data()
        
    def _load_mappings(self):
        """加载dict_v3中的映射关系"""
        emis_item = pd.read_excel(self.dict_v3_path, sheet_name='Emis_item')
        
        # ✅ 保存dict_v3字典供后续使用（特别是feed_requirement创建）
        self.dict_v3 = {'Emis_item': emis_item}
        
        # 加载区域映射关系
        try:
            region_df = pd.read_excel(self.dict_v3_path, sheet_name='region')
            self.m49_to_region = {}
            for _, row in region_df.iterrows():
                m49_code = row.get('M49_Country_Code')
                region = row.get('Region_agg2')
                if pd.notna(m49_code) and pd.notna(region):
                    # ✅ 标准化M49为'xxx格式（单引号+3位数字）
                    m49_str = str(m49_code).strip().lstrip("'\"")
                    if m49_str.isdigit():
                        m49_str = f"'{m49_str.zfill(3)}"
                    self.m49_to_region[m49_str] = str(region)
            logger.info(f"✅ 加载了 {len(self.m49_to_region)} 个国家的区域映射")
            
            # 筛选出有效国家（Region_label_new != 'no'）
            try:
                region_df_full = pd.read_excel(self.dict_v3_path, sheet_name='region')
                valid_countries = region_df_full[
                    region_df_full['Region_label_new'].astype(str).str.lower() != 'no'
                ]['M49_Country_Code'].dropna().unique()
                # ✅ 标准化M49为'xxx格式（单引号+3位数字）
                self.valid_m49_codes = set()
                for m49 in valid_countries:
                    try:
                        m49_str = str(m49).strip().lstrip("'\"")
                        if m49_str.isdigit():
                            m49_str = f"'{m49_str.zfill(3)}"
                        self.valid_m49_codes.add(m49_str)
                    except (ValueError, TypeError):
                        continue
                logger.info(f"✅ 筛选出 {len(self.valid_m49_codes)} 个有效国家（Region_label_new != 'no'）")
            except Exception as e2:
                logger.warning(f"WARNING: failed to filter valid years: {e2}")
                self.valid_m49_codes = set(self.m49_to_region.keys())
        except Exception as e:
            logger.warning(f"WARNING: failed to load process mapping: {e}")
            self.m49_to_region = {}
            self.valid_m49_codes = set()
        
        # 筛选livestock相关的排放过程
        livestock_processes = [
            'Enteric fermentation',
            'Manure management', 
            'Manure applied to soils',
            'Manure left on pasture'
        ]
        
        self.emis_mappings = emis_item[emis_item['Process'].isin(livestock_processes)].copy()
        
        # 构建映射字典
        self.item_production_map = {}
        self.item_yield_map = {}
        self.item_slaughtered_map = {}
        self.item_slaughtered_ratio_map = {}
        self.item_stock_map = {}
        
        # ✅ 关键：反向映射 (Item_Production_Map → Item_Emis)
        self.production_to_emis = {}  # 从生产数据名称反查Item_Emis
        
        for _, row in self.emis_mappings.iterrows():
            # dict_v3中的列名是 'Item_Emis' - 这是模型使用的商品名
            item_emis = row['Item_Emis']
            
            # Production mapping - 建立正向和反向映射
            if pd.notna(row.get('Item_Production_Map')):
                prod_item = row['Item_Production_Map']
                self.item_production_map[item_emis] = {
                    'Item': prod_item,
                    'Element': row.get('Item_Production_Element', 'Production')
                }
                # 反向映射：FAOSTAT名称 → 模型名称
                self.production_to_emis[prod_item] = item_emis
            
            # Yield mapping
            if pd.notna(row.get('Item_Yield_Map')):
                self.item_yield_map[item_emis] = {
                    'Item': row['Item_Yield_Map'],
                    'Element': row.get('Item_Yield_Element', 'Yield')
                }
            
            # Slaughtered mapping
            if pd.notna(row.get('Item_Slaughtered_Map')):
                self.item_slaughtered_map[item_emis] = {
                    'Item': row['Item_Slaughtered_Map'],
                    'Element': row.get('Item_Slaughtered_Element', 'Producing Animals/Slaughtered')
                }
            
            # Slaughtered ratio mapping
            if pd.notna(row.get('Item_SlaughteredRatio_Map')):
                self.item_slaughtered_ratio_map[item_emis] = {
                    'Item': row['Item_SlaughteredRatio_Map'],
                    'Element': row.get('Item_SlaughteredRatio_Element', 'Producing/Slaughtered ratio')
                }
            
            # Stock mapping
            if pd.notna(row.get('Item_Stock_Map')):
                self.item_stock_map[item_emis] = {
                    'Item': row['Item_Stock_Map'],
                    'Element': row.get('Item_Stock_Element', 'Stocks')
                }
    
    def _load_parameters(self):
        """加载GLE参数表"""
        self.gle_params = pd.read_excel(self.gle_params_path)
        
        # 检查参数表的列名（特别是年份列）
        year_cols = [col for col in self.gle_params.columns if str(col).startswith('Y') or str(col).startswith('2')]
        logger.info(f"[INFO] GLE_parameters年份列: {year_cols[:5]}...{year_cols[-3:] if len(year_cols) > 5 else year_cols}")
        
        # 筛选Select=1的数据
        self.gle_params = self.gle_params[self.gle_params['Select'] == 1].copy()
        # ✅ 标准化M49为'xxx格式（单引号+3位数字）
        if 'M49_Country_Code' in self.gle_params.columns:
            def _norm_m49(val):
                s = str(val).strip().lstrip("'\"")
                return f"'{s.zfill(3)}" if s.isdigit() else val
            self.gle_params['M49_Country_Code'] = self.gle_params['M49_Country_Code'].apply(_norm_m49)
        
    def _load_historical_data(self):
        """加载历史数据"""
        # 生产数据
        self.hist_production = pd.read_csv(self.hist_production_path, encoding='latin1')
        
        # 排放数据
        self.hist_emissions = pd.read_csv(self.hist_emissions_path, encoding='latin1')
        
        # 粪便和存栏数据
        self.hist_manure_stock = pd.read_csv(self.hist_manure_stock_path)
        
        # ✅ 修复：标准化M49代码为'xxx格式（单引号+3位数字）
        # 确保与S4_0_main.py传入的格式一致
        def _normalize_m49(s):
            """Normalize M49 to 'xxx format (4→'004', '51'→'051')"""
            def normalize_single(val):
                if pd.isna(val):
                    return None
                s_str = str(val).strip().lstrip("'\"")
                if s_str.isdigit():
                    return f"'{s_str.zfill(3)}"  # ✅ 'xxx格式
                return val
            return s.apply(normalize_single)

        # hist_production uses 'Area Code (M49)'
        if 'Area Code (M49)' in self.hist_production.columns:
            self.hist_production['Area Code (M49)'] = _normalize_m49(self.hist_production['Area Code (M49)'])

        if 'M49_Country_Code' in self.hist_emissions.columns:
            self.hist_emissions['M49_Country_Code'] = _normalize_m49(self.hist_emissions['M49_Country_Code'])

        if 'M49_Country_Code' in self.hist_manure_stock.columns:
            self.hist_manure_stock['M49_Country_Code'] = _normalize_m49(self.hist_manure_stock['M49_Country_Code'])
        
    def _load_historical_emissions(self, years: List[int]) -> Dict[str, Optional[pd.DataFrame]]:
        """
        ✅ 修改：从Emissions_livestock_dairy_split.csv读取历史排放数据
        新文件已在源头完成Buffalo/Camel/Sheep/Goats的dairy/non-dairy拆分
        
        Args:
            years: 需要读取的年份列表
            
        Returns:
            按排放过程分组的历史排放数据
        """
        logger.info(f"\n[DEBUG] Loading historical emissions for years: {years}")
        
        # ✅ 修正：历史排放CSV的Element包含GHG类型
        # 需要分别读取CH4和N2O的数据
        # 使用dict_v3标准的Process名称（带空格）
        element_mappings = {
            # Enteric fermentation只有CH4
            'Enteric fermentation (Emissions CH4)': ('Enteric fermentation', 'CH4'),
            # Manure management有CH4和N2O
            'Manure management (Emissions CH4)': ('Manure management', 'CH4'),
            'Manure management (Emissions N2O)': ('Manure management', 'N2O'),
            # Manure applied和left on pasture只有N2O
            'Manure applied to soils (Emissions N2O)': ('Manure applied to soils', 'N2O'),
            'Manure left on pasture (Emissions N2O)': ('Manure left on pasture', 'N2O'),
        }
        
        # 结果容器：{process: {ghg_type: [dfs]}}
        # 使用标准Process名称
        results_by_ghg = {
            'Enteric fermentation': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure management': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure applied to soils': {'CH4': [], 'N2O': [], 'CO2': []},
            'Manure left on pasture': {'CH4': [], 'N2O': [], 'CO2': []},
        }
        
        # 筛选Select=1的排放数据
        df = self.hist_emissions.copy()
        if 'Select' in df.columns:
            df = df[df['Select'] == 1]
        
        # ✅ 过滤掉Region_label_new='no'的国家
        if 'M49_Country_Code' in df.columns:
            region_df = pd.read_excel(self.dict_v3_path, sheet_name='region')
            # ✅ 标准化M49代码为'xxx格式（单引号+3位数字）
            # dict_v3可能有前导零，CSV可能是整数，统一标准化
            def _normalize_m49_local(s):
                def normalize_single(val):
                    if pd.isna(val):
                        return None
                    s_str = str(val).strip().lstrip("'\"")
                    if s_str.isdigit():
                        return f"'{s_str.zfill(3)}"  # ✅ 'xxx格式
                    return val
                return s.apply(normalize_single)
            
            region_df['M49_Country_Code'] = _normalize_m49_local(region_df['M49_Country_Code'])
            # ✅ df (hist_emissions) 已在_load_historical_data中标准化，无需重复
            
            valid_m49_codes = region_df[region_df['Region_label_new'] != 'no']['M49_Country_Code'].unique()
            rows_before = len(df)
            df = df[df['M49_Country_Code'].isin(valid_m49_codes)].copy()
            rows_after = len(df)
            if rows_before > rows_after:
                logger.info(f"[INFO] 历史排放数据过滤掉 {rows_before - rows_after} 行无效国家")
        
        # 筛选畜牧业相关排放过程
        if 'Element' not in df.columns:
            logger.info("WARNING: historical emissions are missing the Element column; cannot read")
            return {k: None for k in results_by_ghg.keys()}
        
        df_livestock = df[df['Element'].isin(element_mappings.keys())].copy()
        
        if df_livestock.empty:
            logger.info("WARNING: historical emissions block not found in the source file")
            logger.info(f"  可用的Element: {df['Element'].unique()[:10]}")
            return {k: None for k in results_by_ghg.keys()}
        
        logger.info(f"[DEBUG] Found {len(df_livestock)} rows of livestock emissions (before Item filter)")
        
        # ✅ 关键修复：根据dict_v3的Item_Emis过滤Item
        # 只保留dict_v3中定义的Item，过滤掉'Sheep and Goats'等不存在的项
        if 'Item' in df_livestock.columns and hasattr(self, 'emis_mappings') and not self.emis_mappings.empty:
            valid_items = set(self.emis_mappings['Item_Emis'].dropna().unique())
            # 修正：历史CSV中Item是FAOSTAT名称，需要映射到Item_Emis
            # 但历史CSV可能已经是Item_Emis格式，先尝试直接过滤
            items_before = len(df_livestock)
            items_unique_before = df_livestock['Item'].nunique()
            
            df_livestock = df_livestock[df_livestock['Item'].isin(valid_items)].copy()
            
            items_after = len(df_livestock)
            items_unique_after = df_livestock['Item'].nunique()
            
            if items_before > items_after:
                invalid_items = set(df[df['Element'].isin(element_mappings.keys())]['Item'].unique()) - valid_items
                logger.info(f"[INFO] ✅ 按dict_v3过滤Item: {items_before} → {items_after} 行 ({items_unique_before} → {items_unique_after} 种商品)")
                if invalid_items:
                    logger.info(f"  ⚠️ 过滤掉的无效Item ({len(invalid_items)}种): {list(invalid_items)[:10]}")
        
        logger.info(f"[DEBUG] Found {len(df_livestock)} rows of livestock emissions (after Item filter)")
        
        # 转换为长格式
        year_cols = [c for c in df_livestock.columns if isinstance(c, str) and c.startswith('Y')]
        id_cols = [c for c in df_livestock.columns if c not in year_cols]
        
        df_long = df_livestock.melt(id_vars=id_cols, value_vars=year_cols,
                                     var_name='Year', value_name='Value')
        df_long['Year'] = df_long['Year'].str.lstrip('Y').astype(int)
        df_long = df_long[df_long['Year'].isin(years)]
        
        logger.info(f"[DEBUG] After filtering years {years}, got {len(df_long)} rows")
        
        # 按排放过程和GHG类型分组
        for element_name, (process, ghg_type) in element_mappings.items():
            process_df = df_long[df_long['Element'] == element_name].copy()
            
            if process_df.empty:
                continue
            
            logger.info(f"[DEBUG] {element_name}: {len(process_df)} rows")
            
            # 标准化列名
            process_df = process_df.rename(columns={
                'M49_Country_Code': 'M49_Country_Code',
                'Item': 'Item',  # ✅ 保持Item列名不变（CSV中是FAOSTAT名称）
                'Year': 'year',
                'Value': f'{ghg_type}_kt'
            })
            
            # 添加其他GHG列（填0）
            if ghg_type == 'CH4':
                process_df['N2O_kt'] = 0.0
                process_df['CO2_kt'] = 0.0
            elif ghg_type == 'N2O':
                process_df['CH4_kt'] = 0.0
                process_df['CO2_kt'] = 0.0
            else:  # CO2
                process_df['CH4_kt'] = 0.0
                process_df['N2O_kt'] = 0.0
            
            results_by_ghg[process][ghg_type].append(process_df[[
                'M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt'
            ]])
        
        # 合并各排放过程的数据（按GHG类型合并后再按过程合并）
        final_results = {}
        for process, ghg_dict in results_by_ghg.items():
            all_ghg_dfs = []
            for ghg_type, dfs in ghg_dict.items():
                if dfs:
                    all_ghg_dfs.extend(dfs)
            
            if all_ghg_dfs:
                # 合并同一过程的不同GHG数据
                combined = pd.concat(all_ghg_dfs, ignore_index=True)
                # 按国家、商品、年份分组，汇总CH4/N2O/CO2
                result_df = combined.groupby(
                    ['M49_Country_Code', 'Item', 'year'], as_index=False
                ).agg({
                    'CH4_kt': 'sum',
                    'N2O_kt': 'sum',
                    'CO2_kt': 'sum'
                })
                # ✅ Item列已经是FAOSTAT名称，不需要重命名
                result_df['process'] = process  # 添加标准process名称
                final_results[process] = result_df
                logger.info(f"[DEBUG] {process}: final {len(final_results[process])} rows")
            else:
                final_results[process] = None
        
        return final_results
        
    def get_parameter_value(self, 
                           m49_code: int,
                           item: str,
                           process: str,
                           param_name: str,
                           year: int,
                           base_year: int = 2020) -> float:
        """
        获取参数值
        
        Args:
            m49_code: M49国家代码
            item: 物种名称
            process: 排放过程
            param_name: 参数名称
            year: 年份
            base_year: 基准年份（用于未来BASE情景）
            
        Returns:
            参数值
        """
        # 筛选参数
        mask = (
            (self.gle_params['M49_Country_Code'] == m49_code) &
            (self.gle_params['Item'] == item) &
            (self.gle_params['Process'] == process) &
            (self.gle_params['paramName'] == param_name)
        )
        
        param_df = self.gle_params[mask]
        
        if param_df.empty:
            return 0.0
        
        # 历史时期直接读取
        year_col = f'Y{year}'
        if year <= base_year and year_col in param_df.columns:
            value = param_df[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # 未来BASE情景用2020年的值
        base_col = f'Y{base_year}'
        if base_col in param_df.columns:
            value = param_df[base_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        return 0.0
    
    def get_parameter_value_by_species(self,
                                      m49_code: int,
                                      item: str,
                                      process: str,
                                      param_name: str,
                                      species: str,
                                      year: int,
                                      base_year: int = 2020) -> float:
        """
        获取按物种（CH4/N2O/CO2）区分的参数值
        
        Args:
            m49_code: M49国家代码
            item: 物种名称
            process: 排放过程
            param_name: 参数名称
            species: 气体类型（'CH4', 'N2O', 'CO2'）
            year: 年份
            base_year: 基准年份
            
        Returns:
            参数值
        """
        # 筛选参数（包括物种）
        mask = (
            (self.gle_params['M49_Country_Code'] == m49_code) &
            (self.gle_params['Item'] == item) &
            (self.gle_params['Process'] == process) &
            (self.gle_params['paramName'] == param_name)
        )
        
        # ✅ 修复：GLE_parameters.xlsx使用paramMMS列区分CH4/N2O，而非Species列
        if 'paramMMS' in self.gle_params.columns:
            mask = mask & (self.gle_params['paramMMS'] == species)
        elif 'Species' in self.gle_params.columns:
            mask = mask & (self.gle_params['Species'] == species)
        
        param_df = self.gle_params[mask]
        
        if param_df.empty:
            return 0.0
        
        # 历史时期直接读取
        year_col = f'Y{year}'
        if year <= base_year and year_col in param_df.columns:
            value = param_df[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # 未来BASE情景用2020年的值
        base_col = f'Y{base_year}'
        if base_col in param_df.columns:
            value = param_df[base_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        return 0.0
    
    def calculate_milk_animals(self,
                               production_df: pd.DataFrame,
                               year: int) -> pd.DataFrame:
        """
        根据奶类产量计算产奶动物数量（包括牛奶、羊奶、骆驼奶等）
        
        Args:
            production_df: 包含产量的DataFrame，Commodity列包含FAOSTAT Item_Production_Map名称
                          (如'Raw milk of cattle'，非'Cattle, dairy')
            year: 年份
            
        Returns:
            包含产奶动物数量的DataFrame
        """
        logger.info(f"\n=== calculate_milk_animals 调试 (year={year}) ===")
        logger.info(f"[DEBUG] 输入production_df: {len(production_df)} 行")
        if not production_df.empty:
            logger.info(f"[DEBUG] 列名: {list(production_df.columns)}")
            logger.info(f"[DEBUG] 年份分布: {production_df['year'].unique() if 'year' in production_df.columns else 'N/A'}")
        
        # WARNING: filter by year before processing
        if 'year' in production_df.columns:
            production_df = production_df[production_df['year'] == year].copy()
            logger.info(f"[DEBUG] 筛选year={year}后: {len(production_df)} 行")
            if production_df.empty:
                logger.warning(f"❌ WARNING: no data after filtering year={year}; skipping")
                return pd.DataFrame()
        else:
            logger.warning(f"⚠️ WARNING: production_df没有year列！")
        
        # 确定商品列名
        commodity_col = 'Commodity' if 'Commodity' in production_df.columns else (
            'commodity' if 'commodity' in production_df.columns else 'Item'
        )
        
        if commodity_col not in production_df.columns:
            logger.info("WARNING: commodity column not found")
            return pd.DataFrame()
        
        # 识别所有dairy商品（从映射中）
        # Dairy包括：奶类(buffalo/camel/cattle/goats/sheep, dairy)和产蛋鸡(Chickens, layers)
        dairy_production_items = []  # FAOSTAT生产数据名称
        for item_emis, mapping in self.item_production_map.items():
            # ✅ 修复：检查是否为奶类或蛋类商品，排除non-dairy
            item_lower = item_emis.lower()
            if ('dairy' in item_lower and 'non-dairy' not in item_lower) or 'layers' in item_lower:
                fao_item = mapping['Item']  # Item_Production_Map中的名称
                dairy_production_items.append(fao_item)
        
        logger.info(f"识别的dairy FAOSTAT商品 ({len(dairy_production_items)} 种): {dairy_production_items}")
        
        # 检查production_df中有哪些dairy商品（FAOSTAT名称）
        available_items = production_df[commodity_col].unique()
        matching_items = set(available_items) & set(dairy_production_items)
        logger.info(f"production_df中的dairy商品: {list(matching_items)}")
        
        if not matching_items:
            logger.warning("WARNING: production_df has no dairy items")
            return pd.DataFrame()
        
        results = []
        
        # 遍历每个国家和dairy商品（只计算有效国家）
        for m49_code in production_df['M49_Country_Code'].unique():
            # ✅ 修复：规范化M49格式后再比较
            if hasattr(self, 'valid_m49_codes') and self.valid_m49_codes:
                m49_normalized = _normalize_m49_for_comparison(m49_code)
                if m49_normalized not in self.valid_m49_codes:
                    continue
            
            country_prod = production_df[production_df['M49_Country_Code'] == m49_code]
            
            # 处理每个dairy商品（FAOSTAT Item_Production_Map名称）
            for fao_item in matching_items:
                item_data = country_prod[country_prod[commodity_col] == fao_item]
                
                if item_data.empty:
                    continue
                
                production = item_data['production_t'].sum()
                
                if production <= 0:
                    continue
                
                # 反向查找Item_Emis
                if fao_item not in self.production_to_emis:
                    logger.info(f"  WARNING: {fao_item} not found in production mapping")
                    continue
                
                item_emis = self.production_to_emis[fao_item]
                
                # 获取yield (产奶率或产蛋率) - 使用Item_Yield_Map
                if item_emis not in self.item_yield_map:
                    logger.info(f"  WARNING: {item_emis} missing yield mapping")
                    continue
                
                yield_fao_item = self.item_yield_map[item_emis]['Item']
                yield_element = self.item_yield_map[item_emis].get('Element', 'Yield')
                yield_val = self.get_yield_value(m49_code, yield_fao_item, yield_element, year)
                
                # ✅ 关键修复：当yield缺失时，使用合理默认值（基于典型dairy品种产奶率）
                if yield_val <= 0:
                    # 默认dairy品种产奶率 (t/head/year)，基于FAO统计平均值
                    default_yields = {
                        'Buffalo, dairy': 1.5,    # 水牛奶：1.5吨/头/年
                        'Camel, dairy': 1.2,      # 骆驼奶：1.2吨/头/年
                        'Goats, dairy': 0.5,      # 羊奶：0.5吨/头/年
                        'Sheep, dairy': 0.4,      # 绵羊奶：0.4吨/头/年
                        'Cattle, dairy': 6.0,     # 奶牛：6吨/头/年
                        'Chickens, layers': 0.015 # 产蛋鸡：15kg/头/年
                    }
                    
                    yield_val = default_yields.get(item_emis, 1.0)  # 通用默认值1吨/头/年
                    logger.info(f"  [DAIRY_FIX] {item_emis} M49={m49_code} yield缺失，使用默认值 {yield_val:.3f} t/head/year")
                
                if yield_val > 0:
                    # 计算产奶/产蛋动物数
                    producing_animals = production / yield_val
                    
                    # ✅ 修复：Chickens, layers应该使用laying列而不是milk_animals
                    result_dict = {
                        'M49_Country_Code': m49_code,
                        'Item': fao_item,  # 使用FAOSTAT Item名称
                        'Item_Emis': item_emis,  # 模型商品名
                        'year': year,
                        'production_t': production,
                        'yield_t_per_head': yield_val,
                    }
                    
                    # 根据商品类型设置不同的列名
                    if 'layer' in item_emis.lower():
                        result_dict['laying'] = producing_animals
                    else:
                        result_dict['milk_animals'] = producing_animals
                    
                    results.append(result_dict)
                else:
                    logger.warning(f"  WARNING: {item_emis} ({yield_fao_item}) M49={m49_code} yield仍为0，跳过")
        
        logger.info(f"计算得到 {len(results)} 个dairy动物流量记录")
        return pd.DataFrame(results)
    
    def calculate_meat_animals(self,
                               production_df: pd.DataFrame,
                               year: int) -> pd.DataFrame:
        """
        根据肉类产量计算屠宰动物数量
        
        Args:
            production_df: 包含产量的DataFrame，Commodity列包含FAOSTAT Item_Production_Map名称
                          (如'Meat of pig'，非'Swine')
            year: 年份
            
        Returns:
            包含屠宰动物数量的DataFrame
        """
        logger.info(f"\n=== calculate_meat_animals 调试 (year={year}) ===")
        logger.info(f"[DEBUG] 输入production_df: {len(production_df)} 行")
        if not production_df.empty:
            logger.info(f"[DEBUG] 列名: {list(production_df.columns)}")
            logger.info(f"[DEBUG] 年份分布: {production_df['year'].unique() if 'year' in production_df.columns else 'N/A'}")
        
        # WARNING: filter by year before processing
        if 'year' in production_df.columns:
            production_df = production_df[production_df['year'] == year].copy()
            logger.info(f"[DEBUG] 筛选year={year}后: {len(production_df)} 行")
            if production_df.empty:
                logger.warning(f"❌ WARNING: no data after filtering year={year}; skipping")
                return pd.DataFrame()
        else:
            logger.warning(f"⚠️ WARNING: production_df没有year列！")
        
        # 确定商品列名
        commodity_col = 'Commodity' if 'Commodity' in production_df.columns else (
            'commodity' if 'commodity' in production_df.columns else 'Item'
        )
        
        if commodity_col not in production_df.columns:
            logger.info("WARNING: commodity column not found")
            return pd.DataFrame()
        
        # 识别所有肉类商品（从反向映射中）
        # 肉类商品的特征：Item_Cat2='Meat' 或 Item_Emis中包含'non-dairy'
        meat_production_items = []  # FAOSTAT生产数据名称
        for item_emis, mapping in self.item_production_map.items():
            # 检查商品名称中的关键词
            item_lower = item_emis.lower()
            if ('non-dairy' in item_lower or 
                'broiler' in item_lower or
                item_emis in ['Asses', 'Ducks', 'Horses', 'Llamas', 'Mules and hinnies', 
                             'Swine', 'Turkeys']):
                fao_item = mapping['Item']  # Item_Production_Map中的名称
                meat_production_items.append(fao_item)
        
        logger.info(f"识别的肉类FAOSTAT商品 ({len(meat_production_items)} 种): {meat_production_items}")
        
        # 检查production_df中有哪些肉类商品（FAOSTAT名称）
        available_items = production_df[commodity_col].unique()
        matching_items = set(available_items) & set(meat_production_items)
        logger.info(f"production_df中的肉类商品: {list(matching_items)}")
        
        if not matching_items:
            logger.info("WARNING: production_df has no carcass items")
            return pd.DataFrame()
        
        results = []
        
        # 遍历每个国家和肉类商品（只计算有效国家）
        all_m49_codes = production_df['M49_Country_Code'].unique()
        logger.debug(f"[DEBUG] production_df中的M49代码 ({len(all_m49_codes)}个): {all_m49_codes[:10]}")
        
        if hasattr(self, 'valid_m49_codes'):
            logger.debug(f"[DEBUG] valid_m49_codes前10个: {list(self.valid_m49_codes)[:10]}")
        
        for m49_code in all_m49_codes:
            # ✅ 修复：规范化M49格式后再比较
            if hasattr(self, 'valid_m49_codes') and self.valid_m49_codes:
                m49_normalized = _normalize_m49_for_comparison(m49_code)
                if m49_normalized not in self.valid_m49_codes:
                    continue
            
            country_prod = production_df[production_df['M49_Country_Code'] == m49_code]
            
            # 处理每个肉类商品（FAOSTAT Item_Production_Map名称）
            for fao_item in matching_items:
                item_data = country_prod[country_prod[commodity_col] == fao_item]
                
                if item_data.empty:
                    continue
                
                production = item_data['production_t'].sum()
                
                if production <= 0:
                    continue
                
                # 反向查找Item_Emis
                if fao_item not in self.production_to_emis:
                    logger.info(f"  WARNING: {fao_item} not found in production mapping")
                    continue
                
                item_emis = self.production_to_emis[fao_item]
                
                # 获取胴体重(carcass weight) - 使用Item_Yield_Map
                if item_emis not in self.item_yield_map:
                    logger.info(f"  WARNING: {item_emis} missing yield mapping (carcass)")
                    continue
                
                carcass_fao_item = self.item_yield_map[item_emis]['Item']
                carcass_element = self.item_yield_map[item_emis].get('Element', 'Yield')
                carcass_weight = self.get_carcass_weight(m49_code, carcass_fao_item, carcass_element, year)
                
                if carcass_weight > 0:
                    # 屠宰数 = 肉产量 / 胴体重
                    slaughtered = production / carcass_weight
                    
                    results.append({
                        'M49_Country_Code': m49_code,
                        'Item': fao_item,  # 使用FAOSTAT Item名称
                        'Item_Emis': item_emis,  # 模型商品名
                        'year': year,
                        'production_t': production,
                        'carcass_weight_t': carcass_weight,
                        'slaughtered': slaughtered
                    })
                else:
                    logger.warning(f"  WARNING: {item_emis} ({carcass_fao_item}) M49={m49_code} missing yield value")
        
        logger.info(f"计算得到 {len(results)} 个肉类动物流量记录")
        return pd.DataFrame(results)
    
    def calculate_egg_animals(self,
                             production_df: pd.DataFrame,
                             year: int) -> pd.DataFrame:
        """
        根据蛋类产量计算产蛋动物数量
        
        Args:
            production_df: 包含蛋类产量的DataFrame，Commodity列包含FAOSTAT Item_Production_Map名称
                          (如'Eggs from hens'，非'Chickens, eggs')
            year: 年份
            
        Returns:
            包含产蛋动物数量的DataFrame
        """
        # 调试信息
        logger.info(f"\n=== calculate_egg_animals 调试 (year={year}) ===")
        
        # 确定商品列名
        commodity_col = 'Commodity' if 'Commodity' in production_df.columns else (
            'commodity' if 'commodity' in production_df.columns else 'Item'
        )
        
        # 蛋类商品（FAOSTAT生产数据名称）
        egg_production_items = []
        for item_emis, mapping in self.item_production_map.items():
            if 'egg' in item_emis.lower():
                fao_item = mapping['Item']
                egg_production_items.append(fao_item)
        
        logger.info(f"识别的蛋类FAOSTAT商品: {egg_production_items}")
        
        # 检查production_df中有哪些蛋类商品
        available_items = production_df[commodity_col].unique()
        matching_items = set(available_items) & set(egg_production_items)
        logger.info(f"production_df中的蛋类商品: {list(matching_items)}")
        
        results = []
        
        # 只计算有效国家
        for m49_code in production_df['M49_Country_Code'].unique():
            # ✅ 修复：规范化M49格式后再比较
            if hasattr(self, 'valid_m49_codes') and self.valid_m49_codes:
                m49_normalized = _normalize_m49_for_comparison(m49_code)
                if m49_normalized not in self.valid_m49_codes:
                    continue
            
            country_prod = production_df[production_df['M49_Country_Code'] == m49_code]
            
            # 遍历所有蛋类商品（FAOSTAT Item_Production_Map名称）
            for fao_item in matching_items:
                egg_data = country_prod[country_prod[commodity_col] == fao_item]
                
                if egg_data.empty:
                    continue
                
                egg_production = egg_data['production_t'].sum()
                
                if egg_production <= 0:
                    continue
                
                # 反向查找Item_Emis
                if fao_item not in self.production_to_emis:
                    logger.info(f"  WARNING: {fao_item} not found in production mapping")
                    continue
                
                item_emis = self.production_to_emis[fao_item]
                
                # 获取产蛋率(laying rate) - 使用Item_Yield_Map
                if item_emis not in self.item_yield_map:
                    logger.info(f"  WARNING: {item_emis} missing yield mapping (layers)")
                    continue
                
                laying_fao_item = self.item_yield_map[item_emis]['Item']
                laying_element = self.item_yield_map[item_emis].get('Element', 'Yield')
                laying_rate = self.get_laying_rate(m49_code, laying_fao_item, laying_element, year)
                
                if laying_rate > 0:
                    # 产蛋动物数 = 蛋产量 / 产蛋率
                    laying = egg_production / laying_rate
                    
                    results.append({
                        'M49_Country_Code': m49_code,
                        'Item': fao_item,
                        'Item_Emis': item_emis,
                        'year': year,
                        'production_t': egg_production,
                        'laying_rate_t_per_head': laying_rate,
                        'laying': laying
                    })
                else:
                    logger.warning(f"  WARNING: {item_emis} ({laying_fao_item}) has no production data")
        
        logger.info(f"计算得到 {len(results)} 个蛋类动物流量记录")
        return pd.DataFrame(results)
    
    def get_carcass_weight(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        获取胴体重（肉类yield）
        优先级：特定国家 -> 同区域平均值 -> 全球平均值
        
        ✅ 重要：FAOSTAT中carcass weight的原始单位是kg/head，
        本方法返回的是t/head（吨/头），已进行单位转换
        
        Args:
            m49_code: M49国家代码
            item: Item_Yield_Map对应的Item名称（FAOSTAT名称）
            element: Item_Yield_Element对应的Element名称
            year: 年份
            
        Returns:
            float: 胴体重 (t/head)
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        # ✅ FAOSTAT carcass weight 单位是 kg/head，需要转换为 t/head
        KG_TO_TONNE = 1.0 / 1000.0
        
        # 1. 尝试获取特定国家的数据
        hist_yield = self.hist_production[
            (self.hist_production['Area Code (M49)'] == m49_code) &
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not hist_yield.empty and year_col in hist_yield.columns:
            value = hist_yield[year_col].iloc[0]
            if pd.notna(value) and value > 0:
                # ✅ 单位转换: kg/head → t/head
                return float(value) * KG_TO_TONNE
        
        # 2. 如果特定国家缺失，尝试使用同区域平均值
        region = self.m49_to_region.get(m49_code)
        if region:
            # 找出同区域的所有国家M49代码
            region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
            
            if region_m49_codes:
                region_data = self.hist_production[
                    (self.hist_production['Area Code (M49)'].isin(region_m49_codes)) &
                    (self.hist_production['Item'] == item) &
                    (self.hist_production['Element'] == element)
                ]
                
                if not region_data.empty and year_col in region_data.columns:
                    valid_values = region_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
                    valid_values = valid_values[valid_values > 0]
                    
                    if len(valid_values) > 0:
                        region_avg = valid_values.mean()
                        if region_avg > 0:
                            # ✅ 单位转换: kg/head → t/head
                            region_avg_t = region_avg * KG_TO_TONNE
                            logger.debug(f"  ℹ️ {item} M49={m49_code} 使用{region}区域平均胴体重: {region_avg_t:.4f} t/head (原值{region_avg:.1f} kg/head, 基于{len(valid_values)}个国家)")
                            return float(region_avg_t)
        
        # 3. 如果区域也没有数据，使用全球平均值
        global_data = self.hist_production[
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not global_data.empty and year_col in global_data.columns:
            valid_values = global_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            valid_values = valid_values[valid_values > 0]
            
            if len(valid_values) > 0:
                global_avg = valid_values.mean()
                if global_avg > 0:
                    # ✅ 单位转换: kg/head → t/head
                    global_avg_t = global_avg * KG_TO_TONNE
                    logger.debug(f"  ℹ️ {item} M49={m49_code} 使用全球平均胴体重: {global_avg_t:.4f} t/head (原值{global_avg:.1f} kg/head, 基于{len(valid_values)}个国家)")
                    return float(global_avg_t)
        
        return 0.0
    
    def get_laying_rate(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        获取产蛋率
        优先级：特定国家 -> 同区域平均值 -> 全球平均值
        
        Args:
            m49_code: M49国家代码
            item: Item_Yield_Map对应的Item名称（FAOSTAT名称）
            element: Item_Yield_Element对应的Element名称
            year: 年份
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        # 1. 尝试获取特定国家的数据
        hist_yield = self.hist_production[
            (self.hist_production['Area Code (M49)'] == m49_code) &
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not hist_yield.empty and year_col in hist_yield.columns:
            value = hist_yield[year_col].iloc[0]
            if pd.notna(value) and value > 0:
                return float(value)
        
        # 2. 如果特定国家缺失，尝试使用同区域平均值
        region = self.m49_to_region.get(m49_code)
        if region:
            region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
            
            if region_m49_codes:
                region_data = self.hist_production[
                    (self.hist_production['Area Code (M49)'].isin(region_m49_codes)) &
                    (self.hist_production['Item'] == item) &
                    (self.hist_production['Element'] == element)
                ]
                
                if not region_data.empty and year_col in region_data.columns:
                    valid_values = region_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
                    valid_values = valid_values[valid_values > 0]
                    
                    if len(valid_values) > 0:
                        region_avg = valid_values.mean()
                        if region_avg > 0:
                            logger.debug(f"  ℹ️ {item} M49={m49_code} 使用{region}区域平均产蛋率: {region_avg:.4f} t/head (基于{len(valid_values)}个国家)")
                            return float(region_avg)
        
        # 3. 如果区域也没有数据，使用全球平均值
        global_data = self.hist_production[
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not global_data.empty and year_col in global_data.columns:
            valid_values = global_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            valid_values = valid_values[valid_values > 0]
            
            if len(valid_values) > 0:
                global_avg = valid_values.mean()
                if global_avg > 0:
                    logger.debug(f"  ℹ️ {item} M49={m49_code} 使用全球平均产蛋率: {global_avg:.4f} t/head (基于{len(valid_values)}个国家)")
                    return float(global_avg)
        
        return 0.0
    
    def get_yield_value(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        获取产率值 (milk/egg yield)
        优先级：特定国家 -> 同区域平均值 -> 全球平均值
        
        ✅ 重要：FAOSTAT中milk yield的原始单位通常是hg/head或kg/head，
        本方法返回的是t/head（吨/头），已进行单位转换
        
        Args:
            m49_code: M49国家代码
            item: Item_Yield_Map对应的Item名称（FAOSTAT名称）
            element: Item_Yield_Element对应的Element名称
            year: 年份
            
        Returns:
            float: 产率 (t/head/year)
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        def _convert_yield_unit(value: float, unit_str: str, item_name: str) -> float:
            """将FAOSTAT yield单位转换为t/head"""
            if not np.isfinite(value) or value <= 0:
                return 0.0
            u = (unit_str or '').strip().lower()
            item_lower = (item_name or '').lower()
            
            # Milk类产率 (通常是hg/head或kg/head)
            if 'milk' in item_lower:
                if u in {'hg/an', 'hg/animal', 'hg', 'hectogram', '100g/an'}:
                    # hg (100g) -> t: 除以10000
                    return value / 10000.0
                if u in {'kg/an', 'kg/animal', 'kg', 'kilogram'}:
                    # kg -> t: 除以1000
                    return value / 1000.0
                if u in {'t/an', 't/animal', 't', 'tonne'}:
                    return value
                # 默认假设是hg/head（FAOSTAT milk yield标准单位）
                return value / 10000.0
            
            # Egg类产率
            if 'egg' in item_lower:
                if u in {'kg/an', 'kg/animal', 'kg'}:
                    return value / 1000.0
                if u in {'t/an', 't/animal', 't'}:
                    return value
                # 默认假设是kg/head
                return value / 1000.0
            
            # 其他livestock yield（如果走到这里）
            if 'kg' in u:
                return value / 1000.0
            if 'hg' in u or '100g' in u:
                return value / 10000.0
            
            # 保守处理：假设是kg单位
            return value / 1000.0
        
        # 1. 尝试获取特定国家的数据
        hist_yield = self.hist_production[
            (self.hist_production['Area Code (M49)'] == m49_code) &
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        unit_col = 'Unit' if 'Unit' in self.hist_production.columns else None
        
        if not hist_yield.empty and year_col in hist_yield.columns:
            value = hist_yield[year_col].iloc[0]
            if pd.notna(value) and value > 0:
                unit_str = hist_yield[unit_col].iloc[0] if unit_col else ''
                # ✅ 单位转换
                return _convert_yield_unit(float(value), str(unit_str), item)
        
        # ✅ 关键修复：对于Buffalo/Camel/Goats/Sheep dairy，Element='Yield'不存在
        # 需要用Production / Milk Animals计算yield
        if element == 'Yield' and hist_yield.empty:
            # 尝试获取Production和Milk Animals，手动计算yield
            production_data = self.hist_production[
                (self.hist_production['Area Code (M49)'] == m49_code) &
                (self.hist_production['Item'] == item) &
                (self.hist_production['Element'] == 'Production')
            ]
            milk_animals_data = self.hist_production[
                (self.hist_production['Area Code (M49)'] == m49_code) &
                (self.hist_production['Item'] == item) &
                (self.hist_production['Element'] == 'Milk Animals')
            ]
            
            if not production_data.empty and not milk_animals_data.empty:
                if year_col in production_data.columns and year_col in milk_animals_data.columns:
                    prod_value = production_data[year_col].iloc[0]
                    animals_value = milk_animals_data[year_col].iloc[0]
                    
                    if pd.notna(prod_value) and pd.notna(animals_value) and prod_value > 0 and animals_value > 0:
                        # 计算yield = Production / Milk Animals
                        # Production单位通常是tonnes，Milk Animals是head，所以结果是t/head
                        calculated_yield = float(prod_value) / float(animals_value)
                        logger.info(f"  [YIELD_CALC] {item} M49={m49_code}: 从Production({prod_value:.0f}t) / Milk Animals({animals_value:.0f}head) 计算yield = {calculated_yield:.4f} t/head")
                        return calculated_yield
        
        # 2. 如果特定国家缺失，尝试使用同区域平均值
        region = self.m49_to_region.get(m49_code)
        if region:
            region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
            
            if region_m49_codes:
                region_data = self.hist_production[
                    (self.hist_production['Area Code (M49)'].isin(region_m49_codes)) &
                    (self.hist_production['Item'] == item) &
                    (self.hist_production['Element'] == element)
                ]
                
                if not region_data.empty and year_col in region_data.columns:
                    valid_values = region_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
                    valid_values = valid_values[valid_values > 0]
                    
                    if len(valid_values) > 0:
                        region_avg = valid_values.mean()
                        # 获取该区域的单位（取第一个有效值的单位）
                        unit_str = region_data[unit_col].iloc[0] if unit_col else ''
                        # ✅ 单位转换
                        region_avg_t = _convert_yield_unit(region_avg, str(unit_str), item)
                        logger.debug(f"  ℹ️ {item} M49={m49_code} 使用{region}区域平均产率: {region_avg_t:.6f} t/head (基于{len(valid_values)}个国家)")
                        return float(region_avg_t)
        
        # 3. 如果区域也没有数据，使用全球平均值
        global_data = self.hist_production[
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not global_data.empty and year_col in global_data.columns:
            valid_values = global_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            valid_values = valid_values[valid_values > 0]
            
            if len(valid_values) > 0:
                global_avg = valid_values.mean()
                # 获取全球数据的单位
                unit_str = global_data[unit_col].iloc[0] if unit_col else ''
                # ✅ 单位转换
                global_avg_t = _convert_yield_unit(global_avg, str(unit_str), item)
                logger.debug(f"  ℹ️ {item} M49={m49_code} 使用全球平均产率: {global_avg_t:.6f} t/head (基于{len(valid_values)}个国家)")
                return float(global_avg_t)
        
        return 0.0
    
    def calculate_stock_from_animals(self,
                                    animals_df: pd.DataFrame,
                                    year: int) -> pd.DataFrame:
        """
        根据动物流量计算存栏
        
        Args:
            animals_df: 包含producing_animals/slaughtered/milk_animals/laying的DataFrame
            year: 年份
            
        Returns:
            包含stock的DataFrame
        """
        logger.info(f"\n--- 计算存栏 (year={year}) ---")
        logger.info(f"输入animals_df: {len(animals_df)} 行")
        if not animals_df.empty:
            logger.info(f"列名: {list(animals_df.columns)}")
        
        results = []
        
        for _, row in animals_df.iterrows():
            m49_code = row['M49_Country_Code']
            item = row['Item']  # FAOSTAT Item_Production_Map名称
            item_emis = row.get('Item_Emis', item)  # 模型Item_Emis名称
            
            # ✅ 修正：获取slaughtered ratio需要使用Item_SlaughteredRatio_Map
            if item_emis in self.item_slaughtered_ratio_map:
                ratio_item = self.item_slaughtered_ratio_map[item_emis]['Item']
                ratio_element = self.item_slaughtered_ratio_map[item_emis].get('Element', 'Producing/Slaughtered ratio')
                ratio_val = self.get_slaughtered_ratio(m49_code, ratio_item, ratio_element, year)
            else:
                ratio_val = 1.0  # 如果没有映射，使用默认值
            
            if ratio_val <= 0:
                ratio_val = 1.0  # 使用默认比例
            
            # 根据类型计算stock
            stock = 0
            animal_type = None
            
            if 'milk_animals' in row and pd.notna(row['milk_animals']) and row['milk_animals'] > 0:
                stock = row['milk_animals'] / ratio_val
                animal_type = 'milk'
            elif 'slaughtered' in row and pd.notna(row['slaughtered']) and row['slaughtered'] > 0:
                stock = row['slaughtered'] / ratio_val
                animal_type = 'meat'
            elif 'laying' in row and pd.notna(row['laying']) and row['laying'] > 0:
                stock = row['laying'] / ratio_val
                animal_type = 'egg'
            elif 'producing_animals' in row and pd.notna(row['producing_animals']) and row['producing_animals'] > 0:
                stock = row['producing_animals'] / ratio_val
                animal_type = 'producing'
            
            if stock > 0:
                results.append({
                    'M49_Country_Code': m49_code,
                    'Item': item,  # FAOSTAT名称（用于查Production数据）
                    'Item_Emis': item_emis,  # 模型名称（用于查GLE_parameters）
                    'year': year,
                    'stock': stock,
                    'animal_type': animal_type
                })
        
        result_df = pd.DataFrame(results)
        logger.info(f"输出stock_df: {len(result_df)} 行")
        if not result_df.empty:
            logger.info(f"  奶类: {len(result_df[result_df['animal_type']=='milk'])} 行")
            logger.info(f"  肉类: {len(result_df[result_df['animal_type']=='meat'])} 行")
            logger.info(f"  蛋类: {len(result_df[result_df['animal_type']=='egg'])} 行")
        
        return result_df
    
    def get_slaughtered_ratio(self, m49_code: int, item: str, element: str, year: int) -> float:
        """
        获取producing/slaughtered ratio
        优先级：特定国家 -> 同区域平均值 -> 全球平均值 -> 默认值1.0
        
        Args:
            m49_code: M49国家代码
            item: Item_SlaughteredRatio_Map对应的Item名称（FAOSTAT名称）
            element: Item_SlaughteredRatio_Element对应的Element名称（如'Producing/Slaughtered ratio'）
            year: 年份
        """
        year_col = f'Y{year}' if year <= 2020 else 'Y2020'
        
        # 1. 尝试获取特定国家的数据
        hist_ratio = self.hist_production[
            (self.hist_production['Area Code (M49)'] == m49_code) &
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not hist_ratio.empty and year_col in hist_ratio.columns:
            value = hist_ratio[year_col].iloc[0]
            if pd.notna(value):
                return float(value)
        
        # 2. 如果特定国家缺失，尝试使用同区域平均值
        region = self.m49_to_region.get(m49_code)
        if region:
            region_m49_codes = [m49 for m49, reg in self.m49_to_region.items() if reg == region]
            
            if region_m49_codes:
                region_data = self.hist_production[
                    (self.hist_production['Area Code (M49)'].isin(region_m49_codes)) &
                    (self.hist_production['Item'] == item) &
                    (self.hist_production['Element'] == element)
                ]
                
                if not region_data.empty and year_col in region_data.columns:
                    valid_values = region_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(valid_values) > 0:
                        region_avg = valid_values.mean()
                        logger.debug(f"  ℹ️ {item} M49={m49_code} 使用{region}区域平均屠宰比例: {region_avg:.4f} (基于{len(valid_values)}个国家)")
                        return float(region_avg)
        
        # 3. 如果区域也没有数据，使用全球平均值
        global_data = self.hist_production[
            (self.hist_production['Item'] == item) &
            (self.hist_production['Element'] == element)
        ]
        
        if not global_data.empty and year_col in global_data.columns:
            valid_values = global_data[year_col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_values) > 0:
                global_avg = valid_values.mean()
                logger.debug(f"  ℹ️ {item} M49={m49_code} 使用全球平均屠宰比例: {global_avg:.4f} (基于{len(valid_values)}个国家)")
                return float(global_avg)
        
        return 1.0  # 默认比例为1
    
    def calculate_enteric_fermentation(self,
                                      stock_df: pd.DataFrame,
                                      year: int,
                                      scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算肠道发酵排放（Enteric fermentation）
        
        Args:
            stock_df: 包含存栏数据的DataFrame
            year: 年份
            scenario_params: 情景参数
            
        Returns:
            排放DataFrame
        """
        results = []
        
        for _, row in stock_df.iterrows():
            m49_code = row['M49_Country_Code']
            item = row['Item']  # FAOSTAT名称
            item_emis = row.get('Item_Emis', item)  # 模型名称
            stock = row['stock']
            
            # ✅ 关键：使用Item_Emis查询GLE_parameters（参数表使用模型名称）
            ef = self.get_parameter_value(
                m49_code, item_emis, 'Enteric fermentation', 'Emission factor', year
            )
            
            # 应用情景调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, 'Enteric fermentation', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'Enteric fermentation', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                ef *= ef_mult
            
            # 计算CH4排放（kt）
            # ✅ 单位转换: stock(head) × ef(kg CH4/head/year) = kg CH4/year
            # 1 kt = 1000 t = 1,000,000 kg，所以除以1,000,000
            ch4_kt = stock * ef / 1_000_000.0  # kg to kt
            
            results.append({
                'M49_Country_Code': m49_code,
                'Item': item_emis,  # 使用Item_Emis（模型名称）用于汇总
                'year': year,
                'process': 'Enteric fermentation',
                'stock': stock,
                'emission_factor': ef,
                'CH4_kt': ch4_kt,
                'N2O_kt': 0.0,
                'CO2_kt': 0.0
            })
        
        return pd.DataFrame(results)
    
    def calculate_manure_management(self,
                                   stock_df: pd.DataFrame,
                                   year: int,
                                   scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算粪便管理排放（Manure management）
        
        Args:
            stock_df: 包含存栏数据的DataFrame
            year: 年份
            scenario_params: 情景参数
            
        Returns:
            排放DataFrame
        """
        # Defensive: 合并可能重复的存栏记录，避免重复计算导致排放被双计
        if stock_df is None:
            return pd.DataFrame()

        # 确保有关键列
        key_cols = ['M49_Country_Code', 'Item', 'Item_Emis', 'year']
        if set(['M49_Country_Code', 'Item']).issubset(stock_df.columns):
            pre_rows = len(stock_df)
            try:
                if 'Item_Emis' in stock_df.columns:
                    group_cols = key_cols
                else:
                    # 如果没有 Item_Emis，按 Item 合并
                    group_cols = ['M49_Country_Code', 'Item', 'year']

                agg_dict = {}
                # ✅ 修复：聚合 stock 列使用'first'而不是'sum'，避免重复计数
                # 重复记录来自milk/meat filter重叠（如Cattle, non-dairy同时进入两个分类）
                if 'stock' in stock_df.columns:
                    agg_dict['stock'] = 'first'  # 改为first，防止重复求和
                # 对其他数值列采用 first（保留原样）以避免意外求和
                for c in stock_df.columns:
                    if c not in group_cols and c not in agg_dict and pd.api.types.is_numeric_dtype(stock_df[c]):
                        agg_dict[c] = 'first'

                if agg_dict:
                    stock_df = stock_df.groupby(group_cols, as_index=False).agg(agg_dict)
                else:
                    stock_df = stock_df.drop_duplicates(subset=group_cols)

                post_rows = len(stock_df)
                if pre_rows != post_rows:
                    logger.info(f"⚠️ [WARN] calculate_manure_management: 检测到{pre_rows - post_rows}行重复stock记录 ({pre_rows} -> {post_rows})")
                    logger.info(f"    这可能是由于商品同时被milk和meat filter捕获（如'Cattle, non-dairy'）")
            except Exception as e:
                logger.info(f"[WARN] calculate_manure_management: 合并stock出错: {e}")

        results = []

        for _, row in stock_df.iterrows():
            m49_code = row['M49_Country_Code']
            item = row['Item']  # FAOSTAT名称
            item_emis = row.get('Item_Emis', item)  # 模型名称
            stock = row['stock']
            
            # ✅ 关键：使用Item_Emis查询GLE_parameters（参数表使用模型名称）
            n_excretion_rate = self.get_parameter_value(
                m49_code, item_emis, 'Manure management', 'N.excretion.rate', year
            )
            
            # 转换为年排泄量 (kg N/year)
            annual_n_excretion = stock * n_excretion_rate * 365
            
            # ✅ 修正：Environment文件也使用Item_Emis！
            mm_ratio = self.get_manure_management_ratio(m49_code, item_emis, year)
            
            # 应用情景调整：manure management ratio
            if scenario_params and 'manure_management_ratio_multiplier' in scenario_params:
                mm_mult_dict = scenario_params['manure_management_ratio_multiplier']
                country = row.get('country', '')
                mm_mult = mm_mult_dict.get((country, item, year), 1.0)
                if mm_mult == 1.0:
                    mm_mult = mm_mult_dict.get((country, 'All', year), 1.0)
                mm_ratio *= mm_mult
            
            # Managed manure N量
            managed_manure_n = annual_n_excretion * mm_ratio
            
            # ✅ 修复：分别获取CH4和N2O排放因子
            # GLE_parameters中有两行：一行Species='CH4'，一行Species='N2O'
            # 使用Item_Emis查询（GLE_parameters使用模型名称）
            ef_ch4 = self.get_parameter_value_by_species(
                m49_code, item_emis, 'Manure management', 'Emission factor', 'CH4', year
            )
            
            ef_n2o = self.get_parameter_value_by_species(
                m49_code, item_emis, 'Manure management', 'Emission factor', 'N2O', year
            )
            
            # 应用情景调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, 'Manure management', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'Manure management', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                ef_ch4 *= ef_mult
                ef_n2o *= ef_mult
            
            # ✅ 单位转换: kg GHG ÷ 1,000,000 = kt GHG
            ch4_kt = managed_manure_n * ef_ch4 / 1_000_000.0  # kg to kt
            n2o_kt = managed_manure_n * ef_n2o / 1_000_000.0  # kg to kt
            
            # ✅ DEBUG: 输出Australia Cattle non-dairy的详细参数
            if m49_code == "'036" and item_emis == 'Cattle, non-dairy':
                logger.info(f"\n【DEBUG】Australia Cattle, non-dairy - Manure management (year={year})")
                logger.info(f"  Stock (头): {stock:,.0f}")
                logger.info(f"  N.excretion.rate (kg N/head/day): {n_excretion_rate:.6f}")
                logger.info(f"  年排泄量 (kg N/year): {annual_n_excretion:,.0f}")
                logger.info(f"  Manure management ratio: {mm_ratio:.6f}")
                logger.info(f"  Managed manure N (kg N): {managed_manure_n:,.0f}")
                logger.info(f"  EF_CH4 (kg CH4/kg N): {ef_ch4:.6f}")
                logger.info(f"  EF_N2O (kg N2O/kg N): {ef_n2o:.6f}")
                logger.info(f"  CH4排放 (kt): {ch4_kt:.2f}")
                logger.info(f"  N2O排放 (kt): {n2o_kt:.2f}")
            
            # ✅ DEBUG: 输出Australia Goats, dairy的详细参数
            if m49_code == "'036" and item_emis == 'Goats, dairy':
                logger.info(f"\n【DEBUG】Australia Goats, dairy - Manure management (year={year})")
                logger.info(f"  Stock (头): {stock:,.0f}")
                logger.info(f"  N.excretion.rate (kg N/head/day): {n_excretion_rate:.6f}")
                logger.info(f"  年排泄量 (kg N/year): {annual_n_excretion:,.0f}")
                logger.info(f"  Manure management ratio: {mm_ratio:.6f}")
                logger.info(f"  Managed manure N (kg N): {managed_manure_n:,.0f}")
                logger.info(f"  EF_CH4 (kg CH4/kg N): {ef_ch4:.6f}")
                logger.info(f"  EF_N2O (kg N2O/kg N): {ef_n2o:.6f}")
                logger.info(f"  CH4排放 (kt): {ch4_kt:.2f}")
                logger.info(f"  N2O排放 (kt): {n2o_kt:.2f}")
            
            # ✅ DEBUG: 输出Australia Goats, non-dairy的详细参数
            if m49_code == "'036" and item_emis == 'Goats, non-dairy':
                logger.info(f"\n【DEBUG】Australia Goats, non-dairy - Manure management (year={year})")
                logger.info(f"  Stock (头): {stock:,.0f}")
                logger.info(f"  N.excretion.rate (kg N/head/day): {n_excretion_rate:.6f}")
                logger.info(f"  年排泄量 (kg N/year): {annual_n_excretion:,.0f}")
                logger.info(f"  Manure management ratio: {mm_ratio:.6f}")
                logger.info(f"  Managed manure N (kg N): {managed_manure_n:,.0f}")
                logger.info(f"  EF_CH4 (kg CH4/kg N): {ef_ch4:.6f}")
                logger.info(f"  EF_N2O (kg N2O/kg N): {ef_n2o:.6f}")
                logger.info(f"  CH4排放 (kt): {ch4_kt:.2f}")
                logger.info(f"  N2O排放 (kt): {n2o_kt:.2f}")
            
            results.append({
                'M49_Country_Code': m49_code,
                'Item': item_emis,  # 使用Item_Emis（模型名称）用于汇总
                'year': year,
                'process': 'Manure management',
                'stock': stock,
                'annual_n_excretion': annual_n_excretion,
                'mm_ratio': mm_ratio,
                'managed_manure_n': managed_manure_n,
                'CH4_kt': ch4_kt,
                'N2O_kt': n2o_kt,
                'CO2_kt': 0.0
            })
        
        return pd.DataFrame(results)
    
    def get_manure_management_ratio(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        获取manure management ratio
        
        ✅ 注意：Environment_LivestockManure_with_ratio.csv使用Item_Emis（模型名称）！
        ✅ FIX: 如果找不到dairy/non-dairy后缀的Item，回退到不带后缀的基础Item名称
        """
        # 定义dairy/non-dairy到基础Item的映射
        base_item_map = {
            'Cattle, dairy': 'Cattle',
            'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes',
            'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats',
            'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep',
            'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels',
            'Camels, non-dairy': 'Camels',
        }
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            """内部辅助函数：查询指定Item的ratio"""
            hist_mm = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_code) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure management ratio')
            ]
            
            if not hist_mm.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_mm.columns:
                    value = hist_mm[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None
        
        # 1. 首先尝试使用原始Item名称查询
        target_year = min(year, 2020)  # 历史用实际年份，未来用2020
        
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # 2. 如果没找到且Item有dairy/non-dairy后缀，尝试基础Item名称
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                logger.warning(f"⚠️ 未找到{item_emis}的manure management ratio，使用基础Item '{base_item}' 的值: {ratio:.6f} (M49={m49_code}, year={year})")
                return ratio
        
        # 3. 仍未找到，尝试2020年的值（任意一个）
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020
        
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                logger.warning(f"⚠️ 未找到{item_emis}的2020 ratio，使用基础Item '{base_item}' 的值: {ratio_2020:.6f} (M49={m49_code})")
                return ratio_2020
        
        # 4. 完全找不到，返回默认值0.5
        logger.warning(f"⚠️⚠️ 完全未找到{item_emis}的manure management ratio，使用默认值0.5 (M49={m49_code}, year={year})")
        return 0.5
    
    def calculate_manure_applied_to_soils(self,
                                         mm_results: pd.DataFrame,
                                         year: int,
                                         scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算粪便施用到土壤的排放（Manure applied to soils）
        
        Args:
            mm_results: Manure management的计算结果
            year: 年份
            scenario_params: 情景参数
            
        Returns:
            排放DataFrame
        """
        results = []
        
        for _, row in mm_results.iterrows():
            m49_code = row['M49_Country_Code']
            item = row['Item']  # FAOSTAT名称
            item_emis = row.get('Item_Emis', item)  # 模型名称
            managed_manure_n = row['managed_manure_n']
            
            # ✅ 修正：Environment文件使用Item_Emis
            ma_ratio = self.get_manure_applied_ratio(m49_code, item_emis, year)
            
            # Applied to soil的managed manure N量
            applied_manure_n = managed_manure_n * ma_ratio
            
            # ✅ 使用Item_Emis查询emission factor（GLE_parameters使用模型名称）
            ef_n2o = self.get_parameter_value(
                m49_code, item_emis, 'Manure applied to soils', 'Emission factor', year
            )
            
            # 应用情景调整
            if scenario_params and 'ef_multiplier_by' in scenario_params:
                key = (row.get('country'), item, 'Manure applied to soils', year)
                if key in scenario_params['ef_multiplier_by']:
                    ef_n2o *= scenario_params['ef_multiplier_by'][key]
            
            # ✅ 单位转换: kg N2O ÷ 1,000,000 = kt N2O
            n2o_kt = applied_manure_n * ef_n2o / 1_000_000.0  # kg to kt
            
            results.append({
                'M49_Country_Code': m49_code,
                'Item': item_emis,  # 使用Item_Emis（模型名称）用于汇总
                'year': year,
                'process': 'Manure applied to soils',
                'applied_manure_n': applied_manure_n,
                'ma_ratio': ma_ratio,
                'CH4_kt': 0.0,
                'N2O_kt': n2o_kt,
                'CO2_kt': 0.0
            })
        
        return pd.DataFrame(results)
    
    def get_manure_applied_ratio(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        获取manure applied ratio
        
        ✅ 注意：Environment_LivestockManure_with_ratio.csv使用Item_Emis（模型名称）！
        ✅ FIX: 如果找不到dairy/non-dairy后缀的Item，回退到不带后缀的基础Item名称
        """
        base_item_map = {
            'Cattle, dairy': 'Cattle', 'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes', 'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats', 'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep', 'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels', 'Camels, non-dairy': 'Camels',
        }
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            hist_ma = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_code) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure applied ratio')
            ]
            if not hist_ma.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_ma.columns:
                    value = hist_ma[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None
        
        target_year = min(year, 2020)
        
        # 尝试原始Item名称
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # 尝试基础Item名称
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                return ratio
        
        # 尝试2020
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020
        
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                return ratio_2020
        
        return 0.3  # 默认值
    
    def calculate_manure_left_on_pasture(self,
                                        mm_results: pd.DataFrame,
                                        year: int,
                                        scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算粪便遗留在牧场的排放（Manure left on pasture）
        
        Args:
            mm_results: Manure management的计算结果
            year: 年份
            scenario_params: 情景参数
            
        Returns:
            排放DataFrame
        """
        results = []
        
        for _, row in mm_results.iterrows():
            m49_code = row['M49_Country_Code']
            item = row['Item']  # FAOSTAT名称
            item_emis = row.get('Item_Emis', item)  # 模型名称
            annual_n_excretion = row['annual_n_excretion']
            mm_ratio = row['mm_ratio']
            
            # ✅ 修正：Environment文件使用Item_Emis
            lp_ratio_sum = self.get_left_pasture_ratio_sum(m49_code, item_emis, year)
            
            # Manure left_pasture ratio = lp_ratio_sum - mm_ratio
            lp_ratio = lp_ratio_sum - mm_ratio
            lp_ratio = max(0, min(1, lp_ratio))  # 限制在[0,1]
            
            # Left on pasture的manure N量
            left_pasture_n = annual_n_excretion * lp_ratio
            
            # ✅ 使用Item_Emis查询emission factor（GLE_parameters使用模型名称）
            ef_n2o = self.get_parameter_value(
                m49_code, item_emis, 'Manure left on pasture', 'Emission factor', year
            )
            
            # 应用情景调整
            if scenario_params and 'ef_multiplier_by' in scenario_params:
                key = (row.get('country'), item, 'Manure left on pasture', year)
                if key in scenario_params['ef_multiplier_by']:
                    ef_n2o *= scenario_params['ef_multiplier_by'][key]
            
            # ✅ 单位转换: kg N2O ÷ 1,000,000 = kt N2O
            n2o_kt = left_pasture_n * ef_n2o / 1_000_000.0  # kg to kt
            
            results.append({
                'M49_Country_Code': m49_code,
                'Item': item_emis,  # 使用Item_Emis（模型名称）用于汇总
                'year': year,
                'process': 'Manure left on pasture',
                'left_pasture_n': left_pasture_n,
                'lp_ratio': lp_ratio,
                'CH4_kt': 0.0,
                'N2O_kt': n2o_kt,
                'CO2_kt': 0.0
            })
        
        return pd.DataFrame(results)
    
    def get_left_pasture_ratio_sum(self, m49_code: int, item_emis: str, year: int) -> float:
        """
        获取left_pasture_management ratio sum
        
        ✅ 注意：Environment_LivestockManure_with_ratio.csv使用Item_Emis（模型名称）！
        ✅ FIX: 如果找不到dairy/non-dairy后缀的Item，回退到不带后缀的基础Item名称
        """
        base_item_map = {
            'Cattle, dairy': 'Cattle', 'Cattle, non-dairy': 'Cattle',
            'Buffaloes, dairy': 'Buffaloes', 'Buffaloes, non-dairy': 'Buffaloes',
            'Goats, dairy': 'Goats', 'Goats, non-dairy': 'Goats',
            'Sheep, dairy': 'Sheep', 'Sheep, non-dairy': 'Sheep',
            'Camels, dairy': 'Camels', 'Camels, non-dairy': 'Camels',
        }
        
        def _query_ratio(item_name: str, target_year: int) -> float:
            hist_lp = self.hist_manure_stock[
                (self.hist_manure_stock['M49_Country_Code'] == m49_code) &
                (self.hist_manure_stock['Item'] == item_name) &
                (self.hist_manure_stock['Element'] == 'Manure left_pasture_management ratio sum')
            ]
            if not hist_lp.empty:
                year_col = f'Y{target_year}'
                if year_col in hist_lp.columns:
                    value = hist_lp[year_col].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None
        
        target_year = min(year, 2020)
        
        # 尝试原始Item名称
        ratio = _query_ratio(item_emis, target_year)
        if ratio is not None:
            return ratio
        
        # 尝试基础Item名称
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio = _query_ratio(base_item, target_year)
            if ratio is not None:
                return ratio
        
        # 尝试2020
        ratio_2020 = _query_ratio(item_emis, 2020)
        if ratio_2020 is not None:
            return ratio_2020
        
        if item_emis in base_item_map:
            base_item = base_item_map[item_emis]
            ratio_2020 = _query_ratio(base_item, 2020)
            if ratio_2020 is not None:
                return ratio_2020
        
        return 0.7  # 默认值
    
    def merge_dairy_nondairy(self, emissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并dairy和non-dairy品种
        
        Args:
            emissions_df: 排放DataFrame
            
        Returns:
            合并后的DataFrame
        """
        # 如果DataFrame为空或没有Item列，直接返回
        if emissions_df.empty or 'Item' not in emissions_df.columns:
            return emissions_df
        
        # 定义需要合并的物种对
        merge_pairs = {
            ('Buffalo, dairy', 'Buffalo, non-dairy'): 'Buffalo',
            ('Camel, dairy', 'Camel, non-dairy'): 'Camels',
            ('Goats, dairy', 'Goats, non-dairy'): 'Goats',
            ('Sheep, dairy', 'Sheep, non-dairy'): 'Sheep'
        }
        
        result_list = []
        
        for (dairy_item, nondairy_item), merged_item in merge_pairs.items():
            # 找到dairy和non-dairy的数据
            dairy_data = emissions_df[emissions_df['Item'] == dairy_item].copy()
            nondairy_data = emissions_df[emissions_df['Item'] == nondairy_item].copy()
            
            if dairy_data.empty and nondairy_data.empty:
                continue
            
            # 合并数据
            merged_data = pd.concat([dairy_data, nondairy_data], ignore_index=True)
            
            # 按M49_Country_Code, year, process聚合
            agg_dict = {
                'CH4_kt': 'sum',
                'N2O_kt': 'sum',
                'CO2_kt': 'sum'
            }
            
            # 添加其他数值列
            for col in merged_data.columns:
                if col not in ['M49_Country_Code', 'Item', 'year', 'process', 
                              'CH4_kt', 'N2O_kt', 'CO2_kt'] and \
                   pd.api.types.is_numeric_dtype(merged_data[col]):
                    agg_dict[col] = 'sum'
            
            merged = merged_data.groupby(
                ['M49_Country_Code', 'year', 'process'],
                as_index=False
            ).agg(agg_dict)
            
            merged['Item'] = merged_item
            result_list.append(merged)
        
        # 保留原始数据中不需要合并的物种
        items_to_remove = []
        for (dairy, nondairy), _ in merge_pairs.items():
            items_to_remove.extend([dairy, nondairy])
        
        other_data = emissions_df[~emissions_df['Item'].isin(items_to_remove)].copy()
        result_list.append(other_data)
        
        # 合并所有结果
        final_result = pd.concat(result_list, ignore_index=True)
        
        return final_result
    
    def extract_livestock_parameters(self, years: List[int], hist_cutoff_year: int = 2020) -> Dict[str, pd.DataFrame]:
        """
        提取livestock参数用于production_summary
        
        历史年份从Environment_LivestockManure_with_ratio.csv读取
        未来年份使用历史最后一年(2020)的baseline值
        
        Returns:
            包含各参数的DataFrame字典
        """
        try:
            # 读取历史manure stock数据
            manure_df = pd.read_csv(self.hist_manure_stock_path)
            logger.info(f"[DEBUG] 读取历史livestock参数: {len(manure_df)} 行")
            
            # 标准化列名
            manure_df.columns = [c.strip() for c in manure_df.columns]
            
            # 需要的参数列
            param_dfs = {}
            
            # Stock参数
            if 'Stocks' in manure_df.columns:
                stock_df = manure_df[['Area', 'Item', 'Year', 'Stocks']].copy()
                stock_df = stock_df.rename(columns={
                    'Area': 'country',
                    'Item': 'commodity',
                    'Year': 'year',
                    'Stocks': 'stock_head'
                })
                param_dfs['stock'] = stock_df
            
            # ⚠️ Feed requirement创建移到下方（在stock被GLE计算结果覆盖之后）
            # 这样可以确保使用的是GLE计算的stock_df（带M49代码），而不是CSV读取的stock_df（国家名称）
            
            # Manure management ratio
            if 'Manure_management_ratio' in manure_df.columns:
                manure_ratio_df = manure_df[['Area', 'Item', 'Year', 'Manure_management_ratio']].copy()
                manure_ratio_df = manure_ratio_df.rename(columns={
                    'Area': 'country',
                    'Item': 'commodity',
                    'Year': 'year',
                    'Manure_management_ratio': 'manure_management_ratio'
                })
                param_dfs['manure_ratio'] = manure_ratio_df
            
            # ====================================================================
            # 使用GLE计算的slaughter、stock、carcass_yield（如果有的话）
            # 这些数据由run_full_calculation在计算过程中保存
            # ====================================================================
            if self._computed_slaughter:
                computed_slaughter_df = pd.concat(self._computed_slaughter, ignore_index=True)
                param_dfs['slaughter'] = computed_slaughter_df
                logger.info(f"[DEBUG] 导出GLE计算的slaughter: {len(computed_slaughter_df)} 行")
            
            if self._computed_stock:
                computed_stock_df = pd.concat(self._computed_stock, ignore_index=True)
                # 用计算结果覆盖从CSV读取的stock
                param_dfs['stock'] = computed_stock_df
                logger.info(f"[DEBUG] 导出GLE计算的stock: {len(computed_stock_df)} 行")
            
            if self._computed_carcass_yield:
                computed_yield_df = pd.concat(self._computed_carcass_yield, ignore_index=True)
                param_dfs['carcass_yield'] = computed_yield_df
                logger.info(f"[DEBUG] 导出GLE计算的carcass_yield (meat类): {len(computed_yield_df)} 行")
            
            # ✅ 新增：导出dairy yield (milk/egg类)
            if self._computed_dairy_yield:
                computed_dairy_yield_df = pd.concat(self._computed_dairy_yield, ignore_index=True)
                param_dfs['dairy_yield'] = computed_dairy_yield_df
                logger.info(f"[DEBUG] 导出GLE计算的dairy_yield (milk/egg类): {len(computed_dairy_yield_df)} 行")
            
            # ✅ Feed requirement (从dict_v3中的默认值)
            # 在stock被GLE计算结果覆盖之后创建，确保使用M49代码
            if hasattr(self, 'dict_v3') and 'stock' in param_dfs:
                emis_item_df = self.dict_v3.get('Emis_item', pd.DataFrame())
                if not emis_item_df.empty and 'Feed_GE' in emis_item_df.columns:
                    # 创建feed requirement lookup
                    feed_lookup = emis_item_df.set_index('Item_Emis')['Feed_GE'].to_dict()
                    
                    # 使用最新的stock_df（GLE计算结果，带M49代码）
                    stock_df_for_feed = param_dfs['stock']
                    feed_req_df = stock_df_for_feed[['country', 'commodity', 'year']].copy()
                    feed_req_df['feed_requirement_kg_per_head'] = feed_req_df['commodity'].map(feed_lookup)
                    param_dfs['feed_requirement'] = feed_req_df
                    logger.info(f"[DEBUG] 导出GLE计算的feed_requirement: {len(feed_req_df)} 行 (基于GLE计算的stock)")
            
            # 为未来年份扩展参数（使用2020 baseline）
            # 注意：如果_computed_*中已有未来年份数据，则不需要扩展
            future_years = [y for y in years if y > hist_cutoff_year]
            if future_years:
                logger.info(f"[DEBUG] 检查是否需要扩展参数: {future_years}")
                for param_name, param_df in list(param_dfs.items()):
                    if param_df.empty:
                        continue
                    
                    # 检查是否已经有未来年份数据
                    existing_future = param_df[param_df['year'].isin(future_years)]
                    if not existing_future.empty:
                        logger.info(f"  [{param_name}] 已有 {len(existing_future)} 行未来年份数据，跳过扩展")
                        continue
                    
                    # 提取2020年数据作为baseline
                    baseline_2020 = param_df[param_df['year'] == hist_cutoff_year].copy()
                    if baseline_2020.empty:
                        continue
                    
                    # 为每个未来年份复制
                    extended_rows = []
                    for future_year in future_years:
                        future_data = baseline_2020.copy()
                        future_data['year'] = future_year
                        extended_rows.append(future_data)
                    
                    if extended_rows:
                        param_dfs[param_name] = pd.concat([param_df] + extended_rows, ignore_index=True)
                        logger.info(f"  [{param_name}] 扩展后: {len(param_dfs[param_name])} 行")
            
            return param_dfs
            
        except Exception as e:
            logger.info(f"[ERROR] 提取livestock参数失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_calculation(self,
                            production_df: pd.DataFrame,
                            years: List[int],
                            scenario_params: Optional[Dict] = None,
                            hist_cutoff_year: int = 2020) -> Dict[str, pd.DataFrame]:
        """
        运行完整的排放计算流程
        
        Args:
            production_df: 包含未来产量预测的DataFrame
            years: 计算年份列表
            scenario_params: 情景参数
            hist_cutoff_year: 历史数据截止年份（<=此年份的排放直接从历史数据读取）
            
        Returns:
            包含所有排放过程结果的字典
        """
        # WARNING: do not filter Region here - production_df should already be filtered
        # Region过滤应该在S4_0_main.py中生成production_for_livestock时进行
        
        all_results = {
            'Enteric fermentation': [],
            'Manure management': [],
            'Manure applied to soils': [],
            'Manure left on pasture': []
        }
        
        # 分离历史年份和未来年份
        hist_years = [y for y in years if y <= hist_cutoff_year]
        future_years = [y for y in years if y > hist_cutoff_year]
        
        # === 1. 历史年份：直接从Emissions CSV读取 ===
        if hist_years:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading historical emissions for years {min(hist_years)}-{max(hist_years)}...")
            logger.info(f"{'='*60}")
            hist_emissions = self._load_historical_emissions(hist_years)
            if hist_emissions:
                for process, emis_df in hist_emissions.items():
                    if emis_df is not None and not emis_df.empty:
                        all_results[process].append(emis_df)
                        logger.info(f"  [OK] {process}: {len(emis_df)} 行历史数据")
        
        # === 2. 未来年份：计算排放 ===
        logger.info(f"\n[DEBUG] future_years = {future_years}")
        logger.info(f"[DEBUG] production_df shape: {production_df.shape}")
        if not production_df.empty:
            logger.info(f"[DEBUG] production_df years: {production_df['year'].unique() if 'year' in production_df.columns else 'NO year column'}")
            logger.info(f"[DEBUG] production_df columns: {list(production_df.columns)}")
            
            # 打印2080年的数据情况
            if 'year' in production_df.columns:
                year_2080_data = production_df[production_df['year'] == 2080]
                logger.info(f"[DEBUG] 2080年数据行数: {len(year_2080_data)}")
                if not year_2080_data.empty:
                    logger.info(f"[DEBUG] 2080年前5行:")
                    logger.info(year_2080_data.head())
                    if 'Commodity' in year_2080_data.columns:
                        logger.info(f"[DEBUG] 2080年商品: {year_2080_data['Commodity'].unique()}")

            # ❌ BUG FIX: Do NOT normalize production_df M49 codes
            # production_df should already have correct format from S4_0_main
            # Normalizing here would break matching with dict_v3 (which has '004, '008, etc.)
        
        for year in future_years:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {year}...")
            logger.info(f"{'='*60}")
            
            # WARNING: debug: inspect the production statistics
            if 'year' in production_df.columns:
                year_prod = production_df[production_df['year'] == year]
                logger.info(f"[DEBUG] Year {year} 产量数据: {len(year_prod)} 行")
                if not year_prod.empty:
                    logger.info(f"  - 商品数: {year_prod['Commodity'].nunique() if 'Commodity' in year_prod.columns else 'N/A'}")
                    logger.info(f"  - 国家数: {year_prod['M49_Country_Code'].nunique() if 'M49_Country_Code' in year_prod.columns else 'N/A'}")
                    if 'production_t' in year_prod.columns:
                        total_prod = year_prod['production_t'].sum()
                        logger.info(f"  - 总产量: {total_prod:.0f} tons")
                else:
                    logger.info(f"  WARNING: country set has no production data")
            
            # 1. 计算dairy animals (包括奶类和产蛋鸡)
            # 注意：Chickens, layers 在dict_v3中归类为Dairy
            logger.info(f"\n=== 调用calculate_milk_animals (year={year}) ===")
            dairy_animals_df = self.calculate_milk_animals(production_df, year)
            logger.info(f"[返回结果] dairy_animals_df: {len(dairy_animals_df) if not dairy_animals_df.empty else 0} 行")
            if not dairy_animals_df.empty:
                logger.info(f"  - 包含列: {list(dairy_animals_df.columns)}")
                logger.info(f"  - 前3行数据:")
                logger.info(dairy_animals_df.head(3))
            
            # 2. 计算meat animals (非dairy的肉类动物)
            logger.info(f"\n=== 调用calculate_meat_animals (year={year}) ===")
            meat_animals_df = self.calculate_meat_animals(production_df, year)
            logger.info(f"[返回结果] meat_animals_df: {len(meat_animals_df) if not meat_animals_df.empty else 0} 行")
            if not meat_animals_df.empty:
                logger.info(f"  - 包含列: {list(meat_animals_df.columns)}")
                logger.info(f"  - 前3行数据:")
                logger.info(meat_animals_df.head(3))
            
            # ✅ 保存计算的dairy yield数据（从dairy_animals_df提取）
            if not dairy_animals_df.empty and 'yield_t_per_head' in dairy_animals_df.columns:
                dairy_yield_save = dairy_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'yield_t_per_head']].copy()
                dairy_yield_save = dairy_yield_save.rename(columns={
                    'Item_Emis': 'commodity'
                })
                # ✅ 保留M49_Country_Code，同时创建country列
                dairy_yield_save['country'] = dairy_yield_save['M49_Country_Code']
                self._computed_dairy_yield.append(dairy_yield_save)
                logger.info(f"  [保存] dairy_yield: {len(dairy_yield_save)} 行")
            
            # 3. 合并所有动物流量
            all_animals_list = []
            if not dairy_animals_df.empty:
                all_animals_list.append(dairy_animals_df)
            if not meat_animals_df.empty:
                all_animals_list.append(meat_animals_df)
            
            if not all_animals_list:
                logger.info(f"WARNING: Year {year}: no production data for this country; skipping")
                continue
            
            all_animals_df = pd.concat(all_animals_list, ignore_index=True)
            logger.info(f"\n总计 {len(all_animals_df)} 个动物流量记录")
            
            # 5. 计算stock
            stock_df = self.calculate_stock_from_animals(all_animals_df, year)
            
            if stock_df.empty:
                logger.info(f"WARNING: Year {year}: stock_df is empty; skipping emissions")
                continue
            
            # ✅ 保存计算的stock数据
            # 关键修复：保留M49_Country_Code列，同时也提供country列用于兼容性
            if 'stock' in stock_df.columns:
                stock_save = stock_df[['M49_Country_Code', 'Item_Emis', 'year', 'stock']].copy()
                # ✅ 保留M49_Country_Code，同时创建country列（复制M49作为国家标识）
                stock_save = stock_save.rename(columns={
                    'Item_Emis': 'commodity',
                    'stock': 'stock_head'
                })
                # 额外提供country列（与M49相同，供兼容）
                stock_save['country'] = stock_save['M49_Country_Code']
                self._computed_stock.append(stock_save)
            
            # ✅ 保存计算的slaughter数据（从meat_animals_df提取）
            # 关键修复：保留M49_Country_Code列
            if not meat_animals_df.empty and 'slaughtered' in meat_animals_df.columns:
                slaughter_save = meat_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'slaughtered']].copy()
                slaughter_save = slaughter_save.rename(columns={
                    'Item_Emis': 'commodity',
                    'slaughtered': 'slaughter_head'
                })
                # ✅ 保留M49_Country_Code，同时创建country列
                slaughter_save['country'] = slaughter_save['M49_Country_Code']
                
                # 添加carcass_yield（carcass_weight_t是列名）
                if 'carcass_weight_t' in meat_animals_df.columns:
                    yield_save = meat_animals_df[['M49_Country_Code', 'Item_Emis', 'year', 'carcass_weight_t']].copy()
                    yield_save = yield_save.rename(columns={
                        'Item_Emis': 'commodity',
                        'carcass_weight_t': 'yield_t_per_head'
                    })
                    # ✅ 保留M49_Country_Code，同时创建country列
                    yield_save['country'] = yield_save['M49_Country_Code']
                    self._computed_carcass_yield.append(yield_save)
                self._computed_slaughter.append(slaughter_save)
            
            logger.info(f"计算得到 {len(stock_df)} 个存栏记录")
            
            # 6. 计算Enteric fermentation
            ef_results = self.calculate_enteric_fermentation(stock_df, year, scenario_params)
            if not ef_results.empty:
                all_results['Enteric fermentation'].append(ef_results)
                logger.info(f"  [OK] Enteric fermentation: {len(ef_results)} 行")
            
            # 7. 计算Manure management
            mm_results = self.calculate_manure_management(stock_df, year, scenario_params)
            # 防御性：去掉 mm_results 中的完全重复行（按国家/Item/年/三气体值）
            try:
                if mm_results is not None and not mm_results.empty:
                    before_mm = len(mm_results)
                    dup_cols = ['M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
                    if set(dup_cols).issubset(mm_results.columns):
                        mm_results = mm_results.drop_duplicates(subset=dup_cols, keep='first').reset_index(drop=True)
                        after_mm = len(mm_results)
                        if before_mm != after_mm:
                            logger.info(f"[INFO] run_full_calculation: 去除 mm_results 完全重复行 {before_mm} -> {after_mm}")
            except Exception as e:
                logger.info(f"[WARN] run_full_calculation: 去重 mm_results 失败: {e}")
            # ===== Enhanced diagnostic dump for manure management (always write a timestamped snapshot) =====
            try:
                import datetime
                dbg_dir = Path(os.path.join(os.getcwd(), 'debug_outputs'))
                dbg_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

                if not mm_results.empty:
                    # write full mm_results snapshot for this year
                    try:
                        mm_results.to_csv(dbg_dir / f'mm_results_snapshot_{year}_{ts}.csv', index=False)
                    except Exception:
                        pass

                    # detect exact duplicate rows in mm_results
                    try:
                        dup_cols = ['M49_Country_Code', 'Item', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
                        if set(dup_cols).issubset(mm_results.columns):
                            dup_mask_all = mm_results.duplicated(subset=dup_cols, keep=False)
                            if dup_mask_all.any():
                                try:
                                    mm_results[dup_mask_all].to_csv(dbg_dir / f'mm_results_duplicates_{year}_{ts}.csv', index=False)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # write a small summary file for quick review
                    try:
                        with open(dbg_dir / f'mm_debug_summary_{year}_{ts}.txt', 'w', encoding='utf-8') as fh:
                            fh.write(f"year={year}\n")
                            fh.write(f"mm_results rows total: {len(mm_results)}\n")
                            try:
                                # count duplicates (exact)
                                dup_count = int(mm_results.duplicated(subset=dup_cols, keep=False).sum()) if set(dup_cols).issubset(mm_results.columns) else 0
                            except Exception:
                                dup_count = -1
                            fh.write(f"duplicates_exact_count: {dup_count}\n")
                    except Exception:
                        pass
            except Exception:
                # Do not interrupt normal flow if diagnostics fail
                pass
            if not mm_results.empty:
                all_results['Manure management'].append(mm_results)
                logger.info(f"  [OK] Manure management: {len(mm_results)} 行")
            
            # 8. 计算Manure applied to soils
            mas_results = self.calculate_manure_applied_to_soils(mm_results, year, scenario_params)
            if not mas_results.empty:
                all_results['Manure applied to soils'].append(mas_results)
                logger.info(f"  [OK] Manure applied to soils: {len(mas_results)} 行")
            
            # 9. 计算Manure left on pasture
            mlp_results = self.calculate_manure_left_on_pasture(mm_results, year, scenario_params)
            if not mlp_results.empty:
                all_results['Manure left on pasture'].append(mlp_results)
                logger.info(f"  [OK] Manure left on pasture: {len(mlp_results)} 行")
        
        # 合并所有年份的结果
        final_results = {}
        for process, results_list in all_results.items():
            if results_list:
                combined = pd.concat(results_list, ignore_index=True)
                # ✅ 关键修复：不再合并dairy和non-dairy，dict_v3要求严格分离
                # 历史数据已由S4_1_results拆分，未来数据本身就分开计算
                # combined = self.merge_dairy_nondairy(combined)  # ⚠️ 禁用，与要求相惖
                final_results[process] = combined
                
                # WARNING: debug: look at the per-process data distribution
                if 'year' in combined.columns:
                    year_counts = combined['year'].value_counts().sort_index()
                    logger.info(f"\n[DEBUG] {process} 年份分布:")
                    for yr, cnt in year_counts.items():
                        logger.info(f"  - {yr}: {cnt} 行")
        
        return final_results


def calculate_stock_from_optimized_production(
    production_df: pd.DataFrame,
    years: List[int],
    gle_params_path: str,
    hist_production_path: str,
    hist_emissions_path: str,
    hist_manure_stock_path: str,
    dict_v3_path: str,
    universe: Any,
    hist_cutoff_year: int = 2020
) -> pd.DataFrame:
    """
    从优化后的产量计算存栏数量
    
    完整计算链：
        Qs (优化后产量) 
        → calculate_milk_animals / calculate_meat_animals (产量→动物流量)
        → calculate_stock_from_animals (动物流量→存栏)
        → stock_df (用于 build_feed_demand_from_stock)
    
    Args:
        production_df: 优化后的产量DataFrame，包含 [country, year, commodity, production_t] 和 M49_Country_Code
        years: 计算年份列表
        gle_params_path: GLE参数文件路径
        hist_production_path: 历史生产数据路径
        hist_emissions_path: 历史排放数据路径
        hist_manure_stock_path: 历史粪便存栏数据路径
        dict_v3_path: dict_v3文件路径
        universe: Universe对象，包含国家映射
        hist_cutoff_year: 历史数据截止年份
        
    Returns:
        stock_df: 存栏DataFrame，格式为 [country, iso3, year, commodity, stock_head]
                  与 build_feed_demand_from_stock 输入格式兼容
    """
    logger.info("=" * 60)
    logger.info("开始从优化后产量计算存栏 (calculate_stock_from_optimized_production)")
    logger.info("=" * 60)
    
    # 🔍 详细诊断：输入参数检查
    logger.info("\n🔍 [GLE诊断] 输入参数检查:")
    logger.info(f"  - production_df是否为空: {production_df is None or production_df.empty}")
    if production_df is not None and not production_df.empty:
        logger.info(f"  - production_df形状: {production_df.shape}")
        logger.info(f"  - production_df列名: {list(production_df.columns)}")
        logger.info(f"  - production_df年份: {sorted(production_df['year'].unique()) if 'year' in production_df.columns else 'N/A'}")
        logger.info(f"  - production_df国家数: {production_df['country'].nunique() if 'country' in production_df.columns else 'N/A'}")
        logger.info(f"  - production_df商品数: {production_df['commodity'].nunique() if 'commodity' in production_df.columns else 'N/A'}")
        if 'Commodity' in production_df.columns:
            logger.info(f"  - Commodity列存在: ✅")
            logger.info(f"  - Commodity样例: {production_df['Commodity'].head(5).tolist()}")
        else:
            logger.info(f"  - Commodity列存在: ❌")
        if 'production_t' in production_df.columns:
            logger.info(f"  - production_t总和: {production_df['production_t'].sum():,.0f} t")
            logger.info(f"  - production_t非零行数: {(production_df['production_t'] > 0).sum()}/{len(production_df)}")
    logger.info(f"  - years参数: {years}")
    logger.info(f"  - hist_cutoff_year: {hist_cutoff_year}\n")
    
    if production_df is None or production_df.empty:
        logger.warning("❌ production_df 为空，返回空存栏数据")
        return pd.DataFrame(columns=['country', 'iso3', 'year', 'commodity', 'stock_head'])
    
    # 实例化计算器
    calculator = LivestockEmissionsCalculator(
        gle_params_path=gle_params_path,
        hist_production_path=hist_production_path,
        hist_emissions_path=hist_emissions_path,
        hist_manure_stock_path=hist_manure_stock_path,
        dict_v3_path=dict_v3_path
    )
    
    # 只处理未来年份（历史年份使用历史存栏数据）
    future_years = [y for y in years if y > hist_cutoff_year]
    hist_years = [y for y in years if y <= hist_cutoff_year]
    
    all_stock_results = []
    
    # === 1. 未来年份：从优化产量计算存栏 ===
    logger.info(f"处理未来年份 {future_years}，从优化产量计算存栏...")
    
    for year in future_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 处理年份 {year}...")
        logger.info(f"{'='*60}")
        
        # 1a. 计算dairy类动物（奶牛、产蛋鸡等）
        dairy_animals_df = calculator.calculate_milk_animals(production_df, year)
        logger.info(f"  Dairy动物计算结果: {len(dairy_animals_df)} 行")
        
        # 1b. 计算meat类动物（肉牛、猪等）
        meat_animals_df = calculator.calculate_meat_animals(production_df, year)
        logger.info(f"  Meat动物计算结果: {len(meat_animals_df)} 行")
        
        # 1c. 合并所有动物流量
        all_animals_list = []
        if not dairy_animals_df.empty:
            all_animals_list.append(dairy_animals_df)
        if not meat_animals_df.empty:
            all_animals_list.append(meat_animals_df)
        
        if not all_animals_list:
            logger.warning(f"年份 {year} 没有动物流量数据，跳过")
            continue
        
        animals_df = pd.concat(all_animals_list, ignore_index=True)
        
        # 1d. 从动物流量计算存栏
        stock_df_year = calculator.calculate_stock_from_animals(animals_df, year)
        
        if not stock_df_year.empty:
            all_stock_results.append(stock_df_year)
            logger.info(f"  年份 {year}: 计算得到 {len(stock_df_year)} 条存栏记录")
    
    if not all_stock_results:
        logger.warning("没有计算得到任何存栏数据")
        return pd.DataFrame(columns=['country', 'iso3', 'year', 'commodity', 'stock_head'])
    
    # === 2. 合并所有年份的存栏数据 ===
    combined_stock = pd.concat(all_stock_results, ignore_index=True)
    
    # === 3. 格式转换：匹配 build_feed_demand_from_stock 输入格式 ===
    # 原格式: ['M49_Country_Code', 'Item', 'Item_Emis', 'year', 'stock', 'animal_type']
    # 目标格式: ['country', 'iso3', 'year', 'commodity', 'stock_head']
    
    # 创建 M49 → country 映射
    m49_to_country = {}
    m49_to_iso3 = {}
    if universe and hasattr(universe, 'm49_by_country') and universe.m49_by_country:
        for country, m49 in universe.m49_by_country.items():
            # 规范化 M49 为多种格式以便匹配
            m49_normalized = str(m49).strip().lstrip("'\"")
            if m49_normalized.isdigit():
                m49_int_str = str(int(m49_normalized))  # 去除前导零
                m49_to_country[m49_int_str] = country
                m49_to_country[m49_normalized] = country
                m49_to_country[f"'{m49_normalized.zfill(3)}"] = country
            else:
                m49_to_country[m49_normalized] = country
    
    if universe and hasattr(universe, 'iso3_by_country') and universe.iso3_by_country:
        for country, iso3 in universe.iso3_by_country.items():
            m49_to_iso3[country] = iso3
    
    # 转换列名和格式
    result_df = combined_stock.copy()
    
    # 🔍 诊断：以美国为例追踪存栏数据
    us_stock = result_df[result_df['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"").isin(['840', '0840', "'840"])]
    if not us_stock.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 2: 优化后存栏计算完成")
        logger.info("=" * 80)
        logger.info(f"美国存栏数据: {len(us_stock)} 行")
        us_summary = us_stock.groupby(['Item_Emis', 'year'])['stock'].sum().reset_index()
        for _, row in us_summary.head(10).iterrows():
            logger.info(f"  {row['Item_Emis']}: {row['stock']:,.0f} head ({row['year']}年)")
    
    # M49 → country
    def _convert_m49_to_country(m49_val):
        m49_str = str(m49_val).strip().lstrip("'\"")
        # 尝试多种格式匹配
        for fmt in [m49_str, str(int(m49_str)) if m49_str.isdigit() else m49_str, f"'{m49_str.zfill(3)}"]:
            if fmt in m49_to_country:
                return m49_to_country[fmt]
        return None
    
    result_df['country'] = result_df['M49_Country_Code'].apply(_convert_m49_to_country)
    result_df['iso3'] = result_df['country'].map(m49_to_iso3)
    
    # ✅ 关键修复：保持commodity为Item_Emis格式
    # build_feed_demand_from_stock期望Item_Emis格式（如"Cattle, dairy"），
    # 它内部会使用comm_to_species映射将Item_Emis转换为species（如"dairy_cattle"）
    # 因此这里不需要转换，直接使用Item_Emis作为commodity
    result_df['commodity'] = result_df['Item_Emis']
    result_df['stock_head'] = result_df['stock']
    
    # 🔍 诊断：确认commodity格式
    unique_commodity = result_df['commodity'].unique()
    logger.info(f"✅ 存栏数据commodity格式（Item_Emis）: {list(unique_commodity)[:10]}")
    
    # 🔍 诊断：以美国为例追踪commodity数据
    us_converted = result_df[result_df['country'] == 'United States of America']
    if not us_converted.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 3: 存栏数据格式确认")
        logger.info("=" * 80)
        us_sample = us_converted[['commodity', 'stock_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            logger.info(f"  {row['commodity']:25s} | {row['stock_head']:>12,.0f} head ({row['year']}年)")
    
    # 筛选有效数据
    result_df = result_df.dropna(subset=['country'])
    result_df = result_df[result_df['stock_head'] > 0]
    
    # 选择需要的列
    output_cols = ['country', 'iso3', 'year', 'commodity', 'stock_head']
    result_df = result_df[output_cols].copy()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"存栏计算完成: 共 {len(result_df)} 条记录")
    logger.info(f"  - 年份范围: {result_df['year'].min()}-{result_df['year'].max()}")
    logger.info(f"  - 国家数: {result_df['country'].nunique()}")
    logger.info(f"  - 商品数: {result_df['commodity'].nunique()}")
    logger.info(f"  - 总存栏: {result_df['stock_head'].sum():,.0f} head")
    logger.info(f"{'='*60}")
    
    return result_df


def run_livestock_emissions(
    production_df: pd.DataFrame,
    years: List[int],
    gle_params_path: str,
    hist_production_path: str,
    hist_emissions_path: str,
    hist_manure_stock_path: str,
    dict_v3_path: str,
    scenario_params: Optional[Dict] = None,
    hist_cutoff_year: int = 2020
) -> Dict[str, Any]:
    """
    运行畜牧业排放计算的主函数
    
    Args:
        production_df: 产量预测DataFrame
        years: 计算年份列表
        gle_params_path: GLE参数文件路径
        hist_production_path: 历史生产数据路径
        hist_emissions_path: 历史排放数据路径（历史年份直接从此文件读取）
        hist_manure_stock_path: 历史粪便存栏数据路径
        dict_v3_path: dict_v3文件路径
        scenario_params: 情景参数
        hist_cutoff_year: 历史数据截止年份（<=此年份从CSV读取，>此年份才计算）
        
    Returns:
        包含emissions和parameters的字典:
        {
            'emissions': {process_name: DataFrame, ...},  # 排放结果
            'parameters': {  # livestock参数
                'stock': DataFrame,
                'feed_requirement': DataFrame,
                'slaughter': DataFrame,
                'carcass_yield': DataFrame,
                'manure_ratio': DataFrame
            }
        }
    """
    calculator = LivestockEmissionsCalculator(
        gle_params_path=gle_params_path,
        hist_production_path=hist_production_path,
        hist_emissions_path=hist_emissions_path,
        hist_manure_stock_path=hist_manure_stock_path,
        dict_v3_path=dict_v3_path
    )
    
    emissions = calculator.run_full_calculation(
        production_df=production_df,
        years=years,
        scenario_params=scenario_params,
        hist_cutoff_year=hist_cutoff_year
    )
    
    # 提取livestock参数（从历史文件和计算结果）
    parameters = calculator.extract_livestock_parameters(years, hist_cutoff_year)
    
    return {
        'emissions': emissions,
        'parameters': parameters
    }

