# -*- coding: utf-8 -*-
"""
gce_emissions_complete.py - 作物排放完整计算模块（与 FAOSTAT 对齐）

模块定位
--------
本模块实现作物部门（GCE）的完整排放计算引擎，与 livestock 模块类似，提供
无状态、可组合的计算函数。

覆盖的四个主要过程：
1. Crop residues (直接 N2O) - 秸秆还田释放 N2O
2. Burning crop residues (CH4/N2O) - 秸秆焚烧释放 CH4 和 N2O
3. Rice cultivation (CH4) - 稻作水田释放 CH4
4. Synthetic fertilizers (N2O) - 合成肥施用释放 N2O

数据流：
  生产量 (production_t) + 参数 (residue N content, EF) 
  → 残体N含量 or DM含量 
  → 乘以排放因子 
  → 排放量 (N2O_kt, CH4_kt)

关键数据来源：
  - 历史产量: Production_Crops_Livestock_E_All_Data_NOFLAG.csv
  - 历史排放: Emissions_crops_E_All_Data_NOFLAG.csv
  - 参数表: Code/src/GCE_parameters.xlsx (GCE_parameters sheet)
  - dict_v3: 用于 Item 名称映射与 M49 country 过滤
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config_paths import get_input_base, get_src_base


def _norm_m49(val: str) -> str:
    """
    规范化M49代码为3位数字字符串格式
    
    Excel中M49可能以多种格式存储:
    - 带前置引号: '004, '840 (Excel文本格式防止前导零丢失)
    - 纯数字字符串: "004", "840"
    - 整数: 4, 840
    
    规范化为: "'004", "'840" ('xxx格式：单引号+3位数字字符串)
    """
    try:
        s = str(val).strip()
        # 移除Excel文本格式的前置引号
        if s.startswith("'"):
            s = s[1:]
        # 转为整数再转回字符串，补齐3位，添加单引号前缀
        return f"'{int(s):03d}"  # ✅ 'xxx格式
    except (ValueError, AttributeError):
        # 如果转换失败，返回原字符串
        return str(val)


class CropEmissionsCalculator:
    """
    作物排放计算器
    
    职责：
    1. 从 GCE_parameters.xlsx 读取参数
    2. 从历史排放 CSV 读取历史数据
    3. 计算未来年份的各过程排放
    4. 支持情景和 MC 模拟
    """
    
    def __init__(self, 
                 gle_params_path: str,
                 dict_v3_path: str,
                 hist_emissions_crop_path: str,
                 fertilizer_efficiency_path: Optional[str] = None):
        """
        初始化 Crop Emissions Calculator
        
        Args:
            gle_params_path: GCE_parameters.xlsx 路径
            dict_v3_path: dict_v3.xlsx 路径
            hist_emissions_crop_path: Emissions_crops_E_All_Data_NOFLAG.csv 路径
            fertilizer_efficiency_path: Fertilizer_efficiency.xlsx 路径（用于Synthetic fertilizers历史分配）
        """
        self.gle_params_path = gle_params_path
        self.dict_v3_path = dict_v3_path
        self.hist_emissions_crop_path = hist_emissions_crop_path
        self.fertilizer_efficiency_path = fertilizer_efficiency_path
        
        # 加载参数表（GCE_parameters）
        self._load_gce_parameters()
        
        # 加载历史排放数据（用于历史年份直接读取）
        self._load_historical_emissions()
        
        # 加载肥料效率数据（用于Synthetic fertilizers历史分配）
        self._load_fertilizer_efficiency()
        
    def _load_gce_parameters(self) -> None:
        """从 GCE_parameters.xlsx 加载参数表"""
        if not os.path.exists(self.gle_params_path):
            print(f"WARNING: parameter file not found: {self.gle_params_path}")
            self.gce_params = pd.DataFrame()
            return
        
        try:
            # 读取 GCE_parameters sheet
            self.gce_params = pd.read_excel(self.gle_params_path, sheet_name='GCE_parameters')
            print(f"[INFO] 加载 GCE_parameters: {len(self.gce_params)} 行")
            print(f"[DEBUG] 列名: {list(self.gce_params.columns)}")
            
            # 仅保留 Select=1 的行
            if 'Select' in self.gce_params.columns:
                before = len(self.gce_params)
                self.gce_params = self.gce_params[self.gce_params['Select'] == 1].copy()
                after = len(self.gce_params)
                print(f"[INFO] 过滤 Select=1: {before} → {after} 行")
            
            # 规范化 M49 编码（如果存在）
            if 'M49_Country_Code' in self.gce_params.columns:
                self.gce_params['M49_Country_Code'] = self.gce_params['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"❌ 加载 GCE_parameters 失败: {e}")
            self.gce_params = pd.DataFrame()
    
    def _load_historical_emissions(self) -> None:
        """从 Emissions_crops_E_All_Data_NOFLAG.csv 加载历史排放数据"""
        if not os.path.exists(self.hist_emissions_crop_path):
            print(f"WARNING: historical emissions file not found: {self.hist_emissions_crop_path}")
            self.hist_emissions_crop = pd.DataFrame()
            return
        
        try:
            self.hist_emissions_crop = pd.read_csv(self.hist_emissions_crop_path, encoding='utf-8')
            print(f"[INFO] 加载历史 Crop 排放: {len(self.hist_emissions_crop)} 行")
            
            # 规范化 M49 编码
            if 'M49_Country_Code' in self.hist_emissions_crop.columns:
                self.hist_emissions_crop['M49_Country_Code'] = self.hist_emissions_crop['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"❌ 加载历史排放失败: {e}")
            self.hist_emissions_crop = pd.DataFrame()
    
    def _load_fertilizer_efficiency(self) -> None:
        """从 Fertilizer_efficiency.xlsx 加载肥料效率数据"""
        if not self.fertilizer_efficiency_path or not os.path.exists(self.fertilizer_efficiency_path):
            print(f"[INFO] Fertilizer_efficiency.xlsx 未提供或不存在，将使用默认方法")
            self.fertilizer_efficiency = pd.DataFrame()
            return
        
        try:
            self.fertilizer_efficiency = pd.read_excel(self.fertilizer_efficiency_path)
            print(f"[INFO] 加载 Fertilizer_efficiency: {len(self.fertilizer_efficiency)} 行")
            
            # 规范化 M49 编码
            if 'M49_Country_Code' in self.fertilizer_efficiency.columns:
                self.fertilizer_efficiency['M49_Country_Code'] = self.fertilizer_efficiency['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"WARNING: 加载 Fertilizer_efficiency 失败: {e}")
            self.fertilizer_efficiency = pd.DataFrame()
    
    def _standardize_results(self, results: List[Dict]) -> pd.DataFrame:
        """
        将计算结果标准化为统一的列结构
        返回: M49_Country_Code, Item, year, process, CH4_kt, N2O_kt, CO2_kt
        """
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # 确保包含所有必需的列（缺失的设为0）
        for col in ['M49_Country_Code', 'Item', 'year', 'process']:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        for gas_col in ['CH4_kt', 'N2O_kt', 'CO2_kt']:
            if gas_col not in df.columns:
                df[gas_col] = 0.0
        
        # 处理多行结构（如果存在 'gas' 列）
        if 'gas' in df.columns:
            # 按气体类型聚合，确保每行有正确的气体值
            for idx, row in df.iterrows():
                gas = str(row.get('gas', ''))
                if gas == 'CH4':
                    for col in df.columns:
                        if col.startswith('CH4') and col.endswith('_kt') and col != 'CH4_kt':
                            df.at[idx, 'CH4_kt'] = row.get(col, 0)
                elif gas == 'N2O':
                    for col in df.columns:
                        if col.startswith('N2O') and col.endswith('_kt') and col != 'N2O_kt':
                            df.at[idx, 'N2O_kt'] = row.get(col, 0)
                elif gas == 'CO2':
                    for col in df.columns:
                        if col.startswith('CO2') and col.endswith('_kt') and col != 'CO2_kt':
                            df.at[idx, 'CO2_kt'] = row.get(col, 0)
        
        return df[['M49_Country_Code', 'Item', 'year', 'process', 'CH4_kt', 'N2O_kt', 'CO2_kt']]
    
    def get_parameter(self, 
                     m49_code: str, 
                     item: str, 
                     process: str, 
                     param_name: str,
                     year: int) -> Optional[float]:
        """
        从参数表获取单个参数值
        
        查找逻辑：
        1. M49 精确匹配 + Item + Process + ParamName
        2. 若无，尝试 M49='000' (Global) 回退
        3. 若目标年份列不存在或值为NaN，向前查找最近可用年份的值
        """
        # 规范化 M49
        m49 = _norm_m49(m49_code)
        
        # 识别所有年份列（按年份排序）
        year_cols = sorted([int(col) for col in self.gce_params.columns if str(col).isdigit()])
        if not year_cols:
            return None
        
        def _find_value_in_row(row: pd.Series, target_year: int) -> Optional[float]:
            """
            从行中查找参数值，支持年份外推和NaN处理
            
            策略：
            1. 尝试目标年份
            2. 如果不存在或为NaN，向前查找最近可用年份
            """
            # 年份列名：直接用字符串"2000", "2020"等（不带Y前缀）
            year_col = str(target_year)
            
            # 如果目标年份列存在且有值，直接返回
            if year_col in row.index and pd.notna(row[year_col]):
                return float(row[year_col])
            
            # 否则向前查找：使用<=目标年份的最大可用年份
            available_years = [y for y in year_cols if y <= target_year]
            if not available_years:
                # 如果目标年份早于所有数据，使用最早年份
                available_years = [min(year_cols)]
            
            # 从最近年份向前查找第一个非NaN值
            for search_year in reversed(available_years):
                search_col = str(search_year)
                if search_col in row.index and pd.notna(row[search_col]):
                    return float(row[search_col])
            
            return None
        
        # 1. 精确匹配（M49 + Item + Process + ParamName）
        mask = (
            (self.gce_params['M49_Country_Code'].astype(str) == m49) &
            (self.gce_params['Item'].astype(str) == item) &
            (self.gce_params['Process'].astype(str) == process) &
            (self.gce_params['paramName'].astype(str) == param_name)
        )
        
        if mask.any():
            row = self.gce_params[mask].iloc[0]
            val = _find_value_in_row(row, year)
            if val is not None:
                return val
        
        # 2. Global (M49='000' 或 '0') 回退
        for global_m49 in ['000', '0']:
            mask = (
                (self.gce_params['M49_Country_Code'].astype(str) == global_m49) &
                (self.gce_params['Item'].astype(str) == item) &
                (self.gce_params['Process'].astype(str) == process) &
                (self.gce_params['paramName'].astype(str) == param_name)
            )
            if mask.any():
                row = self.gce_params[mask].iloc[0]
                val = _find_value_in_row(row, year)
                if val is not None:
                    return val
        
        return None
    
    def compute_crop_residues_n2o(self,
                                  production_df: pd.DataFrame,
                                  year: int,
                                  scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算 Crop residues 直接 N2O 排放
        
        单位说明（来自GCE_parameters.xlsx）：
        - Residue N content: kg N / tonne product
        - Emission factor: kg N2O / kg N (注意：直接是N2O，不是N2O-N)
        
        计算公式：
        1. residue_n_kg = production_t × Residue_N_content  [tonne × kg/tonne = kg N]
        2. n2o_kg = residue_n_kg × EF                       [kg N × kg N2O/kg N = kg N2O]
        3. n2o_kt = n2o_kg / 1e6                            [kg → kt]
        """
        results = []
        
        # 过程名称和参数名称
        process = "Crop residues"
        residue_n_param = "Residue N content"
        ef_param = "Emission factor"
        
        for _, row in production_df.iterrows():
            m49 = str(row.get('M49_Country_Code', ''))
            item = str(row.get('Item', ''))
            prod_t = float(row.get('production_t', 0))
            
            if prod_t <= 0:
                continue
            
            # 获取残体 N 含量 (kg N / tonne product)
            residue_n = self.get_parameter(m49, item, process, residue_n_param, year)
            if residue_n is None:
                continue
            
            # 获取排放因子 (kg N2O / kg N) - 注意：直接是N2O，不是N2O-N
            ef = self.get_parameter(m49, item, process, ef_param, year)
            if ef is None:
                continue
            
            # 应用情景参数：排放因子调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                # 查找匹配的因子：(country, commodity, process, year)
                # 需要从 m49 获取 country 名称
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, process, year), 1.0)
                # 如果没有精确匹配，尝试 'All' 通配
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', process, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                ef *= ef_mult
            
            # ✅ 修复计算公式：
            # residue_n单位是 kg N / tonne product，乘以production_t(吨)直接得到 kg N
            residue_n_total_kg = prod_t * residue_n  # [tonne × kg/tonne = kg N]
            # EF单位是 kg N2O / kg N，直接得到 kg N2O（不需要×44/28转换）
            n2o_kg = residue_n_total_kg * ef  # [kg N × kg N2O/kg N = kg N2O]
            n2o_kt = n2o_kg / 1e6  # [kg → kt]
            
            # 🔍 调试: U.S. Barley Crop residues
            if _norm_m49(m49) == "'840" and 'Barley' in item:
                print(f"[DEBUG Crop residues] U.S. Barley Y{year}:")
                print(f"  prod_t={prod_t:.2f} t, residue_n={residue_n:.6f} kg/t, ef={ef:.6f}")
                print(f"  N2O_kt={n2o_kt:.4f} kt, CO2eq_kt={n2o_kt * 273.0:.2f} kt")
            
            results.append({
                'M49_Country_Code': m49,
                'Item': item,
                'year': year,
                'process': process,
                'CH4_kt': 0.0,
                'N2O_kt': n2o_kt,
                'CO2_kt': 0.0
            })
        
        return self._standardize_results(results)
    
    def compute_burning_ch4_n2o(self,
                               production_df: pd.DataFrame,
                               year: int,
                               scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算 Burning crop residues 的 CH4 和 N2O 排放
        
        单位说明（来自参数表）：
        - Biomass burning DM content: kg DM / tonne product
        - Emission factor (CH4): kg CH4/kg DM
        - Emission factor (N2O): kg N2O/kg DM
        
        计算公式：
        1. biomass_dm_kg = production_t × dm_content   [tonne × kg/tonne = kg DM]
        2. ch4_kg = biomass_dm_kg × ef_ch4             [kg DM × kg CH4/kg DM = kg CH4]
        3. n2o_kg = biomass_dm_kg × ef_n2o             [kg DM × kg N2O/kg DM = kg N2O]
        
        注意：当前参数表可能不区分 CH4/N2O 的 EF，暂时使用 IPCC 默认比例
        IPCC 典型值：EF_CH4 ≈ 0.0027 kg/kg DM, EF_N2O ≈ 0.00007 kg/kg DM
        """
        results = []
        process_display = "Burning crop residues"
        
        for _, row in production_df.iterrows():
            m49 = str(row.get('M49_Country_Code', ''))
            item = str(row.get('Item', ''))
            prod_t = float(row.get('production_t', 0))
            
            if prod_t <= 0:
                continue
            
            ch4_kt = 0.0
            n2o_kt = 0.0
            
            # 获取生物质干物质含量 (kg DM / tonne product)
            dm_content = self.get_parameter(m49, item, process_display, "Biomass burning DM content", year)
            
            # 获取排放因子 (kg/kg DM)
            # 注意：参数表可能只有一个 EF，此时使用 IPCC 比例分配
            ef = self.get_parameter(m49, item, process_display, "Emission factor", year)
            
            # 应用情景参数：排放因子调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                if ef is not None:
                    ef *= ef_mult
            
            if dm_content is not None and ef is not None and dm_content > 0 and ef > 0:
                # 计算焚烧的生物质干物质量 (kg DM)
                # 注意：dm_content 单位是 kg DM / tonne product，所以直接乘以 production_t(tonne)
                biomass_dm_kg = prod_t * dm_content
                
                # EF 单位是 kg/kg DM
                # 如果参数表只有一个 EF（假设是总排放因子或 CH4 的 EF）
                # 使用 IPCC 典型比例：CH4:N2O ≈ 27:0.07 (质量比约 39:1)
                # 但更常见的做法是 EF 就是 CH4 的，N2O 用独立参数
                
                # 判断 ef 的量级来确定是 CH4 还是总排放
                if ef > 0.001:  # 大于 0.001 kg/kg，可能是 CH4 的 EF
                    ch4_kg = biomass_dm_kg * ef
                    # IPCC: EF_N2O ≈ EF_CH4 / 39
                    n2o_kg = ch4_kg / 39.0
                else:  # 可能是 N2O 的 EF
                    n2o_kg = biomass_dm_kg * ef
                    ch4_kg = n2o_kg * 39.0  # 反推 CH4
                
                ch4_kt = ch4_kg / 1e6
                n2o_kt = n2o_kg / 1e6
            
            if ch4_kt > 0 or n2o_kt > 0:
                results.append({
                    'M49_Country_Code': m49,
                    'Item': item,
                    'year': year,
                    'process': process_display,
                    'CH4_kt': ch4_kt,
                    'N2O_kt': n2o_kt,
                    'CO2_kt': 0.0
                })
        
        return self._standardize_results(results)
    
    def compute_rice_ch4(self,
                        harvest_area_df: pd.DataFrame,
                        year: int,
                        scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算 Rice cultivation 的 CH4 排放
        
        单位说明（来自GCE_parameters.xlsx）：
        - Emission factor: kg CH4/ha
        
        计算公式：
        1. ch4_kg = area_ha × EF     [ha × kg/ha = kg CH4]
        2. ch4_kt = ch4_kg / 1e6     [kg → kt]
        """
        results = []
        process_display = "Rice cultivation"
        
        # DEBUG: 检查传入DataFrame的列名和形状
        print(f"[DEBUG Rice] Year={year}, 收到DataFrame形状: {harvest_area_df.shape}")
        print(f"[DEBUG Rice] 列名: {list(harvest_area_df.columns)}")
        
        for _, row in harvest_area_df.iterrows():
            m49 = str(row.get('M49_Country_Code', ''))
            item = str(row.get('Item', ''))
            commodity = str(row.get('commodity', item))
            area_ha = float(row.get('harvest_area_ha', 0))
            if area_ha == 0:
                area_ha = float(row.get('harvested_area_ha', 0))
            
            # Rice匹配检查
            is_rice = ('Rice' in str(item) or 'Rice' in str(commodity) or 
                      'rice' in str(item).lower() or 'rice' in str(commodity).lower())
            
            # DEBUG: 对中国Rice输出详细信息
            if m49 in ('156', "'156") and is_rice:
                print(f"[DEBUG Rice China] Year={year}, Item={item}, area_ha={area_ha:.2f}")
            
            if area_ha <= 0 or not is_rice:
                continue
            
            # 标准化Item为"Rice"以匹配参数表
            item_for_param = "Rice"
            
            # 获取排放因子 EF (kg CH4/ha)
            ef0 = self.get_parameter(m49, item_for_param, process_display, "Emission factor", year)
            if ef0 is None:
                continue
            
            # 应用情景参数：排放因子调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                ef0 *= ef_mult
            
            # DEBUG: 对中国Rice输出详细计算
            if m49 in ('156', "'156"):
                print(f"[DEBUG Rice China] EF={ef0:.2f} kg CH4/ha, CH4={ef0*area_ha/1e6:.2f} kt")
            
            # ✅ 修复计算公式：EF单位是 kg CH4/ha
            ch4_kg = ef0 * area_ha  # [kg CH4/ha × ha = kg CH4]
            ch4_kt = ch4_kg / 1e6   # [kg → kt]
            
            results.append({
                'M49_Country_Code': m49,
                'Item': item,  # 保留原始Item名称
                'year': year,
                'process': process_display,
                'CH4_kt': ch4_kt,
                'N2O_kt': 0.0,
                'CO2_kt': 0.0
            })
        
        return self._standardize_results(results)
    
    def compute_synthetic_fert_n2o(self,
                                  fert_df: pd.DataFrame,
                                  year: int,
                                  scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算 Synthetic fertilizers 的 N2O 排放（未来年份）
        
        逻辑：
        1. 使用 synthetic_fertilizer_direct_N2O 过程参数
        2. 对每个 M49-Item 读取施肥率和排放因子
        3. 计算 N2O 直接排放
        """
        results = []
        process_display = "Synthetic fertilizers"
        
        for _, row in fert_df.iterrows():
            m49 = str(row.get('M49_Country_Code', ''))
            item = str(row.get('Item', ''))
            area_ha = float(row.get('harvest_area_ha', 0))
            if area_ha == 0:
                area_ha = float(row.get('harvested_area_ha', 0))
            
            if area_ha <= 0:
                continue
            
            # 获取施肥率（使用显示名称，注意参数名拼写）
            fert_rate = self.get_parameter(m49, item, process_display, "Fertlizer rate", year)
            if fert_rate is None:
                continue
            
            # 应用情景参数：施肥密度调整
            if scenario_params and 'fertilizer_rate_multiplier' in scenario_params:
                fert_mult_dict = scenario_params['fertilizer_rate_multiplier']
                country = row.get('country', '')
                fert_mult = fert_mult_dict.get((country, item, year), 1.0)
                if fert_mult == 1.0:
                    fert_mult = fert_mult_dict.get((country, 'All', year), 1.0)
                fert_rate *= fert_mult
            
            # 计算施肥量 (kg N)
            fert_amount_kg = area_ha * fert_rate
            
            # 获取排放因子 EF1（使用显示名称）
            # 单位：kg N2O/kg N （已经是N2O，不是N2O-N，无顸44/28转换）
            ef = self.get_parameter(m49, item, process_display, "Emission factor", year)
            if ef is None:
                continue
            
            # 应用情景参数：排放因子调整
            if scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_mult_dict = scenario_params['emission_factor_multiplier']
                country = row.get('country', '')
                ef_mult = ef_mult_dict.get((country, item, process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, item, 'All', year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', process_display, year), 1.0)
                if ef_mult == 1.0:
                    ef_mult = ef_mult_dict.get((country, 'All', 'All', year), 1.0)
                ef *= ef_mult
            
            # 计算 N2O 排放（EF单位已经是 kg N2O/kg N）
            n2o_kg = fert_amount_kg * ef
            n2o_kt = n2o_kg / 1e6
            
            results.append({
                'M49_Country_Code': m49,
                'Item': item,
                'year': year,
                'process': process_display,
                'CH4_kt': 0.0,
                'N2O_kt': n2o_kt,
                'CO2_kt': 0.0
            })
        
        return self._standardize_results(results)
    
    def _get_historical_emissions(self, year: int) -> Dict[str, pd.DataFrame]:
        """
        从历史排放文件中读取指定年份的排放数据
        
        文件格式：宽格式，每年一列 (Y2000, Y2001, ...)
        Element列包含: 'Crop residues (Emissions N2O)', 'Burning crop residues (Emissions CH4)', etc.
        
        返回: {过程名: DataFrame}，数据已标准化为 (M49, Item, year, process, CH4_kt, N2O_kt, CO2_kt)
        """
        if self.hist_emissions_crop.empty:
            return {}
        
        results = {}
        
        # 检查是否是宽格式（有年份列如Y2000, Y2001等）
        year_cols = [col for col in self.hist_emissions_crop.columns if col.startswith('Y')]
        year_col_name = f'Y{year}'
        
        if year_col_name not in self.hist_emissions_crop.columns:
            print(f"WARNING: year column {year_col_name} not found in historical file")
            return {}
        
        # 提取该年份的数据
        # Element -> (process, gas_type) 的映射
        # ⚠️ 注意：只使用 Total (Emissions N2O)，不要同时使用 Direct emissions 避免重复统计
        element_map = {
            'Crop residues (Emissions N2O)': ('Crop residues', 'N2O'),
            # 'Crop residues (Direct emissions N2O)': ('Crop residues', 'N2O'),  # 已包含在 Total 中，不要重复
            'Burning crop residues (Emissions N2O)': ('Burning crop residues', 'N2O'),
            'Burning crop residues (Emissions CH4)': ('Burning crop residues', 'CH4'),
            'Rice cultivation (Emissions CH4)': ('Rice cultivation', 'CH4'),
            'Synthetic fertilizers (Emissions N2O)': ('Synthetic fertilizers', 'N2O'),
        }
        
        # 按过程收集排放数据
        process_data = {
            'Crop residues': [],
            'Burning crop residues': [],
            'Rice cultivation': [],
            'Synthetic fertilizers': []
        }
        
        for element, (process, gas_type) in element_map.items():
            # 过滤该 Element 的所有数据
            elem_mask = self.hist_emissions_crop['Element'] == element
            if not elem_mask.any():
                continue
            
            elem_df = self.hist_emissions_crop[elem_mask].copy()
            
            # 提取该年份的值
            elem_df['value'] = elem_df[year_col_name]
            
            # 保留必要的列
            elem_df = elem_df[['M49_Country_Code', 'Item', 'value']].copy()
            elem_df['year'] = year
            elem_df['process'] = process
            elem_df['gas'] = gas_type
            
            # 移除 NaN 值
            elem_df = elem_df.dropna(subset=['value'])
            
            if not elem_df.empty:
                process_data[process].append(elem_df)
        
        # 为每个过程合并并透视成 CH4_kt, N2O_kt, CO2_kt 列
        for process_name in ['Crop residues', 'Burning crop residues', 'Rice cultivation', 'Synthetic fertilizers']:
            if not process_data[process_name]:
                continue
            
            # 合并该过程的所有数据
            df = pd.concat(process_data[process_name], ignore_index=True)
            
            # Synthetic fertilizers特殊处理：按19种Item分配
            if process_name == 'Synthetic fertilizers' and not self.fertilizer_efficiency.empty:
                df = self._allocate_synthetic_fertilizers_by_items(df, year)
            
            # 转换为宽格式: 每个气体类型一列
            pivot_df = df.pivot_table(
                index=['M49_Country_Code', 'Item', 'year', 'process'],
                columns='gas',
                values='value',
                aggfunc='sum'
            ).reset_index()
            
            # 清除列名索引名称
            pivot_df.columns.name = None
            
            # 确保所有气体列都存在（缺失的为0）
            for gas in ['CH4', 'N2O', 'CO2']:
                if gas not in pivot_df.columns:
                    pivot_df[f'{gas}_kt'] = 0.0
                else:
                    pivot_df.rename(columns={gas: f'{gas}_kt'}, inplace=True)
            
            # 标准化列名和顺序
            if 'CH4_kt' not in pivot_df.columns:
                pivot_df['CH4_kt'] = 0.0
            if 'N2O_kt' not in pivot_df.columns:
                pivot_df['N2O_kt'] = 0.0
            if 'CO2_kt' not in pivot_df.columns:
                pivot_df['CO2_kt'] = 0.0
            
            results[process_name] = pivot_df[['M49_Country_Code', 'Item', 'year', 'process', 'CH4_kt', 'N2O_kt', 'CO2_kt']].reset_index(drop=True)
        
        return results
    
    def _allocate_synthetic_fertilizers_by_items(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        按19种Item分配Synthetic fertilizers历史排放
        
        逻辑：
        1. 读取Fertilizer_efficiency.xlsx中的N_contentModi_Yxxxx列
        2. 使用dict_v3的Item_Fertilizer_Map映射将Item名称标准化为Item_Emis
        3. 对于无法映射的Item（如Others_crops），将其N施用量等比例分配到其他可映射的Item上
        4. 对每个M49_Country_Code，计算各Item的N施用量占比
        5. 用占比分配总排放到标准化后的Item
        
        Args:
            df: 原始排放数据（Item='Nutrient nitrogen N (total)'）
            year: 年份
        
        Returns:
            分配后的排放数据（使用Item_Emis标准名称）
        """
        if self.fertilizer_efficiency.empty:
            return df
        
        # 年份列名
        n_content_col = f'N_contentModi_Y{year}'
        if n_content_col not in self.fertilizer_efficiency.columns:
            print(f"WARNING: {n_content_col} not found in Fertilizer_efficiency")
            return df
        
        # 提取该年份的N施用量数据
        fert_eff = self.fertilizer_efficiency[['M49_Country_Code', 'Item', n_content_col]].copy()
        fert_eff = fert_eff.rename(columns={n_content_col: 'n_content'})
        fert_eff = fert_eff.dropna(subset=['n_content'])
        
        if fert_eff.empty:
            return df
        
        # ✅ 关键修复：从 dict_v3 读取 Item_Fertilizer_Map -> Item_Emis 的映射
        # 这是官方的标准映射关系
        item_name_mapping = self._get_fertilizer_item_mapping()
        
        # 标记哪些Item可以映射
        fert_eff['Item_Emis'] = fert_eff['Item'].map(item_name_mapping)
        fert_eff['is_mappable'] = fert_eff['Item_Emis'].notna()
        
        # ⚠️ 舍弃无法映射的Item（如Others_crops），不进行等比例分配
        # 原逻辑：将无法映射的Item的N施用量等比例分配到其他可映射的Item
        # 现逻辑：直接舍弃无法映射的Item，只保留可映射的部分
        result_rows = []
        for m49, group in fert_eff.groupby('M49_Country_Code'):
            mappable = group[group['is_mappable']].copy()
            # unmappable = group[~group['is_mappable']].copy()  # 注释掉：不再处理Others_crops
            
            if mappable.empty:
                # 如果没有可映射的Item，跳过该国家
                continue
            
            # 直接使用可映射的Item，不进行Others_crops的等比例分配
            for _, row in mappable.iterrows():
                result_rows.append({
                    'M49_Country_Code': m49,
                    'Item': row['Item_Emis'],  # 使用标准化的Item_Emis名称
                    'n_content': row['n_content']  # 原始N施用量，不加上Others_crops的分配
                })
            
            # === 以下代码注释掉：不再将Others_crops等比例分配 ===
            # # 计算可映射Item的N施用量总和
            # mappable_n_total = mappable['n_content'].sum()
            # 
            # if unmappable.empty:
            #     # 没有不可映射的Item，直接使用原始数据
            #     for _, row in mappable.iterrows():
            #         result_rows.append({
            #             'M49_Country_Code': m49,
            #             'Item': row['Item_Emis'],  # 使用标准化的Item_Emis名称
            #             'n_content': row['n_content']
            #         })
            # else:
            #     # 有不可映射的Item，将其等比例分配
            #     unmappable_n_total = unmappable['n_content'].sum()
            #     
            #     for _, row in mappable.iterrows():
            #         # 原始N施用量
            #         original_n = row['n_content']
            #         # 该Item占可映射Item总量的比例
            #         share_of_mappable = original_n / mappable_n_total if mappable_n_total > 0 else 0
            #         # 分配到该Item的不可映射部分
            #         extra_n = unmappable_n_total * share_of_mappable
            #         # 总N施用量 = 原始 + 分配的
            #         total_n = original_n + extra_n
            #         
            #         result_rows.append({
            #             'M49_Country_Code': m49,
            #             'Item': row['Item_Emis'],  # 使用标准化的Item_Emis名称
            #             'n_content': total_n
            #         })
            # === 注释结束 ===
        
        if not result_rows:
            return df
        
        fert_eff_redistributed = pd.DataFrame(result_rows)
        
        # 计算每个M49的Item占比
        fert_eff_redistributed['total_n'] = fert_eff_redistributed.groupby('M49_Country_Code')['n_content'].transform('sum')
        fert_eff_redistributed['share'] = fert_eff_redistributed['n_content'] / fert_eff_redistributed['total_n']
        fert_eff_redistributed = fert_eff_redistributed[fert_eff_redistributed['share'] > 0]  # 移除零占比
        
        # 合并排放数据（通常Item='Nutrient nitrogen N (total)'）
        # 提取总排放
        total_emis = df[df['Item'].str.contains('Nutrient nitrogen N', na=False, case=False)].copy()
        
        if total_emis.empty:
            # 如果没有total，返回原数据
            return df
        
        # 分配到19种Item
        allocated = []
        for _, emis_row in total_emis.iterrows():
            m49 = emis_row['M49_Country_Code']
            total_value = emis_row['value']
            
            # 该国家的Item占比
            country_shares = fert_eff_redistributed[fert_eff_redistributed['M49_Country_Code'] == m49]
            
            if country_shares.empty:
                # 如果没有分配数据，保留总量
                allocated.append(emis_row)
                continue
            
            # 按占比分配
            for _, share_row in country_shares.iterrows():
                new_row = emis_row.copy()
                new_row['Item'] = share_row['Item']  # 已经是标准化的Item_Emis名称
                new_row['value'] = total_value * share_row['share']
                allocated.append(new_row)
        
        if allocated:
            df_allocated = pd.DataFrame(allocated)
            # 移除原来的total行
            df_no_total = df[~df['Item'].str.contains('Nutrient nitrogen N', na=False, case=False)]
            # 合并
            df = pd.concat([df_no_total, df_allocated], ignore_index=True)
        
        return df
    
    def _get_fertilizer_item_mapping(self) -> dict:
        """
        从 dict_v3 的 Emis_item sheet 获取 Item_Fertilizer_Map -> Item_Emis 的映射
        
        Returns:
            dict: {Item_Fertilizer_Map: Item_Emis}
        """
        if not os.path.exists(self.dict_v3_path):
            # 回退到硬编码映射
            return {
                'Maize': 'Maize (corn)',
                'Potato': 'Potatoes', 
                'Soybean': 'Soya beans',
                'Sugarcane': 'Sugar cane',
                'Barley': 'Barley',
                'Cassava': 'Cassava',
                'Cotton': 'Cotton',
                'Fruits': 'Fruits',
                'Groundnut': 'Groundnut',
                'Oilpalm': 'Oilpalm',
                'Rapeseed': 'Rapeseed',
                'Rice': 'Rice',
                'Rye': 'Rye',
                'Sorghum': 'Sorghum',
                'Sugarbeet': 'Sugarbeet',
                'Sweetpotato': 'Sweetpotato',
                'Vegetables': 'Vegetables',
                'Wheat': 'Wheat',
                'sunflower': 'sunflower',
            }
        
        try:
            emis_item_df = pd.read_excel(self.dict_v3_path, sheet_name='Emis_item')
            synth_items = emis_item_df[emis_item_df['Process'] == 'Synthetic fertilizers']
            
            mapping = {}
            for _, row in synth_items.iterrows():
                fert_map = row.get('Item_Fertilizer_Map')
                item_emis = row.get('Item_Emis')
                if pd.notna(fert_map) and pd.notna(item_emis):
                    mapping[fert_map] = item_emis
            
            return mapping
        except Exception as e:
            print(f"WARNING: 无法从dict_v3读取映射: {e}")
            # 回退到硬编码映射
            return {
                'Maize': 'Maize (corn)',
                'Potato': 'Potatoes', 
                'Soybean': 'Soya beans',
                'Sugarcane': 'Sugar cane',
            }
    
    def run_full_calculation(self,
                            production_df: pd.DataFrame,
                            harvest_area_df: pd.DataFrame,
                            years: List[int],
                            scenario_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        运行完整的 crop 排放计算
        
        逻辑：
        1. 历史年份 (≤2020): 直接从 Emissions_crops_E_All_Data_NOFLAG.csv 读取
        2. 未来年份 (>2020): 通过参数计算得到
        
        Args:
            production_df: production 数据 (M49_Country_Code, Item, year, production_t)
            harvest_area_df: harvest area 数据 (M49_Country_Code, Item, year, harvested_area_ha)
            years: 计算年份列表
            scenario_params: 情景参数
        
        Returns:
            {'Crop residues': df, 'Burning crop residues': df, 'Rice cultivation': df, 'Synthetic fertilizers': df}
        """
        all_results = {}
        
        print(f"\n{'='*60}")
        print("Crop Emissions Calculation")
        print(f"{'='*60}")
        
        # 按历史/未来分类年份
        historical_years = [y for y in years if y <= 2020]
        future_years = [y for y in years if y > 2020]
        
        # 1. 处理历史年份 - 直接从排放文件读取
        if historical_years:
            print(f"[历史年份] 从排放文件直接读取: {historical_years}")
            for year in historical_years:
                hist_data = self._get_historical_emissions(year)
                for process, df in hist_data.items():
                    if not df.empty:
                        all_results.setdefault(process, []).append(df)
        
        # 2. 处理未来年份 - 通过参数计算
        if future_years:
            print(f"[未来年份] 通过参数计算: {future_years}")
            for year in future_years:
                print(f"处理年份: {year}")
                
                # 过滤该年份的数据
                prod_year = production_df[production_df['year'] == year]
                area_year = harvest_area_df[harvest_area_df['year'] == year]
                
                if not prod_year.empty:
                    # 1. Crop residues N2O
                    res = self.compute_crop_residues_n2o(prod_year, year, scenario_params)
                    if not res.empty:
                        all_results.setdefault('Crop residues', []).append(res)
                    
                    # 2. Burning crop residues
                    burn = self.compute_burning_ch4_n2o(prod_year, year, scenario_params)
                    if not burn.empty:
                        all_results.setdefault('Burning crop residues', []).append(burn)
                
                if not area_year.empty:
                    # 3. Rice cultivation CH4
                    rice = self.compute_rice_ch4(area_year, year, scenario_params)
                    if not rice.empty:
                        all_results.setdefault('Rice cultivation', []).append(rice)
                    
                    # 4. Synthetic fertilizers N2O
                    fert = self.compute_synthetic_fert_n2o(area_year, year, scenario_params)
                    if not fert.empty:
                        all_results.setdefault('Synthetic fertilizers', []).append(fert)
        
        # 合并各过程的结果
        final_results = {}
        for process, dfs in all_results.items():
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                final_results[process] = combined
                print(f"[OK] {process}: {len(combined)} rows")
        
        return final_results


def run_crop_emissions(production_df: pd.DataFrame,
                      harvest_area_df: pd.DataFrame,
                      years: List[int],
                      gle_params_path: str,
                      dict_v3_path: str,
                      hist_emissions_crop_path: str,
                      fertilizer_efficiency_path: Optional[str] = None,
                      scenario_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    主函数：运行 Crop 排放计算
    
    Args:
        production_df: production 数据框
        harvest_area_df: harvest area 数据框
        years: 计算年份
        gle_params_path: 参数文件路径
        dict_v3_path: dict_v3 文件路径
        hist_emissions_crop_path: 历史排放 CSV 路径
        fertilizer_efficiency_path: Fertilizer_efficiency.xlsx路径（用于Synthetic fertilizers历史分配）
        scenario_params: 情景参数
    
    Returns:
        {process_name: DataFrame} 字典
    """
    calculator = CropEmissionsCalculator(
        gle_params_path=gle_params_path,
        dict_v3_path=dict_v3_path,
        hist_emissions_crop_path=hist_emissions_crop_path,
        fertilizer_efficiency_path=fertilizer_efficiency_path
    )
    
    return calculator.run_full_calculation(
        production_df=production_df,
        harvest_area_df=harvest_area_df,
        years=years,
        scenario_params=scenario_params
    )


__all__ = [
    'CropEmissionsCalculator',
    'run_crop_emissions',
]
