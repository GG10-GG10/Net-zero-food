# -*- coding: utf-8 -*-
"""
全球土壤排放计算模块 (Global Soil Emissions Module)
用于计算Drained Organic Soils的N2O和CO2排放

包含两部分：
1. 历史排放：从FAO CSV文件读取
2. 未来排放：基于土地利用变化后的面积计算
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def normalize_m49(val) -> str:
    """标准化 M49 代码：格式化为'xxx（单引号+3位数字）
    
    Examples:
        "'001" -> "'001"
        "'156" -> "'156"
        "156" -> "'156"
        1 -> "'001"
    """
    s = str(val).strip()
    # 移除前导单引号
    if s.startswith("'"):
        s = s[1:]
    # 转为整数再格式化为'xxx格式（单引号+三位数字）
    try:
        return f"'{int(s):03d}"  # ✅ 'xxx格式
    except (ValueError, TypeError):
        return f"'{s}" if not s.startswith("'") else s


class DrainedOrganicSoilsEmissions:
    """Drained Organic Soils排放计算类"""
    
    def __init__(self, dict_v3_path: str, soil_params_path: str):
        """
        初始化Drained Organic Soils排放计算器
        
        Args:
            dict_v3_path: dict_v3.xlsx文件路径
            soil_params_path: Soil_parameters.xlsx文件路径
        """
        self.dict_v3_path = dict_v3_path
        self.soil_params_path = soil_params_path
        
        # 从dict_v3读取Emis_item映射表
        self.emis_item_map = self._load_emis_item_map()
        
        # 从Soil_parameters读取参数
        self.soil_params = self._load_soil_parameters()
        
    def _load_emis_item_map(self) -> Dict:
        """从dict_v3的Emis_item sheet读取Process-Item-GHG映射"""
        try:
            df = pd.read_excel(self.dict_v3_path, sheet_name='Emis_item')
            
            # 筛选Drained organic soils相关的行
            mask = df['Process'].str.contains('Drained organic soils', case=False, na=False)
            emis_item_subset = df[mask]
            
            # 构建映射：{process: {item: [ghg_list]}}
            emis_map = {}
            for _, row in emis_item_subset.iterrows():
                process = row['Process']
                item = row['Item_Emis']
                ghg = row['GHG']
                
                if process not in emis_map:
                    emis_map[process] = {}
                if item not in emis_map[process]:
                    emis_map[process][item] = []
                if ghg not in emis_map[process][item]:
                    emis_map[process][item].append(ghg)
            
            logger.info(f"已加载Emis_item映射: {len(emis_map)}个排放过程")
            return emis_map
            
        except Exception as e:
            logger.warning(f"加载Emis_item映射失败: {e}")
            return {}
    
    def _load_soil_parameters(self) -> pd.DataFrame:
        """从Soil_parameters.xlsx读取土壤参数"""
        try:
            df = pd.read_excel(self.soil_params_path)
            logger.info(f"已加载Soil_parameters: {len(df)}行")
            return df
        except Exception as e:
            logger.warning(f"加载Soil_parameters失败: {e}")
            return pd.DataFrame()
    
    def calculate_future_emissions(self,
                                  cropland_area_ha: Dict[Tuple[str, str, int], float],
                                  grassland_area_ha: Dict[Tuple[str, str, int], float]) -> Dict:
        """
        计算未来年份Drained Organic Soils排放
        
        Args:
            cropland_area_ha: {(m49, country, year): area_ha}
            grassland_area_ha: {(m49, country, year): area_ha}
        
        Returns:
            {process_name: DataFrame} 格式的排放结果
        """
        
        logger.info(f"[GSOIL] calculate_future_emissions: cropland输入{len(cropland_area_ha)}条, grassland输入{len(grassland_area_ha)}条")
        
        results = {}
        
        # 处理Cropland organic soils
        cropland_results = self._calculate_organic_soil_emissions(
            area_dict=cropland_area_ha,
            soil_type='Cropland organic soils',
            item_name='Cropland organic soils'
        )
        logger.info(f"[GSOIL] Cropland计算完成: {len(cropland_results)}行")
        if not cropland_results.empty:
            results['Drained organic soils (Cropland)'] = cropland_results
        else:
            logger.warning(f"[GSOIL] Cropland结果为空")
        
        # 处理Grassland organic soils
        grassland_results = self._calculate_organic_soil_emissions(
            area_dict=grassland_area_ha,
            soil_type='Grassland organic soils',
            item_name='Grassland organic soils'
        )
        logger.info(f"[GSOIL] Grassland计算完成: {len(grassland_results)}行")
        if not grassland_results.empty:
            results['Drained organic soils (Grassland)'] = grassland_results
        else:
            logger.warning(f"[GSOIL] Grassland结果为空")
        
        return results
    
    def _calculate_organic_soil_emissions(self,
                                         area_dict: Dict[Tuple[str, str, int], float],
                                         soil_type: str,
                                         item_name: str) -> pd.DataFrame:
        """
        计算有机土壤排放（Cropland或Grassland）
        
        Args:
            area_dict: 面积字典 {(m49, country, year): area_ha}
            soil_type: 土壤类型（用于参数查询，如 'Cropland organic soils'）
            item_name: 项目名称（用于输出）
        
        Returns:
            排放DataFrame
        """
        
        if self.soil_params.empty:
            logger.warning(f"[GSOIL] Soil_parameters为空，无法计算{soil_type}排放")
            return pd.DataFrame()
        
        logger.info(f"[GSOIL] 开始计算{soil_type}，输入{len(area_dict)}条面积记录")
        
        # 标准化 soil_params 中的 M49
        self.soil_params['M49_normalized'] = self.soil_params['M49_Country_Code'].apply(normalize_m49)
        
        # 确保列名为字符串
        self.soil_params.columns = self.soil_params.columns.astype(str)
        
        # 获取可用的年份列（2000-2022）
        # 检查是否有Y前缀
        has_y_prefix = any(col.startswith('Y') and col[1:].isdigit() for col in self.soil_params.columns)
        
        if has_y_prefix:
            year_cols = [f"Y{y}" for y in range(2000, 2023)]
        else:
            year_cols = [str(y) for y in range(2000, 2023)]
            
        available_years = [y for y in year_cols if y in self.soil_params.columns]
        
        if not available_years:
            logger.error(f"[GSOIL] Soil_parameters 中没有年份列！可用列: {list(self.soil_params.columns)}")
            return pd.DataFrame()
        
        logger.info(f"[GSOIL] Soil_parameters 可用年份: {available_years[0]}-{available_years[-1]}")
        
        emissions_list = []
        skipped_no_area_coeff = 0
        skipped_no_ef = 0
        skipped_zero_area = 0
        processed_count = 0
        
        for (m49, country, year), area_ha in area_dict.items():
            if area_ha <= 0:  # 跳过0或负值
                skipped_zero_area += 1
                continue
            
            processed_count += 1
            
            # 标准化输入的 M49
            m49_norm = normalize_m49(m49)
            
            # 选择参数年份：
            # - 历史年份（<=2020）：使用该年份的参数
            # - 未来年份（>2020）：使用2020年作为基准年
            if year <= 2020:
                param_year = min(max(year, 2000), 2020)  # 限制在2000-2020范围内
            else:
                param_year = 2020  # 未来年份使用2020基准年
            param_year_col = f"Y{param_year}" if has_y_prefix else str(param_year)
            
            # 查询 Area correlation
            area_coeff_df = self.soil_params[
                (self.soil_params['M49_normalized'] == m49_norm) &
                (self.soil_params['Item'] == soil_type) &
                (self.soil_params['paramName'] == 'Area correlation')
            ]
            
            if area_coeff_df.empty:
                if processed_count <= 3:  # 只记录前几个失败的例子
                    logger.debug(f"[GSOIL]   跳过 {country}({m49_norm}, 原始{m49}) {year}: 无Area correlation")
                skipped_no_area_coeff += 1
                continue
            
            # 从年份列读取 area coefficient
            area_coeff = area_coeff_df.iloc[0].get(param_year_col, 0)
            if pd.isna(area_coeff) or area_coeff == 0:
                skipped_no_area_coeff += 1
                continue
                
            organic_area_ha = area_ha * area_coeff  # 有机土壤面积 = 总面积 * 系数
            
            if organic_area_ha <= 0:
                continue
            
            # 查询 Emission factors (N2O 和 CO2)
            ef_df = self.soil_params[
                (self.soil_params['M49_normalized'] == m49_norm) &
                (self.soil_params['Item'] == soil_type) &
                (self.soil_params['paramName'] == 'Emission factor')
            ]
            
            if ef_df.empty:
                if processed_count <= 3:
                    logger.debug(f"[GSOIL]   跳过 {country}({m49_norm}) {year}: 无Emission factor")
                skipped_no_ef += 1
                continue
            
            # 遍历所有排放因子行（可能包含 N2O 和 CO2）
            n2o_kt_total = 0.0
            co2_kt_total = 0.0
            has_valid_ef = False
            
            for _, ef_row in ef_df.iterrows():
                emission_factor_t_ha = ef_row.get(param_year_col, 0)  # 单位: t/ha/year
                
                if pd.isna(emission_factor_t_ha) or emission_factor_t_ha == 0:
                    continue
                
                # 排放量 = 有机土壤面积(ha) * 排放因子(t/ha) = t/year = kt/year (因为 t = 0.001 kt)
                emission_kt = organic_area_ha * emission_factor_t_ha / 1000
                
                if emission_kt > 0:
                    has_valid_ef = True
                    param_mms = str(ef_row.get('paramMMS', '')).upper()
                    
                    if param_mms == 'N2O':
                        n2o_kt_total += emission_kt
                    elif param_mms == 'CO2':
                        co2_kt_total += emission_kt
                    # 如果有其他类型（如N2O_CO2用于EF），需要确认其含义，目前假设EF只有N2O或CO2
            
            if has_valid_ef:
                emissions_list.append({
                    'M49_Country_Code': m49_norm,
                    'Country': country,
                    'Year': year,
                    'Item': item_name,
                    'Process': 'Drained organic soils',
                    'GHG': 'N2O' if n2o_kt_total > 0 else 'CO2', # 标记主要GHG，实际输出包含两列
                    'n2o_kt': n2o_kt_total,
                    'co2_kt': co2_kt_total,
                    'ch4_kt': 0.0,
                    'Organic_soil_area_ha': organic_area_ha,
                    'Emission_factor_t_ha': 0.0, # 聚合后不再适用单一EF
                    'Param_year_used': param_year
                })
        
        logger.info(f"[GSOIL] {soil_type}计算完成: 处理{processed_count}条, 跳过零面积{skipped_zero_area}条, 无面积系数{skipped_no_area_coeff}条, 无排放因子{skipped_no_ef}条, 生成{len(emissions_list)}条排放记录")
        
        if not emissions_list:
            logger.warning(f"[GSOIL] {soil_type}没有生成任何排放记录！")
            return pd.DataFrame()
        
        df = pd.DataFrame(emissions_list)
        
        # 数据类型转换
        df['Year'] = df['Year'].astype(int)
        
        # 确保有三个排放列，GSOIL中只有n2o_kt和co2_kt
        df['ch4_kt'] = 0.0  # GSOIL没有CH4排放
        
        # 将动态的n2o_kt/co2_kt列转换为数值
        for col in ['n2o_kt', 'co2_kt']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        
        logger.info(f"计算{item_name}排放: {len(df)}行")
        return df
    
    def load_historical_emissions(self,
                                 emissions_csv_path: str,
                                 universe) -> Optional[pd.DataFrame]:
        """
        从FAO CSV文件读取历史Drained Organic Soils排放
        
        Args:
            emissions_csv_path: Emissions_Drained_Organic_Soils_E_All_Data_NOFLAG.csv路径
            universe: Universe对象（包含M49-国家映射）
        
        Returns:
            历史排放DataFrame
        """
        
        if not Path(emissions_csv_path).exists():
            logger.warning(f"历史排放文件不存在: {emissions_csv_path}")
            return None
        
        try:
            df = pd.read_csv(emissions_csv_path)
            logger.info(f"读取历史排放CSV: {len(df)}行")
            
            # 数据清洗和标准化
            historical = self._process_historical_data(df, universe)
            
            return historical
            
        except Exception as e:
            logger.error(f"读取历史排放CSV失败: {e}")
            return None
    
    def _process_historical_data(self, df: pd.DataFrame, universe) -> pd.DataFrame:
        """
        处理历史排放数据
        
        Args:
            df: 原始CSV DataFrame
            universe: Universe对象
        
        Returns:
            处理后的DataFrame
        """
        
        # 筛选FAO TIER 1数据
        if 'Source' in df.columns:
            df = df[df['Source'].str.contains('TIER 1', case=False, na=False)]
        
        # 筛选排放数据（排除Area行）
        if 'Element' in df.columns:
            # Element 值如: 'Drained organic soils (N2O)', 'Drained organic soils (CO2)', 'Emissions (N2O)', 'Emissions (CO2)'
            mask = df['Element'].str.contains('N2O|CO2', case=False, na=False) & ~df['Element'].str.contains('Area', case=False, na=False)
            df = df[mask]
            
            # 从 Element 提取 GHG 类型
            df['GHG'] = df['Element'].str.extract(r'\(?(N2O|CO2)\)?', expand=False)
            df['GHG'] = df['GHG'].str.upper()
        
        # 标准化M49代码（CSV中已有M49_Country_Code列，格式为'004，需要标准化）
        if 'M49_Country_Code' in df.columns:
            df['M49_Country_Code'] = df['M49_Country_Code'].apply(normalize_m49)
        elif 'Area' in df.columns:
            df['M49_Country_Code'] = df['Area'].apply(normalize_m49)
        
        # 添加国家名称
        if 'M49_Country_Code' in df.columns:
            df['Country'] = df['M49_Country_Code'].map(universe.country_by_m49)
        
        # 重命名排放值列
        value_col = None
        for col in df.columns:
            if col.lower().startswith('y') and col[1:].isdigit():
                value_col = col
                break
        
        if value_col is None:
            logger.warning("未找到排放值列")
            return pd.DataFrame()
        
        # Unpivot年份
        id_vars = [col for col in df.columns if not (col.lower().startswith('y') and col[1:].isdigit())]
        df_melted = pd.melt(df, id_vars=id_vars, var_name='Year', value_name='Value')
        
        # 清理Year列
        df_melted['Year'] = pd.to_numeric(
            df_melted['Year'].str.replace('^Y', '', regex=True),
            errors='coerce'
        ).astype('Int64')
        
        # 清理Value列
        df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
        
        # 删除NaN值
        df_melted = df_melted.dropna(subset=['Year', 'Value'])
        df_melted = df_melted[df_melted['Value'] != 0]  # 删除零值
        
        # 按 GHG 类型 pivot 成列
        if 'GHG' in df_melted.columns and 'Item' in df_melted.columns:
            # 创建宽格式，每个国家-年份-Item组合有独立的 n2o_kt 和 co2_kt 列
            result_list = []
            
            for (m49, country, item, year), group in df_melted.groupby(['M49_Country_Code', 'Country', 'Item', 'Year']):
                row = {
                    'M49_Country_Code': m49,
                    'Country': country,
                    'Item': item,
                    'Year': int(year),
                    'Process': 'Drained organic soils',
                    'ch4_kt': 0.0,
                    'n2o_kt': 0.0,
                    'co2_kt': 0.0
                }
                
                for _, r in group.iterrows():
                    ghg = r.get('GHG', '').upper()
                    value = r.get('Value', 0)
                    if ghg == 'N2O':
                        row['n2o_kt'] = float(value)
                    elif ghg == 'CO2':
                        row['co2_kt'] = float(value)
                
                result_list.append(row)
            
            result_df = pd.DataFrame(result_list)
            logger.info(f"处理后的历史排放: {len(result_df)}行")
            return result_df
        else:
            logger.warning("历史数据缺少必要的列")
            return pd.DataFrame()


def run_drained_organic_soils_emissions(
    universe,
    dict_v3_path: str,
    soil_params_path: str,
    emissions_csv_path: str,
    cropland_area_future: Dict,
    grassland_area_future: Dict,
    historical_years: List[int]
) -> Dict:
    """
    完整的Drained Organic Soils排放计算流程
    
    Args:
        universe: Universe对象
        dict_v3_path: dict_v3.xlsx路径
        soil_params_path: Soil_parameters.xlsx路径
        emissions_csv_path: 历史排放CSV路径
        cropland_area_future: 未来cropland面积字典
        grassland_area_future: 未来grassland面积字典
        historical_years: 历史年份列表
    
    Returns:
        {process_name: DataFrame} 格式的结果字典
    """
    
    logger.info(f"[GSOIL] 开始计算，cropland面积记录数={len(cropland_area_future)}, grassland={len(grassland_area_future)}")
    
    try:
        calc = DrainedOrganicSoilsEmissions(dict_v3_path, soil_params_path)
        logger.info(f"[GSOIL] 初始化成功，soil_params行数={len(calc.soil_params)}")
    except Exception as e:
        logger.error(f"[GSOIL] 初始化失败: {e}")
        return {}
    
    results = {}
    
    # 计算未来排放
    try:
        future_results = calc.calculate_future_emissions(
            cropland_area_ha=cropland_area_future,
            grassland_area_ha=grassland_area_future
        )
        logger.info(f"[GSOIL] 未来排放计算完成，返回{len(future_results)}个process")
        for key, df in future_results.items():
            logger.info(f"  - {key}: {len(df) if isinstance(df, pd.DataFrame) else 'N/A'}行")
        results.update(future_results)
    except Exception as e:
        logger.error(f"[GSOIL] 未来排放计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 读取历史排放
    try:
        historical = calc.load_historical_emissions(emissions_csv_path, universe)
        if historical is not None and not historical.empty:
            # 过滤到指定的历史年份
            if historical_years:
                historical = historical[historical['Year'].isin(historical_years)]
                logger.info(f"[GSOIL] 历史排放过滤到 {len(historical_years)} 个年份: {len(historical)}行")
            
            if not historical.empty:
                # 使用描述性的key，但DataFrame中的Process列已经是标准名称
                results['GSOIL_Historical'] = historical
                logger.info(f"[GSOIL] 历史排放加载成功: {len(historical)}行")
            else:
                logger.warning(f"[GSOIL] 过滤后历史排放为空")
        else:
            logger.warning(f"[GSOIL] 历史排放为空或None")
    except Exception as e:
        logger.error(f"[GSOIL] 历史排放加载失败: {e}")
    
    logger.info(f"[GSOIL] 总计返回{len(results)}个结果")
    return results
