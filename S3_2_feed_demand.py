# -*- coding: utf-8 -*-
"""
S3.2 Feed demand builder
-----------------------
Derives livestock feed requirements (grass + crop) directly from
country-level livestock stocks and parameter tables stored under
input/Land/Feed_pasture/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import logging
import numpy as np
import pandas as pd

from S1_0_schema import Universe
from S2_0_load_data import DataPaths, EmisItemMappings

# 配置logger
logger = logging.getLogger(__name__)


@dataclass
class FeedDemandOutputs:
    crop_feed_demand: pd.DataFrame
    grass_requirement: pd.DataFrame
    species_dm_detail: pd.DataFrame


def build_feed_demand_from_stock(*,
                                 stock_df: pd.DataFrame,
                                 universe: Universe,
                                 maps: EmisItemMappings,
                                 paths: DataPaths,
                                 years: List[int],
                                 conversion_multiplier: Optional[Dict[Tuple[str, str, int], float]] = None) -> FeedDemandOutputs:
    """
    Convert stock_head (by commodity/country/year) into:
      1) species-level DM requirements from Feed_need_per_head...xlsx
      2) grass vs crop DM split via Grass_feed_ratio...
      3) crop-specific feed demand by commodity (converted to grain using dm_conversion_coefficients)
      4) grass DM requirement + implied pasture area using Pasture_DM_yield_by_country.xlsx
    """
    # ✅ 所有DataFrame都包含M49_Country_Code列
    empty = FeedDemandOutputs(
        crop_feed_demand=pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t']),
        grass_requirement=pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','grass_tdm','grass_area_need_ha']),
        species_dm_detail=pd.DataFrame(columns=[
            'country','iso3','m49_code','year','commodity','species',
            'stock_head','kg_dm_per_head','dm_total_kg','grass_dm_kg','crop_dm_kg'
        ])
    )
    if stock_df is None or stock_df.empty:
        return empty
    conv_mult: Dict[Tuple[str, str, int], float] = {}
    for key, val in (conversion_multiplier or {}).items():
        try:
            country, commodity, year = key
            conv_mult[(str(country), str(commodity), int(year))] = float(val)
        except Exception:
            continue

    if not (paths.feed_need_xlsx and os.path.exists(paths.feed_need_xlsx)):
        logger.error(f"[S3_2 ERROR] Feed_need文件不存在或路径为None: {paths.feed_need_xlsx}")
        return empty
    if not (paths.grass_ratio_xlsx and os.path.exists(paths.grass_ratio_xlsx)):
        logger.error(f"[S3_2 ERROR] ❌❌❌ Grass_ratio文件不存在或路径为None: {paths.grass_ratio_xlsx}")
        logger.error(f"[S3_2 ERROR] 这是导致草地需求缺失的根本原因！")
        return empty
    if not (paths.pasture_dm_yield_xlsx and os.path.exists(paths.pasture_dm_yield_xlsx)):
        logger.error(f"[S3_2 ERROR] Pasture_yield文件不存在或路径为None: {paths.pasture_dm_yield_xlsx}")
        return empty

    years = sorted(set(int(y) for y in years))
    logger.info(f"[S3_2 DEBUG] 请求的years: {years}")
    
    comm_to_species = {comm: feed_item for feed_item, comm in (maps.feed_item_to_comm or {}).items()}
    logger.info(f"[S3_2 DEBUG] comm_to_species映射: {len(comm_to_species)} 个commodity")
    if len(comm_to_species) > 0:
        logger.info(f"[S3_2 DEBUG] 映射样例: {list(comm_to_species.items())[:5]}")
    
    # ✅ 注意：comm_to_species的key是Item_Emis，value是Item_Feed_Map（species）
    # 例如：{"Cattle, dairy": "dairy_cattle", "Cattle, non-dairy": "beef_cattle"}
    # 但传入的stock_df.commodity应该已经是Item_Feed_Map格式（species名称）
    # 所以这里的映射是反向的，需要调整！
    
    if not comm_to_species:
        logger.error(f"[S3_2 ERROR] ❌ comm_to_species映射为空！")
        return empty

    country_to_m49 = {}
    m49_to_country = {}
    for country, code in (universe.m49_by_country or {}).items():
        parsed = _parse_m49(code)
        if parsed is None:
            continue
        country_to_m49[country] = parsed
        m49_to_country[parsed] = country

    if not country_to_m49:
        return empty

    stock = stock_df.copy()
    logger.info(f"[S3_2 DEBUG] 输入存栏数据: {len(stock)} 行")
    if 'year' in stock.columns:
        stock_years = sorted(stock['year'].unique())
        logger.info(f"[S3_2 DEBUG] 存栏年份: {stock_years}")
    
    # 诊断：检查传入的commodity值
    if 'commodity' in stock.columns:
        unique_commodities = stock['commodity'].unique()
        logger.info(f"[S3_2 DEBUG] 传入的commodity值样例: {list(unique_commodities)[:10]}")
    
    stock['m49_code'] = stock['country'].map(country_to_m49)
    
    # ✅ 映射commodity（Item_Emis格式，如'Cattle, non-dairy'）到species（Item_Feed_Map格式，如'beef_cattle'）
    # comm_to_species应该是: {Item_Emis: Item_Feed_Map}
    # 但由于feed_item_to_comm反转，实际comm_to_species可能是反向的，需要修正
    
    # 🔍 诊断：打印comm_to_species内容，检查dairy品种
    logger.info(f"[S3_2 DEBUG] comm_to_species映射数量: {len(comm_to_species)}")
    dairy_check = {k: v for k, v in comm_to_species.items() if 'dairy' in str(k).lower() and 'non-dairy' not in str(k).lower()}
    logger.info(f"[S3_2 DEBUG] comm_to_species中dairy品种: {dairy_check}")
    
    # 先尝试直接映射
    stock['species'] = stock['commodity'].map(comm_to_species)
    
    # 如果映射失败（大部分是NaN），说明comm_to_species是反向的，需要反转回来
    unmapped_count = stock['species'].isna().sum()
    if unmapped_count > len(stock) * 0.5:  # 如果超过50%没匹配上
        logger.warning(f"[S3_2 WARNING] comm_to_species映射失败率高({unmapped_count}/{len(stock)})，尝试反转映射...")
        # 反转映射：{commodity: feed_item} -> {feed_item: commodity}
        species_to_comm = {v: k for k, v in comm_to_species.items()}
        # 构建commodity到feed_item的正确映射
        correct_mapping = {}
        for commodity in stock['commodity'].unique():
            if pd.isna(commodity):
                continue
            # 尝试在species_to_comm的值中查找
            for feed_item, comm in species_to_comm.items():
                if comm == commodity:
                    correct_mapping[commodity] = feed_item
                    break
        logger.info(f"[S3_2 DEBUG] 构建的正确映射示例: {list(correct_mapping.items())[:5]}")
        stock['species'] = stock['commodity'].map(correct_mapping)
    
    # ✅ 关键修复：对于仍未映射的dairy品种，直接从dict_v3补充
    still_unmapped = stock['species'].isna()
    if still_unmapped.any():
        unmapped_commodities = stock[still_unmapped]['commodity'].unique()
        dairy_unmapped = [c for c in unmapped_commodities if 'dairy' in str(c).lower() and 'non-dairy' not in str(c).lower()]
        
        if dairy_unmapped:
            logger.warning(f"[S3_2 DAIRY_FIX] 检测到{len(dairy_unmapped)}个dairy品种未映射，从dict_v3补充: {dairy_unmapped}")
            
            # 直接从dict_v3加载dairy映射
            try:
                dict_v3_path = paths.dict_v3_path if hasattr(paths, 'dict_v3_path') else None
                if dict_v3_path and os.path.exists(dict_v3_path):
                    emis_df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
                    dairy_mapping = {}
                    for _, row in emis_df.iterrows():
                        item_emis = row.get('Item_Emis')
                        item_feed = row.get('Item_Feed_Map')
                        if pd.notna(item_emis) and pd.notna(item_feed):
                            if 'dairy' in str(item_emis).lower() and 'non-dairy' not in str(item_emis).lower():
                                dairy_mapping[item_emis] = item_feed
                    
                    logger.info(f"[S3_2 DAIRY_FIX] 从dict_v3加载dairy映射: {dairy_mapping}")
                    
                    # 应用dairy映射到未映射的行
                    for commodity in dairy_unmapped:
                        if commodity in dairy_mapping:
                            mask = (stock['commodity'] == commodity) & stock['species'].isna()
                            stock.loc[mask, 'species'] = dairy_mapping[commodity]
                            logger.info(f"[S3_2 DAIRY_FIX] 修复: {commodity} → {dairy_mapping[commodity]} ({mask.sum()} 行)")
            except Exception as e:
                logger.error(f"[S3_2 DAIRY_FIX] 从dict_v3补充dairy映射失败: {e}")
    
    # 最后检查是否还有未映射的
    unmapped = stock['species'].isna()
    if unmapped.any():
        logger.warning(f"[S3_2 WARNING] {unmapped.sum()} 行commodity无法映射到species，将被过滤")
    
    stock['iso3'] = stock['iso3'].fillna(stock['country'].map(universe.iso3_by_country))
    
    logger.info(f"[S3_2 DEBUG] 映射后: m49_code缺失={stock['m49_code'].isna().sum()}, species缺失={stock['species'].isna().sum()}")
    
    # 🔍 诊断：以美国为例追踪species映射
    us_stock = stock[stock['country'] == 'United States of America']
    if not us_stock.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 4: species映射完成")
        logger.info("=" * 80)
        logger.info(f"美国存栏数据: {len(us_stock)} 行")
        us_sample = us_stock[['commodity', 'species', 'stock_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            logger.info(f"  commodity={row['commodity']:15s} | species={row['species']:15s} | {row['stock_head']:>12,.0f} head ({row['year']}年)")
    stock = stock.dropna(subset=['m49_code', 'species'])
    logger.info(f"[S3_2 DEBUG] dropna后: {len(stock)} 行")
    
    # 🔍 增强诊断：在过滤前检查stock_head列
    logger.info("\n" + "=" * 80)
    logger.info("🔍 [S3_2诊断] stock_head过滤前检查")
    logger.info("=" * 80)
    logger.info(f"stock_head列类型: {stock['stock_head'].dtype}")
    logger.info(f"stock_head非空行数: {stock['stock_head'].notna().sum()}/{len(stock)}")
    logger.info(f"stock_head>0行数（过滤前）: {(stock['stock_head'] > 0).sum()}/{len(stock)}")
    logger.info(f"stock_head总和: {stock['stock_head'].sum():,.0f}")
    if len(stock) > 0:
        logger.info(f"stock_head范围: {stock['stock_head'].min():.2e} ~ {stock['stock_head'].max():.2e}")
        logger.info(f"stock_head样例（前5行）: {stock['stock_head'].head().tolist()}")
    
    stock['stock_head'] = pd.to_numeric(stock['stock_head'], errors='coerce').fillna(0.0)
    
    # 🔍 诊断：numeric转换后的状态
    logger.info(f"numeric转换后stock_head>0行数: {(stock['stock_head'] > 0).sum()}/{len(stock)}")
    logger.info(f"numeric转换后stock_head总和: {stock['stock_head'].sum():,.0f}")
    
    stock = stock[stock['stock_head'] > 0]
    logger.info(f"[S3_2 DEBUG] 过滤stock_head>0后: {len(stock)} 行")
    logger.info("=" * 80 + "\n")
    
    if stock.empty:
        logger.error(f"[S3_2 ERROR] ❌ 存栏数据为空，提前返回！")
        return empty

    dm_per_head = _load_total_dm_per_head(paths.feed_need_xlsx, years)
    crop_share = _load_crop_share(paths.feed_need_xlsx, years)
    dm_conversion = _load_dm_conversion(paths.feed_need_xlsx, years)
    grass_ratio = _load_grass_ratio(paths.grass_ratio_xlsx)
    pasture_yield = _load_pasture_yield(paths.pasture_dm_yield_xlsx)
    
    # ✅ 诊断：确认参数数据的年份范围
    if not dm_per_head.empty and 'year' in dm_per_head.columns:
        param_years = sorted(dm_per_head['year'].unique())
        logger.info(f"[S3_2 DEBUG] dm_per_head年份范围: {param_years[:3]}...{param_years[-3:]}, 共{len(param_years)}年")
        if 2080 in param_years:
            logger.info(f"[S3_2 DEBUG] ✅ dm_per_head包含2080年数据（前向填充成功）")
        else:
            logger.info(f"[S3_2 DEBUG] ❌ dm_per_head缺少2080年数据！")

    if dm_per_head.empty or crop_share.empty or dm_conversion.empty:
        logger.error(f"[S3_2 ERROR] ❌ 参数数据为空: dm_per_head={dm_per_head.empty}, crop_share={crop_share.empty}, dm_conversion={dm_conversion.empty}")
        return empty

    logger.info(f"[S3_2 DEBUG] merge前stock: {len(stock)} 行, dm_per_head: {len(dm_per_head)} 行")
    # 诊断：检查species值匹配
    stock_species = set(stock['species'].unique())
    dm_species = set(dm_per_head['species'].unique())
    logger.info(f"[S3_2 DEBUG] stock中的species ({len(stock_species)}个): {sorted(list(stock_species))[:10]}")
    logger.info(f"[S3_2 DEBUG] dm_per_head中的species ({len(dm_species)}个): {sorted(list(dm_species))[:10]}")
    overlap = stock_species & dm_species
    logger.info(f"[S3_2 DEBUG] 交集species: {len(overlap)} 个")
    if len(overlap) == 0:
        logger.error(f"[S3_2 ERROR] ❌ stock和dm_per_head的species完全不匹配！")
        logger.error(f"[S3_2 ERROR] stock示例: {list(stock_species)[:5]}")
        logger.error(f"[S3_2 ERROR] dm_per_head示例: {list(dm_species)[:5]}")
    
    stock = stock.merge(
        dm_per_head,
        how='left',
        left_on=['species','m49_code','year'],
        right_on=['species','m49_code','year']
    )
    logger.info(f"[S3_2 DEBUG] merge后: {len(stock)} 行")
    
    stock['kg_dm_per_head'] = pd.to_numeric(stock['kg_dm_per_head'], errors='coerce')
    kg_dm_na = stock['kg_dm_per_head'].isna().sum()
    logger.info(f"[S3_2 DEBUG] kg_dm_per_head缺失: {kg_dm_na}/{len(stock)} 行")
    
    # 🔍 诊断：以美国为例追踪DM per head匹配
    us_stock = stock[stock['country'] == 'United States of America']
    if not us_stock.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 5: DM per head参数匹配")
        logger.info("=" * 80)
        logger.info(f"美国存栏数据: {len(us_stock)} 行")
        us_sample = us_stock[['species', 'stock_head', 'kg_dm_per_head', 'year']].head(10)
        for _, row in us_sample.iterrows():
            dm_status = f"{row['kg_dm_per_head']:.1f}" if pd.notna(row['kg_dm_per_head']) else "❌ NaN"
            logger.info(f"  {row['species']:15s} | {row['stock_head']:>12,.0f} head | DM={dm_status:>8s} kg/head ({row['year']}年)")
    
    stock = stock.dropna(subset=['kg_dm_per_head'])
    logger.info(f"[S3_2 DEBUG] dropna(kg_dm_per_head)后: {len(stock)} 行")
    
    if stock.empty:
        logger.error(f"[S3_2 ERROR] ❌ merge dm_per_head后数据为空，提前返回！")
        logger.error(f"[S3_2 ERROR] 可能原因：存栏的species/m49_code/year组合在dm_per_head中找不到匹配")
        return empty
    stock['dm_total_kg'] = stock['stock_head'] * stock['kg_dm_per_head']

    stock = stock.merge(
        grass_ratio,
        how='left',
        on=['species','m49_code']
    )
    stock['grass_ratio'] = stock['grass_ratio'].clip(lower=0.0, upper=1.0).fillna(0.0)
    stock['crop_ratio'] = stock['crop_ratio'].clip(lower=0.0, upper=1.0)
    stock['crop_ratio'] = stock['crop_ratio'].fillna(1.0 - stock['grass_ratio'])
    stock['crop_ratio'] = stock['crop_ratio'].clip(lower=0.0, upper=1.0)
    stock['grass_dm_kg'] = stock['dm_total_kg'] * stock['grass_ratio']
    stock['crop_dm_kg'] = stock['dm_total_kg'] * stock['crop_ratio']

    crop_dm_rows = stock[['country','iso3','m49_code','year','species','dm_total_kg','crop_ratio']].merge(
        crop_share,
        how='left',
        on=['species','m49_code','year']
    )
    crop_dm_rows['share'] = crop_dm_rows['share'].clip(lower=0.0)
    crop_dm_rows['share'] = crop_dm_rows['share'].fillna(0.0)
    crop_dm_rows['crop_dm_kg'] = crop_dm_rows['dm_total_kg'] * crop_dm_rows['crop_ratio'] * crop_dm_rows['share']
    crop_dm_rows = crop_dm_rows[crop_dm_rows['crop_dm_kg'] > 0]
    if crop_dm_rows.empty:
        crop_feed_demand = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t'])
    else:
        crop_dm_rows = crop_dm_rows.merge(
            dm_conversion,
            how='left',
            on=['m49_code','crop','year']
        )
        crop_dm_rows['dm_fraction'] = crop_dm_rows['dm_fraction'].replace(0, np.nan)
        crop_dm_rows = crop_dm_rows.dropna(subset=['dm_fraction'])
        crop_dm_rows['commodity'] = crop_dm_rows['crop'].map((maps.production_by_item or {}))
        crop_dm_rows['commodity'] = crop_dm_rows['commodity'].fillna(crop_dm_rows['crop'])
        if conv_mult:
            keys = list(zip(
                crop_dm_rows['country'].astype(str),
                crop_dm_rows['commodity'].astype(str),
                crop_dm_rows['year'].astype(int)
            ))
            mult = np.array([conv_mult.get(k, 1.0) for k in keys], dtype=float)
            mult = np.where(np.isfinite(mult), mult, 1.0)
            mult = np.clip(mult, 1e-6, None)
            crop_dm_rows['dm_fraction'] = crop_dm_rows['dm_fraction'] * mult
        crop_dm_rows['grain_need_kg'] = crop_dm_rows['crop_dm_kg'] / crop_dm_rows['dm_fraction']
        crop_dm_rows = crop_dm_rows[crop_dm_rows['commodity'].isin(universe.commodities)]
        crop_dm_rows['feed_t'] = crop_dm_rows['grain_need_kg'] / 1000.0
        # ✅ 保留M49_Country_Code列（重命名m49_code为标准列名）
        if 'm49_code' in crop_dm_rows.columns:
            crop_dm_rows['M49_Country_Code'] = crop_dm_rows['m49_code']
        crop_feed_demand = crop_dm_rows.groupby(
            ['M49_Country_Code','country','iso3','year','commodity'],
            as_index=False
        )['feed_t'].sum()

    grass_req = stock.groupby(['country','iso3','m49_code','year'], as_index=False)['grass_dm_kg'].sum()
    logger.info(f"[S3_2 DEBUG] 草地DM需求聚合完成: {len(grass_req)}行, 年份范围: {grass_req['year'].min()}-{grass_req['year'].max()}")
    
    # 🔍 诊断：以美国为例追踪草地DM需求
    us_grass = grass_req[grass_req['country'] == 'United States of America']
    if not us_grass.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 6: 草地DM需求计算")
        logger.info("=" * 80)
        for _, row in us_grass.iterrows():
            logger.info(f"  {row['year']}年: 草地DM需求 = {row['grass_dm_kg']:>15,.0f} kg")
    
    grass_req = grass_req.merge(pasture_yield, how='left', on='m49_code')
    
    # ✅ DEBUG: 检查pasture_yield匹配情况
    missing_yield = grass_req['pasture_yield_kg_per_ha'].isna().sum()
    if missing_yield > 0:
        logger.warning(f"[S3_2 WARN] ⚠️ {missing_yield}/{len(grass_req)}行缺失pasture_yield数据！")
        missing_countries = grass_req[grass_req['pasture_yield_kg_per_ha'].isna()]['country'].unique()
        logger.warning(f"[S3_2 WARN] 缺失yield的国家样例: {list(missing_countries)[:10]}")
    
    grass_req['grass_tdm'] = grass_req['grass_dm_kg'] / 1000.0
    grass_req['grass_area_need_ha'] = grass_req['grass_dm_kg'] / grass_req['pasture_yield_kg_per_ha'].replace(0, np.nan)
    
    # ✅ DEBUG: 检查area计算结果
    area_na_count = grass_req['grass_area_need_ha'].isna().sum()
    if area_na_count > 0:
        logger.warning(f"[S3_2 WARN] ⚠️ {area_na_count}/{len(grass_req)}行的grass_area_need_ha为NaN（可能pasture_yield=0或NaN）")
    
    # 🔍 诊断：以美国为例追踪草地面积需求
    us_grass_area = grass_req[grass_req['country'] == 'United States of America']
    if not us_grass_area.empty:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 [美国数据流] Step 7: 草地面积需求计算 (DM ÷ yield)")
        logger.info("=" * 80)
        for _, row in us_grass_area.iterrows():
            yield_val = row['pasture_yield_kg_per_ha']
            area_val = row['grass_area_need_ha']
            yield_str = f"{yield_val:,.0f}" if pd.notna(yield_val) else "NaN"
            area_str = f"{area_val:,.0f}" if pd.notna(area_val) else "❌ NaN"
            logger.info(f"  {row['year']}年: 草地单产={yield_str:>10s} kg/ha | 面积需求={area_str:>15s} ha")
    
    # ✅ 保留M49_Country_Code列（重命名m49_code为标准列名）
    grass_req['M49_Country_Code'] = grass_req['m49_code']
    grass_requirement = grass_req[['M49_Country_Code','country','iso3','year','grass_tdm','grass_area_need_ha']].copy()
    
    logger.info(f"[S3_2 DEBUG] ✅ grass_requirement生成完成: {len(grass_requirement)}行")
    for yr in [2020, 2080]:
        yr_data = grass_requirement[grass_requirement['year'] == yr]
        if not yr_data.empty:
            total_area = yr_data['grass_area_need_ha'].sum()
            valid_area = yr_data['grass_area_need_ha'].notna().sum()
            logger.info(f"[S3_2 DEBUG]   {yr}年: {len(yr_data)}行, 有效面积数据: {valid_area}行, 总面积: {total_area:,.0f} ha")

    species_dm_detail = stock[['country','iso3','m49_code','year','commodity','species',
                               'stock_head','kg_dm_per_head','dm_total_kg','grass_dm_kg','crop_dm_kg']].copy()

    return FeedDemandOutputs(
        crop_feed_demand=crop_feed_demand,
        grass_requirement=grass_requirement,
        species_dm_detail=species_dm_detail
    )


def _parse_m49(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    digits = ''.join(ch for ch in str(val) if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(3)


def _normalize_m49(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(_parse_m49)


def _load_total_dm_per_head(xlsx_path: str, years: List[int]) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='total_kgDM_per_head')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    value_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    frames = []
    for col in value_cols:
        year = int(col[1:])
        if year not in years:
            continue
        tmp = df[['Species','m49_code', col]].copy()
        tmp = tmp.rename(columns={'Species':'species', col:'kg_dm_per_head'})
        tmp['year'] = year
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = _extend_years(out, ['species','m49_code'], 'kg_dm_per_head', years)
    return out


def _load_crop_share(xlsx_path: str, years: List[int]) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='kgDM_per_head_crop_shares')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['crop'] = df['Crop'].astype(str).str.strip()
    value_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    frames = []
    for col in value_cols:
        year = int(col[1:])
        if year not in years:
            continue
        tmp = df[['Species','m49_code','crop', col]].copy()
        tmp = tmp.rename(columns={'Species':'species', col:'share'})
        tmp['year'] = year
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = _extend_years(out, ['species','m49_code','crop'], 'share', years)
    return out


def _load_dm_conversion(xlsx_path: str, years: List[int]) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='dm_conversion_coefficients')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['crop'] = df['Crop'].astype(str).str.strip()
    value_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
    frames = []
    for col in value_cols:
        year = int(col[1:])
        if year not in years:
            continue
        tmp = df[['m49_code','crop', col]].copy()
        tmp = tmp.rename(columns={col: 'dm_fraction'})
        tmp['year'] = year
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = _extend_years(out, ['m49_code','crop'], 'dm_fraction', years)
    return out


def _load_grass_ratio(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='country_level_weighted')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['species'] = df['Species'].astype(str).str.strip()
    df['grass_ratio'] = pd.to_numeric(df.get('Grass'), errors='coerce')
    df['crop_ratio'] = pd.to_numeric(df.get('Crop'), errors='coerce')
    return df[['species','m49_code','grass_ratio','crop_ratio']].dropna(subset=['m49_code','species'])


def _load_pasture_yield(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='pasture_DM_yield')
    df.columns = [str(c).strip() for c in df.columns]
    df['m49_code'] = _normalize_m49(df['M49_Country_Code'])
    df['pasture_yield_kg_per_ha'] = pd.to_numeric(df.get('mean_AGB_kg_ha_weighted_by_area'), errors='coerce')
    return df[['m49_code','pasture_yield_kg_per_ha']].dropna(subset=['m49_code'])


def _extend_years(df: pd.DataFrame,
                  key_cols: List[str],
                  value_col: str,
                  years: List[int]) -> pd.DataFrame:
    """
    扩展年份数据到所有请求的年份
    ✅ 修复：当从优化存栏动态计算草地需求时，需要扩展到未来年份（2020-2080）
    策略：将最近的历史年份数据（通常是2020年）前向填充到所有未来年份
    
    这是合理的，因为：
    - DM per head (每头干物质需求)：技术参数，短期内相对稳定
    - Crop share (作物饲料分配)：饲料配方，基于历史模式
    - Grass ratio (草料比例)：饲养方式，假设延续
    这些参数的未来变化应该通过scenario调整，而不是完全缺失数据
    """
    if df.empty:
        return df
    pivot = df.pivot_table(index=key_cols, columns='year', values=value_col, aggfunc='last')
    
    # ✅ 为所有请求的年份创建列（如果不存在）
    all_years = sorted(set(years))
    for y in all_years:
        if y not in pivot.columns:
            pivot[y] = np.nan
    
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.ffill(axis=1)  # ✅ 前向填充到所有年份（包括未来）
    pivot = pivot.reset_index()
    long_df = pivot.melt(id_vars=key_cols, var_name='year', value_name=value_col)
    long_df['year'] = long_df['year'].astype(int)
    long_df = long_df[long_df['year'].isin(all_years)]  # ✅ 保留所有请求的年份
    return long_df
