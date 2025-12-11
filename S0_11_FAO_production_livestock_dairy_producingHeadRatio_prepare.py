# -*- coding: utf-8 -*-
"""
FAOSTAT livestock/dairy preprocessing (v6.1).

Steps per M49_Country_Code:
  1. Duplicate Milk Animals + Raw milk of {camel/goats/sheep/buffalo} into {Camel/Goats/Sheep/Buffalo, dairy};
     turn the original rows Select=0 and pad all missing country×species combinations with zeros.
  2. Build non-dairy Stocks: {*, non-dairy} = max(Stocks(*) - Milk Animals(*, dairy), 0) with
     Note='FAO Recalucated' & Select=1; original raw-species Stocks rows Select=0.
  3. Add Production ratio (Raw milk of {camel/cattle/goats/sheep/buffalo}) from production share (fillna(0)).
  4. Add Yield rows (Raw milk of {camel/goats/sheep/buffalo}) = production (t→kg)/Milk Animals(dairy), guarding zero denominators.

Key fixes:
  - Normalize M49 codes (strip quotes / decimals → Int64) so groupby/pivot works.
  - Before computing Production ratio, fill pivot tables with 0 to avoid NaN propagation.
  - pivot(values=year_cols) yields hierarchical (year,item); swaplevel → (item,year).
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set

# ---------------- 配置 ----------------
IN_PATH      = Path("../../input/Production_Trade/retired-unused-raw/Production_Crops_Livestock_E_All_Data_NOFLAG_2.csv")
OUT_PATH     = Path("../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
DICT_V3_PATH = Path("../../src/dict_v3.xlsx")
LOG_PATH     = Path("../../input/Production_Trade/retired-unused-raw") / f"{Path(__file__).stem}.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 是否在末尾打印若干自检（抽样 Canada/India 等）
RUN_SANITY_CHECKS = False


# ============== 通用小工具 ==============

def detect_year_cols(df: pd.DataFrame) -> List[str]:
    years = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"Y\d{4}", c)]
    if not years:
        raise ValueError("未发现年度列（形如 'Y2010'）")
    return years

def normalize_m49(series: pd.Series) -> pd.Series:
    """去掉前导引号/小数，转为可空整数 Int64。"""
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"^'", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def typical_unit(df: pd.DataFrame, mask: pd.Series, unit_col: str|None, default: str="") -> str:
    if not unit_col:
        return ""
    sub = df.loc[mask, unit_col].astype(str)
    sub = sub[sub.str.len() > 0]
    return sub.mode().iloc[0] if not sub.empty else default

def make_rows_from_wide(
    wide_df: pd.DataFrame,             # index: area_code; columns: year_cols
    area_code_col: str,
    year_cols: List[str],
    item_col: str,
    element_col: str,
    area_map: Dict,                    # code -> name
    unit_col: str|None,
    *,
    item_name: str,
    element_name: str,
    unit_val: str|None=None,
    note_val: str="",
    select_val: int=1,
    area_col: str|None="Area",
    m49_col: str|None=None,
    m49_map: Dict|None=None,
) -> pd.DataFrame:
    rows = wide_df.reset_index().rename(columns={wide_df.index.name: area_code_col})
    rows[item_col]    = item_name
    rows[element_col] = element_name
    rows["Note"]      = note_val
    rows["Select"]    = select_val
    if area_col:
        rows[area_col] = rows[area_code_col].map(area_map).fillna("")
    if m49_col and m49_map is not None:
        rows[m49_col] = rows[area_code_col].map(m49_map)
    if unit_col:
        rows[unit_col] = unit_val if unit_val is not None else ""
    meta_cols = [c for c in [area_code_col, area_col, m49_col, item_col, element_col, unit_col, "Note", "Select"] if c and c in rows.columns]
    return rows[meta_cols + year_cols]


def build_area_year_matrix(
    df: pd.DataFrame,
    area_code_col: str,
    year_cols: List[str],
    item_col: str,
    element_col: str,
    all_area_codes: pd.Index,
    *,
    item_name: str,
    element_name: str,
) -> pd.DataFrame:
    """Filter df for (item, element) and return Area×Year matrix (float) aligned to all_area_codes."""
    cols = [area_code_col] + year_cols
    mask = (df[item_col] == item_name) & (df[element_col] == element_name)
    sub = df.loc[mask, cols].copy()
    if sub.empty:
        wide = pd.DataFrame(np.nan, index=all_area_codes, columns=year_cols, dtype=float)
    else:
        wide = (
            sub.groupby(area_code_col, as_index=False)[year_cols]
            .sum()
            .set_index(area_code_col)
            .reindex(all_area_codes)
        )
    wide.index.name = area_code_col
    return wide[year_cols].astype(float)


# ============== 主流程 ==============

def main(in_path: Path = IN_PATH, out_path: Path = OUT_PATH, run_checks: bool = RUN_SANITY_CHECKS):
    # 读取
    df = pd.read_csv(in_path, low_memory=False)

    # 基础列识别
    year_cols   = detect_year_cols(df)
    area_col    = "Area" if "Area" in df.columns else None
    m49_col     = "M49_Country_Code" if "M49_Country_Code" in df.columns else None
    if not m49_col:
        raise ValueError("缺少 'M49_Country_Code' 列，无法定位国家。")
    element_col = "Element"
    item_col    = "Item"
    unit_col    = "Unit" if "Unit" in df.columns else None
    area_code_c = m49_col

    # 数值化年份
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # M49 标准化
    df[m49_col] = normalize_m49(df[m49_col])

    # 确保 Select / Note
    if "Select" not in df.columns: df["Select"] = 1
    if "Note"   not in df.columns: df["Note"]   = ""

    # dict_v3 region sheet：map M49 -> Region metadata
    region_meta = pd.read_excel(
        DICT_V3_PATH,
        sheet_name="region",
        usecols=["M49_Country_Code", "Region_agg2"],
    )
    region_meta["M49_Country_Code"] = pd.to_numeric(region_meta["M49_Country_Code"], errors="coerce").astype("Int64")
    region_meta = region_meta.dropna(subset=["M49_Country_Code"])
    region_m49_to_region = (
        region_meta.dropna(subset=["Region_agg2"])
        .set_index("M49_Country_Code")["Region_agg2"]
        .to_dict()
    )

    area_name_from_code = df.groupby(area_code_c)[area_col].first().to_dict() if area_col else {}
    all_area_codes = pd.Index(df[area_code_c].dropna().unique(), name=area_code_c)
    m49_to_area = area_name_from_code if area_col else {}

    # ------------------------------------------------
    # (1) Milk Animals：复制 Raw milk -> {*, dairy} 并补零
    # ------------------------------------------------
    species_rename = {
        "Raw milk of camel":   "Camel, dairy",
        "Raw milk of goats":   "Goats, dairy",
        "Raw milk of sheep":   "Sheep, dairy",
        "Raw milk of buffalo": "Buffalo, dairy",
    }
    mask_step1_src = (df[element_col]=="Milk Animals") & (df[item_col].isin(species_rename.keys()))
    dup = df.loc[mask_step1_src].copy()
    if not dup.empty:
        dup[item_col] = dup[item_col].map(species_rename)
        dup["Note"] = "FAO Production rename"
        dup["Select"] = 1
        df.loc[mask_step1_src, "Select"] = 0
        df = pd.concat([df, dup], ignore_index=True)

    milk_unit = typical_unit(df, df[element_col]=="Milk Animals", unit_col, default="Head")
    have_pairs = set(
        df.loc[
            (df[element_col]=="Milk Animals") & (df[item_col].isin(species_rename.values())),
            [area_code_c, item_col]
        ].itertuples(index=False, name=None)
    )
    need_pairs = {(ac, it) for ac in all_area_codes for it in species_rename.values()}
    missing_pairs = need_pairs - have_pairs
    if missing_pairs:
        add_rows = []
        for ac, it in sorted(missing_pairs):
            row = {area_code_c: ac, item_col: it, element_col: "Milk Animals", "Note": "FAO Production rename", "Select": 1}
            if area_col:
                row[area_col] = area_name_from_code.get(ac, "")
            if m49_col:
                row[m49_col] = ac
            if unit_col:
                row[unit_col] = milk_unit
            for yc in year_cols:
                row[yc] = 0.0
            add_rows.append(row)
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

    # 复制 dairy Milk Animals 作为 Stocks（若缺失）
    dairy_items = list(species_rename.values())
    existing_dairy_stock = set(
        df.loc[(df[element_col]=="Stocks") & (df[item_col].isin(dairy_items)), item_col].unique()
    )
    need_dairy_stock = set(dairy_items) - existing_dairy_stock
    if need_dairy_stock:
        stock_unit_general = typical_unit(df, df[element_col]=="Stocks", unit_col, default="Head")
        stock_sources = df.loc[
            (df[element_col]=="Milk Animals") & (df[item_col].isin(need_dairy_stock))
        ].copy()
        if not stock_sources.empty:
            stock_sources[element_col] = "Stocks"
            stock_sources["Note"] = "FAO Production rename"
            stock_sources["Select"] = 1
            if unit_col:
                stock_sources[unit_col] = stock_unit_general
            df = pd.concat([df, stock_sources], ignore_index=True)

    # ------------------------------------------------
    # (2) Stocks 非乳用 = max(Stocks(总) - MilkAnimals(dairy), 0)
    # ------------------------------------------------
    species_norm = {"Camels":"Camels","Goats":"Goats","Sheep":"Sheep","Buffalo":"Buffalo","Buffaloes":"Buffalo"}
    nd_name = {
        "Camels": "Camel, non-dairy",
        "Goats": "Goats, non-dairy",
        "Sheep": "Sheep, non-dairy",
        "Buffalo": "Buffalo, non-dairy",
    }

    stocks_df = df.loc[df[element_col]=="Stocks", [area_code_c, item_col] + year_cols].copy()
    stocks_df["species"] = stocks_df[item_col].map(species_norm)
    stocks_df = stocks_df.dropna(subset=["species"])
    stocks_agg = stocks_df.groupby([area_code_c, "species"], as_index=False)[year_cols].sum()

    dairy_map = {
        "Camel, dairy": "Camels",
        "Goats, dairy": "Goats",
        "Sheep, dairy": "Sheep",
        "Buffalo, dairy": "Buffalo",
    }
    milk_df = df.loc[
        (df[element_col]=="Milk Animals") & (df[item_col].isin(dairy_map.keys())),
        [area_code_c, item_col] + year_cols
    ].copy()
    milk_df["species"] = milk_df[item_col].map(dairy_map)
    milk_agg = milk_df.groupby([area_code_c, "species"], as_index=False)[year_cols].sum()

    grid = pd.MultiIndex.from_product([all_area_codes, pd.Index(["Camels","Goats","Sheep","Buffalo"], name="species")],
                                      names=[area_code_c, "species"])
    base = pd.DataFrame(index=grid).reset_index()
    nd = base.merge(stocks_agg, on=[area_code_c, "species"], how="left", suffixes=("", "_stocks"))
    nd = nd.merge(milk_agg, on=[area_code_c, "species"], how="left", suffixes=("_stocks", "_milk"))
    for yc in year_cols:
        s = nd.get(f"{yc}_stocks", nd.get(yc, 0.0))
        m = nd.get(f"{yc}_milk", 0.0)
        nd[yc] = (s.fillna(0.0) - m.fillna(0.0)).clip(lower=0.0)
    nd = nd[[area_code_c, "species"] + year_cols].copy()

    stock_unit_general = typical_unit(df, df[element_col]=="Stocks", unit_col, default="Head")
    nd[item_col] = nd["species"].map(nd_name)
    nd[element_col] = "Stocks"
    nd["Note"] = "FAO Recalucated"
    nd["Select"] = 1
    if unit_col:
        nd[unit_col] = stock_unit_general
    if area_col:
        nd[area_col] = nd[area_code_c].map(area_name_from_code).fillna("")
    nd = nd[[c for c in [area_code_c, area_col, m49_col, item_col, element_col, unit_col, "Note", "Select"] if c in nd.columns] + year_cols]

    df.loc[(df[element_col]=="Stocks") & (df[item_col].isin(species_norm.keys())), "Select"] = 0
    df = pd.concat([df, nd], ignore_index=True)

        # ------------------------------------------------
    # (5) Producing/Slaughtered ratio from dict_v3 Dairy/Meat
    # ------------------------------------------------
    emis_item_df = pd.read_excel(DICT_V3_PATH, sheet_name="Emis_item")
    ratio_specs = emis_item_df.loc[emis_item_df["Item_Cat2"].isin(["Dairy", "Meat"])].copy()
    ratio_specs = ratio_specs.drop_duplicates(subset=["Item_Emis", "Process"])
    ratio_specs = ratio_specs[
        ratio_specs["Item_Stock_Map"].notna()
        & ratio_specs["Item_Stock_Element"].notna()
        & ratio_specs["Item_Slaughtered_Map"].notna()
        & ratio_specs["Item_Slaughtered_Element"].notna()
    ]
    valid_items = set(df[item_col].dropna().unique())
    ratio_specs = ratio_specs[
        ratio_specs["Item_Stock_Map"].isin(valid_items)
        & ratio_specs["Item_Slaughtered_Map"].isin(valid_items)
    ]
    ratio_specs = ratio_specs.drop_duplicates(subset=["Item_Stock_Map"])
    ratio_rows = []
    warn_pairs: Set[Tuple[str, str]] = set()
    if not ratio_specs.empty and m49_col:
        m49_index = pd.Index(df[m49_col].dropna().unique(), name=m49_col).sort_values()

        def build_ratio_matrix(item_name: str, element_name: str) -> pd.DataFrame:
            cols = [m49_col] + year_cols
            mask = (df[item_col] == item_name) & (df[element_col] == element_name)
            sub = df.loc[mask, cols].dropna(subset=[m49_col])
            if sub.empty:
                mat = pd.DataFrame(np.nan, index=m49_index, columns=year_cols, dtype=float)
            else:
                mat = sub.groupby(m49_col)[year_cols].sum().reindex(m49_index)
            return mat

        ratio_unit = "ratio" if unit_col else None
        ratio_specs_records = ratio_specs.to_dict("records")
        for spec in ratio_specs_records:
            stock_item = spec["Item_Stock_Map"]
            stock_element = spec["Item_Stock_Element"]
            prod_item = spec["Item_Slaughtered_Map"]
            prod_element = spec["Item_Slaughtered_Element"]

            prod_matrix = build_ratio_matrix(prod_item, prod_element)
            stock_matrix = build_ratio_matrix(stock_item, stock_element)
            ratio = prod_matrix.divide(stock_matrix)
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio = ratio.bfill(axis=1).ffill(axis=1)

            m49_series = pd.Series(ratio.index, index=ratio.index)
            region_series = m49_series.map(lambda code: region_m49_to_region.get(code, None) if pd.notna(code) else None)
            ratio_with_region = ratio.copy()
            ratio_with_region["__region"] = region_series.values
            region_mean = ratio_with_region.groupby("__region")[year_cols].mean()
            global_mean = ratio[year_cols].mean(axis=0, skipna=True)

            nan_rows = ratio.isna().all(axis=1)
            if nan_rows.any():
                for m49_val in ratio.index[nan_rows]:
                    reg = region_series.loc[m49_val]
                    fill_vals = None
                    if pd.notna(reg) and reg in region_mean.index:
                        fill_vals = region_mean.loc[reg]
                    if fill_vals is None or (hasattr(fill_vals, "isna") and fill_vals.isna().all()):
                        fill_vals = global_mean
                    ratio.loc[m49_val, year_cols] = np.asarray(fill_vals.reindex(year_cols))

            # 若因 Producing Animals/Slaughtered/Laying/Milk Animals 为 0 导致比值为 0，则设为 1
            zero_mask = (ratio == 0) & (prod_matrix == 0)
            if zero_mask.any().any():
                ratio = ratio.mask(zero_mask, 1.0)

            exceeds = ratio.gt(1.0000000001)
            if exceeds.any().any():
                for m49_val in ratio.index[exceeds.any(axis=1)]:
                    warn_pairs.add((str(m49_val), stock_item))

            ratio_reset = ratio.reset_index().rename(columns={m49_col: m49_col})
            if area_col:
                ratio_reset[area_col] = ratio_reset[m49_col].map(m49_to_area).fillna("")
            ratio_reset[item_col] = stock_item
            ratio_reset[element_col] = "Producing/Slaughtered ratio"
            ratio_reset["Note"] = "FAO calculated"
            ratio_reset["Select"] = 1
            if unit_col:
                ratio_reset[unit_col] = ratio_unit
            cols_keep = [c for c in [area_code_c, area_col, m49_col, item_col, element_col, unit_col, "Note", "Select"] if c and c in ratio_reset.columns]
            ratio_rows.append(ratio_reset[cols_keep + year_cols])
    elif not m49_col:
        logger.warning("Cannot compute Producing/Slaughtered ratio because 'M49_Country_Code' column is missing.")
    if ratio_rows:
        df = pd.concat([df, pd.concat(ratio_rows, ignore_index=True)], ignore_index=True)
    if warn_pairs:
        warn_msg = ", ".join(
            f"{code}-{item}" for code, item in sorted(warn_pairs, key=lambda x: (x[0], x[1]))
        )
        logger.warning("Producing/Slaughtered ratio > 1 detected for: %s", warn_msg)

    # Area-Item 升序
    sort_cols = []
    if area_col and area_col in df.columns:
        sort_cols.append(area_col)
    sort_cols.append(m49_col)
    if item_col in df.columns:
        sort_cols.append(item_col)
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # 保存
    df.to_csv(out_path, index=False)
    print(f"Done -> {out_path}")

    # （可选）自检：抽查 Canada/India 的 ratio & yield 是否合理
    if run_checks:
        cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Y")]
        def show_ratio(area_name: str):
            m = (df["Area"]==area_name) & (df["Element"]=="Production ratio")
            print(f"\n[Check] Production ratio — {area_name}")
            print(df.loc[m].set_index("Item")[cols[-6:]].reindex(
                ["Raw milk of camel","Raw milk of cattle","Raw milk of goats","Raw milk of sheep","Raw milk of buffalo"]
            ).fillna(0).round(4))

        def show_yield(area_name: str):
            m = (df["Area"]==area_name) & (df["Element"]=="Yield")
            print(f"\n[Check] Yield (kg/An) — {area_name}")
            print(df.loc[m].set_index("Item")[cols[-6:]].reindex(
                ["Raw milk of camel","Raw milk of goats","Raw milk of sheep","Raw milk of buffalo"]
            ).fillna(0).round(2))

        for area in ["Canada","India"]:
            if area_col and (df["Area"]==area).any():
                show_ratio(area)
                show_yield(area)


if __name__ == "__main__":
    main()
