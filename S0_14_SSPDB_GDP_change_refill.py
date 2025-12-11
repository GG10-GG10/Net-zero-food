# -*- coding: utf-8 -*-
"""
更新内容（修复早年缺失）：
1) 按 SCENARIO+REGION 计算 GDP|PPP 相对 2020 的变化比（change_ratio），
   在输出工作簿中新建 sheet 'change_ratio'，保持原表结构，仅替换年份列为比值。
2) 读取 dict_v3.xlsx['region']（先过滤 Region_label_new != 'no'），
   用 M49_Country_Code 与 Region_map_SSPDB / Region_agg3#income / Region_map_elasCat7 的映射，
   将各 SCENARIO 下各 REGION 的 change_ratio 拓展到国家层面：
   - 若 Region_map_SSPDB != 'Region_Income'：直接映射；
   - 若为 'Region_Income'：先用 (SCENARIO, Region_agg3#income) 的非 NaN 均值填充；
     若仍 NaN，回退到 (SCENARIO, Region_map_elasCat7) 的非 NaN 均值填充。
3) 为避免原始数据库在早期（如 Y2000/Y2005）无值导致残留 NaN：
   - 再做一轮“合并后”的均值回填（先 income，再 elas，再 scenario-only 均值兜底）；
   - 最后对每行按时间轴做一次 bfill（反向填充），用最早可用的后续年份比值回填最早年份。
4) 将国家层结果写入 'change_ratio_country'。
"""

import pandas as pd
import numpy as np

# =========================
# 路径（按需修改）
# =========================
ssp_path  = "../../input/Driver/Income/retired_unused_raw/SSPDB_future_GDP.xlsx"
dict_path = "../../src/dict_v3.xlsx"
out_path  = "../../input/Driver/Income/SSPDB_future_GDP_with_change_ratio.xlsx"

# =========================
# 工具函数
# =========================
def is_year_col(c):
    """识别年份列：支持 int 年份、'2020'、'Y2020'。"""
    try:
        if isinstance(c, int):
            return 1900 <= c <= 2100
        if isinstance(c, str):
            if c.isdigit() and 1900 <= int(c) <= 2100:
                return True
            if c.startswith("Y") and c[1:].isdigit() and 1900 <= int(c[1:]) <= 2100:
                return True
        return False
    except Exception:
        return False

def year_to_int(c):
    """将列名解析为年份整数（用于排序）。"""
    if isinstance(c, int):
        return c
    if isinstance(c, str):
        if c.isdigit():
            return int(c)
        if c.startswith("Y") and c[1:].isdigit():
            return int(c[1:])
    # 不应命中此分支
    return 9999

# =========================
# 1) 读取 SSPDB 并计算 change_ratio
# =========================
xls = pd.ExcelFile(ssp_path)
df_ssp = pd.read_excel(ssp_path, sheet_name="SSPDB_future_GDP")

# 必要列
for req in ["SCENARIO", "REGION"]:
    if req not in df_ssp.columns:
        raise KeyError(f"'SSPDB_future_GDP' 缺少必要列：{req}")

# 若存在 VARIABLE 列，仅保留 GDP|PPP
if "VARIABLE" in df_ssp.columns:
    df_ssp = df_ssp[df_ssp["VARIABLE"] == "GDP|PPP"].copy()

# 年份列与基准列
year_cols = [c for c in df_ssp.columns if is_year_col(c)]
if not year_cols:
    raise ValueError("未检测到年份列，请确认列名包含年份（如 2020 或 Y2020）。")
year_cols = sorted(year_cols, key=year_to_int)

base_col = None
for candidate in ["2020", 2020, "Y2020"]:
    if candidate in df_ssp.columns:
        base_col = candidate
        break
if base_col is None:
    raise KeyError("未找到基准年份 2020 的列（接受 '2020'、2020 或 'Y2020'）。")

# 计算相对 2020 的变化比（逐行）
df_ratio = df_ssp.copy()
base_vals = df_ratio[base_col].replace(0, np.nan)
for yc in year_cols:
    df_ratio[yc] = df_ratio[yc] / base_vals

# 将原有各表复制到新文件，并追加 change_ratio
with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
    for sheet in xls.sheet_names:
        pd.read_excel(ssp_path, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)
    df_ratio.to_excel(writer, sheet_name="change_ratio", index=False)

# =========================
# 2) change_ratio（区域） -> 国家
# =========================
# 仅保留映射所需列：SCENARIO、REGION + 年份列
keep_cols = ["SCENARIO", "REGION"] + year_cols
cr_reg = df_ratio[keep_cols].copy()
cr_reg["SCENARIO"] = cr_reg["SCENARIO"].astype(str).str.strip()
cr_reg["REGION"]   = cr_reg["REGION"].astype(str).str.strip()

# 读取 region 映射，并过滤 Region_label_new != 'no'
region_df = pd.read_excel(dict_path, sheet_name="region", dtype={"M49_Country_Code": str})
region_df["M49_Country_Code"]   = region_df["M49_Country_Code"].astype(str)
for c in ["Region_map_SSPDB", "Region_agg3#income", "Region_map_elasCat7", "Region_label_new"]:
    if c in region_df.columns:
        region_df[c] = region_df[c].astype(str).str.strip()

if "Region_label_new" in region_df.columns:
    region_df = region_df[region_df["Region_label_new"] != "no"].copy()

need_cols = ["M49_Country_Code", "Region_map_SSPDB", "Region_agg3#income", "Region_map_elasCat7"]
missing = [c for c in need_cols if c not in region_df.columns]
if missing:
    raise KeyError("dict_v3.xlsx['region'] 缺少列: " + ", ".join(missing))

# 2.1 直接映射（非 'Region_Income'）
direct_map = region_df[region_df["Region_map_SSPDB"] != "Region_Income"].copy()
scenarios = cr_reg["SCENARIO"].dropna().unique()
scen_df = pd.DataFrame({"SCENARIO": scenarios})

direct_expanded = (
    direct_map.assign(_k=1)
    .merge(scen_df.assign(_k=1), on="_k", how="left")
    .drop(columns=["_k"])
)
direct_expanded["SCENARIO"]         = direct_expanded["SCENARIO"].astype(str).str.strip()
direct_expanded["Region_map_SSPDB"] = direct_expanded["Region_map_SSPDB"].astype(str).str.strip()

direct_expanded = direct_expanded.merge(
    cr_reg,
    left_on=["SCENARIO", "Region_map_SSPDB"],
    right_on=["SCENARIO", "REGION"],
    how="left",
    suffixes=("", "_cr")
)

country_cols = ["M49_Country_Code", "SCENARIO", "Region_map_SSPDB", "Region_agg3#income", "Region_map_elasCat7"]
direct_country = direct_expanded[country_cols + year_cols].copy()

# 2.2 'Region_Income' 情形：先按 income 分组填，再按 elasCat7 回退
income_map = region_df[region_df["Region_map_SSPDB"] == "Region_Income"].copy()
if not income_map.empty:
    income_expanded = (
        income_map.assign(_k=1)
        .merge(scen_df.assign(_k=1), on="_k", how="left")
        .drop(columns=["_k"])
    )
    for yc in year_cols:
        income_expanded[yc] = np.nan
    income_country = income_expanded[country_cols + year_cols].copy()
else:
    income_country = pd.DataFrame(columns=country_cols + year_cols)

# A) 用 direct_country 的 (SCENARIO, Region_agg3#income) 分组均值填 income_country
mean_income = (
    direct_country
    .groupby(["SCENARIO", "Region_agg3#income"], dropna=False)[year_cols]
    .mean(numeric_only=True)
    .reset_index()
)
if not income_country.empty:
    income_filled = income_country.merge(
        mean_income,
        on=["SCENARIO", "Region_agg3#income"],
        how="left",
        suffixes=("", "_mean_income")
    )
    for yc in year_cols:
        income_filled[yc] = income_filled[yc].where(~income_filled[yc].isna(),
                                                    income_filled[f"{yc}_mean_income"])
        income_filled.drop(columns=[f"{yc}_mean_income"], inplace=True)
else:
    income_filled = income_country.copy()

# B) 若仍 NaN，按 (SCENARIO, Region_map_elasCat7) 分组均值回退
mean_elas = (
    pd.concat([direct_country, income_filled], ignore_index=True)
    .groupby(["SCENARIO", "Region_map_elasCat7"], dropna=False)[year_cols]
    .mean(numeric_only=True)
    .reset_index()
)
if not income_filled.empty:
    income_filled2 = income_filled.merge(
        mean_elas,
        on=["SCENARIO", "Region_map_elasCat7"],
        how="left",
        suffixes=("", "_mean_elas")
    )
    for yc in year_cols:
        income_filled2[yc] = income_filled2[yc].where(~income_filled2[yc].isna(),
                                                      income_filled2[f"{yc}_mean_elas"])
        income_filled2.drop(columns=[f"{yc}_mean_elas"], inplace=True)
else:
    income_filled2 = income_filled.copy()

# 合并国家层
change_ratio_country = pd.concat([direct_country, income_filled2], ignore_index=True)

# =========================
# 3) 额外稳健回填 + 时间轴反向填充
# =========================
# 3.1 再用 (SCENARIO, Region_agg3#income) 的均值回扫
mean_income2 = (
    change_ratio_country[~change_ratio_country[year_cols].isna().all(axis=1)]
    .groupby(["SCENARIO", "Region_agg3#income"], dropna=False)[year_cols]
    .mean(numeric_only=True)
    .reset_index()
)
tmp = change_ratio_country.merge(
    mean_income2,
    on=["SCENARIO", "Region_agg3#income"],
    how="left",
    suffixes=("", "_mean_income2")
)
for yc in year_cols:
    tmp[yc] = tmp[yc].where(~tmp[yc].isna(), tmp[f"{yc}_mean_income2"])
    tmp.drop(columns=[f"{yc}_mean_income2"], inplace=True)
change_ratio_country = tmp

# 3.2 再用 (SCENARIO, Region_map_elasCat7) 的均值回扫
mean_elas2 = (
    change_ratio_country[~change_ratio_country[year_cols].isna().all(axis=1)]
    .groupby(["SCENARIO", "Region_map_elasCat7"], dropna=False)[year_cols]
    .mean(numeric_only=True)
    .reset_index()
)
tmp = change_ratio_country.merge(
    mean_elas2,
    on=["SCENARIO", "Region_map_elasCat7"],
    how="left",
    suffixes=("", "_mean_elas2")
)
for yc in year_cols:
    tmp[yc] = tmp[yc].where(~tmp[yc].isna(), tmp[f"{yc}_mean_elas2"])
    tmp.drop(columns=[f"{yc}_mean_elas2"], inplace=True)
change_ratio_country = tmp

# 3.3 最后用 SCENARIO 均值兜底（极少数仍 NaN 的位置）
scen_means = (
    change_ratio_country[~change_ratio_country[year_cols].isna().all(axis=1)]
    .groupby(["SCENARIO"], dropna=False)[year_cols]
    .mean(numeric_only=True)
    .reset_index()
)
tmp = change_ratio_country.merge(scen_means, on=["SCENARIO"], how="left", suffixes=("", "_scen_mean"))
for yc in year_cols:
    tmp[yc] = tmp[yc].where(~tmp[yc].isna(), tmp[f"{yc}_scen_mean"])
    tmp.drop(columns=[f"{yc}_scen_mean"], inplace=True)
change_ratio_country = tmp

# 3.4 对每行按时间轴做一次反向填充（bfill），用最早可用的后续年份回填最早年份（如 Y2000/Y2005）
year_cols_sorted = sorted(year_cols, key=year_to_int)
change_ratio_country[year_cols_sorted] = change_ratio_country[year_cols_sorted].bfill(axis=1)

# 排序并写入
change_ratio_country = change_ratio_country.sort_values(["SCENARIO", "M49_Country_Code"]).reset_index(drop=True)
with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    change_ratio_country.to_excel(writer, sheet_name="change_ratio_country", index=False)

# 诊断输出（可注释）
residual_nan_rows = int(change_ratio_country[year_cols_sorted].isna().any(axis=1).sum())
print({
    "output_file": out_path,
    "sheets_written": ["change_ratio", "change_ratio_country"],
    "n_country_rows": int(change_ratio_country.shape[0]),
    "residual_nan_rows": residual_nan_rows,
    "year_cols": [str(c) for c in year_cols_sorted]
})
