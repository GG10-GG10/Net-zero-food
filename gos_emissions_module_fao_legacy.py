
# -*- coding: utf-8 -*-
"""
gv_emissions_module.py（排水有机土壤 GV 计算模块 · 与作物/牲畜风格一致）
==================================================================

模块作用
--------
- 读取“参数 WIDE 表”（支持 Excel/CSV，年份横向、国家/品类/过程/参数分层；
  兼容你之前的 GV 参数表：若检测到 'emission_process'/'land_use'/'climate_domain'，
  会自动标准化为 'process'/'commodity'/'ParamCLM'）。
- 统一从参数表取数（get_param_wide）：支持 iso3/country→ALL→GLOBAL 的回退，
  支持年份向左/向右就近填补，支持“气候带 ParamCLM”维度过滤。
- 计算 GV 两个过程：
    1) Drained organic soils - N2O (direct)  —— 直接 N₂O
    2) Drained organic soils - CO2 (on-site) —— 现场 CO₂（由 C→CO₂）
- 输出单位：kt（千吨），与作物/牲畜模块保持一致。

对接约定
--------
- 活动数据 areas_df（长表）至少包含：
    ['country','iso3','year','land_use','climate_zone','area_ha']
  其中 land_use ∈ {'Cropland','Grassland'}；climate_zone 与参数表中的 ParamCLM 一致
  （如 'Boreal/Cool temperate','Warm temperate','Tropical/Sub-tropical'）。
- 参数表：
    * 列：country, iso3, commodity(=land_use), process, parameter, units, 1961..
    * 可带 ParamCLM（或 climate_domain），作为“气候带”过滤键；
    * 也兼容你早前的 GV WIDE：emission_process→process，land_use→commodity，
      climate_domain→ParamCLM，EF_value→常数列（若无年度列）。

"""

from __future__ import annotations
from typing import Optional, Dict, List, Any, Iterable
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 常量与进制
# ─────────────────────────────────────────────────────────────────────────────
N2O_N_TO_N2O = 44.0/28.0  # kg N2O-N → kg N2O
C_TO_CO2     = 44.0/12.0  # t C → t CO2

PROC_N2O = "Drained organic soils - N2O (direct)"
PROC_CO2 = "Drained organic soils - CO2 (on-site)"

# ─────────────────────────────────────────────────────────────────────────────
# 参数表读取 & 标准化
# ─────────────────────────────────────────────────────────────────────────────
def load_params_wide(path: str) -> pd.DataFrame:
    """
    读取并“标准化” GV 参数 WIDE 表：
    - 若存在 'emission_process'，复制为 'process'；
    - 若存在 'land_use'，复制为 'commodity'（与作物/牲畜模块键一致）；
    - 若存在 'climate_domain'，复制为 'ParamCLM'（气候带过滤键）；
    - 若存在 'EF_value' 而无年度列，保留该常数列供后续回退；
    - 年份列（看起来像 '1990'..）将转为数值。
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=0, dtype=str)

    # 列名清洗
    cols = {c: c for c in df.columns}
    # 标准化典型别名
    if "emission_process" in df.columns and "process" not in df.columns:
        df["process"] = df["emission_process"]
    if "land_use" in df.columns and "commodity" not in df.columns:
        df["commodity"] = df["land_use"]
    if "climate_domain" in df.columns and "ParamCLM" not in df.columns:
        df["ParamCLM"] = df["climate_domain"]

    # 确保关键列存在
    for k in ["country","iso3","commodity","process","parameter","units","source","notes","ParamCLM"]:
        if k not in df.columns:
            df[k] = ""

    # 年份列转为数值
    for c in list(df.columns):
        if isinstance(c, str) and c.isdigit():
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # EF_value 常数列（可选）转数值
    if "EF_value" in df.columns:
        df["EF_value"] = pd.to_numeric(df["EF_value"], errors="coerce")

    # 统一字符串类型
    for k in ["country","iso3","commodity","process","parameter","units","source","notes","ParamCLM"]:
        df[k] = df[k].fillna("").astype(str)

    return df


def _year_value(row: pd.Series, year: int) -> Optional[float]:
    """从一行中按年份列取值；若当年为空，向左（过去）→向右（未来）就近回填；若全年皆空，返回 None。"""
    ycols = [c for c in row.index if isinstance(c, str) and c.isdigit()]
    if str(year) in row.index and pd.notna(row[str(year)]):
        return float(row[str(year)])
    # 向过去
    for y in sorted([int(c) for c in ycols if int(c) < year], reverse=True):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    # 向未来
    for y in sorted([int(c) for c in ycols if int(c) > year]):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    # 若无年度列，则尝试常数列 EF_value
    if "EF_value" in row.index and pd.notna(row["EF_value"]):
        return float(row["EF_value"])
    return None


def get_param_wide(
    P: pd.DataFrame,
    *, country: str, iso3: str, commodity: str,
    process: str, parameter: str, year: int,
    ParamCLM: Optional[str] = None,
    default: Optional[float] = None
) -> Optional[float]:
    """
    从 WIDE 表读取标量参数；与作物/牲畜版风格一致：
      - 维度：country/iso3、commodity(=land_use)、process、parameter、ParamCLM(=climate zone)、year
      - 回退：本国→本国+ALL→GLOBAL/GLB（country in {GLOBAL/World} 或 iso3 in {GLB}）
      - 年度：当年→向左/向右就近→常数列 EF_value
    """
    cc_keys: List[str] = [x for x in [(iso3 or "").strip(), (country or "").strip()] if x]
    if not cc_keys:
        cc_keys = [""]

    # 过滤：过程、商品（地类）、参数
    sub = P[(P["process"] == process) & (P["commodity"] == commodity) & (P["parameter"] == parameter)].copy()

    # 气候带过滤（ParamCLM）
    if ParamCLM:
        if "ParamCLM" in sub.columns:
            sub = sub[sub["ParamCLM"] == ParamCLM]
        elif "climate_domain" in sub.columns:
            sub = sub[sub["climate_domain"] == ParamCLM]

    # 1) 本国/ISO 精确
    s1 = sub[(sub["country"].isin(cc_keys)) | (sub["iso3"].isin(cc_keys))]
    if not s1.empty:
        v = _year_value(s1.iloc[0], year)
        if v is not None:
            return v
    # 2) 本国 + ALL（商品层 ALL）
    s2 = P[(P["process"] == process) & (P["commodity"] == "ALL") & (P["parameter"] == parameter)]
    if ParamCLM:
        if "ParamCLM" in s2.columns:
            s2 = s2[s2["ParamCLM"] == ParamCLM]
        elif "climate_domain" in s2.columns:
            s2 = s2[s2["climate_domain"] == ParamCLM]
    s2 = s2[(s2["country"].isin(cc_keys)) | (s2["iso3"].isin(cc_keys))]
    if not s2.empty:
        v = _year_value(s2.iloc[0], year)
        if v is not None:
            return v
    # 3) GLOBAL/GLB
    s3 = P[(P["process"] == process) & (P["parameter"] == parameter) & (P["commodity"].isin([commodity,"ALL"]))]
    s3 = s3[(s3["country"].isin(["GLOBAL","Global","World"])) | (s3["iso3"].isin(["GLB","WORLD"]))]
    if ParamCLM:
        if "ParamCLM" in s3.columns:
            s3 = s3[s3["ParamCLM"] == ParamCLM]
        elif "climate_domain" in s3.columns:
            s3 = s3[s3["climate_domain"] == ParamCLM]
    if not s3.empty:
        v = _year_value(s3.iloc[0], year)
        if v is not None:
            return v

    return default


# ─────────────────────────────────────────────────────────────────────────────
# 单位处理：将 EF 解释为“每公顷每年”的排放强度，并换算到目标气体单位
# ─────────────────────────────────────────────────────────────────────────────
def _ef_to_n2o_kg_per_ha(ef_value: float, units: str) -> float:
    """
    将多种单位的 N2O 因子，统一换算为“kg N2O/ha/yr”。
    支持：kg N2O-N/ha/yr（×44/28）、kg N2O/ha/yr（原样）、t N2O/ha/yr（×1000）等。
    """
    u = (units or "").lower().replace(" ", "")
    x = float(ef_value or 0.0)
    if "n2o-n" in u:
        return x * N2O_N_TO_N2O
    if "kgn2o" in u:
        return x
    if "tn2o" in u:
        return x * 1000.0
    # 兜底：按 kg N2O 处理
    return x


def _ef_to_co2_t_per_ha(ef_value: float, units: str) -> float:
    """
    将多种单位的 CO2 因子，统一换算为“t CO2/ha/yr”。
    支持：t C/ha/yr（×44/12）、t CO2/ha/yr（原样）、kg CO2/ha/yr（/1000）等。
    """
    u = (units or "").lower().replace(" ", "")
    x = float(ef_value or 0.0)
    if ("tc/ha" in u) or ("tcha" in u) or ("tcperha" in u):
        return x * C_TO_CO2
    if ("tco2/ha" in u) or ("tco2ha" in u) or ("tco2perha" in u):
        return x
    if ("kgco2/ha" in u) or ("kgco2ha" in u):
        return x / 1000.0
    # 若单位未知，保守认为给的是 t CO2/ha/yr
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 过程计算
# ─────────────────────────────────────────────────────────────────────────────
def compute_gv_n2o(P: pd.DataFrame, areas_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    直接 N2O：GV（排水有机土壤）
    输入：areas_df 至少含 ['country','iso3','year','land_use','climate_zone','area_ha']
    参数表：process=PROC_N2O，parameter='EF_value' 或年度列；commodity=land_use；ParamCLM=climate_zone
    输出列：N2O_kt
    """
    req = {"country","iso3","year","land_use","climate_zone","area_ha"}
    if not req.issubset(areas_df.columns):
        missing = req - set(areas_df.columns)
        raise ValueError(f"areas_df 缺少列: {missing}")

    out = []
    for _, r in areas_df.iterrows():
        if int(r["year"]) != year:
            continue
        c,i3,lu,cz = str(r["country"]), str(r["iso3"]), str(r["land_use"]), str(r["climate_zone"])
        A = float(r["area_ha"] or 0.0)

        # 取 EF 数值与单位
        ef = get_param_wide(P, country=c, iso3=i3, commodity=lu, process=PROC_N2O,
                            parameter="EF_value", year=year, ParamCLM=cz, default=None)
        if ef is None:
            raise ValueError(f"缺 N2O EF：{c}/{i3}/{year}/{lu}/{cz}")
        # 单位字段：优先本行参数表单位（按相同筛选条件选取第一行的 units）
        sub = P[(P["process"]==PROC_N2O) & (P["commodity"]==lu) & (P["parameter"]=="EF_value")]
        if "ParamCLM" in P.columns:
            sub = sub[sub["ParamCLM"]==cz]
        elif "climate_domain" in P.columns:
            sub = sub[sub["climate_domain"]==cz]
        if not sub.empty and isinstance(sub.iloc[0].get("units", None), str):
            units = sub.iloc[0]["units"]
        else:
            units = "kg N2O-N ha-1 yr-1"  # 缺省

        ef_kgN2O_per_ha = _ef_to_n2o_kg_per_ha(ef, units)
        N2O_kt = A * ef_kgN2O_per_ha / 1e6  # kg → kt

        out.append({
            "country": c, "iso3": i3, "year": year,
            "land_use": lu, "climate_zone": cz,
            "process": PROC_N2O, "gas": "N2O",
            "N2O_kt": N2O_kt
        })

    return pd.DataFrame(out)


def compute_gv_co2(P: pd.DataFrame, areas_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    CO2（on-site）：GV（排水有机土壤）
    输入：areas_df 至少含 ['country','iso3','year','land_use','climate_zone','area_ha']
    参数表：process=PROC_CO2，parameter='EF_value' 或年度列；commodity=land_use；ParamCLM=climate_zone
    输出列：CO2_kt
    """
    req = {"country","iso3","year","land_use","climate_zone","area_ha"}
    if not req.issubset(areas_df.columns):
        missing = req - set(areas_df.columns)
        raise ValueError(f"areas_df 缺少列: {missing}")

    out = []
    for _, r in areas_df.iterrows():
        if int(r["year"]) != year:
            continue
        c,i3,lu,cz = str(r["country"]), str(r["iso3"]), str(r["land_use"]), str(r["climate_zone"])
        A = float(r["area_ha"] or 0.0)

        ef = get_param_wide(P, country=c, iso3=i3, commodity=lu, process=PROC_CO2,
                            parameter="EF_value", year=year, ParamCLM=cz, default=None)
        if ef is None:
            raise ValueError(f"缺 CO2 EF：{c}/{i3}/{year}/{lu}/{cz}")
        sub = P[(P["process"]==PROC_CO2) & (P["commodity"]==lu) & (P["parameter"]=="EF_value")]
        if "ParamCLM" in P.columns:
            sub = sub[sub["ParamCLM"]==cz]
        elif "climate_domain" in P.columns:
            sub = sub[sub["climate_domain"]==cz]
        if not sub.empty and isinstance(sub.iloc[0].get("units", None), str):
            units = sub.iloc[0]["units"]
        else:
            units = "t C ha-1 yr-1"  # 缺省（IPCC 缺省为 C 基）

        ef_tCO2_per_ha = _ef_to_co2_t_per_ha(ef, units)
        CO2_kt = A * ef_tCO2_per_ha / 1e3  # t → kt

        out.append({
            "country": c, "iso3": i3, "year": year,
            "land_use": lu, "climate_zone": cz,
            "process": PROC_CO2, "gas": "CO2",
            "CO2_kt": CO2_kt
        })

    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# 编排器（供 main 调用）
# ─────────────────────────────────────────────────────────────────────────────
def run_gv(
    P: pd.DataFrame,
    *, year: int,
    areas_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    统一编排 GV 两个过程；
    返回 dict：{'gv_n2o': df, 'gv_co2': df}
    """
    res: Dict[str, pd.DataFrame] = {}
    res["gv_n2o"] = compute_gv_n2o(P, areas_df, year)
    res["gv_co2"] = compute_gv_co2(P, areas_df, year)
    return res


__all__ = [
    "load_params_wide", "get_param_wide",
    "compute_gv_n2o", "compute_gv_co2",
    "run_gv",
    "PROC_N2O", "PROC_CO2",
]
