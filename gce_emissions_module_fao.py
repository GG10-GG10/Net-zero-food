
# -*- coding: utf-8 -*-
"""
gce_emissions_module.py（中文注释加强版）
======================================

模块定位
--------
本模块实现作物部门（GCE：Global Crop Emissions）的排放计算引擎，
与 FAOSTAT GCE 域及 IPCC 2006 指南（Volume 4）对齐，提供**无状态、可组合**的
过程级函数（纯函数），便于在你们的 `main` 中直接导入与编排。

覆盖内容（Tier‑1 / Tier‑1.5 脚手架）
-----------------------------------
1) 作物秸秆还田 —— **直接 N₂O**
2) 作物秸秆还田 —— **间接 N₂O**（经淋洗/径流；可选大气沉降支路）
3) 露天焚烧秸秆 —— **CH₄ 与 N₂O**
4) 稻作栽培 —— **CH₄**
5) 合成氮肥 —— **直接与间接 N₂O**

设计要点
--------
- **参数访问**：从“WIDE 参数簿”读取（行键：country/iso3/commodity/process/parameter；列为年份 1961..），
  年份按列取值；查找次序为 **ISO3 优先 → GLOBAL/GLB 回退**。
- **纯函数**：每个 `compute_*` 仅做输入→输出转换，无外部副作用。
- **单位透明**：函数注释中清晰注明所需输入列和单位，显式进行单位换算。
- **鲁棒性**：焚烧过程的燃料负荷 MB（kgDM/ha）若未显式给出，将根据产量/单产与 RPR 自动推导（Tier‑1.5）。

典型用法（在 main 中）
----------------------
    from gce_emissions_module import load_params_wide, run_gce
    P = load_params_wide("GCE_parameters_WIDE_v3_IPCC.xlsx")

    results = run_gce(
        params=P,
        residues_df=...,      # 秸秆还田直接/间接 N₂O
        burning_df=...,       # 露天焚烧 CH₄/N₂O
        rice_df=...,          # 稻作 CH₄
        fertilizers_df=...    # 合成肥 N₂O（直接/间接）
    )
    # `results` 为 {模块名: DataFrame} 的字典

参考资料（参数真值来源）
------------------------
- IPCC 2006 Guidelines for National GHG Inventories, Volume 4.
  *Ch.11（N₂O：土壤/作物残体）、Ch.2（通用系数）、Ch.5（耕地）、稻作良好实践等*
- EFDB（IPCC Emission Factor Database）
- FAOSTAT GCE 方法说明（意图对齐）

符号与常量
----------
- `N2O_N_TO_N2O = 44/28`：将 **kg N₂O‑N → kg N₂O** 的化学计量换算。
- `G_PER_M2_TO_KT_PER_HA = 10,000 / 1e9`：将 **g/m² × ha → kt** 的换算因子。

局限 / 待扩展
-------------
- 假定活动数据整洁；包含基本一致性检查。
- 稻作水分制度份额等国家特异信息建议进入 WIDE（或在活动表直接给面积拆分）。
- 可按需增补甲烷氧化、季节长度、有机改良系数等参数（取参模式已支持）。
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, Mapping
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 全局常量 —— 单位换算
# ─────────────────────────────────────────────────────────────────────────────
#: kg N2O‑N → kg N2O（分子量 44/28）
N2O_N_TO_N2O = 44.0 / 28.0
#: g/m²（季）× ha → kt；1 ha = 10,000 m²，1 kt = 10^9 g
G_PER_M2_TO_KT_PER_HA = 1e4 / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# 参数工具（WIDE 结构）
# ─────────────────────────────────────────────────────────────────────────────
def load_params_wide(xlsx_path: str) -> pd.DataFrame:
    """
    读取 WIDE 参数簿（Excel，表名 `parameters_wide`）

    期望结构
    --------
    列：
      - country   : str（例如 "GLOBAL"）
      - iso3      : str（例如 "GLB" 或 "CHN"）
      - commodity : str（作物名，如 "Wheat"、"Rice-paddy"，或肥料用 "ALL"）
      - process   : str（过程名，如 "crop_residues_direct_N2O"、"rice_cultivation_CH4" 等）
      - parameter : str（参数名）
      - units/source/notes : 文档信息
      - 1961..2022 : 各年份的数值列（float）
    """
    return pd.read_excel(xlsx_path, sheet_name="parameters_wide")


def get_param_wide(params_wide: pd.DataFrame,
                   country: str,
                   iso3: str,
                   year: int,
                   commodity: str,
                   process: str,
                   parameter: str) -> Optional[float]:
    """
    从 WIDE 参数簿中取单个参数值，按 **ISO3 → GLOBAL/GLB** 的优先级回退。

    查找逻辑
    --------
    1) ISO3 精确命中（例如 "CHN"）；若该年份列非空则返回；
    2) GLOBAL 回退：country == "GLOBAL" 且 iso3 == "GLB" 的行。

    返回
    ----
    float 或 None（未命中）。
    """
    col = str(int(year))
    sub = params_wide[
        (params_wide["commodity"] == commodity) &
        (params_wide["process"] == process) &
        (params_wide["parameter"] == parameter)
    ]
    # 1) ISO3 优先
    v = sub[sub["iso3"] == iso3]
    if not v.empty and col in v.columns and pd.notna(v.iloc[0][col]):
        return float(v.iloc[0][col])
    # 2) GLOBAL/GLB 回退
    v = sub[(sub["country"] == "GLOBAL") & (sub["iso3"] == "GLB")]
    if not v.empty and col in v.columns and pd.notna(v.iloc[0][col]):
        return float(v.iloc[0][col])
    return None


def _need(params_wide: pd.DataFrame,
          country: str, iso3: str, year: int,
          commodity: str, process: str,
          names: Iterable[str]) -> Dict[str, Any]:
    """
    一次取回**一组**参数；若在 GLOBAL 回退后仍缺，直接抛错并点名缺项。

    用途
    ----
    供各 `compute_*` 函数“速取+校验”使用，便于**快速失败**与定位。

    返回
    ----
    {parameter_name: value} 字典（float）。
    """
    out: Dict[str, Any] = {}
    missing = []
    for n in names:
        v = get_param_wide(params_wide, country, iso3, year, commodity, process, n)
        if v is None:
            missing.append(n)
        out[n] = v
    if missing:
        raise ValueError(
            f"参数缺失: {country}/{iso3}/{year}/{commodity}/{process}: {missing}"
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 共享推导函数（纯辅助，不改动外部状态）
# ─────────────────────────────────────────────────────────────────────────────
def _derive_residue_N(row: pd.Series, P: Mapping[str, float]) -> float:
    """
    推导“作物残体中的氮量”（单位：kg N）。

    所需活动数据（其一）
    --------------------
    - `production_t`（t）
      或
    - `harvested_area_ha`（ha）与 `yield_t_per_ha`（t/ha）

    所需参数（来自 P）
    ------------------
    - RPR_aboveground_per_crop（单位产品的地上残体-干重比）
    - R_S_root_to_shoot       （根/茎比；无则按 0 处理）
    - N_frac_above / N_frac_below（残体干重中的含氮分数）
    - DM_content_residue（残体干物质系数；若 RPR 已是干基可设 1.0）

    公式
    ----
      Prod_kg     = production_t × 1000
      Above_DM_kg = Prod_kg × RPR × DM
      Below_DM_kg = Above_DM_kg × R/S
      N_res_kg    = Above_DM_kg × N_frac_above + Below_DM_kg × N_frac_below
    """
    # 若活动表已直接给出 N_in_residues_kgN，则优先使用（跳过推导）
    if "N_in_residues_kgN" in row and pd.notna(row["N_in_residues_kgN"]):
        return float(row["N_in_residues_kgN"])

    # 1) 获取产量吨
    if pd.notna(row.get("production_t")):
        prod_t = float(row["production_t"])
    elif pd.notna(row.get("harvested_area_ha")) and pd.notna(row.get("yield_t_per_ha")):
        prod_t = float(row["harvested_area_ha"]) * float(row["yield_t_per_ha"])
    else:
        raise ValueError("推导残体氮需要 production_t 或 (harvested_area_ha 与 yield_t_per_ha)。")

    # 2) 计算残体干重（地上 + 地下）
    above_dm_kg = prod_t * 1000.0 * P["RPR_aboveground_per_crop"] * P["DM_content_residue"]
    below_dm_kg = above_dm_kg * P["R_S_root_to_shoot"]

    # 3) 含氮量汇总（kg N）
    return above_dm_kg * P["N_frac_above"] + below_dm_kg * P["N_frac_below"]


def _derive_MB_if_needed(row: pd.Series,
                         P_proc: Mapping[str, float],
                         P_crop: Mapping[str, float]) -> float:
    """
    推导露天焚烧的燃料负荷 MB（kgDM/ha），在**未显式给出**时使用。

    优先级
    ------
    1) 使用活动表中的 `MB_fuel_biomass_kgDM_per_ha`；
    2) 若无，使用**过程参数**中的 `MB_fuel_biomass_kgDM_per_ha`；
    3) 若仍无，用“单产 × RPR（× DM）”推导：
         MB ≈ yield_t_per_ha × 1000 × RPR × DM
       其中 `yield_t_per_ha` 可由 `production_t / harvested_area_ha` 反推。
    """
    # 1) 行级覆盖
    if "MB_fuel_biomass_kgDM_per_ha" in row and pd.notna(row["MB_fuel_biomass_kgDM_per_ha"]):
        return float(row["MB_fuel_biomass_kgDM_per_ha"])
    # 2) 过程级默认
    if P_proc.get("MB_fuel_biomass_kgDM_per_ha") is not None:
        return float(P_proc["MB_fuel_biomass_kgDM_per_ha"])
    # 3) 由单产与 RPR 推导
    if pd.notna(row.get("yield_t_per_ha")):
        yld = float(row["yield_t_per_ha"])
    elif pd.notna(row.get("production_t")) and pd.notna(row.get("harvested_area_ha")) and float(row["harvested_area_ha"]) != 0.0:
        yld = float(row["production_t"]) / float(row["harvested_area_ha"])
    else:
        raise ValueError("推导 MB 需要 MB_fuel_biomass_kgDM_per_ha 或 (yield/production+area)。")
    return yld * 1000.0 * float(P_crop["RPR_aboveground_per_crop"]) * float(P_crop["DM_content_residue"])


# ─────────────────────────────────────────────────────────────────────────────
# 过程计算函数（逐过程）
# ─────────────────────────────────────────────────────────────────────────────
def compute_crop_residues_direct(params_wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    作物残体**直接 N₂O**（IPCC Ch.11）

    输入 df（最少列）
    ----------------
    - country, iso3, year, commodity
    - 二选一：production_t（t） 或 (harvested_area_ha, yield_t_per_ha)
    - 可选：N_in_residues_kgN（若给出，跳过 RPR 推导）

    取用参数（_need 保证存在）
    ------------------------
    - EF1_N2O_N_per_N
    - RPR_aboveground_per_crop, R_S_root_to_shoot
    - N_frac_above, N_frac_below
    - DM_content_residue
    - Frac_burnt_on_site（还田中被现场焚烧的比例，用于折减）
    - Combustion_factor_Cf（燃烧有效性）

    计算要点
    --------
    1) 得到残体氮 N_res（kg N），扣除现场焚烧（乘 (1 - Frac_burnt_on_site × Cf)）；
    2) 乘以 EF1 得到 N₂O‑N；
    3) ×(44/28) 转为 N₂O，最后 /1e6 → kt。
    """
    rows = []
    for _, r in df.iterrows():
        c,i3,y,com = r["country"], r["iso3"], int(r["year"]), r["commodity"]
        P = _need(params_wide, c,i3,y,com,"crop_residues_direct_N2O",
                  ["EF1_N2O_N_per_N","RPR_aboveground_per_crop","R_S_root_to_shoot",
                   "N_frac_above","N_frac_below","DM_content_residue",
                   "Frac_burnt_on_site","Combustion_factor_Cf"])
        # 残体氮（kg N）
        N_res = _derive_residue_N(r, P)
        # 扣除现场焚烧部分
        N_left = N_res * max(0.0, 1.0 - P["Frac_burnt_on_site"] * P["Combustion_factor_Cf"])
        # EF1 → N2O‑N → N2O(kt)
        N2O_N = N_left * P["EF1_N2O_N_per_N"]
        rows.append({"country":c,"iso3":i3,"year":y,"commodity":com,
                     "process":"crop_residues_direct_N2O",
                     "N2O_kt": (N2O_N * N2O_N_TO_N2O) / 1e6})
    return pd.DataFrame(rows)


def compute_crop_residues_indirect(params_wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    作物残体**间接 N₂O**（IPCC Ch.11）：经**淋洗/径流**与（可选）**大气沉降**路径。

    输入 df 同 `compute_crop_residues_direct`。

    取用参数
    --------
    必需：RPR_aboveground_per_crop、R_S_root_to_shoot、N_frac_above、N_frac_below、DM_content_residue、
          Frac_burnt_on_site、Combustion_factor_Cf、FracLEACH_N、EF5_N2O_N_per_Nleach
    可选：EF4_N2O_N_per_Nvolatilized、FracGASF_volatilization（若两者齐备，加上大气沉降支路）

    计算
    ----
      N_left = 残体氮折减后
      N2O_N = N_left × FracLEACH_N × EF5   [+  N_left × FracGASF × EF4]
    """
    rows = []
    for _, r in df.iterrows():
        c,i3,y,com = r["country"], r["iso3"], int(r["year"]), r["commodity"]
        P = _need(params_wide, c,i3,y,com,"crop_residues_indirect_N2O",
                  ["RPR_aboveground_per_crop","R_S_root_to_shoot","N_frac_above","N_frac_below",
                   "DM_content_residue","Frac_burnt_on_site","Combustion_factor_Cf",
                   "FracLEACH_N","EF5_N2O_N_per_Nleach"])
        N_res  = _derive_residue_N(r, P)
        N_left = N_res * max(0.0, 1.0 - P["Frac_burnt_on_site"] * P["Combustion_factor_Cf"])
        N2O_N  = N_left * P["FracLEACH_N"] * P["EF5_N2O_N_per_Nleach"]

        # 可选：大气沉降路径（两个参数都存在才计入）
        EF4 = get_param_wide(params_wide, c,i3,y,com,"crop_residues_indirect_N2O","EF4_N2O_N_per_Nvolatilized")
        FracGASF = get_param_wide(params_wide, c,i3,y,com,"crop_residues_indirect_N2O","FracGASF_volatilization")
        if (EF4 is not None) and (FracGASF is not None):
            N2O_N += N_left * FracGASF * EF4

        rows.append({"country":c,"iso3":i3,"year":y,"commodity":com,
                     "process":"crop_residues_indirect_N2O",
                     "N2O_kt": (N2O_N * N2O_N_TO_N2O) / 1e6})
    return pd.DataFrame(rows)


def compute_burning(params_wide: pd.DataFrame, df: pd.DataFrame, gas: str="CH4") -> pd.DataFrame:
    """
    露天焚烧秸秆（IPCC Ch.2/Ch.5；EF 参考 Table 2.5；Cf 参考 Table 2.6）。

    输入 df（最少列）
    ----------------
    - country, iso3, year, commodity
    - harvested_area_ha : float
    - 可选：
        • MB_fuel_biomass_kgDM_per_ha —— 若缺失将通过 `_derive_MB_if_needed` 用单产×RPR 推导
        • yield_t_per_ha 或 (production_t 与 harvested_area_ha) —— 仅当需要推导 MB 时使用

    取用参数（过程层）
    ------------------
      - MB_fuel_biomass_kgDM_per_ha（kgDM/ha，可选默认）
      - Combustion_factor_Cf（燃烧有效性）
      - Fraction_burnt_area（发生焚烧的面积比例）
      - EF_g_per_kgDM（每 kg 干物质的排放因子，单位 g 气体/kgDM；气体种类相关）
    取用参数（作物层；仅用于 MB 推导）
      - RPR_aboveground_per_crop, DM_content_residue

    计算
    ----
      Burned_DM_kg = harvested_area_ha × MB × Cf × Fraction_burnt_area
      Emission_kg  = Burned_DM_kg × EF_g_per_kgDM / 1000
      Output kt    = Emission_kg / 1000
    """
    assert gas in ("CH4","N2O"), "gas 必须为 'CH4' 或 'N2O'"
    proc = f"burning_residues_{gas}"
    out_col = f"{gas}_kt"
    rows = []
    for _, r in df.iterrows():
        c,i3,y,com = r["country"], r["iso3"], int(r["year"]), r["commodity"]
        # 过程参数
        P_proc = _need(params_wide, c,i3,y,com,proc,
                       ["MB_fuel_biomass_kgDM_per_ha","Combustion_factor_Cf","Fraction_burnt_area","EF_g_per_kgDM"])
        # 作物参数（供 MB 推导）
        P_crop = _need(params_wide, c,i3,y,com,"crop_residues_direct_N2O",
                       ["RPR_aboveground_per_crop","DM_content_residue"])
        MB = _derive_MB_if_needed(r, P_proc, P_crop)
        burned_dm_kg = float(r["harvested_area_ha"]) * MB * P_proc["Combustion_factor_Cf"] * P_proc["Fraction_burnt_area"]
        emission_kg  = burned_dm_kg * P_proc["EF_g_per_kgDM"] / 1000.0
        rows.append({"country":c,"iso3":i3,"year":y,"commodity":com,"process":proc, out_col: emission_kg/1e3})
    return pd.DataFrame(rows)


def compute_synth_fert_direct(params_wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    合成氮肥 —— **直接 N₂O**（IPCC Ch.11 Table 11.1）。

    输入 df
    -------
    - country, iso3, year
    - N_synthetic_kg : float（合成氮肥施氮量，kg N）

    参数
    ----
    - EF1_N2O_N_per_N（kg N₂O‑N / kg N）
    """
    rows = []
    for _, r in df.iterrows():
        c,i3,y = r["country"], r["iso3"], int(r["year"])
        EF1 = get_param_wide(params_wide, c,i3,y,"ALL","synthetic_fertilizer_direct_N2O","EF1_N2O_N_per_N")
        if EF1 is None:
            raise ValueError(f"缺 EF1: {c}/{i3}/{y} synthetic_fertilizer_direct_N2O")
        N2O_N = float(r["N_synthetic_kg"]) * EF1
        rows.append({"country":c,"iso3":i3,"year":y,"commodity":"ALL",
                     "process":"synthetic_fertilizer_direct_N2O",
                     "N2O_kt": (N2O_N * N2O_N_TO_N2O) / 1e6})
    return pd.DataFrame(rows)


def compute_synth_fert_indirect(params_wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    合成氮肥 —— **间接 N₂O**（经挥发与淋洗/径流）。

    输入 df
    -------
    - country, iso3, year
    - N_synthetic_kg : float（kg N）

    参数
    ----
    - FracGASF_volatilization（挥发分数，kg NH₃‑N+NOx‑N / kg N）
    - EF4_N2O_N_per_Nvolatilized（kg N₂O‑N / kg(NH₃‑N+NOx‑N)）
    - FracLEACH_N（淋洗/径流分数，kg N leached / kg N）
    - EF5_N2O_N_per_Nleach（kg N₂O‑N / kg N leached）
    """
    rows = []
    for _, r in df.iterrows():
        c,i3,y = r["country"], r["iso3"], int(r["year"])
        FracGASF = get_param_wide(params_wide, c,i3,y,"ALL","synthetic_fertilizer_indirect_N2O","FracGASF_volatilization")
        EF4      = get_param_wide(params_wide, c,i3,y,"ALL","synthetic_fertilizer_indirect_N2O","EF4_N2O_N_per_Nvolatilized")
        FracLEACH= get_param_wide(params_wide, c,i3,y,"ALL","synthetic_fertilizer_indirect_N2O","FracLEACH_N")
        EF5      = get_param_wide(params_wide, c,i3,y,"ALL","synthetic_fertilizer_indirect_N2O","EF5_N2O_N_per_Nleach")
        if None in (FracGASF, EF4, FracLEACH, EF5):
            raise ValueError(f"间接参数缺失: {c}/{i3}/{y}")
        N = float(r["N_synthetic_kg"])
        N2O_N = N*FracGASF*EF4 + N*FracLEACH*EF5
        rows.append({"country":c,"iso3":i3,"year":y,"commodity":"ALL",
                     "process":"synthetic_fertilizer_indirect_N2O",
                     "N2O_kt": (N2O_N * N2O_N_TO_N2O) / 1e6})
    return pd.DataFrame(rows)


def compute_rice_ch4(params_wide: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    稻作栽培 **CH₄**（与 FAOSTAT 对齐的做法，参考 IPCC/Good Practice）

    输入 df
    -------
    - country, iso3, year
    - 面积拆分二选一：
        • 显式给出：area_irrigated_ha, area_rainfed_ha, area_upland_ha
      或
        • 给出总面积 area_rice_ha（模块将读取参数 Share_area_* 拆分）
    - 可选：若 WIDE 提供有机改良等修正因子，会一并读取。

    参数（WIDE）
    ------------
    - EF0_base_CH4_g_m2_season（基线，灌溉稻 g CH₄ / m²·季）
    - SF_rainfed, SF_upland（相对于灌溉稻的缩放因子）
    - CF_organic_amendments（有机改良修正系数）
    - Frac_farmers_using_organic（采用有机改良的农户比例）
    - Share_area_irrigated / rainfed / upland（用于从总面积拆分）

    计算
    ----
      org_factor = (1 - frac_org) + frac_org * CF_org
      EF_ir = EF0 * 1.0       * org_factor
      EF_rf = EF0 * SF_rainfed* org_factor
      EF_up = EF0 * SF_upland * org_factor
      CH4_kt = (EF_ir*A_ir + EF_rf*A_rf + EF_up*A_up) × G_PER_M2_TO_KT_PER_HA
    """
    rows = []
    for _, r in df.iterrows():
        c,i3,y = r["country"], r["iso3"], int(r["year"])
        com = "Rice-paddy"
        P = _need(params_wide, c,i3,y,com,"rice_cultivation_CH4",
                  ["EF0_base_CH4_g_m2_season","SF_rainfed","SF_upland",
                   "CF_organic_amendments","Frac_farmers_using_organic",
                   "Share_area_irrigated","Share_area_rainfed","Share_area_upland"])
        # 面积：优先使用显式拆分，否则由总面积与份额拆分
        if all(k in r and pd.notna(r[k]) for k in ["area_irrigated_ha","area_rainfed_ha","area_upland_ha"]):
            A_ir, A_rf, A_up = float(r["area_irrigated_ha"]), float(r["area_rainfed_ha"]), float(r["area_upland_ha"])
        else:
            if "area_rice_ha" not in r or pd.isna(r["area_rice_ha"]):
                raise ValueError("稻作 CH₄ 需要 (area_irrigated_ha, area_rainfed_ha, area_upland_ha) 或 area_rice_ha。")
            A_total = float(r["area_rice_ha"])
            A_ir = A_total * P["Share_area_irrigated"]
            A_rf = A_total * P["Share_area_rainfed"]
            A_up = A_total * P["Share_area_upland"]
        # 有机改良修正
        frac_org = P["Frac_farmers_using_organic"]
        CF_org   = P["CF_organic_amendments"]
        org_factor = (1 - frac_org) + frac_org * CF_org
        # 各水分制度的 EF（g/m²·季）
        EF_ir = P["EF0_base_CH4_g_m2_season"] * 1.0            * org_factor
        EF_rf = P["EF0_base_CH4_g_m2_season"] * P["SF_rainfed"] * org_factor
        EF_up = P["EF0_base_CH4_g_m2_season"] * P["SF_upland"]  * org_factor
        # 转 kt
        CH4_kt = (EF_ir*A_ir + EF_rf*A_rf + EF_up*A_up) * G_PER_M2_TO_KT_PER_HA
        rows.append({"country":c,"iso3":i3,"year":y,"commodity":com,
                     "process":"rice_cultivation_CH4","CH4_kt": CH4_kt})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 顶层编排器（供 main 调用）
# ─────────────────────────────────────────────────────────────────────────────
def run_gce(params: pd.DataFrame,
            residues_df: Optional[pd.DataFrame]=None,
            burning_df: Optional[pd.DataFrame]=None,
            rice_df: Optional[pd.DataFrame]=None,
            fertilizers_df: Optional[pd.DataFrame]=None) -> Dict[str, pd.DataFrame]:
    """
    统一编排四类过程。传入你已有的活动表（可以为 None）。

    参数
    ----
    params : `load_params_wide` 读到的 WIDE 参数表
    residues_df : 作物残体活动表（用于直接与间接 N₂O）
        必含列：country, iso3, year, commodity；其一：production_t 或 (harvested_area_ha, yield_t_per_ha)
        可选：N_in_residues_kgN（若给出将跳过 RPR 推导）
    burning_df : 露天焚烧活动表
        必含列：country, iso3, year, commodity, harvested_area_ha
        可选：MB_fuel_biomass_kgDM_per_ha；若无需推导 MB 则需提供单产或(产量+面积)
    rice_df : 稻作 CH₄ 活动表
        必含：country, iso3, year；面积拆分显式或 total+参数份额拆分
    fertilizers_df : 合成氮肥活动表
        必含：country, iso3, year, N_synthetic_kg（kg N）

    返回
    ----
    dict[str, DataFrame]：可能包含
      - 'residue_direct', 'residue_indirect'
      - 'burning_ch4', 'burning_n2o'
      - 'rice_ch4'
      - 'fert_direct', 'fert_indirect'
    """
    out: Dict[str, pd.DataFrame] = {}

    if residues_df is not None and not residues_df.empty:
        out["residue_direct"]  = compute_crop_residues_direct(params, residues_df)
        out["residue_indirect"]= compute_crop_residues_indirect(params, residues_df)

    if burning_df is not None and not residues_df is None:  # 允许空 DataFrame 判断
        if not burning_df.empty:
            out["burning_ch4"] = compute_burning(params, burning_df, gas="CH4")
            out["burning_n2o"] = compute_burning(params, burning_df, gas="N2O")

    if rice_df is not None and not rice_df.empty:
        out["rice_ch4"] = compute_rice_ch4(params, rice_df)

    if fertilizers_df is not None and not fertilizers_df.empty:
        out["fert_direct"]  = compute_synth_fert_direct(params, fertilizers_df)
        out["fert_indirect"]= compute_synth_fert_indirect(params, fertilizers_df)

    return out


__all__ = [
    "load_params_wide", "get_param_wide",
    "compute_crop_residues_direct", "compute_crop_residues_indirect",
    "compute_burning", "compute_synth_fert_direct", "compute_synth_fert_indirect",
    "compute_rice_ch4", "run_gce",
]
