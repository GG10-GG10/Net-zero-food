
# -*- coding: utf-8 -*-
"""
lme_manure_module_fao_wide_cn.py
================================
【畜禽粪便（Livestock Manure）— 统一从“WIDE 参数表”读取的计算模块 / 中文注释版】

本模块实现 FAOSTAT “Livestock Manure” 域的 Tier 1 计算逻辑，
并将所有系数（N.excretion.rate、各 MMS 份额 MS、系统损失 Frac.loss、
挥发/淋洗分数 Frac.GASM/Frac.LEACH）统一从**同一张宽表（WIDE）**读取，
以便与作物模块/畜禽模块使用体验保持一致。

一、输入数据（WIDE 参数表）
--------------------------
- 文件：CSV 或 XLSX（推荐 CSV 以提升 I/O 性能）；若为 XLSX，默认工作表名 "LIV_parameters_WIDE"
- 列（必须）：
  AreaCode, IPCC_AreaCode, ItemCode, ItemName, Process, ParamName, ParamMMS, ParamCode, Units, Source, 1990..2022
  * 年份列为 4 位数字列名（字符串或数字均可读，内部会自动识别）
  * 该表由 “Parameters_Livestock.xlsx::parameters_AreaCode” 整理生成（见上游脚本）

- 关键 ParamName（大小写不敏感，支持别名）：
  * "N.excretion.rate"    ：单位 kg N head-1 yr-1（头数×该因子 → 粪便总排泄 N）
  * "MS"（ParamMMS=系统名）：分配到各管理途径/系统的份额（含 Pasture / Burned for fuel）
  * "Frac.loss"（ParamMMS=系统名）：该系统内的合计损失分数（NH3/NOx/N2O/N2/渗漏合并）
  * "Frac.GASM"           ：挥发分数（NH3+NOx，总量层面用一个分数）
  * "Frac.LEACH"          ：淋洗/径流分数

- 系统名（ParamMMS）：
  * 特殊路径（不计入“系统损失”环节）："Pasture", "Burned for fuel"
  * 管理系统（参与系统损失）：
    ["Lagoon","Slurry","Solid storage","Drylot","Daily spread",
     "Anaerobic digester","Pit < 1 month","Pit ≥ 1 month","Other"]

二、外部人口学输入（存栏）
--------------------------
- 由调用方提供一张 tidy 表（DataFrame），至少包含：
  AreaCode, year, ItemCode（或 ItemName）, head（存栏数，单位 head）

三、输出
--------
- 返回整洁表（tidy）：列为 ["AreaCode","year","ItemCode","ItemName","element","value_ktN"]
- 单位：**kt N / yr**（内部完成 kg→kt 的换算）
- 要素（element）与 FAOSTAT 对齐：
  1) Amount excreted in manure (N content)
  2) Manure left on pasture (N content)
  3) Manure left on pasture that volatilises (N content)
  4) Manure left on pasture that leaches (N content)
  5) Manure treated (N content)
  6) Losses from manure treated (N content)
  7) Manure applied to soils (N content)
  8) Manure applied to soils that volatilises (N content)
  9) Manure applied to soils that leaches (N content)

四、核心公式（与 FAOSTAT 文档一致）
----------------------------------
- 粪便总排泄： N_total = heads × N.excretion.rate（kg N/年）
- 留牧地：     N_pasture = N_total × [MS_Pasture + 0.5 × MS_BurnedForFuel]
- 留牧地挥发/淋洗： N_pasture × Frac.GASM / Frac.LEACH
- 进入管理系统：    N_treated = N_total × Σ(MMS 份额；不含 Pasture/Burned)
- 系统内损失：     Loss = Σ [N_total × MS_sys × Frac.loss_sys]
- 施用到土壤：     N_applied = max(N_treated − Loss, 0)
- 施用后挥发/淋洗：N_applied × Frac.GASM / Frac.LEACH

五、使用示例
------------
    from lme_manure_module_fao_wide_cn import load_parameters_wide, run_lme_from_wide
    P = load_parameters_wide("LIV_parameters_WIDE_1990_2022.csv")
    populations = ... # DataFrame(AreaCode, year, ItemCode/ItemName, head)
    out = run_lme_from_wide(P, populations, years=[2020])
    print(out.head())

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# 常量：kg → kt 的换算
KG_TO_KT = 1e-6

# 参与“系统损失”的 MMS 列表（IPCC/FAOSTAT 语境常用）
MMS_CANON = [
    "Lagoon", "Slurry", "Solid storage", "Drylot", "Daily spread",
    "Anaerobic digester", "Pit < 1 month", "Pit ≥ 1 month", "Other"
]

# 特殊路径：不参与“系统损失”计算，但在 Pasture 中计入 + 在 Burned 中取 50% 计入 Pasture（尿 N）
SPECIAL_PATHS = ["Pasture", "Burned for fuel"]

# ──────────────────────────────────────────────────────────────────────────────
# 1) 载入宽表（WIDE）
# ──────────────────────────────────────────────────────────────────────────────

def load_parameters_wide(path: str, sheet: str = "LIV_parameters_WIDE") -> pd.DataFrame:
    """
    读取宽表（CSV / XLSX），并标准化字段：
    - 将 ID 字段转为数值 Int64，便于与外部存栏表 join；
    - 保留全部年份列（1990..2022），供按需取值；
    - 额外生成小写匹配字段 _param / _mms / _item / _process。
    """
    # 支持 CSV 与 Excel 两种格式
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)

    # 去除列名空白
    df.columns = [str(c).strip() for c in df.columns]

    # 将 ID 列转为 Int64（支持缺失）
    for c in ["AreaCode","IPCC_AreaCode","ItemCode","ParamCode"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # 添加便于不区分大小写匹配的辅助列
    df["_param"] = df.get("ParamName","").astype(str).str.strip().str.lower()
    df["_mms"]   = df.get("ParamMMS","").astype(str).str.strip()
    df["_item"]  = df.get("ItemName","").astype(str).str.strip()
    df["_process"] = df.get("Process","").astype(str).str.strip()

    # 识别年份列（字符串形式也可）
    years = [int(c) for c in df.columns if c.isdigit()]
    df["_years"] = [years]*len(df)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 2) 参数取值工具（按主键 + 年份取值，支持别名与回溯）
# ──────────────────────────────────────────────────────────────────────────────

# ParamName 别名表（大小写不敏感）
ALIASES = {
    "n.excretion.rate": ["n.excretion.rate", "nex", "n_excretion_rate"],
    "ms": ["ms","share","share.ms","mms.share"],
    "frac.loss": ["frac.loss","loss.frac","loss_fraction"],
    "frac.gasm": ["frac.gasm","frac_nh3_nox","frac.volatilisation"],
    "frac.leach": ["frac.leach","frac_leaching","frac.runoff"],
}

def _match_param(df: pd.DataFrame, pname: str) -> pd.Series:
    """
    判断某一行是否命中指定 ParamName（允许别名）。
    返回一个布尔 Series，表示逐行命中情况。
    """
    key = pname.strip().lower()
    cands = ALIASES.get(key, [key])
    return df["_param"].isin(cands)

def _row_year_value(row: pd.Series, year: int) -> Optional[float]:
    """
    针对一行记录，取指定年份的数值；当年为空时：
    1) 向过去最近的有值年份回溯；
    2) 若仍为空，再向未来最近的有值年份前溯。
    成功返回 float，失败返回 None。
    """
    years = [int(c) for c in row.index if str(c).isdigit()]
    years.sort()
    # 当年
    if str(year) in row.index and pd.notna(row[str(year)]):
        return float(row[str(year)])
    # 向过去回溯
    for y in sorted([y for y in years if y < year], reverse=True):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    # 向未来前溯
    for y in sorted([y for y in years if y > year]):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    return None

def get_param(
    P: pd.DataFrame, *, year: int, areacode: int,
    itemcode: Optional[int] = None, itemname: Optional[str] = None,
    param: str, mms: Optional[str] = None, default: Optional[float] = None
) -> Optional[float]:
    """
    从宽表中检索参数值。
    匹配优先级：
      (1) AreaCode + (ItemCode 或 ItemName) + ParamName(+ParamMMS)
      (2) AreaCode + ParamName(+ParamMMS)（不限定 Item）
      (3) 全局（AreaCode==0）的 ParamName(+ParamMMS)

    参数
    ----
    P         : load_parameters_wide() 的返回 DataFrame
    year      : 年份（int）
    areacode  : 国家/地区代码（FAOSTAT AreaCode）
    itemcode  : 畜种代码（可选）
    itemname  : 畜种名称（可选；若提供 itemcode，可不传）
    param     : ParamName（支持别名；大小写不敏感）
    mms       : 管理系统/路径名（ParamMMS），如 "Slurry"、"Pasture"
    default   : 若未检索到时返回的默认值

    返回
    ----
    float 或 None
    """
    sub = P.copy()

    # 1) 先按 AreaCode 过滤
    sub = sub[(sub["AreaCode"].astype("Int64")==pd.Series([areacode]*len(sub), dtype="Int64"))]

    # 2) Item 过滤（优先 ItemCode，再退回 ItemName）；若两者皆无，则不加限制
    if itemcode is not None and "ItemCode" in sub.columns:
        sub_item = sub[sub["ItemCode"].astype("Int64")==pd.Series([itemcode]*len(sub), dtype="Int64")]
    else:
        sub_item = pd.DataFrame(columns=sub.columns)
    if itemname:
        sub_name = sub[sub["_item"].str.lower()==str(itemname).strip().lower()]
    else:
        sub_name = pd.DataFrame(columns=sub.columns)
    sub_any = pd.concat([sub_item, sub_name]).drop_duplicates() if not sub_item.empty or not sub_name.empty else sub

    # 3) 在候选集中匹配 ParamName（及可选的 ParamMMS）
    m = _match_param(sub_any, param)
    if mms is not None:
        m = m & (sub_any["_mms"].str.lower()==str(mms).strip().lower())
    cand = sub_any[m]
    if not cand.empty:
        r = cand.iloc[0]  # 多行时取第一行（可按 Source/Units 再加规则）
        return _row_year_value(r, year)

    # 4) 回退：仅按 AreaCode + ParamName(+ParamMMS)（不限定 Item）
    m = _match_param(sub, param)
    if mms is not None:
        m = m & (sub["_mms"].str.lower()==str(mms).strip().lower())
    cand = sub[m]
    if not cand.empty:
        r = cand.iloc[0]
        return _row_year_value(r, year)

    # 5) 全局回退：AreaCode==0
    g = P[P["AreaCode"].fillna(0).astype(int)==0]
    if not g.empty:
        m = _match_param(g, param)
        if mms is not None:
            m = m & (g["_mms"].str.lower()==str(mms).strip().lower())
        cand = g[m]
        if not cand.empty:
            r = cand.iloc[0]
            return _row_year_value(r, year)

    return default

def get_share(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], system: str) -> float:
    """读取某管理路径/系统的份额 MS（缺失返回 0）"""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="MS", mms=system, default=0.0)
    return float(v) if v is not None else 0.0

def get_loss_frac(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], system: str) -> float:
    """读取某系统的合计损失分数 Frac.loss（缺失返回 0）"""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="Frac.loss", mms=system, default=0.0)
    return float(v) if v is not None else 0.0

def get_frac_scalar(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], key: str, default: float) -> float:
    """读取全局挥发/淋洗分数（Frac.GASM / Frac.LEACH），支持区域或全局默认值"""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param=key, default=default)
    return float(v) if v is not None else default

# ──────────────────────────────────────────────────────────────────────────────
# 3) 单条记录计算（AreaCode × year × 畜种）
# ──────────────────────────────────────────────────────────────────────────────

def compute_record(
    P: pd.DataFrame, *, year: int, areacode: int,
    itemcode: Optional[int], itemname: Optional[str],
    head: float
) -> List[Dict[str, object]]:
    """
    计算一个“国家×年份×畜种”的全部要素（element）。
    返回一个字典列表（每个元素代表一个 element 的 kt N 值）。
    """
    # ① Nex：kg N / 头 / 年
    Nex = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="N.excretion.rate")
    if Nex is None or not np.isfinite(Nex):
        # 若缺少 Nex，则返回 NaN 以提示上游补充参数
        return [{
            "AreaCode": areacode, "year": year, "ItemCode": itemcode, "ItemName": itemname,
            "element": "Amount excreted in manure (N content)", "value_ktN": np.nan
        }]

    # ② 粪便总排泄（kg N）
    N_total_kg = float(head) * float(Nex)

    # ③ 份额 MS（Pasture / Burned / 各 MMS 系统）
    ms = {sys: get_share(P, year, areacode, itemcode, itemname, sys) for sys in (SPECIAL_PATHS + MMS_CANON)}
    # 纠偏：负数归零；若总和>0，则按总和归一化为 1（保证物理一致性）
    ms = {k: max(0.0, float(v or 0.0)) for k,v in ms.items()}
    ssum = sum(ms.values())
    if ssum > 0:
        ms = {k: v/ssum for k, v in ms.items()}

    # ④ 挥发/淋洗分数（缺失给默认值；建议在 WIDE 表中明确给值）
    FracGASM  = get_frac_scalar(P, year, areacode, itemcode, itemname, "Frac.GASM", default=0.10)
    FracLEACH = get_frac_scalar(P, year, areacode, itemcode, itemname, "Frac.LEACH", default=0.30)

    # ⑤ 留牧地（Pasture）+ 燃料燃烧的一半（尿 N）
    MS_past = ms.get("Pasture", 0.0)
    MS_burn = ms.get("Burned for fuel", 0.0)
    N_pasture_kg = N_total_kg * (MS_past + 0.5*MS_burn)
    N_past_vol_kg   = N_pasture_kg * FracGASM
    N_past_leach_kg = N_pasture_kg * FracLEACH

    # ⑥ 进入管理系统（不含 Pasture/Burned）
    MS_treated = sum(v for k,v in ms.items() if k not in SPECIAL_PATHS)
    N_treated_kg = N_total_kg * MS_treated

    # ⑦ 系统内损失（逐系统：N_total × MS_sys × Frac.loss_sys）
    loss_kg = 0.0
    for sys in MMS_CANON:
        share = ms.get(sys, 0.0)
        if share <= 0: 
            continue
        L = get_loss_frac(P, year, areacode, itemcode, itemname, sys)
        loss_kg += N_total_kg * share * float(L or 0.0)

    # ⑧ 施用到土壤（损失后剩余全部视为施用）
    N_applied_kg = max(N_treated_kg - loss_kg, 0.0)
    N_appl_vol_kg   = N_applied_kg * FracGASM
    N_appl_leach_kg = N_applied_kg * FracLEACH

    # 工具：输出统一为 kt N
    def row(elem, kg): 
        return {
            "AreaCode": areacode, "year": year,
            "ItemCode": itemcode, "ItemName": itemname,
            "element": elem, "value_ktN": float(kg) * KG_TO_KT
        }

    out = [
        row("Amount excreted in manure (N content)", N_total_kg),
        row("Manure left on pasture (N content)", N_pasture_kg),
        row("Manure left on pasture that volatilises (N content)", N_past_vol_kg),
        row("Manure left on pasture that leaches (N content)", N_past_leach_kg),
        row("Manure treated (N content)", N_treated_kg),
        row("Losses from manure treated (N content)", loss_kg),
        row("Manure applied to soils (N content)", N_applied_kg),
        row("Manure applied to soils that volatilises (N content)", N_appl_vol_kg),
        row("Manure applied to soils that leaches (N content)", N_appl_leach_kg),
    ]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 4) 统一编排：按国家×年份×畜种 批量计算
# ──────────────────────────────────────────────────────────────────────────────

def run_lme_from_wide(
    params_wide: pd.DataFrame,
    populations: pd.DataFrame,
    years: Optional[Iterable[int]] = None,
    itemcode_col: str = "ItemCode",
    itemname_col: str = "ItemName",
    head_col: str = "head"
) -> pd.DataFrame:
    """
    批量计算畜禽粪便 N 流量（全部系数从 WIDE 表读取）。

    参数
    ----
    params_wide : 由 load_parameters_wide() 读入后的 DataFrame
    populations : 由调用方提供的存栏表，至少含字段：
                  AreaCode, year, ItemCode（或 ItemName）, head
    years       : 需要计算的年份列表（默认使用 populations 中出现的所有年份）
    itemcode_col: 畜种代码列名（默认 "ItemCode"）
    itemname_col: 畜种名称列名（默认 "ItemName"）
    head_col    : 存栏列名（默认 "head"）

    返回
    ----
    整洁表（tidy）：AreaCode, year, ItemCode, ItemName, element, value_ktN
    """
    P = params_wide.copy()

    # 标准化存栏表字段
    pop = populations.copy()
    pop.columns = [c.strip() for c in pop.columns]
    pop["AreaCode"] = pd.to_numeric(pop["AreaCode"], errors="coerce").astype("Int64")
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce").astype(int)
    pop[head_col] = pd.to_numeric(pop[head_col], errors="coerce").fillna(0.0)

    # 年份范围
    if years is None:
        years = sorted(pop["year"].unique().tolist())
    years = [int(y) for y in years]

    rows = []
    # 逐国家×年份×畜种聚合后计算（同一畜种多行 head 会相加）
    gcols = ["AreaCode","year", itemcode_col if itemcode_col in pop.columns else "ItemCode", itemname_col if itemname_col in pop.columns else "ItemName"]
    for (ac, y, ic, iname), g in pop.groupby(gcols, dropna=False):
        head = float(g[head_col].sum())
        recs = compute_record(
            P, year=int(y),
            areacode=int(ac) if pd.notna(ac) else 0,
            itemcode=int(ic) if pd.notna(ic) else None,
            itemname=str(iname) if pd.notna(iname) else None,
            head=head
        )
        rows.extend(recs)

    out = pd.DataFrame(rows)
    out["value_ktN"] = out["value_ktN"].astype(float)
    return out

__all__ = ["load_parameters_wide","get_param","run_lme_from_wide","compute_record","MMS_CANON","SPECIAL_PATHS"]
