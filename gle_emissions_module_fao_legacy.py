
# -*- coding: utf-8 -*-
"""
gle_emissions_module_fao_legacy.py
==================================

**【重要说明】本模块已被弃用，保留作为参考**
----------------------------------------------
- 本文件原名为 gle_emissions_module_fao.py
- 在主程序 S4_0_main.py 中从未被真正调用（仅作为占位符）
- 现在使用新模块 gle_emissions_complete.py 进行畜牧业排放计算
- 保留本文件是为了：
  1. 作为历史参考和文档
  2. 其MC采样和回归功能可能在将来被借鉴
  3. 便于对比新旧实现方法

如需使用畜牧业排放计算，请使用: gle_emissions_complete.py
更新日期: 2025-11-10

原模块定位
--------
本模块是 **畜牧业温室气体排放计算引擎**，风格与作物模块一致，供 `main` 直接调用。
它读取你维护的 **WIDE 参数表**（FAOSTAT 过程名），并依据活动数据计算以下过程排放：
- 肠道发酵（Enteric fermentation）→ CH₄
- 牧场粪尿沉积（Manure left on pastures, PRP）→ 直接/间接 N₂O
- 粪污管理（Manure management, MM）→ CH₄、直接/间接 N₂O
- 粪肥还田（Manure applied to soils, MAS）→ 直接/间接 N₂O

在此基础上，模块扩展了 **Monte Carlo（MC）不确定性链路**：
- 以 **产率（yield）/胴体率（carcass率）** 作为 MC 抽样的**变换因子**；
- 由预测的 **meat/dairy 产量** 推导 **动物流量**（肉类：屠宰头数；奶类：泌乳头数）；
- 使用从 FAO 历史数据拟合出的 **“屠宰头数 → 存栏（stock）”回归关系**，将动物流量转换为存栏；
- 再用存栏计算各过程排放；
- 支持对 MC 结果进行分位数汇总。

核心优点
--------
1) **参数完全外置**（WIDE）：process/parameter/ParamMMS/年份横向展开，国家/品类/全球层级回退；
2) **活动数据最少依赖**：未来只有产量也能算（先由 yield/carcass 推导屠宰，再由回归推存栏）。
3) **MC 就绪**：replicate 作为样本维度贯穿全链路，并可一键汇总 P5/P50/P95。

使用总览
--------
1) 用 FAO 历史长表（含 Stocks/Slaughtered 指标）整理训练集 → `fit_stock_from_slaughter()` 拟合“屠宰→存栏”；
2) 预测期给出产量（meat/dairy），+ MC 抽样（产率/胴体率）→ `run_gle_mc()` 一键得到排放（含 replicate）；
3) 用 `summarize_mc_results()` 按国家/品类/过程聚合求分位。

约定与单位
----------
- `headcount`：年平均存栏数（头）
- `slaughtered_animals`：年度屠宰头数（头）
- `production_t`：产量（吨/年），肉类为胴体重，奶类为牛奶产量
- `yield_t_per_head`：肉类的单位“可食产量”产率（吨/头·年）；
- `carcass_rate`：胴体率（0~1），将“动物体产量”转换为胴体重；
- `milk_yield_t_per_head`：奶牛单位产奶量（吨/头·年）
- 输出排放单位：`CH4_kt`/`N2O_kt`（千吨）

依赖
----
- pandas、numpy

"""  # 以上为模块说明

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Iterable, Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# FAOSTAT 过程名常量（严格使用官方名称，便于参数表一致）
# ──────────────────────────────────────────────────────────────────────────────
FAO_PROC_ENTERIC = "Enteric fermentation"
FAO_PROC_PRP     = "Manure left on pastures"
FAO_PROC_MM      = "Manure management"
FAO_PROC_MAS     = "Manure applied to soils"

# ──────────────────────────────────────────────────────────────────────────────
# 参数别名映射（不同资料里同一参数可能存在不同字段名，这里统一兼容）
# 键为“标准名”，值为“候选名列表”（按顺序尝试，命中即用）
# ──────────────────────────────────────────────────────────────────────────────
PARAM_ALIASES: Dict[str, List[str]] = {
    # CH4 排放因子（kg CH4/头·年）
    "Methane_emissions_for_animal": [
        "Methane_emissions_for_animal", "EF_CH4_kg_head_yr", "EF_CH4_per_head"
    ],
    # 年氮排泄（kg N/头·年，Nex）
    "N.excretion.rate": [
        "N.excretion.rate", "N_excretion_rate", "Nex"
    ],
    # N2O 直接排放因子（kg N2O-N / kg N），不同场景下记为 EF1/EF3
    "N2O.emis.factor": [
        "N2O.emis.factor", "EF1_N2O_N_per_N", "EF3_N2O_N_per_N", "EF3", "EF1"
    ],
    # 挥发相关分数与因子（间接 N2O）
    "FracGASM_volatilization": [
        "FracGASM_volatilization", "FracGASF_volatilization", "FracGASM", "FracGASF"
    ],
    "EF4_N2O_N_per_Nvolatilized": [
        "EF4_N2O_N_per_Nvolatilized", "EF4"
    ],
    # 淋洗相关分数与因子（间接 N2O）
    "FracLEACH": [
        "FracLEACH", "FracLEACH_N"
    ],
    "EF5_N2O_N_per_Nleached": [
        "EF5_N2O_N_per_Nleached", "EF5"
    ],
    # MMS 份额（管理系统分配 share），ParamMMS 指定子系统，如 pasture/slurry/solid/…
    "MS": [
        "MS", "MMS_share", "MMS"
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 参数表读取与取值（WIDE 结构）
# ──────────────────────────────────────────────────────────────────────────────
def load_params_wide(path: str) -> pd.DataFrame:
    """
    读取 WIDE 参数簿（CSV / XLSX 均可）。
    期望列：country, iso3, commodity, process, parameter, ParamMMS, units, source, notes, 1961..
    - 年份列为字符串列名（"1961"~），函数内会尝试转换为数值型，便于后续插值/回退。
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=0, dtype=str)

    # 将“看起来像年份”的列统一转为数值（便于做数值判断/就近回填）
    for c in df.columns:
        if isinstance(c, str) and c.isdigit():
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 确保关键列存在并为字符串
    for k in ["country","iso3","commodity","process","parameter","ParamMMS","units","source","notes"]:
        if k not in df.columns:
            df[k] = ""
        else:
            df[k] = df[k].fillna("").astype(str)
    return df


def _year_value(row: pd.Series, year: int) -> Optional[float]:
    """
    从某一行的年度列中取值：
    - 优先取目标年；若为空，则向“过去”回溯最近非空；仍空，再向“未来”寻找最近非空。
    - 全为空则返回 None。
    """
    ycols = [c for c in row.index if isinstance(c, str) and c.isdigit()]
    # 目标年
    if str(year) in row.index and pd.notna(row[str(year)]):
        return float(row[str(year)])
    # 向过去回溯
    for y in sorted([int(c) for c in ycols if int(c) < year], reverse=True):
        v = row[str(y)]
        if pd.notna(v):
            return float(v)
    # 向未来回溯
    for y in sorted([int(c) for c in ycols if int(c) > year]):
        v = row[str(y)]
        if pd.notna(v):
            return float(v)
    return None


def _expand_param_names(parameter: str) -> List[str]:
    """将请求的参数名展开为“标准名+别名们”的候选序列（按顺序尝试）。"""
    p = parameter.strip()
    if p in PARAM_ALIASES:
        return list(dict.fromkeys(PARAM_ALIASES[p]))  # 去重保序
    return [p]


def get_param_wide(
    P: pd.DataFrame,
    *, country: str, iso3: str, commodity: str,
    process: str, parameter: str, year: int,
    ParamMMS: Optional[str] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    从 WIDE 参数簿检索单个“标量参数”值。
    匹配维度：country/iso3、commodity、process、parameter（含别名）、ParamMMS（可选）、year（横向）。

    匹配优先级（从高到低）：
      1) 本国/ISO3 + 指定 commodity + process + parameter (+ ParamMMS)
      2) 本国/ISO3 + ALL           + process + parameter (+ ParamMMS)
      3) GLOBAL/GLB/World          + (commodity 或 ALL) + process + parameter (+ ParamMMS)

    返回 float 或 default（均可能为 None）。
    """
    country = (country or "").strip()
    iso3 = (iso3 or "").strip()
    commodity = (commodity or "").strip()
    process = (process or "").strip()
    parameter = (parameter or "").strip()
    mms = (ParamMMS or "").strip()

    param_names = _expand_param_names(parameter)

    def _fetch(df: pd.DataFrame, cc: List[str], it: List[str]) -> Optional[float]:
        sub = df[(df["process"] == process) & (df["commodity"].isin(it))]
        sub = sub[sub["parameter"].isin(param_names)]
        # MMS 子系统过滤（如 pasture/slurry/solid/...）
        if mms:
            if "ParamMMS" in sub.columns:
                sub = sub[sub["ParamMMS"].str.lower() == mms.lower()]
            else:
                return None
        # 国家/ISO3 匹配
        sub = sub[sub["country"].isin(cc) | sub["iso3"].isin(cc)]
        if sub.empty:
            return None
        # 若多行命中，取第一行（一般唯一）
        row = sub.iloc[0]
        return _year_value(row, year)

    # 1) 本国/ISO + commodity
    cc = [x for x in [iso3, country] if x]
    it = [commodity]
    v = _fetch(P, cc, it)
    if v is not None:
        return v

    # 2) 本国/ISO + ALL
    it = ["ALL"]
    v = _fetch(P, cc, it)
    if v is not None:
        return v

    # 3) GLOBAL 回退
    cc = ["GLOBAL", "GLB", "World", "GLOBAL/GLB"]
    it = [commodity, "ALL"]
    v = _fetch(P, cc, it)
    if v is not None:
        return v

    return default


# ──────────────────────────────────────────────────────────────────────────────
# FAO 历史数据 → 训练集（“屠宰 vs 存栏”）
# ──────────────────────────────────────────────────────────────────────────────
def prepare_regression_data_from_fao(
    fao_df: pd.DataFrame,
    *, indicator_col: str = "item",
    value_col: str = "value",
    slaughter_labels: List[str] = None,
    stock_labels: List[str] = None
) -> pd.DataFrame:
    """
    将 FAOSTAT 畜牧**长表**（同一 DataFrame 内既有 Stocks 也有 Slaughtered）整理为**宽表**训练集：
    输出列：country, iso3, commodity, year, slaughtered_animals, stock_headcount

    参数
    ----
    indicator_col : 区分“指标种类”的列名（例如 'item' 或 'element' 等）
    value_col     : 数值列名
    slaughter_labels : 识别“屠宰”指标的标签列表（默认含常见英文/中文）
    stock_labels     : 识别“存栏”指标的标签列表（默认含常见英文/中文）
    """
    # 默认的标签集合（如你的 FAO 导出用其他文案，请传参覆盖）
    s_labels = slaughter_labels or ["Slaughtered", "屠宰头数", "Animals slaughtered"]
    k_labels = stock_labels or ["Stocks", "存栏", "Stocks - Headcount"]

    need_cols = ["country","iso3","commodity","year",indicator_col,value_col]
    for c in need_cols:
        if c not in fao_df.columns:
            raise ValueError(f"FAO 数据缺列 {c}")

    d = fao_df.copy()
    # 仅保留“屠宰/存栏”两类指标的记录
    d = d[d[indicator_col].isin(s_labels + k_labels)].copy()

    # 映射成两个类别：slaughtered_animals / stock_headcount
    d["__ind__"] = np.where(d[indicator_col].isin(s_labels), "slaughtered_animals", "stock_headcount")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    # 透视成宽表（每个键一行，包含两个指标列）
    wide = (d
        .pivot_table(index=["country","iso3","commodity","year"],
                     columns="__ind__", values=value_col, aggfunc="sum")
        .reset_index())

    # 兜底：若某列缺失则创建空列
    if "slaughtered_animals" not in wide.columns:
        wide["slaughtered_animals"] = np.nan
    if "stock_headcount" not in wide.columns:
        wide["stock_headcount"] = np.nan

    return wide


# ──────────────────────────────────────────────────────────────────────────────
# 拟合：屠宰 → 存栏（国家×品类回归 + 品类全局回归）
# ──────────────────────────────────────────────────────────────────────────────
def fit_stock_from_slaughter(
    train_df: pd.DataFrame,
    intercept: bool = True,
    min_years: int = 3,
) -> Dict[str, Any]:
    """
    以“屠宰头数”为自变量、以“年平均存栏”为因变量的线性回归：
        stock_headcount ≈ a + b × slaughtered_animals
    - 优先在 **国家×品类** 维度上拟合；样本不足则保存 **stock/slaughter** 的**中位数比值**作为兜底；
    - 同时拟合 **品类全局** 回归用于回退。

    要求列
    ------
    country, iso3, commodity, year, slaughtered_animals, stock_headcount

    返回
    ----
    models: dict，包含
      - "by_key": {(iso3_or_country, commodity): {"a","b","ratio","n","r2"}}
      - "by_commodity": {commodity: {"a","b","ratio","n","r2"}}
    """
    df = train_df.copy()
    for c in ["slaughtered_animals","stock_headcount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["slaughtered_animals","stock_headcount"])
    df = df[(df["slaughtered_animals"] >= 0) & (df["stock_headcount"] >= 0)].copy()

    by_key: Dict[Tuple[str,str], Dict[str, float]] = {}
    for (country, iso3, commodity), sub in df.groupby(["country","iso3","commodity"], dropna=False):
        key = ((iso3 or country or "").strip(), str(commodity))
        s = sub.dropna(subset=["slaughtered_animals","stock_headcount"])
        if len(s) == 0:
            continue
        x = s["slaughtered_animals"].astype(float).values
        y = s["stock_headcount"].astype(float).values

        # 比值中位数（stock/slaughter），在样本不足或回归不稳时兜底使用
        ratio = float(np.median(y / np.maximum(x, 1e-9)))

        # 默认：充当初值
        a = 0.0
        b = ratio
        r2 = np.nan

        # 样本达到阈值，执行最小二乘回归
        if len(s) >= max(3, min_years) and np.nanmax(x) > 0:
            if intercept:
                # 带截距：y = a + b x
                X = np.vstack([np.ones_like(x), x]).T
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # 解 [a, b]
                a, b = float(beta[0]), float(beta[1])
                yhat = a + b * x
            else:
                # 过原点：y = b x
                b = float(np.sum(x*y) / (np.sum(x*x) or 1e-9))
                a = 0.0
                yhat = b * x
            # R² 评估
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2)) or 1e-9
            r2 = 1.0 - ss_res/ss_tot

        by_key[key] = {"a": a, "b": b, "ratio": ratio, "n": int(len(s)), "r2": r2}

    # 品类全局回归（作为回退层）
    by_com: Dict[str, Dict[str, float]] = {}
    for commodity, s in df.groupby("commodity", dropna=False):
        x = s["slaughtered_animals"].astype(float).values
        y = s["stock_headcount"].astype(float).values
        ratio = float(np.median(y / np.maximum(x, 1e-9)))
        a = 0.0
        b = ratio
        r2 = np.nan
        if len(s) >= max(10, 3*min_years) and np.nanmax(x) > 0:
            X = np.vstack([np.ones_like(x), x]).T
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = float(beta[0]), float(beta[1])
            yhat = a + b*x
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2)) or 1e-9
            r2 = 1.0 - ss_res/ss_tot
        by_com[str(commodity)] = {"a": a, "b": b, "ratio": ratio, "n": int(len(s)), "r2": r2}

    return {"by_key": by_key, "by_commodity": by_com}


# ──────────────────────────────────────────────────────────────────────────────
# MC：产量 +（产率/胴体率）→ 动物流量（屠宰或泌乳）
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_prod(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保存在 `production_t` 列：
    - 若已有则直接返回；
    - 否则尝试从 `meat_production_t` 或 `milk_production_t` 构造；
    - 都没有则抛错（提示调用者补足列）。
    """
    if "production_t" in df.columns:
        return df
    d = df.copy()
    if "meat_production_t" in d.columns:
        d["production_t"] = d["meat_production_t"]
    elif "milk_production_t" in d.columns:
        d["production_t"] = d["milk_production_t"]
    else:
        raise ValueError("需要列 'production_t'，或 'meat_production_t' / 'milk_production_t' 之一。")
    return d


def mc_animals_from_production(
    prod_df: pd.DataFrame,
    mc_draws: pd.DataFrame,
    *, product_type_col: str = "product_type",
    meat_label: str = "meat",
    dairy_label: str = "dairy",
) -> pd.DataFrame:
    """
    基于“预测产量 + MC 抽样的产率/胴体率”计算“动物流量”，并按照 `replicate` 展开。

    入参要求
    --------
    prod_df：至少含键列 `country, iso3, commodity, year` 与产量列：
      - `production_t`（或 `meat_production_t` / `milk_production_t` 其一）
      - `product_type`（可选，取 {meat, dairy}；若缺失会按 commodity 字符串包含 'milk'/'dairy' 自动推断）
    mc_draws：至少含键列 `country, iso3, commodity, year, replicate`，并包含：
      - 肉类：`yield_t_per_head`（吨/头·年）、`carcass_rate`（0~1）
      - 奶类：`milk_yield_t_per_head`（吨/头·年）

    计算规则
    --------
    肉类（product_type==meat）：
        屠宰头数 = 产量吨 / (单位产率 × 胴体率) = production_t / (yield_t_per_head × carcass_rate)
    奶类（product_type==dairy）：
        泌乳头数 = 产量吨 / 单位产奶量 = production_t / milk_yield_t_per_head

    返回
    ----
    一个按 replicate 展开的 DataFrame，新增列：`replicate`, `animals_flow`（肉类=屠宰；奶类=泌乳）。
    """
    P = _ensure_prod(prod_df).copy()
    P[product_type_col] = P.get(product_type_col, "").fillna("")

    # 若缺 product_type，则基于 commodity 文案推断（含 'milk'/'dairy' → dairy；其余 → meat）
    def infer_t(commodity: str, cur: str) -> str:
        if cur:
            return cur
        s = str(commodity).lower()
        if "milk" in s or "dairy" in s:
            return dairy_label
        return meat_label
    P[product_type_col] = [infer_t(c, t) for c, t in zip(P["commodity"], P[product_type_col])]

    # 合并 MC 抽样（按 country/iso3/commodity/year 连接，抽样维度 replicate 来自 mc_draws）
    keys = ["country","iso3","commodity","year"]
    need_cols = keys + ["production_t", product_type_col]
    for c in need_cols:
        if c not in P.columns:
            raise ValueError(f"prod_df 缺列 {c}")
    for c in keys + ["replicate"]:
        if c not in mc_draws.columns:
            raise ValueError(f"mc_draws 缺列 {c}")

    merged = P.merge(mc_draws, on=keys, how="left", suffixes=("",""))
    if merged["replicate"].isna().any():
        missing = merged[merged["replicate"].isna()][keys].drop_duplicates()
        if len(missing) > 0:
            raise ValueError("存在生产记录未匹配到 MC 抽样（按 country/iso3/commodity/year）。请检查 mc_draws。")

    # 逐行计算 animals_flow（根据 product_type 采用不同公式）
    def calc_row(row: pd.Series) -> float:
        pt = row[product_type_col]
        prod = float(row["production_t"] or 0.0)
        if pt == meat_label:
            y = float(row.get("yield_t_per_head", np.nan))     # 吨/头·年
            c = float(row.get("carcass_rate", np.nan))         # 0~1
            if np.isnan(y) or np.isnan(c) or c <= 0 or y <= 0:
                return np.nan
            return prod / (y * c)
        else:  # 奶业
            m = float(row.get("milk_yield_t_per_head", np.nan))
            if np.isnan(m) or m <= 0:
                return np.nan
            return prod / m

    merged["animals_flow"] = merged.apply(calc_row, axis=1)

    # 若出现 NaN，说明某些抽样记录缺关键参数（yield/carcass 或 milk_yield）
    if merged["animals_flow"].isna().any():
        raise ValueError("存在无法计算 animals_flow 的记录（检查 MC 抽样列：yield_t_per_head/carcass_rate 或 milk_yield_t_per_head）。")

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# MC：动物流量 → 存栏（通过“屠宰→存栏”回归模型）
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict_stock(
    flow_df: pd.DataFrame,
    models: Dict[str, Any],
    *, link: str = "slaughter",
    clamp_nonneg: bool = True,
) -> pd.DataFrame:
    """
    使用 `fit_stock_from_slaughter()` 的模型，将 MC 生成的“动物流量”映射为“存栏 headcount”。

    参数
    ----
    flow_df : `mc_animals_from_production()` 的输出，至少含：
        country, iso3, commodity, year, replicate, animals_flow
    models  : `fit_stock_from_slaughter()` 的返回字典（含 by_key 与 by_commodity）
    link    : 暂仅支持 'slaughter'（屠宰→存栏）。若传 'milking'，需你另行训练对应模型。
    clamp_nonneg : 是否将负值钳制为 0（默认为 True）

    返回
    ----
    在原表基础上新增：
        - headcount_est：预测的存栏
        - headcount    ：若原本不存在该列，会回填为 headcount_est
    """
    out = flow_df.copy()
    by_key = models.get("by_key", {})
    by_com = models.get("by_commodity", {})

    for i, r in out.iterrows():
        # 键优先顺序：ISO3（若空则用 country） + commodity
        key = ((r.get("iso3") or "").strip() or (r.get("country") or "").strip(), str(r.get("commodity")))
        x = float(r.get("animals_flow") or 0.0)  # 自变量

        a = b = ratio = None
        if key in by_key:
            m = by_key[key]; a, b, ratio = m["a"], m["b"], m["ratio"]
        elif str(r.get("commodity")) in by_com:
            m = by_com[str(r.get("commodity"))]; a, b, ratio = m["a"], m["b"], m["ratio"]

        # 估计逻辑：优先回归（a+b*x），否则用比值（ratio*x），再否则置 0
        if a is not None and b is not None:
            est = a + b * x
        elif ratio is not None:
            est = ratio * x
        else:
            est = 0.0

        if clamp_nonneg and est < 0:
            est = 0.0

        out.at[i, "headcount_est"] = est
        if "headcount" not in out.columns or pd.isna(out.at[i, "headcount"]):
            out.at[i, "headcount"] = est

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 各过程排放计算（非 MC）——与前版一致，但中文注释更细
# ──────────────────────────────────────────────────────────────────────────────
def compute_enteric_ch4(P: pd.DataFrame, df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    肠道发酵 CH₄（Enteric fermentation）

    输入 df 至少包含：country, iso3, year, commodity, headcount
    参数获取：process=Enteric fermentation，parameter ∈ {"Methane_emissions_for_animal","EF_CH4_kg_head_yr"}
    计算：CH4_kt = headcount × EF_CH4(kg/头·年) / 1e6
    """
    out = df.copy()
    if "headcount" not in out.columns:
        raise ValueError("enteric_df 需要列 headcount。")

    out["CH4_kt"] = 0.0
    for i, r in out.iterrows():
        ef = get_param_wide(
            P, country=r["country"], iso3=r["iso3"], commodity=r["commodity"],
            process=FAO_PROC_ENTERIC, parameter="Methane_emissions_for_animal",
            year=year, default=0.0
        ) or 0.0
        head = float(r.get("headcount", 0) or 0)
        out.at[i, "CH4_kt"] = head * ef / 1e6  # kg → kt

    out["process"] = FAO_PROC_ENTERIC
    out["gas"] = "CH4"
    return out


def _nex(P: pd.DataFrame, r: pd.Series, year: int, process: str) -> float:
    """读取年氮排泄 Nex（kg N/头·年），缺失返回 0。"""
    v = get_param_wide(
        P, country=r["country"], iso3=r["iso3"], commodity=r["commodity"],
        process=process, parameter="N.excretion.rate", year=year, default=None
    )
    return float(v) if v is not None else 0.0


def _mms_share(P: pd.DataFrame, r: pd.Series, year: int, mms: str) -> float:
    """读取 MMS 子系统份额（0~1），process=Manure management, parameter=MS, ParamMMS=<mms>。"""
    v = get_param_wide(
        P, country=r["country"], iso3=r["iso3"], commodity=r["commodity"],
        process=FAO_PROC_MM, parameter="MS", ParamMMS=mms, year=year, default=0.0
    )
    return float(v or 0.0)


def compute_prp_n2o(P: pd.DataFrame, df: pd.DataFrame, year: int, include_indirect: bool=True) -> Dict[str, pd.DataFrame]:
    """
    牧场粪尿沉积（PRP）N₂O：直接 + 间接（可选）

    需要 df 列：country, iso3, year, commodity, headcount
    关键参数：
      - Nex：process=PRP, parameter="N.excretion.rate"
      - 牧场份额：process=MM, parameter="MS", ParamMMS="pasture"
      - 直接 EF：process=PRP, parameter="N2O.emis.factor"
      - 间接：FracGASM/EF4、FracLEACH/EF5（process=PRP）
    """
    res_dir = df.copy(); res_dir["N2O_kt"] = 0.0
    res_ind = df.copy(); res_ind["N2O_kt"] = 0.0

    if "headcount" not in df.columns:
        raise ValueError("prp_df 需要列 headcount。")

    for i, r in df.iterrows():
        # 沉积到牧场的氮量（kg N/年）= 存栏 × 年排泄 × pasture 份额
        nex = _nex(P, r, year, FAO_PROC_PRP)
        share_pasture = _mms_share(P, r, year, "pasture")
        N_to_PRP = float(r.get("headcount", 0) or 0) * nex * share_pasture

        # 直接 N2O：EF3PRP（kg N2O-N/kg N）
        ef3 = get_param_wide(
            P, country=r["country"], iso3=r["iso3"], commodity=r["commodity"],
            process=FAO_PROC_PRP, parameter="N2O.emis.factor", year=year, default=0.0
        ) or 0.0
        N2O_N_dir = N_to_PRP * ef3
        res_dir.at[i, "N2O_kt"] = N2O_N_dir * (44.0/28.0) / 1e6  # N2O-N → N2O → kt

        # 间接 N2O：挥发与淋洗路径（可选）
        if include_indirect:
            frac_gasm = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_PRP, "FracGASM_volatilization", year, default=0.0) or 0.0
            ef4 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_PRP, "EF4_N2O_N_per_Nvolatilized", year, default=0.0) or 0.0
            frac_leach = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_PRP, "FracLEACH", year, default=0.0) or 0.0
            ef5 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_PRP, "EF5_N2O_N_per_Nleached", year, default=0.0) or 0.0
            N2O_N_ind = N_to_PRP*frac_gasm*ef4 + N_to_PRP*frac_leach*ef5
            res_ind.at[i, "N2O_kt"] = N2O_N_ind * (44.0/28.0) / 1e6

    res_dir["process"] = FAO_PROC_PRP; res_dir["gas"] = "N2O"
    if include_indirect:
        res_ind["process"] = FAO_PROC_PRP; res_ind["gas"] = "N2O"
        return {"prp_direct": res_dir, "prp_indirect": res_ind}
    else:
        return {"prp_direct": res_dir}


def compute_mm_ch4(P: pd.DataFrame, df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    粪污管理（MM）CH₄：
    - 若参数表提供 process=MM 下的 CH₄ EF（kg/头·年），则计算；否则结果为 0。
    """
    out = df.copy(); out["CH4_kt"] = 0.0
    if "headcount" not in out.columns:
        raise ValueError("mm_df 需要列 headcount。")

    for i, r in out.iterrows():
        ef = get_param_wide(
            P, country=r["country"], iso3=r["iso3"], commodity=r["commodity"],
            process=FAO_PROC_MM, parameter="Methane_emissions_for_animal",
            year=year, default=0.0
        ) or 0.0
        head = float(r.get("headcount", 0) or 0)
        out.at[i, "CH4_kt"] = head * ef / 1e6

    out["process"] = FAO_PROC_MM; out["gas"] = "CH4"
    return out


def compute_mm_n2o(P: pd.DataFrame, df: pd.DataFrame, year: int, include_indirect: bool=True) -> Dict[str, pd.DataFrame]:
    """
    粪污管理（MM）N₂O：直接 + 间接（可选）
    - 进入管理系统的氮量 = 存栏 × Nex × (1 - pasture 份额)
    - 直接：EF3MM；间接：FracGASM/EF4 与 FracLEACH/EF5
    """
    res_dir = df.copy(); res_dir["N2O_kt"] = 0.0
    res_ind = df.copy(); res_ind["N2O_kt"] = 0.0
    if "headcount" not in df.columns:
        raise ValueError("mm_df 需要列 headcount。")

    for i, r in df.iterrows():
        nex = _nex(P, r, year, FAO_PROC_MM)
        share_pasture = _mms_share(P, r, year, "pasture")
        N_to_MM = float(r.get("headcount", 0) or 0) * nex * (1.0 - share_pasture)

        ef3mm = get_param_wide(
            P, r["country"], r["iso3"], r["commodity"],
            FAO_PROC_MM, "N2O.emis.factor", year, default=0.0
        ) or 0.0
        N2O_N_dir = N_to_MM * ef3mm
        res_dir.at[i, "N2O_kt"] = N2O_N_dir * (44.0/28.0) / 1e6

        if include_indirect:
            frac_gasm = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MM, "FracGASM_volatilization", year, default=0.0) or 0.0
            ef4 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MM, "EF4_N2O_N_per_Nvolatilized", year, default=0.0) or 0.0
            frac_leach = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MM, "FracLEACH", year, default=0.0) or 0.0
            ef5 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MM, "EF5_N2O_N_per_Nleached", year, default=0.0) or 0.0
            N2O_N_ind = N_to_MM*frac_gasm*ef4 + N_to_MM*frac_leach*ef5
            res_ind.at[i, "N2O_kt"] = N2O_N_ind * (44.0/28.0) / 1e6

    res_dir["process"] = FAO_PROC_MM; res_dir["gas"] = "N2O"
    if include_indirect:
        res_ind["process"] = FAO_PROC_MM; res_ind["gas"] = "N2O"
        return {"mm_direct": res_dir, "mm_indirect": res_ind}
    else:
        return {"mm_direct": res_dir}


def compute_mas_n2o(P: pd.DataFrame, df: pd.DataFrame, year: int, include_indirect: bool=True) -> Dict[str, pd.DataFrame]:
    """
    粪肥还田（MAS）N₂O：直接 + 间接（可选）
    - 若已给出 `N_applied_to_soils_kgN`，直接使用；
    - 否则：从 Nex 与 MMS 份额推导“进入土壤的氮”（估算进入土壤的 MMS 份额缺省为 0.5）。
    """
    res_dir = df.copy(); res_dir["N2O_kt"] = 0.0
    res_ind = df.copy(); res_ind["N2O_kt"] = 0.0

    for i, r in df.iterrows():
        N_applied = r.get("N_applied_to_soils_kgN", None)

        if (pd.isna(N_applied) or N_applied is None):
            # 没有直接给 N_applied，就需要依赖存栏和 MMS 份额推导；因此必须有 headcount
            if "headcount" not in df.columns:
                raise ValueError("mas_df 缺少 headcount 且未提供 N_applied_to_soils_kgN。")

            nex = _nex(P, r, year, FAO_PROC_MAS)
            share_pasture = _mms_share(P, r, year, "pasture")
            N_to_MM = float(r.get("headcount", 0) or 0) * nex * (1.0 - share_pasture)

            # 估算“进入土壤”的 MMS 份额：尝试若干标签（daily_spread/solid/compost/deep_litter/drylot）；
            # 若都没有，采用保守中值 0.5。
            share_soil = 0.0
            for tag in ["daily_spread","solid","compost","deep_litter","drylot"]:
                share_soil += _mms_share(P, r, year, tag)
            if share_soil == 0.0:
                share_soil = 0.5

            N_applied = N_to_MM * share_soil  # kg N/年

        # 直接 N2O：EF1（或统一名 N2O.emis.factor）
        ef1 = get_param_wide(
            P, r["country"], r["iso3"], r["commodity"],
            FAO_PROC_MAS, "N2O.emis.factor", year, default=0.0
        ) or 0.0
        N2O_N_dir = float(N_applied) * ef1
        res_dir.at[i, "N2O_kt"] = N2O_N_dir * (44.0/28.0) / 1e6

        if include_indirect:
            frac_gasm = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MAS, "FracGASM_volatilization", year, default=0.0) or 0.0
            ef4 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MAS, "EF4_N2O_N_per_Nvolatilized", year, default=0.0) or 0.0
            frac_leach = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MAS, "FracLEACH", year, default=0.0) or 0.0
            ef5 = get_param_wide(P, r["country"], r["iso3"], r["commodity"], FAO_PROC_MAS, "EF5_N2O_N_per_Nleached", year, default=0.0) or 0.0
            N2O_N_ind = float(N_applied)*frac_gasm*ef4 + float(N_applied)*frac_leach*ef5
            res_ind.at[i, "N2O_kt"] = N2O_N_ind * (44.0/28.0) / 1e6

    res_dir["process"] = FAO_PROC_MAS; res_dir["gas"] = "N2O"
    if include_indirect:
        res_ind["process"] = FAO_PROC_MAS; res_ind["gas"] = "N2O"
        return {"mas_direct": res_dir, "mas_indirect": res_ind}
    else:
        return {"mas_direct": res_dir}


# ──────────────────────────────────────────────────────────────────────────────
# 编排：非 MC（需要 headcount 已知）
# ──────────────────────────────────────────────────────────────────────────────
def run_gle(
    P: pd.DataFrame, *, year: int,
    enteric_df: Optional[pd.DataFrame] = None,
    prp_df: Optional[pd.DataFrame] = None,
    mm_df: Optional[pd.DataFrame] = None,
    mas_df: Optional[pd.DataFrame] = None,
    include_indirect: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    非 MC 情景：各过程活动表中必须已经包含 `headcount` 列。
    返回值是一个字典：键为过程名称，值为对应结果 DataFrame。
    """
    out: Dict[str, pd.DataFrame] = {}

    if enteric_df is not None and len(enteric_df) > 0:
        out["enteric"] = compute_enteric_ch4(P, enteric_df, year)

    if prp_df is not None and len(prp_df) > 0:
        out.update(compute_prp_n2o(P, prp_df, year, include_indirect=include_indirect))

    if mm_df is not None and len(mm_df) > 0:
        out["mm_ch4"] = compute_mm_ch4(P, mm_df, year)
        out.update(compute_mm_n2o(P, mm_df, year, include_indirect=include_indirect))

    if mas_df is not None and len(mas_df) > 0:
        out.update(compute_mas_n2o(P, mas_df, year, include_indirect=include_indirect))

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 编排：MC 情景（产量 + MC 抽样 → 动物流量 → 存栏 → 排放）
# ──────────────────────────────────────────────────────────────────────────────
def run_gle_mc(
    P: pd.DataFrame, *, year: int,
    base_df: pd.DataFrame,
    mc_draws: pd.DataFrame,
    models_slaughter_stock: Dict[str, Any],
    product_type_col: str = "product_type",
    include_indirect: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    MC 全流程：
      base_df（只有产量） + mc_draws（产率/胴体率抽样）
        → `mc_animals_from_production` 计算动物流量（meat: 屠宰；dairy: 泌乳）
        → `mc_predict_stock` 用“屠宰→存栏”回归转换为 headcount（保持 replicate）
        → compute_* 按过程计算排放
    返回：包含 replicate 的结果字典（enteric/prp_direct/...）。
    """
    # 1) 产量 + 抽样 → 动物流量（每条生产记录 × 多个 replicate）
    flow = mc_animals_from_production(base_df, mc_draws, product_type_col=product_type_col)

    # 2) 动物流量 → 存栏（按国家×品类的屠宰→存栏回归或比值回退）
    flow2 = mc_predict_stock(flow, models_slaughter_stock, link="slaughter")

    # 3) 用同一份 flow2（含 headcount & replicate）计算各过程排放
    res: Dict[str, pd.DataFrame] = {}
    res["enteric"] = compute_enteric_ch4(P, flow2, year)
    res.update(compute_prp_n2o(P, flow2, year, include_indirect=include_indirect))
    res["mm_ch4"] = compute_mm_ch4(P, flow2, year)
    res.update(compute_mm_n2o(P, flow2, year, include_indirect=include_indirect))
    res.update(compute_mas_n2o(P, flow2, year, include_indirect=include_indirect))

    return res


# ──────────────────────────────────────────────────────────────────────────────
# MC 结果汇总（按分组列求分位数）
# ──────────────────────────────────────────────────────────────────────────────
def summarize_mc_results(
    results: Dict[str, pd.DataFrame],
    group_cols: List[str],
    value_cols: List[str],
    qs=(0.05, 0.5, 0.95)
) -> Dict[str, pd.DataFrame]:
    """
    对 `run_gle_mc()` 的（含 replicate）输出进行汇总：
    - 按 group_cols 分组（例如 ["country","iso3","commodity","year","process","gas"]）
    - 对 value_cols（例如 ["CH4_kt","N2O_kt"]）计算分位数（默认 P5/P50/P95）

    返回：同结构的 DataFrame 字典（各键与原结果一致）。
    """
    out: Dict[str, pd.DataFrame] = {}
    for key, df in results.items():
        # 若这个过程没有 replicate（理论上 MC 都会有），则原样返回
        if "replicate" not in df.columns:
            out[key] = df.copy()
            continue

        g = df.groupby(group_cols, dropna=False)
        qs_names = [f"q{int(q*100)}" for q in qs]
        frames = []
        for v in value_cols:
            if v not in df.columns:
                continue
            # groupby-quantile：得到 (group × quantile) 的二维表
            agg = g[v].quantile(q=qs).unstack(level=-1)
            agg.columns = qs_names
            frames.append(agg)

        out[key] = pd.concat(frames, axis=1).reset_index() if frames else \
                   df.groupby(group_cols, dropna=False).size().reset_index(name="n")
    return out


__all__ = [
    # 参数访问
    "load_params_wide", "get_param_wide",
    # FAO→训练集 与 回归
    "prepare_regression_data_from_fao", "fit_stock_from_slaughter",
    # MC：产量→动物流量→存栏
    "mc_animals_from_production", "mc_predict_stock",
    # 过程排放（非 MC）
    "compute_enteric_ch4", "compute_prp_n2o", "compute_mm_ch4", "compute_mm_n2o", "compute_mas_n2o",
    # 编排器
    "run_gle", "run_gle_mc",
    # MC 汇总
    "summarize_mc_results",
]
