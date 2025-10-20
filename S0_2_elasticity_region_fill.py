# -*- coding: utf-8 -*-
"""
按“国家→区域 + 商品→四大类(Crop/Meat/Dairy/Other)”两级规则进行区域均值填补，
并在原工作簿基础上保留所有 sheet，仅替换 6 个目标表。

(1) 读取 region_map.xlsx
    - 'region' sheet：Country（或同义列）→ Region（或同义列）
    - 'item'   sheet：Commodity → Commodity_agg（目标四类：Crop/Meat/Dairy/Other）

(2) 非交叉四表：Supply-Temperature, Supply-Own-Price, Demand-Own-Price, Demand-Income
    - 先用 Region×Commodity 的非 NaN 均值填 Elasticity_mean/min/max
    - 若 Region×Commodity 仍无均值，则用 Region×Commodity_agg 的均值填补
    - 还没有则保留为 NaN

(3) 交叉两表：Demand_Cross_mean, Supply_Cross_mean
    - 仅做 Region 层面的补充：0 -> NaN；合并 Region；按 Region×Commodity 逐列求均值回填；剩余 NaN -> 0
    - 不做 Commodity_agg（Crop/Meat/Dairy/Other）层面的回填

注意：
- 仅替换上述 6 张表，其余 sheet 原样拷贝数据（不保证保留单元格格式/公式）。
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------
# 路径设置（同目录运行可直接用默认）
# -----------------------
IN_ELA  = Path("../../src/Elasticity_v3_processed.xlsx")
IN_RMAP = Path("../../src/dict_v3.xlsx")
OUT_ELA = Path("../../src/Elasticity_v3_processed_filled_by_region.xlsx")

# 目标表名
NON_CROSS_SHEETS  = ["Supply-Temperature", "Supply-Own-Price", "Demand-Own-Price", "Demand-Income"]
CROSS_MEAN_SHEETS = ["Demand_Cross_mean", "Supply_Cross_mean"]


# -----------------------
# 工具函数：字符串规范化 / 列名模糊识别
# -----------------------
def norm(s):
    """轻度清洗：去首尾空格，统一破折号，合并多空格；NaN 原样返回。"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s

def canon(name: str) -> str:
    """列名归一化（用于模糊匹配列名）。"""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")

def find_col(columns, candidates):
    """
    在给定 columns 中搜索候选列名（大小写/下划线/短横线不敏感）。
    candidates: 可能的同义列名列表（如 ["Country", "Region_label_new"]）
    返回：命中的真实列名；若都未命中，返回 None。
    """
    cmap = {canon(c): c for c in columns}
    for cand in candidates:
        key = canon(cand)
        if key in cmap:
            return cmap[key]
    return None


# -----------------------
# 读取 Country→Region & Commodity→Commodity_agg
# -----------------------
def load_mappings(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)
    sheets = set(xls.sheet_names)

    # 1) region sheet：Country→Region
    reg = pd.read_excel(xls, sheet_name="region")
    reg.columns = [str(c).strip() for c in reg.columns]

    reg["Country"] = reg["Region_label_new"].map(norm)
    reg["Region"]  = reg["Region_agg4"].map(norm)
    reg = reg.dropna(subset=["Country", "Region"]).drop_duplicates(subset=["Country"])

    # 2) item sheet：Commodity→Commodity_agg
    itm = pd.read_excel(xls, sheet_name="Emis_item")
    itm.columns = [str(c).strip() for c in itm.columns]

    itm["Commodity"]     = itm["Item Elasticity Map"].map(norm)
    itm["Commodity_agg"] = itm["Item Cat2"].map(norm)

    # 仅允许四类；若有空或不在集合内，统一归为 "Other"
    allowed = {"crop", "meat", "dairy", "other"}
    itm["Commodity_agg"] = itm["Commodity_agg"].apply(
        lambda x: "Other" if pd.isna(x) or str(x).strip().lower() not in allowed else str(x).strip().title()
    )

    return reg, itm


# -----------------------
# 非交叉四表：两级回填（Region×Commodity → Region×Commodity_agg）
# -----------------------
def fill_non_cross_with_region_and_cat(df: pd.DataFrame, reg: pd.DataFrame, itm: pd.DataFrame) -> pd.DataFrame:
    """
    对非交叉 sheet（含 Elasticity_* 列）进行两级回填：
      第1级：Region×Commodity 均值
      第2级：Region×Commodity_agg 均值
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    # 规范关键列
    for c in ["Country", "Commodity"]:
        if c in df.columns:
            df[c] = df[c].map(norm)

    # 需要回填的目标列
    raw_target_cols = [c for c in ["Elasticity_mean", "Elasticity_min", "Elasticity_max"] if c in df.columns]
    numeric_targets = [c for c in raw_target_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_targets:
        # 没有目标列则直接返回
        return df

    # 合并 Region 信息
    df = df.merge(reg, on="Country", how="left")

    # ------- 第1级：Region×Commodity 均值 -------
    # 仅对有 Region 的行参与分组
    has_region = df["Region"].notna()

    if numeric_targets:
        grp1 = df.loc[has_region].groupby(["Region", "Commodity"])[numeric_targets]
        reg_comm_means = grp1.transform("mean").reindex(df.index)

        # 用第1级均值填补
        for col in numeric_targets:
            mask = df[col].isna() & df["Region"].notna() & reg_comm_means[col].notna()
            df.loc[mask, col] = reg_comm_means.loc[mask, col]

    # ------- 第2级：Region×Commodity_agg 均值 -------
    # 合并 Commodity_agg
    df = df.merge(itm, on="Commodity", how="left")
    # 若 Commodity 未映射到四类，已在 load_mappings 时归为 "Other"
    has_region_cat = df["Region"].notna() & df["Commodity_agg"].notna()
    if has_region_cat.any() and numeric_targets:
        grp2 = df.loc[has_region_cat].groupby(["Region", "Commodity_agg"])[numeric_targets]
        reg_cat_means = grp2.transform("mean").reindex(df.index)

        for col in numeric_targets:
            mask = df[col].isna() & has_region_cat & reg_cat_means[col].notna()
            df.loc[mask, col] = reg_cat_means.loc[mask, col]

    # 清理辅助列
    return df.drop(columns=[c for c in ["Region", "Commodity_agg"] if c in df.columns])


# -----------------------
# 交叉两表：0->NaN，Region×Commodity 均值回填，剩余 NaN -> 0
# -----------------------
def fill_cross_mean_with_region(df: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """
    对交叉均值表（宽表）：Country, Commodity + 若干 cross-item 列。
    逻辑：
      - 先把所有 0 视为缺失（置 NaN）
      - 合并 Region；按 Region×Commodity 对每个 cross 列分别计算均值回填
      - 其余 NaN 再填回 0
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for c in ["Country", "Commodity"]:
        if c in df.columns:
            df[c] = df[c].map(norm)

    # 交叉列：除 Country / Commodity 外的全部列
    cross_cols_all = [c for c in df.columns if c not in ["Country", "Commodity"]]
    num_cols = [c for c in cross_cols_all if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df

    # 0 -> NaN
    df[num_cols] = df[num_cols].replace(0, np.nan)

    # 合并 Region
    df = df.merge(reg, on="Country", how="left")
    has_region = df["Region"].notna()

    if has_region.any():
     # 逐列在 Region×Commodity 分组后求均值（只对数值列）
        grp = df.loc[has_region].groupby(["Region", "Commodity"])[num_cols]
        reg_comm_means = grp.transform("mean").reindex(df.index)
        for col in num_cols:
             mask = df[col].isna() & df["Region"].notna() & reg_comm_means[col].notna()
             df.loc[mask, col] = reg_comm_means.loc[mask, col]
    # 残余 NaN -> 0
    df[num_cols] = df[num_cols].fillna(0)

    return df.drop(columns=["Region"])


# -----------------------
# 主流程：读取→处理→写出（保留所有 sheet）
# -----------------------
# 1) 载入映射
reg, itm = load_mappings(IN_RMAP)

# 2) 读取原始工作簿的所有 sheet（字典：sheet_name -> DataFrame）
xls = pd.ExcelFile(IN_ELA)
all_sheets = {}
for name in xls.sheet_names:
    all_sheets[name] = pd.read_excel(xls, sheet_name=name)

# 3) 处理 6 个目标表
processed = {}

# 非交叉四表：两级回填
for name in NON_CROSS_SHEETS:
    if name in all_sheets:
        df = all_sheets[name]
        processed[name] = fill_non_cross_with_region_and_cat(df, reg, itm)

# 交叉两表：0->NaN->区域均值->0
for name in CROSS_MEAN_SHEETS:
    if name in all_sheets:
        df = all_sheets[name]
        processed[name] = fill_cross_mean_with_region(df, reg)

# 4) 写出：保留所有 sheet，仅替换被处理的 6 张表
with pd.ExcelWriter(OUT_ELA, engine="xlsxwriter") as writer:
    for name, df in all_sheets.items():
        if name in processed:           # 用处理后的
            processed[name].to_excel(writer, sheet_name=name[:31], index=False)
        else:                            # 其它原样写回
            df.to_excel(writer, sheet_name=name[:31], index=False)

print(f"完成：{OUT_ELA.resolve()}")

