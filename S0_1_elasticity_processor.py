# -*- coding: utf-8 -*-
"""
S0.5 Elasticity Processor — 多对多映射版（仅输出“宽表”）【完整注释版】
================================================================
目标：
  按“国家 × 目标商品(Commodity Map/All) × Element（六类）”生成弹性结果的**宽表**；
  其中交叉价格(Cross-Price)的列集固定为《Item map》中的 **Commodity All** 白名单，
  对每个 (Country, Commodity) 的交叉向量**先初始化为 0**，若有观测再覆盖。

数据来源（Excel：Elasticity_v3.xlsx）：
  - raw_all    : 原始弹性条目（含 Commodity, Cross_price_item, Element Map, D/S, Region, Selected, Elasticity）
  - Item map   : 物项映射（Commodity Map, Commodity Raw, Commodity All；可选 Cross_price_item Map/Raw）
  - Region map : 地区映射（Region_label_new, Region_Elasticity_v3, Region_Elasticity_keep）

核心规则：
  1) 仅保留 raw_all 中 Selected==1 的记录；
  2) 使用“Map↔Raw **多对多**映射”：
     - forward:  map_token_lc -> {raw 原样字符串, ...}
     - reverse:  raw_lc        -> {map_token_lc, ...}
     说明：Map 列可分号分隔，代表多个“目标 token”都映射到该 Raw；一个 Map token 也可出现于多行⇒对应多 Raw；
  3) Element 分类：严格依赖 Element Map + D/S（Cross_price_item **不参与**分类），归一得到 6 类：
     Demand-Cross-Price / Demand-Income / Demand-Own-Price /
     Supply-Cross-Price / Supply-Own-Price / Supply-Temperature
  4) 三层回退：
     国家（Region_Elasticity_v3）→ 区域层（Region_Elasticity_keep）→ 同 keep 组国家聚合（peer）
     - 非交叉：在多个 raw/区域上聚合时：mean 用 n 加权，min 取最小，max 取最大，n 求和；
     - 交叉：先把 (main_raw, cross_raw) 聚合到国家/区域/peer，再将 cross_raw 通过 reverse 映射到目标 token；
             同一目标 token 汇总多个来源时，仍按 n 加权等规则聚合；最终以 Commodity All 列集输出，缺失填 0。
  5) 输出：
     - Demand/Supply Cross：各出 4 张矩阵（mean/min/max/n），行 (Country, Commodity)，列=Commodity All；
     - 非交叉 4 类：各 1 张表，列为 (Elasticity_mean/min/max/n, source_level)。

"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

INPUT_XLSX  = Path("../../src/Elasticity_v3.xlsx")
OUTPUT_XLSX = Path("../../src/Elasticity_v3_processed_out3.xlsx")

# -----------------------------
# 文本规范化工具
# -----------------------------
def norm(s):
    """轻量清洗：去首尾空格、统一破折号、压缩多空格。保留原大小写用于展示。"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def canon(s):
    """强规范化：在 norm 基础上，转小写、把 '_' 和 '-' 视作空格、再去首尾空格。
    用于“无关大小写/连字符/多空格”的匹配键。"""
    if pd.isna(s):
        return np.nan
    s0 = norm(s).lower().replace("_", " ").replace("-", " ").strip()
    return s0

def lc(x):
    """仅在是 str 时转小写+去首尾空格；否则原样返回。"""
    return x.lower().strip() if isinstance(x, str) else x

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """在 df.columns 中寻找候选列名（大小写/连字符/多空格无关），返回真实列名。"""
    cmap = {canon(c): c for c in df.columns}
    for cand in candidates:
        key = canon(cand)
        if key in cmap:
            return cmap[key]
    return None

# -----------------------------
# Map↔Raw 多对多 映射
# -----------------------------
def build_maps_many_to_many(df_map: pd.DataFrame, map_col: str, raw_col: str):
    """从 Item map 构建：
       forward: { map_token_lc -> set(raw 原样) }
       reverse: { raw_lc        -> set(map_token_lc) }
       - Map 列支持分号分隔；
       - 键使用 lc 形态做大小写无关匹配；
       - Raw 值保留原样字符串，便于审阅导出。"""
    forward: Dict[str, Set[str]] = defaultdict(set)
    reverse: Dict[str, Set[str]] = defaultdict(set)
    # 逐行读取映射：每行的 Commodity Map 可能是“a;b;c”
    for _, r in df_map[[map_col, raw_col]].dropna(how="any").iterrows():
        raw_v = norm(r[raw_col])     # 原样清洗
        raw_lc = lc(raw_v)           # 小写键
        for tok in str(r[map_col]).split(";"):
            token = norm(tok)
            if not token:
                continue
            token_lc = lc(token)     # 目标 token（小写键）
            forward[token_lc].add(raw_v)
            reverse[raw_lc].add(token_lc)
    return forward, reverse

# -----------------------------
# Element 分类（严格基于 Element Map + D/S）
# -----------------------------
def classify_element_ds(element_text: Optional[str], ds_text: Optional[str]) -> Tuple[str, str]:
    """把一行记录归入六类元素之一；Cross_price_item 不参与分类（仅用于交叉映射阶段）。"""
    # 只区分 Demand / Supply 两类（缩写/大小写/空格容错）
    ds = (str(ds_text).strip().title() if pd.notna(ds_text) else "Demand")
    ds = "Supply" if ds.lower().startswith("sup") else "Demand"
    # 统一 Element Map 文本形态
    em = canon(element_text)
    # 首选精确名（允许大小写/连字符差异）
    canonical = {
        "demand cross price":      ("Demand-Cross-Price", "Demand"),
        "demand income":           ("Demand-Income",      "Demand"),
        "demand own price":        ("Demand-Own-Price",   "Demand"),
        "supply cross price":      ("Supply-Cross-Price", "Supply"),
        "supply own price":        ("Supply-Own-Price",   "Supply"),
        "supply temperature":      ("Supply-Temperature", "Supply"),
    }
    if em in canonical:
        return canonical[em]
    # 关键字兜底（Element Map 不规范时）
    if em is not np.nan:
        if "temperature" in em or "temp" in em:
            return ("Supply-Temperature", "Supply")
        if "income" in em:
            return ("Demand-Income", "Demand")
        if "cross" in em:
            return (f"{ds}-Cross-Price", ds)
        if "own" in em or "own price" in em or "price elasticity" in em or "price" in em:
            return (f"{ds}-Own-Price", ds)
    # 再兜底：按 D/S 归为 Own-Price
    return (f"{ds}-Own-Price", ds)

# =============================
#            主流程
# =============================
# ---------- 读取与规范 ----------
xls = pd.ExcelFile(INPUT_XLSX)
raw_all    = pd.read_excel(xls, sheet_name="raw_all")
item_map   = pd.read_excel(xls, sheet_name="Item map")
region_map = pd.read_excel(xls, sheet_name="Region map")

# 列名去空格，避免隐性不匹配
raw_all.columns    = [c.strip() for c in raw_all.columns]
item_map.columns   = [c.strip() for c in item_map.columns]
region_map.columns = [c.strip() for c in region_map.columns]

# 关键列标准化 / 数值化
for c in ["Commodity", "Element", "Element Map", "Cross_price_item", "D/S", "Region", "Selected", "Elasticity"]:
    if c in raw_all.columns:
        if c in ["Selected", "Elasticity"]:
            raw_all[c] = pd.to_numeric(raw_all[c], errors="coerce")
        else:
            raw_all[c] = raw_all[c].map(norm)

# 仅保留 Selected==1 的条目，并进行 Element 分类
raw_sel = raw_all[raw_all["Selected"] == 1].copy()
raw_sel["Element_Class"], raw_sel["DS_Class"] = zip(*raw_sel.apply(
    lambda r: classify_element_ds(r.get("Element Map"), r.get("D/S")), axis=1
))

# ---------- Item map：列定位与白名单 ----------
cm_map_col  = find_col(item_map, ["Commodity Map"])
cm_raw_col  = find_col(item_map, ["Commodity Raw"])
cm_all_col  = find_col(item_map, ["Commodity All"])
if not (cm_map_col and cm_raw_col and cm_all_col):
    raise ValueError("Item map 缺少 Commodity Map/Raw/All 列")

# 目标商品白名单（列名/目标 token 的来源）
target_all: List[str] = [norm(x) for x in item_map[cm_all_col].dropna().unique().tolist()]
target_all_lc = set([x.lower() for x in target_all])

# 构建主维度（Commodity）的多对多映射
commodity_fwd_m2m, commodity_rev_m2m = build_maps_many_to_many(item_map, cm_map_col, cm_raw_col)

# 若提供 Cross_price_item Map/Raw，则优先用于交叉维度映射；否则回退到 commodity_rev_m2m
cpi_map_col = find_col(item_map, ["Cross_price_item Map", "Cross item Map", "Cross-price item Map", "Cross price item Map"])
cpi_raw_col = find_col(item_map, ["Cross_price_item Raw", "Cross item Raw", "Cross-price item Raw", "Cross price item Raw"])
if cpi_map_col and cpi_raw_col:
    cross_fwd_m2m, cross_rev_m2m = build_maps_many_to_many(item_map, cpi_map_col, cpi_raw_col)
else:
    cross_fwd_m2m, cross_rev_m2m = {}, {}

# ---------- Region map：建立回退链 ----------
rl_col   = find_col(region_map, ["Region_label_new"])
v3_col   = find_col(region_map, ["Region_Elasticity_v3"])
keep_col = find_col(region_map, ["Region_Elasticity_keep"])
if not rl_col:
    raise ValueError("Region map 缺少 Region_label_new 列")

region_map[rl_col]   = region_map[rl_col].map(norm)
if v3_col:
    region_map[v3_col]   = region_map[v3_col].map(norm)
if keep_col:
    region_map[keep_col] = region_map[keep_col].map(norm)

# 国家 → 所属 v3 区域；国家 → keep 组
country_to_region: Dict[str, str] = {}
country_to_keep  : Dict[str, Optional[str]] = {}
for _, r in region_map.iterrows():
    cty = r.get(rl_col)
    v3  = r.get(v3_col) or cty if v3_col else cty
    kp  = r.get(keep_col) if keep_col else None
    if pd.notna(cty):
        country_to_region[cty] = v3
        country_to_keep[cty]   = kp

# 需要输出的国家清单（以 Region map 中的国家为准）
target_countries = [x for x in region_map[rl_col].dropna().unique().tolist()]

# keep 组 → (去重后的) v3 区域列表，用于“peer”回退（同组国家聚合）
group_to_regionlist: Dict[str, List[str]] = defaultdict(list)
for cty, grp in country_to_keep.items():
    reg = country_to_region.get(cty)
    if grp and reg:
        group_to_regionlist[grp].append(reg)
for grp in list(group_to_regionlist.keys()):
    group_to_regionlist[grp] = sorted(set(group_to_regionlist[grp]))

# ---------- 预聚合：先按 (Element, main_raw, Region) 聚合原始值 ----------
# 非交叉：dict[(ec, main_raw_lc, region_lc)] -> (mean, min, max, n)
non_cross_by_key: Dict[Tuple[str, str, str], Tuple[float, float, float, int]] = {}
for (ec, com, reg), grp in raw_sel.groupby(["Element_Class", "Commodity", "Region"]):
    vals = pd.to_numeric(grp["Elasticity"], errors="coerce").dropna()
    if len(vals):
        non_cross_by_key[(ec, lc(com), lc(reg))] = (float(vals.mean()), float(vals.min()), float(vals.max()), int(vals.shape[0]))

# 交叉：dict[(ec, main_raw_lc, region_lc)] -> {cross_raw_lc: (mean, min, max, n)}
cross_by_key: Dict[Tuple[str, str, str], Dict[str, Tuple[float, float, float, int]]] = defaultdict(dict)
if "Cross_price_item" in raw_sel.columns:
    for (ec, com, reg, cp), grp in raw_sel.groupby(["Element_Class", "Commodity", "Region", "Cross_price_item"]):
        vals = pd.to_numeric(grp["Elasticity"], errors="coerce").dropna()
        if len(vals):
            cross_by_key[(ec, lc(com), lc(reg))][lc(cp)] = (float(vals.mean()), float(vals.min()), float(vals.max()), int(vals.shape[0]))

# ---------- 聚合器与取数助手 ----------
def agg_stats(stats_list):
    """把若干 (mean, min, max, n) 聚合：mean 按 n 加权；min 取最小；max 取最大；n 求和。"""
    if not stats_list:
        return None
    m_sum = 0.0; n_sum = 0; mi = np.inf; ma = -np.inf
    for (mean_v, min_v, max_v, n_v) in stats_list:
        m_sum += mean_v * n_v
        n_sum += n_v
        mi = min(mi, min_v)
        ma = max(ma, max_v)
    if n_sum == 0:
        return None
    mean_w = m_sum / n_sum
    if mi == np.inf:    mi = mean_w
    if ma == -np.inf:   ma = mean_w
    return (mean_w, mi, ma, int(n_sum))

def fetch_non_cross(ec: str, main_raws: Set[str], region: Optional[str] = None,
                    keep_grp: Optional[str] = None, scope: str = "country"):
    """在指定 scope 内（country/region/peer），将 main_raws 的非交叉条目聚合并返回统计。"""
    if scope == "country" and region:
        reg_candidates = [lc(region)]
    elif scope == "region" and keep_grp:
        reg_candidates = [lc(keep_grp)]
    elif scope == "peer" and keep_grp:
        reg_candidates = [lc(r) for r in group_to_regionlist.get(keep_grp, [])]
    else:
        reg_candidates = []
    stats = []
    for reg_lc in reg_candidates:
        for raw_name in main_raws:
            st = non_cross_by_key.get((ec, lc(raw_name), reg_lc))
            if st:
                stats.append(st)
    return agg_stats(stats)

def fetch_cross_dict(ec: str, main_raws: Set[str], region: Optional[str] = None,
                     keep_grp: Optional[str] = None, scope: str = "country"):
    """在指定 scope 内（country/region/peer），收集 (main_raw, cross_raw) 并按 cross_raw 聚合。"""
    if scope == "country" and region:
        reg_candidates = [lc(region)]
    elif scope == "region" and keep_grp:
        reg_candidates = [lc(keep_grp)]
    elif scope == "peer" and keep_grp:
        reg_candidates = [lc(r) for r in group_to_regionlist.get(keep_grp, [])]
    else:
        reg_candidates = []
    bucket: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    for reg_lc in reg_candidates:
        for raw_name in main_raws:
            dd = cross_by_key.get((ec, lc(raw_name), reg_lc), {})
            for cp_raw_lc, st in dd.items():
                bucket[cp_raw_lc].append(st)
    # 返回：{cross_raw_lc: (mean, min, max, n)}（已按 main_raw 合并）
    out: Dict[str, Tuple[float, float, float, int]] = {}
    for cp_raw_lc, lst in bucket.items():
        st = agg_stats(lst)
        if st:
            out[cp_raw_lc] = st
    return out

# ---------- 输出容器：交叉矩阵 & 非交叉表 ----------
# cross：分别收集 mean/min/max/n 的“宽表行”；最后再写入工作簿
cross_rows_by_stat: Dict[Tuple[str, str], List[Dict[str, object]]] = {
    ("Demand-Cross-Price", "mean"): [],
    ("Demand-Cross-Price", "min"):  [],
    ("Demand-Cross-Price", "max"):  [],
    ("Demand-Cross-Price", "n"):    [],
    ("Supply-Cross-Price", "mean"): [],
    ("Supply-Cross-Price", "min"):  [],
    ("Supply-Cross-Price", "max"):  [],
    ("Supply-Cross-Price", "n"):    [],
}
# 非交叉：四类 Element 各一张
non_cross_rows: Dict[str, List[Dict[str, object]]] = {
    "Demand-Income": [],
    "Demand-Own-Price": [],
    "Supply-Own-Price": [],
    "Supply-Temperature": [],
}

# ---------- 主循环：按 Element × Country × Commodity(Map token) 逐一生成 ----------
for ec in ["Demand-Cross-Price", "Demand-Income", "Demand-Own-Price",
           "Supply-Cross-Price", "Supply-Own-Price", "Supply-Temperature"]:
    for country in target_countries:
        reg_name = country_to_region.get(country)     # 该国对应的 v3 区域
        keep_grp = country_to_keep.get(country)       # 该国对应的 keep 组（区域聚合标签）
        for tgt in target_all:
            # 关键：一个“目标 token”可能映射到**多个** raw（多对多）
            main_raws = commodity_fwd_m2m.get(tgt.lower(), set())

            if ec in ["Demand-Cross-Price", "Supply-Cross-Price"]:
                # ---------- 交叉：先抓 cross_raw 的统计，再映射到目标 Cross token ----------
                dd = fetch_cross_dict(ec, main_raws, region=reg_name, scope="country")
                if not dd and keep_grp:
                    dd = fetch_cross_dict(ec, main_raws, keep_grp=keep_grp, scope="region")
                if not dd and keep_grp:
                    dd = fetch_cross_dict(ec, main_raws, keep_grp=keep_grp, scope="peer")

                # present 保存“每个 Cross 目标 token”的聚合（初始化为空；稍后按白名单补 0）
                present = defaultdict(lambda: {"m": 0.0, "mi": np.inf, "ma": -np.inf, "n": 0})
                if dd:
                    for cp_raw_lc, st in dd.items():
                        # 先把 cross_raw_lc → 目标 token（优先 Cross 专用映射；若无则回退 commodity_rev）
                        if cross_rev_m2m and (cp_raw_lc in cross_rev_m2m):
                            targets_lc = list(cross_rev_m2m[cp_raw_lc])
                        else:
                            targets_lc = list(commodity_rev_m2m.get(cp_raw_lc, []))
                        # 只保留白名单中的 Cross 目标
                        targets_lc = [t for t in targets_lc if t in target_all_lc]
                        if not targets_lc:
                            continue
                        mean_i, min_i, max_i, n_i = st
                        # 将同一目标（可能由多个 cross_raw 来的）按 n 加权聚合到 present
                        for cp_tgt_lc in targets_lc:
                            present[cp_tgt_lc]["m"]  += mean_i * n_i
                            present[cp_tgt_lc]["n"]  += n_i
                            present[cp_tgt_lc]["mi"]  = min(present[cp_tgt_lc]["mi"], min_i)
                            present[cp_tgt_lc]["ma"]  = max(present[cp_tgt_lc]["ma"], max_i)

                # 组装四张矩阵的“行”：先把 Commodity All 全列初始化为 0，再用 present 覆盖
                base_mean = {"Country": country, "Commodity": tgt}
                base_min  = {"Country": country, "Commodity": tgt}
                base_max  = {"Country": country, "Commodity": tgt}
                base_n    = {"Country": country, "Commodity": tgt}
                for cp_name in target_all:
                    key_lc = cp_name.lower()
                    d = present.get(key_lc, None)
                    if d and d["n"] > 0:
                        mean_w = d["m"] / d["n"]
                        base_mean[cp_name] = mean_w
                        base_min[cp_name]  = d["mi"] if d["mi"] != np.inf else mean_w
                        base_max[cp_name]  = d["ma"] if d["ma"] != -np.inf else mean_w
                        base_n[cp_name]    = int(d["n"])
                    else:
                        base_mean[cp_name] = 0.0
                        base_min[cp_name]  = 0.0
                        base_max[cp_name]  = 0.0
                        base_n[cp_name]    = 0
                cross_rows_by_stat[(ec, "mean")].append(base_mean)
                cross_rows_by_stat[(ec, "min")].append(base_min)
                cross_rows_by_stat[(ec, "max")].append(base_max)
                cross_rows_by_stat[(ec, "n")].append(base_n)

            else:
                # ---------- 非交叉：按三层回退在 main_raws 集合上聚合 ----------
                st = fetch_non_cross(ec, main_raws, region=reg_name, scope="country")
                if not st and keep_grp:
                    st = fetch_non_cross(ec, main_raws, keep_grp=keep_grp, scope="region")
                if not st and keep_grp:
                    st = fetch_non_cross(ec, main_raws, keep_grp=keep_grp, scope="peer")
                if not st:
                    # 完全找不到：保留行，但用 NaN/0 占位
                    non_cross_rows[ec].append({
                        "Country": country, "Commodity": tgt,
                        "Elasticity_mean": np.nan, "Elasticity_min": np.nan, "Elasticity_max": np.nan,
                        "n": 0, "source_level": "missing"
                    })
                else:
                    non_cross_rows[ec].append({
                        "Country": country, "Commodity": tgt,
                        "Elasticity_mean": st[0], "Elasticity_min": st[1], "Elasticity_max": st[2],
                        "n": st[3], "source_level": "country/region/peer"
                    })

# ---------- 写出：各表到一个 Excel 工作簿 ----------
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    # 交叉价：Demand/Supply × (mean/min/max/n)
    for ec in ["Demand-Cross-Price", "Supply-Cross-Price"]:
        for stat in ["mean", "min", "max", "n"]:
            rows = cross_rows_by_stat[(ec, stat)]
            df = pd.DataFrame(rows)
            # 列顺序：Country, Commodity, 然后 Commodity All
            ordered_cols = ["Country", "Commodity"] + target_all
            # 某些目标可能未出现（极少），确保列齐全
            for col in target_all:
                if col not in df.columns:
                    df[col] = 0 if stat in ["mean", "min", "max"] else 0
            df = df[ordered_cols]
            sheet = f"{ec.split('-')[0]}_Cross_{stat}"[:31]  # Excel sheet 名最长 31 字符
            df.to_excel(writer, sheet_name=sheet, index=False)

    # 非交叉四类
    for ec in ["Demand-Income", "Demand-Own-Price", "Supply-Own-Price", "Supply-Temperature"]:
        df = pd.DataFrame(non_cross_rows[ec])
        ordered_cols = ["Country", "Commodity", "Elasticity_mean", "Elasticity_min", "Elasticity_max", "n", "source_level"]
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64")
        df = df[ordered_cols]
        sheet = ec[:31]
        df.to_excel(writer, sheet_name=sheet, index=False)

    # 附：两张映射审阅表，便于核对 Fish/Durum 等多对多归属
    fwd_rows = [[tok_lc, ";".join(sorted(set(raws)))] for tok_lc, raws in commodity_fwd_m2m.items()]
    pd.DataFrame(fwd_rows, columns=["Map_token(lc)", "Raw_list"]).to_excel(writer, sheet_name="commodity_forward_m2m", index=False)
    rev_rows = [[raw_lc, ";".join(sorted(set(toks)))] for raw_lc, toks in commodity_rev_m2m.items()]
    pd.DataFrame(rev_rows, columns=["Raw(lc)", "Map_tokens(lc)"]).to_excel(writer, sheet_name="commodity_reverse_m2m", index=False)

print("Done:", OUTPUT_XLSX)
