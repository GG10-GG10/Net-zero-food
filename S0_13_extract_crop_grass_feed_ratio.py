# -*- coding: utf-8 -*-
"""
End-to-end pipeline:
1) Parse Table S10 (pp.60–70) -> region_level
2) (Optional) Expand region_level to country_level by Table S4 (p.36)
3) Parse Table S11 (pp.71–81) -> region weights (n_herd_1000)
4) Weighted region means of S10 by S11 (per Species_System×Region)
5) WRD-fill for missing weighted cells
6) Read item_map & region_country_map; build country_level_weighted as
   Item_Emis × M49_Country_Code cross, mapping to region-weighted shares
7) Write a final Excel with all sheets

Author: GG10 helper
"""

import os
import re
import numpy as np
import pandas as pd
import pdfplumber

# =========================
# User CONFIG (edit here)
# =========================
PDF_PATH = r"/mnt/data/sapp.pdf"  # 原始PDF（含 S4/S10/S11）
OUT_XLSX_FINAL = r"/mnt/data/TableS10_p60_70_combined_with_country_weighted.xlsx"

# S10/S4/S11 页码（人类编号，从1开始）
S10_START_PAGE, S10_END_PAGE = 60, 70
S4_PAGE = 36
S11_START_PAGE, S11_END_PAGE = 71, 81

# 是否保留“旧的 country_level”（按 S4 展开区域→国家）
GENERATE_OLD_COUNTRY_LEVEL = False

# =========================
# Constants
# =========================
SPECIES = ["BOVD","BOVO","SGTD","SGTO"]
REGIONS = ["CIS","EAS","EUR","LAM","MNA","NAM","OCE","SAS","SEA","SSA","WRD"]
SUBTYPES_S10 = ["LGA","LGH","LGT","MRA","MRH","MRT","Other","URBAN"]
SUBTYPES_S11 = ["ANY"] + SUBTYPES_S10

# =========================
# Helpers
# =========================
def extract_text_range(pdf_path: str, start_page: int, end_page: int):
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in range(start_page - 1, end_page):
            if 0 <= p < len(pdf.pages):
                text = pdf.pages[p].extract_text(x_tolerance=2, y_tolerance=2) or ""
                out.append((p + 1, text))
    return out

def parse_table_s10(pdf_path: str, start_page=60, end_page=70) -> pd.DataFrame:
    """
    Parse Table S10 region rows. Each data row:
      Subtype + [13 numeric: solCHO..k5] + [optional: Starch] + [up to 4 shares: Grass, grain, stover, occasional]
    Returns columns:
      Species_System, Region, Subtype, solCHO..k5, Starch, Grass (%), grain (%), stover (%), occasional (%), _page
    """
    text_by_page = extract_text_range(pdf_path, start_page, end_page)
    rows = []
    species, region = None, None
    num_pat = re.compile(r'^-?\d+(?:\.\d+)?$')

    for page_no, text in text_by_page:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line in SPECIES:
                species = line;  continue
            if line in REGIONS:
                region = line;   continue

            parts = line.split()
            if parts and parts[0] in SUBTYPES_S10 and species and region:
                subtype = parts[0]
                toks = [re.sub(r'[^0-9\.\-]', '', t) for t in parts[1:]]
                toks = [t for t in toks if t and num_pat.match(t)]

                base = [float(x) for x in toks[:13]] if len(toks) >= 13 else [None]*13
                rest = toks[13:] if len(toks) > 13 else []

                solCHO, solCP, Ash, degNDF, degCP, CP, Fat, ME, NDF, k1, k4, k2, k5 = base
                Starch = None
                shares = [0,0,0,0]  # Grass, grain, stover, occasional

                if rest:
                    # If first rest token looks like a fraction (<=1.5), treat as Starch
                    try:
                        first = float(rest[0])
                        if ('.' in rest[0]) and (first <= 1.5):
                            Starch = first; rest = rest[1:]
                    except Exception:
                        pass
                    for i in range(min(4, len(rest))):
                        try: shares[i] = int(round(float(rest[i])))
                        except Exception: shares[i] = 0

                rows.append({
                    "Species_System": species, "Region": region, "Subtype": subtype,
                    "solCHO": solCHO, "solCP": solCP, "Ash": Ash,
                    "degNDF": degNDF, "degCP": degCP, "CP": CP, "Fat": Fat,
                    "ME": ME, "NDF": NDF, "k1": k1, "k4": k4, "k2": k2, "k5": k5,
                    "Starch": Starch,
                    "Grass (%)": shares[0], "grain (%)": shares[1], "stover (%)": shares[2], "occasional (%)": shares[3],
                    "_page": page_no
                })

    df = pd.DataFrame(rows)
    # sort
    if not df.empty:
        r_order = {r:i for i,r in enumerate(REGIONS)}
        s_order = {s:i for i,s in enumerate(SUBTYPES_S10)}
        df["region_order"] = df["Region"].map(r_order)
        df["subtype_order"] = df["Subtype"].map(s_order)
        df.sort_values(["Species_System","region_order","subtype_order"], inplace=True)
        df.drop(columns=["region_order","subtype_order"], inplace=True)
    return df

def fill_regions_with_wrd(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Species_System, Subtype), if some Region is missing,
    copy that combo's WRD row and set Region to the missing one.
    """
    out = df.copy()
    for (sp, st), grp in df.groupby(["Species_System","Subtype"]):
        have = set(grp["Region"].unique())
        wrd = df[(df["Species_System"]==sp) & (df["Subtype"]==st) & (df["Region"]=="WRD")]
        if not len(wrd):  # no world row, skip
            continue
        for r in REGIONS:
            if r == "WRD":
                continue
            if r not in have:
                add = wrd.copy(); add["Region"] = r
                out = pd.concat([out, add], ignore_index=True)
    return out

def parse_table_s4_mapping(pdf_path: str, page_no=36) -> pd.DataFrame:
    """
    Parse Table S4 region->country lists; map EUR-Former USSR to CIS for S10/11 compatibility.
    Returns columns: S4_region, Subregion, Country, Region_S10
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_no - 1].extract_text(x_tolerance=2, y_tolerance=2) or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    region_acros = {"EUR","OCE","NAM","LAM","EAS","SEA","SAS","MNA","SSA"}
    current_region, buf, blocks = None, [], []

    for ln in lines:
        toks = ln.split()
        if toks and toks[0] in region_acros and len(toks) <= 3:
            if current_region and buf:
                blocks.append((current_region, "\n".join(buf))); buf = []
            current_region = toks[0];  continue
        if current_region: buf.append(ln)
    if current_region and buf:
        blocks.append((current_region, "\n".join(buf)))

    recs = []
    for reg, block in blocks:
        parts = [p.strip() for p in block.split("\n") if p.strip()]
        merged, i = [], 0
        while i < len(parts):
            cur = parts[i]
            if ("," not in cur) and i+1 < len(parts) and ("," in parts[i+1] or " and " in parts[i+1]):
                merged.append(cur + " " + parts[i+1]); i += 2
            else:
                merged.append(cur); i += 1

        for ln in merged:
            if "," in ln:
                m = re.match(r"^(.*?)[\s]{2,}(.+)$", ln)
                if m:
                    sub, countries = m.group(1).strip(), m.group(2).strip()
                else:
                    toks = ln.split(); sub = " ".join(toks[:2]); countries = " ".join(toks[2:])
                countries = re.sub(r"\s+", " ", countries)
                for c in [x.strip() for x in countries.split(",") if x.strip()]:
                    c2 = re.sub(r"\(.*?\)", "", c).strip()
                    recs.append({"S4_region": reg, "Subregion": sub, "Country": c2})
            else:
                c2 = re.sub(r"\(.*?\)", "", ln).strip()
                if c2 and not any(ch.isdigit() for ch in c2):
                    recs.append({"S4_region": reg, "Subregion": None, "Country": c2})

    df = pd.DataFrame(recs).drop_duplicates().reset_index(drop=True)

    def to_s10(row):
        s4 = row["S4_region"]; sub = (row["Subregion"] or "").lower()
        if s4 == "EUR" and "former ussr" in sub:
            return "CIS"
        return {"EUR":"EUR","OCE":"OCE","NAM":"NAM","LAM":"LAM","EAS":"EAS","SEA":"SEA","SAS":"SAS","MNA":"MNA","SSA":"SSA"}[s4]

    df["Region_S10"] = df.apply(to_s10, axis=1)
    # minimal naming cleanup
    df["Country"] = df["Country"].replace({
        "United States of America": "United States",
        "Fiji Islands": "Fiji",
        "Korea DPR": "Korea, DPR",
        "Congo Republic": "Congo",
        "Democratic Republic of Congo": "Congo, DR",
        "Netherland Antilles": "Netherlands Antilles",
        "Macedonia": "North Macedonia",
        "Bahamas": "The Bahamas",
    })
    return df

def parse_table_s11(pdf_path: str, start_page=71, end_page=81) -> pd.DataFrame:
    """
    Parse Table S11 rows starting with Subtype tokens; expect 10 numerics:
      n_productive_1000, n_herd_1000, col3(weight gain), col4(milk), manureN, entericCH4, mmCH4, mmN2O, mcN2O, mgN2O
    We keep both milk & weight gain columns; for乳/肉分别解读。
    """
    text_by_page = extract_text_range(pdf_path, start_page, end_page)
    rows = []
    species, region = None, None
    isnum = lambda t: re.fullmatch(r'-?\d+(?:\.\d+)?', t) is not None

    for page_no, text in text_by_page:
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line in SPECIES:
                species = line;  region = None;  continue
            if line in REGIONS and len(line) <= 4:
                region = line;   continue

            parts = line.split()
            if parts and (parts[0] in SUBTYPES_S11) and species and region:
                subtype = parts[0]
                nums = [float(t) for t in parts[1:] if isnum(t)]
                if len(nums) < 10:
                    continue  # wrapped or incomplete; skip
                n_prod, n_herd = nums[0], nums[1]
                col3, col4 = nums[2], nums[3]
                manureN, entericCH4, mmCH4, mmN2O, mcN2O, mgN2O = nums[4:10]

                # For dairy (BOVD/SGTD) vs other (BOVO/SGTO)
                if species in {"BOVD","SGTD"}:
                    avg_milk, weight_gain = col4, col3
                else:
                    avg_milk, weight_gain = col4, col3

                rows.append({
                    "Species_System": species, "Region": region, "Subtype": subtype,
                    "n_productive_1000": n_prod, "n_herd_1000": n_herd,
                    "avg_milk_kg_d": avg_milk, "weight_gain_g_d": weight_gain,
                    "manureN_kgBW075_yr": manureN,
                    "enteric_CH4_kgCO2e_perkgBW075_yr": entericCH4,
                    "manure_mgmt_CH4_kgCO2e_perkgBW075_yr": mmCH4,
                    "manure_mgmt_N2O_kgCO2e_perkgBW075_yr": mmN2O,
                    "manure_cropland_N2O_kgCO2e_perkgBW075_yr": mcN2O,
                    "manure_grassland_N2O_kgCO2e_perkgBW075_yr": mgN2O,
                    "_page": page_no
                })
    return pd.DataFrame(rows)

def build_schema_codebook() -> pd.DataFrame:
    rows = [
        ("Species_System","BOVD/BOVO/SGTD/SGTO"),
        ("Region","CIS/EAS/EUR/LAM/MNA/NAM/OCE/SAS/SEA/SSA/WRD"),
        ("Subtype","LGA/LGH/LGT/MRA/MRH/MRT/Other/URBAN（S11还含ANY聚合）"),
        ("solCHO","soluble carbohydrates"),
        ("solCP","soluble crude protein"),
        ("Ash","ash"),
        ("degNDF","degradable NDF fraction"),
        ("degCP","degradable CP fraction"),
        ("CP","crude protein (%)"),
        ("Fat","fat (%)"),
        ("ME","metabolizable energy"),
        ("NDF","neutral detergent fiber"),
        ("k1","rate constant k1"),
        ("k4","rate constant k4"),
        ("k2","rate constant k2"),
        ("k5","rate constant k5"),
        ("Starch","starch fraction (0.x if present)"),
        ("Grass (%)","forage grass share"),
        ("grain (%)","grain/concentrate share"),
        ("stover (%)","crop residue share"),
        ("occasional (%)","occasional feeds share"),
        ("_page","source PDF page"),
    ]
    return pd.DataFrame(rows, columns=["Column","Description"])

# =========================
# 1) Parse S10 & S4, assemble base sheets
# =========================
df_s10 = parse_table_s10(PDF_PATH, S10_START_PAGE, S10_END_PAGE)
df_s10 = fill_regions_with_wrd(df_s10)  # 区域缺失用WRD行补齐

df_s4 = parse_table_s4_mapping(PDF_PATH, page_no=S4_PAGE)
df_schema = build_schema_codebook()

# （可选）旧式 country_level
if GENERATE_OLD_COUNTRY_LEVEL:
    country_level = df_s10.merge(
        df_s4[["Region_S10","Country"]],
        left_on="Region", right_on="Region_S10", how="left"
    ).drop(columns=["Region_S10"])
else:
    country_level = None  # 不再生成

# =========================
# 2) Parse S11 & build weighted region means
# =========================
df_s11 = parse_table_s11(PDF_PATH, S11_START_PAGE, S11_END_PAGE)
w = (df_s11[df_s11["Subtype"].str.upper() != "ANY"]
       .groupby(["Species_System","Region","Subtype"], as_index=False)
       .agg(n_herd_1000=("n_herd_1000","sum")))

# 对 S10 region_level 的所有数值列做权重平均
key_cols = ["Species_System","Region","Subtype"]
num_cols = [c for c in df_s10.columns if c not in key_cols + ["_page"]]
m = df_s10.merge(w, how="left", on=key_cols, validate="m:1")
m["weight"] = m["n_herd_1000"].fillna(0.0)

def wavg(group, col):
    w = group["weight"].to_numpy()
    x = group[col].to_numpy(dtype=float)
    mask = (~np.isnan(x)) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])

weighted_rows = []
for (sp, rg), grp in m.groupby(["Species_System","Region"]):
    row = {"Species_System": sp, "Region": rg}
    for col in num_cols:
        row[col] = wavg(grp, col)
    weighted_rows.append(row)
df_weighted = pd.DataFrame(weighted_rows)

# WRD 回填：按同物种的 WRD 行补齐空值
df_weighted_wrd = df_weighted.copy()
for sp in df_weighted_wrd["Species_System"].unique():
    base = df_weighted_wrd[(df_weighted_wrd["Species_System"]==sp) & (df_weighted_wrd["Region"]=="WRD")]
    if base.empty: 
        continue
    wrd_vals = base.iloc[0][[c for c in df_weighted_wrd.columns if c not in ["Species_System","Region"]]]
    mask = (df_weighted_wrd["Species_System"]==sp) & (df_weighted_wrd["Region"]!="WRD")
    for col in wrd_vals.index:
        df_weighted_wrd.loc[mask & df_weighted_wrd[col].isna(), col] = wrd_vals[col]

# =========================
# 3) Build country_level_weighted (Item_Emis × M49)
#    需要工作簿中已有的 item_map & region_country_map
# =========================
# 这里假设两张映射表来自同一最终输出文件（若来自外部文件，可改路径）
# 为确保 M49 以字符串保留格式，dtype 指定为 str
item_map = pd.DataFrame({
    "Item_Emis": [], "Item_map": []
})
region_country_map = pd.DataFrame({
    "M49_Country_Code": [], "Region_map": []
})

# 如你已有“item_map / region_country_map”在其它xlsx，可在此读取；为便于一次成品，
# 也可以把它们临时写入再并回（此处假定你会在写出后手动补上或本地已有版本）。
# ——若你已有现成映射，替换为：
# item_map = pd.read_excel("your_mapping.xlsx", sheet_name="item_map", dtype={"Item_Emis":str, "Item_map":str})
# region_country_map = pd.read_excel("your_mapping.xlsx", sheet_name="region_country_map",
#                                    dtype={"M49_Country_Code":str, "Region_map":str})

# 若你希望本脚本直接从最终文件读取（二次运行场景），可在写文件后再读回追加；见下方“写文件”之后的二次合并段。

# =========================
# 4) 写出初版文件（不含 country_level_weighted；稍后二次打开合并）
# =========================
with pd.ExcelWriter(OUT_XLSX_FINAL, engine="xlsxwriter") as w:
    df_s10.to_excel(w, sheet_name="region_level", index=False)
    if GENERATE_OLD_COUNTRY_LEVEL and country_level is not None:
        country_level.to_excel(w, sheet_name="country_level", index=False)
    df_s4.to_excel(w, sheet_name="region_country_map", index=False)
    df_schema.to_excel(w, sheet_name="schema_codebook", index=False)
    df_weighted.to_excel(w, sheet_name="region_weighted_by_S11", index=False)
    df_weighted_wrd.to_excel(w, sheet_name="region_weighted_by_S11_WRDfill", index=False)
    # 预留空表，提醒后续需要 item_map / region_country_map
    if item_map.empty:
        pd.DataFrame(columns=["Item_Emis","Item_map"]).to_excel(w, sheet_name="item_map", index=False)

# =========================
# 5) 若“item_map / region_country_map”已存在于 OUT_XLSX_FINAL，则读取并生成 country_level_weighted
#    （这是针对你已有映射表的常见使用路径）
# =========================
wb = pd.read_excel(OUT_XLSX_FINAL, sheet_name=None)
if "item_map" in wb and "region_country_map" in wb:
    imap = wb["item_map"].copy()
    rmap = wb["region_country_map"].copy()
    # 保证关键列为字符串（M49 按原格式保留）
    if "M49_Country_Code" in rmap.columns:
        rmap["M49_Country_Code"] = rmap["M49_Country_Code"].astype(str)
    if "Item_Emis" in imap.columns:
        imap["Item_Emis"] = imap["Item_Emis"].astype(str)
    if "Item_map" in imap.columns:
        imap["Item_map"] = imap["Item_map"].astype(str)

    shares_sel = df_weighted_wrd[["Species_System","Region","Grass (%)","grain (%)","stover (%)","occasional (%)"]].copy()

    imap["_k"] = 1
    rmap["_k"] = 1
    cross = imap.merge(rmap, on="_k").drop(columns="_k")

    country_level_weighted = cross.merge(
        shares_sel, left_on=["Item_map","Region_map"],
        right_on=["Species_System","Region"], how="left"
    )
    out_cols = ["Item_Emis","Item_map","M49_Country_Code","Region_map","Grass (%)","grain (%)","stover (%)","occasional (%)"]
    country_level_weighted = country_level_weighted[out_cols].copy()

    # 回写：保留所有原有sheet，新增 country_level_weighted
    with pd.ExcelWriter(OUT_XLSX_FINAL, engine="xlsxwriter") as w:
        for name, df in wb.items():
            df.to_excel(w, sheet_name=name, index=False)
        country_level_weighted.to_excel(w, sheet_name="country_level_weighted", index=False)

print(f"[DONE] Wrote final workbook -> {OUT_XLSX_FINAL}")
