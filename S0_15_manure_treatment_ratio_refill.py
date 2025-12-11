# -*- coding: utf-8 -*-
"""
1) Read Environment_LivestockManure_E_All_Data_NOFLAG.csv and dict_v3.xlsx['region'].
2) Compute ratios (management/applied/left_pasture). Missing values: row bfill/ffill -> Region_agg2->Item mean -> global Item mean.
3) If left_pasture + management > 1, clamp left_pasture = 1 - management and log the correction.
4) Append ratio rows (Unit='ratio'), sorted by Area->Item.
5) Outputs: CSV + Excel log (warnings/corrections).
"""
import pandas as pd
import numpy as np
from pathlib import Path

env_path  = Path("../../input/Manure_Stock/retired-unused-raw/Environment_LivestockManure_E_All_Data_NOFLAG.csv")
dict_path = Path("../../src/dict_v3.xlsx")

out_csv   = Path("../../input/Manure_Stock/Environment_LivestockManure_with_ratios_v2.csv")
logs_xlsx = Path("../../input/Manure_Stock/retired-unused-raw/Environment_LivestockManure_ratios_logs.xlsx")

df = pd.read_csv(env_path, dtype={"M49_Country_Code": str})
region_map = pd.read_excel(dict_path, sheet_name="region", dtype={"M49_Country_Code": str})
if "Region_label_new" in region_map.columns:
    region_map = region_map[region_map["Region_label_new"] != "no"].copy()

id_cols = ["M49_Country_Code","Area","Item Code","Item Code (CPC)","Item"]
req_cols = id_cols + ["Element","Unit"]
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise KeyError(f"missing columns in the source CSV: {missing}")

year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Y") and c[1:].isdigit()]
year_cols = sorted(year_cols, key=lambda s: int(s[1:]))

def sel(elem_name: str) -> pd.DataFrame:
    """Return subset for the given element (ID columns + year columns)."""
    return df[df["Element"] == elem_name][id_cols + year_cols].copy()

E_EXCRETED = "Amount excreted in manure (N content)"
E_MANAGED  = "Manure management (manure treated, N content)"
E_APPLIED  = "Manure applied to soils (N content)"
E_PASTURE  = "Manure left on pasture (N content)"

excreted = sel(E_EXCRETED)
managed  = sel(E_MANAGED)
applied  = sel(E_APPLIED)
pasture  = sel(E_PASTURE)

def safe_div(num_df: pd.DataFrame, den_df: pd.DataFrame) -> pd.DataFrame:
    """Align IDs and divide year-by-year with guards for zero denominators."""
    m = num_df.merge(den_df, on=id_cols, how="outer", suffixes=("_n","_d"))
    out = m[id_cols].copy()
    for yc in year_cols:
        den = m[f"{yc}_d"].replace(0, np.nan)
        out[yc] = (m[f"{yc}_n"] / den).replace([np.inf, -np.inf], np.nan)
    return out

mm_ratio = safe_div(managed, excreted)  # management / excreted
ap_ratio = safe_div(applied, managed)   # applied / management
lp_ratio = safe_div(pasture, excreted)  # left_pasture / excreted

for d in (mm_ratio, ap_ratio, lp_ratio):
    d[year_cols] = d[year_cols].bfill(axis=1).ffill(axis=1)

# ---------- Region / global fallback for all-NaN rows ----------
if "Region_agg2" not in region_map.columns:
    raise KeyError("dict_v3['region'] is missing Region_agg2")
map_keep = region_map[["M49_Country_Code","Region_agg2"]].copy()

def fill_all_nan_rows(ratio_df: pd.DataFrame) -> pd.DataFrame:
    x = ratio_df.merge(map_keep, on="M49_Country_Code", how="left")
    have_vals = ~x[year_cols].isna().all(axis=1)

    reg_means = (
        x.loc[have_vals]
         .groupby(["Region_agg2","Item"], dropna=False)[year_cols]
         .mean(numeric_only=True)
         .reset_index()
         .rename(columns={c: f"{c}_regmean" for c in year_cols})
    )
    glob_means = (
        x.loc[have_vals]
         .groupby(["Item"], dropna=False)[year_cols]
         .mean(numeric_only=True)
         .reset_index()
         .rename(columns={c: f"{c}_globmean" for c in year_cols})
    )

    all_nan_mask = x[year_cols].isna().all(axis=1)
    if all_nan_mask.any():
        sub = (x.loc[all_nan_mask]
                 .merge(reg_means, on=["Region_agg2","Item"], how="left")
                 .merge(glob_means, on=["Item"],   how="left"))
        for yc in year_cols:
            v = sub[f"{yc}_regmean"]
            v = v.where(~v.isna(), sub[f"{yc}_globmean"])
            sub[yc] = v
            del sub[f"{yc}_regmean"]; del sub[f"{yc}_globmean"]
        x.loc[all_nan_mask, year_cols] = sub[year_cols].values

    return x.drop(columns=["Region_agg2"])

mm_ratio = fill_all_nan_rows(mm_ratio)
ap_ratio = fill_all_nan_rows(ap_ratio)
lp_ratio = fill_all_nan_rows(lp_ratio)

mm_idx = mm_ratio.set_index(id_cols)
lp_idx = lp_ratio.set_index(id_cols)

corr_records = []
for yc in year_cols:
    s_mm, s_lp = mm_idx[yc], lp_idx[yc]
    over1 = (s_mm + s_lp) > 1
    if over1.any():
        corr_df = (
            pd.DataFrame(index=s_lp.index[over1])
              .assign(old_lp_ratio=s_lp.loc[over1].values,
                      mm_ratio=s_mm.loc[over1].values,
                      new_lp_ratio=(1 - s_mm.loc[over1]).values)
              .reset_index()
        )
        corr_df["year"] = yc
        corr_records.append(corr_df)
        lp_idx.loc[over1, yc] = (1 - s_mm.loc[over1])

lp_ratio = lp_idx.reset_index()
corrections_df = (
    pd.concat(corr_records, ignore_index=True)
    if corr_records else
    pd.DataFrame(columns=id_cols + ["old_lp_ratio","mm_ratio","new_lp_ratio","year"])
)

def to_rows(ratio_df: pd.DataFrame, element_name: str) -> pd.DataFrame:
    out = ratio_df.copy()
    out["Element"] = element_name
    out["Unit"] = "ratio"
    out["Element Code"] = np.nan
    return out[id_cols + ["Element Code","Element","Unit"] + year_cols]

mm_rows = to_rows(mm_ratio, "Manure management ratio")
ap_rows = to_rows(ap_ratio, "Manure applied ratio")
lp_rows = to_rows(lp_ratio, "Manure left_pasture ratio")

df_out = pd.concat([df, mm_rows, ap_rows, lp_rows], ignore_index=True)
df_out = df_out.sort_values(["Area","Item"]).reset_index(drop=True)

warnings_list = []
ap_gt1 = (ap_rows[year_cols] > 1).any(axis=1)
if ap_gt1.any():
    w1 = ap_rows.loc[ap_gt1, ["M49_Country_Code","Area","Item"]].drop_duplicates().copy()
    w1["warning"] = "Manure applied ratio > 1"
    warnings_list.append(w1)

warnings_df = (
    pd.concat(warnings_list, ignore_index=True)
    if warnings_list else
    pd.DataFrame(columns=["M49_Country_Code","Area","Item","warning"])
)

df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(logs_xlsx, engine="openpyxl") as writer:
    warnings_df.to_excel(writer, sheet_name="warnings",   index=False)
    corrections_df.to_excel(writer, sheet_name="corrections", index=False)

print({
    "result_csv": str(out_csv),
    "logs_workbook": str(logs_xlsx),
    "n_rows_result": int(df_out.shape[0]),
    "n_warning_rows": int(warnings_df.shape[0]),
    "n_corrections": int(corrections_df.shape[0])
})
