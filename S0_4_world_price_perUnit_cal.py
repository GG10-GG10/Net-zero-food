# -*- coding: utf-8 -*-
"""
Compute 'Production Value per Unit' for World (FAOSTAT) with correct units:
- Value: parse thousand/million/billion + currency (US$ vs I$) + basis (constant 2014–2016 vs current)
- Per-unit = (Gross Production Value in currency units) / (Production in tonnes)
- Output columns: Area Code, Area Code (M49), Area, Item, Element, Unit, Y2010..Y2021
- Files:
    /mnt/data/Value_of_Production_E_All_Data_NOFLAG.csv
    /mnt/data/Production_Crops_Livestock_E_All_Data_NOFLAG.csv
    -> /mnt/data/World_Production_Value_per_Unit.csv
"""

import pandas as pd
import numpy as np
import re

VALUE_PATH = "../../input/Price_Cost/Price/Value_of_Production_E_All_Data_NOFLAG.csv"
PROD_PATH  = "../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
OUT_PATH   = "../../input/Price_Cost/Price/World_Production_Value_per_Unit.csv"

# ---- 1) Read
value_df = pd.read_csv(VALUE_PATH, low_memory=False)
prod_df  = pd.read_csv(PROD_PATH,  low_memory=False)

# ---- 2) Filter: World + required Elements
# Value: any Element containing 'Gross Production Value' (covers constant/current USD/I$ variants)
value_world = value_df[
    (value_df["Area"] == "World") &
    (value_df["Element"].str.contains("Gross Production Value", case=False, na=False))
].copy()

# Production: Element exactly 'Production' (tonnes)
prod_world = prod_df[
    (prod_df["Area"] == "World") &
    (prod_df["Element"].str.fullmatch(r"Production", case=False, na=False))
].copy()

# Years window
year_cols = [f"Y{y}" for y in range(2010, 2022)]

def keep_and_numeric(df, keep_cols, years):
    cols = [c for c in keep_cols if c in df.columns] + [c for c in years if c in df.columns]
    out = df[cols].copy()
    for c in years:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

key_cols = ["Area Code", "Area Code (M49)", "Area", "Item"]
val_keep = key_cols + ["Element", "Unit"]
prod_keep = key_cols + ["Element", "Unit"]

value_world_n = keep_and_numeric(value_world, val_keep, year_cols)
prod_world_n  = keep_and_numeric(prod_world,  prod_keep,  year_cols)

# Suffix years to avoid collision
value_world_n = value_world_n.rename(columns={c: f"{c}__VAL" for c in year_cols if c in value_world_n.columns})
prod_world_n  = prod_world_n.rename(columns={c: f"{c}__PROD" for c in year_cols if c in prod_world_n.columns})

# ---- 3) Merge on keys
merge_on = [c for c in key_cols if c in value_world_n.columns and c in prod_world_n.columns]
merged = value_world_n.merge(prod_world_n, on=merge_on, how="inner", validate="many_to_many")

# ---- 4) Parse Value units (scale + currency + basis)
# scale: thousand->1e3, million->1e6, billion->1e9
# currency: 'USD' or 'Int$' (PPP)
# basis: '2014–2016 const' or 'current' (or None if not detectable)
def parse_value_unit(element_text, unit_text):
    t = f"{str(element_text)} {str(unit_text)}"
    t_low = t.lower()

    # scale
    scale = 1.0
    if "thousand" in t_low:
        scale = 1e3
    elif "million" in t_low:
        scale = 1e6
    elif "billion" in t_low:
        scale = 1e9

    # currency
    currency = "USD"
    # detect variants of I$: "I$", "international $/dollars"
    if re.search(r"\bi\$|\binternational\b", t_low):
        currency = "Int$"

    # basis
    basis = None
    if "constant 2014-2016" in t_low or "constant 2014–2016" in t_low:
        basis = "2014–2016 const"
    elif "current" in t_low:
        basis = "current"

    return scale, currency, basis

def build_output_unit(currency, basis):
    return f"{currency} ({basis}) per tonne" if basis else f"{currency} per tonne"

parsed = merged.apply(lambda r: parse_value_unit(r.get("Element_x", ""), r.get("Unit_x", "")), axis=1)
merged["_scale"]    = [p[0] for p in parsed]
merged["_currency"] = [p[1] for p in parsed]
merged["_basis"]    = [p[2] for p in parsed]

# ---- 5) Compute per-unit by year
out = merged[merge_on].copy()
for y in year_cols:
    vy = f"{y}__VAL"
    py = f"{y}__PROD"
    if vy in merged.columns and py in merged.columns:
        numer = merged[vy].astype(float) * merged["_scale"].astype(float)   # convert thousand/million/billion to 1:1 currency
        denom = merged[py].astype(float)                                     # tonnes
        out[y] = np.where((denom > 0) & np.isfinite(denom), numer / denom, np.nan)

# ---- 6) Labels & ordering
out["Element"] = "Production Value per Unit"
out["Unit"] = [build_output_unit(c, b) for c, b in zip(merged["_currency"], merged["_basis"])]

# backfill id columns if needed
for col in ["Area Code", "Area Code (M49)", "Area"]:
    if col not in out.columns and col in merged.columns:
        out[col] = merged[col]

final_cols = ["Area Code", "Area Code (M49)", "Area", "Item", "Element", "Unit"] + year_cols
final_cols = [c for c in final_cols if c in out.columns]
out = out[final_cols].copy()

# Optional: sort
sort_cols = [c for c in ["Area Code", "Item", "Unit"] if c in out.columns]
if sort_cols:
    out = out.sort_values(sort_cols, kind="mergesort")

# ---- 7) Save
out.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")

