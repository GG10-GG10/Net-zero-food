# -*- coding: utf-8 -*-
"""
Build world unit values (USD per m³) for Industrial roundwood & Wood fuel
from a FAOSTAT Forestry_E_All_Data_NOFLAG.csv (wide format).

Outputs columns:
  product, year, uv_export_usd_per_m3, uv_import_usd_per_m3, uv_avg_usd_per_m3,
  export_qty_m3, export_value_1000usd, import_qty_m3, import_value_1000usd

Usage:
  python build_world_price_from_NOFLAG.py --input /path/Forestry_E_All_Data_NOFLAG.csv \
      --output /path/world_price_roundwood_woodfuel_nominalUSD_from_NOFLAG.csv

Notes:
- Excludes 'World'、各大洲、EU 等聚合区域，仅汇总国家。
- 价格用“值/量”计算：分别基于出口与进口（单位：USD/m³），并提供均值（不是加权）；值单位应为“1000 USD/US$”，量单位应为“m3/m³”。
- 文件为 NOFLAG 宽表（Y1961–Y2023），脚本会自动 melt 成长表。
"""

import re
import argparse
import pandas as pd
import numpy as np

AGGREGATE_AREAS = {
    "World","Africa","Americas","Asia","Europe","Oceania",
    "European Union (27)","European Union (28)","EU-27","EU-28",
    "Western Africa","Eastern Africa","Northern Africa","Middle Africa","Southern Africa",
    "Northern America","Latin America and the Caribbean","Central America","Caribbean",
    "Eastern Asia","Southern Asia","South-eastern Asia","Western Asia","Central Asia",
    "Northern Europe","Southern Europe","Eastern Europe","Western Europe",
    "Australia and New Zealand","Melanesia","Micronesia","Polynesia",
    "Land Locked Developing Countries","Least Developed Countries","Small Island Developing States"
}

DEFAULT_PRODUCTS = ["Industrial roundwood","Wood fuel"]

def load_wide_and_melt(fp: str, products=DEFAULT_PRODUCTS) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    # Columns like: Area, Item, Element, Unit, Y1961...Y2023
    id_cols = ["Area","Item","Element","Unit"]
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"Missing expected column '{c}' in {fp}")
    ycols = [c for c in df.columns if re.fullmatch(r"Y\d{4}", str(c))]
    if not ycols:
        raise ValueError("This file doesn't look like a NOFLAG wide file (no Y#### columns).")

    # Filter
    df = df[df["Item"].isin(products)]
    df = df[~df["Area"].isin(AGGREGATE_AREAS)]

    # Melt
    long = df.melt(id_vars=id_cols, value_vars=ycols, var_name="Year", value_name="Value")
    long["Year"] = long["Year"].str.replace("Y","", regex=False).astype(int)
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

    # Normalize strings
    long["el"] = long["Element"].astype(str).str.lower()
    long["unit_l"] = long["Unit"].astype(str).str.strip().str.lower()
    return long

def build_world_uv(long: pd.DataFrame) -> pd.DataFrame:
    # Masks for units
    m3 = long["unit_l"].isin(["m3","m³"])
    usd1000 = long["unit_l"].str.contains("1000", na=False) & long["unit_l"].str.contains("usd", na=False)

    iq = long[(long["el"].eq("import quantity")) & m3][["Item","Year","Value"]] \
            .groupby(["Item","Year"], as_index=False)["Value"].sum().rename(columns={"Value":"import_qty_m3"})
    iv = long[(long["el"].eq("import value")) & usd1000][["Item","Year","Value"]] \
            .groupby(["Item","Year"], as_index=False)["Value"].sum().rename(columns={"Value":"import_value_1000usd"})
    eq = long[(long["el"].eq("export quantity")) & m3][["Item","Year","Value"]] \
            .groupby(["Item","Year"], as_index=False)["Value"].sum().rename(columns={"Value":"export_qty_m3"})
    ev = long[(long["el"].eq("export value")) & usd1000][["Item","Year","Value"]] \
            .groupby(["Item","Year"], as_index=False)["Value"].sum().rename(columns={"Value":"export_value_1000usd"})

    tmp = iq.merge(iv, on=["Item","Year"], how="outer") \
            .merge(eq, on=["Item","Year"], how="outer") \
            .merge(ev, on=["Item","Year"], how="outer")

    # Compute USD/m3
    tmp["uv_import_usd_per_m3"] = (tmp["import_value_1000usd"]*1000.0) / tmp["import_qty_m3"]
    tmp["uv_export_usd_per_m3"] = (tmp["export_value_1000usd"]*1000.0) / tmp["export_qty_m3"]
    tmp["uv_avg_usd_per_m3"] = tmp[["uv_import_usd_per_m3","uv_export_usd_per_m3"]].mean(axis=1, skipna=True)

    out = tmp.rename(columns={"Item":"product","Year":"year"}).sort_values(["product","year"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to FAOSTAT Forestry_E_All_Data_NOFLAG.csv")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--products", nargs="*", default=DEFAULT_PRODUCTS, help="Items to include (exact Item names)")
    args = ap.parse_args()

    long = load_wide_and_melt(args.input, products=args.products)
    out = build_world_uv(long)

    # Column order
    cols = ["product","year",
            "uv_export_usd_per_m3","uv_import_usd_per_m3","uv_avg_usd_per_m3",
            "export_qty_m3","export_value_1000usd","import_qty_m3","import_value_1000usd"]
    out = out[cols]
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}  (rows={len(out)})")

if __name__ == "__main__":
    main()
