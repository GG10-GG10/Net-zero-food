
# -*- coding: utf-8 -*-
"""
S0_6_fill_fish_prices_reg5_reg2_world.py

用途：
- 读取 fish_price.xlsx（sheet 默认 'aquaculture_price'）与 dict_v3.xlsx（sheet 'region'）
- 先按 Region_agg5（洲）逐年均值填充缺失；若仍缺再按 Region_agg2 填充
- 然后用“向前滑动平均”（窗口 W，默认 3）把每个国家的序列回填到 2002 年
- 最后追加一行“World”汇总价（各年简单平均），并导出 CSV / XLSX

命令行示例：
    python S0_6_fill_fish_prices_reg5_reg2_world.py \
        --in /mnt/data/fish_price.xlsx \
        --sheet aquaculture_price \
        --dict /mnt/data/dict_v3.xlsx \
        --out_csv /mnt/data/fish_price_filled_reg5_reg2_backMA3_world_Y2002_2022.csv \
        --out_xlsx /mnt/data/fish_price_filled_reg5_reg2_backMA3_world_Y2002_2022.xlsx \
        --win 3
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_year_cols(df: pd.DataFrame, start=2002, end=2022):
    """Ensure Y{start}..Y{end} columns exist (inclusive)."""
    years = [f"Y{y}" for y in range(start, end+1)]
    for y in years:
        if y not in df.columns:
            df[y] = np.nan
    return years

def region_mean_fill(df: pd.DataFrame, years, region_col: str):
    """For each year column, fill NaN with mean by region_col (if region available)."""
    if region_col not in df.columns:
        return df
    for y in years:
        means = df.groupby(region_col)[y].mean()
        miss  = df[y].isna() & df[region_col].notna()
        if miss.any():
            df.loc[miss, y] = df.loc[miss].apply(lambda r: means.get(r[region_col], np.nan), axis=1)
    return df

def backward_moving_average(df: pd.DataFrame, years, W=3):
    """Descending years fill: for Y_t NaN, use mean(Y_{t+1..t+W}) if any exist."""
    years_sorted = sorted(years, key=lambda c: int(c[1:]))
    for i in range(len(years_sorted)-2, -1, -1):  # from (latest-2) down to earliest
        y = years_sorted[i]
        later_cols = years_sorted[i+1 : i+1+W]
        if not later_cols:
            continue
        later_mean = df[later_cols].mean(axis=1, skipna=True)
        fill_mask = df[y].isna() & later_mean.notna()
        df.loc[fill_mask, y] = later_mean[fill_mask]
    return df

def add_world_row(df: pd.DataFrame, years, meta_defaults=None):
    """Append a World row with simple mean across all rows for each year."""
    meta_defaults = meta_defaults or {}
    world_vals = df[years].mean(axis=0, skipna=True).to_dict()
    # Ensure metadata columns exist
    for c in ["Area Code","M49 Code_xxx","Region_label_new","Region","Region_agg5","Region_agg2"]:
        if c not in df.columns:
            df[c] = np.nan
    # Build world row
    world_row = {c: np.nan for c in df.columns}
    world_row.update({
        "Area Code": meta_defaults.get("Area Code", 5000),
        "M49 Code_xxx": meta_defaults.get("M49 Code_xxx", "001"),
        "Region_label_new": meta_defaults.get("Region_label_new", "no"),
        "Region": meta_defaults.get("Region", "World"),
        "Region_agg5": meta_defaults.get("Region_agg5", np.nan),
        "Region_agg2": meta_defaults.get("Region_agg2", np.nan),
    })
    for y in years:
        world_row[y] = world_vals.get(y, np.nan)
    df = pd.concat([df, pd.DataFrame([world_row])], ignore_index=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to fish_price.xlsx")
    ap.add_argument("--sheet", dest="sheet_name", default="aquaculture_price")
    ap.add_argument("--dict", dest="dict_path", required=True, help="Path to dict_v3.xlsx (region sheet)")
    ap.add_argument("--out_csv", dest="out_csv", required=True)
    ap.add_argument("--out_xlsx", dest="out_xlsx", required=False, default=None)
    ap.add_argument("--win", dest="win", type=int, default=3)
    args = ap.parse_args()

    in_path   = Path(args.in_path)
    dict_path = Path(args.dict_path)
    out_csv   = Path(args.out_csv)
    out_xlsx  = Path(args.out_xlsx) if args.out_xlsx else None
    W         = args.win

    # Read inputs
    fish   = pd.read_excel(in_path, sheet_name=args.sheet_name)
    region = pd.read_excel(dict_path, sheet_name="region")

    # Keep valid areas (exclude Region_label_new == 'no')
    region_valid = region[region["Region_label_new"].astype(str).str.lower() != "no"].copy()
    region_valid["_AreaCode_merge"] = pd.to_numeric(region_valid["Area Code"], errors="coerce").astype("Int64")

    # Map regions
    fish["_AreaCode_merge"] = pd.to_numeric(fish["Area Code"], errors="coerce").astype("Int64")
    cols_to_add = [c for c in ["Region_agg5","Region_agg2"] if c in region_valid.columns]
    fish = fish.merge(region_valid[["_AreaCode_merge"] + cols_to_add].drop_duplicates(),
                      on="_AreaCode_merge", how="left")

    # Ensure years
    years = ensure_year_cols(fish, start=2002, end=2022)

    # Reorder columns: meta first if present
    meta_pref = ["Area Code","M49 Code_xxx","Region_label_new","Region","Region_agg5","Region_agg2"]
    meta_cols = [c for c in meta_pref if c in fish.columns]
    fish = fish[meta_cols + years + [c for c in fish.columns if c not in meta_cols + years]]

    # (1) Region_agg5 mean fill, (2) then Region_agg2 mean fill
    fish = region_mean_fill(fish, years, "Region_agg5")
    if "Region_agg2" in fish.columns:
        # 仅对仍缺的单元格再尝试 Region_agg2
        miss_any = fish[years].isna().any(axis=1)
        if miss_any.any():
            fish2 = fish.loc[miss_any].copy()
            fish2 = region_mean_fill(fish2, years, "Region_agg2")
            fish.loc[miss_any, years] = fish2[years].values

    # (3) Backward moving-average to 2002
    fish = backward_moving_average(fish, years, W=W)

    # (4) Append World row
    fish = add_world_row(fish, years, meta_defaults={"Area Code": 5000, "M49 Code_xxx": "001", "Region_label_new": "no", "Region": "World"})

    # Sort and save
    if "Area Code" in fish.columns:
        fish = fish.sort_values("Area Code", kind="mergesort").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fish.to_csv(out_csv, index=False)

    if out_xlsx is not None:
        try:
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
                fish.to_excel(writer, sheet_name="filled", index=False)
        except Exception as e:
            print(f"[warn] XLSX export failed: {e}", file=sys.stderr)

    # Report
    nan_rate = float(fish[years].isna().mean().mean())
    print({"out_csv": str(out_csv), "out_xlsx": (None if out_xlsx is None else str(out_xlsx)),
           "rows": len(fish), "nan_share_all_years": nan_rate})

if __name__ == "__main__":
    main()
