
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from pathlib import Path

PRICES = Path("/mnt/data/Prices_E_All_Data_NOFLAG.csv")
FX     = Path("/mnt/data/Exchange_rate_E_All_Data_NOFLAG.csv")
DICT   = Path("/mnt/data/dict_v3.xlsx")
WORLD  = Path("/mnt/data/World_Production_Value_per_Unit.xlsx")
OUT    = Path("/mnt/data/Prices_with_USDtrans_USDfinal_SUPERSET_Y2002_2022.csv")

def norm(s: str) -> str:
    return (str(s).strip().replace("\\xa0"," ").replace("–","-").replace("  "," "))

def fx_direction(fx_df: pd.DataFrame) -> str:
    txt = (" ".join(fx_df.get("Unit","").astype(str).tolist()) + " " +
           " ".join(fx_df.get("Element","").astype(str).tolist())).lower()
    if "lcu per us" in txt or "lcu per usd" in txt or "lcu per us$" in txt: return "LCU_per_USD"
    if "usd per lcu" in txt or "us$ per lcu" in txt: return "USD_per_LCU"
    return "LCU_per_USD"

def to_usd(lcu, fxv, direction):
    if pd.isna(lcu) or pd.isna(fxv): return np.nan
    return (lcu / fxv) if direction=="LCU_per_USD" else (lcu * fxv)

def is_usd_per_tonne(elem: str) -> bool:
    s = str(elem).upper().replace(" ", "")
    return ("USD/TONNE" in s) or ("US$/TONNE" in s)

def is_lcu_per_tonne(elem: str) -> bool:
    s = str(elem).upper().replace(" ", "")
    return "LCU/TONNE" in s

def is_slc_per_tonne(elem: str) -> bool:
    s = str(elem).upper().replace(" ", "")
    return "SLC/TONNE" in s

# ---- Load
prices = pd.read_csv(PRICES)
fx     = pd.read_csv(FX)
region = pd.read_excel(DICT, sheet_name="region")
emis   = pd.read_excel(DICT, sheet_name="Emis_item")
fao    = pd.read_excel(DICT, sheet_name="FAO_Item").rename(columns=lambda x: str(x).strip())

# ---- Cols
cols = prices.columns.tolist()
year_cols_all = [c for c in cols if str(c).startswith("Y") and str(c)[1:].isdigit()]
year_cols = [c for c in year_cols_all if 2002 <= int(c[1:]) <= 2022]
year_cols.sort(key=lambda c: int(c[1:]))
area_code_col="Area Code"; area_m49_col="Area Code (M49)"; area_col="Area"
item_code_col="Item Code"; item_col="Item"
element_code_col="Element Code" if "Element Code" in cols else None
element_col="Element"; unit_col="Unit"; months_col="Months" if "Months" in cols else None

# ---- Region map
region = region[region["Region_label_new"].astype(str).str.lower()!="no"].copy()
region["_AreaCode_merge"] = pd.to_numeric(region["Area Code"], errors="coerce").astype("Int64")
region_map = region[["_AreaCode_merge","Region_agg5"]].drop_duplicates()

# ---- 38 targets
emis_items = [norm(x) for x in emis["Item_Price_Map"].dropna().astype(str).tolist() if norm(x).lower()!="no"]
four = ["Eggs Primary","Vegetables Primary","Fruit Primary","Fibre Crops, Fibre Equivalent"]
targets = sorted(set(emis_items).union(four))

# ---- USD_trans
prices["_unit_norm"] = prices[unit_col].astype(str).str.upper().str.strip()
prices["_AreaCode_merge"] = pd.to_numeric(prices[area_code_col], errors="coerce").astype("Int64")
subset = prices[[area_code_col, area_m49_col, area_col, item_code_col, item_col, element_code_col, element_col, unit_col] + year_cols + ["_unit_norm","_AreaCode_merge"]].copy()
long0 = subset.melt(id_vars=[area_code_col, area_m49_col, area_col, item_code_col, item_col, element_code_col, element_col, unit_col,"_unit_norm","_AreaCode_merge"],
                    value_vars=year_cols, var_name="Year", value_name="Value")
idx_cols = [area_code_col, area_m49_col, area_col, item_code_col, item_col, element_code_col, element_col,"_AreaCode_merge","Year"]
slc_series = long0[long0["_unit_norm"]=="SLC"].set_index(idx_cols)["Value"]
lcu_series = long0[long0["_unit_norm"]=="LCU"].set_index(idx_cols)["Value"]
tuples = sorted(set(slc_series.index).union(lcu_series.index))
mi = pd.MultiIndex.from_tuples(tuples, names=idx_cols)
comb = (mi.to_frame(index=False)
          .merge(slc_series.rename("SLC").reset_index(), on=idx_cols, how="left")
          .merge(lcu_series.rename("LCU").reset_index(), on=idx_cols, how="left"))
fx["_AreaCode_merge"] = pd.to_numeric(fx.get("Area Code", fx.get("Area")), errors="coerce").astype("Int64")
fx_long = fx.melt(id_vars=["_AreaCode_merge"], value_vars=[c for c in fx.columns if str(c).startswith("Y")], var_name="Year", value_name="FX")
fx_long = fx_long[fx_long["Year"].isin(year_cols)]
comb = comb.merge(fx_long, on=["_AreaCode_merge","Year"], how="left")
direction = fx_direction(fx)
comb["USD_trans"] = comb.apply(lambda r: r["SLC"] if pd.notna(r["SLC"]) else to_usd(r["LCU"], r["FX"], direction), axis=1)
usd_trans_rows = comb[pd.notna(comb["USD_trans"])]
usd_trans_wide = usd_trans_rows.pivot_table(index=[area_code_col, area_m49_col, area_col, item_code_col, item_col, element_code_col, element_col],
                                            columns="Year", values="USD_trans", aggfunc="first").reset_index()
usd_trans_wide[unit_col] = "USD_trans"
if months_col: usd_trans_wide[months_col] = "Annual value"
need_cols = prices.columns.tolist()
for c in need_cols:
    if c not in usd_trans_wide.columns and (c in year_cols): continue
    if c not in usd_trans_wide.columns: usd_trans_wide[c]=np.nan
usd_trans_wide = usd_trans_wide[need_cols]

# ---- USD_final base (ALL items)
prices_aug = pd.concat([prices, usd_trans_wide], ignore_index=True, sort=False)
prices_aug["_AreaCode_merge"] = pd.to_numeric(prices_aug[area_code_col], errors="coerce").astype("Int64")
prices_aug = prices_aug.merge(region_map[["_AreaCode_merge","Region_agg5"]], on="_AreaCode_merge", how="left")
pp_mask = prices_aug[element_col].astype(str).str.contains("Producer Price", case=False, na=False) & prices_aug[element_col].astype(str).str.contains("/tonne", case=False, na=False)

pp_usd = pp_mask & prices_aug[element_col].astype(str).apply(is_usd_per_tonne) & (prices_aug[unit_col].astype(str).str.upper()=="USD")
pp_lcu = pp_mask & prices_aug[element_col].astype(str).apply(is_lcu_per_tonne) & (prices_aug[unit_col]=="USD_trans")
pp_slc = pp_mask & prices_aug[element_col].astype(str).apply(is_slc_per_tonne) & (prices_aug[unit_col]=="USD_trans")

def to_long(df, tag):
    if df.empty:
        return pd.DataFrame(columns=[area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5","Year","Value","src"])
    L = df.melt(id_vars=[area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5", unit_col],
                value_vars=year_cols, var_name="Year", value_name="Value").drop(columns=[unit_col])
    L["src"]=tag
    return L

L_usd = to_long(prices_aug.loc[pp_usd, [area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5", unit_col] + year_cols], "USD")
L_lcu = to_long(prices_aug.loc[pp_lcu, [area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5", unit_col] + year_cols], "USD_LCUtrans")
L_slc = to_long(prices_aug.loc[pp_slc, [area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5", unit_col] + year_cols], "USD_SLCtrans")
pp_all = pd.concat([L_usd, L_lcu, L_slc], ignore_index=True)

base_keys = [area_code_col, area_m49_col, area_col, item_code_col, item_col, "Region_agg5","Year"]
base = pp_all[base_keys].drop_duplicates()
for nm, L in [("v_usd", L_usd), ("v_lcu", L_lcu), ("v_slc", L_slc)]:
    tmp = L[base_keys+["Value"]].copy(); tmp = tmp.rename(columns={"Value": nm});
    base = base.merge(tmp, on=base_keys, how="left")

def pick(a,b,c):
    if pd.notna(a): return a
    if pd.notna(b): return b
    if pd.notna(c): return c
    return np.nan

base["USD_final"] = base.apply(lambda r: pick(r["v_usd"], r["v_lcu"], r["v_slc"]), axis=1)

# Same-continent mean fill
mean_reg = (base.dropna(subset=["USD_final"])
                 .groupby(["Region_agg5", item_col, "Year"])["USD_final"]
                 .mean().reset_index().rename(columns={"USD_final":"MeanReg"}))
base = base.merge(mean_reg, on=["Region_agg5", item_col, "Year"], how="left")
mnan = base["USD_final"].isna()
base.loc[mnan, "USD_final"] = base.loc[mnan, "MeanReg"]

# Sticky USD
usd_only = L_usd.rename(columns={"Value":"USD_only"})[base_keys+["USD_only"]]
base = base.merge(usd_only, on=base_keys, how="left")
mask = base["USD_final"].isna() & base["USD_only"].notna()
base.loc[mask, "USD_final"] = base.loc[mask, "USD_only"]

usd_final_all = base.pivot_table(index=[area_code_col, area_m49_col, area_col, item_code_col, item_col],
                                 columns="Year", values="USD_final", aggfunc="first").reset_index()
usd_final_all[unit_col]="USD_final"
if months_col: usd_final_all[months_col]="Annual value"
usd_final_all[element_col]="Producer Price (USD/tonne)"

# ---- Four groups via FAO membership
groups = ["Eggs Primary","Vegetables Primary","Fruit Primary","Fibre Crops, Fibre Equivalent"]
members = fao[fao["Item Group"].astype(str).isin(groups)][["Item","Item Code","Item Group"]].rename(columns={"Item Group":"GroupItem"})
U_long = usd_final_all.melt(id_vars=[area_code_col, area_m49_col, area_col, item_code_col, item_col, unit_col] + ([months_col] if months_col else []) + ([element_col] if element_col else []),
                            value_vars=year_cols, var_name="Year", value_name="Val")
U_long = U_long.merge(members.rename(columns={"Item":"MemberItem"}), left_on=item_col, right_on="MemberItem", how="inner")
grp = U_long.groupby([area_code_col, area_m49_col, area_col, "GroupItem","Year"])["Val"].mean().reset_index()
grp_wide = grp.pivot_table(index=[area_code_col, area_m49_col, area_col, "GroupItem"], columns="Year", values="Val", aggfunc="first").reset_index()
grp_wide = grp_wide.rename(columns={"GroupItem": item_col})
# codes for group items
group_code_map = fao.set_index("Item")["Item Code"].to_dict()
grp_wide[item_code_col] = grp_wide[item_col].map(group_code_map)
grp_wide[unit_col]="USD_final"
if months_col: grp_wide[months_col]="Annual value"
grp_wide[element_col]="Producer Price (USD/tonne)"
usd_final_all = pd.concat([usd_final_all, grp_wide], ignore_index=True, sort=False)

# ---- TARGET skeleton & reuse
allowed_areas = set(region["Area Code"].astype(int).tolist())
areas = (prices[[area_code_col, area_m49_col, area_col]].drop_duplicates()
          .loc[lambda d: d[area_code_col].astype(int).isin(allowed_areas)])
areas["key"]=1
items_df = pd.DataFrame({item_col: targets}); items_df["key"]=1
skel = areas.merge(items_df, on="key").drop(columns=["key"])
# item codes
code_map = {}
tmp = pd.concat([fao[["Item","Item Code"]].rename(columns={"Item":item_col,"Item Code":item_code_col}), usd_final_all[[item_col,item_code_col]]], ignore_index=True).dropna().drop_duplicates()
for _,r in tmp.iterrows(): code_map[norm(r[item_col])] = r[item_code_col]
skel[item_code_col] = skel[item_col].map(lambda x: code_map.get(norm(x), np.nan)).astype("float64")
skel[unit_col]="USD_final"
if months_col: skel[months_col]="Annual value"
skel[element_col]="Producer Price (USD/tonne)"
for y in year_cols: skel[y]=np.nan

merge_keys = [area_code_col, area_m49_col, area_col, item_code_col, item_col, unit_col] + ([months_col] if months_col else []) + ([element_col] if element_col else [])
sub_target = usd_final_all[usd_final_all[item_col].isin(targets)].copy()
m = skel.merge(sub_target, on=merge_keys, how="left", suffixes=("","_exist"))
for y in year_cols:
    col_e = f"{y}_exist"
    if col_e in m.columns:
        m[y] = m[y].where(~m[col_e].notna(), m[col_e])
        m.drop(columns=[col_e], inplace=True)

# ---- World × PPI fallback; then global mean fallback
wx = pd.ExcelFile(WORLD)
wdf = pd.read_excel(WORLD, sheet_name=wx.sheet_names[0])
unit_norm = wdf["Unit"].astype(str).str.replace("-", "–").str.strip()
mask_world = (wdf["Area"].astype(str).str.strip().str.lower()=="world") & (unit_norm=="Int$ (2014–2016 const) per tonne")
wdf = wdf.loc[mask_world].copy()
y1416 = [c for c in wdf.columns if c.startswith("Y") and c[1:].isdigit() and 2014 <= int(c[1:]) <= 2016]
wdf["World_IntConst_mean_2014_2016"] = wdf[y1416].astype(float).mean(axis=1, skipna=True)
world_mean = wdf[["Item","World_IntConst_mean_2014_2016"]].copy()

ppi_mask = prices[element_col].astype(str).str.contains("Producer Price Index (2014-2016 = 100)", case=False, na=False)
ppi = prices.loc[ppi_mask, [area_code_col, item_col] + year_cols].copy()

val_long = m.melt(id_vars=merge_keys, value_vars=year_cols, var_name="Year", value_name="Val")
val_long = val_long.merge(world_mean, on=item_col, how="left")
ppi_long = ppi.melt(id_vars=[area_code_col, item_col], value_vars=year_cols, var_name="Year", value_name="PPI")
val_long = val_long.merge(ppi_long, on=[area_code_col, item_col, "Year"], how="left")

mna = val_long["Val"].isna()
w = val_long["World_IntConst_mean_2014_2016"]; p = val_long["PPI"]
val_long.loc[mna & w.notna() & p.notna(), "Val"] = (w[mna & w.notna() & p.notna()] * p[mna & w.notna() & p.notna()] / 100.0)
val_long.loc[mna & w.notna() & p.isna(),  "Val"] = w[mna & w.notna() & p.isna()]

global_mean = (val_long.dropna(subset=["Val"]).groupby([item_col,"Year"])["Val"].mean().reset_index().rename(columns={"Val":"GlobalMean"}))
val_long = val_long.merge(global_mean, on=[item_col,"Year"], how="left")
mna2 = val_long["Val"].isna() & val_long["GlobalMean"].notna()
val_long.loc[mna2, "Val"] = val_long.loc[mna2, "GlobalMean"]

target_block = val_long.pivot_table(index=merge_keys, columns="Year", values="Val", aggfunc="first").reset_index()
keys_df = m[merge_keys].drop_duplicates()
target_block = keys_df.merge(target_block, on=merge_keys, how="left")

# ---- SUPerset output
base_all = pd.concat([prices, usd_trans_wide, usd_final_all], ignore_index=True, sort=False)
mask_rm = (base_all[unit_col]=="USD_final") & (base_all[item_col].isin(targets))
base_all = base_all.loc[~mask_rm].copy()
base_all = pd.concat([base_all, target_block], ignore_index=True, sort=False)

# Keep only 2002–2022
drop_cols = [c for c in base_all.columns if str(c).startswith("Y") and str(c)[1:].isdigit() and int(c[1:])>2022]
if drop_cols: base_all = base_all.drop(columns=drop_cols)

sort_keys = [c for c in ["Area Code","Item","Unit"] if c in base_all.columns]
base_all = base_all.sort_values(sort_keys, ascending=True, kind="mergesort").reset_index(drop=True)
base_all.to_csv(OUT, index=False)
print({"out_path": str(OUT), "rows": len(base_all)})
