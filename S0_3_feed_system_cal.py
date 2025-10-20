
# feedreq_v3.py
# - Uses dict_v3.xlsx (sheet 'region') for country -> region mappings
# - Filters to countries where Region_label_new != 'no' (your 199 target; in this file it's 198 rows)
# - Attaches Region_label_new and Region_agg1..Region_agg5 to outputs
# - Keeps system shares / rations parameterization as in v2 (region heuristics) but *labels* follow dict_v3.xlsx

import pandas as pd, numpy as np
from pathlib import Path
from config_paths import get_input_base

YEARS = list(range(2010, 2021))
YCOLS  = [f"Y{y}" for y in YEARS]

ITEM_TO_CROP = {
    'Wheat and products': 'Wheat',
    'Maize and products': 'Maize',
    'Barley and products': 'Barley',
    'Sorghum and products': 'Sorghum',
    'Rice and products': 'Rice',
    'Millet and products': 'Millet',
    'Cassava and products': 'Cassava',
    'Soyabeans': 'Soybean',
}
ITEMS = list(ITEM_TO_CROP.keys())

SPECIES_LIST = ['layers','broilers','ducks','turkeys','geese_guinea','pigs',
                'dairy_cattle','beef_cattle','buffalo','sheep','goat']

SYSTEMS_BY_SPECIES = {
    'layers': ['industrial','backyard'],
    'broilers': ['industrial','backyard'],
    'ducks': ['industrial','backyard'],
    'turkeys': ['industrial','backyard'],
    'geese_guinea': ['industrial','backyard'],
    'pigs': ['industrial','intermediate','backyard'],
    'dairy_cattle': ['grassland','mixed','feedlot'],
    'beef_cattle': ['grassland','mixed','feedlot'],
    'buffalo': ['grassland','mixed'],
    'sheep': ['grassland','mixed'],
    'goat': ['grassland','mixed'],
}

GRASS_FRAC_BY_SYSTEM = {
    'grassland': 0.85,
    'mixed':     0.50,
    'feedlot':   0.20,
    'industrial':0.00,
    'intermediate':0.00,
    'backyard':  0.00,
}

def to_head_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    if 'Unit' in out.columns and out['Unit'].nunique()==1 and out['Unit'].iloc[0]=='1000 An':
        for y in YEARS:
            col = f"Y{y}"
            if col in out.columns: out[col] = out[col] * 1000.0
    return out.drop(columns=['Unit'], errors='ignore')

def auto_region(area: str) -> str:
    a = (area or '').lower()
    NA   = {'united states of america','canada','greenland'}
    WE   = {'germany','france','italy','spain','netherlands','belgium','austria','switzerland','united kingdom','ireland','norway','sweden','finland','denmark','portugal','greece','luxembourg','iceland'}
    EE   = {'poland','czechia','hungary','romania','bulgaria','slovakia','slovenia','croatia','serbia','bosnia and herzegovina','albania','north macedonia','belarus','ukraine','moldova','lithuania','latvia','estonia','montenegro'}
    RUSS = {'russian federation'}
    ESEA = {'china, mainland','japan','republic of korea','democratic people\'s republic of korea','viet nam','thailand','indonesia','malaysia','philippines','cambodia','myanmar','lao people\'s democratic republic','mongolia','timor-leste','singapore','brunei darussalam'}
    SA   = {'india','pakistan','bangladesh','nepal','sri lanka','bhutan','maldives','afghanistan'}
    LAC  = {'brazil','argentina','mexico','chile','peru','colombia','ecuador','bolivia (plurinational state of)','paraguay','uruguay','costa rica','panama','guatemala','honduras','el salvador','nicaragua','cuba','haiti','dominican republic','jamaica','trinidad and tobago','bahamas','barbados','guyana','suriname','belize','grenada','saint lucia','saint vincent and the grenadines','antigua and barbuda','dominica','saint kitts and nevis'}
    SSA  = {'nigeria','ethiopia','kenya','tanzania, united republic of','south africa','ghana','uganda',"cote d'ivoire",'cameroon','senegal','burkina faso','mali','niger','benin','togo','liberia','sierra leone','guinea','guinea-bissau','gambia','mauritania','chad','central african republic','sudan','south sudan','eritrea','djibouti','somalia','angola','mozambique','zambia','zimbabwe','botswana','namibia','lesotho','eswatini','madagascar','rwanda','burundi','malawi','cabo verde','seychelles','mauritius'}
    NENA = {'egypt','morocco','algeria','tunisia','libya','western sahara','saudi arabia','united arab emirates','qatar','bahrain','kuwait','oman','yemen','iraq','syrian arab republic','lebanon','jordan','israel','state of palestine','iran (islamic republic of)'}
    OC   = {'australia','new zealand','papua new guinea','fiji','solomon islands','vanuatu','samoa','tonga','kiribati','micronesia (federated states of)','marshall islands','palau','nauru','tuvalu','new caledonia','french polynesia'}
    for reg, s in [('NA',NA),('WE',WE),('EE',EE),('RUSS',RUSS),('ESEA',ESEA),('SA',SA),('LAC',LAC),('SSA',SSA),('NENA',NENA),('OC',OC)]:
        if a in s: return reg
    return 'GLOBAL'

def default_system_shares() -> pd.DataFrame:
    rows = []
    regions = ['NA','WE','EE','RUSS','ESEA','SA','LAC','SSA','NENA','OC','GLOBAL']
    for r in regions:
        for sp in SPECIES_LIST:
            syss = SYSTEMS_BY_SPECIES[sp]
            if sp in ['layers','broilers','ducks','turkeys','geese_guinea','pigs']:
                if r in ['NA','WE','ESEA','OC']: shares = {'industrial':0.85,'intermediate':0.0,'backyard':0.15}
                elif r in ['LAC','RUSS','EE']:  shares = {'industrial':0.70,'intermediate':0.10,'backyard':0.20}
                elif r in ['SA','NENA']:        shares = {'industrial':0.60,'intermediate':0.10,'backyard':0.30}
                elif r in ['SSA']:              shares = {'industrial':0.40,'intermediate':0.20,'backyard':0.40}
                else:                            shares = {'industrial':0.70,'intermediate':0.10,'backyard':0.20}
                if 'intermediate' not in syss and 'intermediate' in shares:
                    s = shares.copy(); s.pop('intermediate', None)
                    tot = sum(s.values()); shares = {k:v/tot for k,v in s.items()}
            else:
                if r in ['NA','WE','EE','RUSS','OC']:
                    s = {'grassland':0.30,'mixed':0.50,'feedlot':0.20} if 'feedlot' in syss else {'grassland':0.40,'mixed':0.60}
                elif r in ['LAC','ESEA','SA','NENA']:
                    s = {'grassland':0.50,'mixed':0.50,'feedlot':0.00} if 'feedlot' in syss else {'grassland':0.60,'mixed':0.40}
                else:
                    s = {'grassland':0.70,'mixed':0.30,'feedlot':0.00} if 'feedlot' in syss else {'grassland':0.80,'mixed':0.20}
                shares = s
            tot = sum(shares.get(sys,0.0) for sys in syss) or 1.0
            for sys in syss:
                rows.append({'ParamRegion': r,'Species': sp,'System': sys,'Share': shares.get(sys,0.0)/tot})
    return pd.DataFrame(rows)

def seed_rations() -> pd.DataFrame:
    rows = []
    def add_row(param_region, species, system, w):
        s = sum(w.values()) or 1.0
        for it in ITEMS:
            rows.append({'ParamRegion': param_region,'Species': species,'System': system,'Item': it,'Weight': w.get(it,0.0)/s})
    regions10 = ['NA','RUSS','WE','EE','NENA','ESEA','OC','SA','LAC','SSA','GLOBAL']
    for r in regions10:
        if r in ['WE','EE','RUSS','NENA']: w_layers = {'Wheat and products':0.45,'Maize and products':0.25,'Barley and products':0.10,'Soyabeans':0.20}
        elif r in ['NA','ESEA','LAC','OC']: w_layers = {'Wheat and products':0.20,'Maize and products':0.50,'Barley and products':0.05,'Soyabeans':0.25}
        elif r in ['SA','SSA']:              w_layers = {'Wheat and products':0.25,'Maize and products':0.40,'Barley and products':0.05,'Sorghum and products':0.05,'Soyabeans':0.25}
        else:                                 w_layers = {'Wheat and products':0.30,'Maize and products':0.40,'Barley and products':0.05,'Soyabeans':0.25}
        add_row(r,'layers','industrial', w_layers); add_row(r,'layers','backyard', w_layers)

        if r in ['NA','ESEA','LAC','OC']: w_bro = {'Wheat and products':0.10,'Maize and products':0.55,'Barley and products':0.05,'Sorghum and products':0.05,'Soyabeans':0.25}
        elif r in ['WE','EE','RUSS','NENA']: w_bro = {'Wheat and products':0.30,'Maize and products':0.35,'Barley and products':0.10,'Soyabeans':0.25}
        elif r in ['SA','SSA']:              w_bro = {'Wheat and products':0.15,'Maize and products':0.45,'Sorghum and products':0.10,'Millet and products':0.05,'Soyabeans':0.25}
        else:                                 w_bro = {'Wheat and products':0.20,'Maize and products':0.45,'Barley and products':0.05,'Soyabeans':0.30}
        for sp in ['broilers','ducks','turkeys','geese_guinea']:
            add_row(r, sp, 'industrial', w_bro); add_row(r, sp, 'backyard', w_bro)

        if r in ['NA','ESEA','LAC','OC']: w_pig_ind = {'Maize and products':0.45,'Wheat and products':0.15,'Barley and products':0.08,'Soyabeans':0.25,'Sorghum and products':0.05,'Cassava and products':0.02}
        elif r in ['WE','EE','RUSS','NENA']: w_pig_ind = {'Maize and products':0.30,'Wheat and products':0.30,'Barley and products':0.20,'Soyabeans':0.20}
        elif r in ['SA','SSA']: w_pig_ind = {'Maize and products':0.35,'Wheat and products':0.15,'Barley and products':0.05,'Soyabeans':0.20,'Sorghum and products':0.15,'Cassava and products':0.10}
        else: w_pig_ind = {'Maize and products':0.35,'Wheat and products':0.25,'Barley and products':0.15,'Soyabeans':0.20,'Sorghum and products':0.05}
        w_pig_int = w_pig_ind.copy(); w_pig_bk = w_pig_ind.copy()
        if r in ['SSA','SA']:
            w_pig_int['Sorghum and products'] += 0.05; w_pig_int['Cassava and products'] += 0.05
            w_pig_bk['Sorghum and products']  += 0.08; w_pig_bk['Cassava and products']  += 0.07
        if r in ['WE','EE','RUSS']:
            w_pig_int['Wheat and products'] += 0.05; w_pig_int['Barley and products']+= 0.05
            w_pig_bk['Wheat and products']  += 0.08; w_pig_bk['Barley and products'] += 0.07
        add_row(r,'pigs','industrial', w_pig_ind)
        add_row(r,'pigs','intermediate', w_pig_int)
        add_row(r,'pigs','backyard', w_pig_bk)

        if r in ['NA','OC']:       w_dairy = {'Maize and products':0.45,'Barley and products':0.15,'Wheat and products':0.25,'Soyabeans':0.15}
        elif r in ['WE','EE','RUSS']: w_dairy = {'Wheat and products':0.35,'Barley and products':0.35,'Maize and products':0.20,'Soyabeans':0.10}
        elif r in ['LAC','ESEA']:  w_dairy = {'Maize and products':0.50,'Wheat and products':0.20,'Barley and products':0.10,'Soyabeans':0.20}
        elif r in ['SA']:          w_dairy = {'Maize and products':0.40,'Wheat and products':0.30,'Rice and products':0.10,'Barley and products':0.05,'Soyabeans':0.15}
        elif r in ['NENA']:        w_dairy = {'Wheat and products':0.40,'Barley and products':0.35,'Maize and products':0.15,'Soyabeans':0.10}
        else:                      w_dairy = {'Maize and products':0.50,'Sorghum and products':0.20,'Millet and products':0.10,'Wheat and products':0.10,'Soyabeans':0.10}
        w_beef = w_dairy.copy()
        w_beef_feedlot = w_beef.copy(); w_beef_feedlot['Maize and products'] = w_beef_feedlot.get('Maize and products',0)+0.10
        for sys in ['grassland','mixed','feedlot']:
            if sys in ['grassland','mixed']:
                add_row(r,'dairy_cattle',sys, w_dairy); add_row(r,'beef_cattle',sys, w_beef)
            else:
                add_row(r,'dairy_cattle',sys, w_dairy); add_row(r,'beef_cattle',sys, w_beef_feedlot)

        w_smallrum = w_dairy.copy()
        for sp in ['buffalo','sheep','goat']:
            for sys in ['grassland','mixed']:
                add_row(r, sp, sys, w_smallrum)
    return pd.DataFrame(rows)

def aggregate_rations(rations_df: pd.DataFrame, system_shares_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pr in rations_df['ParamRegion'].unique():
        for sp in rations_df['Species'].unique():
            df_rs = rations_df[(rations_df['ParamRegion']==pr) & (rations_df['Species']==sp)]
            if df_rs.empty: continue
            ss = system_shares_df[(system_shares_df['ParamRegion']==pr) & (system_shares_df['Species']==sp)]
            if ss.empty: ss = system_shares_df[(system_shares_df['ParamRegion']=='GLOBAL') & (system_shares_df['Species']==sp)]
            for item in ITEMS:
                w = 0.0
                for _, srow in ss.iterrows():
                    w_sys = df_rs[(df_rs['System']==srow['System']) & (df_rs['Item']==item)]['Weight']
                    if not w_sys.empty:
                        w += float(srow['Share']) * float(w_sys.iloc[0])
                rows.append({'ParamRegion': pr,'Species': sp,'Item': item,'Weight': w})
    out = pd.DataFrame(rows)
    out['Weight'] = out['Weight'].fillna(0.0)
    out['Weight'] = out.groupby(['ParamRegion','Species'])['Weight'].transform(lambda s: s/(s.sum() if s.sum()>0 else 1.0))
    return out

def load_inputs(base: Path):
    fbs  = pd.read_csv(base/"FoodBalanceSheets_E_All_Data_NOFLAG.csv")
    prod = pd.read_csv(base/"Production_Crops_Livestock_E_All_Data_NOFLAG.csv")
    gleam_tab31 = pd.read_excel(base/"GLEAM_3.0_Supplement_S1.xlsx", sheet_name="Tab. S.3.1")
    dm_df = gleam_tab31[['Unnamed: 1','Unnamed: 2']].dropna()
    dm_df.columns = ['Crop','DM_percent']
    dm_map = {item: float(dm_df.set_index('Crop').loc[crop, 'DM_percent'])
              for item, crop in ITEM_TO_CROP.items()
              if crop in dm_df.set_index('Crop').index}
    # dict_v3 region
    reg = pd.read_excel(base/"dict_v3.xlsx", sheet_name="region")
    reg_valid = reg[reg['Region_label_new'].astype(str).str.lower()!='no'].copy()
    # Keep only the columns we need
    reg_keep = reg_valid[['Area Code','Region_label_new','Region_agg1','Region_agg2','Region_agg3','Region_agg4','Region_agg5']].copy()
    return fbs, prod, dm_map, reg_keep

def prep_feed_matrices(fbs: pd.DataFrame, dm_map: dict, valid_codes: set) -> pd.DataFrame:
    feed = fbs[fbs['Element']=='Feed'][['Area','Area Code','Item','Unit'] + YCOLS].copy()
    feed = feed[feed['Item'].isin(ITEMS)].copy()
    feed = feed[feed['Area Code'].isin(valid_codes)].copy()
    feed[YCOLS] = feed[YCOLS].fillna(0.0) * 1_000.0 * 1_000.0
    for item in ITEMS:
        dm_frac = (dm_map.get(item, 0.0)) / 100.0
        feed.loc[feed['Item']==item, [f"DM_{y}" for y in YEARS]]   = feed.loc[feed['Item']==item, YCOLS].values * dm_frac
        feed.loc[feed['Item']==item, [f"ASIS_{y}" for y in YEARS]] = feed.loc[feed['Item']==item, YCOLS].values
    return feed[['Area','Area Code','Item'] + [f"DM_{y}" for y in YEARS] + [f"ASIS_{y}" for y in YEARS]]

def to_head_df_filter(df: pd.DataFrame, valid_codes: set) -> pd.DataFrame:
    if df.empty: return df
    out = df[df['Area Code'].isin(valid_codes)].copy()
    if 'Unit' in out.columns and out['Unit'].nunique()==1 and out['Unit'].iloc[0]=='1000 An':
        for y in YEARS:
            col = f"Y{y}"
            if col in out.columns: out[col] = out[col] * 1000.0
    return out.drop(columns=['Unit'], errors='ignore')

def build_heads_all(prod: pd.DataFrame, valid_codes: set) -> dict:
    chickens_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item']=='Chickens')][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    layers_hen     = to_head_df_filter(prod[(prod['Element']=='Laying') & (prod['Item'].isin(['Hen eggs in shell, fresh','Eggs Primary']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    layers = layers_hen.groupby(['Area','Area Code'])[YCOLS].max().reset_index() if not layers_hen.empty else pd.DataFrame(columns=['Area','Area Code']+YCOLS)
    broilers = (chickens_stock.set_index(['Area','Area Code']) - layers.set_index(['Area','Area Code']).reindex(chickens_stock.set_index(['Area','Area Code']).index).fillna(0.0)).clip(lower=0.0).reset_index()
    ducks   = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Ducks']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    turkeys = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Turkeys']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    geese   = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Geese and guinea fowls']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)

    cattle_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item']=='Cattle')][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    dairy_cattle = to_head_df_filter(prod[(prod['Element']=='Milk Animals') & (prod['Item'].isin(['Raw milk of cattle']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    beef_cattle  = (cattle_stock.set_index(['Area','Area Code']) - dairy_cattle.set_index(['Area','Area Code']).reindex(cattle_stock.set_index(['Area','Area Code']).index).fillna(0.0)).clip(lower=0.0).reset_index()

    buffalo_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Buffaloes']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    dairy_buffalo = to_head_df_filter(prod[(prod['Element']=='Milk Animals') & (prod['Item'].isin(['Raw milk of buffalo']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    buffalo      = (buffalo_stock.set_index(['Area','Area Code']) - dairy_buffalo.set_index(['Area','Area Code']).reindex(buffalo_stock.set_index(['Area','Area Code']).index).fillna(0.0)).clip(lower=0.0).reset_index()

    sheep_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Sheep']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    dairy_sheep = to_head_df_filter(prod[(prod['Element']=='Milk Animals') & (prod['Item'].isin(['Raw milk of sheep']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    sheep       = (sheep_stock.set_index(['Area','Area Code']) - dairy_sheep.set_index(['Area','Area Code']).reindex(sheep_stock.set_index(['Area','Area Code']).index).fillna(0.0)).clip(lower=0.0).reset_index()

    goat_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Goats']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    dairy_goat = to_head_df_filter(prod[(prod['Element']=='Milk Animals') & (prod['Item'].isin(['Raw milk of goats']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)
    goat       = (goat_stock.set_index(['Area','Area Code']) - dairy_goat.set_index(['Area','Area Code']).reindex(goat_stock.set_index(['Area','Area Code']).index).fillna(0.0)).clip(lower=0.0).reset_index()

    pigs_stock = to_head_df_filter(prod[(prod['Element']=='Stocks') & (prod['Item'].isin(['Swine / pigs']))][['Area','Area Code','Unit'] + YCOLS].copy(), valid_codes)

    return {
        'layers': layers, 'broilers': broilers, 'ducks': ducks, 'turkeys': turkeys, 'geese_guinea': geese,
        'dairy_cattle': dairy_cattle, 'beef_cattle': beef_cattle,
        'buffalo': buffalo, 'sheep': sheep, 'goat': goat,
        'pigs': pigs_stock,
    }

def build_country_region_from_dict(feed_df: pd.DataFrame, reg_keep: pd.DataFrame) -> pd.DataFrame:
    # Map by Area Code
    cr = feed_df[['Area','Area Code']].drop_duplicates().merge(reg_keep, on='Area Code', how='left')
    # Build a ParamRegion for parameters using auto_region on country names
    cr['ParamRegion'] = cr['Area'].str.lower().map(auto_region)
    # For transparency, report any missing region mapping rows
    return cr

def seed_pig_cycle_days() -> pd.DataFrame:
    rows = []
    for r in ['NA','WE','EE','RUSS','ESEA','SA','LAC','SSA','NENA','OC','GLOBAL']:
        rows += [{'ParamRegion': r,'System':'industrial','cycle_days':180},
                 {'ParamRegion': r,'System':'intermediate','cycle_days':210},
                 {'ParamRegion': r,'System':'backyard','cycle_days':240}]
    return pd.DataFrame(rows)

def build_pigs_equiv(country_region: pd.DataFrame, system_shares_df: pd.DataFrame, pig_cycle_days: pd.DataFrame, pigs_slaughter: pd.DataFrame) -> pd.DataFrame:
    if pigs_slaughter is None or pigs_slaughter.empty:
        return pd.DataFrame(columns=['Area','Area Code']+[f"Y{y}" for y in YEARS])
    sl = pigs_slaughter.set_index(['Area','Area Code'])
    out = pd.DataFrame({'Area': sl.index.get_level_values(0).values,
                        'Area Code': sl.index.get_level_values(1).values})
    for y in YEARS: out[f"Y{y}"] = 0.0
    for (area, ac) in sl.index:
        param_region = country_region.loc[country_region['Area Code']==ac, 'ParamRegion']
        pr = param_region.iloc[0] if not param_region.empty else 'GLOBAL'
        shares = system_shares_df[(system_shares_df['ParamRegion']==pr) & (system_shares_df['Species']=='pigs')]
        if shares.empty: shares = system_shares_df[(system_shares_df['ParamRegion']=='GLOBAL') & (system_shares_df['Species']=='pigs')]
        cyc = 0.0
        for _, r in shares.iterrows():
            cd = pig_cycle_days[(pig_cycle_days['ParamRegion']==pr) & (pig_cycle_days['System']==r['System'])]['cycle_days']
            if cd.empty: cd = pig_cycle_days[(pig_cycle_days['ParamRegion']=='GLOBAL') & (pig_cycle_days['System']==r['System'])]['cycle_days']
            cyc += r['Share'] * (float(cd.iloc[0]) if not cd.empty else 200.0)
        for y in YEARS:
            val = sl.loc[(area, ac), f"Y{y}"] if f"Y{y}" in sl.columns else 0.0
            out.loc[out['Area Code']==ac, f"Y{y}"] = float(val) * (cyc/365.0)
    return out.reset_index(drop=True)

def allocate_by_region(feed_df: pd.DataFrame, basis_prefix: str, country_region: pd.DataFrame, rations_region: pd.DataFrame, heads_map: dict) -> dict:
    alloc = {sp: [] for sp in heads_map.keys()}
    basis_cols = [f"{basis_prefix}_{y}" for y in YEARS]
    for _, row in feed_df[['Area','Area Code','Item'] + basis_cols].iterrows():
        area, ac, item = row['Area'], row['Area Code'], row['Item']
        pr = country_region.loc[country_region['Area Code']==ac, 'ParamRegion']
        param_region = pr.iloc[0] if not pr.empty else 'GLOBAL'
        rr = rations_region[(rations_region['ParamRegion']==param_region) & (rations_region['Item']==item)]
        if rr.empty: rr = rations_region[(rations_region['ParamRegion']=='GLOBAL') & (rations_region['Item']==item)]
        w_map = {sp: (float(rr[rr['Species']==sp]['Weight'].iloc[0]) if not rr[rr['Species']==sp].empty else 0.0) for sp in heads_map.keys()}
        heads_at_area = {sp: (dfh[(dfh['Area Code']==ac)].iloc[0] if (not dfh.empty and ac in dfh['Area Code'].values) else None) for sp, dfh in heads_map.items()}
        for y in YEARS:
            total_val = float(row.get(f"{basis_prefix}_{y}", 0.0) or 0.0)
            weights = {sp: (float(heads_at_area[sp][f"Y{y}"]) if (heads_at_area[sp] is not None and f"Y{y}" in heads_at_area[sp]) else 0.0) * w_map.get(sp,0.0) for sp in heads_map.keys()}
            totw = sum(weights.values())
            for sp in heads_map.keys():
                alloc_val = 0.0 if (totw<=0 or total_val<=0) else total_val * (weights[sp]/totw)
                alloc[sp].append({'Area': area,'Area Code': ac,'Item': item,'Year': y,'alloc': alloc_val})
    return {sp: pd.DataFrame(rows) for sp, rows in alloc.items()}

def per_head_wide(alloc_df: pd.DataFrame, heads_df: pd.DataFrame) -> pd.DataFrame:
    if alloc_df is None or alloc_df.empty:
        return pd.DataFrame(columns=['Area','Area Code','Item']+[f"Y{y}" for y in YEARS])
    df = alloc_df.copy()
    df['col'] = df['Year'].apply(lambda y: f"Y{y}")
    wide = df.pivot_table(index=['Area','Area Code','Item'], columns='col', values='alloc', aggfunc='sum').reset_index()
    heads = heads_df.drop_duplicates(['Area','Area Code']) if not heads_df.empty else pd.DataFrame(columns=['Area','Area Code']+[f"Y{y}" for y in YEARS])
    for y in YEARS:
        col = f"Y{y}"
        if col in wide.columns and col in heads.columns:
            denom = heads.set_index(['Area','Area Code']).reindex(wide.set_index(['Area','Area Code']).index)[col].values
            with np.errstate(divide='ignore', invalid='ignore'):
                wide[col] = np.where(denom>0, wide[col]/denom, 0.0)
    return wide.replace([np.inf,-np.inf], np.nan).fillna(0.0)

def compute_grass_ratio(country_region: pd.DataFrame, system_shares_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cr in country_region.iterrows():
        area, ac, pr = cr['Area'], cr['Area Code'], cr['ParamRegion']
        for sp in SPECIES_LIST:
            ss = system_shares_df[(system_shares_df['ParamRegion']==pr) & (system_shares_df['Species']==sp)]
            if ss.empty: ss = system_shares_df[(system_shares_df['ParamRegion']=='GLOBAL') & (system_shares_df['Species']==sp)]
            g = 0.0
            for _, r in ss.iterrows():
                g += float(r['Share']) * GRASS_FRAC_BY_SYSTEM.get(r['System'], 0.0)
            for y in YEARS:
                rows.append({'Area': area, 'Area Code': ac, 'Species': sp, 'Year': y, 'grass_feed_ratio': g})
    return pd.DataFrame(rows)

def main():
    base = Path(get_input_base())
    fbs, prod, dm_map, reg_keep = load_inputs(base)
    valid_codes = set(reg_keep['Area Code'].unique())
    feed = prep_feed_matrices(fbs, dm_map, valid_codes)
    heads_raw = build_heads_all(prod, valid_codes)
    # Build country->region mapping (dict sheet) + param-region (for system shares/rations)
    country_region = build_country_region_from_dict(feed, reg_keep)
    # Params
    system_shares_df = default_system_shares()
    rations_df      = seed_rations()
    rations_region  = aggregate_rations(rations_df, system_shares_df)
    # Pigs equiv
    # Heads map
    heads_map = {
        'layers': heads_raw['layers'], 'broilers': heads_raw['broilers'], 'ducks': heads_raw['ducks'],
        'turkeys': heads_raw['turkeys'], 'geese_guinea': heads_raw['geese_guinea'],
        'pigs': heads_raw['pigs'],
        'dairy_cattle': heads_raw['dairy_cattle'], 'beef_cattle': heads_raw['beef_cattle'],
        'buffalo': heads_raw['buffalo'], 'sheep': heads_raw['sheep'], 'goat': heads_raw['goat'],
    }
    # Allocate
    feed_dm   = feed[['Area','Area Code','Item'] + [f"DM_{y}" for y in YEARS]]
    feed_asis = feed[['Area','Area Code','Item'] + [f"ASIS_{y}" for y in YEARS]]
    alloc_dm   = allocate_by_region(feed_dm, 'DM', country_region, rations_region, heads_map)
    alloc_asis = allocate_by_region(feed_asis,'ASIS',country_region, rations_region, heads_map)
    # Per-head
    per_dm_wide = {sp: per_head_wide(df, heads_map[sp]) for sp, df in alloc_dm.items()}
    per_as_wide = {sp: per_head_wide(df, heads_map[sp]) for sp, df in alloc_asis.items()}
    # Grass
    grass_ratio = compute_grass_ratio(country_region, system_shares_df)
    # Crop DM total per head
    crop_dm_total_rows = []
    for sp, df in per_dm_wide.items():
        if df is None or df.empty: 
            continue
        d = df.copy(); d['Species'] = sp
        for y in YEARS: d[f"Y{y}"] = d[f"Y{y}"].fillna(0.0)
        g = d.groupby(['Area','Area Code'])[[f"Y{y}" for y in YEARS]].sum().reset_index()
        g['Species'] = sp
        crop_dm_total_rows.append(g[['Species','Area','Area Code'] + [f"Y{y}" for y in YEARS]])
    crop_dm_total = pd.concat(crop_dm_total_rows, ignore_index=True) if crop_dm_total_rows else pd.DataFrame(columns=['Species','Area','Area Code']+[f"Y{y}" for y in YEARS])

    # Attach region labels from dict to all outputs
    attach_cols = ['Region_label_new','Region_agg1','Region_agg2','Region_agg3','Region_agg4','Region_agg5']
    cr_slim = country_region[['Area','Area Code'] + attach_cols].drop_duplicates()

    def attach_region(df):
        return df.merge(cr_slim, on=['Area','Area Code'], how='left')

    per_dm_wide = {sp: attach_region(df) for sp, df in per_dm_wide.items()}
    per_as_wide = {sp: attach_region(df) for sp, df in per_as_wide.items()}
    grass_ratio = attach_region(grass_ratio)
    crop_dm_total = attach_region(crop_dm_total)

    # Estimate grass DM from ratio
    gr_w = grass_ratio.pivot_table(index=['Area','Area Code','Species'], columns='Year', values='grass_feed_ratio').reset_index()
    gr_w.columns = ['Area','Area Code','Species'] + [f"Y{int(c)}" for c in gr_w.columns[3:]]
    grass_dm_est = crop_dm_total.merge(gr_w, on=['Area','Area Code','Species'], how='left', suffixes=('_cropDM','_r'))
    def calc_grass(row, y):
        crop = row.get(f"Y{y}_cropDM", np.nan); r = row.get(f"Y{y}_r", np.nan)
        if pd.isna(crop) or pd.isna(r) or r>=1.0: 
            return (np.nan if pd.isna(crop) else 0.0)
        return float(crop) * r / (1.0 - r)
    if not grass_dm_est.empty:
        # Prepare crop DM total per head with clean column names
        crop_dm_total_final = grass_dm_est[['Species','Area','Area Code'] + [f"Y{y}_cropDM" for y in YEARS]].copy()
        crop_dm_total_final.rename(columns={f"Y{y}_cropDM": f"Y{y}" for y in YEARS}, inplace=True)
        # Grass DM per head estimated
        grass_vals = []
        for _, row in grass_dm_est.iterrows():
            new = {'Species': row['Species'], 'Area': row['Area'], 'Area Code': row['Area Code']}
            for y in YEARS: new[f"Y{y}"] = calc_grass(row, y)
            grass_vals.append(new)
        grass_dm_final = pd.DataFrame(grass_vals)
        crop_dm_total_final = attach_region(crop_dm_total_final)
        grass_dm_final = attach_region(grass_dm_final)
    else:
        crop_dm_total_final = pd.DataFrame(columns=['Species','Area','Area Code']+[f"Y{y}" for y in YEARS])
        grass_dm_final      = pd.DataFrame(columns=['Species','Area','Area Code']+[f"Y{y}" for y in YEARS])

    # Export
    out = base/"unit_feed_crops_per_head_region_system_2010_2020_v3.xlsx"
    with pd.ExcelWriter(out, engine='xlsxwriter') as w:
        for sp in per_dm_wide.keys():
            d1 = per_dm_wide[sp].copy(); d1.insert(0,'Species', sp)
            d1.to_excel(w, sheet_name=f"{sp}_kgDM_per_head", index=False)
            d2 = per_as_wide[sp].copy(); d2.insert(0,'Species', sp)
            d2.to_excel(w, sheet_name=f"{sp}_kgASIS_per_head", index=False)
        # One-sheet
        frames = []
        for sp, df in per_dm_wide.items():
            if df.empty: continue
            t = df.copy(); t.insert(0,'Species', sp); t.insert(1,'Basis','kgDM')
            frames.append(t[['Species','Basis','Area','Area Code','Item','Region_label_new','Region_agg1','Region_agg2','Region_agg3','Region_agg4','Region_agg5']+[f"Y{y}" for y in YEARS]])
        for sp, df in per_as_wide.items():
            if df.empty: continue
            t = df.copy(); t.insert(0,'Species', sp); t.insert(1,'Basis','kgASIS')
            frames.append(t[['Species','Basis','Area','Area Code','Item','Region_label_new','Region_agg1','Region_agg2','Region_agg3','Region_agg4','Region_agg5']+[f"Y{y}" for y in YEARS]])
        if frames:
            one = pd.concat(frames, ignore_index=True)
            one.to_excel(w, sheet_name="per_head_all", index=False)
        # grass
        grass_ratio.to_excel(w, sheet_name="grass_feed_ratio_long", index=False)
        crop_dm_total_final.to_excel(w, sheet_name="crop_total_kgDM_per_head", index=False)
        grass_dm_final.to_excel(w, sheet_name="grass_total_kgDM_per_head_est", index=False)
        # region dictionary
        country_region.to_excel(w, sheet_name="country_to_region(dict_v3)", index=False)

        # grass assumptions
        pd.DataFrame({'System': list(GRASS_FRAC_BY_SYSTEM.keys()), 'grass_DM_fraction': list(GRASS_FRAC_BY_SYSTEM.values())}).to_excel(w, sheet_name="grass_assumptions", index=False)

if __name__ == "__main__":
    main()
