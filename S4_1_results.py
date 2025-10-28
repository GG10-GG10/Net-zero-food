# -*- coding: utf-8 -*-
"""
S4.1_results.py 鈥?缁熶竴姹囨€讳笌瀵煎嚭锛堝惈 GHG 鍒嗚В涓?AR6 GWP100 鍙ｅ緞锛?- 姹囨€绘帓鏀撅細Country鈥揚rocess鈥揅ommodity锛孋ountry鈥揚rocess锛孋ountry锛堝惈 CH4/N2O/CO2 鍒?& CO2eq_AR6锛?- 姹囨€讳緵闇€锛欳ountry鈥揅ommodity 鐨?Qs/Qd/Import/Export/Price
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# AR6 GWP100
GWP100_AR6 = {'CO2': 1.0, 'CH4': 27.2, 'N2O': 273.0}

def _col(df, name_like_list):
    cols = [c for c in df.columns for n in name_like_list if n.lower() in str(c).lower()]
    return cols[0] if cols else None

def _norm_emis_df(df: pd.DataFrame,
                  process_label: Optional[str]=None,
                  process_meta: Optional[dict]=None) -> pd.DataFrame:
    """
    褰掍竴鍖栨帓鏀炬槑缁嗗埌缁熶竴鍒楋細
    country, iso3, year, process, commodity, co2_kt, ch4_kt, n2o_kt, co2e_kt_ar6
    - 鑻ュ師琛ㄥ彧鏈夋€婚噺 co2e_kt锛屼篃杩斿洖骞跺敖閲忔媶瑙ｄ负 0/0/0 + co2e_kt
    - 鑻?process_meta 缁欏嚭璇ヨ繃绋嬮粯璁ゆ皵浣擄紙濡?'gas'=='CH4'锛夛紝鍙敤鏉ュ～鍏呭厹搴?    """
    if df is None or len(df)==0:
        return pd.DataFrame(columns=['country','iso3','year','process','commodity',
                                     'co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6'])
    z = df.copy()
    z.columns = [str(c).strip() for c in z.columns]

    c_ctry = _col(z, ['country','area'])
    c_iso3 = _col(z, ['iso3'])
    c_year = _col(z, ['year','time'])
    c_comm = _col(z, ['commodity','item','product'])
    c_co2  = _col(z, ['co2_kt','co2 kt','co2 (kt)'])
    c_ch4  = _col(z, ['ch4_kt','ch4 kt','ch4 (kt)'])
    c_n2o  = _col(z, ['n2o_kt','n2o kt','n2o (kt)'])
    c_co2e = _col(z, ['co2e_kt','co2eq_kt','co2e'])

    out = pd.DataFrame({
        'country':   z[c_ctry] if c_ctry else None,
        'iso3':      z[c_iso3] if c_iso3 else None,
        'year':      z[c_year] if c_year else None,
        'commodity': z[c_comm] if c_comm else 'ALL',
    })

    if process_label is not None:
        out['process'] = process_label
    else:
        c_proc = _col(z, ['process'])
        out['process'] = z[c_proc] if c_proc else 'Unknown'

    out['co2_kt'] = pd.to_numeric(z[c_co2], errors='coerce') if c_co2 else 0.0
    out['ch4_kt'] = pd.to_numeric(z[c_ch4], errors='coerce') if c_ch4 else 0.0
    out['n2o_kt'] = pd.to_numeric(z[c_n2o], errors='coerce') if c_n2o else 0.0

    if c_co2e:
        out['co2e_kt_ar6'] = pd.to_numeric(z[c_co2e], errors='coerce')
    else:
        out['co2e_kt_ar6'] = (
            out['co2_kt'] * GWP100_AR6['CO2'] +
            out['ch4_kt'] * GWP100_AR6['CH4'] +
            out['n2o_kt'] * GWP100_AR6['N2O']
        )

    # 鑻ヤ笁姘旈兘涓?浣嗙煡閬撹杩囩▼鐗瑰畾姘斾綋锛堟潵鑷?meta锛夛紝鐢ㄦ€婚噺鎸?GWP 鍙嶆帹璇ユ皵浣撶殑 kt
    if (out[['co2_kt','ch4_kt','n2o_kt']].sum(axis=1)==0).all() and process_meta:
        gas = None
        if isinstance(process_meta, dict):
            gas = process_meta.get('gas') or process_meta.get('GHG') or process_meta.get('Gas')
            if isinstance(gas, str):
                gas = gas.upper()
        if gas in ('CH4','N2O','CO2'):
            col = {'CH4':'ch4_kt','N2O':'n2o_kt','CO2':'co2_kt'}[gas]
            out[col] = out['co2e_kt_ar6'] / GWP100_AR6[gas]

    out = out.dropna(subset=['co2e_kt_ar6'])
    return out[['country','iso3','year','process','commodity',
                'co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']]

def summarize_emissions(fao_results: Dict[str, Any],
                        extra_emis: Optional[pd.DataFrame]=None,
                        process_meta_map: Optional[dict]=None) -> Dict[str, pd.DataFrame]:
    """
    鎶?orchestrator 鍚勬ā鍧楄緭鍑虹粺涓€鍒颁竴涓暱琛紝骞剁敓鎴愬绮掑害姹囨€伙紱
    鍖哄垎 CO2/CH4/N2O锛屽苟鎸?AR6 GWP100 璁＄畻 co2e_kt_ar6銆?    """
    frames: List[pd.DataFrame] = []

    def meta_for(label: str) -> Optional[dict]:
        pm = process_meta_map or {}
        name = label.split(':',1)[1] if ':' in label else label
        return pm.get(name)

    gce = (fao_results or {}).get('GCE', {})
    for k, df in (gce or {}).items():
        if isinstance(df, pd.DataFrame):
            frames.append(_norm_emis_df(df, process_label=f'GCE:{k}',
                                        process_meta=meta_for(f'GCE:{k}')))

    gle = (fao_results or {}).get('GLE', [])
    for yout in gle or []:
        if isinstance(yout, dict):
            for k, df in yout.items():
                if isinstance(df, pd.DataFrame):
                    frames.append(_norm_emis_df(df, process_label=f'GLE:{k}',
                                                process_meta=meta_for(f'GLE:{k}')))

    gos = (fao_results or {}).get('GOS', [])
    for yout in gos or []:
        if isinstance(yout, dict):
            for k, df in yout.items():
                if isinstance(df, pd.DataFrame):
                    frames.append(_norm_emis_df(df, process_label=f'GOS:{k}',
                                                process_meta=meta_for(f'GOS:{k}')))

    gfe = (fao_results or {}).get('GFE', {})
    for k, df in (gfe or {}).items():
        if isinstance(df, pd.DataFrame):
            frames.append(_norm_emis_df(df, process_label=f'GFE:{k}',
                                        process_meta=meta_for(f'GFE:{k}')))

    luf = (fao_results or {}).get('LUF', None)
    if isinstance(luf, pd.DataFrame):
        frames.append(_norm_emis_df(luf, process_label='Land-use fires',
                                    process_meta=meta_for('Land-use fires')))

    if extra_emis is not None:
        frames.append(_norm_emis_df(extra_emis, process_label=None))

    if frames:
        long = pd.concat(frames, ignore_index=True)
    else:
        long = pd.DataFrame(columns=['country','iso3','year','process','commodity',
                                     'co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6'])

    by_ctry_proc_comm = long.groupby(['country','year','process','commodity'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry_proc = long.groupby(['country','year','process'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry = long.groupby(['country','year'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum().rename(
            columns={'co2e_kt_ar6':'co2e_kt_total_ar6'}
        )
    by_year = long.groupby(['year'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum().rename(
            columns={'co2e_kt_ar6':'co2e_kt_total_ar6'}
        )

    return {'by_ctry_proc_comm': by_ctry_proc_comm,
            'by_ctry_proc': by_ctry_proc,
            'by_ctry': by_ctry,
            'by_year': by_year,
            'long': long}


def summarize_emissions_from_detail(emis_detail: pd.DataFrame,
                                    *,
                                    process_meta_map: Optional[dict] = None,
                                    allowed_years: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
    empty = {
        'by_ctry_proc_comm': pd.DataFrame(columns=['country','year','process','commodity','co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']),
        'by_ctry_proc': pd.DataFrame(columns=['country','year','process','co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']),
        'by_ctry': pd.DataFrame(columns=['country','year','co2_kt','ch4_kt','n2o_kt','co2e_kt_total_ar6']),
        'by_year': pd.DataFrame(columns=['year','co2_kt','ch4_kt','n2o_kt','co2e_kt_total_ar6']),
        'long': pd.DataFrame(columns=['country','iso3','year','process','commodity','co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6'])
    }
    if emis_detail is None or len(emis_detail) == 0:
        return empty
    df = emis_detail.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'year' not in df.columns or 'country' not in df.columns or 'process' not in df.columns:
        return empty
    if allowed_years is not None:
        allowed = set(allowed_years)
        df = df[df['year'].isin(allowed)]
    if df.empty:
        return empty
    if 'iso3' not in df.columns:
        df['iso3'] = None
    df['process'] = df['process'].astype(str).str.strip()
    df['commodity'] = df['commodity'].astype(str).str.strip() if 'commodity' in df.columns else 'ALL'
    df['emissions_tco2e'] = pd.to_numeric(df.get('emissions_tco2e', 0.0), errors='coerce').fillna(0.0)
    df = df[df['emissions_tco2e'] != 0.0]
    if df.empty:
        return empty
    meta = process_meta_map or {}
    def _gas(proc: str) -> str:
        info = meta.get(proc) if isinstance(meta, dict) else None
        gas = None
        if isinstance(info, dict):
            gas = info.get('gas') or info.get('GHG') or info.get('Gas')
        return str(gas).upper() if isinstance(gas, str) else ''
    df['gas'] = df['process'].apply(_gas)
    df['co2e_kt_ar6'] = df['emissions_tco2e'] / 1000.0
    df['co2_kt'] = 0.0
    df['ch4_kt'] = 0.0
    df['n2o_kt'] = 0.0
    for gas_name, col in (('CO2', 'co2_kt'), ('CH4', 'ch4_kt'), ('N2O', 'n2o_kt')):
        mask = df['gas'] == gas_name
        if not mask.any():
            continue
        if gas_name == 'CO2':
            df.loc[mask, col] = df.loc[mask, 'co2e_kt_ar6']
        else:
            divisor = GWP100_AR6.get(gas_name, 1.0)
            if divisor > 0:
                df.loc[mask, col] = df.loc[mask, 'co2e_kt_ar6'] / divisor
    cols = ['country','iso3','year','process','commodity','co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']
    long_df = df[cols].copy()
    by_ctry_proc_comm = long_df.groupby(['country','year','process','commodity'], as_index=False)[['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry_proc = long_df.groupby(['country','year','process'], as_index=False)[['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry = long_df.groupby(['country','year'], as_index=False)[['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry.rename(columns={'co2e_kt_ar6':'co2e_kt_total_ar6'}, inplace=True)
    by_year = long_df.groupby(['year'], as_index=False)[['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_year.rename(columns={'co2e_kt_ar6':'co2e_kt_total_ar6'}, inplace=True)
    return {
        'by_ctry_proc_comm': by_ctry_proc_comm,
        'by_ctry_proc': by_ctry_proc,
        'by_ctry': by_ctry,
        'by_year': by_year,
        'long': long_df
    }
def summarize_market(model, var, universe, data=None, price_df: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    """
    浠庤В鍑虹殑鍙橀噺鎶藉彇 Qs/Qd/Import/Export銆俻rice_df 鍙€夛細
    columns: country, iso3, year, commodity, price
    """
    rows = []
    try:
        Qs = var['Qs']; Qd = var['Qd']
    except Exception:
        return pd.DataFrame(columns=['country','year','commodity','Qs','Qd','Import','Export','price'])

    Pc = var.get('Pc', {})
    Pnet = var.get('Pnet', {})
    W = var.get('W', {})
    Cij = var.get('C', {})
    Eij = var.get('E', {})
    seen = set()
    def _safe_float(val, default=np.nan):
        try:
            if val is None:
                return default
            out = float(val)
            if np.isnan(out):
                return default
            return out
        except Exception:
            return default

    for (i, j, t), svar in Qs.items():
        q_supply = svar.X if hasattr(svar, 'X') else np.nan
        dvar = Qd.get((i, j, t))
        q_demand = dvar.X if dvar is not None and hasattr(dvar, 'X') else np.nan
        imp = max(q_demand - q_supply, 0.0) if np.isfinite(q_supply) and np.isfinite(q_demand) else np.nan
        exp = max(q_supply - q_demand, 0.0) if np.isfinite(q_supply) and np.isfinite(q_demand) else np.nan
        price = Pc.get((j, t))
        price_val = price.X if price is not None and hasattr(price, 'X') else np.nan
        price_net = Pnet.get((i, j, t))
        price_net_val = price_net.X if price_net is not None and hasattr(price_net, 'X') else np.nan
        w = W.get((i, j, t))
        unit_cost = w.X if w is not None and hasattr(w, 'X') else np.nan
        cost = Cij.get((i, j, t))
        cost_val = cost.X if cost is not None and hasattr(cost, 'X') else np.nan
        emis = Eij.get((i, j, t))
        emis_val = emis.X if emis is not None and hasattr(emis, 'X') else np.nan
        rows.append((i, t, j, q_supply, q_demand, imp, exp, price_val, price_net_val, unit_cost, cost_val, emis_val))
        seen.add((i, j, t))

    import_slack = var.get('Import') or {}
    export_slack = var.get('Export') or {}
    slack_keys = set()
    if isinstance(import_slack, dict):
        slack_keys.update(import_slack.keys())
    if isinstance(export_slack, dict):
        slack_keys.update(export_slack.keys())
    for key in sorted(slack_keys):
        j, t = key
        imp_var = import_slack.get(key)
        exp_var = export_slack.get(key)
        imp_val = imp_var.X if imp_var is not None and hasattr(imp_var, 'X') else np.nan
        exp_val = exp_var.X if exp_var is not None and hasattr(exp_var, 'X') else np.nan
        if not (np.isfinite(imp_val) or np.isfinite(exp_val)):
            continue
        rows.append(('ROW', t, j, np.nan, np.nan,
                     imp_val if np.isfinite(imp_val) else np.nan,
                     exp_val if np.isfinite(exp_val) else np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan))

    nodes = getattr(data, 'nodes', []) if data is not None else []
    for n in nodes:
        key = (n.country, n.commodity, n.year)
        if key in seen:
            continue
        supply = _safe_float(getattr(n, 'Q0', None), 0.0)
        demand_source = getattr(n, 'D0', None)
        if demand_source is None:
            demand_source = getattr(n, 'Q0', None)
        demand = _safe_float(demand_source, 0.0)
        imp = max(demand - supply, 0.0)
        exp = max(supply - demand, 0.0)
        price_val = _safe_float(getattr(n, 'P0', None))
        rows.append((n.country, n.year, n.commodity, supply, demand, imp, exp,
                     price_val, price_val, np.nan, np.nan, np.nan))

    out = pd.DataFrame(rows, columns=['country','year','commodity','Qs','Qd','Import','Export',
                                      'price_global','price_net','unit_abatement_cost','abatement_cost','emissions'])
    if price_df is not None and len(price_df):
        out = out.merge(price_df[['country','year','commodity','price']],
                        on=['country','year','commodity'], how='left')
    if not out.empty:
        mask = out['commodity'].astype(str).str.strip().isin({'1', '2'})
        out = out[~mask].reset_index(drop=True)
    return out

