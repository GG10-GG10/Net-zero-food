# -*- coding: utf-8 -*-
"""
S4.1_results.py — 统一汇总与导出（含 GHG 分解与 AR6 GWP100 口径）
- 汇总排放：Country–Process–Commodity，Country–Process，Country（含 CH4/N2O/CO2 列 & CO2eq_AR6）
- 汇总供需：Country–Commodity 的 Qs/Qd/Import/Export/Price
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
    归一化排放明细到统一列：
    country, iso3, year, process, commodity, co2_kt, ch4_kt, n2o_kt, co2e_kt_ar6
    - 若原表只有总量 co2e_kt，也返回并尽量拆解为 0/0/0 + co2e_kt
    - 若 process_meta 给出该过程默认气体（如 'gas'=='CH4'），可用来填充兜底
    """
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

    # 若三气都为0但知道该过程特定气体（来自 meta），用总量按 GWP 反推该气体的 kt
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
    把 orchestrator 各模块输出统一到一个长表，并生成多粒度汇总；
    区分 CO2/CH4/N2O，并按 AR6 GWP100 计算 co2e_kt_ar6。
    """
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

    by_ctry_proc_comm = long.groupby(['country','process','commodity'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry_proc = long.groupby(['country','process'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum()
    by_ctry = long.groupby(['country'], as_index=False)[
        ['co2_kt','ch4_kt','n2o_kt','co2e_kt_ar6']].sum().rename(
            columns={'co2e_kt_ar6':'co2e_kt_total_ar6'}
        )

    return {'by_ctry_proc_comm': by_ctry_proc_comm,
            'by_ctry_proc': by_ctry_proc,
            'by_ctry': by_ctry,
            'long': long}

def summarize_market(model, var, universe, price_df: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    """
    从解出的变量抽取 Qs/Qd/Import/Export。price_df 可选：
    columns: country, iso3, year, commodity, price
    """
    rows = []
    try:
        Qs = var['Qs']; Qd = var['Qd']; Im = var['Import']; Ex = var['Export']
        for (i,j,t) in Qs.keys():
            rows.append((i, t, j,
                         Qs[i,j,t].X if hasattr(Qs[i,j,t], 'X') else np.nan,
                         Qd[i,j,t].X if hasattr(Qd[i,j,t], 'X') else np.nan,
                         Im[i,j,t].X if hasattr(Im[i,j,t], 'X') else np.nan,
                         Ex[i,j,t].X if hasattr(Ex[i,j,t], 'X') else np.nan))
    except Exception:
        return pd.DataFrame(columns=['country','year','commodity','Qs','Qd','Import','Export','price'])

    out = pd.DataFrame(rows, columns=['country','year','commodity','Qs','Qd','Import','Export'])
    if price_df is not None and len(price_df):
        out = out.merge(price_df[['country','year','commodity','price']],
                        on=['country','year','commodity'], how='left')
    return out
