from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

# Processes/GHG names must follow dict_v3 definitions
ALLOWED_PROCESSES = ('Savanna fire', 'Peatlands fire')
ALLOWED_GHG = ('CH4', 'N2O', 'CO2')
YEAR_START = 2010
YEAR_END = 2020
YEAR_COLUMNS = [f'Y{year}' for year in range(YEAR_START, YEAR_END + 1)]
GWP_LOOKUP = {'CH4': 28.0, 'N2O': 265.0, 'CO2': 1.0}


def _load_process_item_map(dict_v3_path: str) -> Dict[str, str]:
    """
    Load the Process -> Item_Emis mapping for Savanna/Peatlands fire from dict_v3.
    Falls back to using the process name itself when dict_v3 is unavailable.
    """
    default_mapping = {proc: proc for proc in ALLOWED_PROCESSES}
    if not dict_v3_path or not os.path.exists(dict_v3_path):
        return default_mapping

    try:
        emis_item_df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
    except Exception:
        return default_mapping

    emis_item_df = emis_item_df[emis_item_df['Process'].isin(ALLOWED_PROCESSES)]
    emis_item_df = emis_item_df[['Process', 'Item_Emis']].dropna().drop_duplicates()
    mapping = {row['Process']: row['Item_Emis'] for _, row in emis_item_df.iterrows()}
    return {**default_mapping, **mapping}


def _normalize_m49(value: object) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    return text if text.startswith("'") else f"'{text.zfill(3)}"


def _select_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce').fillna(0.0)
    return numeric == 1.0


def _safe_country_lookup(m49: pd.Series,
                         country_by_m49: Optional[Dict[str, str]]) -> pd.Series:
    if not country_by_m49:
        return pd.Series([None] * len(m49), index=m49.index)
    return m49.map(country_by_m49)


def _safe_iso_lookup(country: pd.Series,
                     iso3_by_country: Optional[Dict[str, str]]) -> pd.Series:
    if not iso3_by_country:
        return pd.Series([None] * len(country), index=country.index)
    return country.map(iso3_by_country)


def _extend_to_future_years(df: pd.DataFrame,
                            future_years: Iterable[int]) -> pd.DataFrame:
    future_years = sorted({y for y in future_years if y > YEAR_END})
    if not future_years or df.empty:
        return pd.DataFrame(columns=df.columns)

    base_df = df[df['year'] == YEAR_END].copy()
    if base_df.empty:
        return pd.DataFrame(columns=df.columns)

    replicated = []
    for future_year in future_years:
        tmp = base_df.copy()
        tmp['year'] = future_year
        replicated.append(tmp)
    return pd.concat(replicated, ignore_index=True) if replicated else pd.DataFrame(columns=df.columns)


def load_fixed_gfire_emissions(excel_path: str,
                               dict_v3_path: str,
                               active_years: Optional[Sequence[int]],
                               *,
                               country_by_m49: Optional[Dict[str, str]] = None,
                               iso3_by_country: Optional[Dict[str, str]] = None
                               ) -> Dict[str, pd.DataFrame]:
    """
    Load historical Savanna/Peatlands fire emissions, convert to kt CO2e,
    and extend future years by repeating Y2020 values for each (M49, Process, GHG).
    """
    if not excel_path or not os.path.exists(excel_path):
        raise FileNotFoundError(f"Emission workbook not found: {excel_path}")

    raw_df = pd.read_excel(excel_path)
    required_cols = {'Select', 'Land Category', 'Species', 'M49_Country_Code'}
    missing_cols = required_cols - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {excel_path}: {missing_cols}")

    year_cols = [col for col in YEAR_COLUMNS if col in raw_df.columns]
    if not year_cols:
        raise ValueError(f"No Y{YEAR_START}-Y{YEAR_END} columns found in {excel_path}")

    select_mask = _select_mask(raw_df['Select'])
    filtered = raw_df[select_mask &
                      raw_df['Land Category'].isin(ALLOWED_PROCESSES) &
                      raw_df['Species'].isin(ALLOWED_GHG)].copy()
    if filtered.empty:
        empty_cols = ['M49_Country_Code', 'Country', 'iso3', 'Process',
                      'Item', 'GHG', 'year', 'value']
        empty_df = pd.DataFrame(columns=empty_cols)
        return {
            'historical': empty_df.copy(),
            'future_extension': empty_df.copy(),
            'combined': empty_df.copy(),
        }

    filtered['M49_Country_Code'] = filtered['M49_Country_Code'].apply(_normalize_m49)
    process_item_map = _load_process_item_map(dict_v3_path)

    id_vars = ['M49_Country_Code', 'Land Category', 'Species']
    long_df = filtered.melt(id_vars=id_vars,
                            value_vars=year_cols,
                            var_name='year',
                            value_name='value_mt')
    long_df['year'] = long_df['year'].astype(str).str.lstrip('Y').astype(int)
    long_df = long_df[(long_df['year'] >= YEAR_START) & (long_df['year'] <= YEAR_END)]

    long_df['value_mt'] = pd.to_numeric(long_df['value_mt'], errors='coerce').fillna(0.0)
    long_df['Process'] = long_df['Land Category'].astype(str)
    long_df['Item'] = long_df['Process'].map(process_item_map).fillna(long_df['Process'])
    long_df['GHG'] = long_df['Species'].astype(str)

    long_df['Country'] = _safe_country_lookup(long_df['M49_Country_Code'], country_by_m49)
    if 'UNFCCC country/GROUP' in filtered.columns:
        fallback_country = (
            filtered[['M49_Country_Code', 'UNFCCC country/GROUP']]
            .dropna(subset=['UNFCCC country/GROUP'])
            .drop_duplicates(subset=['M49_Country_Code'])
            .set_index('M49_Country_Code')['UNFCCC country/GROUP']
        )
        if not fallback_country.empty:
            mapped = long_df['M49_Country_Code'].map(fallback_country)
            long_df['Country'] = long_df['Country'].fillna(mapped)
    long_df['iso3'] = _safe_iso_lookup(long_df['Country'], iso3_by_country)

    # Convert Mt to kt only (no GWP conversion - output should be raw GHG mass in kt,
    # consistent with other emission modules like GCE and GLE)
    long_df['value'] = long_df['value_mt'] * 1000.0  # convert Mt to kt

    base_columns = ['M49_Country_Code', 'Country', 'iso3', 'Process', 'Item', 'GHG', 'year', 'value']
    historical_df = long_df[base_columns].copy()
    historical_df['module'] = 'GFIRE_FIXED'

    future_years = list(active_years or [])
    future_df = _extend_to_future_years(historical_df, future_years)
    if not future_df.empty:
        future_df['module'] = 'GFIRE_FIXED'

    combined_df = pd.concat([historical_df, future_df], ignore_index=True) \
        if not historical_df.empty or not future_df.empty \
        else pd.DataFrame(columns=historical_df.columns if len(historical_df.columns) else base_columns + ['module'])

    return {
        'historical': historical_df,
        'future_extension': future_df,
        'combined': combined_df,
    }
