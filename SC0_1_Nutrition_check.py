# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, Iterable

import numpy as np
import pandas as pd

from config_paths import get_results_base
from S1_0_schema import ScenarioConfig
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    load_population_wpp,
    load_nutrient_factors_from_dict_v3,
    load_intake_targets,
)

ANALYSIS_YEARS = list(range(2010, 2021))


def _safe_float(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def _format_m49(code: Any) -> str:
    if code in (None, "", np.nan):
        return ""
    try:
        return f"'{int(float(code)):03d}"  # ✅ 'xxx格式
    except Exception:
        return str(code).strip()


def _load_node_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find detailed_node_summary file at: {path}")
    df = pd.read_csv(path)
    required_cols = {'country', 'year', 'commodity', 'demand_t'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"detailed_node_summary is missing required columns: {sorted(missing)}")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df[df['year'].between(min(ANALYSIS_YEARS), max(ANALYSIS_YEARS))]
    df['demand_t'] = pd.to_numeric(df['demand_t'], errors='coerce')
    return df


def _aggregate_qd(df: pd.DataFrame,
                  nutrient_map: Dict[str, float]) -> Dict[Tuple[str, int], float]:
    if df.empty or not nutrient_map:
        return {}
    tmp = df[['country', 'year', 'commodity', 'demand_t']].copy()
    tmp['factor'] = tmp['commodity'].map(nutrient_map)
    tmp = tmp.dropna(subset=['factor'])
    tmp['value'] = pd.to_numeric(tmp['demand_t'], errors='coerce') * pd.to_numeric(tmp['factor'], errors='coerce')
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=['value'])
    grouped = tmp.groupby(['country', 'year'])['value'].sum()
    return {(str(country), int(year)): float(val) for (country, year), val in grouped.items()}


def _build_rhs(intake_targets: Dict[str, float],
               population_map: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    rhs: Dict[Tuple[str, int], float] = {}
    if not intake_targets or not population_map:
        return rhs
    for (country, year), pop in population_map.items():
        if year not in ANALYSIS_YEARS:
            continue
        target = intake_targets.get(country)
        if target is None:
            continue
        try:
            rhs[(country, year)] = float(target) * 365.0 * float(pop)
        except Exception:
            continue
    return rhs


def _build_sheet(countries: Iterable[str],
                 m49_map: Dict[str, str],
                 qd_totals: Dict[Tuple[str, int], float],
                 rhs_totals: Dict[Tuple[str, int], float]) -> pd.DataFrame:
    rows = []
    for country in sorted(set(countries)):
        row = {
            "M49_Country_Code": _format_m49(m49_map.get(country)),
            "Country": country,
        }
        for year in ANALYSIS_YEARS:
            qd_val = qd_totals.get((country, year), np.nan)
            rhs_val = rhs_totals.get((country, year), np.nan)
            row[f"Y{year}_Qd"] = qd_val
            row[f"Y{year}_RHS"] = rhs_val
        rows.append(row)
    return pd.DataFrame(rows)


def _build_per_capita_sheet(countries: Iterable[str],
                            m49_map: Dict[str, str],
                            qd_totals: Dict[Tuple[str, int], float],
                            population_map: Dict[Tuple[str, int], float]) -> pd.DataFrame:
    rows = []
    for country in sorted(set(countries)):
        row = {
            "M49_Country_Code": _format_m49(m49_map.get(country)),
            "Country": country,
        }
        for year in ANALYSIS_YEARS:
            qd_val = qd_totals.get((country, year), np.nan)
            pop = population_map.get((country, year), np.nan)
            if np.isfinite(qd_val) and np.isfinite(pop) and pop > 0:
                row[f"Y{year}"] = qd_val / pop / 365.0
            else:
                row[f"Y{year}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare historical Qd nutrient totals against nutrition RHS requirements."
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="BASE",
        help="Scenario/output folder name under the output directory (default: BASE).",
    )
    parser.add_argument(
        "--node-file",
        type=str,
        default="",
        help="Optional explicit path to detailed_node_summary.csv. "
             "When omitted, the script reads output/<scenario>/DS/detailed_node_summary.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="SC0_1_Nutrition_check.xlsx",
        help="Output filename (written under output/<scenario>/Analysis unless absolute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DataPaths()
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)

    ds_dir = Path(get_results_base(args.scenario_id)) / "DS"
    node_file = Path(args.node_file) if args.node_file else ds_dir / "detailed_node_summary.csv"
    node_df = _load_node_detail(node_file)

    pop_map = load_population_wpp(paths.population_wpp_csv, universe)
    pop_map = {(country, year): val for (country, year), val in pop_map.items() if year in ANALYSIS_YEARS}

    nutrient_maps = {
        'energy': load_nutrient_factors_from_dict_v3(paths.dict_v3_path, 'energy'),
        'protein': load_nutrient_factors_from_dict_v3(paths.dict_v3_path, 'protein'),
        'fat': load_nutrient_factors_from_dict_v3(paths.dict_v3_path, 'fat'),
    }
    intake_targets = {
        'energy': load_intake_targets(paths.intake_constraint_xlsx, 'energy'),
        'protein': load_intake_targets(paths.intake_constraint_xlsx, 'protein'),
        'fat': load_intake_targets(paths.intake_constraint_xlsx, 'fat'),
    }

    qd_totals = {key: _aggregate_qd(node_df, nutrient_maps[key]) for key in nutrient_maps}
    rhs_totals = {key: _build_rhs(intake_targets[key], pop_map) for key in intake_targets}

    all_countries = set(node_df['country'].astype(str).unique())
    all_countries.update(country for country, _ in pop_map.keys())

    m49_map = {country: universe.m49_by_country.get(country) for country in all_countries}

    sheets = {
        'Energy': _build_sheet(all_countries, m49_map, qd_totals['energy'], rhs_totals['energy']),
        'Protein': _build_sheet(all_countries, m49_map, qd_totals['protein'], rhs_totals['protein']),
        'Fat': _build_sheet(all_countries, m49_map, qd_totals['fat'], rhs_totals['fat']),
        'energy_RHS_recal': _build_per_capita_sheet(all_countries, m49_map, qd_totals['energy'], pop_map),
        'protein_RHS_recal': _build_per_capita_sheet(all_countries, m49_map, qd_totals['protein'], pop_map),
        'fat_RHS_recal': _build_per_capita_sheet(all_countries, m49_map, qd_totals['fat'], pop_map),
    }

    analysis_dir = Path(get_results_base(args.scenario_id)) / "Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = analysis_dir / output_path

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Nutrition check workbook saved to {output_path}")


if __name__ == "__main__":
    main()
