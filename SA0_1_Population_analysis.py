# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Dict, Any

import numpy as np
import pandas as pd

from config_paths import get_results_base
from S1_0_schema import ScenarioConfig
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    load_population_wpp,
)


def _safe_float(value: Any) -> float:
    """Convert to float and guard against non-finite values."""
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


def build_population_growth_table(paths: DataPaths,
                                  base_year: int,
                                  analysis_years: Sequence[int]) -> pd.DataFrame:
    """Prepare per-country growth rates relative to ``base_year``."""
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
    pop_map = load_population_wpp(paths.population_wpp_csv, universe)
    target_years = [y for y in sorted(set(analysis_years)) if y != base_year]
    rate_year = base_year + 60
    cagr_col = f"compound_rate_{base_year}_{rate_year}"

    records = []
    for country in sorted(universe.countries):
        base_pop = _safe_float(pop_map.get((country, base_year)))
        if not np.isfinite(base_pop) or base_pop <= 0:
            continue
        row: Dict[str, Any] = {
            "M49_Country_Code": _format_m49(universe.m49_by_country.get(country)),
            "Country": country,
            "Base_Year": base_year,
            "Population_Base": base_pop,
        }
        for year in target_years:
            pop_val = _safe_float(pop_map.get((country, year)))
            col = f"growth_rate_{year}"
            if np.isfinite(pop_val):
                row[col] = (pop_val / base_pop) - 1.0
            else:
                row[col] = np.nan

        pop_for_rate = _safe_float(pop_map.get((country, rate_year)))
        if np.isfinite(pop_for_rate):
            row[cagr_col] = (pop_for_rate / base_pop) ** (1.0 / 60.0) - 1.0
        else:
            row[cagr_col] = np.nan

        records.append(row)

    if not records:
        return pd.DataFrame()

    # ensure consistent column order
    growth_cols = [f"growth_rate_{year}" for year in target_years]
    ordered_cols = ["M49_Country_Code", "Country", "Base_Year", "Population_Base", *growth_cols, cagr_col]
    df = pd.DataFrame(records)
    existing_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]
    df = df.sort_values(by=["M49_Country_Code", "Country"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze population growth relative to the model baseline year."
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="Baseline year used in the main model (default: 2020).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
        help="Future years to compare against the base year.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="BASE",
        help="Scenario/output folder name under the output directory (default: BASE).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="SA0_1_Population_analysis.xlsx",
        help="Output filename (written under output/<scenario>/Analysis unless an absolute path is provided).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DataPaths()
    df = build_population_growth_table(paths, args.base_year, args.years)
    if df.empty:
        print("No population data found for the requested configuration.")
        return

    analysis_dir = Path(get_results_base(args.scenario_id)) / "Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = analysis_dir / output_path

    df.to_excel(output_path, index=False)
    print(f"Population growth analysis saved to {output_path}")


if __name__ == "__main__":
    main()
