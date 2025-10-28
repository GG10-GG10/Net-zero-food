# -*- coding: utf-8 -*-
from __future__ import annotations

# argparse disabled
import os
import sys
import traceback
import logging
import tempfile
import ctypes
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp

from S1_0_schema import ScenarioConfig, ScenarioData
# ---------------- configuration (edit here instead of CLI) ----------------
CFG = {
    'run_mode': 'BASE',  # options: 'single' (single scenario/BASE) | 'scenario' (batch from Scenario sheet) | 'mc' (Monte Carlo)
    'solve': True,
    'use_fao_modules': True,
    'premacc_e0': False,
    'mc_samples': 1000,
    'land_carbon_price': 10.0,
    'intake_indicator': 'energy',
    'future_last_only': True,
}
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    make_nodes_skeleton,
    apply_supply_ty_elasticity,
    apply_demand_elasticities_to_nodes,
    build_demand_components_from_fbs,
    apply_fbs_components_to_nodes,
    build_production_from_faostat,
    build_gce_activity_tables,
    build_livestock_stock_from_env,
    build_gv_areas_from_inputs,
    build_land_use_fires_timeseries,
    load_prices,
    load_intake_constraint,
    load_land_area_limits,
)
try:
    from S3_0_ds_emis_mc_full import build_model as build_model  # prefer full model if available
except Exception as exc:
    print("无法导入 S3_0_ds_emis_mc_full.build_model，原因如下：", file=sys.stderr)
    print(f"{exc}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
from S3_5_land_use_change import compute_luc_areas, LUCConfig
from S3_1_emissions_orchestrator_fao import EmissionsFAO, FAOPaths
from S4_1_results import summarize_emissions, summarize_emissions_from_detail, summarize_market
from S3_6_scenarios import (
    load_scenarios,
    apply_scenario_to_data,
    load_mc_specs,
    draw_mc_to_params,
)
from config_paths import get_input_base, get_results_base

paths = DataPaths()

# ---------------- helpers ----------------

def _is_lulucf_process(name: str, meta_map: dict) -> bool:
    mm = (meta_map or {}).get(name, {})
    cat = str(mm.get('category') or mm.get('Category') or mm.get('sector') or '').lower()
    name_l = (name or '').lower()
    return ('lulu' in cat) or ('forest' in name_l) or ('savanna' in name_l) or ('drained organic' in name_l) or ('net forest' in name_l)

import pickle
def _load_macc_df(path: str = os.path.join(get_input_base(), 'MACC-Global-US.pkl')) -> pd.DataFrame:
    try:
        return pickle.load(open(path, 'rb'))
    except Exception:
        return pd.DataFrame()

def _macc_fraction_for(price: float, df: pd.DataFrame, country: str, process: str) -> float:
    """Return cumulative abatement fraction for given land carbon price, country, and process."""
    if df is None or len(df) == 0 or price <= 0:
        return 0.0
    z = df[(df['Country'] == country) & (df['Process'] == process)]
    if not len(z):
        return 0.0
    z = z.sort_values('marginal_cost_$per_tco2e')
    z2 = z[z['marginal_cost_$per_tco2e'] <= price]
    if not len(z2):
        return 0.0
    return float(z2['cumulative_fraction_of_process'].max())

model_logger: Optional[logging.Logger] = None


def _windows_short_path(path: Path) -> Optional[Path]:
    if os.name != 'nt':
        return None
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    except AttributeError:
        return None
    buffer_len = 32768
    buffer = ctypes.create_unicode_buffer(buffer_len)
    result = get_short_path(str(path), buffer, buffer_len)
    if result == 0:
        return None
    return Path(buffer.value)


def _configure_model_logger(log_dir: Path, scenario_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = f"nzf_model_{scenario_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    # Ensure a fresh log file each run
    model_log_file = log_dir / "model.log"
    try:
        if model_log_file.exists():
            model_log_file.unlink()
    except Exception:
        # Best-effort cleanup; proceed even if deletion fails
        pass
    handler = logging.FileHandler(model_log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def _log_step(message: str, *, print_console: bool = True) -> None:
    if print_console:
        print(f"[NZF] {message}")
    if model_logger:
        model_logger.info(message)

def build_detailed_outputs(data: ScenarioData,
                           var: Dict[str, Dict],
                           *,
                           model_status: Optional[int],
                           population_map: Dict[Tuple[str,int], float],
                           energy_map: Dict[str, float],
                           protein_map: Dict[str, float],
                           fertilizers_df: Optional[pd.DataFrame],
                           land_areas_df: Optional[pd.DataFrame],
                           land_cp_by_year: Dict[int, float]) -> Dict[str, pd.DataFrame]:
    allowed_status = {None, gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT,
                      gp.GRB.INTERRUPTED, gp.GRB.USER_OBJ_LIMIT, gp.GRB.INF_OR_UNBD}
    if model_status not in allowed_status:
        _log_step(f"\u6a21\u578b\u72b6\u6001 {model_status} \u975e\u53ef\u7528\u72b6\u6001\uff0c\u8df3\u8fc7\u8be6\u7ec6\u8f93\u51fa")
        return {}
    if model_status not in (None, gp.GRB.OPTIMAL):
        _log_step(f"\u6a21\u578b\u72b6\u6001 {model_status} \u975e\u6700\u4f18\uff0c\u6309\u5f53\u524d\u89e3\u5199\u51fa\u7ed3\u679c")
    Qs = var.get('Qs'); Qd = var.get('Qd')
    Pc = var.get('Pc', {}); Pnet = var.get('Pnet', {})
    W = var.get('W', {}); Cij = var.get('C', {}); Eij = var.get('E', {})
    if Qs is None or Qd is None:
        return {}

    def _val(v):
        try:
            return float(v.X)
        except Exception:
            return np.nan

    def _baseline_float(val, default=np.nan):
        try:
            if val is None:
                return default
            out = float(val)
            if np.isnan(out):
                return default
            return out
        except Exception:
            return default

    idx = {(n.country, n.commodity, n.year): n for n in data.nodes}
    fert_map: Dict[Tuple[str,int], float] = {}
    if isinstance(fertilizers_df, pd.DataFrame) and not fertilizers_df.empty:
        for r in fertilizers_df[['country','year','n_fert_t']].itertuples(index=False):
            try:
                fert_map[(str(r[0]), int(r[1]))] = float(r[2])
            except Exception:
                continue

    def _land_category(name: Any) -> Optional[str]:
        s = str(name).lower()
        if 'forest' in s:
            return 'forest_area_ha'
        if 'pasture' in s or 'grass' in s or 'grazing' in s or 'meadow' in s:
            return 'pasture_area_ha'
        if 'cropland' in s or 'arable' in s or 'cultivated' in s or 'crop' in s:
            return 'cropland_area_ha'
        return None

    land_summary_map: Dict[Tuple[str,int], Dict[str, float]] = defaultdict(dict)
    if isinstance(land_areas_df, pd.DataFrame) and not land_areas_df.empty:
        tmp = land_areas_df.copy()
        tmp['category'] = tmp['land_use'].apply(_land_category)
        tmp = tmp.dropna(subset=['category'])
        if len(tmp):
            agg = tmp.groupby(['country','year','category'], as_index=False)['area_ha'].sum()
            for r in agg.itertuples(index=False):
                try:
                    land_summary_map[(str(r.country), int(r.year))][str(r.category)] = float(r.area_ha)
                except Exception:
                    continue

    rows: List[Dict[str, Any]] = []
    supply_by = defaultdict(float)
    demand_by = defaultdict(float)
    cost_by = defaultdict(float)
    emis_by = defaultdict(float)
    energy_total = defaultdict(float)
    protein_total = defaultdict(float)
    population_by: Dict[Tuple[str,int], float] = {}
    emis_detail_rows: List[Dict[str, Any]] = []

    for key, node in idx.items():
        i, j, t = key
        s_var = Qs.get(key)
        d_var = Qd.get(key)
        if s_var is not None or d_var is not None:
            supply = _val(s_var)
            demand = _val(d_var)
            price = _val(Pc.get((j, t)))
            price_net = _val(Pnet.get(key))
            unit_cost = _val(W.get(key))
            abat_cost = _val(Cij.get(key))
            emis = _val(Eij.get(key))
        else:
            supply = _baseline_float(getattr(node, 'Q0', None), 0.0)
            demand = _baseline_float(getattr(node, 'D0', None), supply)
            price = _baseline_float(getattr(node, 'P0', None))
            price_net = price
            unit_cost = np.nan
            abat_cost = np.nan
            emis = np.nan

        imp = max(demand - supply, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
        exp = max(supply - demand, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
        supply_by[(i, t)] += 0.0 if not np.isfinite(supply) else supply
        demand_by[(i, t)] += 0.0 if not np.isfinite(demand) else demand
        cost_by[(i, t)] += 0.0 if not np.isfinite(abat_cost) else abat_cost
        emis_by[(i, t)] += 0.0 if not np.isfinite(emis) else emis
        e_factor = energy_map.get(j)
        if np.isfinite(demand) and e_factor is not None:
            energy_total[(i, t)] += demand * float(e_factor)
        p_factor = protein_map.get(j)
        if np.isfinite(demand) and p_factor is not None:
            protein_total[(i, t)] += demand * float(p_factor)

        pop_val = _baseline_float(population_map.get((i, t)), np.nan)
        if np.isfinite(pop_val):
            population_by[(i, t)] = pop_val

        land_price = land_cp_by_year.get(t) if land_cp_by_year else None
        row = {
            'country': i,
            'year': t,
            'commodity': j,
            'supply_t': supply,
            'demand_t': demand,
            'imports_t': imp,
            'exports_t': exp,
            'price_global': price,
            'price_net': price_net,
            'unit_abatement_cost': unit_cost,
            'abatement_cost_total': abat_cost,
            'emissions_total': emis,
            'yield0_t_per_ha': node.meta.get('yield0'),
            'land_carbon_price': float(land_price) if land_price is not None else np.nan,
            'population': pop_val,
        }
        rows.append(row)

        e0_map = getattr(node, 'e0_by_proc', {}) or {}
        if e0_map:
            safe_supply = supply if np.isfinite(supply) and supply >= 0.0 else max(_baseline_float(getattr(node, 'Q0', None), 0.0), 0.0)
            proc_emis: Dict[str, float] = {}
            for proc, intensity in e0_map.items():
                try:
                    val = float(intensity) * safe_supply
                except Exception:
                    val = 0.0
                if np.isfinite(val) and val != 0.0:
                    proc_emis[str(proc).strip()] = val
            base_total = sum(max(v, 0.0) for v in proc_emis.values())
            scale = 1.0
            if np.isfinite(emis) and base_total > 0:
                scale = float(emis) / base_total if base_total else 1.0
            for proc_name, raw_val in proc_emis.items():
                adj_val = raw_val * scale if base_total > 0 else raw_val
                emis_detail_rows.append({
                    'country': i,
                    'iso3': getattr(node, 'iso3', ''),
                    'year': t,
                    'commodity': j,
                    'process': proc_name,
                    'emissions_tco2e': float(adj_val) if np.isfinite(adj_val) else 0.0,
                })

    fert_eff_map: Dict[Tuple[str,int], float] = {}
    for key, total_supply in supply_by.items():
        fert_val = fert_map.get(key)
        if fert_val is not None and fert_val > 0:
            fert_eff_map[key] = total_supply / fert_val

    energy_pc = {}
    protein_pc = {}
    for key, total in energy_total.items():
        pop = population_by.get(key, _baseline_float(population_map.get(key), np.nan))
        if pop and pop > 0:
            energy_pc[key] = total / float(pop)
    for key, total in protein_total.items():
        pop = population_by.get(key, _baseline_float(population_map.get(key), np.nan))
        if pop and pop > 0:
            protein_pc[key] = total / float(pop)

    for row in rows:
        key = (row['country'], row['year'])
        row['fertilizer_input_n_t'] = fert_map.get(key, np.nan)
        row['fertilizer_efficiency_t_output_per_tN'] = fert_eff_map.get(key, np.nan)
        land_vals = land_summary_map.get(key, {})
        row['cropland_area_ha'] = land_vals.get('cropland_area_ha', np.nan)
        row['forest_area_ha'] = land_vals.get('forest_area_ha', np.nan)
        row['pasture_area_ha'] = land_vals.get('pasture_area_ha', np.nan)
        row['per_capita_kcal'] = energy_pc.get(key, np.nan)
        row['per_capita_protein_g'] = protein_pc.get(key, np.nan)
        if np.isnan(row.get('population', np.nan)):
            row['population'] = population_by.get(key, _baseline_float(population_map.get(key), np.nan))

    node_df = pd.DataFrame(rows)

    country_year_rows: List[Dict[str, Any]] = []
    for (country, year), total_supply in supply_by.items():
        total_demand = demand_by.get((country, year), np.nan)
        imports = max(total_demand - total_supply, 0.0) if np.isfinite(total_supply) and np.isfinite(total_demand) else np.nan
        exports = max(total_supply - total_demand, 0.0) if np.isfinite(total_supply) and np.isfinite(total_demand) else np.nan
        land_vals = land_summary_map.get((country, year), {})
        land_price = land_cp_by_year.get(year) if land_cp_by_year else None
        row = {
            'country': country,
            'year': year,
            'total_supply_t': total_supply,
            'total_demand_t': total_demand,
            'imports_t': imports,
            'exports_t': exports,
            'net_trade_t': total_supply - total_demand if np.isfinite(total_supply) and np.isfinite(total_demand) else np.nan,
            'fertilizer_input_n_t': fert_map.get((country, year), np.nan),
            'fertilizer_efficiency_t_output_per_tN': fert_eff_map.get((country, year), np.nan),
            'cropland_area_ha': land_vals.get('cropland_area_ha', np.nan),
            'forest_area_ha': land_vals.get('forest_area_ha', np.nan),
            'pasture_area_ha': land_vals.get('pasture_area_ha', np.nan),
            'total_abatement_cost': cost_by.get((country, year), np.nan),
            'total_emissions': emis_by.get((country, year), np.nan),
            'land_carbon_price': float(land_price) if land_price is not None else np.nan,
            'population': population_by.get((country, year), _baseline_float(population_map.get((country, year)), np.nan)),
            'energy_total_kcal': energy_total.get((country, year), np.nan),
            'per_capita_kcal': energy_pc.get((country, year), np.nan),
            'protein_total_g': protein_total.get((country, year), np.nan),
            'per_capita_protein_g': protein_pc.get((country, year), np.nan),
        }
        country_year_rows.append(row)

    nutrition_rows: List[Dict[str, Any]] = []
    keys = set(list(energy_total.keys()) + list(protein_total.keys()))
    for country, year in sorted(keys):
        nutrition_rows.append({
            'country': country,
            'year': year,
            'energy_total_kcal': energy_total.get((country, year), np.nan),
            'per_capita_kcal': energy_pc.get((country, year), np.nan),
            'protein_total_g': protein_total.get((country, year), np.nan),
            'per_capita_protein_g': protein_pc.get((country, year), np.nan),
            'population': population_by.get((country, year), _baseline_float(population_map.get((country, year)), np.nan)),
        })

    land_rows: List[Dict[str, Any]] = []
    for (country, year), vals in land_summary_map.items():
        land_rows.append({
            'country': country,
            'year': year,
            'cropland_area_ha': vals.get('cropland_area_ha', np.nan),
            'forest_area_ha': vals.get('forest_area_ha', np.nan),
            'pasture_area_ha': vals.get('pasture_area_ha', np.nan),
        })

    return {
        'node_detail': node_df,
        'country_year_summary': pd.DataFrame(country_year_rows),
        'nutrition_per_capita': pd.DataFrame(nutrition_rows),
        'land_use_summary': pd.DataFrame(land_rows),
        'emissions_detail': pd.DataFrame(emis_detail_rows),
    }

# -------------- main pipeline --------------

def run_one_pipeline(paths: DataPaths,
                     pre_macc_e0: bool = False,
                     *,
                     scenario_id: str = 'BASE',
                     scenario_params: Optional[Dict[str, Any]] = None,
                     solve: bool = False,
                     use_fao_modules: bool = False,
                     save_root: Optional[str] = None,
                     future_last_only: bool = False) -> str:
    """Run one scenario (or MC draw) end-to-end and write results under {save_root}/{scenario_id}/"""
    global model_logger
    if save_root is None:
        outdir = Path(get_results_base(scenario_id))
    else:
        save_root_path = Path(save_root)
        outdir = save_root_path if save_root_path.name == scenario_id else save_root_path / scenario_id
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = outdir / "Log"
    if model_logger:
        for handler in list(model_logger.handlers):
            model_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    model_logger = _configure_model_logger(log_dir, scenario_id)
    model_log_path = log_dir / "model.log"
    _log_step(f"开始运行情景 {scenario_id}")
    _log_step(f"日志文件路径：{log_dir}")
    _log_step(f"模型日志文件：{model_log_path}")
    # 1) Universe & nodes
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
    nodes = make_nodes_skeleton(universe)
    EXCLUDE_COMMS = {'1', '2'}
    if EXCLUDE_COMMS:
        nodes = [n for n in nodes if str(getattr(n, 'commodity', '')).strip() not in EXCLUDE_COMMS]
    if future_last_only:
        future_years = [y for y in universe.years if y > 2020]
        last_year = max(future_years) if len(future_years) else None
        keep = set([y for y in universe.years if y <= 2020] + ([last_year] if last_year is not None else []))
        nodes = [n for n in nodes if n.year in keep]
    data = ScenarioData(nodes=nodes, universe=universe, config=cfg)
    active_years = sorted({n.year for n in data.nodes})
    _log_step("\u5df2\u6784\u5efa Universe \u4e0e\u8282\u70b9")
    
    # 2) Elasticities
    apply_supply_ty_elasticity(data.nodes, paths.elasticity_xlsx)
    # Demand cross-price & population elasticities
    apply_demand_elasticities_to_nodes(data.nodes, universe, paths.elasticity_xlsx)
    # Temperature multiplier
    try:
        from S2_0_load_data import apply_temperature_multiplier_to_nodes
        apply_temperature_multiplier_to_nodes(paths.temperature_xlsx, data.nodes)
    except Exception:
        pass
    _log_step("\u5df2\u52a0\u8f7d\u4f9b\u9700\u5f39\u6027\u4e0e\u6e29\u5ea6\u4e58\u5b50")

    # 3) Activities (S2 auto-build)
    production_df = build_production_from_faostat(paths.production_faostat_csv, universe)
    prod_lookup: Dict[Tuple[str, str, int], float] = {}
    latest_hist_prod: Dict[Tuple[str, str], Tuple[int, float]] = {}
    if len(production_df):
        for r in production_df.itertuples(index=False):
            try:
                country = str(getattr(r, 'country'))
                commodity = str(getattr(r, 'commodity'))
                year = int(getattr(r, 'year'))
                value = float(getattr(r, 'production_t') or 0.0)
            except Exception:
                continue
            prod_lookup[(country, commodity, year)] = value
            if year <= cfg.years_hist_end and value > 0:
                key = (country, commodity)
                prev = latest_hist_prod.get(key)
                if prev is None or year >= prev[0]:
                    latest_hist_prod[key] = (year, value)
        for n in data.nodes:
            val = prod_lookup.get((n.country, n.commodity, n.year))
            if val is None and n.year > cfg.years_hist_end:
                hist = latest_hist_prod.get((n.country, n.commodity))
                if hist:
                    val = hist[1]
            if val is not None:
                n.Q0 = float(val)
    gce_act = build_gce_activity_tables(production_csv=paths.production_faostat_csv,
                                        fbs_csv=paths.fbs_csv,
                                        historical_fert_xlsx=paths.historical_fert_xlsx,
                                        universe=universe)
    data.gce_residues_df    = gce_act['residues_df']
    data.gce_burning_df     = gce_act['burning_df']
    data.gce_rice_df        = gce_act['rice_df']
    data.gce_fertilizers_df = gce_act['fertilizers_df']

    # 4) Demand from FBS (Food/Feed/Seed) with mapped commodities
    fbs_comp = build_demand_components_from_fbs(
        paths.fbs_csv,
        universe,
        production_lookup=prod_lookup,
        latest_hist_prod=latest_hist_prod,
    )
    apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)
    latest_hist_demand: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for n in data.nodes:
        if n.year <= cfg.years_hist_end:
            d0 = float(getattr(n, 'D0', 0.0) or 0.0)
            if d0 > 0.0:
                key = (n.country, n.commodity)
                prev = latest_hist_demand.get(key)
                if prev is None or n.year >= prev[0]:
                    latest_hist_demand[key] = (n.year, d0)
    for n in data.nodes:
        if n.year > cfg.years_hist_end:
            d0 = float(getattr(n, 'D0', 0.0) or 0.0)
            if d0 <= 0.0:
                prev = latest_hist_demand.get((n.country, n.commodity))
                if prev is not None and prev[1] > 0.0:
                    n.D0 = prev[1]
    _log_step("\u5df2\u5e94\u7528 FBS \u9700\u6c42\u62c6\u5206")

    latest_hist_supply: Dict[Tuple[str, str], float] = {k: v[1] for k, v in latest_hist_prod.items()}

    data.gle_livestock_stock_df = build_livestock_stock_from_env(paths.livestock_patterns_csv, universe)
    data.gos_areas_df           = build_gv_areas_from_inputs(paths.inputs_landuse_csv, universe)
    fires_df                    = build_land_use_fires_timeseries(paths.emis_fires_csv, universe)
    # 4.1) Derive yield0 (t/ha) from production & area for historical baseline and assign to nodes
    try:
        from S2_0_load_data import compute_yield_from_prod_area, assign_yield0_to_nodes
        ydf = compute_yield_from_prod_area(paths.production_faostat_csv, paths.inputs_landuse_csv, universe)
        assign_yield0_to_nodes(data.nodes, ydf, hist_start=cfg.years_hist_start, hist_end=cfg.years_hist_end)
    except Exception:
        pass
    _log_step("\u5df2\u6784\u5efa\u751f\u4ea7/\u571f\u5730/\u80a5\u6599\u7b49\u6d3b\u52a8\u6570\u636e")


    # 5) Apply scenario parameters to data (feed reduction, ruminant cap, land carbon price)
    scen = scenario_params or {}
    # feed efficiency (rate) 鈫?shrink D0 by (1 - r) for matched (i,j,t)
    feed_red = scen.get('feed_reduction_by', {})
    if feed_red:
        idx = {(n.country, n.commodity, n.year): n for n in data.nodes}
        for (i,j,t), r in feed_red.items():
            n = idx.get((i,j,t))
            if n is not None and r is not None and r > 0:
                n.D0 = float(n.D0) * max(0.0, 1.0 - float(r))

    # ruminant caps: store in node meta; hard constraints will be added later
    rumi_cap = scen.get('ruminant_intake_cap', {})
    if rumi_cap:
        for n in data.nodes:
            cap = rumi_cap.get((n.country, n.year))
            if cap is not None:
                n.meta['ruminant_cap_total'] = float(cap)

    # Optional: ef multipliers from MC 鈫?will apply after e0 is computed
    ef_mult = scen.get('ef_multiplier_by', {})
    _log_step("\u5df2\u5e94\u7528\u60c5\u666f\u53c2\u6570\u5230\u8282\u70b9")

    hist_end = cfg.years_hist_end
    MIN_BASELINE = 1e-6
    adjusted_nodes: List[Any] = []
    dropped_future = 0
    for n in data.nodes:
        if n.year > hist_end:
            key = (n.country, n.commodity)
            q0 = float(getattr(n, 'Q0', 0.0) or 0.0)
            d0 = float(getattr(n, 'D0', 0.0) or 0.0)
            if q0 <= 0.0:
                prev_q = latest_hist_supply.get(key)
                if prev_q is not None and prev_q > 0.0:
                    q0 = float(prev_q)
                    n.Q0 = q0
            if d0 <= 0.0:
                prev_d = latest_hist_demand.get(key)
                if prev_d is not None and prev_d[1] > 0.0:
                    d0 = float(prev_d[1])
                    n.D0 = d0
            if q0 <= 0.0 and d0 <= 0.0:
                dropped_future += 1
                continue
            if q0 <= 0.0:
                q0 = max(d0, MIN_BASELINE)
                n.Q0 = q0
            if d0 <= 0.0:
                d0 = max(q0, MIN_BASELINE)
                n.D0 = d0
        adjusted_nodes.append(n)
    if dropped_future:
        _log_step(f"\u672a\u6765\u8282\u70b9\u4e2d\u5220\u9664 {dropped_future} \u4e2a\u57fa\u51c6\u4e3a 0 \u7684\u8282\u70b9")
    data.nodes = adjusted_nodes
    nodes = data.nodes

    # 6) LUC (passes land carbon price for intensification proxy)
    land_cp_by_year: Dict[int, float] = scen.get('land_carbon_price_by_year', {})
    # Build demand_df for LUC
    crop_rows = [(n.country, n.year, n.commodity, float(n.D0 if n.D0 > 0 else n.Q0)) for n in data.nodes]
    crop_demand_df = pd.DataFrame(crop_rows, columns=['country', 'year', 'commodity', 'demand_t'])
    _ = compute_luc_areas(demand_df=crop_demand_df,
                          production_df=production_df,
                          cfg=LUCConfig(land_carbon_price_per_tco2=float(land_cp_by_year.get(2080, 0.0))))
    _log_step("\u5df2\u5b8c\u6210\u571f\u5730\u5229\u7528\u53d8\u5316\u524d\u7f6e\u8ba1\u7b97")

    # 7) Emission intensities from *_fao.py (strict)
    if use_fao_modules:
        md = Path(get_input_base())
        module_paths = {
            k: str(md / k) for k in [
                'gce_emissions_module_fao.py',
                'gle_emissions_module_fao.py',
                'gos_emissions_module_fao.py',
                'gf_emissions_module_fao.py',
                'gfe_emissions_module_fao.py',
                'lme_manure_module_fao.py'
            ]
        }
        from S2_0_load_data import attach_emission_factors_from_fao_modules, run_fao_modules_and_cache
        attach_emission_factors_from_fao_modules(
            data.nodes,
            params_wide=None,
            production_df=production_df,
            crop_activity={'residues': data.gce_residues_df,
                           'burning':  data.gce_burning_df,
                           'rice':     data.gce_rice_df,
                           'fertilizers': data.gce_fertilizers_df},
            livestock_activity={'year': 0,
                                'enteric': data.gle_livestock_stock_df,
                                'prp':     data.gle_livestock_stock_df,
                                'mm':      data.gle_livestock_stock_df,
                                'mas':     data.gle_livestock_stock_df},
            soils_activity={'year': 0, 'areas': data.gos_areas_df},
            forest_activity=None,
            module_paths=module_paths,
        )
        try:
            _fao_runs = run_fao_modules_and_cache(
                data.nodes,
                livestock_stock_df=data.gle_livestock_stock_df,
                module_paths=module_paths,
            )
        except Exception:
            _fao_runs = {}
        _log_step("\u5df2\u4ece FAO \u6a21\u5757\u8bfb\u53d6\u6392\u653e\u5f3a\u5ea6")
    else:
        _fao_runs = {}
        _log_step("\u8df3\u8fc7 FAO \u6a21\u5757\u6392\u653e\u5f3a\u5ea6\uff08use_fao_modules=False\uff09")

    # 7.1) Apply MC ef multipliers (if any)
    if ef_mult:
        for n in data.nodes:
            e0 = getattr(n, 'e0_by_proc', {}) or {}
            for p in list(e0.keys()):
                f = ef_mult.get((n.country, n.commodity, p, n.year))
                if f is not None:
                    e0[p] = max(0.0, float(e0[p]) * float(f))
            n.e0_by_proc = e0

    # 7.2) LULUCF 涓撳睘 MACC锛氭牴鎹湡鍦扮⒊浠锋寜鍥藉脳杩囩▼绠椻€滃彲鍑忔帓浠介鈥濓紝鎶樺噺 e0_p
    if pre_macc_e0:
        # NOTE: pre_macc_e0 涓?S3.0 鍐呴儴鈥滃湡鍦扮⒊浠访桳ULUCF 涓撳睘 MACC鈥濆簲浜岄€変竴锛?        # - pre_macc_e0=True锛氬湪寤烘ā鍓嶇洿鎺ユ寜 MACC 鎶樺噺 e0锛圫3.0 鍐呴儴 land_carbon_price 璇风疆 0锛?        # - pre_macc_e0=False锛氫笉鍋氬墠缃姌鍑忥紱鑻ラ渶 MACC锛岃鍦?S3.0 鐨?SolveOpt.land_carbon_price 涓祴鍊?        # 渚涘簲渚у崟浣嶅姞浠凤紙tax_unit = 浠锋牸 脳 e_land锛夊彲涓庝换涓€鏂规骞跺瓨锛屼笉鏋勬垚鍙岃銆?        macc_df = _load_macc_df(os.path.join(get_input_base(), 'MACC-Global-US.pkl'))
        lulucf_procs = [p for p in universe.processes if _is_lulucf_process(p, universe.process_meta)]
        if len(macc_df):
            for n in data.nodes:
                e0 = getattr(n, 'e0_by_proc', {}) or {}
                price = float(land_cp_by_year.get(n.year, 0.0))
                if price > 0 and e0:
                    for p in list(e0.keys()):
                        if p in lulucf_procs:
                            f = _macc_fraction_for(price, macc_df, n.country, p)  # 0..1
                            if f > 0:
                                e0[p] = max(0.0, float(e0[p]) * (1.0 - f))
                n.e0_by_proc = e0

    # 7.3) Convert land carbon price to supply-side unit tax adder: tax_unit = price * e_land
    if land_cp_by_year:
        lulucf_procs = [p for p in universe.processes if _is_lulucf_process(p, universe.process_meta)]
        for n in data.nodes:
            price = float(land_cp_by_year.get(n.year, 0.0))
            if price <= 0:
                continue
            e0 = getattr(n, 'e0_by_proc', {}) or {}
            e_land = sum(v for k, v in e0.items() if k in lulucf_procs)
            if e_land > 0:
                setattr(n, 'tax_unit', float(getattr(n, 'tax_unit', 0.0)) + price * e_land)

    # 8) Build model & add ruminant hard-cap constraints
    # Nutrition and land constraints for future years
    from S2_0_load_data import (
        load_population_wpp,
        load_nutrient_factors_from_dict_v3,
        load_intake_targets,
        build_nutrition_rhs_for_future,
        load_income_multipliers_from_sspdb,
    )
    pop_map = load_population_wpp(paths.population_wpp_csv, universe)
    intake_targets = load_intake_targets(paths.intake_constraint_xlsx, CFG['intake_indicator'])
    nut_rhs = build_nutrition_rhs_for_future(universe, pop_map, intake_targets)
    nutrient_energy_map = load_nutrient_factors_from_dict_v3(paths.dict_v3_path, 'energy')
    nutrient_protein_map = load_nutrient_factors_from_dict_v3(paths.dict_v3_path, 'protein')
    nutrient_map = load_nutrient_factors_from_dict_v3(paths.dict_v3_path, CFG['intake_indicator'])
    # Filter to food-related commodities using dict_v3 cat2, drop non-food categories if present
    try:
        cat2 = universe.item_cat2_by_commodity or {}
        nutrient_map = {k: v for k, v in nutrient_map.items() if str(cat2.get(k, '')).lower().find('non') == -1}
    except Exception:
        pass
    # Income multipliers from SSPDB (relative to 2020)
    inc_mult = load_income_multipliers_from_sspdb(paths.income_sspdb_xlsx, paths.sspdb_scenario, universe)
    land_limits = load_land_area_limits(paths.inputs_landuse_csv)
    _log_step("\u5df2\u52a0\u8f7d\u4eba\u53e3/\u8425\u517b/\u571f\u5730/\u6536\u5165\u9a71\u52a8")

    model, var = build_model(
        data,
        nutrition_rhs=nut_rhs if len(nut_rhs) else None,
        nutrient_per_unit_by_comm=nutrient_map if len(nutrient_map) else None,
        land_area_limits=land_limits,
        yield_t_per_ha_default=LUCConfig().yield_t_per_ha_default,
        land_carbon_price_by_year=scenario_params.get('land_carbon_price_by_year', {}) if scenario_params else {},
        population_by_country_year=pop_map,
        income_mult_by_country_year=inc_mult,
        macc_path=paths.macc_pkl,
    )
    gurobi_log_path = log_dir / "gurobi.log"
    # Ensure Gurobi log starts fresh
    try:
        if gurobi_log_path.exists():
            gurobi_log_path.unlink()
    except Exception:
        pass

    def _set_gurobi_log(path: Path) -> Optional[str]:
        try:
            model.Params.LogFile = str(path)
            return None
        except Exception:
            try:
                model.setParam('LogFile', str(path))
                return None
            except Exception as exc:
                return str(exc)

    def _try_paths(candidates: List[Tuple[str, Path]]) -> Optional[Path]:
        for label, candidate in candidates:
            err = _set_gurobi_log(candidate)
            if err is None:
                _log_step(label.format(path=candidate))
                return candidate
            _log_step(f"\u8bbe\u7f6e Gurobi \u65e5\u5fd7\u6587\u4ef6\u5931\u8d25\uff0c\u539f\u56e0: {err}\uff08\u5c1d\u8bd5\u8def\u5f84\uff1a{candidate}\uff09")
        return None

    candidate_paths: List[Tuple[str, Path]] = [("Gurobi \u65e5\u5fd7\u6587\u4ef6\uff1a{path}", gurobi_log_path)]
    if os.name == 'nt':
        short_full = _windows_short_path(gurobi_log_path)
        if short_full and short_full != gurobi_log_path:
            candidate_paths.append(("Gurobi \u65e5\u5fd7\u6587\u4ef6\uff08\u77ed\u8def\u5f84\uff09\uff1a{path}", short_full))
        else:
            short_parent = _windows_short_path(gurobi_log_path.parent)
            if short_parent:
                short_candidate = short_parent / gurobi_log_path.name
                if short_candidate != gurobi_log_path:
                    candidate_paths.append(("Gurobi \u65e5\u5fd7\u6587\u4ef6\uff08\u77ed\u8def\u5f84\uff09\uff1a{path}", short_candidate))

    fallback_path = Path(tempfile.gettempdir()) / f"gurobi_{scenario_id}.log"
    # Also clear fallback path if present
    try:
        if fallback_path.exists():
            fallback_path.unlink()
    except Exception:
        pass
    candidate_paths.append(("Gurobi 日志文件（临时目录）：{path}", fallback_path))

    gurobi_log_actual = _try_paths(candidate_paths)
    if gurobi_log_actual is None:
        _log_step("仍未成功写入 Gurobi 日志文件，请检查目录权限或 8.3 短路径设置")
    else:
        _log_step("已构建优化模型")
        try:
            model.write(os.path.join(log_dir, f"model_{scenario_id}.lp"))
        except Exception as exc:
            _log_step(f"写 LP 文件失败：{exc}")

    # Ruminant hard cap: sum_j Qd[i,j,t] <= cap(i,t)
    RUMINANT_COMMS = [
        'Meat of cattle with the bone, fresh or chilled',
        'Meat of buffalo, fresh or chilled',
        'Meat of sheep, fresh or chilled',
        'Meat of goat, fresh or chilled'
    ]
    try:
        Qd = var['Qd']
        countries = sorted(set(n.country for n in data.nodes))
        for i in countries:
            for t in universe.years:
                # find cap from any node meta (we stored same cap across ruminant nodes of that country-year)
                caps = [float(n.meta.get('ruminant_cap_total', 0.0))
                        for n in data.nodes if n.country == i and n.year == t]
                cap_val = max(caps) if caps else 0.0
                if cap_val > 0:
                    expr = gp.LinExpr()
                    for j in RUMINANT_COMMS:
                        try:
                            expr += Qd[i, j, t]
                        except Exception:
                            pass
                    model.addConstr(expr <= cap_val, name=f"ruminant_cap[{i},{t}]")
    except Exception:
        pass

    solve_status = None
    if solve:
        try:
            model.setParam('DualReductions', 0)
        except Exception:
            pass
        model.optimize()
        solve_status = model.Status
        _log_step(f"模型求解完成，状态={solve_status}")
        if solve_status != gp.GRB.OPTIMAL:
            try:
                model.write(os.path.join(log_dir, f"model_{scenario_id}_status{solve_status}.lp"))
            except Exception as exc:
                _log_step(f"写 status LP 失败：{exc}")
            try:
                model.setParam('InfUnbdInfo', 1)
            except Exception:
                pass
            if solve_status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                try:
                    pass
                   # model.computeIIS()
                   # model.write(os.path.join(log_dir, f"model_{scenario_id}.ilp"))
                   # _log_step('已生成 IIS 文件')
                except Exception as exc:
                    _log_step(f"写 IIS 失败：{exc}")
    else:
        solve_status = getattr(model, 'Status', None)

    allowed_emis_status = {None,
                           gp.GRB.OPTIMAL,
                           gp.GRB.SUBOPTIMAL,
                           gp.GRB.TIME_LIMIT,
                           gp.GRB.INTERRUPTED,
                           gp.GRB.USER_OBJ_LIMIT}
    can_emit = (solve_status in allowed_emis_status)
    emis_sum: Dict[str, pd.DataFrame] = {}

    if not can_emit:
        _log_step(f"\u6a21\u578b\u72b6\u6001={solve_status}\uff0c\u8df3\u8fc7 Emis \u6c47\u603b\uff0c\u4f46\u4fdd\u7559 DS \u8f93\u51fa")
        fao_results = None
    else:
        # 9) Run FAO orchestrator for reporting (kept for factor computations/cache)
        _log_step("\u5f00\u59cb\u7ec4\u88c5 FAO \u6392\u653e\u6570\u636e")
        fao = EmissionsFAO(FAOPaths(
            gce_params=os.path.join(get_input_base(), 'GCE_parameters.xlsx'),
            gfe_params=os.path.join(get_input_base(), 'GFE_parameters.xlsx'),
            gle_params=os.path.join(get_input_base(), 'GLE_parameters.xlsx'),
            gos_params=None,
            lme_params=os.path.join(get_input_base(), 'Livestock_Manure_parameters.xlsx')
        ))
        fao_results = fao.run_all(
            crop_activity={'residues_df': data.gce_residues_df,
                           'burning_df': data.gce_burning_df,
                           'rice_df': data.gce_rice_df,
                           'fertilizers_df': data.gce_fertilizers_df},
            livestock_stock=data.gle_livestock_stock_df,
            gv_areas=data.gos_areas_df,
            fires_df=fires_df,
            iso3_list=None,
            years=universe.years
        )
        _log_step("FAO \u6392\u653e\u6570\u636e\u7ec4\u88c5\u5b8c\u6210")

    price_df = load_prices(paths.prices_csv, universe) if os.path.exists(paths.prices_csv) else None
    _log_step("\u5f00\u59cb\u751f\u6210\u5e02\u573a\u6c47\u603b")
    market_sum = summarize_market(model if solve else None, var, universe, data, price_df=price_df)
    _log_step("\u5e02\u573a\u6c47\u603b\u5b8c\u6210")

    _log_step("\u5f00\u59cb\u6784\u9020\u8be6\u7ec6\u8f93\u51fa DataFrame")
    detailed_outputs = build_detailed_outputs(
        data,
        var,
        model_status=solve_status,
        population_map=pop_map,
        energy_map=nutrient_energy_map or {},
        protein_map=nutrient_protein_map or {},
        fertilizers_df=data.gce_fertilizers_df,
        land_areas_df=data.gos_areas_df,
        land_cp_by_year=land_cp_by_year,
    )
    _log_step("\u8be6\u7ec6\u8f93\u51fa DataFrame \u5df2\u751f\u6210")
    emission_detail_df = detailed_outputs.get('emissions_detail')
    if can_emit:
        if isinstance(emission_detail_df, pd.DataFrame) and len(emission_detail_df):
            _log_step("\u5f00\u59cb\u751f\u6210\u6392\u653e\u6c47\u603b")
            emis_sum = summarize_emissions_from_detail(emission_detail_df,
                                                       process_meta_map=universe.process_meta,
                                                       allowed_years=active_years)
            _log_step("\u6392\u653e\u6c47\u603b\u5b8c\u6210")
        elif fao_results is not None:
            _log_step("\u6392\u653e\u8be6\u7ec6\u6570\u636e\u4e3a\u7a7a\uff0c\u4f7f\u7528 FAO \u6a21\u5757\u8f93\u51fa\u6c47\u603b")
            emis_sum = summarize_emissions(fao_results, extra_emis=None,
                                           process_meta_map=universe.process_meta)
            if active_years:
                filt_years = set(active_years)
                for k, df in list(emis_sum.items()):
                    if isinstance(df, pd.DataFrame) and 'year' in df.columns:
                        emis_sum[k] = df[df['year'].isin(filt_years)].reset_index(drop=True)
            _log_step("\u6392\u653e\u6c47\u603b\u5b8c\u6210\uff08FAO \u8f93\u51fa\uff09")
        else:
            _log_step("\u6392\u653e\u6c47\u603b\u8df3\u8fc7\uff0c\u6ca1\u6709\u53ef\u7528\u7684\u6392\u653e\u6570\u636e")
    else:
        _log_step("\u5df2\u751f\u6210\u5e02\u573a\u548c DS \u6c47\u603b\uff08\u6392\u653e\u8df3\u8fc7\uff09")

# 10) Write outputs
    emis_dir = outdir / "Emis"
    ds_dir = outdir / "DS"
    emis_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    outdir_str = str(outdir)
    emis_dir_str = str(emis_dir)
    ds_dir_str = str(ds_dir)
    for k, df in emis_sum.items():
        df.to_csv(os.path.join(emis_dir_str, f"emis_{k}.csv"), index=False, encoding='utf-8-sig')
    if market_sum is not None and len(market_sum):
        market_sum.to_csv(os.path.join(ds_dir_str, "market_summary.csv"), index=False, encoding='utf-8-sig')
    for name, df in (detailed_outputs or {}).items():
        if df is not None and len(df):
            fname = {
                'node_detail': 'detailed_node_summary.csv',
                'country_year_summary': 'country_year_summary.csv',
                'nutrition_per_capita': 'nutrition_per_capita.csv',
                'land_use_summary': 'land_use_summary.csv',
            }.get(name, f"{name}.csv")
            df.to_csv(os.path.join(ds_dir_str, fname), index=False, encoding='utf-8-sig')

    if gurobi_log_actual and gurobi_log_actual != gurobi_log_path and gurobi_log_actual.exists():
        try:
            shutil.copyfile(gurobi_log_actual, gurobi_log_path)
            _log_step(f"Gurobi 日志已从 {gurobi_log_actual} 复制到 {gurobi_log_path}")
        except Exception as exc:
            _log_step(f"Gurobi 日志复制到目标目录失败：{exc}")

    _log_step(f"\u7ed3\u679c\u5199\u5165\u76ee\u5f55\uff1a{outdir_str}")
    if model_logger:
        for handler in model_logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    return outdir_str

# -------------- CLI --------------

def main():

    if CFG['run_mode']=='scenario':
        # Batch scenarios from Scenario sheet
        cfg = ScenarioConfig()
        universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
        effects = load_scenarios(paths.scenario_config_xlsx, universe, sheet='Scenario')
        ids = sorted({e.scenario_id for e in effects}) or ['BASE']
        for sid in ids:
            scen = apply_scenario_to_data(effects, sid, universe, make_nodes_skeleton(universe))
            run_one_pipeline(paths, pre_macc_e0=CFG['premacc_e0'], scenario_id=sid, scenario_params=scen,
                             solve=CFG['solve'], use_fao_modules=CFG['use_fao_modules'],
                             future_last_only=CFG['future_last_only'])
    elif CFG['run_mode']=='mc':
        # Monte Carlo from MC sheet using full-model cache
        cfg = ScenarioConfig()
        universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
        # Build base data once
        nodes = make_nodes_skeleton(universe)
        if CFG['future_last_only']:
            future_years = [y for y in universe.years if y > 2020]
            last_year = max(future_years) if len(future_years) else None
            keep = set([y for y in universe.years if y <= 2020] + ([last_year] if last_year is not None else []))
            nodes = [n for n in nodes if n.year in keep]
        data = ScenarioData(nodes=nodes, universe=universe, config=cfg)

        # Elasticities & drivers
        apply_supply_ty_elasticity(data.nodes, paths.elasticity_xlsx)
        apply_demand_elasticities_to_nodes(data.nodes, universe, paths.elasticity_xlsx)
        try:
            from S2_0_load_data import apply_temperature_multiplier_to_nodes
            apply_temperature_multiplier_to_nodes(paths.temperature_xlsx, data.nodes)
        except Exception:
            pass

        # Production baseline for MC cache
        production_df = build_production_from_faostat(paths.production_faostat_csv, universe)
        prod_lookup: Dict[Tuple[str, str, int], float] = {}
        latest_hist_prod: Dict[Tuple[str, str], Tuple[int, float]] = {}
        if len(production_df):
            for r in production_df.itertuples(index=False):
                try:
                    country = str(getattr(r, 'country'))
                    commodity = str(getattr(r, 'commodity'))
                    year = int(getattr(r, 'year'))
                    value = float(getattr(r, 'production_t') or 0.0)
                except Exception:
                    continue
                prod_lookup[(country, commodity, year)] = value
                if year <= cfg.years_hist_end and value > 0:
                    key = (country, commodity)
                    prev = latest_hist_prod.get(key)
                    if prev is None or year >= prev[0]:
                        latest_hist_prod[key] = (year, value)
            for n in data.nodes:
                val = prod_lookup.get((n.country, n.commodity, n.year))
                if val is None and n.year > cfg.years_hist_end:
                    hist = latest_hist_prod.get((n.country, n.commodity))
                    if hist:
                        val = hist[1]
                if val is not None:
                    n.Q0 = float(val)

        # Demand from FBS
        fbs_comp = build_demand_components_from_fbs(
            paths.fbs_csv,
            universe,
            production_lookup=prod_lookup,
            latest_hist_prod=latest_hist_prod,
        )
        apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)
        latest_hist_demand: Dict[Tuple[str, str], Tuple[int, float]] = {}
        for n in data.nodes:
            if n.year <= cfg.years_hist_end:
                d0 = float(getattr(n, 'D0', 0.0) or 0.0)
                if d0 > 0.0:
                    key = (n.country, n.commodity)
                    prev = latest_hist_demand.get(key)
                    if prev is None or n.year >= prev[0]:
                        latest_hist_demand[key] = (n.year, d0)
        for n in data.nodes:
            if n.year > cfg.years_hist_end:
                d0 = float(getattr(n, 'D0', 0.0) or 0.0)
                if d0 <= 0.0:
                    prev = latest_hist_demand.get((n.country, n.commodity))
                    if prev is not None and prev[1] > 0.0:
                        n.D0 = prev[1]

        # Nutrition RHS & land limits
        from S2_0_load_data import (
            load_population_wpp,
            load_intake_targets,
            build_nutrition_rhs_for_future,
            load_income_multipliers_from_sspdb,
        )
        pop_map = load_population_wpp(paths.population_wpp_csv, universe)
        intake_targets = load_intake_targets(paths.intake_constraint_xlsx, CFG['intake_indicator'])
        nut_rhs = build_nutrition_rhs_for_future(universe, pop_map, intake_targets)
        land_limits = load_land_area_limits(paths.inputs_landuse_csv)
        inc_mult = load_income_multipliers_from_sspdb(paths.income_sspdb_xlsx, paths.sspdb_scenario, universe)

        # Build base model cache
        from S3_0_ds_emis_mc_full import build_model_cache, run_mc
        cache = build_model_cache(data,
                                  nutrition_rhs=nut_rhs if len(nut_rhs) else None,
                                  nutrient_per_unit_by_comm=None,
                                  land_area_limits=land_limits,
                                  land_carbon_price_by_year=None,
                                  population_by_country_year=pop_map,
                                  income_mult_by_country_year=inc_mult,
                                  macc_path=paths.macc_pkl)

        # Load specs & run MC
        specs = load_mc_specs(paths.scenario_config_xlsx)
        run_mc(data, universe, specs, n_samples=int(CFG['mc_samples']), seed=42,
               save_prefix=os.path.join(get_results_base(), 'MC', 'mc_full'))
    else:
        # Single quick run (optionally with a constant land carbon price)
        sid = "BASE"
        lcp = float(CFG['land_carbon_price'] or 0.0)
        scen = {'land_carbon_price_by_year': {y: lcp for y in range(2010, 2081, 10)}}
        run_one_pipeline(paths, pre_macc_e0=CFG['premacc_e0'], scenario_id=sid, scenario_params=scen,
                         solve=CFG['solve'], use_fao_modules=CFG['use_fao_modules'],
                         future_last_only=CFG['future_last_only'])

if __name__ == '__main__':
    main()


