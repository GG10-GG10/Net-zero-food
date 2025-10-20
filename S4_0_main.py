# -*- coding: utf-8 -*-
from __future__ import annotations

# argparse disabled
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import gurobipy as gp

from S1_0_schema import ScenarioConfig, ScenarioData
# ---------------- configuration (edit here instead of CLI) ----------------
CFG = {
    'run_mode': 'single',
    'solve': True,
    'use_fao_modules': False,
    'premacc_e0': False,
    'mc_samples': 100,
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
except Exception:
    from S3_0_ds_emis_mc import build_model  # fallback simple SD model
from S3_5_land_use_change import compute_luc_areas, LUCConfig
from S3_1_emissions_orchestrator_fao import EmissionsFAO, FAOPaths
from S4_1_results import summarize_emissions, summarize_market
from S3_6_scenarios import (
    load_scenarios,
    apply_scenario_to_data,
    load_mc_specs,
    draw_mc_to_params,
)
from config_paths import get_input_base, get_results_base

# ---------------- helpers ----------------

try:
    from S3_0_ds_emis_mc_full import build_model as build_model
except Exception:
    from S3_0_ds_emis_mc import build_model
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

# -------------- main pipeline --------------

def run_one_pipeline(paths: DataPaths,
                     pre_macc_e0: bool = False,
                     *,
                     scenario_id: str = 'BASE',
                     scenario_params: Optional[Dict[str, Any]] = None,
                     solve: bool = False,
                     use_fao_modules: bool = False,
                     save_root: str = get_results_base(),
                     future_last_only: bool = False) -> str:
    """Run one scenario (or MC draw) end-to-end and write results under {save_root}/{scenario_id}/"""
    # 1) Universe & nodes
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
    nodes = make_nodes_skeleton(universe)
    if future_last_only:
        future_years = [y for y in universe.years if y > 2020]
        last_year = max(future_years) if len(future_years) else None
        keep = set([y for y in universe.years if y <= 2020] + ([last_year] if last_year is not None else []))
        nodes = [n for n in nodes if n.year in keep]
    data = ScenarioData(nodes=nodes, universe=universe, config=cfg)

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

    # 3) Demand from FBS (Food/Feed/Seed)
    fbs_comp = build_demand_components_from_fbs(paths.fbs_csv, universe)
    apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)

    # 4) Activities (S2 auto-build)
    production_df = build_production_from_faostat(paths.production_faostat_csv, universe)
    gce_act = build_gce_activity_tables(production_csv=paths.production_faostat_csv,
                                        fbs_csv=paths.fbs_csv,
                                        historical_fert_xlsx=paths.historical_fert_xlsx,
                                        universe=universe)
    data.gce_residues_df    = gce_act['residues_df']
    data.gce_burning_df     = gce_act['burning_df']
    data.gce_rice_df        = gce_act['rice_df']
    data.gce_fertilizers_df = gce_act['fertilizers_df']

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

    # 6) LUC (passes land carbon price for intensification proxy)
    land_cp_by_year: Dict[int, float] = scen.get('land_carbon_price_by_year', {})
    # Build demand_df for LUC
    crop_rows = [(n.country, n.year, n.commodity, float(n.D0 if n.D0 > 0 else n.Q0)) for n in data.nodes]
    crop_demand_df = pd.DataFrame(crop_rows, columns=['country', 'year', 'commodity', 'demand_t'])
    _ = compute_luc_areas(demand_df=crop_demand_df,
                          production_df=production_df,
                          cfg=LUCConfig(land_carbon_price_per_tco2=float(land_cp_by_year.get(2080, 0.0))))

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
    pop_map = load_population_wpp(paths.population_wpp_csv)
    intake_targets = load_intake_targets(paths.intake_constraint_xlsx, CFG['intake_indicator'])
    nut_rhs = build_nutrition_rhs_for_future(universe, pop_map, intake_targets)
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

    if solve:
        model.optimize()

    # 9) Run FAO orchestrator for reporting + summarize (with process_meta to infer gases if needed)
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

    emis_sum = summarize_emissions(fao_results, extra_emis=None,
                                   process_meta_map=universe.process_meta)
    price_df = load_prices(paths.prices_csv, universe) if os.path.exists(paths.prices_csv) else None
    market_sum = summarize_market(model if solve else None, var, universe, price_df=price_df)

    # 10) Write outputs
    outdir = os.path.join(save_root, scenario_id)
    os.makedirs(outdir, exist_ok=True)
    for k, df in emis_sum.items():
        df.to_csv(os.path.join(outdir, f"emis_{k}.csv"), index=False, encoding='utf-8-sig')
    if market_sum is not None and len(market_sum):
        market_sum.to_csv(os.path.join(outdir, "market_summary.csv"), index=False, encoding='utf-8-sig')

    return outdir

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
                             save_root=get_results_base(), future_last_only=CFG['future_last_only'])
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

        # Demand from FBS
        fbs_comp = build_demand_components_from_fbs(paths.fbs_csv, universe)
        apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)

        # Nutrition RHS & land limits
        from S2_0_load_data import (
            load_population_wpp,
            load_intake_targets,
            build_nutrition_rhs_for_future,
            load_income_multipliers_from_sspdb,
        )
        pop_map = load_population_wpp(paths.population_wpp_csv)
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
                         save_root=get_results_base(), future_last_only=CFG['future_last_only'])

if __name__ == '__main__':
    main()

