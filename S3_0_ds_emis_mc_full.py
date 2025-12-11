# -*- coding: utf-8 -*-
"""
Full supply-demand + emissions + MACC cost model (aligned to S1_0_schema.Node).

Features
- Log-log supply and demand with cross-price and population elasticities.
- Global market clearing per (commodity, year).
- Emissions as e0_by_proc * Qs with abatement decisions per process driven by MACC.
- Land carbon price objective term for LULUCF processes.
- Optional nutrition and land constraints as in simplified model.

Inputs expected on Node
- Q0, D0, P0, tax_unit, eps_supply, eps_demand
- optional: eps_pop_demand (per country), epsD_row (dict commodity->epsilon)
- e0_by_proc: baseline process intensities

This file intentionally does not depend on nzf_schema; it works with S1_0_schema.Node.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
import logging
import math
import pandas as pd
import gurobipy as gp
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
def _log_pwl(model: gp.Model, x: gp.Var, y: gp.Var, xmin: float, xmax: float, npts: int, name: str):
    """
    Build a log PWL with log-spaced breakpoints to retain resolution at realistic volumes.
    Evenly spaced (linear) breakpoints between 1e-12 and 1e12 create huge gaps; here we
    space points in log-domain so Qs in tens/ hundreds map to sensible ln values.
    """
    lo = max(1e-12, xmin)
    hi = max(lo * 1.01, xmax)
    min_step_rel = 1e-9
    # 放宽绝对间距，避免靠近 0 的断点过于密集触发 Gurobi 容差
    min_step_abs = 1e-5
    # If range is extremely narrow, force just two well-separated points
    if hi / lo < 1.05:
        xs_unique = [lo, max(hi, lo * 1.05, lo + min_step_abs)]
    else:
        steps = max(2, min(100, npts))
        ratio = (hi / lo) ** (1.0 / (steps - 1))
        xs = [lo * (ratio ** k) for k in range(steps)]
        xs_unique = []
        for v in xs:
            # drop near-duplicates using both relative and absolute gaps
            if not xs_unique:
                xs_unique.append(v)
                continue
            prev = xs_unique[-1]
            sep_tol = max(min_step_abs, min_step_rel * max(1.0, abs(prev)))
            if abs(v - prev) > sep_tol:
                xs_unique.append(v)
        if len(xs_unique) < 2:
            bump = max(min_step_abs, 0.01 * max(1.0, abs(lo)))
            xs_unique = [lo, max(lo + bump, hi)]
        # 如果低端区域点过多，压缩为首/中/尾三个点，避免“靠近零的多点”报错
        small_thr = max(lo * 1e3, 1e-3)
        small_idx = [idx for idx, val in enumerate(xs_unique) if val <= small_thr]
        if len(small_idx) > 3:
            keep_idx = {small_idx[0], small_idx[len(small_idx) // 2], small_idx[-1]}
            xs_unique = [v for idx, v in enumerate(xs_unique) if idx not in small_idx or idx in keep_idx]
    ys = [math.log(v) for v in xs_unique]
    try:
        model.addGenConstrPWL(x, y, xs_unique, ys, name=name)
    except gp.GurobiError as err:
        try:
            lb = getattr(x, "LB", None)
            ub = getattr(x, "UB", None)
        except Exception:
            lb = ub = None
        msg = f"[PWL_ERROR] name={name}, x_bounds=({lb},{ub}), xmin={xmin}, xmax={xmax}, steps={len(xs_unique)}, err={err}"
        # 优先写到根 logger（由 S4_0_main 配置到 model.log），再回退到模块 logger 和 stderr
        try:
            root_logger = logging.getLogger()
            root_logger.error(msg)
        except Exception:
            pass
        try:
            logging.getLogger(__name__).error(msg)
        except Exception:
            pass
        print(msg, file=sys.stderr)
        raise


def _is_lulucf_process(name: str) -> bool:
    if not name:
        return False
    s = str(name).lower()
    keys = ['forest', 'net forest', 'afforest', 'deforest', 'savanna', 'drained organic', 'organic soil', 'peat', 'lulucf', 'land use']
    return any(k in s for k in keys)


def _read_macc(macc_path: Optional[str]) -> pd.DataFrame:
    if not macc_path:
        return pd.DataFrame()
    try:
        return pd.read_pickle(macc_path)
    except Exception:
        try:
            # sometimes saved via pickle.dump
            import pickle
            return pickle.load(open(macc_path, 'rb'))
        except Exception:
            return pd.DataFrame()


def build_model(
    data: Any,
    *,
    price_bounds=(1e-3, 1e6),
    qty_bounds=(1e-12, 1e12),
    # 减少默认断点数，配合更大绝对间距降低近零密度
    npts_log: int = 15,
    nutrition_rhs: Optional[Dict[Tuple[str,int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str,int], float]] = None,
    grass_area_by_region_year: Optional[Dict[Tuple[str,int], float]] = None,  # ✅ 草地面积约束
    forest_area_by_region_year: Optional[Dict[Tuple[str,int], float]] = None,  # ✅ 森林面积约束
    yield_t_per_ha_default: float = 3.0,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    macc_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str,int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str,int], float]] = None,
    max_growth_rate_per_period: Optional[float] = None,
    max_decline_rate_per_period: Optional[float] = None,
    lp_output_path: Optional[str] = None,
) -> Tuple[gp.Model, Dict[str, Dict[Tuple[str, str, int], gp.Var]]]:
    nodes_all = list(getattr(data, 'nodes', []) or [])
    univ = getattr(data, 'universe', None)
    cfg = getattr(data, 'config', None)
    hist_end = getattr(cfg, 'years_hist_end', 2020) if cfg else 2020
    nodes = [n for n in nodes_all if getattr(n, 'year', hist_end + 1) > hist_end]

    countries = sorted({n.country for n in nodes})
    commodities = sorted({n.commodity for n in nodes})
    years = sorted({n.year for n in nodes})

    Pmin, Pmax = price_bounds
    Qmin, Qmax = qty_bounds

    # Historical基准最大产量（用于未来变量上界，避免跨期约束失效时出现数量级跳变）
    hist_max_by_comm: Dict[Tuple[str, str], float] = {}
    hist_max_by_commodity: Dict[str, List[float]] = {}
    for n in data.nodes:
        if n.year <= 2020:
            key = (n.country, n.commodity)
            try:
                val = float(getattr(n, 'q0_with_ty', lambda: getattr(n, 'Q0', 0.0))())
            except Exception:
                val = float(getattr(n, 'Q0', 0.0) or 0.0)
            if val > 0:
                hist_max_by_comm[key] = max(hist_max_by_comm.get(key, 0.0), val)
                hist_max_by_commodity.setdefault(n.commodity, []).append(val)
    countries_all = list(univ.countries) if univ and getattr(univ, 'countries', None) else sorted({n.country for n in nodes_all})
    missing_roundwood = [c for c in countries_all if (c, 'Roundwood') not in hist_max_by_comm]
    try:
        logging.getLogger(__name__).info(
            "[NZF] Roundwood 基准产量缺失国家: %d / %d, 示例: %s",
            len(missing_roundwood), len(countries_all), missing_roundwood[:5])
    except Exception:
        pass

    m = gp.Model("nzf_full")
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = 3

    idx = {(n.country, n.commodity, n.year): n for n in nodes}

    # Global prices by (commodity, year)
    Pc: Dict[Tuple[str,int], gp.Var] = {}
    lnPc: Dict[Tuple[str,int], gp.Var] = {}
    stock: Dict[Tuple[str,int], gp.Var] = {}
    for j in commodities:
        for t in years:
            Pc[j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pc[{j},{t}]")
            lnPc[j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPc[{j},{t}]")
            _log_pwl(m, Pc[j, t], lnPc[j, t], Pmin, Pmax, npts_log, f"logPc[{j},{t}]")
            # 库存吸收当期全球净过剩，容量约束在市场平衡处设置
            stock[j, t] = m.addVar(lb=0.0, ub=Qmax, name=f"stock[{j},{t}]")

    # Node variables by (i,j,t)
    Pnet: Dict[Tuple[str,str,int], gp.Var] = {}
    lnPnet: Dict[Tuple[str,str,int], gp.Var] = {}
    Qs: Dict[Tuple[str,str,int], gp.Var] = {}
    Qd: Dict[Tuple[str,str,int], gp.Var] = {}
    lnQs: Dict[Tuple[str,str,int], gp.Var] = {}
    lnQd: Dict[Tuple[str,str,int], gp.Var] = {}
    Wunit: Dict[Tuple[str,str,int], gp.Var] = {}
    Taux: Dict[Tuple[str,str,int], gp.Var] = {}

    # Emissions and cost per node
    Eij: Dict[Tuple[str,str,int], gp.Var] = {}
    Cij: Dict[Tuple[str,str,int], gp.Var] = {}

    total_C = gp.LinExpr(0.0)
    total_E_land = gp.LinExpr(0.0)
    total_E_other = gp.LinExpr(0.0)

    # Preload MACC
    macc_df = _read_macc(macc_path)

    # Caches to support in-place updates
    constr_supply: Dict[Tuple[str,str,int], gp.Constr] = {}
    constr_demand: Dict[Tuple[str,str,int], gp.Constr] = {}
    constr_Edef: Dict[Tuple[str,str,int], gp.Constr] = {}
    proc_cap_constr: Dict[Tuple[str,str,int,str,int], gp.Constr] = {}
    proc_cap_basecoeff: Dict[Tuple[str,str,int,str,int], float] = {}
    e0_by_node: Dict[Tuple[str,str,int], Dict[str, float]] = {}
    nutri_constr: Dict[Tuple[str,int], gp.Constr] = {}
    landU_constr: Dict[Tuple[str,int], gp.Constr] = {}
    
    # Slack variables for supply/demand soft constraints (prevent infeasibility from rigid log-linear equations)
    supply_slack_pos: Dict[Tuple[str,str,int], gp.Var] = {}  # Positive deviation: actual supply > predicted
    supply_slack_neg: Dict[Tuple[str,str,int], gp.Var] = {}  # Negative deviation: actual supply < predicted
    demand_slack_pos: Dict[Tuple[str,str,int], gp.Var] = {}  # Positive deviation: actual demand > predicted
    demand_slack_neg: Dict[Tuple[str,str,int], gp.Var] = {}  # Negative deviation: actual demand < predicted

    for (i, j, t), n in idx.items():
        # Vars
        Pnet[i, j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pnet[{i},{j},{t}]")
        lnPnet[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPnet[{i},{j},{t}]")
        _log_pwl(m, Pnet[i, j, t], lnPnet[i, j, t], Pmin, Pmax, npts_log, f"logPnet[{i},{j},{t}]")

        # Upper bound: prefer own historical max, else use commodity average of nonzero hist max
        hist_max = hist_max_by_comm.get((i, j), None)
        fallback_avg = None
        vals = hist_max_by_commodity.get(j)
        if vals:
            try:
                fallback_avg = float(sum(vals) / len(vals))
            except Exception:
                fallback_avg = None
        base_ub = None
        if hist_max is not None and hist_max > 0:
            base_ub = hist_max
        elif fallback_avg is not None and fallback_avg > 0:
            base_ub = fallback_avg
        else:
            base_ub = Qmax

        q_ub = base_ub
        if t > 2020:
            growth_rate = max_growth_rate_per_period if max_growth_rate_per_period is not None else 0.0
            year_diff = t - 2020
            q_ub = min(Qmax, base_ub * (1.0 + growth_rate) ** year_diff)
        else:
            q_ub = min(Qmax, base_ub)
        
        # ADAPTIVE LOWER BOUND: Use Q0/D0 to set variable bounds, allowing small baseline values
        # This preserves calibration integrity while ensuring PWL feasibility
        Q0_raw = float(getattr(n, 'q0_with_ty', lambda: getattr(n, 'Q0', 0.0))())
        D0_raw = float(getattr(n, 'D0', 0.0) or 0.0)
        # CRITICAL FIX: Use very small lower bound (1e-12) for Qs/Qd to allow slack-driven deviations
        # The slack in supply/demand equations can change lnQs/lnQd significantly, so Qs = exp(lnQs)
        # must be able to reach very small values without violating variable bounds
        qs_lb = 1e-12  # Very small to allow any lnQs value from slack
        qd_lb = 1e-12  # Very small to allow any lnQd value from slack
        
        # DEBUG: Log small baseline nodes
        if i == 'Myanmar' and j == 'Oats':
            logging.getLogger(__name__).info(
                "[NZF][DEBUG] Myanmar Oats %d: Q0_raw=%.2e, D0_raw=%.2e, qs_lb=%.2e, qd_lb=%.2e, Qmin=%.2e",
                t, Q0_raw, D0_raw, qs_lb, qd_lb, Qmin)
        
        Qs[i, j, t] = m.addVar(lb=qs_lb, ub=q_ub, name=f"Qs[{i},{j},{t}]")
        Qd[i, j, t] = m.addVar(lb=qd_lb, ub=Qmax, name=f"Qd[{i},{j},{t}]")
        lnQs[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQs[{i},{j},{t}]")
        lnQd[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQd[{i},{j},{t}]")
        _log_pwl(m, Qs[i, j, t], lnQs[i, j, t], qs_lb, Qmax, npts_log, f"logQs[{i},{j},{t}]")
        _log_pwl(m, Qd[i, j, t], lnQd[i, j, t], qd_lb, Qmax, npts_log, f"logQd[{i},{j},{t}]")
        
        # Create slack variables for supply/demand soft constraints
        supply_slack_pos[i, j, t] = m.addVar(lb=0.0, name=f"s_supply_pos[{i},{j},{t}]")
        supply_slack_neg[i, j, t] = m.addVar(lb=0.0, name=f"s_supply_neg[{i},{j},{t}]")
        demand_slack_pos[i, j, t] = m.addVar(lb=0.0, name=f"s_demand_pos[{i},{j},{t}]")
        demand_slack_neg[i, j, t] = m.addVar(lb=0.0, name=f"s_demand_neg[{i},{j},{t}]")

        # Price linkage (basic tau only) - NOTE: pnet2 constraint below adds MACC unit-adder w
        tau = float(getattr(n, 'tax_unit', 0.0) or 0.0)
        # REMOVED: m.addConstr(Pnet[i, j, t] == Pc[j, t] - tau, name=f"pnet[{i},{j},{t}]")
        # The pnet2 constraint (line ~390) properly handles Pnet = Pc - tau - w
        # Having both pnet and pnet2 causes infeasibility when w != 0

        # Calibrate alpha_s and alpha_d using baseline Q0/D0 and P0
        # Supply with yield, temperature, and cross-price elasticities
        # NOTE: Q0_raw and D0_raw already defined above for adaptive variable bounds
        # Use original values for calibration (no clamping) - PWL range now extends to match
        Q0_cal = max(1e-12, Q0_raw)  # Only prevent log(0), preserve small positive values
        P0 = float(getattr(n, 'P0', 1.0) or 1.0)
        Pnet0 = max(1e-12, P0 - tau)
        eps_s = float(getattr(n, 'eps_supply', 0.0) or 0.0)
        eta_y = float(getattr(n, 'eps_supply_yield', 0.0) or 0.0)
        eta_temp = float(getattr(n, 'eps_supply_temp', 0.0) or 0.0)
        Ymult = float(getattr(n, 'Ymult', 1.0) or 1.0)
        Tmult = float(getattr(n, 'Tmult', 1.0) or 1.0)
        row_s = getattr(n, 'epsS_row', {}) or {}
        # Baseline calibration with cross-price elasticities
        Pnet0_by_comm = {c: max(1e-12, float(getattr(idx.get((i, c, t)), 'P0', 1.0) or 1.0) - tau) for c in commodities}
        ln_base_s = eps_s * math.log(Pnet0) + sum(float(row_s.get(mm, 0.0)) * math.log(Pnet0_by_comm.get(mm, 1.0)) for mm in commodities)
        alpha_s = math.log(Q0_cal) - ln_base_s - eta_y * math.log(max(1e-12, Ymult)) - eta_temp * math.log(max(1e-12, Tmult))
        # Only include cross-price terms for commodities that exist for this country-year (after name mapping)
        # 使用已创建的 lnPnet 变量；若对应节点尚未创建，跳过该交叉项
        cross_price_terms = gp.quicksum(float(row_s.get(mm, 0.0)) * lnPnet[(i, mm, t)] for mm in row_s.keys() if (i, mm, t) in lnPnet)
        # ADD SLACK: lnQs + slack_neg - slack_pos = RHS (allows deviation from predicted supply)
        cs = m.addConstr(lnQs[i, j, t] + supply_slack_neg[i, j, t] - supply_slack_pos[i, j, t] == alpha_s + eps_s * lnPnet[i, j, t] + cross_price_terms + eta_y * math.log(max(1e-12, Ymult)) + eta_temp * math.log(max(1e-12, Tmult)),
                         name=f"supply[{i},{j},{t}]")
        constr_supply[(i, j, t)] = cs

        # Demand with cross-price and population (population/income as relative changes vs base year)
        # NOTE: D0_raw already defined above for adaptive variable bounds
        eps_d = float(getattr(n, 'eps_demand', 0.0) or 0.0)
        eps_pop = float(getattr(n, 'eps_pop_demand', 0.0) or 0.0)
        eps_inc = float(getattr(n, 'eps_income_demand', 0.0) or 0.0)
        row = getattr(n, 'epsD_row', {}) or {}
        Pc0_by_comm = {c: float(getattr(n, 'P0', 1.0) or 1.0) for c in commodities}
        ln_base = eps_d * math.log(Pc0_by_comm[j]) + sum(float(row.get(mm, 0.0)) * math.log(Pc0_by_comm[mm]) for mm in commodities)
        # base population/income at hist_end (e.g., 2020); fallback to 1 to avoid log(0)
        pop_base = float(population_by_country_year.get((i, hist_end), 1.0) if population_by_country_year else 1.0)
        inc_base = float(income_mult_by_country_year.get((i, hist_end), 1.0) if income_mult_by_country_year else 1.0)
        # NOTE: D0_raw already defined above for adaptive variable bounds
        # Use original value for calibration (only prevent log(0)) - PWL range now extends to match
        D0_cal = max(1e-12, D0_raw)  # Only prevent log(0), preserve small positive values
        # FIXED: alpha_d should NOT subtract eps_pop*log(pop_base) or eps_inc*log(inc_base)
        # because the constraint uses RELATIVE changes: lnPop = log(pop_t/pop_base), lnInc = log(inc_t/inc_base)
        # At base year: lnPop = 0, lnInc = 0, so calibration is simply: alpha_d = log(D0) - ln_base
        alpha_d = math.log(D0_cal) - ln_base
        pop_t = float(population_by_country_year.get((i, t), pop_base) if population_by_country_year else pop_base)
        inc_t = float(income_mult_by_country_year.get((i, t), inc_base) if income_mult_by_country_year else inc_base)
        lnPop = math.log(max(1e-12, pop_t / max(1e-12, pop_base)))
        lnInc = math.log(max(1e-12, inc_t / max(1e-12, inc_base)))
        # ADD SLACK: lnQd + slack_neg - slack_pos = RHS (allows deviation from predicted demand)
        cd = m.addConstr(lnQd[i, j, t] + demand_slack_neg[i, j, t] - demand_slack_pos[i, j, t] == alpha_d + eps_d * lnPc[j, t] + gp.quicksum(float(row.get(mm, 0.0)) * lnPc[mm, t] for mm in commodities) + eps_pop * lnPop + eps_inc * lnInc,
                         name=f"demand[{i},{j},{t}]")
        if i == 'Myanmar' and j == 'Oats' and t == 2080:
            try:
                rhs_const = alpha_d + eps_pop * lnPop + eps_inc * lnInc
                logging.getLogger(__name__).info(
                    "[NZF][DEBUG] Myanmar Oats 2080 DEMAND: D0_raw=%.4e, D0_cal=%.4e, eps_d=%.4f, eps_pop=%.4f, eps_inc=%.4f, "
                    "pop_base=%.2e, pop_t=%.2e, inc_base=%.4f, inc_t=%.4f, lnPop=%.4f, lnInc=%.4f, ln_base=%.4f, "
                    "alpha_d=%.4f, RHS_const=%.4f",
                    D0_raw, D0_cal, eps_d, eps_pop, eps_inc, pop_base, pop_t, inc_base, inc_t, lnPop, lnInc, ln_base, alpha_d, rhs_const)
            except Exception:
                pass
        if i == 'Brazil' and j == 'Roundwood' and t == 2080:
            try:
                logging.getLogger(__name__).info(
                    "[NZF][DEBUG] BR Roundwood 2080 demand: D0=%.3f eps_d=%.4f eps_pop=%.4f eps_inc=%.4f pop_base=%.3f inc_base=%.3f lnPop=%.3f lnInc=%.3f ln_base=%.3f alpha_d=%.3f RHS=%.3f",
                    D0_raw, eps_d, eps_pop, eps_inc, pop_base, inc_base, lnPop, lnInc, ln_base, alpha_d,
                    alpha_d + eps_d * math.log(Pc0_by_comm[j]) + eps_pop * lnPop + eps_inc * lnInc)
            except Exception:
                pass
        constr_demand[(i, j, t)] = cd

        # Emissions and MACC abatement cost
        Eij[i, j, t] = m.addVar(lb=0.0, name=f"E[{i},{j},{t}]")
        Cij[i, j, t] = m.addVar(lb=0.0, name=f"C[{i},{j},{t}]")
        # NOTE: w is unit abatement cost adder (USD/tonne product), cap at 1e+04 for numerical stability
        # This avoids McCormick Big-M values of 1e+14 when combined with large Qs bounds
        w_upper = 1e+04  # 10,000 USD/tonne is reasonable upper bound for marginal abatement cost
        w = m.addVar(lb=0.0, ub=w_upper, name=f"w[{i},{j},{t}]")  # unit cost adder
        t_aux = m.addVar(lb=0.0, name=f"t[{i},{j},{t}]")      # auxiliary t = w * Qs
        Wunit[i, j, t] = w
        Taux[i, j, t] = t_aux

        e0_map = getattr(n, 'e0_by_proc', {}) or {}
        e0_by_node[(i, j, t)] = dict(e0_map)
        sum_e0 = sum(float(v) for v in e0_map.values())

        # Build abatement variables per process from MACC (if available)
        a_by_proc: Dict[str, List[gp.Var]] = {}
        a_vars: List[gp.Var] = []
        cost_terms = []
        if not macc_df.empty and sum_e0 > 0:
            for proc, e0p in e0_map.items():
                e0p = float(e0p)
                if e0p <= 0:
                    continue
                dfp = macc_df[(macc_df['Country'] == i) & (macc_df['Process'] == proc)] if 'Country' in macc_df.columns else pd.DataFrame()
                if dfp.empty:
                    continue
                if 'cumulative_fraction_of_process' in dfp.columns and 'marginal_cost_$per_tco2e' in dfp.columns:
                    dfp = dfp[['cumulative_fraction_of_process','marginal_cost_$per_tco2e']].dropna().sort_values('cumulative_fraction_of_process')
                    prev = 0.0
                    for _, r in dfp.iterrows():
                        frac = float(r['cumulative_fraction_of_process'])
                        mu = float(r['marginal_cost_$per_tco2e'])
                        if frac <= prev:
                            continue
                        # Create linear constraint: a - coeff*Qs <= 0, where coeff = (螖frac * e0p)
                        coeff = (frac - prev) * e0p
                        a = m.addVar(lb=0.0, name=f"a[{i},{j},{proc},{_}]")
                        ccap = m.addConstr(a - coeff * Qs[i, j, t] <= 0.0, name=f"cap[{i},{j},{proc},{_}]")
                        proc_cap_constr[(i, j, t, proc, _)] = ccap
                        proc_cap_basecoeff[(i, j, t, proc, _)] = coeff
                        a_vars.append(a)
                        a_by_proc.setdefault(proc, []).append(a)
                        cost_terms.append(mu * a)
                        prev = frac
                else:
                    # Fallback: single linear segment up to full e0p*Qs with median cost 0
                    a = m.addVar(lb=0.0, name=f"a[{i},{j},{proc},0]")
                    ccap = m.addConstr(a - (1.0 * e0p) * Qs[i, j, t] <= 0.0, name=f"cap[{i},{j},{proc},0]")
                    proc_cap_constr[(i, j, t, proc, 0)] = ccap
                    proc_cap_basecoeff[(i, j, t, proc, 0)] = (1.0 * e0p)
                    a_vars.append(a)
                    a_by_proc.setdefault(proc, []).append(a)
                    cost_terms.append(0.0 * a)

        # Emission definition
        emis_expr = sum_e0 * Qs[i, j, t]
        if a_vars:
            emis_expr -= gp.quicksum(a_vars)
        ce = m.addConstr(Eij[i, j, t] == emis_expr,
                         name=f"E_def[{i},{j},{t}]")
        constr_Edef[(i, j, t)] = ce
        # Cost definition
        if cost_terms:
            m.addConstr(Cij[i, j, t] == gp.quicksum(cost_terms), name=f"C_def[{i},{j},{t}]")
        else:
            m.addConstr(Cij[i, j, t] == 0.0, name=f"C_def[{i},{j},{t}]")

        # Accumulate totals: abatement cost only
        total_C += Cij[i, j, t]
        # LULUCF split for land carbon price term
        e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
        # Sum of abatement on LULUCF processes
        abat_land = gp.quicksum(a for p, lst in a_by_proc.items() if _is_lulucf_process(p) for a in lst) if a_by_proc else 0.0
        total_E_land += e_land * Qs[i, j, t] - abat_land
        total_E_other += (sum_e0 - e_land) * Qs[i, j, t]

        # McCormick envelope tying unit adder to cost: t_aux = w * Qs; enforce t_aux == Cij to get w = C/Q
        # ONLY add McCormick constraints when there are cost_terms (MACC data exists)
        # Otherwise, w=0 and no bilinear relaxation needed
        if cost_terms:
            # Use w_upper (not Pmax) to keep Big-M values reasonable for numerical stability
            # NUMERICAL STABILITY FIX: Cap qU for McCormick to avoid wU*qU > 1e+10
            # This keeps mc2/mc4 RHS values below 1e+10 instead of 1e+14
            # The actual Qs variable still has full q_ub bound; this is just for Big-M relaxation
            q_ub_mc = min(q_ub, 1e+06)  # Cap McCormick qU at 1e+06 to keep wU*qU = 1e+04*1e+06 = 1e+10
            wL, wU = 0.0, w_upper
            qL, qU = Qmin, q_ub_mc
            m.addConstr(t_aux >= wL * Qs[i, j, t] + qL * w - wL * qL, name=f"mc1[{i},{j},{t}]")
            m.addConstr(t_aux >= wU * Qs[i, j, t] + qU * w - wU * qU, name=f"mc2[{i},{j},{t}]")
            m.addConstr(t_aux <= wU * Qs[i, j, t] + qL * w - wU * qL, name=f"mc3[{i},{j},{t}]")
            m.addConstr(t_aux <= wL * Qs[i, j, t] + qU * w - wL * qU, name=f"mc4[{i},{j},{t}]")
            m.addConstr(t_aux == Cij[i, j, t], name=f"t_eq_C[{i},{j},{t}]")
            # Price linkage includes unit adder
            m.addConstr(Pnet[i, j, t] == Pc[j, t] - (tau + w), name=f"pnet2[{i},{j},{t}]")
        else:
            # No MACC data: fix w=0, t_aux=0, simplify price linkage
            m.addConstr(w == 0.0, name=f"w_zero[{i},{j},{t}]")
            m.addConstr(t_aux == 0.0, name=f"t_zero[{i},{j},{t}]")
            m.addConstr(Pnet[i, j, t] == Pc[j, t] - tau, name=f"pnet2[{i},{j},{t}]")

    # Market clearing per (j,t)
    # Add slack variables to handle potential supply-demand imbalances
    # excess: supply exceeds demand (stock accumulation)
    # shortage: demand exceeds supply (unmet demand)
    excess = {}
    shortage = {}
    for j in commodities:
        for t in years:
            supply_sum = gp.quicksum(Qs[i, j, t] for i in countries if (i,j,t) in Qs)
            demand_sum = gp.quicksum(Qd[i, j, t] for i in countries if (i,j,t) in Qd)
            # Create slack variables for supply-demand balance
            excess[j, t] = m.addVar(lb=0.0, name=f"excess[{j},{t}]")
            shortage[j, t] = m.addVar(lb=0.0, name=f"shortage[{j},{t}]")
            # Market clearing with slack: supply + shortage = demand + excess + stock
            m.addConstr(supply_sum + shortage[j, t] == demand_sum + excess[j, t] + stock[j, t], name=f"clear[{j},{t}]")
            # Stock is bounded by excess (can't store what we don't produce)
            m.addConstr(stock[j, t] <= excess[j, t], name=f"stock_cap[{j},{t}]")

    # Nutrition constraint (future only): sum_j nutrient_per_unit[j] * Qd[i,j,t] >= RHS[i,t]
    if nutrition_rhs and nutrient_per_unit_by_comm:
        for i in countries:
            for t in years:
                if t <= 2020:  # historical years unconstrained
                    continue
                rhs = nutrition_rhs.get((i, t))
                if rhs is None:
                    continue
                expr = gp.LinExpr(0.0)
                for j in commodities:
                    if (i, j, t) in Qd:
                        v = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                        if v > 0:
                            expr += v * Qd[i, j, t]
                if expr.size() > 0:
                    cn = m.addConstr(expr >= float(rhs), name=f"nutri[{i},{t}]")
                    nutri_constr[(i, t)] = cn

    # Land area upper bound per (i,t)
    # 约束含义：耕地面积 + 草地面积 + 森林面积 ≤ 总土地上限
    # 未来年份统一使用2020年的土地上限数据
    # ✅ 森林面积作为基期固定值参与约束
    if land_area_limits:
        BASE_YEAR_FOR_LAND = 2020
        for i in countries:
            # ✅ 统一使用2020年的土地上限数据（单位: 1000 ha）
            limit = land_area_limits.get((i, BASE_YEAR_FOR_LAND))
            if limit is None:
                continue
            # ✅ 跳过 NaN 值
            try:
                limit_float = float(limit) * 1000.0  # 转换为 ha
                if np.isnan(limit_float) or np.isinf(limit_float):
                    continue
            except (ValueError, TypeError):
                continue
            
            # ✅ 获取基期森林面积（2020年）
            forest_area_ha = 0.0
            if forest_area_by_country_year:
                forest_area_ha = forest_area_by_country_year.get((i, BASE_YEAR_FOR_LAND), 0.0)
            
            for t in years:
                # ✅ 关键修复：土地约束 = 耕地面积 + 草地面积 + 森林面积 <= 总土地上限
                # 耕地面积 = Σ(Qs / yield0)
                expr = gp.LinExpr(0.0)
                for j in commodities:
                    if (i,j,t) not in Qs:
                        continue
                    y0 = None
                    n = idx.get((i, j, t))
                    if n is not None:
                        y0 = float(getattr(n, 'meta', {}).get('yield0', 0.0))
                    coef = 1.0 / max(1e-12, y0 if y0 and y0>0 else yield_t_per_ha_default)
                    expr += coef * Qs[i, j, t]
                
                # ✅ 加入草地面积（从外部数据）
                grass_area = 0.0
                if grass_area_by_region_year:
                    grass_area = grass_area_by_region_year.get((i, t), 0.0)
                    if grass_area > 0:
                        expr.addConstant(grass_area)  # 草地面积加到约束左侧
                
                # ✅ 加入森林面积（基期固定值）
                if forest_area_ha > 0:
                    expr.addConstant(forest_area_ha)  # 森林面积加到约束左侧
                
                if expr.size() > 0:
                    cl = m.addConstr(expr <= limit_float, name=f"landU[{i},{t}]")
                    landU_constr[(i, t)] = cl

    # Growth rate constraints: limit change between consecutive years
    growth_constr_upper: Dict[Tuple[str,str,int], gp.Constr] = {}
    growth_constr_lower: Dict[Tuple[str,str,int], gp.Constr] = {}
    if max_growth_rate_per_period is not None or max_decline_rate_per_period is not None:
        growth_rate = max_growth_rate_per_period if max_growth_rate_per_period is not None else 0.5
        decline_rate = max_decline_rate_per_period if max_decline_rate_per_period is not None else 0.5
        # 历史最大产量，用于当历史变量缺失时提供锚点
        hist_max: Dict[Tuple[str, str], float] = {}
        for i in countries:
            for j in commodities:
                vals = []
                for t in years:
                    if t <= 2020:
                        n_hist = idx.get((i, j, t))
                        if n_hist:
                            try:
                                vals.append(float(n_hist.q0_with_ty()))
                            except Exception:
                                vals.append(float(getattr(n_hist, 'Q0', 0.0)))
                if vals:
                    hist_max[(i, j)] = max(v for v in vals if v is not None)
        
        for i in countries:
            for j in commodities:
                years_sorted = sorted([t for t in years if (i,j,t) in idx])  # use all years with nodes (even if Qs not a var)
                for idx_t in range(1, len(years_sorted)):
                    t_prev = years_sorted[idx_t - 1]
                    t_curr = years_sorted[idx_t]
                    year_diff = t_curr - t_prev
                    
                    n_prev = idx.get((i, j, t_prev))
                    Q0_prev = float(getattr(n_prev, 'q0_with_ty', lambda: getattr(n_prev, 'Q0', 0.0))()) if n_prev else 0.0
                    # 若无历史变量则使用历史最大值
                    hist_anchor = hist_max.get((i, j), 0.0)
                    baseline_val = Q0_prev if Q0_prev > 0 else hist_anchor
                    
                    # Only apply constraint if baseline production is positive
                    if baseline_val <= 1e-3 or (i, j, t_curr) not in Qs:
                        continue

                    # Compound growth rate for multi-year periods
                    max_mult_upper = (1.0 + growth_rate) ** year_diff
                    max_mult_lower = (1.0 - decline_rate) ** year_diff

                    if (i, j, t_prev) in Qs:
                        # Upper bound: Qs[t] <= Qs[t-1] * (1 + growth_rate)^year_diff
                        cg_upper = m.addConstr(
                            Qs[i, j, t_curr] <= Qs[i, j, t_prev] * max_mult_upper,
                            name=f"growth_upper[{i},{j},{t_curr}]"
                        )
                        growth_constr_upper[(i, j, t_curr)] = cg_upper
                        
                        # Lower bound: Qs[t] >= Qs[t-1] * (1 - growth_rate)^year_diff
                        cg_lower = m.addConstr(
                            Qs[i, j, t_curr] >= Qs[i, j, t_prev] * max_mult_lower,
                            name=f"growth_lower[{i},{j},{t_curr}]"
                        )
                        growth_constr_lower[(i, j, t_curr)] = cg_lower
                    else:
                        # 历史期没有优化变量：使用历史基准锚定未来
                        cg_upper = m.addConstr(
                            Qs[i, j, t_curr] <= baseline_val * max_mult_upper,
                            name=f"growth_upper_base[{i},{j},{t_curr}]"
                        )
                        growth_constr_upper[(i, j, t_curr)] = cg_upper
                        cg_lower = m.addConstr(
                            Qs[i, j, t_curr] >= baseline_val * max_mult_lower,
                            name=f"growth_lower_base[{i},{j},{t_curr}]"
                        )
                        growth_constr_lower[(i, j, t_curr)] = cg_lower
        # 对所有未来年份再叠加一次绝对上界：Qs[t] <= hist_max*(1+g)^(t-2020)
        for (i, j), base_val in hist_max.items():
            if base_val is None or base_val <= 1e-3:
                continue
            for t in years:
                if t <= 2020 or (i, j, t) not in Qs:
                    continue
                year_diff = t - 2020
                max_mult_upper = (1.0 + growth_rate) ** year_diff
                cg_upper_abs = m.addConstr(
                    Qs[i, j, t] <= base_val * max_mult_upper,
                    name=f"growth_upper_abs[{i},{j},{t}]"
                )
                growth_constr_upper[(i, j, t, 'abs')] = cg_upper_abs

    # Objective: total abatement cost + land carbon price term
    obj = total_C
    if land_carbon_price_by_year:
        for (i, j, t), n in idx.items():
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp > 0:
                e0_map = getattr(n, 'e0_by_proc', {}) or {}
                e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
                # Rebuild abat_land expression analogously
                # Note: without holding references to a_by_proc outside loop, approximate benefit by cp * total_E_land already accumulated
                obj += cp * (e_land * Qs[i, j, t])  # benefit for BAU emissions
        # Since total_E_land already subtracts abatements, subtract cp * Σ a(LULUCF)
        # This is approximated within per-node accumulation above.

    # Add penalty for supply-demand imbalances (excess/shortage) AND supply/demand equation slack
    # Use a high penalty to minimize slack usage while keeping model feasible
    SLACK_PENALTY = 1e6  # High penalty to minimize imbalances
    for j in commodities:
        for t in years:
            obj += SLACK_PENALTY * (excess[j, t] + shortage[j, t])
    
    # Add penalty for supply/demand soft constraint violations (higher penalty = stricter adherence to log-linear equations)
    EQUATION_SLACK_PENALTY = 1e5  # Slightly lower than market clearing slack to prioritize market equilibrium
    for i, j, t in supply_slack_pos.keys():
        obj += EQUATION_SLACK_PENALTY * (supply_slack_pos[i, j, t] + supply_slack_neg[i, j, t])
        obj += EQUATION_SLACK_PENALTY * (demand_slack_pos[i, j, t] + demand_slack_neg[i, j, t])

    # Set objective with slack penalties to minimize imbalances
    m.setObjective(obj, gp.GRB.MINIMIZE)
    
    # expose cache for in-place MC updates
    m._nzf_cache = dict(
        Pc=Pc, lnPc=lnPc, Pnet=Pnet, lnPnet=lnPnet, Qs=Qs, Qd=Qd, lnQs=lnQs, lnQd=lnQd,
        W=Wunit, Cij=Cij, Eij=Eij, t_aux=Taux,
        constr_supply=constr_supply, constr_demand=constr_demand,
        constr_Edef=constr_Edef, proc_cap_constr=proc_cap_constr, proc_cap_basecoeff=proc_cap_basecoeff,
        e0_by_node=e0_by_node, nutri_constr=nutri_constr, landU_constr=landU_constr,
        growth_constr_upper=growth_constr_upper, growth_constr_lower=growth_constr_lower,
        stock=stock, excess=excess, shortage=shortage,
        supply_slack_pos=supply_slack_pos, supply_slack_neg=supply_slack_neg,
        demand_slack_pos=demand_slack_pos, demand_slack_neg=demand_slack_neg,
        countries=countries, commodities=commodities, years=years,
    )
    if lp_output_path:
        try:
            m.write(lp_output_path)
        except Exception as err:
            logging.getLogger(__name__).warning(f"Failed to write LP to {lp_output_path}: {err}")
    m.update()

    var = {
        "Qs": Qs,
        "Qd": Qd,
        "Pc": Pc,
        "Pnet": Pnet,
        "W": Wunit,
        "C": Cij,
        "E": Eij,
        "Stock": stock,
    }
    return m, var


# ===================== MC cache (in-place updates scaffold) =====================
@dataclass
class SolveOpt:
    price_bounds: Tuple[float, float] = (1e-3, 1e6)
    qty_bounds: Tuple[float, float] = (1e-12, 1e12)
    npts_log: int = 25
    land_carbon_price_by_year: Optional[Dict[int, float]] = None
    grb_output: int = 0


@dataclass
class ModelCache:
    model: gp.Model
    Pc: Dict[Tuple[str,int], gp.Var]
    lnPc: Dict[Tuple[str,int], gp.Var]
    Pnet: Dict[Tuple[str,str,int], gp.Var]
    lnPnet: Dict[Tuple[str,str,int], gp.Var]
    Qs: Dict[Tuple[str,str,int], gp.Var]
    Qd: Dict[Tuple[str,str,int], gp.Var]
    lnQs: Dict[Tuple[str,str,int], gp.Var]
    lnQd: Dict[Tuple[str,str,int], gp.Var]
    W: Dict[Tuple[str,str,int], gp.Var]
    Cij: Dict[Tuple[str,str,int], gp.Var]
    Eij: Dict[Tuple[str,str,int], gp.Var]
    Stock: Dict[Tuple[str,int], gp.Var]
    constr_supply: Dict[Tuple[str,str,int], gp.Constr]
    constr_demand: Dict[Tuple[str,str,int], gp.Constr]
    alpha_s: Dict[Tuple[str,str,int], float]
    alpha_d: Dict[Tuple[str,str,int], float]
    eps_s: Dict[Tuple[str,str,int], float]
    eps_d: Dict[Tuple[str,str,int], float]
    eps_pop: Dict[Tuple[str,str,int], float]
    eta_y: Dict[Tuple[str,str,int], float]
    eta_temp: Dict[Tuple[str,str,int], float]
    ymult0: Dict[Tuple[str,str,int], float]
    tmult0: Dict[Tuple[str,str,int], float]
    # references to builder cache for advanced updates
    _builder: dict


def build_model_cache(data: Any, **kwargs) -> ModelCache:
    m, var = build_model(data, **kwargs)
    # Use builder-exposed cache to populate ModelCache quickly
    c = getattr(m, '_nzf_cache', {})
    Pc = c.get('Pc', {}); lnPc = c.get('lnPc', {})
    Pnet = c.get('Pnet', {}); lnPnet = c.get('lnPnet', {})
    Qs = c.get('Qs', {}); Qd = c.get('Qd', {}); lnQs = c.get('lnQs', {}); lnQd = c.get('lnQd', {})
    W = c.get('W', {}); Cij = c.get('Cij', {}); Eij = c.get('Eij', {})
    stock = c.get('stock', {})
    constr_supply = c.get('constr_supply', {}); constr_demand = c.get('constr_demand', {})
    constr_Edef = c.get('constr_Edef', {})

    alpha_s = {}; alpha_d = {}; eps_s = {}; eps_d = {}; eps_pop = {}
    eta_y = {}; eta_temp = {}; ymult0 = {}; tmult0 = {}
    idx = {(n.country, n.commodity, n.year): n for n in getattr(data, 'nodes', []) or []}
    for (i, j, t), n in idx.items():
        key = (i, j, t)
        eps_s[key] = float(getattr(n, 'eps_supply', 0.0) or 0.0)
        eps_d[key] = float(getattr(n, 'eps_demand', 0.0) or 0.0)
        eps_pop[key] = float(getattr(n, 'eps_pop_demand', 0.0) or 0.0)
        eta_y[key] = float(getattr(n, 'eps_supply_yield', 0.0) or 0.0)
        eta_temp[key] = float(getattr(n, 'eps_supply_temp', 0.0) or 0.0)
        ymult0[key] = float(getattr(n, 'Ymult', 1.0) or 1.0)
        tmult0[key] = float(getattr(n, 'Tmult', 1.0) or 1.0)

        # Estimate alpha_s/d used in build (approx)
        P0 = float(getattr(n, 'P0', 1.0) or 1.0)
        tau = float(getattr(n, 'tax_unit', 0.0) or 0.0)
        Q0_eff = float(getattr(n, 'q0_with_ty', lambda: getattr(n, 'Q0', 0.0))())
        Pnet0 = max(1e-12, P0 - tau)
        alpha_s[key] = math.log(max(1e-12, Q0_eff)) - eps_s[key] * math.log(Pnet0) - eta_y[key]*math.log(max(1e-12, ymult0[key])) - eta_temp[key]*math.log(max(1e-12, tmult0[key]))
        D0 = float(getattr(n, 'D0', 0.0) or 0.0)
        # Cross-price base omitted; absorb into alpha_d approx
        alpha_d[key] = math.log(max(1e-12, D0))

    return ModelCache(m, Pc, lnPc, Pnet, lnPnet, Qs, Qd, lnQs, lnQd,
                      W, Cij, Eij, stock,
                      constr_supply, constr_demand, alpha_s, alpha_d,
                      eps_s, eps_d, eps_pop, eta_y, eta_temp, ymult0, tmult0,
                      _builder=c)


def apply_sample_updates(cache: ModelCache,
                         *,
                         pop_mult_by_country: Optional[Dict[str, float]] = None,
                         income_mult_by_country: Optional[Dict[str, float]] = None,
                         yield_mult_by_node: Optional[Dict[Tuple[str,str], float]] = None,
                         temp_mult_by_node: Optional[Dict[Tuple[str,str], float]] = None,
                         e0_mult_by_node_proc: Optional[Dict[Tuple[str,str,int,str], float]] = None,
                         land_cp_by_year: Optional[Dict[int, float]] = None,
                         nutrition_rhs: Optional[Dict[Tuple[str,int], float]] = None,
                         land_limits: Optional[Dict[Tuple[str,int], float]] = None) -> None:
    """Update key constraints in-place for MC: population and yield scalers.
    - pop_mult_by_country: updates demand RHS via eps_pop * ln(pop_mult)
    - yield_mult_by_node: updates supply RHS via eta_y * ln(y_mult)
    """
    m = cache.model
    # Update demand RHS (population & income)
    if pop_mult_by_country:
        for (i, j, t), con in cache.constr_demand.items():
            em = cache.eps_pop.get((i, j, t), 0.0)
            if em == 0.0:
                continue
            mult = float(pop_mult_by_country.get(i, 1.0) or 1.0)
            if mult <= 0:
                continue
            # Adjust RHS by eps_pop * ln(mult)
            con.RHS = con.RHS + em * math.log(mult)
    if income_mult_by_country:
        for (i, j, t), con in cache.constr_demand.items():
            # reuse alpha_d approximation; simply shift RHS by 畏inc ln(mult)
            em = float(getattr(cache, 'eps_income', {}).get((i, j, t), 0.0)) if hasattr(cache, 'eps_income') else 0.0
            mult = float(income_mult_by_country.get(i, 1.0) or 1.0)
            if em and mult > 0:
                con.RHS = con.RHS + em * math.log(mult)
    # Update supply RHS (yield, temperature)
    if yield_mult_by_node:
        for (i, j, t), con in cache.constr_supply.items():
            ey = cache.eta_y.get((i, j, t), 0.0)
            if ey == 0.0:
                continue
            mult = float(yield_mult_by_node.get((i, j), 1.0) or 1.0)
            if mult <= 0:
                continue
            con.RHS = con.RHS + ey * math.log(mult)
    if temp_mult_by_node:
        for (i, j, t), con in cache.constr_supply.items():
            et = cache.eta_temp.get((i, j, t), 0.0)
            mult = float(temp_mult_by_node.get((i, j), 1.0) or 1.0)
            if et and mult > 0:
                con.RHS = con.RHS + et * math.log(mult)

    # Update e0 & MACC caps coefficients
    if e0_mult_by_node_proc:
        m = cache.model
        c = getattr(m, '_nzf_cache', {})
        Edef = c.get('constr_Edef', {})
        cap_constr = c.get('proc_cap_constr', {})
        base = c.get('proc_cap_basecoeff', {})
        e0_map = c.get('e0_by_node', {})
        # Update coefficients in E_def for Qs
        for (i, j, t), ce in Edef.items():
            e0s = e0_map.get((i, j, t), {})
            new_sum = 0.0
            for p, v in e0s.items():
                f = float(e0_mult_by_node_proc.get((i, j, t, p), 1.0) or 1.0)
                new_sum += float(v) * f
            # change coefficient on Qs in E_def: Eij == new_sum * Qs - 危 a
            qvar = c.get('Qs', {}).get((i, j, t))
            if qvar is not None:
                m.chgCoeff(ce, qvar, new_sum)
        # Update each MACC cap coefficient a - coeff*Qs <= 0
        for key, con in cap_constr.items():
            i, j, t, p, seg = key
            qvar = c.get('Qs', {}).get((i, j, t))
            if qvar is None:
                continue
            base_coeff = base.get(key, 0.0)  # base contains 螖frac * e0p
            f = float(e0_mult_by_node_proc.get((i, j, t, p), 1.0) or 1.0)
            m.chgCoeff(con, qvar, - base_coeff * f)  # note coefficient is negative on LHS (a - coeff*Qs)

    # Update nutrition & land RHS
    if nutrition_rhs:
        for (i, t), con in (getattr(cache.model, '_nzf_cache', {}).get('nutri_constr', {}) or {}).items():
            rhs = nutrition_rhs.get((i, t))
            if rhs is not None:
                con.RHS = float(rhs)
    if land_limits:
        for (i, t), con in (getattr(cache.model, '_nzf_cache', {}).get('landU_constr', {}) or {}).items():
            lim = land_limits.get((i, t))
            if lim is not None:
                con.RHS = float(lim)

    # Optionally update objective with new land carbon price time series
    if land_cp_by_year is not None:
        m = cache.model
        c = getattr(m, '_nzf_cache', {})
        Qs = c.get('Qs', {})
        e0_map = c.get('e0_by_node', {})
        obj = gp.LinExpr(0.0)
        # Keep existing cost terms Cij in objective
        for v in m.getVars():
            if v.VarName.startswith('C['):
                obj += v
        for (i, j, t), n in e0_map.items():
            cp = float(land_cp_by_year.get(t, 0.0) or 0.0)
            if cp <= 0:
                continue
            e_land = sum(float(v) for p, v in n.items() if _is_lulucf_process(p))
            qvar = Qs.get((i, j, t))
            if qvar is not None and e_land > 0:
                obj += cp * e_land * qvar
        m.setObjective(obj, gp.GRB.MINIMIZE)
    m.update()


# ===================== MC driver: from MC sheet specs =====================
def _select_processes(universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(getattr(universe, 'processes', []) or [])
    return [key] if key in (getattr(universe, 'processes', []) or []) else []

def _select_commodities(universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(getattr(universe, 'commodities', []) or [])
    k = str(key).lower()
    if k in ('crop','meat','dairy','other'):
        cat2 = getattr(universe, 'item_cat2_by_commodity', {}) or {}
        return [c for c in getattr(universe, 'commodities', []) if (cat2.get(c, '').lower()==k)]
    return [key] if key in (getattr(universe, 'commodities', []) or []) else []

def run_mc(data: Any, universe: Any, specs_df: pd.DataFrame, *, n_samples: int=100, seed: int=42,
           save_prefix: str = 'mc_full_results') -> pd.DataFrame:
    """Run MC sampling based on MC sheet specifications.
    Supports EF multipliers by (Process, Item, optional GHG) with [Min_bound, Max_bound].
    """
    cache = build_model_cache(data,
                              nutrition_rhs=None,
                              nutrient_per_unit_by_comm=None,
                              land_area_limits=None,
                              land_carbon_price_by_year=None,
                              population_by_country_year=None,
                              income_mult_by_country_year=None,
                              macc_path=None)
    m = cache.model

    # Prepare specs rows
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Required columns
    c_elem = 'Element'; c_proc = 'Process'; c_item = 'Item'; c_min='Min_bound'; c_max='Max_bound'
    if not all(c in df.columns for c in [c_elem, c_proc, c_item, c_min, c_max]):
        return pd.DataFrame()
    c_ghg  = 'GHG' if 'GHG' in df.columns else None

    rng = np.random.default_rng(seed)
    recs = []
    for s in range(1, n_samples+1):
        # Build multipliers map for this sample
        ef_mult: Dict[Tuple[str,str,int,str], float] = {}
        yld_mult: Dict[Tuple[str,str], float] = {}
        # process gas map
        gas_by_proc = {p: ((getattr(universe, 'process_meta', {}) or {}).get(p, {}).get('gas') or '').upper() for p in getattr(universe, 'processes', [])}
        for r in df.itertuples(index=False):
            elem = str(getattr(r, c_elem))
            proc = str(getattr(r, c_proc)) if c_proc in df.columns else 'All'
            item = str(getattr(r, c_item)) if c_item in df.columns else 'All'
            lo = float(getattr(r, c_min, 1.0)); hi = float(getattr(r, c_max, 1.0))
            if hi < lo:
                lo, hi = hi, lo
            draw = float(rng.uniform(lo, hi))
            # Only support EF/emission intensity elements
            if 'ef' in elem.lower() or 'emission' in elem.lower():
                procs = _select_processes(universe, proc)
                comms = _select_commodities(universe, item)
                # assign to all node-years (historical+future)
                for n in getattr(data, 'nodes', []) or []:
                    if n.commodity in comms and procs:
                        for p in procs:
                            ef_mult[(n.country, n.commodity, n.year, p)] = draw
            elif 'yield' in elem.lower():
                comms = _select_commodities(universe, item)
                for n in getattr(data, 'nodes', []) or []:
                    if n.commodity in comms:
                        yld_mult[(n.country, n.commodity)] = draw
        # Apply updates & solve
        apply_sample_updates(cache, e0_mult_by_node_proc=ef_mult, yield_mult_by_node=yld_mult if yld_mult else None)
        m.optimize()
        status = m.Status
        row = {'sample': s, 'status': status}
        if status == gp.GRB.OPTIMAL:
            # totals
            totE = 0.0; totC = 0.0
            for v in m.getVars():
                name = v.VarName
                if name.startswith('E['):
                    totE += v.X
                elif name.startswith('C['):
                    totC += v.X
            row['E_total'] = totE; row['C_total'] = totC
            # Prices
            for (j,t), v in cache.Pc.items():
                row[f'{j}__Pc_{t}'] = v.X
            # By node
            for (i,j,t), q in cache.Qs.items():
                row[f'{i}::{j}::{t}__Qs'] = q.X
            for (i,j,t), q in cache.Qd.items():
                row[f'{i}::{j}::{t}__Qd'] = q.X
        recs.append(row)

    out = pd.DataFrame(recs)
    # basic summary
    num = out.drop(columns=['sample','status'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    if num.shape[1] > 0 and num.shape[0] > 0:
        qs = num.quantile([0.5, 0.05, 0.95], interpolation='linear')
        qs.index = ['median','p05','p95']
        qs.to_csv(f'{save_prefix}__summary.csv', encoding='utf-8-sig')
    out.to_csv(f'{save_prefix}__samples.csv', index=False, encoding='utf-8-sig')

