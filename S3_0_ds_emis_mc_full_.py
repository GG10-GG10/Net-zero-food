

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
import math
import pandas as pd
import gurobipy as gp
from dataclasses import dataclass
import numpy as np
import pandas as pd


def _log_pwl(model: gp.Model, x: gp.Var, y: gp.Var, xmin: float, xmax: float, npts: int, name: str):
    xs = [xmin + k * (xmax - xmin) / max(1, (npts - 1)) for k in range(max(2, npts))]
    ys = [math.log(max(1e-12, v)) for v in xs]
    model.addGenConstrPWL(x, y, xs, ys, name=name)


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
    npts_log: int = 25,
    nutrition_rhs: Optional[Dict[Tuple[str,int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str,int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
    macc_path: Optional[str] = None,
    population_by_country_year: Optional[Dict[Tuple[str,int], float]] = None,
    income_mult_by_country_year: Optional[Dict[Tuple[str,int], float]] = None,
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

    m = gp.Model("nzf_full")
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = 3

    idx = {(n.country, n.commodity, n.year): n for n in nodes}

    # Global prices by (commodity, year)
    Pc: Dict[Tuple[str,int], gp.Var] = {}
    lnPc: Dict[Tuple[str,int], gp.Var] = {}
    import_slack: Dict[Tuple[str,int], gp.Var] = {}
    export_slack: Dict[Tuple[str,int], gp.Var] = {}
    for j in commodities:
        for t in years:
            Pc[j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pc[{j},{t}]")
            lnPc[j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPc[{j},{t}]")
            _log_pwl(m, Pc[j, t], lnPc[j, t], Pmin, Pmax, npts_log, f"logPc[{j},{t}]")
            import_slack[j, t] = m.addVar(lb=0.0, name=f"import_slack[{j},{t}]")
            export_slack[j, t] = m.addVar(lb=0.0, name=f"export_slack[{j},{t}]")

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

    for (i, j, t), n in idx.items():
        # Vars
        Pnet[i, j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pnet[{i},{j},{t}]")
        lnPnet[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPnet[{i},{j},{t}]")
        _log_pwl(m, Pnet[i, j, t], lnPnet[i, j, t], Pmin, Pmax, npts_log, f"logPnet[{i},{j},{t}]")

        Qs[i, j, t] = m.addVar(lb=Qmin, ub=Qmax, name=f"Qs[{i},{j},{t}]")
        Qd[i, j, t] = m.addVar(lb=Qmin, ub=Qmax, name=f"Qd[{i},{j},{t}]")
        lnQs[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQs[{i},{j},{t}]")
        lnQd[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQd[{i},{j},{t}]")
        _log_pwl(m, Qs[i, j, t], lnQs[i, j, t], Qmin, Qmax, npts_log, f"logQs[{i},{j},{t}]")
        _log_pwl(m, Qd[i, j, t], lnQd[i, j, t], Qmin, Qmax, npts_log, f"logQd[{i},{j},{t}]")

        # Price linkage
        tau = float(getattr(n, 'tax_unit', 0.0) or 0.0)
        m.addConstr(Pnet[i, j, t] == Pc[j, t] - tau, name=f"pnet[{i},{j},{t}]")

        # Calibrate alpha_s and alpha_d using baseline Q0/D0 and P0
        # Supply with yield and temperature elasticities
        Q0_eff = float(getattr(n, 'q0_with_ty', lambda: getattr(n, 'Q0', 0.0))())
        P0 = float(getattr(n, 'P0', 1.0) or 1.0)
        Pnet0 = max(1e-12, P0 - tau)
        eps_s = float(getattr(n, 'eps_supply', 0.0) or 0.0)
        eta_y = float(getattr(n, 'eps_supply_yield', 0.0) or 0.0)
        eta_temp = float(getattr(n, 'eps_supply_temp', 0.0) or 0.0)
        Ymult = float(getattr(n, 'Ymult', 1.0) or 1.0)
        Tmult = float(getattr(n, 'Tmult', 1.0) or 1.0)
        alpha_s = math.log(max(1e-12, Q0_eff)) - eps_s * math.log(Pnet0) - eta_y * math.log(max(1e-12, Ymult)) - eta_temp * math.log(max(1e-12, Tmult))
        cs = m.addConstr(lnQs[i, j, t] == alpha_s + eps_s * lnPnet[i, j, t] + eta_y * math.log(max(1e-12, Ymult)) + eta_temp * math.log(max(1e-12, Tmult)),
                         name=f"supply[{i},{j},{t}]")
        constr_supply[(i, j, t)] = cs

        # Demand with cross-price and population
        D0 = float(getattr(n, 'D0', 0.0) or 0.0)
        eps_d = float(getattr(n, 'eps_demand', 0.0) or 0.0)
        eps_pop = float(getattr(n, 'eps_pop_demand', 0.0) or 0.0)
        eps_inc = float(getattr(n, 'eps_income_demand', 0.0) or 0.0)
        row = getattr(n, 'epsD_row', {}) or {}
        # baseline ln Qd = ln D0 = alpha_d + eps_d ln(Pc0[j]) + sum_m epsD_row[m] ln(Pc0[m]) + eps_pop ln(Pop0)
        Pc0_by_comm = {c: float(getattr(n, 'P0', 1.0) or 1.0) for c in commodities}
        ln_base = eps_d * math.log(Pc0_by_comm[j]) + sum(float(row.get(mm, 0.0)) * math.log(Pc0_by_comm[mm]) for mm in commodities)
        # Population base unknown; set Pop0=1 unless provided
        pop0 = 1.0
        alpha_d = math.log(max(1e-12, D0)) - ln_base - eps_pop * math.log(max(1e-12, pop0))
        lnPop = math.log(max(1e-12, float(population_by_country_year.get((i, t), 1.0)) if population_by_country_year else 1.0))
        lnInc = math.log(max(1e-12, float(income_mult_by_country_year.get((i, t), 1.0)) if income_mult_by_country_year else 1.0))
        cd = m.addConstr(lnQd[i, j, t] == alpha_d + eps_d * lnPc[j, t] + gp.quicksum(float(row.get(mm, 0.0)) * lnPc[mm, t] for mm in commodities) + eps_pop * lnPop + eps_inc * lnInc,
                         name=f"demand[{i},{j},{t}]")
        constr_demand[(i, j, t)] = cd

        # Emissions and MACC abatement cost
        Eij[i, j, t] = m.addVar(lb=0.0, name=f"E[{i},{j},{t}]")
        Cij[i, j, t] = m.addVar(lb=0.0, name=f"C[{i},{j},{t}]")
        w = m.addVar(lb=0.0, ub=Pmax, name=f"w[{i},{j},{t}]")  # unit cost adder
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

        # Accumulate totals
        total_C += Cij[i, j, t]
        # LULUCF split for land carbon price term
        e_land = sum(float(v) for p, v in e0_map.items() if _is_lulucf_process(p))
        # Sum of abatement on LULUCF processes
        abat_land = gp.quicksum(a for p, lst in a_by_proc.items() if _is_lulucf_process(p) for a in lst) if a_by_proc else 0.0
        total_E_land += e_land * Qs[i, j, t] - abat_land
        total_E_other += (sum_e0 - e_land) * Qs[i, j, t]

        # McCormick envelope tying unit adder to cost: t_aux = w * Qs; enforce t_aux == Cij to get w = C/Q
        wL, wU = 0.0, Pmax
        qL, qU = Qmin, Qmax
        m.addConstr(t_aux >= wL * Qs[i, j, t] + qL * w - wL * qL, name=f"mc1[{i},{j},{t}]")
        m.addConstr(t_aux >= wU * Qs[i, j, t] + qU * w - wU * qU, name=f"mc2[{i},{j},{t}]")
        m.addConstr(t_aux <= wU * Qs[i, j, t] + qL * w - wU * qL, name=f"mc3[{i},{j},{t}]")
        m.addConstr(t_aux <= wL * Qs[i, j, t] + qU * w - wL * qU, name=f"mc4[{i},{j},{t}]")
        m.addConstr(t_aux == Cij[i, j, t], name=f"t_eq_C[{i},{j},{t}]")
        # Price linkage includes unit adder
        m.addConstr(Pnet[i, j, t] == Pc[j, t] - (tau + w), name=f"pnet2[{i},{j},{t}]")

    # Market clearing per (j,t)
    for j in commodities:
        for t in years:
            supply_sum = gp.quicksum(Qs[i, j, t] for i in countries if (i,j,t) in Qs)
            demand_sum = gp.quicksum(Qd[i, j, t] for i in countries if (i,j,t) in Qd)
            m.addConstr(supply_sum + import_slack[j, t] == demand_sum + export_slack[j, t],
                        name=f"clear[{j},{t}]")

    # Nutrition constraint (future only): sum_j nutrient_per_unit[j] * Qs[i,j,t] >= RHS[i,t]
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
                    if (i,j,t) in Qs:
                        v = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                        if v > 0:
                            expr += v * Qs[i, j, t]
                if expr.size() > 0:
                    cn = m.addConstr(expr >= float(rhs), name=f"nutri[{i},{t}]")
                    nutri_constr[(i, t)] = cn

    # Land area upper bound per (i,t)
    if land_area_limits:
        for i in countries:
            for t in years:
                limit = land_area_limits.get((i, t))
                if limit is None:
                    continue
                # Use node-level yield0 if available (avg 2010-2020), otherwise default
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
                cl = m.addConstr(expr <= float(limit), name=f"landU[{i},{t}]")
                landU_constr[(i, t)] = cl

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
        # Since total_E_land already subtracts abatements, subtract cp * 危 a(LULUCF)
        # This is approximated within per-node accumulation above.

    slack_penalty = 1e-6
    for v in import_slack.values():
        obj += slack_penalty * v
    for v in export_slack.values():
        obj += slack_penalty * v

    m.setObjective(obj, gp.GRB.MINIMIZE)
    # expose cache for in-place MC updates
    m._nzf_cache = dict(
        Pc=Pc, lnPc=lnPc, Pnet=Pnet, lnPnet=lnPnet, Qs=Qs, Qd=Qd, lnQs=lnQs, lnQd=lnQd,
        W=Wunit, Cij=Cij, Eij=Eij, t_aux=Taux,
        constr_supply=constr_supply, constr_demand=constr_demand,
        constr_Edef=constr_Edef, proc_cap_constr=proc_cap_constr, proc_cap_basecoeff=proc_cap_basecoeff,
        e0_by_node=e0_by_node, nutri_constr=nutri_constr, landU_constr=landU_constr,
        import_slack=import_slack, export_slack=export_slack,
        countries=countries, commodities=commodities, years=years,
    )
    m.update()

    var = {
        "Qs": Qs,
        "Qd": Qd,
        "Pc": Pc,
        "Pnet": Pnet,
        "W": Wunit,
        "C": Cij,
        "E": Eij,
        "Import": import_slack,
        "Export": export_slack,
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
    ImportSlack: Dict[Tuple[str,int], gp.Var]
    ExportSlack: Dict[Tuple[str,int], gp.Var]
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
    import_slack = c.get('import_slack', {}); export_slack = c.get('export_slack', {})
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
                      W, Cij, Eij, import_slack, export_slack,
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

