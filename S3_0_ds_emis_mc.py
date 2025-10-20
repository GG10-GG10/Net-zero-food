# -*- coding: utf-8 -*-
"""
Supply–demand model with market clearing and emissions, compatible with S4_0_main.

- Variables per (commodity j, year t): consumer price Pc[j,t] and its log.
- Variables per (country i, commodity j, year t): net producer price Pnet[i,j,t],
  supply Qs[i,j,t], demand Qd[i,j,t], and their logs (PWL-approximated).
- Constraints: log–log supply and demand equations, market clearing per (j,t),
  price linkage Pnet = Pc - tax_unit. Emissions can be attached upstream.
- Returns (model, var) where var['Qd'] maps (i,j,t) -> gp.Var for downstream caps.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import math
import gurobipy as gp


def _log_pwl(model: gp.Model, x: gp.Var, y: gp.Var, xmin: float, xmax: float, npts: int, name: str):
    xs = [xmin + k * (xmax - xmin) / max(1, (npts - 1)) for k in range(max(2, npts))]
    ys = [math.log(max(1e-12, v)) for v in xs]
    model.addGenConstrPWL(x, y, xs, ys, name=name)


def build_model(
    data: Any,
    *,
    price_bounds=(1e-3, 1e6),
    qty_bounds=(1e-6, 1e12),
    npts_log: int = 25,
    nutrition_rhs: Optional[Dict[Tuple[str,int], float]] = None,
    nutrient_per_unit_by_comm: Optional[Dict[str, float]] = None,
    land_area_limits: Optional[Dict[Tuple[str,int], float]] = None,
    yield_t_per_ha_default: float = 3.0,
    land_carbon_price_by_year: Optional[Dict[int, float]] = None,
) -> Tuple[gp.Model, Dict[str, Dict[Tuple[str, str, int], gp.Var]]]:
    nodes = list(getattr(data, 'nodes', []) or [])

    countries = sorted({n.country for n in nodes})
    commodities = sorted({n.commodity for n in nodes})
    years = sorted({n.year for n in nodes})

    Pmin, Pmax = price_bounds
    Qmin, Qmax = qty_bounds

    m = gp.Model("nzf_sd")
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = 3

    # Indexers
    idx = {(n.country, n.commodity, n.year): n for n in nodes}

    # Variables
    Pc: Dict[Tuple[str, int], gp.Var] = {}
    lnPc: Dict[Tuple[str, int], gp.Var] = {}
    for j in commodities:
        for t in years:
            Pc[j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pc[{j},{t}]")
            lnPc[j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPc[{j},{t}]")
            _log_pwl(m, Pc[j, t], lnPc[j, t], Pmin, Pmax, npts_log, f"logPc[{j},{t}]")

    Pnet: Dict[Tuple[str, str, int], gp.Var] = {}
    lnPnet: Dict[Tuple[str, str, int], gp.Var] = {}
    Qs: Dict[Tuple[str, str, int], gp.Var] = {}
    Qd: Dict[Tuple[str, str, int], gp.Var] = {}
    lnQs: Dict[Tuple[str, str, int], gp.Var] = {}
    lnQd: Dict[Tuple[str, str, int], gp.Var] = {}

    for (i, j, t), n in idx.items():
        Pnet[i, j, t] = m.addVar(lb=Pmin, ub=Pmax, name=f"Pnet[{i},{j},{t}]")
        lnPnet[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnPnet[{i},{j},{t}]")
        _log_pwl(m, Pnet[i, j, t], lnPnet[i, j, t], Pmin, Pmax, npts_log, f"logPnet[{i},{j},{t}]")

        Qs[i, j, t] = m.addVar(lb=Qmin, ub=Qmax, name=f"Qs[{i},{j},{t}]")
        Qd[i, j, t] = m.addVar(lb=Qmin, ub=Qmax, name=f"Qd[{i},{j},{t}]")
        lnQs[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQs[{i},{j},{t}]")
        lnQd[i, j, t] = m.addVar(lb=-gp.GRB.INFINITY, name=f"lnQd[{i},{j},{t}]")
        _log_pwl(m, Qs[i, j, t], lnQs[i, j, t], Qmin, Qmax, npts_log, f"logQs[{i},{j},{t}]")
        _log_pwl(m, Qd[i, j, t], lnQd[i, j, t], Qmin, Qmax, npts_log, f"logQd[{i},{j},{t}]")

    # Equations
    for (i, j, t), n in idx.items():
        # Net producer price linkage: Pnet = Pc - tax_unit
        tau = float(getattr(n, 'tax_unit', 0.0) or 0.0)
        m.addConstr(Pnet[i, j, t] == Pc[j, t] - tau, name=f"pnet[{i},{j},{t}]")

        # Supply log–log: ln Qs = ln(Q0_eff) + eps_supply * ln Pnet
        Q0_eff = float(getattr(n, 'q0_with_ty', lambda: getattr(n, 'Q0', 0.0))())
        lnQ0eff = math.log(max(1e-12, Q0_eff))
        eps_s = float(getattr(n, 'eps_supply', 0.0) or 0.0)
        m.addConstr(lnQs[i, j, t] == lnQ0eff + eps_s * lnPnet[i, j, t], name=f"supply[{i},{j},{t}]")

        # Demand log–log: ln Qd = ln(D0) + eps_demand * ln Pc
        D0 = float(getattr(n, 'D0', 0.0) or 0.0)
        lnD0 = math.log(max(1e-12, D0))
        eps_d = float(getattr(n, 'eps_demand', 0.0) or 0.0)
        m.addConstr(lnQd[i, j, t] == lnD0 + eps_d * lnPc[j, t], name=f"demand[{i},{j},{t}]")

    # Market clearing per (j,t): sum_i Qs = sum_i Qd
    for j in commodities:
        for t in years:
            m.addConstr(gp.quicksum(Qs[i, j, t] for i in countries if (i, j, t) in Qs)
                         == gp.quicksum(Qd[i, j, t] for i in countries if (i, j, t) in Qd),
                         name=f"clear[{j},{t}]")

    # Nutrition constraint
    if nutrition_rhs and nutrient_per_unit_by_comm:
        for i in countries:
            for t in years:
                rhs = nutrition_rhs.get((i, t))
                if rhs is None:
                    continue
                expr = gp.LinExpr(0.0)
                for j in commodities:
                    if (i,j,t) in Qs:
                        kpu = float(nutrient_per_unit_by_comm.get(j, 0.0) or 0.0)
                        if kpu > 0:
                            expr += kpu * Qs[i, j, t]
                if len(expr.getVars()) > 0:
                    m.addConstr(expr >= float(rhs), name=f"nutri[{i},{t}]")

    # Land area upper bound
    if land_area_limits:
        for i in countries:
            for t in years:
                limit = land_area_limits.get((i, t))
                if limit is None:
                    continue
                expr = gp.quicksum((1.0 / max(1e-12, yield_t_per_ha_default)) * Qs[i, j, t]
                                   for j in commodities if (i,j,t) in Qs)
                m.addConstr(expr <= float(limit), name=f"landU[{i},{t}]")

    # Land carbon price objective term (LULUCF-only emissions proxy)
    obj = gp.LinExpr(0.0)
    if land_carbon_price_by_year:
        # simple proxy: e_land = sum_{p in LULUCF} e0_p; total cost = cp[t] * e_land * Qs
        for (i, j, t), n in idx.items():
            cp = float(land_carbon_price_by_year.get(t, 0.0) or 0.0)
            if cp <= 0:
                continue
            e0 = getattr(n, 'e0_by_proc', {}) or {}
            e_land = 0.0
            for p, v in e0.items():
                name = (p or '').lower()
                if ('forest' in name) or ('savanna' in name) or ('drained organic' in name) or ('lulucf' in name) or ('land use' in name):
                    e_land += float(v)
            if e_land > 0:
                obj += cp * e_land * Qs[i, j, t]

    m.setObjective(obj)
    m.update()

    return m, {"Qd": Qd}
