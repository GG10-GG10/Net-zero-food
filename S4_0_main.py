# -*- coding: utf-8 -*-
from __future__ import annotations
# argparse disabled
import os
import sys
import traceback
import logging
import logging.handlers
import tempfile
import ctypes
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import gurobipy as gp
from S1_0_schema import ScenarioConfig, ScenarioData
# ---------------- configuration (edit here instead of CLI) ----------------
CFG = {
    'run_mode': 'scenario',  # options: 'single' (single scenario/BASE) | 'scenario' (batch from Scenario sheet) | 'mc' (Monte Carlo)
    'solve': True,
    'use_fao_modules': True,
    'premacc_e0': False,
    'mc_samples': 1000,
    'land_carbon_price': 10.0,
    'intake_indicator': 'energy',  # options: 'energy' | 'protein' | 'fat'
    'future_last_only': False,  # 改为False，包含所有年份（历史 + 所有未来年份）
    'max_growth_rate': 0.1,  # 年增长率：6%/年，60年约33倍
    'max_decline_rate': 0.05,  # 年下降率：5%/年
    'enable_growth_constraints': True,  # Enable growth rate constraints to prevent order-of-magnitude jumps
    'enable_decline_constraint': False,  # 启用下降率约束（默认关闭，避免与历史锚定冲突）
    'iis_timeout': 60,  # IIS计算超时（秒），设为0完全禁用IIS计算，避免模型不可行时卡住
    # === 简化模型选项 ===
    'use_linear_model': True,  # 使用线性弹性模型（无PWL），大幅加快求解
    'use_regional_aggregation': True,  # 使用区域聚合（194国 → 25区域），减少变量
    # === 历史最大产量锚定 ===
    'enable_hist_max_anchor': True,  # 启用历史最大产量锚定约束
    # === 草地处理方式 ===
    'grassland_method': 'dynamic',  # 'dynamic' (方案A, 默认): 草地作为优化变量 | 'static' (方案B): 草地作为外生参数+迭代
    # === 市场失衡限制 ===
    'max_slack_rate': None,  # 短缺/过剩上限：不超过总供给的1% (0.01=1%, 0.05=5%, None=不限制)
    # === 成本计算方法 ===
    'cost_calculation_method': 'unit_cost',  # 'MACC' (基于MACC曲线) | 'unit_cost' (基于给定单位成本)
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
    load_luh2_land_cover,
    load_land_cover_from_excel,  # 简化版土地覆盖读取
    load_forest_ef_from_excel,   # 简化版Forest EF读取
    load_roundwood_supply,
    load_production_statistics,
    load_prices,
    load_intake_constraint,
    load_land_area_limits,
    load_trade_import_export,
    load_emis_item_mappings,
    EmisItemMappings,
    assign_grassland_coef_to_nodes,  # ✅ 新增：分配grassland系数
    load_unit_cost_data,  # ✅ 新增：加载单位成本数据
    load_process_cost_mapping,  # ✅ 新增：加载process到cost的映射
)
try:
    from S3_0_ds_emis_mc_full import build_model as build_model  # prefer full model if available
except Exception as exc:
    print("无法导入 S3_0_ds_emis_mc_full.build_model，原因如下：", file=sys.stderr)
    print(f"{exc}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# 线性区域模型（简化版，无PWL）
try:
    from S3_0_ds_linear_regional import solve_linear_regional, aggregate_nodes_to_regions, load_historical_max_production
    _LINEAR_MODEL_AVAILABLE = True
except ImportError:
    _LINEAR_MODEL_AVAILABLE = False
from S3_5_land_use_change import compute_luc_areas, LUCConfig
from S3_2_feed_demand import build_feed_demand_from_stock
from luc_emission_module import run_luc_bookkeeping, run_luc_emissions_future
from luc_historical_module import read_luc_historical_emissions
from S3_1_emissions_orchestrator_fao import EmissionsFAO, FAOPaths
from S4_1_results import summarize_emissions, summarize_emissions_from_detail, summarize_market
from S3_6_scenarios import (
    load_scenarios,
    apply_scenario_to_data,
    load_mc_specs,
    draw_mc_to_params,
)
from gle_emissions_complete import run_livestock_emissions, calculate_stock_from_optimized_production
from gce_emissions_complete import run_crop_emissions
from gfire_emission_fixed_module import load_fixed_gfire_emissions
from gsoil_emission_complete import run_drained_organic_soils_emissions
from config_paths import get_input_base, get_results_base, get_src_base
paths = DataPaths()
item_maps = load_emis_item_mappings(paths.dict_v3_path)
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
class _DedupFilter(logging.Filter):
    """过滤重复日志，确保同样的消息只记录一次。"""
    _seen_messages = set()
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg in self._seen_messages:
            return False
        self._seen_messages.add(msg)
        return True
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
def _build_coarse_transitions_from_deltas(deltas: Optional[pd.DataFrame],
                                          iso_map: Dict[str, str]) -> pd.DataFrame:
    columns = ['country', 'iso3', 'year',
               'forest_to_cropland', 'forest_to_pasture',
               'cropland_to_forest', 'pasture_to_forest']
    if not isinstance(deltas, pd.DataFrame) or deltas.empty:
        return pd.DataFrame(columns=columns)
    df = deltas.copy()
    df['iso3'] = df['country'].map(iso_map)
    df = df.dropna(subset=['iso3'])
    records: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        try:
            country = str(getattr(row, 'country'))
            iso3 = str(getattr(row, 'iso3'))
            year = int(getattr(row, 'year'))
            dc = float(getattr(row, 'd_cropland_ha', 0.0) or 0.0)
            dg = float(getattr(row, 'd_grassland_ha', 0.0) or 0.0)
            df_forest = float(getattr(row, 'd_forest_ha', 0.0) or 0.0)
        except Exception:
            continue
        fc = max(dc, 0.0)
        fp = max(dg, 0.0)
        cf = max(-dc, 0.0)
        pf = max(-dg, 0.0)
        forest_loss = max(-df_forest, 0.0)
        forest_gain = max(df_forest, 0.0)
        if fc + fp > 0.0:
            if forest_loss > 0.0:
                scale = forest_loss / (fc + fp)
                fc *= scale
                fp *= scale
            else:
                fc = 0.0
                fp = 0.0
        if cf + pf > 0.0:
            if forest_gain > 0.0:
                scale = forest_gain / (cf + pf)
                cf *= scale
                pf *= scale
            else:
                cf = 0.0
                pf = 0.0
        if (fc + fp + cf + pf) <= 0.0:
            continue
        records.append({
            'country': country,
            'iso3': iso3,
            'year': year,
            'forest_to_cropland': max(fc, 0.0),
            'forest_to_pasture': max(fp, 0.0),
            'cropland_to_forest': max(cf, 0.0),
            'pasture_to_forest': max(pf, 0.0),
        })
    if not records:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame.from_records(records, columns=columns)
    return out.groupby(['country', 'iso3', 'year'], as_index=False).sum()
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
class _StreamTee:
    """Mirror stdout/stderr to the logger while keeping console output."""
    def __init__(self, logger_func, mirror_stream):
        self.logger_func = logger_func
        self.mirror_stream = mirror_stream
        self._buffer = ""
    def write(self, message: str) -> int:
        if self.mirror_stream:
            try:
                self.mirror_stream.write(message)
                self.mirror_stream.flush()
            except Exception:
                pass
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                try:
                    self.logger_func(line)
                except Exception:
                    pass
        return len(message)
    def flush(self) -> None:
        if self.mirror_stream:
            try:
                self.mirror_stream.flush()
            except Exception:
                pass
        if self._buffer.strip():
            try:
                self.logger_func(self._buffer.rstrip("\n"))
            except Exception:
                pass
            self._buffer = ""
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
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR
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
    # Reset root handlers so they do not accumulate between runs
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    # Restore std streams before wiring tees (prevents nested wrappers)
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    # Ensure a fresh log file each run
    model_log_file = log_dir / "model.log"
    try:
        if model_log_file.exists():
            model_log_file.unlink()
    except Exception:
        # Best-effort cleanup; proceed even if deletion fails
        pass
    # File handler with timestamped lines
    file_handler = logging.FileHandler(model_log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(_DedupFilter())
    # Console handler keeps stdout readable
    console_handler = logging.StreamHandler(stream=_ORIGINAL_STDOUT)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(_DedupFilter())
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    # Capture warnings into logging so they land in model.log
    logging.captureWarnings(True)
    # Tee stdout/stderr into logger while preserving console output
    sys.stdout = _StreamTee(root_logger.info, _ORIGINAL_STDOUT)
    sys.stderr = _StreamTee(root_logger.error, _ORIGINAL_STDERR)
    return logger
def _log_step(message: str, *, print_console: bool = True) -> None:
    msg = f"[NZF] {message}"
    if model_logger:
        model_logger.info(msg)
    elif print_console:
        print(msg)
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
                      gp.GRB.INTERRUPTED, gp.GRB.USER_OBJ_LIMIT, gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE}
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
    import_by = defaultdict(float)
    export_by = defaultdict(float)
    energy_total = defaultdict(float)
    protein_total = defaultdict(float)
    population_by: Dict[Tuple[str,int], float] = {}
    emis_detail_rows: List[Dict[str, Any]] = []
    # WARNING: derive the years to output (historical 2010-2020 plus configured future years)
    valid_years_for_output = set()
    if data and hasattr(data, 'nodes'):
        valid_years_for_output = {n.year for n in data.nodes}
    nodes_skipped = 0
    nodes_processed = 0
    for key, node in idx.items():
        i, j, t = key
        # WARNING: skip nodes whose year is not in the valid output list
        if valid_years_for_output and t not in valid_years_for_output:
            nodes_skipped += 1
            continue
        nodes_processed += 1
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
        import_hist = None
        export_hist = None
        meta = getattr(node, 'meta', None)
        if isinstance(meta, dict):
            import_hist = meta.get('import_hist')
            export_hist = meta.get('export_hist')
        if s_var is not None or d_var is not None:
            imp = max(demand - supply, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
            exp = max(supply - demand, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
        else:
            imp = max(demand - supply, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
            exp = max(supply - demand, 0.0) if np.isfinite(supply) and np.isfinite(demand) else np.nan
        if import_hist is not None or export_hist is not None:
            imp = float(import_hist or 0.0)
            exp = float(export_hist or 0.0)
        supply_by[(i, t)] += 0.0 if not np.isfinite(supply) else supply
        demand_by[(i, t)] += 0.0 if not np.isfinite(demand) else demand
        cost_by[(i, t)] += 0.0 if not np.isfinite(abat_cost) else abat_cost
        emis_by[(i, t)] += 0.0 if not np.isfinite(emis) else emis
        if np.isfinite(imp):
            import_by[(i, t)] += imp
        if np.isfinite(exp):
            export_by[(i, t)] += exp
        e_factor = energy_map.get(j)
        if np.isfinite(demand) and e_factor is not None and np.isfinite(e_factor):
            energy_total[(i, t)] += demand * float(e_factor)
        p_factor = protein_map.get(j)
        if np.isfinite(demand) and p_factor is not None and np.isfinite(p_factor):
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
                # 使用标准化的列名以匹配summarize_emissions_from_detail的期望
                emis_detail_rows.append({
                    'M49': getattr(node, 'm49', ''),
                    'Country': i,  # 标准化为大写
                    'year': t,
                    'Item': j,  # 从 'commodity' 改为 'Item'
                    'Process': proc_name,  # 从 'process' 改为 'Process'
                    'GHG': 'CO2e',  # 添加温室气体类型
                    'value': float(adj_val) if np.isfinite(adj_val) else 0.0,  # 从 'emissions_tco2e' 改为 'value'
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
            energy_pc[key] = total / (float(pop) * 365.0)
    for key, total in protein_total.items():
        pop = population_by.get(key, _baseline_float(population_map.get(key), np.nan))
        if pop and pop > 0:
            protein_pc[key] = total / (float(pop) * 365.0)
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
            'imports_t': import_by.get((country, year), imports),
            'exports_t': export_by.get((country, year), exports),
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
        # ✅ 只输出历史年份(<=2020)到 LUH2base 表，未来年份不需要
        if year > 2020:
            continue
        land_rows.append({
            'country': country,
            'year': year,
            'cropland_area_ha': vals.get('cropland_area_ha', np.nan),
            'forest_area_ha': vals.get('forest_area_ha', np.nan),
            'pasture_area_ha': vals.get('pasture_area_ha', np.nan),
        })
    print(f"  - nodes_processed: {nodes_processed}")
    print(f"  - nodes_skipped: {nodes_skipped}")
    print(f"  - emis_detail_rows: {len(emis_detail_rows)} 行")
    if emis_detail_rows:
        years_in_emis = sorted(set(r['year'] for r in emis_detail_rows if 'year' in r))
        print(f"  - emis_detail_rows包含的年份: {years_in_emis}")
    return {
        'node_detail': node_df,
        'country_year_summary': pd.DataFrame(country_year_rows),
        'nutrition_per_capita': pd.DataFrame(nutrition_rows),
        'land_use_LUH2_summary': pd.DataFrame(land_rows),
        'emissions_detail': pd.DataFrame(emis_detail_rows),
    }
def build_production_summary(
    node_df: pd.DataFrame,
    *,
    yield_df: pd.DataFrame,
    area_df: pd.DataFrame,
    fertilizer_eff_df: pd.DataFrame,
    fertilizer_amt_df: pd.DataFrame,
    feed_req_df: pd.DataFrame,
    slaughter_df: pd.DataFrame,
    livestock_yield_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    manure_ratio_df: pd.DataFrame,
    luc_area_df: pd.DataFrame,
    feed_amount_df: Optional[pd.DataFrame] = None,
    grass_share_df: Optional[pd.DataFrame] = None,
    feed_eff_df: Optional[pd.DataFrame] = None,
    scenario_ctx: Optional[Dict] = None,
    m49_by_country: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    columns = [
        'M49_Country_Code','country','iso3','year','commodity','production_t','production_unit',
        'yield_t_per_ha','yield_unit','crop_area_ha','crop_area_unit',
        'fertilizer_efficiency_kgN_per_ha','fertilizer_efficiency_unit',
        'fertilizer_n_input_t','fertilizer_amount_unit','slaughter_head','slaughter_unit',
        'yield_t_per_head','yield_unit_livestock','stock_head','stock_unit',
        'feed_requirement_kg_per_head','feed_requirement_unit','feed_amount_t','feed_amount_unit',
        'grass_feed_share','feed_efficiency_multiplier',
        'pasture_area_ha','pasture_area_unit','manure_management_ratio','manure_ratio_unit'
    ]
    if node_df is None or node_df.empty:
        return pd.DataFrame(columns=columns)
    
    # 构建国家→M49映射（标准化格式：'xxx，单引号+3位数字）
    def _normalize_m49(m49_val):
        """将M49标准化为'xxx格式（单引号+3位数字，如'004）"""
        if m49_val is None or pd.isna(m49_val):
            return ''
        m49_str = str(m49_val).strip().strip("'\"")
        try:
            m49_int = int(m49_str)
            # ✅ 确保格式：'004（单引号前缀+3位数字）
            return f"'{m49_int:03d}"
        except Exception:
            return str(m49_val)
    
    # 首先尝试从参数传入的m49_by_country使用（来自universe）
    country_to_m49 = {}
    if m49_by_country:
        # 标准化传入的m49_by_country映射
        for country, m49 in m49_by_country.items():
            country_to_m49[str(country).strip()] = _normalize_m49(m49)
    
    # 如果还没有建立映射，尝试从node_df提取
    if not country_to_m49 and 'm49' in node_df.columns:
        # 从node_df提取M49映射
        tmp = node_df[['country', 'm49']].dropna().drop_duplicates()
        for _, row in tmp.iterrows():
            country = str(row['country']).strip()
            m49 = _normalize_m49(row['m49'])
            if m49:
                country_to_m49[country] = m49
    
    def _collect_iso_lookup(df: Optional[pd.DataFrame]) -> Dict[Any, Any]:
        if df is None or df.empty:
            return {}
        if 'country' not in df.columns or 'iso3' not in df.columns:
            return {}
        tmp = df[['country', 'iso3']].dropna().drop_duplicates()
        if tmp.empty:
            return {}
        return tmp.set_index('country')['iso3'].to_dict()
    initial_iso_lookup: Dict[Any, Any] = {}
    for src in [
        node_df,
        yield_df,
        area_df,
        fertilizer_eff_df,
        fertilizer_amt_df,
        feed_req_df,
        slaughter_df,
        livestock_yield_df,
        stock_df,
        manure_ratio_df,
        luc_area_df,
    ]:
        initial_iso_lookup.update(_collect_iso_lookup(src))
    
    # 提取M49列（如果存在）
    base_cols = ['country', 'year', 'commodity', 'supply_t']
    if 'm49' in node_df.columns:
        base_cols.insert(0, 'm49')
    if 'iso3' in node_df.columns:
        base_cols.insert(1, 'iso3')
    summary = node_df[base_cols].copy()
    
    # 标准化M49格式（如果存在）
    if 'm49' in summary.columns:
        summary['m49'] = summary['m49'].apply(lambda x: _normalize_m49(x) if pd.notna(x) else '')
        # 如果没有m49，从country_to_m49映射填充
        if summary['m49'].isna().any() or (summary['m49'] == '').any():
            summary['m49'] = summary.apply(
                lambda row: country_to_m49.get(str(row['country']).strip(), '') 
                if pd.isna(row.get('m49')) or row.get('m49') == '' else row['m49'],
                axis=1
            )
        # 重命名为标准列名
        summary = summary.rename(columns={'m49': 'M49_Country_Code'})
    else:
        # 如果node_df中没有m49，从country_to_m49创建
        summary['M49_Country_Code'] = summary['country'].map(lambda c: country_to_m49.get(str(c).strip(), ''))
    
    # 重新排列列，确保M49在首位
    cols = list(summary.columns)
    if 'M49_Country_Code' in cols:
        cols.remove('M49_Country_Code')
        cols = ['M49_Country_Code'] + cols
        summary = summary[cols]
    
    if 'iso3' not in summary.columns:
        summary['iso3'] = summary['country'].map(initial_iso_lookup)
    else:
        summary['iso3'] = summary['iso3'].where(
            summary['iso3'].notna(), summary['country'].map(initial_iso_lookup)
        )
    if summary['iso3'].notna().any():
        mask = summary['iso3'].notna()
        summary.loc[mask, 'iso3'] = summary.loc[mask, 'iso3'].astype(str).str.strip()
    summary = summary.rename(columns={'supply_t': 'production_t'})
    # WARNING: fallback: pull parameter columns from node_df for future years
    # node中的yield0 → yield_t_per_ha
    # 注意：for livestock, don't fill yield_t_per_ha - use yield_t_per_head instead
    if 'yield0_t_per_ha' in node_df.columns and 'commodity' in summary.columns:
        # 定义livestock商品列表
        # 注意：这些是Item_Emis标准名称，stock_df/slaughter_df中的commodity已经映射为Item_Emis
        livestock_items = {
            'Cattle, dairy',
            'Cattle, non-dairy',
            'Buffalo, dairy',
            'Buffalo, non-dairy',
            'Sheep, dairy',
            'Sheep, non-dairy',
            'Goats, dairy',
            'Goats, non-dairy',
            'Swine',
            'Chickens, broilers',
            'Chickens, layers',
            'Ducks',
            'Turkeys',
            'Horses',
            'Asses',
            'Mules and hinnies',
            'Camel, dairy',
            'Camel, non-dairy',
            'Llamas',
        }
        # 只填充非livestock商品的yield_t_per_ha
        is_livestock = summary['commodity'].isin(livestock_items)
        node_yield = node_df.set_index(['country', 'year', 'commodity'])['yield0_t_per_ha'] if 'yield0_t_per_ha' in node_df.columns else None
        if node_yield is not None:
            summary.loc[~is_livestock, 'yield_t_per_ha_from_node'] = summary.loc[~is_livestock].apply(
                lambda row: node_yield.get((row['country'], row['year'], row['commodity']), None),
                axis=1
            )
    use_iso3 = summary['iso3'].notna().any()
    iso_lookup = summary[['country', 'iso3']].dropna().drop_duplicates().set_index('country')['iso3'].to_dict()
    
    # 定义livestock商品列表（用于yield过滤）
    # 注意：这些是Item_Emis标准名称，因为stock_df/slaughter_df中的commodity已经映射为Item_Emis
    livestock_items = {
        'Cattle, dairy',
        'Cattle, non-dairy',
        'Buffalo, dairy',
        'Buffalo, non-dairy',
        'Sheep, dairy',
        'Sheep, non-dairy',
        'Goats, dairy',
        'Goats, non-dairy',
        'Swine',
        'Chickens, broilers',
        'Chickens, layers',
        'Ducks',
        'Turkeys',
        'Horses',
        'Asses',
        'Mules and hinnies',
        'Camel, dairy',
        'Camel, non-dairy',
        'Llamas',
    }
    
    def _merge(df, value_col, alias: Optional[str]=None, skip_livestock=False):
        nonlocal summary
        if df is None or df.empty or value_col not in df.columns:
            return
        
        # 如果skip_livestock为True且value_col是yield_t_per_ha，排除livestock商品
        if skip_livestock and value_col == 'yield_t_per_ha':
            df = df[~df['commodity'].isin(livestock_items)].copy() if 'commodity' in df.columns else df.copy()
            if df.empty:
                return
        
        # ✅ 使用M49_Country_Code作为唯一标识符进行join
        join_keys = ['M49_Country_Code', 'year', 'commodity']
        
        df = df.copy()
        if 'M49_Country_Code' not in df.columns:
            print(f"  [WARN] _merge({value_col}): DataFrame无M49_Country_Code列，跳过！")
            return
        
        # 标准化M49格式
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49)
        
        cols = join_keys + [value_col]
        if not set(cols).issubset(df.columns):
            return
        rename_map = {}
        if alias and alias != value_col:
            rename_map[value_col] = alias
        data = df[cols].drop_duplicates().rename(columns=rename_map)
        
        # 记录merge前的非空值数量
        non_null_before = summary[alias or value_col].notna().sum() if (alias or value_col) in summary.columns else 0
        
        summary = summary.merge(data, how='left', on=join_keys)
        
        # 记录merge后的非空值数量，用于验证merge是否有效
        target_col = alias or value_col
        non_null_after = summary[target_col].notna().sum() if target_col in summary.columns else 0
        new_filled = non_null_after - non_null_before
        
        # DEBUG: 如果merge没有填充任何值，发出警告
        if new_filled == 0 and len(data) > 0:
            print(f"  [WARN] _merge({value_col}): 虽然有{len(data)}行输入数据，但没有填充任何值!")
            print(f"    输入M49样本: {df['M49_Country_Code'].unique()[:3].tolist()}")
            print(f"    summary M49样本: {summary['M49_Country_Code'].unique()[:3].tolist()}")
    # WARNING: for future years (>2020), extend historical DataFrames using 2020 baseline plus scenario adjustments
    # 1. 识别summary中的未来年份
    future_years = [y for y in summary['year'].unique() if y > 2020]
    hist_year_2020 = 2020
    # 2. 扩展各参数DataFrames：为未来年份复制2020年baseline，然后应用scenario调整
    def extend_to_future_years(df: pd.DataFrame, value_col: str, future_yrs: List[int]) -> pd.DataFrame:
        """扩展DataFrame到未来年份：复制2020年数据到未来年份"""
        if df is None or df.empty or value_col not in df.columns:
            return df
        if not future_yrs:
            return df
        # 提取2020年数据
        baseline_2020 = df[df['year'] == hist_year_2020].copy()
        if baseline_2020.empty:
            print(f"[WARN] {value_col}: 没有2020年baseline数据，无法扩展")
            return df
        # 为每个未来年份复制2020年数据
        extended_rows = []
        for yr in future_yrs:
            yr_data = baseline_2020.copy()
            yr_data['year'] = yr
            extended_rows.append(yr_data)
        if extended_rows:
            extended_df = pd.concat([df] + extended_rows, ignore_index=True)
            return extended_df
        return df
    # 扩展所有参数DataFrames
    # NOTE: 仅对历史参数扩展到未来年份以保持兼容
    # 但对于yield等技术参数，未来年份由模型优化决定，不应固定为2020年值
    yield_df = extend_to_future_years(yield_df, 'yield_t_per_ha', future_years)
    # area 不复制 2020 基准到未来，避免未来面积被固定为历史值
    # fertilizer_eff_df: 肥料效率取决于产量和投入，不应复制
    # fertilizer_eff_df = extend_to_future_years(fertilizer_eff_df, 'fertilizer_efficiency_kgN_per_ha', future_years)
    # fertilizer_amt_df: 肥料投入量取决于模型优化，不应复制
    # fertilizer_amt_df = extend_to_future_years(fertilizer_amt_df, 'fertilizer_n_input_t', future_years)
    # ❌ 不再扩展livestock参数！GLE模块已经导出了未来年份的计算数据
    # 扩展会导致重复数据：2020 baseline + GLE计算的2080数据
    # slaughter_df = extend_to_future_years(slaughter_df, 'slaughter_head', future_years)
    # livestock_yield_df = extend_to_future_years(livestock_yield_df, 'yield_t_per_head', future_years)
    # stock_df = extend_to_future_years(stock_df, 'stock_head', future_years)
    # feed_req_df: 饲料需求取决于产量和饮食结构，不应复制2020年值
    # feed_req_df = extend_to_future_years(feed_req_df, 'feed_requirement_kg_per_head', future_years)
    manure_ratio_df = extend_to_future_years(manure_ratio_df, 'manure_management_ratio', future_years)
    # 3. 应用scenario调整 (如果有scenario_ctx)
    if scenario_ctx and isinstance(scenario_ctx, dict):
        # feed efficiency调整：dm_conversion_multiplier
        dm_mult = scenario_ctx.get('dm_conversion_multiplier', {})
        if dm_mult and feed_req_df is not None and not feed_req_df.empty and 'feed_requirement_kg_per_head' in feed_req_df.columns:
            def apply_dm_mult_to_row(row):
                key = (row['country'], row['commodity'], row['year'])
                mult = dm_mult.get(key, 1.0)
                if mult != 1.0 and row['year'] > 2020:
                    row = row.copy()
                    row['feed_requirement_kg_per_head'] = row['feed_requirement_kg_per_head'] * mult
                return row
            feed_req_df = pd.DataFrame([apply_dm_mult_to_row(row) for idx, row in feed_req_df.iterrows()])
    # 现在执行merge
    # 注意：yield_df中的livestock产率不应填充（livestock使用yield_t_per_head）
    _merge(yield_df, 'yield_t_per_ha', skip_livestock=True)
    if area_df is not None and not area_df.empty and 'area_ha' in area_df.columns:
        # ✅ 使用M49_Country_Code作为唯一标识符
        join_keys = ['M49_Country_Code', 'year', 'commodity']
        cols = join_keys + ['area_ha']
        area_tmp = area_df.copy()
        if 'M49_Country_Code' in area_tmp.columns:
            area_tmp['M49_Country_Code'] = area_tmp['M49_Country_Code'].apply(_normalize_m49)
            if set(cols).issubset(area_tmp.columns):
                area_tmp = area_tmp[cols].drop_duplicates()
                area_tmp = area_tmp.rename(columns={'area_ha': 'crop_area_ha'})
                summary = summary.merge(area_tmp, how='left', on=join_keys)
    _merge(fertilizer_eff_df, 'fertilizer_efficiency_kgN_per_ha')
    _merge(fertilizer_amt_df, 'fertilizer_n_input_t')
    # 注意: slaughter_df 和 livestock_yield_df 可能使用不同的 Item_XXX_Map 名称
    # 它们将在下面的 commodity 映射区块完成后再合并，确保名称已转换为 production_map
    # WARNING: After Q0 fix, summary['commodity'] now uses Item_Emis (standard model names like "Cattle, non-dairy")
    # Each parameter DataFrame uses different Item_XXX_Map naming (from FAOSTAT):
    #   - stock_df uses Item_Stock_Map names (e.g., "Cattle, dairy")
    #   - slaughter_df uses Item_Slaughtered_Map names (e.g., "Meat of cattle with the bone, fresh or chilled")
    #   - livestock_yield_df uses Item_Yield_Map names
    # 
    # KEY INSIGHT (from gle_emissions_complete.py):
    # The correct approach is to build REVERSE mappings:
    #   Item_Stock_Map → Item_Emis (反向映射)
    #   Item_Slaughtered_Map → Item_Emis
    #   Item_Yield_Map → Item_Emis
    # This way stock_df/slaughter_df commodity names can be matched to summary commodity names (Item_Emis)
    
    try:
        # NOTE: livestock参数DataFrames (stock_df, slaughter_df, livestock_yield_df)
        # 已在load_production_statistics中通过build_faostat_production_indicators完成了从
        # FAOSTAT名称到Item_Emis的映射，使用dict_v3的Item_Stock_Map/Item_Slaughtered_Map/Item_Yield_Map等。
        # 因此这些DataFrame的commodity列已经是Item_Emis标准名称，可以直接merge，无需再映射。
        _merge(stock_df, 'stock_head')
        _merge(slaughter_df, 'slaughter_head')
        _merge(livestock_yield_df, 'yield_t_per_head')
        _merge(feed_req_df, 'feed_requirement_kg_per_head')
        _merge(manure_ratio_df, 'manure_management_ratio')
        # Merge feed-related DataFrames if available
        if feed_amount_df is not None and not feed_amount_df.empty:
            _merge(feed_amount_df, 'feed_amount_t')
        if grass_share_df is not None and not grass_share_df.empty:
            _merge(grass_share_df, 'grass_feed_share')
        if feed_eff_df is not None and not feed_eff_df.empty:
            _merge(feed_eff_df, 'feed_efficiency_multiplier')
    except Exception as e:
        print(f"[WARN] 无法加载commodity映射，livestock参数可能无法正确匹配: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: 直接merge（可能会失败）
        _merge(stock_df, 'stock_head')
        _merge(feed_req_df, 'feed_requirement_kg_per_head')
        _merge(manure_ratio_df, 'manure_management_ratio')
    
    if luc_area_df is not None and not luc_area_df.empty and 'land_use' in luc_area_df.columns:
        # ✅ 修复：grassland_area_ha 和 pasture_area_ha 是同一个概念的不同名称
        # S3_5 输出是 grassland_ha -> melt 后变成 grassland_area_ha
        pasture = luc_area_df[luc_area_df['land_use'].isin(['pasture_area_ha', 'grassland_area_ha'])]
        if not pasture.empty:
            # ✅ 使用M49_Country_Code作为唯一标识符
            pasture_tmp = pasture.copy()
            if 'M49_Country_Code' in pasture_tmp.columns:
                pasture_tmp['M49_Country_Code'] = pasture_tmp['M49_Country_Code'].apply(_normalize_m49)
                cols = ['M49_Country_Code', 'year', 'area_ha']
                if set(cols).issubset(pasture_tmp.columns):
                    pasture_tmp = pasture_tmp[cols].drop_duplicates()
                    pasture_tmp = pasture_tmp.rename(columns={'area_ha': 'pasture_area_ha'})
                    summary = summary.merge(pasture_tmp, how='left', on=['M49_Country_Code', 'year'])
        else:
            print(f"[DEBUG] ⚠️ luc_area_df 中没有 pasture/grassland 数据! land_use 值: {luc_area_df['land_use'].unique().tolist()}")
    if isinstance(feed_amount_df, pd.DataFrame) and not feed_amount_df.empty:
        tmp = feed_amount_df.copy()
        if 'feed_t' in tmp.columns:
            tmp = tmp.rename(columns={'feed_t': 'feed_amount_override'})
            _merge(tmp, 'feed_amount_override')
        elif 'feed_amount_t' in tmp.columns:
            tmp = tmp.rename(columns={'feed_amount_t': 'feed_amount_override'})
            _merge(tmp, 'feed_amount_override')
    if isinstance(feed_eff_df, pd.DataFrame) and not feed_eff_df.empty:
        _merge(feed_eff_df, 'feed_efficiency_multiplier')
    if isinstance(grass_share_df, pd.DataFrame) and not grass_share_df.empty:
        # ✅ 使用M49_Country_Code作为唯一标识符
        grass_tmp = grass_share_df.copy()
        if 'M49_Country_Code' in grass_tmp.columns:
            grass_tmp['M49_Country_Code'] = grass_tmp['M49_Country_Code'].apply(_normalize_m49)
            cols = ['M49_Country_Code', 'year', 'grass_feed_share']
            if set(cols).issubset(grass_tmp.columns):
                grass_tmp = grass_tmp[cols].drop_duplicates()
                summary = summary.merge(grass_tmp, how='left', on=['M49_Country_Code', 'year'])
    
    # ✅ 确保所有必需的列都存在（如果_merge跳过，这些列可能不存在）
    required_cols = {
        'yield_t_per_ha': np.nan,
        'crop_area_ha': np.nan,
        'fertilizer_efficiency_kgN_per_ha': np.nan,
        'fertilizer_n_input_t': np.nan,
        'slaughter_head': np.nan,
        'yield_t_per_head': np.nan,
        'stock_head': np.nan,
        'feed_requirement_kg_per_head': np.nan,
        'manure_management_ratio': np.nan,
        'grass_feed_share': np.nan,
        'feed_efficiency_multiplier': np.nan,
        'pasture_area_ha': np.nan,
    }
    for col, default_val in required_cols.items():
        if col not in summary.columns:
            summary[col] = default_val
    
    feed_req_series = summary['feed_requirement_kg_per_head'] if 'feed_requirement_kg_per_head' in summary.columns else pd.Series(np.nan, index=summary.index)
    stock_series = summary['stock_head'] if 'stock_head' in summary.columns else pd.Series(np.nan, index=summary.index)
    feed_amount_calc = np.where(
        feed_req_series.notna() & stock_series.notna(),
        feed_req_series * stock_series / 1000.0,
        np.nan,
    )
    summary['feed_amount_t'] = feed_amount_calc
    if 'feed_amount_override' in summary.columns:
        summary['feed_amount_t'] = summary['feed_amount_override'].where(summary['feed_amount_override'].notna(),
                                                                         summary['feed_amount_t'])
        summary = summary.drop(columns=['feed_amount_override'])
    # 补足未来年的作物面积：按产量 / 单产重新计算，避免2020面积被硬复制
    # 优先使用节点里的 yield0 (yield_t_per_ha_from_node)，否则用合并得到的历史/情景 yield
    if 'yield_t_per_ha_from_node' in summary.columns:
        # ✅ 确保yield_t_per_ha列存在
        if 'yield_t_per_ha' not in summary.columns:
            summary['yield_t_per_ha'] = np.nan
        summary['yield_t_per_ha'] = summary['yield_t_per_ha'].combine_first(summary['yield_t_per_ha_from_node'])
        summary = summary.drop(columns=['yield_t_per_ha_from_node'])
    future_mask = summary['year'] > 2020
    if 'production_t' in summary.columns and 'yield_t_per_ha' in summary.columns:
        prod = summary['production_t']
        yld = summary['yield_t_per_ha']
        recomputed_area = np.where(
            future_mask & prod.notna() & yld.notna() & (yld > 0),
            prod / yld,
            np.nan
        )
        if 'crop_area_ha' not in summary.columns:
            summary['crop_area_ha'] = np.nan
        summary['crop_area_ha'] = summary['crop_area_ha'].where(~future_mask | summary['crop_area_ha'].notna(),
                                                                recomputed_area)
        # 如果未来面积已存在但为零，也用计算值填充
        summary['crop_area_ha'] = summary['crop_area_ha'].where(~(future_mask & (summary['crop_area_ha'] == 0)),
                                                                recomputed_area)
    
    # ============================================================================
    # 注意：slaughter_head和stock_head现在直接从GLE模块导出的计算结果读取
    # GLE在计算排放时会使用 slaughter = production / carcass_weight
    # 并保存到 livestock_parameters['slaughter'] 和 livestock_parameters['stock']
    # 这些数据在本函数调用前已经通过slaughter_df和stock_df参数传入
    # ============================================================================
    
    # 基于面积和施肥强度计算肥料投入（kgN/ha -> tN）
    if 'fertilizer_n_input_t' not in summary.columns:
        summary['fertilizer_n_input_t'] = np.nan
    if 'fertilizer_efficiency_kgN_per_ha' in summary.columns and 'crop_area_ha' in summary.columns:
        eff = summary['fertilizer_efficiency_kgN_per_ha']
        area = summary['crop_area_ha']
        fert_calc = np.where(
            eff.notna() & area.notna() & (area > 0),
            eff * area / 1000.0,  # kgN/ha * ha -> kgN -> tN
            np.nan
        )
        summary['fertilizer_n_input_t'] = summary['fertilizer_n_input_t'].where(summary['fertilizer_n_input_t'].notna(),
                                                                                fert_calc)
        # 未来年份强制更新为计算值，避免沿用历史常数
        summary['fertilizer_n_input_t'] = summary['fertilizer_n_input_t'].where(~future_mask, fert_calc)
    summary['production_unit'] = 't'
    summary['yield_unit'] = 't/ha'
    summary['crop_area_unit'] = 'ha'
    summary['fertilizer_efficiency_unit'] = 'kgN/ha'
    summary['fertilizer_amount_unit'] = 'tN'
    summary['slaughter_unit'] = 'head'
    summary['yield_unit_livestock'] = 't/head'
    summary['stock_unit'] = 'head'
    summary['feed_requirement_unit'] = 'kg/head'
    summary['feed_amount_unit'] = 't'
    summary['pasture_area_unit'] = 'ha'
    summary['manure_ratio_unit'] = 'ratio'
    for col in ['crop_area_ha','fertilizer_efficiency_kgN_per_ha','fertilizer_n_input_t',
                'slaughter_head','yield_t_per_head','stock_head',
                'feed_requirement_kg_per_head','feed_amount_t','pasture_area_ha',
                'manure_management_ratio']:
        if col not in summary.columns:
            summary[col] = np.nan
    for col in ['grass_feed_share','feed_efficiency_multiplier']:
        if col not in summary.columns:
            summary[col] = np.nan
    summary = summary[columns].copy()
    
    # ✅ 关键修复：去除重复数据
    # 按照country-year-commodity去重，保留最后一次merge的结果（通常是最准确的）
    before_dedup = len(summary)
    summary = summary.drop_duplicates(subset=['country', 'year', 'commodity'], keep='last')
    after_dedup = len(summary)
    if before_dedup != after_dedup:
        print(f"[INFO] 去重：删除了 {before_dedup - after_dedup} 条重复记录")
    
    summary = summary.sort_values(['country','year','commodity']).reset_index(drop=True)
    return summary
def apply_trade_baseline_to_nodes(nodes: List[Node],
                                  *,
                                  hist_end: int,
                                  trade_imports: Dict[Tuple[str, str, int], float],
                                  trade_exports: Dict[Tuple[str, str, int], float]) -> Dict[Tuple[str, str], Tuple[int, float]]:
    """Override historical D0 using trade flows: Qd = Qs + Import - Export."""
    latest_hist: Dict[Tuple[str, str], Tuple[int, float]] = {}
    if not trade_imports and not trade_exports:
        for n in nodes:
            if n.year <= hist_end:
                d0 = float(getattr(n, 'D0', 0.0) or 0.0)
                key = (n.country, n.commodity)
                prev = latest_hist.get(key)
                if prev is None or n.year >= prev[0]:
                    latest_hist[key] = (n.year, d0)
        for n in nodes:
            if n.year > hist_end:
                if float(getattr(n, 'D0', 0.0) or 0.0) <= 0.0:
                    prev = latest_hist.get((n.country, n.commodity))
                    if prev is not None and prev[1] > 0.0:
                        n.D0 = prev[1]
        return latest_hist
    for n in nodes:
        key_full = (n.country, n.commodity, n.year)
        if n.year <= hist_end:
            imp = float(trade_imports.get(key_full, 0.0) or 0.0)
            exp = float(trade_exports.get(key_full, 0.0) or 0.0)
            supply = float(getattr(n, 'Q0', 0.0) or 0.0)
            demand = max(supply + imp - exp, 0.0)
            n.D0 = demand
            try:
                if isinstance(n.meta, dict):
                    n.meta['import_hist'] = imp
                    n.meta['export_hist'] = exp
            except Exception:
                pass
            key = (n.country, n.commodity)
            prev = latest_hist.get(key)
            if prev is None or n.year >= prev[0]:
                latest_hist[key] = (n.year, demand)
    for n in nodes:
        if n.year > hist_end:
            if float(getattr(n, 'D0', 0.0) or 0.0) <= 0.0:
                prev = latest_hist.get((n.country, n.commodity))
                if prev is not None and prev[1] > 0.0:
                    n.D0 = prev[1]
    return latest_hist
# -------------- main pipeline --------------
def run_one_pipeline(paths: DataPaths,
                     pre_macc_e0: bool = False,
                     *,
                     scenario_id: str = 'BASE',
                     scenario_params: Optional[Dict[str, Any]] = None,
                     scenario_effects: Optional[list] = None,  # ✅ 新增：传入scenario effects而非计算好的params
                     solve: bool = False,
                     use_fao_modules: bool = False,
                     save_root: Optional[str] = None,
                     future_last_only: bool = False,
                     use_linear: bool = True) -> str:
    """Run one scenario (or MC draw) end-to-end and write results under {save_root}/{scenario_id}/"""
    import time
    start_time = time.time()
    print(f"\n{'='*100}")
    print(f"[START] 开始运行情景: {scenario_id}")
    print(f"{'='*100}\n")
    
    # 定义历史期结束年份（与build_model保持一致）
    hist_end_year = 2020
    
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
    
    # ✅ 注意：此时scenario_ctx仅用于land_carbon_price等简单参数
    # ruminant_intake_cap等需要D0数据的参数将在加载D0后重新计算
    scenario_ctx = dict(scenario_params) if isinstance(scenario_params, dict) else {}
    
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
    _log_step(f"active_years: {active_years}")
    _log_step(f"年份范围: {min(active_years)} - {max(active_years)}, 共 {len(active_years)} 年")
    _log_step("\u5df2\u6784\u5efa Universe \u4e0e\u8282\u70b9")
    luc_area_long = pd.DataFrame()
    luc_emis_detail = pd.DataFrame()
    luc_emis_summary = pd.DataFrame()
    coarse_transitions_df = pd.DataFrame()
    roundwood_supply_df = pd.DataFrame()
    base_area_df = pd.DataFrame()
    gfire_emission_frames: Dict[str, pd.DataFrame] = {}
    gfire_combined_df = pd.DataFrame()
    gfire_excel_path = os.path.join(get_input_base(), 'Emission', 'Emission_LULUCF_Historical_updated.xlsx')
    try:
        gfire_emission_frames = load_fixed_gfire_emissions(
            excel_path=gfire_excel_path,
            dict_v3_path=paths.dict_v3_path,
            active_years=active_years,
            country_by_m49=universe.country_by_m49,
            iso3_by_country=universe.iso3_by_country,
        )
        gfire_combined_df = gfire_emission_frames.get('combined', pd.DataFrame())
        _log_step(f"加载GFIRE固定排放: hist={len(gfire_emission_frames.get('historical', pd.DataFrame()))}, "
                  f"future={len(gfire_emission_frames.get('future_extension', pd.DataFrame()))}, "
                  f"combined={len(gfire_combined_df)}")
    except Exception as exc:
        gfire_emission_frames = {}
        gfire_combined_df = pd.DataFrame()
        _log_step(f"WARNING: Savanna/Peatlands fire emissions merge failed: {exc}")
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
    production_stats = load_production_statistics(paths, universe)
    # ✅ 所有默认DataFrame都包含M49_Country_Code列
    production_df = production_stats.get('production', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','production_t']))
    yield_hist_df = production_stats.get('yield', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_ha']))
    area_hist_df = production_stats.get('area', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','area_ha']))
    slaughter_hist_df = production_stats.get('slaughter', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','slaughter_head']))
    livestock_yield_df = production_stats.get('livestock_yield', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','yield_t_per_head']))
    stock_hist_df = production_stats.get('stock', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','stock_head']))
    fertilizer_eff_df = production_stats.get('fertilizer_efficiency', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','fertilizer_efficiency_kgN_per_ha']))
    fertilizer_amt_df = production_stats.get('fertilizer_amount', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','fertilizer_n_input_t']))
    feed_req_df = production_stats.get('feed_requirement', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_requirement_kg_per_head']))
    manure_ratio_df = production_stats.get('manure_management_ratio', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','manure_management_ratio']))
    # Temporary debug: if DEBUG_DUMP_INTERMEDIATE is set, write key DataFrames to workspace and exit early
    try:
        if os.environ.get('DEBUG_DUMP_INTERMEDIATE') == '1':
            debug_dir = Path(r"g:\我的云端硬盘\Work\Net-zero food\Code\bin\new\tools\debug_out")
            debug_dir.mkdir(parents=True, exist_ok=True)
            def _maybe_save(df, name):
                if isinstance(df, pd.DataFrame) and len(df):
                    p = debug_dir / f"{name}.csv"
                    # Save only 2020 and 2080 rows to keep files small
                    try:
                        df[df['year'].isin([2020,2080])].to_csv(p, index=False, encoding='utf-8-sig')
                    except Exception:
                        df.head(200).to_csv(p, index=False, encoding='utf-8-sig')
            _maybe_save(production_df, 'production')
            _maybe_save(stock_hist_df, 'stock')
            _maybe_save(slaughter_hist_df, 'slaughter')
            _maybe_save(livestock_yield_df, 'livestock_yield')
            _maybe_save(feed_req_df, 'feed_requirement')
            _maybe_save(manure_ratio_df, 'manure_management_ratio')
            return str(Path(get_results_base()) / scenario_id)
    except Exception as _e:
        pass  # 忽略dump失败
    # Debug: 打印livestock DataFrames的实际情况
    print(f"  slaughter_hist_df: {len(slaughter_hist_df)} rows")
    print(f"  livestock_yield_df: {len(livestock_yield_df)} rows")
    print(f"  stock_hist_df: {len(stock_hist_df)} rows")
    print(f"  feed_req_df: {len(feed_req_df)} rows")
    print(f"  manure_ratio_df: {len(manure_ratio_df)} rows")
    # ✅ 所有DataFrame都添加M49_Country_Code列
    feed_override_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t'])
    grass_requirement_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','grass_tdm','grass_area_need_ha'])
    grass_share_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','grass_feed_share'])
    feed_eff_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_efficiency_multiplier'])
    try:
        feed_outputs = build_feed_demand_from_stock(
            stock_df=stock_hist_df,
            universe=universe,
            maps=item_maps,
            paths=paths,
            years=universe.years,
            conversion_multiplier=scenario_ctx.get('dm_conversion_multiplier', {})
        )
        if isinstance(feed_outputs.crop_feed_demand, pd.DataFrame) and not feed_outputs.crop_feed_demand.empty:
            feed_override_df = feed_outputs.crop_feed_demand
        if isinstance(feed_outputs.grass_requirement, pd.DataFrame) and not feed_outputs.grass_requirement.empty:
            grass_requirement_df = feed_outputs.grass_requirement
        if isinstance(feed_outputs.species_dm_detail, pd.DataFrame) and not feed_outputs.species_dm_detail.empty:
            detail = feed_outputs.species_dm_detail.copy()
            if 'iso3' not in detail.columns or detail['iso3'].isna().all():
                detail['iso3'] = detail['country'].map(universe.iso3_by_country)
            # ✅ 添加M49_Country_Code列（从m49_code重命名）
            if 'm49_code' in detail.columns and 'M49_Country_Code' not in detail.columns:
                detail['M49_Country_Code'] = detail['m49_code']
            # ✅ groupby包含M49_Country_Code
            group_cols = ['country','iso3','year']
            if 'M49_Country_Code' in detail.columns:
                group_cols = ['M49_Country_Code'] + group_cols
            grp = detail.groupby(group_cols, as_index=False)[['grass_dm_kg','dm_total_kg']].sum()
            grp['grass_feed_share'] = np.where(grp['dm_total_kg'] > 0, grp['grass_dm_kg'] / grp['dm_total_kg'], np.nan)
            # ✅ 选择列时包含M49_Country_Code
            select_cols = ['country','iso3','year','grass_feed_share']
            if 'M49_Country_Code' in grp.columns:
                select_cols = ['M49_Country_Code'] + select_cols
            grass_share_df = grp[select_cols]
        conv_map = scenario_ctx.get('dm_conversion_multiplier', {})
        if conv_map:
            rows = []
            for (country, commodity, year), mult in conv_map.items():
                # ✅ 添加M49_Country_Code
                m49_code = universe.m49_by_country.get(country, '')
                rows.append({
                    'M49_Country_Code': m49_code,
                    'country': country,
                    'iso3': universe.iso3_by_country.get(country),
                    'year': int(year),
                    'commodity': commodity,
                    'feed_efficiency_multiplier': float(mult)
                })
            feed_eff_df = pd.DataFrame(rows)
    except Exception as exc:
        _log_step(f"Feed demand reconstruction failed, fallback to FAOSTAT feed data. Detail: {exc}")
    
    roundwood_supply_df = load_roundwood_supply(paths.trade_forestry_csv, universe, years=active_years)
    
    # ✅ 将 Roundwood 产量数据合并到 production_df，使其参与 DS 供需模型
    if isinstance(roundwood_supply_df, pd.DataFrame) and not roundwood_supply_df.empty:
        try:
            rw_prod = roundwood_supply_df.copy()
            rw_prod['commodity'] = 'Roundwood'
            rw_prod['production_t'] = rw_prod['roundwood_m3']  # m³ 作为产量单位
            # 添加 M49_Country_Code 列
            rw_prod['M49_Country_Code'] = rw_prod['country'].map(universe.m49_by_country)
            # 选择与 production_df 相同的列
            rw_cols = ['M49_Country_Code', 'country', 'iso3', 'year', 'commodity', 'production_t']
            rw_prod = rw_prod[[c for c in rw_cols if c in rw_prod.columns]]
            # 合并到 production_df
            production_df = pd.concat([production_df, rw_prod], ignore_index=True)
            _log_step(f"✅ 已将 Roundwood 产量数据 ({len(rw_prod)} 行) 合并到 production_df")
        except Exception as e:
            _log_step(f"⚠️ 合并 Roundwood 到 production_df 失败: {e}")
    
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
        if isinstance(roundwood_supply_df, pd.DataFrame) and not roundwood_supply_df.empty:
            for r in roundwood_supply_df.itertuples(index=False):
                try:
                    country = str(getattr(r, 'country'))
                    year = int(getattr(r, 'year'))
                    value = float(getattr(r, 'roundwood_m3') or 0.0)
                except Exception:
                    continue
                key = (country, 'Roundwood', year)
                prod_lookup[key] = value
                if year <= cfg.years_hist_end and value > 0:
                    prev = latest_hist_prod.get((country, 'Roundwood'))
                    if prev is None or year >= prev[0]:
                        latest_hist_prod[(country, 'Roundwood')] = (year, value)
        # 为所有未来年份节点填充Q0，从hist_end年复制基准值
        _log_step(f"latest_hist_prod中的记录数: {len(latest_hist_prod)}")
        
        # ✅ 诊断：检查production_df中的畜牧商品
        livestock_keywords = ['cattle', 'swine', 'pig', 'chicken', 'sheep', 'goat', 'buffalo', 'dairy', 'meat', 'milk', 'egg']
        if len(production_df) > 0:
            all_commodities = production_df['commodity'].unique()
            livestock_commodities_in_prod = [c for c in all_commodities if any(kw in str(c).lower() for kw in livestock_keywords)]
            _log_step(f"[诊断] production_df中的畜牧商品 ({len(livestock_commodities_in_prod)}个): {livestock_commodities_in_prod[:10]}")
            # 检查2020年数据
            y2020_prod = production_df[production_df['year'] == 2020]
            if len(y2020_prod) > 0:
                livestock_2020 = y2020_prod[y2020_prod['commodity'].apply(lambda c: any(kw in str(c).lower() for kw in livestock_keywords))]
                _log_step(f"[诊断] 2020年畜牧商品数据: {len(livestock_2020)} 行")
        
        # ✅ 诊断：检查universe.commodities中的畜牧商品
        livestock_in_universe = [c for c in universe.commodities if any(kw in str(c).lower() for kw in livestock_keywords)]
        _log_step(f"[诊断] universe.commodities中的畜牧商品 ({len(livestock_in_universe)}个): {livestock_in_universe}")
        
        nodes_filled_q0 = 0
        nodes_missing_q0_livestock = []  # 记录缺少Q0的畜牧节点
        for n in data.nodes:
            val = prod_lookup.get((n.country, n.commodity, n.year))
            if val is None and n.year > cfg.years_hist_end:
                hist = latest_hist_prod.get((n.country, n.commodity))
                if hist:
                    val = hist[1]
                    nodes_filled_q0 += 1
                else:
                    # 记录缺少Q0的畜牧节点
                    if any(kw in str(n.commodity).lower() for kw in livestock_keywords):
                        if len(nodes_missing_q0_livestock) < 10:  # 只记录前10个
                            nodes_missing_q0_livestock.append((n.country, n.commodity, n.year))
            if val is not None:
                n.Q0 = float(val)
        _log_step(f"为未来年份填充Q0的节点数: {nodes_filled_q0}")
        if nodes_missing_q0_livestock:
            _log_step(f"[诊断] ⚠️ 缺少Q0的畜牧节点示例: {nodes_missing_q0_livestock}")
    gce_act = build_gce_activity_tables(production_csv=paths.production_faostat_csv,
                                        fbs_csv=paths.fbs_csv,
                                        fertilizer_eff_xlsx=paths.fertilizer_efficiency_xlsx,
                                        universe=universe)
    data.gce_residues_df    = gce_act['residues_df']
    data.gce_burning_df     = gce_act['burning_df']
    data.gce_rice_df        = gce_act['rice_df']
    data.gce_fertilizers_df = gce_act['fertilizers_df']
    trade_imports, trade_exports = load_trade_import_export(paths, universe)
    # 4) Demand from FBS (Food/Feed/Seed) with mapped commodities
    fbs_comp = build_demand_components_from_fbs(
        paths.fbs_csv,
        universe,
        production_lookup=prod_lookup,
        latest_hist_prod=latest_hist_prod,
        feed_override_df=feed_override_df,
    )
    apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)
    latest_hist_demand = apply_trade_baseline_to_nodes(
        data.nodes,
        hist_end=cfg.years_hist_end,
        trade_imports=trade_imports,
        trade_exports=trade_exports,
    )
    # 记录弹性缺失情况，方便定位 Qs/Qd 可能被锁死的节点
    zero_supply_eps = []
    zero_demand_eps = []
    zero_supply_yield = []
    zero_supply_temp = []
    zero_pop_eps = []
    zero_inc_eps = []
    ignore_commodities = {
        'Forestland', 'Cropland organic soils', 'Grassland organic soils',
        'Peatlands fire', 'Savanna fires', 'De/Reforestation_crop area', 'De/Reforestation_pasture area'
    }
    cross_demo_country = "China"
    cross_demo_comm = "Chickens, layers"
    crossD_zero_count = crossS_zero_count = None
    for n in data.nodes:
        if n.year <= cfg.years_hist_end:
            continue
        if n.commodity in ignore_commodities:
            continue
        if n.country == cross_demo_country and n.commodity == cross_demo_comm:
            row_d = getattr(n, 'epsD_row', {}) or {}
            row_s = getattr(n, 'epsS_row', {}) or {}
            crossD_zero_count = sum(1 for v in row_d.values() if float(v or 0.0) == 0.0)
            crossS_zero_count = sum(1 for v in row_s.values() if float(v or 0.0) == 0.0)
        if float(getattr(n, 'eps_supply', 0.0) or 0.0) == 0.0:
            zero_supply_eps.append((n.country, n.commodity, n.year))
        if float(getattr(n, 'eps_supply_yield', 0.0) or 0.0) == 0.0:
            zero_supply_yield.append((n.country, n.commodity, n.year))
        if float(getattr(n, 'eps_supply_temp', 0.0) or 0.0) == 0.0:
            zero_supply_temp.append((n.country, n.commodity, n.year))
        if float(getattr(n, 'eps_demand', 0.0) or 0.0) == 0.0:
            zero_demand_eps.append((n.country, n.commodity, n.year))
        if float(getattr(n, 'eps_pop_demand', 0.0) or 0.0) == 0.0:
            zero_pop_eps.append((n.country, n.commodity, n.year))
        if float(getattr(n, 'eps_income_demand', 0.0) or 0.0) == 0.0:
            zero_inc_eps.append((n.country, n.commodity, n.year))
    def _log_zero(label: str, arr: list):
        if not arr:
            _log_step(f"{label}：0 个")
            return
        if len(arr) <= 100:
            _log_step(f"{label}：{len(arr)} 个，列表：{arr}")
        else:
            _log_step(f"{label}：{len(arr)} 个，示例前3: {arr[:3]}")
    _log_zero("供给自价弹性为0的未来节点", zero_supply_eps)
    _log_zero("供给产率弹性为0的未来节点", zero_supply_yield)
    _log_zero("供给温度弹性为0的未来节点", zero_supply_temp)
    _log_zero("需求自价弹性为0的未来节点", zero_demand_eps)
    _log_zero("需求人口弹性为0的未来节点", zero_pop_eps)
    _log_zero("需求收入弹性为0的未来节点", zero_inc_eps)
    if crossD_zero_count is not None:
        _log_zero("需求交叉弹性为0的类型数 (China, Chickens, layers)", [(cross_demo_country, cross_demo_comm, crossD_zero_count)])
    if crossS_zero_count is not None:
        _log_zero("供给交叉弹性为0的类型数 (China, Chickens, layers)", [(cross_demo_country, cross_demo_comm, crossS_zero_count)])
    # 对缺失需求的商品：基准年（<=hist_end）若 D0<=0 且 Q0>0，则用 Q0 作为自给需求，避免无需求导致全球供给>需求
    restored_demand = 0
    for n in data.nodes:
        if n.year <= cfg.years_hist_end:
            q0 = float(getattr(n, 'Q0', 0.0) or 0.0)
            d0 = float(getattr(n, 'D0', 0.0) or 0.0)
            if d0 <= 0.0 and q0 > 0.0:
                n.D0 = q0
                restored_demand += 1
    if restored_demand:
        _log_step(f"基准年缺失需求已用本国供应回填：{restored_demand} 个节点")
    _log_step("\u5df2\u5e94\u7528 FBS \u9700\u6c42\u62c6\u5206\uff0c\u5df2\u6839\u636e\u8d38\u6613\u6570\u636e\u66f4\u65b0\u57fa\u51c6\u9700\u6c42")
    
    # ========================================================================
    # ✅ 关键修复：在加载完所有基础数据（Q0, D0）后，重新计算所有情景参数
    # ========================================================================
    # 原问题：main()中提前调用apply_scenario_to_data使用空skeleton nodes，
    #        导致依赖Q0/D0的参数（如ruminant_intake_cap）计算错误
    # 解决方案：如果传入了scenario_effects，在此处使用完整数据重新计算
    # ========================================================================
    
    if scenario_effects is not None:
        _log_step("="*80)
        _log_step(f"🔄 使用完整数据重新计算情景参数: {scenario_id}")
        _log_step("="*80)
        
        # 使用已填充Q0/D0的nodes重新计算情景参数
        scenario_ctx = apply_scenario_to_data(scenario_effects, scenario_id, universe, data.nodes)
        
        _log_step(f"✓ 情景参数重新计算完成")
        _log_step(f"  - feed_reduction_by: {len(scenario_ctx.get('feed_reduction_by', {}))} 条")
        _log_step(f"  - ruminant_intake_cap: {len(scenario_ctx.get('ruminant_intake_cap', {}))} 条")
        _log_step(f"  - land_carbon_price_by_year: {len(scenario_ctx.get('land_carbon_price_by_year', {}))} 年份")
        _log_step(f"  - emission_factor_multiplier: {len(scenario_ctx.get('emission_factor_multiplier', {}))} 条")
        _log_step(f"  - fertilizer_rate_multiplier: {len(scenario_ctx.get('fertilizer_rate_multiplier', {}))} 条")
        _log_step(f"  - yield_multiplier: {len(scenario_ctx.get('yield_multiplier', {}))} 条")
        _log_step(f"  - manure_management_ratio_multiplier: {len(scenario_ctx.get('manure_management_ratio_multiplier', {}))} 条")
        
        # 检查ruminant_intake_cap是否有效（不全为0）
        rumi_cap = scenario_ctx.get('ruminant_intake_cap', {})
        if rumi_cap:
            values = list(rumi_cap.values())
            non_zero = sum(1 for v in values if v > 0)
            _log_step(f"  - ruminant_intake_cap 非零记录: {non_zero}/{len(values)} ({non_zero/len(values)*100:.1f}%)")
            if non_zero > 0:
                # 显示示例
                sample_items = [(k, v) for k, v in rumi_cap.items() if v > 0][:3]
                for (country, year), cap in sample_items:
                    _log_step(f"    例: {country} {year}年 cap={cap:.2f} t")
        
        _log_step("="*80)
    elif scenario_params and 'ruminant_intake_cap' in scenario_params:
        # 如果传入的是scenario_params（旧方式），检查并修复ruminant_intake_cap
        from S3_6_scenarios import RUMINANT_COMMS
        rumi_cap_values = list(scenario_params['ruminant_intake_cap'].values())
        if rumi_cap_values and all(v == 0.0 for v in rumi_cap_values):
            _log_step("⚠️ 检测到 ruminant_intake_cap 全为0，使用当前节点数据重新计算...")
            
            # 重新计算 2020 年的 ruminant 需求基准
            base_rumi = {}
            for n in data.nodes:
                if n.year == 2020 and n.commodity in RUMINANT_COMMS:
                    d0 = float(getattr(n, 'D0', 0.0) or 0.0)
                    base_rumi[n.country] = base_rumi.get(n.country, 0.0) + d0
            
            _log_step(f"   2020年ruminant需求基准: {len(base_rumi)} 个国家, 总计 {sum(base_rumi.values()):.2f} t")
            
            # 从 effects 中找到 ruminant_reduction 配置
            from S3_6_scenarios import load_scenarios, _linear_path_2020_2080
            effects = load_scenarios(paths.scenario_config_xlsx, universe, sheet='Scenario')
            rumi_effects = [e for e in effects if e.scenario_id == scenario_id and 
                           e.kind in ('ruminant_intake_decreasing_ratio', 'ruminant_intake_ratio', 'ruminant_reduction')]
            
            if rumi_effects and base_rumi:
                eff = rumi_effects[0]  # 取第一个配置
                timeline = _linear_path_2020_2080(eff.value_2080, eff.unit, universe.years)
                
                # 重新计算每个国家、每年的 cap
                new_rumi_cap = {}
                for country in base_rumi.keys():
                    for year, val in timeline.items():
                        if eff.unit == 'rate':
                            cap = (1.0 - val) * base_rumi[country]
                            new_rumi_cap[(country, year)] = max(0.0, cap)
                
                # 更新 scenario_ctx
                scenario_ctx['ruminant_intake_cap'] = new_rumi_cap
                _log_step(f"   ✓ 重新计算完成: {len(new_rumi_cap)} 个 (country, year) 组合")
                
                # 示例输出
                if new_rumi_cap:
                    sample_country = list(base_rumi.keys())[0]
                    _log_step(f"   示例 ({sample_country}):")
                    _log_step(f"      2020 基准需求: {base_rumi[sample_country]:.2f} t")
                    for y in [2030, 2050, 2080]:
                        if (sample_country, y) in new_rumi_cap:
                            cap = new_rumi_cap[(sample_country, y)]
                            reduction_pct = (1.0 - cap/base_rumi[sample_country]) * 100
                            _log_step(f"      {y} 年上限: {cap:.2f} t (减少 {reduction_pct:.1f}%)")
            else:
                if not rumi_effects:
                    _log_step("   ✗ 未找到 ruminant_reduction 情景配置")
                if not base_rumi:
                    _log_step("   ✗ 未找到 2020 年 ruminant 需求数据")
    
    latest_hist_supply: Dict[Tuple[str, str], float] = {k: v[1] for k, v in latest_hist_prod.items()}
    data.gle_livestock_stock_df = build_livestock_stock_from_env(paths.livestock_patterns_csv, universe)
    
    # ======= 土地覆盖数据加载 - 开关控制 =======
    # paths.use_simplified_land_cover: True=从简化Excel读取, False=从LUH2 NetCDF提取
    if paths.use_simplified_land_cover:
        _log_step("[LAND] 使用简化Excel文件读取土地覆盖数据")
        luh2_land_df = load_land_cover_from_excel(paths.land_cover_base_xlsx, universe, years=active_years)
    else:
        _log_step("[LAND] 从LUH2 NetCDF提取土地覆盖数据")
        luh2_land_df = load_luh2_land_cover(paths.luh2_states_nc, paths.luh2_mask_nc, universe, years=active_years)
    
    if isinstance(luh2_land_df, pd.DataFrame) and not luh2_land_df.empty:
        data.gos_areas_df = luh2_land_df
        base_area_df = luh2_land_df.pivot_table(
            index=['country', 'iso3', 'year'],
            columns='land_use',
            values='area_ha',
            aggfunc='sum'
        ).reset_index()
        base_area_df.columns.name = None
    else:
        data.gos_areas_df = build_gv_areas_from_inputs(paths.inputs_landuse_csv, universe)
        base_area_df = pd.DataFrame()
    fires_df = build_land_use_fires_timeseries(paths.emis_fires_csv, universe)
    # 4.1) Derive yield0 (t/ha) from FAOSTAT历史产率并赋值
    try:
        from S2_0_load_data import assign_yield0_to_nodes
        assign_yield0_to_nodes(data.nodes, yield_hist_df, hist_start=cfg.years_hist_start, hist_end=cfg.years_hist_end)
    except Exception:
        pass
    
    # 4.2) ✅ 计算并分配grassland系数（ha/ton）到livestock节点
    try:
        assign_grassland_coef_to_nodes(
            nodes=data.nodes,
            paths=paths,
            universe=universe,
            maps=item_maps,
            stock_df=data.gle_livestock_stock_df
        )
    except Exception as e:
        _log_step(f"⚠️ 分配grassland系数失败: {e}")
        import traceback
        traceback.print_exc()
    
    _log_step("\u5df2\u6784\u5efa\u751f\u4ea7/\u571f\u5730/\u80a5\u6599\u7b49\u6d3b\u52a8\u6570\u636e")
    # 5) Apply scenario parameters to data (feed reduction, ruminant cap, land carbon price, yield, etc.)
    scen = scenario_ctx
    # feed efficiency (rate) → shrink D0 by (1 - r) for matched (i,j,t)
    feed_red = scen.get('feed_reduction_by', {})
    if feed_red:
        idx = {(n.country, n.commodity, n.year): n for n in data.nodes}
        for (i,j,t), r in feed_red.items():
            n = idx.get((i,j,t))
            if n is not None and r is not None and r > 0:
                n.D0 = float(n.D0) * max(0.0, 1.0 - float(r))
    
    # yield multiplier (rate) → increase Q0 and yield by (1 + r) for matched (i,j,t)
    yield_mult = scen.get('yield_multiplier', {})
    if yield_mult:
        idx = {(n.country, n.commodity, n.year): n for n in data.nodes}
        for (i,j,t), mult in yield_mult.items():
            n = idx.get((i,j,t))
            if n is not None and mult is not None and mult != 1.0:
                n.Q0 = float(n.Q0) * float(mult)
                if hasattr(n, 'yield0') and n.yield0 is not None:
                    n.yield0 = float(n.yield0) * float(mult)
    
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
    
    # ❌ 注释掉solve之前的LUC计算：应该在优化后基于Qs重新计算
    # 原因：solve前使用Q0（初始产量），不包含优化后的产量和动态草地需求
    # 正确的LUC计算在solve后的line 2270-2561，基于Qs和动态草地需求
    # 
    # # Build production_df for LUC (使用生产量而非需求量，与production_summary一致)
    # # ✅ 修复：使用Q0（供应量）而非D0（需求量）计算面积，保持与production_summary.crop_area_ha一致
    # # ✅ 关键修复：只保留2020年（基准年）和未来年份，避免历史年份（2010-2019）参与逐期累积计算
    # crop_rows = [(n.country, n.year, n.commodity, float(n.Q0 if n.Q0 > 0 else n.D0)) for n in data.nodes if n.year >= 2020]
    # crop_demand_df = pd.DataFrame(crop_rows, columns=['country', 'year', 'commodity', 'demand_t'])
    # _log_step(f"LUC计算数据年份: {sorted(crop_demand_df['year'].unique())}")
    # luc_result = compute_luc_areas(
    #     demand_df=crop_demand_df,
    #     production_df=production_df,
    #     crop_yield_df=yield_hist_df,  # ✅ 修复：传入真实单产数据，避免使用默认值3.0 t/ha
    #     grass_requirement_df=grass_requirement_df,
    #     base_area_df=base_area_df,
    #     cfg=LUCConfig(land_carbon_price_per_tco2=float(land_cp_by_year.get(2080, 0.0)))
    # )
    # _log_step("已完成土地利用变化前置计算")
    _log_step("跳过solve前的LUC计算，将在优化后基于Qs和动态草地需求重新计算")
    
    # ✅ 初始化空的LUC结果（solve后会重新计算）
    luc_area_df = pd.DataFrame()
    luc_period_start_df = pd.DataFrame()
    luc_period_end_df = pd.DataFrame()
    luc_deltas_df = pd.DataFrame()
    
    # 处理期初面积 (period_start)
    luc_period_start_long = pd.DataFrame()
    if isinstance(luc_period_start_df, pd.DataFrame) and not luc_period_start_df.empty:
        luc_period_start_df = luc_period_start_df.copy()
        if 'M49_Country_Code' not in luc_period_start_df.columns and 'country' in luc_period_start_df.columns:
            luc_period_start_df['M49_Country_Code'] = luc_period_start_df['country'].map(universe.m49_by_country)
        luc_period_start_df['iso3'] = luc_period_start_df['country'].map(universe.iso3_by_country)
        luc_period_start_df = luc_period_start_df.dropna(subset=['iso3'])
    
    # 处理期末面积 (period_end)
    luc_period_end_long = pd.DataFrame()
    if isinstance(luc_period_end_df, pd.DataFrame) and not luc_period_end_df.empty:
        luc_period_end_df = luc_period_end_df.copy()
        if 'M49_Country_Code' not in luc_period_end_df.columns and 'country' in luc_period_end_df.columns:
            luc_period_end_df['M49_Country_Code'] = luc_period_end_df['country'].map(universe.m49_by_country)
        luc_period_end_df['iso3'] = luc_period_end_df['country'].map(universe.iso3_by_country)
        luc_period_end_df = luc_period_end_df.dropna(subset=['iso3'])
    
    # 处理逐期变化量 (deltas) - 用于生成 luc_land_area_change_by_period
    luc_change_long = pd.DataFrame()
    if isinstance(luc_deltas_df, pd.DataFrame) and not luc_deltas_df.empty:
        luc_deltas_df = luc_deltas_df.copy()
        if 'M49_Country_Code' not in luc_deltas_df.columns and 'country' in luc_deltas_df.columns:
            luc_deltas_df['M49_Country_Code'] = luc_deltas_df['country'].map(universe.m49_by_country)
        luc_deltas_df['iso3'] = luc_deltas_df['country'].map(universe.iso3_by_country)
        luc_deltas_df = luc_deltas_df.dropna(subset=['iso3'])
        # melt 变化量数据
        delta_cols = [c for c in luc_deltas_df.columns if c.startswith('d_') and c.endswith('_ha')]
        if delta_cols:
            id_vars = ['country', 'iso3', 'year']
            if 'M49_Country_Code' in luc_deltas_df.columns:
                id_vars = ['M49_Country_Code'] + id_vars
            luc_change_long = luc_deltas_df.melt(
                id_vars=id_vars,
                value_vars=delta_cols,
                var_name='land_use_change',
                value_name='change_ha'
            )
    
    # 兼容旧接口：luc_area_long 使用期末面积
    luc_area_long = pd.DataFrame()
    if isinstance(luc_area_df, pd.DataFrame) and not luc_area_df.empty:
        luc_area_df = luc_area_df.copy()
        # ✅ 添加M49_Country_Code（如果不存在）
        if 'M49_Country_Code' not in luc_area_df.columns and 'country' in luc_area_df.columns:
            luc_area_df['M49_Country_Code'] = luc_area_df['country'].map(universe.m49_by_country)
        luc_area_df['iso3'] = luc_area_df['country'].map(universe.iso3_by_country)
        luc_area_df = luc_area_df.dropna(subset=['iso3'])
        value_cols = [c for c in luc_area_df.columns if c.endswith('_ha') and c not in {'country', 'iso3', 'year', 'M49_Country_Code'}]
        rename_map = {c: c.replace('_ha', '_area_ha') for c in value_cols if not c.endswith('_area_ha')}
        luc_area_df = luc_area_df.rename(columns=rename_map)
        value_cols = [rename_map.get(c, c) for c in value_cols]
        # ✅ melt时保留M49_Country_Code
        id_vars = ['country', 'iso3', 'year']
        if 'M49_Country_Code' in luc_area_df.columns:
            id_vars = ['M49_Country_Code'] + id_vars
        luc_area_long = luc_area_df.melt(
            id_vars=id_vars,
            value_vars=value_cols,
            var_name='land_use',
            value_name='area_ha'
        )
    
    # 构建 coarse_transitions (森林⇔耕地/草地转换)
    coarse_transitions_df = _build_coarse_transitions_from_deltas(luc_deltas_df, universe.iso3_by_country)
    if not coarse_transitions_df.empty:
        coarse_transitions_df['iso3'] = coarse_transitions_df['iso3'].astype(str)
    roundwood_for_luc = None
    if isinstance(roundwood_supply_df, pd.DataFrame) and not roundwood_supply_df.empty:
        roundwood_for_luc = roundwood_supply_df[['iso3', 'year', 'roundwood_m3']].copy()
        roundwood_for_luc['iso3'] = roundwood_for_luc['iso3'].astype(str)
    _log_step(
        "LUC 输入摘要："
        f" area_rows={0 if luc_area_df is None else len(luc_area_df)}, "
        f"base_cols={list(base_area_df.columns) if isinstance(base_area_df, pd.DataFrame) and len(base_area_df.columns) <= 10 else 'N/A'}, "
        f"coarse_rows={0 if coarse_transitions_df is None else len(coarse_transitions_df)}, "
        f"coarse_cols={list(coarse_transitions_df.columns) if isinstance(coarse_transitions_df, pd.DataFrame) and len(coarse_transitions_df.columns) <= 10 else 'N/A'}, "
        f"roundwood_rows={0 if roundwood_for_luc is None else len(roundwood_for_luc)}"
    )
    try:
        luc_emis_summary, luc_transitions_hist = run_luc_bookkeeping(
            luh_file=paths.luh2_states_nc,
            mask_file=paths.luh2_mask_nc,
            years=active_years,
            param_excel=paths.luc_param_xlsx,
            coarse_transitions_df=coarse_transitions_df if not coarse_transitions_df.empty else None,
            roundwood_supply_df=roundwood_for_luc if roundwood_for_luc is not None and not roundwood_for_luc.empty else None,
            transitions_file=paths.luh2_transitions_nc if os.path.exists(paths.luh2_transitions_nc) else None,
        )
        _log_step("\u5df2\u751f\u6210 LUC \u8bb0\u8d26\u6392\u653e")
    except Exception as exc:
        detail = traceback.format_exc()
        _log_step(f"LUC \u8bb0\u8d26\u6a21\u5757\u6267\u884c\u5931\u8d25\uff1a{exc}\n{detail}")
        raise
    if isinstance(luc_emis_summary, pd.DataFrame) and not luc_emis_summary.empty:
        iso_to_country = {iso: cty for cty, iso in universe.iso3_by_country.items()}
        luc_emis_summary = luc_emis_summary.copy()
        luc_emis_summary['country'] = luc_emis_summary['iso3'].map(iso_to_country)
        luc_emis_summary = luc_emis_summary.dropna(subset=['country'])
        detail_frames: List[pd.DataFrame] = []
        for col, label in [
            ('F_veg_co2', 'LUC:Vegetation'),
            ('F_soil_co2', 'LUC:Soils'),
            ('F_hwp_co2', 'LUC:HWP'),
            ('F_inst_co2', 'LUC:Instant'),
        ]:
            if col not in luc_emis_summary.columns:
                continue
            tmp = luc_emis_summary[['country', 'iso3', 'year']].copy()
            tmp['commodity'] = 'ALL'
            tmp['process'] = label
            tmp['emissions_tco2e'] = luc_emis_summary[col]
            detail_frames.append(tmp)
        if 'total_co2' in luc_emis_summary.columns:
            total_df = luc_emis_summary[['country', 'iso3', 'year']].copy()
            total_df['commodity'] = 'ALL'
            total_df['process'] = 'LUC:Total'
            total_df['emissions_tco2e'] = luc_emis_summary['total_co2']
            detail_frames.append(total_df)
        luc_emis_detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    else:
        luc_emis_detail = pd.DataFrame(columns=['country', 'iso3', 'year', 'commodity', 'process', 'emissions_tco2e'])
    # 7) Emission intensities from *_fao.py (strict)
    if use_fao_modules:
        md = Path(get_input_base())
        # 注意: gle_emissions_module_fao.py 已重命名为 _legacy.py 并不再使用
        # 现在使用 gle_emissions_complete.py 进行完整的畜牧业排放计算
        module_paths = {
            k: str(md / k) for k in [
                'gce_emissions_module_fao.py',
                # 'gle_emissions_module_fao.py',  # 已弃用，改用 gle_emissions_complete.py
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
    # 7.2) LULUCF-only MACC: derive abatement share per country/process from land carbon price and reduce e0_p
    if pre_macc_e0:
        # NOTE: pre_macc_e0 and the S3.0 in-model land_carbon_price MACC path are mutually exclusive.
        # - pre_macc_e0=True: apply MACC to e0 before modeling (set S3.0 land_carbon_price to 0).
        # - pre_macc_e0=False: skip upfront MACC; if needed, set SolveOpt.land_carbon_price in S3.0.
        # Supply-side unit tax adders (tax_unit = price * e_land) can coexist without double counting.
        macc_df = _load_macc_df(os.path.join(get_input_base(), 'MACC-Global-US.pkl'))
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
    _log_step("已加载人口/营养/土地/收入驱动")

    # ========== 读取BASE情景数据（用于单位成本法计算）==========
    baseline_emissions = {}  # {(country, commodity, year, process): tCO2e}
    baseline_production = {}  # {(country, commodity, year): production_t}
    baseline_prices = {}  # {(commodity, year): price}
    
    is_base_scenario = (scenario_id.upper() == 'BASE')
    cost_method = CFG.get('cost_calculation_method', 'MACC')
    
    if not is_base_scenario and cost_method == 'unit_cost':
        _log_step("="*80)
        _log_step("[BASE数据加载] 当前为非BASE情景，需要加载BASE情景数据作为基准")
        _log_step("="*80)
        
        # 构建BASE结果路径
        base_results_path = Path(get_results_base('BASE'))
        base_emis_path = base_results_path / 'Emis' / 'emissions_summary.xlsx'
        base_prod_path = base_results_path / 'DS' / 'production_summary.csv'
        base_detail_path = base_results_path / 'DS' / 'detailed_node_summary.csv'
        base_market_path = base_results_path / 'DS' / 'market_summary.csv'
        
        try:
            # 1. 读取BASE排放数据
            if base_emis_path.exists():
                _log_step(f"  读取BASE排放数据: {base_emis_path}")
                # 使用By_Country_Process_Item sheet，包含最详细的数据
                base_emis_df = pd.read_excel(base_emis_path, sheet_name='By_Country_Process_Item')
                
                # 仅提取CO2eq数据（GHG='CO2eq'）作为基准
                base_emis_df = base_emis_df[base_emis_df['GHG'] == 'CO2eq'].copy()
                
                # 年份列格式: Y2010, Y2011, ..., Y2080
                year_cols = [col for col in base_emis_df.columns if col.startswith('Y')]
                
                # 构建baseline_emissions字典
                for _, row in base_emis_df.iterrows():
                    try:
                        country = str(row.get('Region_label_new', ''))
                        commodity = str(row.get('Item', ''))
                        process = str(row.get('Process', ''))
                        
                        if not country or not process:
                            continue
                        
                        # 遍历所有年份列
                        for year_col in year_cols:
                            try:
                                year = int(year_col[1:])  # 去掉'Y'前缀，如Y2020 -> 2020
                                value = float(row.get(year_col, 0))
                                
                                if value > 0:  # 只保存非零值
                                    # commodity可能为空（某些排放不分商品）
                                    baseline_emissions[(country, commodity, year, process)] = value
                            except Exception:
                                continue
                    except Exception as e:
                        continue
                
                _log_step(f"  成功加载 {len(baseline_emissions)} 条BASE排放记录（CO2eq）")
            else:
                _log_step(f"  ⚠️ BASE排放数据文件不存在: {base_emis_path}")
            
            # 2. 读取BASE产量数据
            if base_prod_path.exists():
                _log_step(f"  读取BASE产量数据: {base_prod_path}")
                base_prod_df = pd.read_csv(base_prod_path)
                
                for _, row in base_prod_df.iterrows():
                    try:
                        country = str(row.get('country', ''))
                        commodity = str(row.get('commodity', ''))
                        year = int(row.get('year', 0))
                        production = float(row.get('production_t', row.get('supply_t', 0)))
                        
                        if country and commodity and year > 0:
                            baseline_production[(country, commodity, year)] = production
                    except Exception as e:
                        continue
                
                _log_step(f"  成功加载 {len(baseline_production)} 条BASE产量记录")
            else:
                _log_step(f"  ⚠️ BASE产量数据文件不存在: {base_prod_path}")
            
            # 3. 读取BASE价格数据（从市场汇总或详细节点数据）
            if base_market_path.exists():
                _log_step(f"  读取BASE价格数据: {base_market_path}")
                base_market_df = pd.read_csv(base_market_path)
                
                for _, row in base_market_df.iterrows():
                    try:
                        commodity = str(row.get('commodity', ''))
                        year = int(row.get('year', 0))
                        price = float(row.get('price_global', row.get('price', 0)))
                        
                        if commodity and year > 0 and price > 0:
                            baseline_prices[(commodity, year)] = price
                    except Exception as e:
                        continue
                
                _log_step(f"  成功加载 {len(baseline_prices)} 条BASE价格记录")
            elif base_detail_path.exists():
                # 备选：从detailed_node_summary读取价格
                _log_step(f"  从详细节点数据读取BASE价格: {base_detail_path}")
                base_detail_df = pd.read_csv(base_detail_path)
                
                for _, row in base_detail_df.iterrows():
                    try:
                        commodity = str(row.get('commodity', ''))
                        year = int(row.get('year', 0))
                        price = float(row.get('price_global', row.get('price', 0)))
                        
                        if commodity and year > 0 and price > 0:
                            # 使用第一个遇到的价格（全球统一价格）
                            if (commodity, year) not in baseline_prices:
                                baseline_prices[(commodity, year)] = price
                    except Exception as e:
                        continue
                
                _log_step(f"  成功加载 {len(baseline_prices)} 条BASE价格记录")
            else:
                _log_step(f"  ⚠️ BASE价格数据文件不存在")
            
            _log_step("="*80)
            
        except Exception as e:
            _log_step(f"⚠️ 加载BASE情景数据时出错: {e}")
            import traceback
            traceback.print_exc()
            _log_step("将继续运行，但单位成本法可能无法正确计算")
    
    # ========== 加载成本数据（如果使用unit_cost方法）==========
    unit_cost_data = None
    process_cost_mapping = None
    
    if cost_method == 'unit_cost':
        _log_step("="*80)
        _log_step("成本计算方法: 【单位成本法】")
        _log_step("="*80)
        try:
            unit_cost_data, process_cost_mapping = load_unit_cost_data(
                cost_xlsx_path=paths.unit_cost_xlsx,
                dict_v3_path=paths.dict_v3_path
            )
            _log_step(f"成功加载单位成本数据: {len(unit_cost_data)} 条记录")
            _log_step(f"Process成本映射: {len(process_cost_mapping)} 个过程")
            
            # 显示部分映射示例
            if process_cost_mapping:
                sample_mappings = list(process_cost_mapping.items())[:5]
                _log_step(f"Process映射示例: {sample_mappings}")
            
        except Exception as e:
            _log_step(f"⚠️ 加载单位成本数据失败: {e}")
            import traceback
            traceback.print_exc()
            unit_cost_data = {}
            process_cost_mapping = {}
    else:
        _log_step("="*80)
        _log_step("成本计算方法: 【MACC曲线法】")
        _log_step("="*80)

    # ========== 模型选择：线性区域模型 vs 完整PWL模型 ==========
    use_linear = CFG.get('use_linear_model', False) and _LINEAR_MODEL_AVAILABLE
    
    # 初始化 gurobi_log_path（后续会更新）
    gurobi_log_path = log_dir / "gurobi.log"
    
    if use_linear:
        # === 使用简化线性区域模型（无PWL，快速求解）===
        _log_step("="*80)
        _log_step("D-S 模型配置: 【线性模型】(Linear Regional Model)")
        _log_step("  - 使用简化线性供需均衡优化")
        _log_step("  - 求解器: Gurobi (线性优化)")
        _log_step(f"  - Gurobi日志文件: {gurobi_log_path}")
        _log_step("="*80)
        
        # 过滤掉非商品项目（Item_Cat2 = 'no' 的土地类别、火灾等）
        # 这些项目没有真实的供需关系，不应参与价格出清
        non_tradable_items = {
            'Savanna_fires', 'Peatlands_fire', 'Forestland',
            'Cropland_organic_soils', 'Grassland_organic_soils',
            'De/Reforestation_crop_area', 'De/Reforestation_pasture_area',
            'De/Reforestation_crop area', 'De/Reforestation_pasture area',
            # 添加任何其他 Item_Cat2='no' 的项目
        }
        commodities = [c for c in universe.commodities if c not in non_tradable_items]
        _log_step(f"商品列表: {len(commodities)} 个 (已排除 {len(universe.commodities) - len(commodities)} 个非商品项目)")
        
        # 加载历史最大产量数据（可选）
        hist_max_production = None
        if CFG.get('enable_hist_max_anchor', False):
            hist_max_csv = Path(get_input_base()) / "Production_Trade" / "S0_19_historical_max_production.csv"
            if hist_max_csv.exists():
                try:
                    hist_max_production = load_historical_max_production(
                        csv_path=str(hist_max_csv),
                        dict_v3_path=str(paths.dict_v3_path),
                    )
                    _log_step(f"已加载历史最大产量锚定: {len(hist_max_production)} 个 (区域, 商品)")
                except Exception as e:
                    _log_step(f"警告: 无法加载历史最大产量数据: {e}")
            else:
                _log_step(f"警告: 历史最大产量文件不存在: {hist_max_csv}")
        
        # ✅ 关键修复：将国家级土地限制聚合到区域级（线性模型使用34个区域）
        # land_limits 是 {(country, year): land_area_ha}，需要转换为 {(region, year): sum_ha}
        regional_land_limits: Dict[Tuple[str, int], float] = {}
        if land_limits:
            from S3_0_ds_linear_regional import get_region
            _log_step(f"开始聚合土地限制: 从 {len(land_limits)} 个国家级记录聚合到区域级...")
            nan_count = 0
            for (country, year), area_ha in land_limits.items():
                # ✅ 过滤NaN值，避免Gurobi报错
                if area_ha is None or (isinstance(area_ha, float) and np.isnan(area_ha)):
                    nan_count += 1
                    continue
                region = get_region(country, dict_v3_path=str(paths.dict_v3_path))
                if region:
                    key = (region, year)
                    regional_land_limits[key] = regional_land_limits.get(key, 0.0) + float(area_ha)
            if nan_count > 0:
                _log_step(f"  ⚠️ 跳过 {nan_count} 个NaN值")
            _log_step(f"  ✅ 聚合完成: {len(regional_land_limits)} 个 (区域, 年份) 组合")
            _log_step(f"  📝 注意: 数据单位为 1000 ha，模型中会转换为 ha；所有年份统一使用2020年土地约束")
            # 打印几个区域的土地限制示例
            sample_regions = ['China', 'U.S.', 'India', 'Brazil', 'EUR-Continental']
            for sample_region in sample_regions:
                val_2020 = regional_land_limits.get((sample_region, 2020))
                if val_2020:
                    _log_step(f"    [{sample_region}, 2020]: {val_2020:,.0f} (1000 ha) = {val_2020 * 1000:,.0f} ha")
        
        # ✅ 关键修复：将国家级营养约束聚合到区域级
        regional_nut_rhs: Dict[Tuple[str, int], float] = {}
        if nut_rhs:
            from S3_0_ds_linear_regional import get_region
            _log_step(f"开始聚合营养约束: 从 {len(nut_rhs)} 个国家级记录聚合到区域级...")
            nan_count = 0
            for (country, year), rhs in nut_rhs.items():
                # ✅ 过滤NaN值
                if rhs is None or (isinstance(rhs, float) and np.isnan(rhs)):
                    nan_count += 1
                    continue
                region = get_region(country, dict_v3_path=str(paths.dict_v3_path))
                if region:
                    key = (region, year)
                    regional_nut_rhs[key] = regional_nut_rhs.get(key, 0.0) + float(rhs)
            if nan_count > 0:
                _log_step(f"  ⚠️ 跳过 {nan_count} 个NaN值")
            _log_step(f"  ✅ 聚合完成: {len(regional_nut_rhs)} 个 (区域, 年份) 组合")
        
        # ✅ 将草地面积聚合到区域级（用于土地约束）
        regional_grass_area: Dict[Tuple[str, int], float] = {}
        if isinstance(grass_requirement_df, pd.DataFrame) and not grass_requirement_df.empty:
            from S3_0_ds_linear_regional import get_region
            _log_step(f"开始聚合草地面积: 从 {len(grass_requirement_df)} 行国家级数据聚合到区域级...")
            for _, row in grass_requirement_df.iterrows():
                country = row.get('country', '')
                year = int(row.get('year', 0))
                grass_ha = float(row.get('grass_area_need_ha', 0) or 0)
                if not country or year == 0 or grass_ha <= 0:
                    continue
                region = get_region(country, dict_v3_path=str(paths.dict_v3_path))
                if region:
                    key = (region, year)
                    regional_grass_area[key] = regional_grass_area.get(key, 0.0) + grass_ha
            _log_step(f"  ✅ 聚合完成: {len(regional_grass_area)} 个 (区域, 年份) 组合")
            
            # ✅ [CRITICAL FIX] 用2020年草地面积初始化未来年（避免土地约束中草地=0）
            # 未来年草地会在solve后基于优化后的livestock重新计算，但solve时需要初始值
            if regional_grass_area:
                BASE_YEAR = 2020
                regions_with_2020 = {r for (r, y) in regional_grass_area.keys() if y == BASE_YEAR}
                future_years_needed = [y for y in active_years if y > BASE_YEAR]
                
                if regions_with_2020 and future_years_needed:
                    _log_step(f"  🔧 [土地约束修复] 用2020年草地面积初始化未来年（{len(future_years_needed)}个年份）...")
                    fill_count = 0
                    for region in regions_with_2020:
                        grass_2020 = regional_grass_area.get((region, BASE_YEAR), 0.0)
                        if grass_2020 > 0:
                            for future_year in future_years_needed:
                                if (region, future_year) not in regional_grass_area:
                                    regional_grass_area[(region, future_year)] = grass_2020
                                    fill_count += 1
                    _log_step(f"  ✅ 已用2020年数据填充 {fill_count} 个未来年草地面积")
                    _log_step(f"  📝 说明：这是初始值，solve后会基于优化后livestock重新计算")
            
            # 打印几个区域的草地面积示例
            for sample_region in ['China', 'U.S.', 'India', 'Brazil']:
                val_2020 = regional_grass_area.get((sample_region, 2020))
                val_2080 = regional_grass_area.get((sample_region, 2080))
                if val_2020:
                    _log_step(f"    [{sample_region}]: 2020={val_2020:,.0f} ha" + (f", 2080={val_2080:,.0f} ha (初始)" if val_2080 else ""))
        
        # ✅ 将森林面积聚合到区域级（用于土地约束）
        regional_forest_area: Dict[Tuple[str, int], float] = {}
        if isinstance(luh2_land_df, pd.DataFrame) and not luh2_land_df.empty:
            from S3_0_ds_linear_regional import get_region
            _log_step(f"开始聚合森林面积: 从土地覆盖数据聚合到区域级...")
            # 过滤出森林数据（land_use = 'forest' 或 'forest_area_ha'）
            forest_df = luh2_land_df[luh2_land_df['land_use'].isin(['forest', 'forest_area_ha'])].copy()
            if not forest_df.empty:
                for _, row in forest_df.iterrows():
                    country = row.get('country', '')
                    year = int(row.get('year', 0))
                    forest_ha = float(row.get('area_ha', 0) or 0)
                    if not country or year == 0 or forest_ha <= 0:
                        continue
                    region = get_region(country, dict_v3_path=str(paths.dict_v3_path))
                    if region:
                        key = (region, year)
                        regional_forest_area[key] = regional_forest_area.get(key, 0.0) + forest_ha
                _log_step(f"  ✅ 聚合完成: {len(regional_forest_area)} 个 (区域, 年份) 组合")
                # 打印几个区域的森林面积示例
                for sample_region in ['China', 'U.S.', 'India', 'Brazil']:
                    val_2020 = regional_forest_area.get((sample_region, 2020))
                    val_2080 = regional_forest_area.get((sample_region, 2080))
                    if val_2020:
                        _log_step(f"    [{sample_region}]: 2020={val_2020:,.0f} ha" + (f", 2080={val_2080:,.0f} ha" if val_2080 else ""))
            else:
                _log_step(f"  ⚠️ 土地覆盖数据中没有森林数据（land_use 值: {luh2_land_df['land_use'].unique().tolist()}）")
        
        # ========== 土地约束参数诊断（solve前最终检查）==========
        _log_step("\n" + "="*80)
        _log_step("[土地约束参数诊断] solve_linear_regional 调用前参数检查")
        _log_step("="*80)
        _log_step(f"  regional_land_limits: {len(regional_land_limits) if regional_land_limits else 0} 个 (区域, 年份)")
        _log_step(f"  regional_grass_area: {len(regional_grass_area) if regional_grass_area else 0} 个 (区域, 年份)")
        _log_step(f"  regional_forest_area: {len(regional_forest_area) if regional_forest_area else 0} 个 (区域, 年份)")
        
        # DEBUG: 列出regional_forest_area的前10个key
        if regional_forest_area:
            sample_keys = list(regional_forest_area.items())[:10]
            _log_step(f"  [DEBUG] regional_forest_area样例: {sample_keys}")
        
        # 打印几个关键区域的完整参数（用于验证）
        diagnostic_regions = ['China', 'U.S.', 'India', 'Brazil', 'EUR-Continental']
        diagnostic_years = [2020, 2080]
        _log_step("\n  [关键区域参数示例]")
        for reg in diagnostic_regions:
            for yr in diagnostic_years:
                if (reg, yr) in regional_land_limits or (reg, yr) in regional_grass_area or (reg, yr) in regional_forest_area:
                    land_limit = regional_land_limits.get((reg, yr), 'N/A')
                    grass = regional_grass_area.get((reg, yr), 0.0)
                    forest = regional_forest_area.get((reg, yr), 0.0)
                    
                    if land_limit != 'N/A':
                        land_limit_str = f"{float(land_limit)*1000:,.0f} ha"
                        available = float(land_limit) * 1000 - grass - forest
                        _log_step(f"    {reg} {yr}: land_limit={land_limit_str}, grass={grass:,.0f} ha, forest={forest:,.0f} ha")
                        _log_step(f"      => available_for_cropland={available:,.0f} ha")
        _log_step("="*80 + "\n")
        
        # 调用线性模型求解
        # ✅ 关键修复：传入与PWL模型一致的所有约束参数（使用区域级聚合数据）
        linear_result = solve_linear_regional(
            nodes=data.nodes,
            commodities=commodities,
            years=active_years,
            time_limit=300.0,
            dict_v3_path=str(paths.dict_v3_path),
            output_dir=str(log_dir),  # IIS 文件输出到 Log 目录
            gurobi_log_path=str(gurobi_log_path) if gurobi_log_path else None,
            # ===== 约束参数（使用区域级聚合数据）=====
            nutrition_rhs=regional_nut_rhs if regional_nut_rhs else None,  # ✅ 区域级营养约束
            nutrient_per_unit_by_comm=nutrient_map if len(nutrient_map) else None,
            land_area_limits=regional_land_limits if regional_land_limits else None,  # ✅ 区域级土地约束
            grass_area_by_region_year=regional_grass_area if regional_grass_area else None,  # ✅ 区域级草地面积
            forest_area_by_region_year=regional_forest_area if regional_forest_area else None,  # ✅ 区域级森林面积
            yield_t_per_ha_default=LUCConfig().yield_t_per_ha_default,
            grassland_method=CFG.get('grassland_method', 'dynamic'),  # ✅ 草地处理方式（dynamic或static）
            # ===== 人口与收入驱动 =====
            population_by_country_year=pop_map,
            income_mult_by_country_year=inc_mult,
            # ===== 排放与MACC =====
            macc_path=paths.macc_pkl,
            land_carbon_price_by_year=scenario_params.get('land_carbon_price_by_year', {}) if scenario_params else {},
            # ===== 增长约束 =====
            max_growth_rate_per_period=CFG.get('max_growth_rate') if CFG.get('enable_growth_constraints', False) else None,
            max_decline_rate_per_period=CFG.get('max_decline_rate') if CFG.get('enable_growth_constraints', False) and CFG.get('enable_decline_constraint', False) else None,
            hist_max_production=hist_max_production,
            # ===== 市场失衡限制 =====
            max_slack_rate=CFG.get('max_slack_rate'),  # 短缺/过剩上限
            # ===== 单位成本法参数 =====
            unit_cost_data=unit_cost_data,
            baseline_scenario_result={'baseline_emissions': baseline_emissions} if baseline_emissions else None,
            process_cost_mapping=process_cost_mapping,
            cost_calculation_method=cost_method,
        )
        
        solve_status = linear_result['status']
        _log_step(f"线性模型求解完成，状态={solve_status}")
        _log_step(f"[诊断] linear_result keys: {list(linear_result.keys())}")
        _log_step(f"[诊断] solve_status == 2? {solve_status == 2}")
        _log_step(f"[诊断] solve_status type: {type(solve_status)}")
        
        # 将区域结果分解回国家级，并应用到节点
        if solve_status == 2:  # OPTIMAL
            from S3_0_ds_linear_regional import disaggregate_to_countries, apply_results_to_nodes
            
            # 分解区域结果到国家级
            country_result = disaggregate_to_countries(data.nodes, linear_result)
            
            # 应用结果到节点（设置 n.P, n.Q, n.D）
            apply_results_to_nodes(data.nodes, country_result)
            
            _log_step(f"已将区域结果分解到 {len(country_result.get('Qs', {}))} 个国家级供需量")
        else:
            _log_step(f"⚠️ 线性模型未达到OPTIMAL状态(status={solve_status})，使用空结果集")
            country_result = {'status': solve_status, 'Qs': {}, 'Qd': {}, 'Pc': {}}
        
        # ✅ 修复：将线性模型的country_result转换为类Gurobi的var字典，供build_detailed_outputs使用
        # 创建虚拟的Gurobi变量对象，以保持与PWL模型的兼容性
        class VarWrapper:
            """包装字典值为类Gurobi变量对象"""
            def __init__(self, value):
                self.X = value
        
        var = {}
        if 'Pc' in country_result:
            var['Pc'] = {key: VarWrapper(val) for key, val in country_result['Pc'].items() if val is not None}
        if 'Qs' in country_result:
            var['Qs'] = {(i, j, t): VarWrapper(val) for (i, j, t), val in country_result['Qs'].items() if val is not None}
        if 'Qd' in country_result:
            var['Qd'] = {(i, j, t): VarWrapper(val) for (i, j, t), val in country_result['Qd'].items() if val is not None}
        if 'Eij' in country_result:
            var['E'] = {(i, j, t): VarWrapper(val) for (i, j, t), val in country_result['Eij'].items() if val is not None}
        if 'Cij' in country_result:
            var['C'] = {(i, j, t): VarWrapper(val) for (i, j, t), val in country_result['Cij'].items() if val is not None}
        
        _log_step(f"✓ 已从线性模型结果创建var字典: Pc={len(var.get('Pc', {}))}个, Qs={len(var.get('Qs', {}))}个, Qd={len(var.get('Qd', {}))}个")
        
        # ✅ 诊断：检查Qs变量中的畜牧商品和年份
        if var.get('Qs'):
            qs_keys = list(var['Qs'].keys())
            # 统计年份分布
            qs_years = {}
            qs_livestock_2080 = []
            livestock_keywords = ['cattle', 'swine', 'pig', 'chicken', 'sheep', 'goat', 'buffalo', 'dairy', 'meat', 'milk', 'egg']
            for (i, j, t) in qs_keys:
                qs_years[t] = qs_years.get(t, 0) + 1
                # 检查2080年的畜牧商品
                if t == 2080 and any(kw in str(j).lower() for kw in livestock_keywords):
                    val = var['Qs'][(i, j, t)].X
                    if val > 0 and len(qs_livestock_2080) < 10:
                        qs_livestock_2080.append((i, j, t, val))
            _log_step(f"[诊断] var['Qs']年份分布: {qs_years}")
            _log_step(f"[诊断] var['Qs']中2080年畜牧商品示例 ({len(qs_livestock_2080)}个): {qs_livestock_2080}")
            # 检查所有畜牧商品（不限年份）
            all_livestock_in_qs = set(j for (i, j, t) in qs_keys if any(kw in str(j).lower() for kw in livestock_keywords))
            _log_step(f"[诊断] var['Qs']中的所有畜牧商品 ({len(all_livestock_in_qs)}个): {list(all_livestock_in_qs)[:15]}")
        
        # 跳过后续的 Gurobi 模型构建/求解
        model = None
        # 线性模型不使用 Gurobi 日志，初始化为 None
        gurobi_log_path = None
        gurobi_log_actual = None
        
    else:
        # === 使用完整PWL模型 ===
        _log_step("="*80)
        _log_step("D-S 模型配置: 【PWL分段线性模型】(Piecewise Linear Model)")
        _log_step("  - 使用完整供需均衡 + 分段线性替代函数")
        _log_step("  - 求解器: Gurobi (非线性优化)")
        _log_step("="*80)
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
            max_growth_rate_per_period=CFG.get('max_growth_rate') if CFG.get('enable_growth_constraints', False) else None,
            max_decline_rate_per_period=CFG.get('max_decline_rate') if CFG.get('enable_growth_constraints', False) else None,
        )
    
    # ========== 仅在使用完整PWL模型时处理 Gurobi 日志和求解 ==========
    if not use_linear:
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
        
        # 求解完整PWL模型
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
                # === 快速诊断：输出模型统计和约束/变量边界信息 ===
                _log_step("=== 模型不可行/无界，开始快速诊断 ===")
                try:
                    _log_step(f"模型统计: 变量数={model.NumVars}, 约束数={model.NumConstrs}, 非零元素={model.NumNZs}")
                except Exception:
                    pass
                
                # 检查变量边界冲突（LB > UB）
                bound_conflicts = []
                try:
                    for v in model.getVars():
                        if v.LB > v.UB + 1e-6:
                            bound_conflicts.append(f"  {v.VarName}: LB={v.LB:.4g} > UB={v.UB:.4g}")
                        if len(bound_conflicts) >= 20:
                            break
                    if bound_conflicts:
                        _log_step(f"发现 {len(bound_conflicts)} 个变量边界冲突（LB>UB）:")
                        for bc in bound_conflicts[:10]:
                            _log_step(bc)
                        if len(bound_conflicts) > 10:
                            _log_step(f"  ...还有 {len(bound_conflicts)-10} 个")
                except Exception as e:
                    _log_step(f"检查变量边界时出错: {e}")
                
                # 检查约束边界（RHS过紧或冲突）
                tight_constrs = []
                try:
                    for c in model.getConstrs():
                        sense = c.Sense
                        rhs = c.RHS
                        # 检查等式约束的极端RHS
                        if sense == '=' and abs(rhs) > 1e15:
                            tight_constrs.append(f"  {c.ConstrName}: RHS={rhs:.4g} (极大值)")
                        if len(tight_constrs) >= 10:
                            break
                    if tight_constrs:
                        _log_step(f"发现可疑约束RHS:")
                        for tc in tight_constrs:
                            _log_step(tc)
                except Exception as e:
                    _log_step(f"检查约束RHS时出错: {e}")
                
                # 写出简化诊断文件（约束名列表）
                try:
                    diag_path = os.path.join(log_dir, f"model_{scenario_id}_diagnosis.txt")
                    with open(diag_path, 'w', encoding='utf-8') as f:
                        f.write(f"模型状态: {solve_status}\n")
                        f.write(f"变量数: {model.NumVars}\n")
                        f.write(f"约束数: {model.NumConstrs}\n\n")
                        f.write("=== 约束列表（前500个）===\n")
                        for i, c in enumerate(model.getConstrs()):
                            if i >= 500:
                                f.write(f"...还有 {model.NumConstrs - 500} 个约束\n")
                                break
                            f.write(f"{c.ConstrName}: {c.Sense} {c.RHS}\n")
                        if bound_conflicts:
                            f.write("\n=== 变量边界冲突 ===\n")
                            for bc in bound_conflicts:
                                f.write(bc + "\n")
                    _log_step(f"诊断信息已写入: {diag_path}")
                except Exception as e:
                    _log_step(f"写诊断文件失败: {e}")
                
                # === IIS 计算（可选）===
                iis_timeout = CFG.get('iis_timeout', 60)
                if iis_timeout <= 0:
                    _log_step("IIS计算已禁用（iis_timeout<=0），使用上述快速诊断信息排查问题")
                else:
                    try:
                        # 设置IIS计算超时，避免大模型卡住
                        _log_step(f"开始计算IIS（设置{iis_timeout}秒超时）...")
                        model.setParam('IISMethod', 0)  # 使用快速IIS方法
                        model.setParam('TimeLimit', iis_timeout)  # IIS计算超时
                        model.computeIIS()
                        iis_path = os.path.join(log_dir, f"model_{scenario_id}.ilp")
                        model.write(iis_path)
                        _log_step(f"IIS 文件已写入: {iis_path}")
                    except gp.GurobiError as ge:
                        _log_step(f"IIS计算超时或失败（GurobiError）：{ge}")
                        _log_step("请参考上述快速诊断信息或增大iis_timeout重试")
                    except Exception as exc:
                        _log_step(f"写 IIS 失败：{exc}")
        else:
            solve_status = getattr(model, 'Status', None)
    # === END of PWL model section ===
    
    # 统一处理 solve_status（线性模型已在前面设置）
    allowed_emis_status = {None,
                           2,  # OPTIMAL
                           gp.GRB.OPTIMAL,
                           gp.GRB.SUBOPTIMAL,
                           gp.GRB.TIME_LIMIT,
                           gp.GRB.INTERRUPTED,
                           gp.GRB.USER_OBJ_LIMIT}
    can_emit = (solve_status in allowed_emis_status)
    
    # ==========================================================================
    # 计算 Production value 类型的减排成本（单位成本法后处理）
    # ==========================================================================
    production_value_cost = 0.0
    if cost_method == 'unit_cost' and process_cost_mapping and can_emit:
        # 检查是否有 Production value 类型的过程
        prod_value_processes = [p for p, cost_name in process_cost_mapping.items() 
                                if cost_name == 'Production value']
        
        if prod_value_processes and baseline_production and baseline_prices:
            _log_step("[UNIT_COST] 开始计算 Production value 类型减排成本...")
            _log_step(f"[UNIT_COST] 涉及 {len(prod_value_processes)} 个过程: {prod_value_processes[:5]}...")
            
            try:
                from unit_cost_calculation import calculate_production_value_cost
                
                # 提取当前情景的产量（从模型结果）
                current_production = {}
                if var and 'Qs' in var:
                    for (country, commodity, year), qs_var in var['Qs'].items():
                        try:
                            prod_val = float(qs_var.X) if hasattr(qs_var, 'X') else float(qs_var)
                            current_production[(country, commodity, year)] = prod_val
                        except Exception:
                            continue
                
                # 提取市场价格（从模型结果或使用baseline价格）
                market_prices = {}
                if var and 'Pc' in var:
                    for (commodity, year), pc_var in var['Pc'].items():
                        try:
                            price_val = float(pc_var.X) if hasattr(pc_var, 'X') else float(pc_var)
                            market_prices[(commodity, year)] = price_val
                        except Exception:
                            continue
                
                # 如果模型没有价格变量，使用baseline价格
                if not market_prices and baseline_prices:
                    market_prices = baseline_prices.copy()
                    _log_step("[UNIT_COST] 使用BASE情景价格进行Production value成本计算")
                
                # 识别LUC相关的商品（De/Reforestation主要影响作物生产）
                # 简化：假设所有商品都可能受LUC影响，按产量变化计算
                luc_processes = set(prod_value_processes)
                
                # 计算Production value成本
                if current_production and market_prices:
                    prod_value_costs = calculate_production_value_cost(
                        current_production=current_production,
                        baseline_production=baseline_production,
                        market_prices=market_prices,
                        luc_processes=luc_processes
                    )
                    
                    # 汇总总成本
                    production_value_cost = sum(prod_value_costs.values())
                    _log_step(f"[UNIT_COST] Production value 总成本: {production_value_cost:.2e} USD")
                    
                    # 输出详细成本（前10个最大的）
                    if prod_value_costs:
                        sorted_costs = sorted(prod_value_costs.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)
                        _log_step("[UNIT_COST] 前10个最大的Production value成本:")
                        for i, (key, cost) in enumerate(sorted_costs[:10], 1):
                            country, commodity, year = key
                            _log_step(f"  {i}. {country}, {commodity}, {year}: {cost:.2e} USD")
                else:
                    _log_step("[UNIT_COST] ⚠️ 缺少产量或价格数据，无法计算Production value成本")
            
            except ImportError:
                _log_step("[UNIT_COST] ⚠️ 无法导入 unit_cost_calculation 模块")
            except Exception as e:
                _log_step(f"[UNIT_COST] ⚠️ 计算 Production value 成本时出错: {e}")
                import traceback
                _log_step(traceback.format_exc())
    emis_sum: Dict[str, pd.DataFrame] = {}
    if not can_emit:
        _log_step(f"\u6a21\u578b\u72b6\u6001={solve_status}\uff0c\u8df3\u8fc7 Emis \u6c47\u603b\uff0c\u4f46\u4fdd\u7559 DS \u8f93\u51fa")
        fao_results = None
    else:
        # 9) 初始化fao_results为空dict，后续由各个模块填充
        # （旧的fao.run_all()已废弃，使用新的run_crop_emissions和run_livestock_emissions）
        _log_step("初始化FAO排放结果容器")
        fao_results = {
            'GCE': {},      # Crop排放（dict格式，由run_crop_emissions填充）
            'GLE': [],      # Livestock排放（list格式，由run_livestock_emissions填充）
            'GSOIL': [],    # Soil排放
            'LUC': [],      # LUC排放
            'GFIRE': []     # Fire排放
        }
        _log_step(f"FAO 排放数据容器初始化完成，GCE类型: {type(fao_results['GCE'])}, 内容: {fao_results['GCE']}")
        
        # ========== ✅ 关键修复：基于优化后的 Qs 重新计算 LUC 面积变化 ==========
        # 原来的 luc_deltas_df 是基于模型求解前的 Q0 计算的，与 production_summary 不一致
        # 现在使用优化后的 var['Qs'] 重新计算，确保 LUC 面积与 production_summary.crop_area_ha 一致
        _log_step("基于优化后的产量(Qs)重新计算LUC面积变化...")
        try:
            Qs_dict = var.get('Qs', {}) if var else {}
            if Qs_dict:
                # 1. 从 Qs 提取优化后的产量
                # ✅ 关键修复：只保留2020年（基准年）和未来年份，避免历史年份参与逐期累积
                qs_rows = []
                all_qs_years = set()
                filtered_qs_years = set()
                for (country, commodity, year), qs_var in Qs_dict.items():
                    try:
                        year_int = int(year)
                        all_qs_years.add(year_int)
                        # 过滤：只保留>=2020的年份
                        if year_int < 2020:
                            continue
                        filtered_qs_years.add(year_int)
                        qs_val = float(qs_var.X) if hasattr(qs_var, 'X') else float(qs_var)
                        if qs_val > 0:
                            qs_rows.append({
                                'country': country,
                                'year': year_int,
                                'commodity': commodity,
                                'production_t': qs_val
                            })
                    except Exception:
                        continue
                
                # 诊断：显示年份过滤效果
                _log_step(f"[LUC年份过滤] Qs原始年份: {sorted(all_qs_years)}")
                _log_step(f"[LUC年份过滤] 过滤后年份: {sorted(filtered_qs_years)}")
                _log_step(f"[LUC年份过滤] 过滤掉: {sorted(all_qs_years - filtered_qs_years)}")
                
                if qs_rows:
                    qs_prod_df = pd.DataFrame(qs_rows)
                    _log_step(f"  从 Qs 提取产量数据: {len(qs_prod_df)} 行")
                    
                    # 2. 合并单产数据计算面积
                    if isinstance(yield_hist_df, pd.DataFrame) and not yield_hist_df.empty:
                        yield_tmp = yield_hist_df[['country', 'year', 'commodity', 'yield_t_per_ha']].copy()
                        qs_prod_df = qs_prod_df.merge(yield_tmp, on=['country', 'year', 'commodity'], how='left')
                        qs_prod_df['yield_t_per_ha'] = qs_prod_df['yield_t_per_ha'].replace(0, np.nan).fillna(3.0)
                        qs_prod_df['crop_area_ha'] = qs_prod_df['production_t'] / qs_prod_df['yield_t_per_ha']
                        
                        # 🔍 诊断：检查Qs和单产数据
                        _log_step("\n[LAND DIAG] 🔍 土地需求计算诊断:")
                        for yr in [2020, 2030, 2050, 2080]:
                            yr_data = qs_prod_df[qs_prod_df['year'] == yr]
                            if not yr_data.empty:
                                total_prod = yr_data['production_t'].sum()
                                avg_yield = yr_data['yield_t_per_ha'].mean()
                                total_area = yr_data['crop_area_ha'].sum()
                                _log_step(f"  {yr}年: 总产量={total_prod:,.0f} t, 平均单产={avg_yield:.2f} t/ha, 总面积={total_area:,.0f} ha")
                        
                        # 3. 按 country-year 汇总总耕地面积
                        crop_area_by_cy = qs_prod_df.groupby(['country', 'year'], as_index=False)['crop_area_ha'].sum()
                        crop_area_by_cy = crop_area_by_cy.rename(columns={'crop_area_ha': 'new_cropland_ha'})
                        _log_step(f"[DIAG] crop_area_by_cy: {len(crop_area_by_cy)} 行")
                        for yr in [2020, 2080]:
                            yr_data = crop_area_by_cy[crop_area_by_cy['year'] == yr]
                            _log_step(f"[DIAG]   {yr}年: {len(yr_data)} 个国家, 总面积={yr_data['new_cropland_ha'].sum():,.0f} ha")
                        
                        # 🔍 诊断：检查面积汇总
                        _log_step("\n[LAND DIAG] 🔍 全球耕地面积汇总:")
                        for yr in [2020, 2030, 2050, 2080]:
                            yr_area = crop_area_by_cy[crop_area_by_cy['year'] == yr]
                            if not yr_area.empty:
                                total = yr_area['new_cropland_ha'].sum()
                                _log_step(f"  {yr}年: {total:,.0f} ha")
                        
                        # 4. 获取基准年(2020)的面积 - ✅ 必须使用同一数据源（模型计算的面积）
                        base_year = 2020
                        base_areas = pd.DataFrame()
                        
                        # ✅ 关键修复：统一使用模型计算的cropland面积作为基准
                        # 理由：未来年份(2080)的new_cropland_ha是从Qs产量反推的
                        #       如果基准用LUH2数据，会导致不同定义混用，d_cropland会异常偏大
                        # 解决：2020年和2080年都使用crop_area_by_cy（从产量反推）
                        base_areas = crop_area_by_cy[crop_area_by_cy['year'] == base_year][['country', 'new_cropland_ha']].copy()
                        base_areas = base_areas.rename(columns={'new_cropland_ha': 'base_cropland_ha'})
                        _log_step(f"[LUC BASE] 使用模型计算的2020年cropland面积作为基准: {len(base_areas)} 个国家")
                        
                        # 诊断：打印2020年基准总面积
                        base_total = base_areas['base_cropland_ha'].sum()
                        _log_step(f"[LUC BASE] 2020年基准cropland总面积: {base_total:,.0f} ha")
                        
                        # 5. 计算 delta（相对基准年的变化）
                        future_areas = crop_area_by_cy[crop_area_by_cy['year'] > base_year].copy()
                        _log_step(f"[DIAG] future_areas初始化后: {len(future_areas)} 行, 年份={sorted(future_areas['year'].unique())}")
                        if not future_areas.empty and not base_areas.empty:
                            future_areas = future_areas.merge(base_areas, on='country', how='left')
                            future_areas['base_cropland_ha'] = future_areas['base_cropland_ha'].fillna(0)
                            future_areas['d_cropland_ha'] = future_areas['new_cropland_ha'] - future_areas['base_cropland_ha']
                            _log_step(f"[DIAG] future_areas merge后: {len(future_areas)} 行")
                            
                            # 🔍 诊断：检查耕地变化量
                            _log_step("\n[LAND DIAG] 🔍 全球耕地变化量 (d_cropland_ha):")
                            for yr in [2030, 2050, 2080]:
                                yr_data = future_areas[future_areas['year'] == yr]
                                if not yr_data.empty:
                                    d_crop_total = yr_data['d_cropland_ha'].sum()
                                    base_total = yr_data['base_cropland_ha'].sum()
                                    new_total = yr_data['new_cropland_ha'].sum()
                                    _log_step(f"  {yr}年: 基准={base_total:,.0f} ha, 未来={new_total:,.0f} ha, 变化={d_crop_total:,.0f} ha")
                            
                            # 🔍 检查是否为累积问题
                            yr_2030 = future_areas[future_areas['year'] == 2030]['d_cropland_ha'].sum()
                            yr_2050 = future_areas[future_areas['year'] == 2050]['d_cropland_ha'].sum()
                            yr_2080 = future_areas[future_areas['year'] == 2080]['d_cropland_ha'].sum()
                            if yr_2030 > 0 and yr_2050 > 0 and yr_2080 > 0:
                                ratio_50_30 = yr_2050 / yr_2030
                                ratio_80_50 = yr_2080 / yr_2050
                                _log_step(f"\n[LAND DIAG] 🔍 增长倍数检查:")
                                _log_step(f"  2050/2030 = {ratio_50_30:.2f}x")
                                _log_step(f"  2080/2050 = {ratio_80_50:.2f}x")
                                if ratio_80_50 > 3.0:
                                    _log_step(f"  ⚠️  警告: 2080年变化量相对2050年异常增大 ({ratio_80_50:.1f}x)！")
                                    _log_step(f"  可能原因: 需求累积、单产下降、或基准年数据有误")
                            
                            # 6. 添加 iso3 和 M49
                            future_areas['iso3'] = future_areas['country'].map(universe.iso3_by_country)
                            future_areas['M49_Country_Code'] = future_areas['country'].map(universe.m49_by_country)
                            _log_step(f"[DIAG] future_areas添加iso3/M49后: {len(future_areas)} 行")
                            
                            # ========== ✅ 关键修复：从优化后产量动态计算草地需求 ==========
                            # 原来使用静态的 grass_requirement_df（基于历史存栏计算）
                            # 现在从优化后的 Qs 重新计算：Qs → 存栏 → 草地需求
                            _log_step("[DYNAMIC GRASS] 开始从优化后产量动态计算草地需求...")
                            dynamic_grass_requirement_df = pd.DataFrame()
                            try:
                                # 6a. 筛选畜牧产品的优化产量
                                livestock_commodities = []
                                if hasattr(item_maps, 'feed_item_to_comm') and item_maps.feed_item_to_comm:
                                    livestock_commodities = list(item_maps.feed_item_to_comm.values())
                                _log_step(f"[DYNAMIC GRASS]   畜牧商品列表: {len(livestock_commodities)} 种")
                                
                                # 准备畜牧产量DataFrame（添加M49）
                                livestock_prod_df = qs_prod_df.copy()
                                livestock_prod_df['iso3'] = livestock_prod_df['country'].map(universe.iso3_by_country)
                                livestock_prod_df['M49_Country_Code'] = livestock_prod_df['country'].map(universe.m49_by_country)
                                
                                # 映射 commodity → FAOSTAT Item 名称（用于 gle 计算）
                                # Item_Emis → Item_Production_Map
                                emis_item_df = pd.read_excel(paths.dict_v3_path, sheet_name='Emis_item')
                                emis_to_faostat = {}
                                for _, row in emis_item_df.iterrows():
                                    item_emis = row.get('Item_Emis')
                                    item_prod = row.get('Item_Production_Map')
                                    if pd.notna(item_emis) and pd.notna(item_prod):
                                        emis_to_faostat[item_emis] = item_prod
                                
                                # 创建 Commodity 列（FAOSTAT格式）
                                livestock_prod_df['Commodity'] = livestock_prod_df['commodity'].map(emis_to_faostat)
                                livestock_prod_df['Commodity'] = livestock_prod_df['Commodity'].fillna(livestock_prod_df['commodity'])
                                
                                _log_step(f"[DYNAMIC GRASS]   准备畜牧产量数据: {len(livestock_prod_df)} 行")
                                
                                # 🔍 详细诊断：检查livestock_prod_df的数据质量
                                _log_step("\n" + "=" * 80)
                                _log_step("🔍 [GRASSLAND诊断] livestock_prod_df数据质量检查")
                                _log_step("=" * 80)
                                _log_step(f"数据形状: {livestock_prod_df.shape}")
                                _log_step(f"列名: {list(livestock_prod_df.columns)}")
                                _log_step(f"年份: {sorted(livestock_prod_df['year'].unique())}")
                                _log_step(f"国家数: {livestock_prod_df['country'].nunique()}")
                                _log_step(f"commodity数: {livestock_prod_df['commodity'].nunique()}")
                                _log_step(f"Commodity数: {livestock_prod_df['Commodity'].nunique()}")
                                _log_step(f"production_t非零行数: {(livestock_prod_df['production_t'] > 0).sum()}/{len(livestock_prod_df)}")
                                _log_step(f"production_t总和: {livestock_prod_df['production_t'].sum():,.0f} t")
                                _log_step(f"production_t范围: {livestock_prod_df['production_t'].min():.2e} ~ {livestock_prod_df['production_t'].max():.2e}")
                                
                                # 检查commodity映射情况
                                unmapped = livestock_prod_df[livestock_prod_df['Commodity'] == livestock_prod_df['commodity']]
                                if len(unmapped) > 0:
                                    _log_step(f"⚠️ 未成功映射的commodity: {unmapped['commodity'].unique()[:10].tolist()}")
                                
                                # 检查M49列
                                if 'M49_Country_Code' in livestock_prod_df.columns:
                                    m49_null = livestock_prod_df['M49_Country_Code'].isna().sum()
                                    _log_step(f"M49_Country_Code缺失: {m49_null}/{len(livestock_prod_df)}")
                                
                                # 样例数据（前10行）
                                _log_step("\n样例数据（前10行）:")
                                sample = livestock_prod_df[['country', 'year', 'commodity', 'Commodity', 'production_t']].head(10)
                                for idx, row in sample.iterrows():
                                    _log_step(f"  {row['country'][:20]:20s} | {row['year']} | {row['commodity'][:20]:20s} → {str(row['Commodity'])[:30]:30s} | {row['production_t']:>12,.0f} t")
                                
                                # 🔍 诊断：以美国为例追踪Qs（优化后产量）
                                us_qs = livestock_prod_df[livestock_prod_df['country'] == 'United States of America']
                                if not us_qs.empty:
                                    _log_step("\n" + "=" * 80)
                                    _log_step("🔍 [美国数据流] Step 1: 优化后畜牧产量 (Qs)")
                                    _log_step("=" * 80)
                                    _log_step(f"美国畜牧产量数据: {len(us_qs)} 行")
                                    _log_step(f"美国年份: {sorted(us_qs['year'].unique())}")
                                    _log_step(f"美国M49码: {us_qs['M49_Country_Code'].iloc[0] if 'M49_Country_Code' in us_qs.columns else 'N/A'}")
                                    _log_step(f"美国畜牧商品: {us_qs['commodity'].unique().tolist()}")
                                    us_summary = us_qs.groupby(['commodity', 'year'])['production_t'].sum().reset_index()
                                    _log_step("\n美国各商品产量汇总:")
                                    for _, row in us_summary.head(15).iterrows():
                                        _log_step(f"  {row['commodity']:20s}: {row['production_t']:>12,.0f} t ({row['year']}年)")
                                    _log_step("=" * 80 + "\n")
                                
                                # 6b. 调用 calculate_stock_from_optimized_production 计算存栏
                                if len(livestock_prod_df) > 0:
                                    optimized_stock_df = calculate_stock_from_optimized_production(
                                        production_df=livestock_prod_df,
                                        years=active_years,
                                        gle_params_path=os.path.join(get_src_base(), 'GLE_parameters.xlsx'),
                                        hist_production_path=paths.production_faostat_csv,
                                        hist_emissions_path=os.path.join(get_input_base(), 'Emission', 'Emissions_livestock_dairy_split.csv'),  # ✅ 使用已拆分dairy/non-dairy的文件
                                        hist_manure_stock_path=paths.livestock_patterns_csv,
                                        dict_v3_path=paths.dict_v3_path,
                                        universe=universe,
                                        hist_cutoff_year=2020
                                    )
                                    
                                    # 🔍 详细诊断：检查返回的存栏数据
                                    _log_step("\n" + "=" * 80)
                                    _log_step("🔍 [GRASSLAND诊断] calculate_stock返回结果检查")
                                    _log_step("=" * 80)
                                    _log_step(f"optimized_stock_df行数: {len(optimized_stock_df)}")
                                    if not optimized_stock_df.empty:
                                        _log_step(f"列名: {list(optimized_stock_df.columns)}")
                                        _log_step(f"年份: {sorted(optimized_stock_df['year'].unique()) if 'year' in optimized_stock_df.columns else 'N/A'}")
                                        _log_step(f"国家数: {optimized_stock_df['country'].nunique() if 'country' in optimized_stock_df.columns else 'N/A'}")
                                        _log_step(f"commodity数: {optimized_stock_df['commodity'].nunique() if 'commodity' in optimized_stock_df.columns else 'N/A'}")
                                        
                                        # 检查stock_head列
                                        if 'stock_head' in optimized_stock_df.columns:
                                            stock_sum = optimized_stock_df['stock_head'].sum()
                                            stock_nonzero = (optimized_stock_df['stock_head'] > 0).sum()
                                            _log_step(f"stock_head列存在: ✅")
                                            _log_step(f"  - 总存栏: {stock_sum:,.0f} head")
                                            _log_step(f"  - 非零行数: {stock_nonzero}/{len(optimized_stock_df)}")
                                            if stock_sum > 0:
                                                _log_step(f"  - 范围: {optimized_stock_df['stock_head'].min():.2e} ~ {optimized_stock_df['stock_head'].max():.2e}")
                                            else:
                                                _log_step(f"  - ❌ 所有stock_head都是0！")
                                        else:
                                            _log_step(f"❌ stock_head列不存在！实际列名: {list(optimized_stock_df.columns)}")
                                        
                                        # 美国数据样例
                                        us_stock = optimized_stock_df[optimized_stock_df['country'] == 'United States of America']
                                        if not us_stock.empty:
                                            _log_step(f"\n美国存栏数据样例（{len(us_stock)}行）:")
                                            for idx, row in us_stock.head(10).iterrows():
                                                comm = row.get('commodity', 'N/A')
                                                yr = row.get('year', 'N/A')
                                                stock = row.get('stock_head', 0)
                                                _log_step(f"  {comm:20s} | {yr} | {stock:>15,.0f} head")
                                        else:
                                            _log_step(f"⚠️ 美国没有存栏数据")
                                    else:
                                        _log_step(f"❌ optimized_stock_df为空！")
                                    _log_step("=" * 80 + "\n")
                                    
                                    _log_step(f"[DYNAMIC GRASS]   优化后存栏计算完成: {len(optimized_stock_df)} 行")
                                    
                                    # 6c. 从优化后存栏计算草地需求
                                    if not optimized_stock_df.empty:
                                        _log_step("\n" + "=" * 80)
                                        _log_step("🔍 [GRASSLAND诊断] 准备调用build_feed_demand_from_stock")
                                        _log_step("=" * 80)
                                        _log_step(f"输入参数:")
                                        _log_step(f"  - stock_df行数: {len(optimized_stock_df)}")
                                        _log_step(f"  - stock_df列名: {list(optimized_stock_df.columns) if not optimized_stock_df.empty else 'N/A'}")
                                        if not optimized_stock_df.empty and 'stock_head' in optimized_stock_df.columns:
                                            _log_step(f"  - stock_head>0的行数: {(optimized_stock_df['stock_head'] > 0).sum()}")
                                        _log_step(f"[DYNAMIC GRASS]   调用build_feed_demand_from_stock计算草地需求...")
                                        dynamic_feed_outputs = build_feed_demand_from_stock(
                                            stock_df=optimized_stock_df,
                                            universe=universe,
                                            maps=item_maps,
                                            paths=paths,
                                            years=active_years,
                                            conversion_multiplier=scenario_ctx.get('dm_conversion_multiplier', {})
                                        )
                                        # 🔍 详细诊断：检查build_feed_demand返回结果
                                        _log_step("\n" + "=" * 80)
                                        _log_step("🔍 [GRASSLAND诊断] build_feed_demand_from_stock返回结果检查")
                                        _log_step("=" * 80)
                                        _log_step(f"返回类型: {type(dynamic_feed_outputs)}")
                                        _log_step(f"是否有grass_requirement属性: {hasattr(dynamic_feed_outputs, 'grass_requirement')}")
                                        if hasattr(dynamic_feed_outputs, 'grass_requirement'):
                                            _log_step(f"grass_requirement类型: {type(dynamic_feed_outputs.grass_requirement)}")
                                        
                                        if hasattr(dynamic_feed_outputs, 'grass_requirement'):
                                            grass_req = dynamic_feed_outputs.grass_requirement
                                            if isinstance(grass_req, pd.DataFrame):
                                                _log_step(f"[DYNAMIC GRASS]   grass_requirement是DataFrame: 形状={grass_req.shape}, 为空={grass_req.empty}")
                                                if not grass_req.empty:
                                                    # ✅ 关键诊断：检查grass_area_need_ha列的有效性
                                                    if 'grass_area_need_ha' in grass_req.columns:
                                                        valid_area_count = grass_req['grass_area_need_ha'].notna().sum()
                                                        total_area = grass_req['grass_area_need_ha'].sum()
                                                        _log_step(f"[DYNAMIC GRASS]   grass_area_need_ha列: {valid_area_count}/{len(grass_req)} 行非NaN, 总计={total_area:,.0f} ha")
                                                        if valid_area_count == 0:
                                                            _log_step(f"[DYNAMIC GRASS]   ❌❌❌ 所有grass_area_need_ha都是NaN！数据不可用！")
                                                            _log_step(f"[DYNAMIC GRASS]   可能原因：pasture_yield数据缺失或为0")
                                                    else:
                                                        _log_step(f"[DYNAMIC GRASS]   ⚠️ grass_requirement缺少grass_area_need_ha列！")
                                                        _log_step(f"[DYNAMIC GRASS]   可用列: {list(grass_req.columns)}")
                                                    
                                                    dynamic_grass_requirement_df = grass_req
                                                    _log_step(f"[DYNAMIC GRASS]   ✅ 动态草地需求计算完成: {len(dynamic_grass_requirement_df)} 行")
                                                    # 打印对比
                                                    if 'year' in dynamic_grass_requirement_df.columns:
                                                        years_list = sorted(dynamic_grass_requirement_df['year'].unique())
                                                        _log_step(f"[DYNAMIC GRASS]   年份范围: {years_list}")
                                                        for yr in years_list[:3]:
                                                            yr_area = dynamic_grass_requirement_df[dynamic_grass_requirement_df['year'] == yr]['grass_area_need_ha'].sum()
                                                            _log_step(f"[DYNAMIC GRASS]     {yr}年全球草地需求: {yr_area:,.0f} ha")
                                                else:
                                                    _log_step(f"[DYNAMIC GRASS]   ⚠️ grass_requirement是空DataFrame")
                                            else:
                                                _log_step(f"[DYNAMIC GRASS]   ⚠️ grass_requirement不是DataFrame: {type(grass_req)}")
                                        else:
                                            _log_step(f"[DYNAMIC GRASS]   ⚠️ dynamic_feed_outputs没有grass_requirement属性")
                            except Exception as e:
                                _log_step(f"[DYNAMIC GRASS]   ⚠️ 动态草地计算失败: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # 使用动态计算的草地需求，如果有的话
                            effective_grass_df = dynamic_grass_requirement_df if not dynamic_grass_requirement_df.empty else grass_requirement_df
                            
                            # ✅ 诊断：确认effective_grass_df的年份范围
                            if isinstance(effective_grass_df, pd.DataFrame) and not effective_grass_df.empty and 'year' in effective_grass_df.columns:
                                eff_years = sorted(effective_grass_df['year'].unique())
                                _log_step(f"[DYNAMIC GRASS] effective_grass_df年份范围: {eff_years[:5]}...{eff_years[-5:] if len(eff_years) > 5 else ''}")
                                _log_step(f"[DYNAMIC GRASS] effective_grass_df总行数: {len(effective_grass_df)}")
                                _log_step(f"[DYNAMIC GRASS] 2080年是否存在: {'是' if 2080 in eff_years else '❌ 不存在'}")
                                if 2080 in eff_years:
                                    yr_2080 = effective_grass_df[effective_grass_df['year'] == 2080]
                                    _log_step(f"[DYNAMIC GRASS] 2080年数据: {len(yr_2080)} 行, 总草地需求={yr_2080['grass_area_need_ha'].sum():,.0f} ha")
                                else:
                                    _log_step(f"[DYNAMIC GRASS] ❌❌❌ 关键问题：effective_grass_df缺少2080年数据！")
                                    _log_step(f"[DYNAMIC GRASS] 这将导致未来年份草地变化计算错误（fallback到基准年，d_grass=0）")
                            else:
                                _log_step(f"[DYNAMIC GRASS] ⚠️ effective_grass_df为空或无year列")
                                _log_step(f"[DYNAMIC GRASS]   effective_grass_df类型: {type(effective_grass_df)}")
                                _log_step(f"[DYNAMIC GRASS]   dynamic_grass_requirement_df行数: {len(dynamic_grass_requirement_df) if isinstance(dynamic_grass_requirement_df, pd.DataFrame) else 'N/A'}")
                                _log_step(f"[DYNAMIC GRASS]   grass_requirement_df行数: {len(grass_requirement_df) if isinstance(grass_requirement_df, pd.DataFrame) else 'N/A'}")
                            # ========== END 动态草地需求计算 ==========
                            
                            # 7. ✅ 从草地需求计算草地变化
                            # ========== 关键修复：使用同一数据源对比基准年和目标年 ==========
                            # 错误做法：用LUH2的实际草地面积(33亿ha)对比草料需求面积(3600万ha)
                            # 正确做法：
                            # - 基准年数据：从静态grass_requirement_df（有2020年历史数据）
                            # - 目标年数据：优先从动态dynamic_grass_requirement_df（如有），否则用静态
                            # - 计算增量变化 = 目标年草料需求 - 基准年草料需求
                            future_areas['d_grassland_ha'] = 0.0  # 默认值
                            
                            # ✅ 始终从静态数据获取基准年草地需求（因为动态数据通常只有未来年份）
                            grass_need_base_by_country = {}
                            if isinstance(grass_requirement_df, pd.DataFrame) and not grass_requirement_df.empty:
                                grass_base_static = grass_requirement_df[grass_requirement_df['year'] == base_year].copy()
                                if not grass_base_static.empty:
                                    grass_need_base_by_country = grass_base_static.groupby('country')['grass_area_need_ha'].sum().to_dict()
                                    base_total = sum(grass_need_base_by_country.values())
                                    _log_step(f"[DEBUG PASTURE]   使用静态草料需求基准年({base_year})作为比较基准: {len(grass_need_base_by_country)} 个国家, 全球总计={base_total:,.0f} ha")
                                    
                                    # 🔍 诊断：对比2080年的草地需求
                                    if isinstance(effective_grass_df, pd.DataFrame) and not effective_grass_df.empty:
                                        grass_2080 = effective_grass_df[effective_grass_df['year'] == 2080]
                                        if not grass_2080.empty:
                                            grass_2080_total = grass_2080['grass_area_need_ha'].sum()
                                            _log_step(f"\n[LAND DIAG] 🔍 全球草地需求汇总:")
                                            _log_step(f"  2020年（基准）: {base_total:,.0f} ha")
                                            _log_step(f"  2080年（目标）: {grass_2080_total:,.0f} ha")
                                            _log_step(f"  变化量:         {grass_2080_total - base_total:,.0f} ha")
                                        else:
                                            _log_step(f"[DEBUG PASTURE]   ⚠️ effective_grass_df缺少2080年数据！")
                            
                            if isinstance(effective_grass_df, pd.DataFrame) and not effective_grass_df.empty:
                                _log_step("[DEBUG PASTURE] 从草地需求计算草地面积变化...")
                                is_dynamic = not dynamic_grass_requirement_df.empty
                                _log_step(f"[DEBUG PASTURE]   目标年数据使用{'动态(优化后)' if is_dynamic else '静态(历史)'}草地需求")
                                
                                if grass_need_base_by_country:
                                    # 逐行计算草地变化（使用草料需求的增量）
                                    for idx, row in future_areas.iterrows():
                                        country = row['country']
                                        yr = int(row['year'])
                                        base_grass_need = grass_need_base_by_country.get(country, 0.0)
                                        
                                        # 获取该年的草地需求（从动态或静态数据）
                                        yr_grass_rows = effective_grass_df[
                                            (effective_grass_df['country'] == country) & 
                                            (effective_grass_df['year'] == yr)
                                        ]
                                        # ✅ 如果该年没有数据，保持草地不变（d_grass=0），不要用base_grass_need代替
                                        if yr_grass_rows.empty:
                                            yr_grass_need = base_grass_need  # 没有预测数据时保持不变
                                        else:
                                            yr_grass_need = yr_grass_rows['grass_area_need_ha'].sum()
                                        
                                        # ✅ 正确计算：草料需求的增量变化
                                        d_grass = yr_grass_need - base_grass_need
                                        future_areas.loc[idx, 'd_grassland_ha'] = d_grass
                                        
                                        # DEBUG: 以美国为例打印
                                        # 🔍 诊断：以美国为例打印详细数据
                                        if 'United States' in str(country):
                                            _log_step(f"\n{'=' * 80}")
                                            _log_step(f"🔍 [美国数据流] Step 8: 草地变化量计算 ({yr}年)")
                                            _log_step(f"{'=' * 80}")
                                            _log_step(f"  基准年(2020)草地需求: {base_grass_need:>15,.0f} ha")
                                            _log_step(f"  目标年({yr})草地需求:  {yr_grass_need:>15,.0f} ha")
                                            _log_step(f"  草地变化量 (Δ):        {d_grass:>15,.0f} ha")
                                            if d_grass == 0:
                                                _log_step(f"  ⚠️ 草地变化为0！检查yr_grass_need是否正确查询到{yr}年数据")
                                            _log_step(f"{'=' * 80}\n")
                                else:
                                    _log_step("[DEBUG PASTURE]   ⚠️ 草地需求数据中没有基准年数据，草地变化保持为0")
                            else:
                                _log_step("[DEBUG PASTURE]   ⚠️ 草地需求数据为空，草地变化保持为0")
                            
                            # 8. 森林变化 = -(耕地变化 + 草地变化)（总面积不变）
                            future_areas['d_forest_ha'] = -(future_areas['d_cropland_ha'] + future_areas['d_grassland_ha'])
                            _log_step(f"[DIAG] future_areas计算d_forest后: {len(future_areas)} 行")
                            
                            # ✅ 诊断：检查草地变化数据
                            d_grass_total = future_areas['d_grassland_ha'].sum()
                            d_grass_nonzero = (future_areas['d_grassland_ha'] != 0).sum()
                            _log_step(f"[DEBUG PASTURE] 全球草地变化汇总: {d_grass_total:,.0f} ha, 非零国家数: {d_grass_nonzero}")
                            if d_grass_nonzero == 0:
                                _log_step(f"  ⚠️ 警告：所有国家的d_grassland_ha都为0！检查草地需求数据")
                                # 打印草地需求数据状态（使用effective_grass_df）
                                if isinstance(effective_grass_df, pd.DataFrame):
                                    _log_step(f"  目标年数据使用: {'dynamic' if not dynamic_grass_requirement_df.empty else 'static'} grass_requirement_df, {len(effective_grass_df)} 行")
                                    if not effective_grass_df.empty:
                                        sample = effective_grass_df.head(3)
                                        _log_step(f"  样本:\n{sample.to_string()}")
                                else:
                                    _log_step(f"  ⚠️ effective_grass_df 为空或None！")
                            
                            # DEBUG: 检查 U.S. 的具体数据
                            us_data = future_areas[future_areas['country'].str.contains('United States', case=False, na=False)]
                            if not us_data.empty:
                                for _, row in us_data.iterrows():
                                    _log_step(f"  [US DEBUG] {row['year']}: base_cropland={row.get('base_cropland_ha', 0):,.0f} ha, "
                                             f"new_cropland={row.get('new_cropland_ha', 0):,.0f} ha, "
                                             f"d_cropland={row.get('d_cropland_ha', 0):,.0f} ha")
                            
                            # 8. 更新 luc_deltas_df
                            luc_deltas_df = future_areas[['M49_Country_Code', 'country', 'iso3', 'year', 
                                                          'd_cropland_ha', 'd_grassland_ha', 'd_forest_ha']].copy()
                            
                            # 🔍 诊断：保存原始deltas数据
                            try:
                                debug_log_dir = log_dir  # 使用情景的Log目录
                                debug_deltas_path = debug_log_dir / "luc_deltas_debug.csv"
                                luc_deltas_df.to_csv(debug_deltas_path, index=False, encoding='utf-8-sig')
                                _log_step(f"  🔍 保存诊断数据: {debug_deltas_path}")
                            except Exception as e:
                                _log_step(f"  ⚠️ 保存deltas诊断数据失败: {e}")
                            
                            # DEBUG: 打印重新计算的 LUC deltas
                            for yr in sorted(luc_deltas_df['year'].unique()):
                                yr_data = luc_deltas_df[luc_deltas_df['year'] == yr]
                                d_crop_sum = yr_data['d_cropland_ha'].sum()
                                d_grass_sum = yr_data['d_grassland_ha'].sum()
                                d_forest_sum = yr_data['d_forest_ha'].sum()
                                print(f"[LUC RECALC] {yr}年全球: d_cropland={d_crop_sum:,.0f} ha, d_grassland={d_grass_sum:,.0f} ha, d_forest={d_forest_sum:,.0f} ha")
                            
                            # ✅ 关键修复：立即重新构建 coarse_transitions_df（确保包含grassland数据）
                            coarse_transitions_df = _build_coarse_transitions_from_deltas(luc_deltas_df, universe.iso3_by_country)
                            if not coarse_transitions_df.empty:
                                coarse_transitions_df['iso3'] = coarse_transitions_df['iso3'].astype(str)
                                _log_step(f"  ✅ 重新构建 coarse_transitions_df: {len(coarse_transitions_df)} 行")
                                
                                # 🔍 诊断：保存约束后的transitions数据
                                try:
                                    debug_coarse_path = os.path.join(log_dir, "coarse_transitions_debug.csv")
                                    coarse_transitions_df.to_csv(debug_coarse_path, index=False, encoding='utf-8-sig')
                                    _log_step(f"  🔍 保存诊断数据: {debug_coarse_path}")
                                except Exception as e:
                                    _log_step(f"  ⚠️ 保存coarse诊断数据失败: {e}")
                                
                                # 诊断：检查 pasture 转换数据
                                for yr in [2030, 2050, 2080]:
                                    yr_data = coarse_transitions_df[coarse_transitions_df['year'] == yr]
                                    if not yr_data.empty:
                                        f_to_p = yr_data['forest_to_pasture'].sum()
                                        p_to_f = yr_data['pasture_to_forest'].sum()
                                        _log_step(f"  [PASTURE CHECK] {yr}年: forest_to_pasture={f_to_p:,.0f}, pasture_to_forest={p_to_f:,.0f} ha")
                            else:
                                _log_step(f"  ⚠️ coarse_transitions_df 重构后为空！")
                            
                            # ✅ 同步更新 luc_period_start_df 和 luc_period_end_df（用于输出csv）
                            # 期初面积 = 2020年基准面积
                            # 期末面积 = 2080年新面积
                            if not base_areas.empty:
                                # ✅ 诊断：检查base_area_df的状态
                                if isinstance(base_area_df, pd.DataFrame) and not base_area_df.empty:
                                    _log_step(f"[BASE AREA] base_area_df状态: {len(base_area_df)} 行, 列={list(base_area_df.columns)}")
                                    base_2020 = base_area_df[base_area_df['year'] == 2020]
                                    if not base_2020.empty:
                                        _log_step(f"[BASE AREA] 2020年基准数据: {len(base_2020)} 个国家")
                                        # 检查森林和草地列
                                        forest_cols_found = [c for c in base_2020.columns if 'forest' in c.lower()]
                                        grass_cols_found = [c for c in base_2020.columns if 'grass' in c.lower() or 'pasture' in c.lower()]
                                        _log_step(f"[BASE AREA] 森林相关列: {forest_cols_found}")
                                        _log_step(f"[BASE AREA] 草地相关列: {grass_cols_found}")
                                        # 打印样本值
                                        if forest_cols_found:
                                            forest_col = forest_cols_found[0]
                                            total_forest = base_2020[forest_col].sum()
                                            _log_step(f"[BASE AREA] 2020年全球森林面积(从{forest_col}): {total_forest:,.0f} ha")
                                    else:
                                        _log_step(f"[BASE AREA] ❌ base_area_df中没有2020年数据！")
                                else:
                                    _log_step(f"[BASE AREA] ❌ base_area_df为空或不存在！")
                                
                                # 构建期初面积（2020基准）
                                _log_step(f"[PERIOD BUILD] future_areas行数: {len(future_areas)}, 年份: {sorted(future_areas['year'].unique())}")
                                _log_step(f"[PERIOD BUILD] future_areas国家数: {future_areas['country'].nunique()}")
                                period_start_rows = []
                                period_end_rows = []
                                for _, row in future_areas.iterrows():
                                    ctry = row['country']
                                    yr = row['year']
                                    base_crop = row.get('base_cropland_ha', 0)
                                    new_crop = row.get('new_cropland_ha', 0)
                                    d_grass = row.get('d_grassland_ha', 0)  # ✅ 使用计算好的草地变化
                                    
                                    # ✅ 诊断：如果d_grass=0，打印警告
                                    if d_grass == 0 and 'United States' in ctry:
                                        _log_step(f"  [WARN] {ctry} {yr}年 d_grassland_ha=0，检查future_areas['d_grassland_ha']是否有值")
                                        _log_step(f"    row keys: {list(row.keys())}")
                                    
                                    # 从 base_area_df 获取2020年的真实森林和草地面积
                                    base_forest = 0.0
                                    base_grass = 0.0
                                    if isinstance(base_area_df, pd.DataFrame) and not base_area_df.empty:
                                        ctry_base = base_area_df[(base_area_df['country'] == ctry) & (base_area_df['year'] == base_year)]
                                        if not ctry_base.empty:
                                            # ✅ 修复：pivot后列名是 forest/grassland，不是 forest_ha/grassland_ha
                                            if 'forest_ha' in ctry_base.columns:
                                                base_forest = float(ctry_base['forest_ha'].iloc[0])
                                            elif 'forest_area_ha' in ctry_base.columns:
                                                base_forest = float(ctry_base['forest_area_ha'].iloc[0])
                                            elif 'forest' in ctry_base.columns:
                                                base_forest = float(ctry_base['forest'].iloc[0])
                                            if 'grassland_ha' in ctry_base.columns:
                                                base_grass = float(ctry_base['grassland_ha'].iloc[0])
                                            elif 'pasture_area_ha' in ctry_base.columns:
                                                base_grass = float(ctry_base['pasture_area_ha'].iloc[0])
                                            elif 'grassland' in ctry_base.columns:
                                                base_grass = float(ctry_base['grassland'].iloc[0])
                                    
                                    # ✅ 诊断：打印base值
                                    if 'United States' in ctry:
                                        _log_step(f"  [DEBUG] {ctry} {yr}年: base_forest={base_forest:,.0f}, base_grass={base_grass:,.0f}, d_grass={d_grass:,.0f}")
                                    
                                    # ✅ 计算未来年份的绝对面积（用于LUC模块）
                                    # 耕地：new_crop（来自产量反推）
                                    # 草地：base_grass + d_grass（基准 + 变化）
                                    # 森林：反推（总面积守恒）
                                    new_grass = base_grass + d_grass
                                    new_forest = base_forest + base_crop - new_crop + base_grass - new_grass  # 总面积守恒
                                    
                                    # ✅ 确保森林面积不为负
                                    if new_forest < 0:
                                        _log_step(f"  [WARN] {ctry} {yr}年森林面积计算为负: {new_forest:,.0f}, 限制为0")
                                        _log_step(f"    base_forest={base_forest:,.0f}, base_crop={base_crop:,.0f}, new_crop={new_crop:,.0f}, base_grass={base_grass:,.0f}, new_grass={new_grass:,.0f}")
                                        new_forest = 0.0
                                    
                                    # period_start_rows: 2020基准面积（对所有未来年份都用2020作为基准）
                                    period_start_rows.append({
                                        'M49_Country_Code': row.get('M49_Country_Code'),
                                        'country': ctry,
                                        'iso3': row.get('iso3'),
                                        'year': yr,
                                        'cropland_ha': base_crop,
                                        'forest_ha': base_forest,
                                        'grassland_ha': base_grass,
                                    })
                                    
                                    # period_end_rows: 未来年份绝对面积
                                    period_end_rows.append({
                                        'M49_Country_Code': row.get('M49_Country_Code'),
                                        'country': ctry,
                                        'iso3': row.get('iso3'),
                                        'year': yr,
                                        'cropland_ha': new_crop,      # ✅ 绝对面积
                                        'forest_ha': new_forest,      # ✅ 绝对面积
                                        'grassland_ha': new_grass,    # ✅ 绝对面积
                                    })
                                
                                # ✅ 修复：period_end_rows现在存储delta，不需要重新计算森林/草地
                                # 因为d_grass直接来自future_areas['d_grassland_ha']（模型计算）
                                # d_forest = -(d_crop + d_grass)，假设总面积守恒
                                
                                # 转换为DataFrame
                                if period_start_rows and period_end_rows:
                                    new_period_start = pd.DataFrame(period_start_rows)
                                    new_period_end = pd.DataFrame(period_end_rows)
                                    
                                    # ✅ 诊断：检查未来年份数据
                                    future_years_in_start = sorted([y for y in new_period_start['year'].unique() if y > 2020])
                                    future_years_in_end = sorted([y for y in new_period_end['year'].unique() if y > 2020])
                                    _log_step(f"  [PERIOD CHECK] 新生成的期初数据年份: {future_years_in_start}")
                                    _log_step(f"  [PERIOD CHECK] 新生成的期末数据年份: {future_years_in_end}")
                                    
                                    # 合并历史数据（如果有）
                                    if isinstance(luc_period_start_df, pd.DataFrame) and not luc_period_start_df.empty:
                                        hist_start = luc_period_start_df[luc_period_start_df['year'] <= base_year].copy()
                                        luc_period_start_df = pd.concat([hist_start, new_period_start], ignore_index=True)
                                    else:
                                        luc_period_start_df = new_period_start.copy()
                                    
                                    if isinstance(luc_period_end_df, pd.DataFrame) and not luc_period_end_df.empty:
                                        hist_end = luc_period_end_df[luc_period_end_df['year'] <= base_year].copy()
                                        luc_period_end_df = pd.concat([hist_end, new_period_end], ignore_index=True)
                                    else:
                                        luc_period_end_df = new_period_end.copy()
                                    
                                    # ✅ 验证：确认未来年份在最终DataFrame中
                                    final_years_start = sorted(luc_period_start_df['year'].unique())
                                    final_years_end = sorted(luc_period_end_df['year'].unique())
                                    _log_step(f"  ✅ 最终 luc_period_start_df: {len(luc_period_start_df)} 行, 年份: {final_years_start}")
                                    _log_step(f"  ✅ 最终 luc_period_end_df: {len(luc_period_end_df)} 行, 年份: {final_years_end}")
                                    
                                    # ✅ 诊断：打印2080年绝对面积样本
                                    if 2080 in luc_period_end_df['year'].values:
                                        yr_2080_start = luc_period_start_df[luc_period_start_df['year'] == 2080]
                                        yr_2080_end = luc_period_end_df[luc_period_end_df['year'] == 2080]
                                        if not yr_2080_start.empty and not yr_2080_end.empty:
                                            crop_start = yr_2080_start['cropland_ha'].sum()
                                            crop_end = yr_2080_end['cropland_ha'].sum()
                                            grass_start = yr_2080_start['grassland_ha'].sum()
                                            grass_end = yr_2080_end['grassland_ha'].sum()
                                            _log_step(f"  [AREA CHECK] 2080年: 耕地 {crop_start:,.0f}→{crop_end:,.0f} (Δ={crop_end-crop_start:,.0f}), 草地 {grass_start:,.0f}→{grass_end:,.0f} (Δ={grass_end-grass_start:,.0f}) ha")
                                else:
                                    _log_step(f"  ⚠️ period_rows 为空，未能生成期初/期末数据")
                            
                            # ✅ 关键：同步更新 luc_change_long（用于输出 luc_land_area_change_by_period.csv）
                            delta_cols = [c for c in luc_deltas_df.columns if c.startswith('d_') and c.endswith('_ha')]
                            if delta_cols:
                                id_vars = ['country', 'iso3', 'year']
                                if 'M49_Country_Code' in luc_deltas_df.columns:
                                    id_vars = ['M49_Country_Code'] + id_vars
                                luc_change_long = luc_deltas_df.melt(
                                    id_vars=id_vars,
                                    value_vars=delta_cols,
                                    var_name='land_use_change',
                                    value_name='change_ha'
                                )
                                _log_step(f"  ✅ 同步更新 luc_change_long: {len(luc_change_long)} 行")
                            
                            _log_step(f"  ✅ 基于Qs重新计算LUC deltas完成: {len(luc_deltas_df)} 行")
                        else:
                            _log_step("  ⚠️ 无法计算LUC deltas: 缺少未来年份数据或基准年数据")
                    else:
                        _log_step("  ⚠️ 无法计算LUC deltas: yield_hist_df 为空")
            else:
                _log_step("  ⚠️ Qs 为空，保留原有 luc_deltas_df")
        except Exception as e:
            _log_step(f"  ⚠️ 基于Qs重新计算LUC失败: {e}")
            import traceback
            traceback.print_exc()
        # ========== END 重新计算 LUC ==========
        
        # 9.5) 读取历史LUC排放并计算未来LUC排放
        _log_step("开始处理LUC排放（历史+未来）")
        try:
            luc_hist_path = os.path.join(get_input_base(), 'Emission', 'Emission_LULUCF_Historical_updated.xlsx')
            # 读取历史LUC排放
            luc_hist_df = pd.DataFrame()
            if os.path.exists(luc_hist_path):
                try:
                    luc_hist_df = read_luc_historical_emissions(
                        hist_file=luc_hist_path,
                        dict_v3_path=paths.dict_v3_path,
                        years=active_years
                    )
                    _log_step(f"读取历史LUC排放: {len(luc_hist_df)} 行")
                except Exception as e:
                    _log_step(f"[WARN] 读取历史LUC排放失败: {e}")
            # 计算未来LUC排放（>2020年份）
            luc_future_dict = {}
            try:
                # ✅ 🔥 CRITICAL FIX: 准备luc_area_df供run_luc_emissions_future使用
                # 优先级调整：luc_period_end_df > coarse_transitions_df > luc_deltas_df > luc_area_df
                # 原因：luc_period_end_df包含完整的grassland_ha数据（来自dynamic_grass_requirement_df merge）

                luc_area_for_future = pd.DataFrame()
                iso3_to_country = {v: k for k, v in universe.iso3_by_country.items()}
                iso3_to_m49_map = {}
                for iso, ctry in iso3_to_country.items():
                    m49 = universe.m49_by_country.get(ctry)
                    if m49:
                        try:
                            iso3_to_m49_map[iso] = f"'{int(float(m49)):03d}"  # ✅ 'xxx格式
                        except Exception:
                            iso3_to_m49_map[iso] = str(m49).strip()

                # ✅ 优先级1: 使用luc_period_end_df（包含完整grassland_ha）
                if isinstance(luc_period_end_df, pd.DataFrame) and not luc_period_end_df.empty:
                    _log_step("  [INFO] 🔥 【优先级1】使用 luc_period_end_df（包含绝对grassland面积）构建LUC排放输入")
                    luc_area_for_future = luc_period_end_df.copy()
                    
                    # 添加iso3（如果缺失）
                    if 'iso3' not in luc_area_for_future.columns:
                        luc_area_for_future['iso3'] = luc_area_for_future['country'].map(universe.iso3_by_country)
                    
                    # 添加M49（如果缺失）
                    if 'M49_Country_Code' not in luc_area_for_future.columns:
                        luc_area_for_future['M49_Country_Code'] = luc_area_for_future['iso3'].map(iso3_to_m49_map)
                    
                    # ✅ 关键验证：检查grassland_ha列
                    if 'grassland_ha' in luc_area_for_future.columns:
                        grass_nonzero = (luc_area_for_future['grassland_ha'] > 0).sum()
                        grass_total = luc_area_for_future['grassland_ha'].sum()
                        _log_step(f"  [INFO] ✅ grassland_ha列存在: {grass_nonzero}/{len(luc_area_for_future)} 行非零, 总计={grass_total:,.0f} ha")
                        
                        # 检查2080年数据
                        if 'year' in luc_area_for_future.columns and 2080 in luc_area_for_future['year'].values:
                            grass_2080 = luc_area_for_future[luc_area_for_future['year'] == 2080]['grassland_ha'].sum()
                            _log_step(f"  [INFO] 2080年grassland delta总计: {grass_2080:,.0f} ha")
                            if grass_2080 == 0:
                                _log_step(f"  [WARN] ❌ 2080年grassland delta=0，De/Reforestation_pasture排放将为0！")
                    else:
                        _log_step(f"  [WARN] ❌ luc_period_end_df缺少grassland_ha列！De/Reforestation_pasture排放将失败！")
                    
                    # ✅ 关键：计算delta = period_end - period_start
                    # period_end和period_start现在都存储绝对面积
                    _log_step(f"  [INFO] 从period_start/end计算delta（期末-期初）")
                    
                    if isinstance(luc_period_start_df, pd.DataFrame) and not luc_period_start_df.empty:
                        # 合并start和end数据
                        luc_area_for_future = luc_period_end_df.copy()
                        luc_area_for_future = luc_area_for_future.merge(
                            luc_period_start_df[['country', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']],
                            on=['country', 'year'],
                            how='left',
                            suffixes=('_end', '_start')
                        )
                        
                        # 计算delta
                        luc_area_for_future['cropland_ha'] = luc_area_for_future['cropland_ha_end'] - luc_area_for_future['cropland_ha_start'].fillna(0)
                        luc_area_for_future['forest_ha'] = luc_area_for_future['forest_ha_end'] - luc_area_for_future['forest_ha_start'].fillna(0)
                        luc_area_for_future['grassland_ha'] = luc_area_for_future['grassland_ha_end'] - luc_area_for_future['grassland_ha_start'].fillna(0)
                        
                        # 删除临时列
                        luc_area_for_future = luc_area_for_future.drop(columns=[c for c in luc_area_for_future.columns if c.endswith('_start') or c.endswith('_end')])
                        
                        _log_step(f"  [INFO] ✅ Delta计算完成: cropland, forest, grassland列现在是期末-期初的变化量")
                    
                    # 筛选未来年份
                    luc_area_for_future = luc_area_for_future[luc_area_for_future['year'] > 2020]
                    
                # ✅ 优先级2: 使用coarse_transitions_df（经森林约束）
                elif isinstance(coarse_transitions_df, pd.DataFrame) and not coarse_transitions_df.empty:
                    _log_step("  [INFO] 🔥 使用 coarse_transitions_df（经森林约束）计算LUC排放")
                    
                    # 从 coarse_transitions 构建 luc_area_for_future
                    luc_area_for_future = coarse_transitions_df.copy()
                    
                    # 添加 M49_Country_Code（如果不存在）
                    if 'M49_Country_Code' not in luc_area_for_future.columns:
                        luc_area_for_future['M49_Country_Code'] = luc_area_for_future['iso3'].map(iso3_to_m49_map)
                    
                    # 计算净变化（已经受森林约束）
                    # forest_to_cropland - cropland_to_forest = 净耕地增加（从森林转换）
                    luc_area_for_future['cropland_ha'] = (
                        luc_area_for_future.get('forest_to_cropland', 0) - 
                        luc_area_for_future.get('cropland_to_forest', 0)
                    )
                    luc_area_for_future['grassland_ha'] = (
                        luc_area_for_future.get('forest_to_pasture', 0) - 
                        luc_area_for_future.get('pasture_to_forest', 0)
                    )
                    # 森林变化 = -(耕地 + 草地)，保持总面积不变
                    luc_area_for_future['forest_ha'] = -(
                        luc_area_for_future['cropland_ha'] + 
                        luc_area_for_future['grassland_ha']
                    )
                    
                    if 'year' in luc_area_for_future.columns:
                        luc_area_for_future['year'] = luc_area_for_future['year'].astype(int)
                        luc_area_for_future = luc_area_for_future[luc_area_for_future['year'] > 2020]
                    
                    luc_area_for_future = luc_area_for_future.dropna(subset=['M49_Country_Code'])
                    
                    # ✅ DEBUG: 数据质量检查
                    cols_present = [c for c in ['cropland_ha', 'forest_ha', 'grassland_ha'] if c in luc_area_for_future.columns]
                    non_zero_count = (luc_area_for_future[cols_present] != 0).any(axis=1).sum() if cols_present else 0
                    
                    # 统计2080年数据（关键年份）
                    if 2080 in luc_area_for_future['year'].values:
                        data_2080 = luc_area_for_future[luc_area_for_future['year'] == 2080]
                        crop_2080 = data_2080['cropland_ha'].sum()
                        forest_2080 = data_2080['forest_ha'].sum()
                        _log_step(f"  [INFO] ✅ 2080年净变化（经森林约束）: cropland={crop_2080:,.0f} ha, forest={forest_2080:,.0f} ha")
                    
                    _log_step(f"  [INFO] 准备LUC排放计算数据（coarse_transitions）: {len(luc_area_for_future)} 行, 非零={non_zero_count}")
                    
                    # ✅ 验证grassland_ha列
                    if 'grassland_ha' in luc_area_for_future.columns:
                        grass_count = (luc_area_for_future['grassland_ha'] != 0).sum()
                        _log_step(f"  [INFO] grassland_ha非零行数: {grass_count}")
                    else:
                        _log_step(f"  [WARN] ❌ coarse_transitions未包含grassland_ha，将导致排放缺失！")
                    
                elif isinstance(luc_deltas_df, pd.DataFrame) and not luc_deltas_df.empty:
                    # ✅ 优先级3: 使用原始 luc_deltas_df（但添加警告）
                    _log_step("  [WARN] ⚠️ 【优先级3】luc_period_end_df/coarse_transitions_df不可用，回退使用luc_deltas_df（可能缺grassland）")
                    
                    luc_area_for_future = luc_deltas_df.copy()
                    if 'iso3' not in luc_area_for_future.columns:
                        luc_area_for_future['iso3'] = luc_area_for_future.get('country', '').map(universe.iso3_by_country)
                    luc_area_for_future['M49_Country_Code'] = luc_area_for_future['iso3'].map(iso3_to_m49_map)
                    rename_delta_cols = {
                        'd_cropland_ha': 'cropland_ha',
                        'd_forest_ha': 'forest_ha',
                        'd_grassland_ha': 'grassland_ha'
                    }
                    luc_area_for_future = luc_area_for_future.rename(columns=rename_delta_cols)
                    if 'year' in luc_area_for_future.columns:
                        luc_area_for_future['year'] = luc_area_for_future['year'].astype(int)
                        luc_area_for_future = luc_area_for_future[luc_area_for_future['year'] > 2020]
                    luc_area_for_future = luc_area_for_future.dropna(subset=['M49_Country_Code'])
                    
                    # ✅ 数据合理性检查
                    cols_present = [c for c in ['cropland_ha', 'forest_ha', 'grassland_ha'] if c in luc_area_for_future.columns]
                    if cols_present:
                        max_crop = luc_area_for_future['cropland_ha'].abs().max() if 'cropland_ha' in cols_present else 0
                        if max_crop > 1e8:  # 超过100M ha异常
                            _log_step(f"  [WARN] ⚠️ 检测到异常大的cropland值: {max_crop:,.0f} ha")
                            _log_step(f"         可能原因: 单位错误、累积值、或总量误当delta")
                    
                    non_zero_count = (luc_area_for_future[cols_present] != 0).any(axis=1).sum() if cols_present else 0
                    _log_step(f"  [INFO] 准备LUC排放计算数据（luc_deltas）: {len(luc_area_for_future)} 行, 非零={non_zero_count}")
                    
                    # ✅ 验证grassland_ha列
                    if 'grassland_ha' in luc_area_for_future.columns:
                        grass_count = (luc_area_for_future['grassland_ha'] != 0).sum()
                        _log_step(f"  [INFO] grassland_ha非零行数: {grass_count}")
                    else:
                        _log_step(f"  [WARN] ❌ luc_deltas_df未包含grassland_ha，需要添加！")
                    
                elif isinstance(luc_area_df, pd.DataFrame) and not luc_area_df.empty:
                    # ✅ 优先级4（最后回退）: 使用luc_area_df
                    _log_step("  [WARN] ⚠️ 【优先级4】所有优先数据源不可用，最终回退使用luc_area_df")
                    if 'iso3' in luc_area_df.columns:
                        luc_area_for_future = luc_area_df.copy()
                        luc_area_for_future['M49_Country_Code'] = luc_area_for_future['iso3'].map(iso3_to_m49_map)
                        rename_cols = {
                            'cropland_area_ha': 'cropland_ha',
                            'forest_area_ha': 'forest_ha',
                            'grassland_area_ha': 'grassland_ha'
                        }
                        for old, new in rename_cols.items():
                            if old in luc_area_for_future.columns:
                                luc_area_for_future = luc_area_for_future.rename(columns={old: new})

                        if not luc_area_for_future.empty and 'year' in luc_area_for_future.columns:
                            try:
                                luc_area_for_future['year'] = luc_area_for_future['year'].astype(int)
                                df_calc = luc_area_for_future[luc_area_for_future['year'] >= 2020].copy()
                                years_present = sorted(df_calc['year'].unique())
                                if 2020 not in years_present:
                                    _log_step('  [WARN] LUC????2020???????base_area_df??...')
                                    if isinstance(base_area_df, pd.DataFrame) and not base_area_df.empty:
                                        base_2020 = base_area_df[base_area_df['year'] == 2020].copy()
                                        if not base_2020.empty:
                                            base_2020['M49_Country_Code'] = base_2020['iso3'].map(iso3_to_m49_map)
                                            base_2020 = base_2020.dropna(subset=['M49_Country_Code'])
                                            cols_needed = ['M49_Country_Code', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']
                                            cols_available = [c for c in cols_needed if c in base_2020.columns]
                                            if len(cols_available) >= 3:
                                                df_calc = pd.concat([df_calc, base_2020[cols_available]], ignore_index=True)
                                                _log_step(f"  [INFO] ??? {len(base_2020)} ?2020????")
                                            else:
                                                _log_step(f"  [WARN] base_area_df?????: {cols_needed}")
                                        else:
                                            _log_step('  [WARN] base_area_df???2020???')
                                    else:
                                        _log_step('  [WARN] base_area_df???')
                                if 'M49_Country_Code' in df_calc.columns:
                                    df_calc = df_calc.sort_values(by=['M49_Country_Code', 'year'])
                                    cols_to_diff = ['cropland_ha', 'forest_ha', 'grassland_ha']
                                    cols_present = [c for c in cols_to_diff if c in df_calc.columns]
                                    if cols_present:
                                        for col in cols_present:
                                            df_calc[col] = df_calc.groupby('M49_Country_Code')[col].diff()
                                        luc_area_for_future = df_calc[df_calc['year'] > 2020].copy()
                                        non_zero_count = (luc_area_for_future[cols_present] != 0).any(axis=1).sum()
                                        _log_step(f"  [INFO] ???LUC?????? (?? {len(cols_present)} ?????, {len(luc_area_for_future)} ?????)")
                                else:
                                    _log_step('  [WARN] ??M49_Country_Code?????LUC??')
                            except Exception as e:
                                _log_step(f"  [WARN] ??LUC????: {e}")
                                import traceback
                                _log_step(traceback.format_exc())
                
                # ✅ 从模型优化结果 var['Qs'] 中提取 Roundwood 产量（未来年份由模型预测）
                roundwood_for_future = pd.DataFrame()
                try:
                    Qs_dict = var.get('Qs', {}) if var else {}
                    if Qs_dict:
                        # 提取 Roundwood 的产量数据
                        rw_rows = []
                        for (country, commodity, year), qs_var in Qs_dict.items():
                            if commodity == 'Roundwood':
                                try:
                                    production = float(qs_var.X) if hasattr(qs_var, 'X') else float(qs_var)
                                    iso3 = universe.iso3_by_country.get(country, '')
                                    m49 = universe.m49_by_country.get(country, '')
                                    if m49:
                                        try:
                                            m49_fmt = f"'{int(float(m49)):03d}"
                                        except Exception:
                                            m49_fmt = str(m49).strip()
                                    else:
                                        m49_fmt = ''
                                    rw_rows.append({
                                        'country': country,
                                        'iso3': iso3,
                                        'year': int(year),
                                        'roundwood_m3': production,  # 模型优化的产量
                                        'M49_Country_Code': m49_fmt
                                    })
                                except Exception:
                                    pass
                        if rw_rows:
                            roundwood_for_future = pd.DataFrame(rw_rows)
                            _log_step(f"✅ 从模型结果提取 Roundwood 产量: {len(roundwood_for_future)} 行")
                            # 检查未来年份数据
                            future_rw = roundwood_for_future[roundwood_for_future['year'] > 2020]
                            if len(future_rw) > 0:
                                _log_step(f"  未来年份 Roundwood: {len(future_rw)} 行，年份 {sorted(future_rw['year'].unique())}")
                            # ✅ 诊断：检查Roundwood产量在不同年份的变化
                            us_rw = roundwood_for_future[roundwood_for_future['country'] == 'United States of America']
                            if not us_rw.empty:
                                for year in [2020, 2030, 2050, 2080]:
                                    year_data = us_rw[us_rw['year'] == year]
                                    if not year_data.empty:
                                        prod = year_data['roundwood_m3'].iloc[0]
                                        _log_step(f"  [诊断] U.S. Roundwood {year}: {prod:,.0f} m³")
                        else:
                            _log_step(f"⚠️ 模型中 Qs 无 Roundwood 数据（Qs_dict keys 数量: {len(Qs_dict)}）")
                except Exception as e:
                    _log_step(f"⚠️ 从模型结果提取 Roundwood 失败: {e}")
                
                # 如果模型中没有 Roundwood（可能未纳入 universe.commodities），回退到历史数据
                if roundwood_for_future.empty and isinstance(roundwood_supply_df, pd.DataFrame) and not roundwood_supply_df.empty:
                    _log_step("⚠️ 模型中无 Roundwood 产量，使用历史数据（仅历史年份有效）")
                    roundwood_for_future = roundwood_supply_df.copy()
                    if 'iso3' in roundwood_for_future.columns:
                        iso3_to_country = {v: k for k, v in universe.iso3_by_country.items()}
                        iso3_to_m49_map = {}
                        for iso, ctry in iso3_to_country.items():
                            m49 = universe.m49_by_country.get(ctry)
                            if m49:
                                try:
                                    iso3_to_m49_map[iso] = f"'{int(float(m49)):03d}"
                                except Exception:
                                    iso3_to_m49_map[iso] = str(m49).strip()
                        roundwood_for_future['M49_Country_Code'] = roundwood_for_future['iso3'].map(iso3_to_m49_map)
                    if 'roundwood_m3' not in roundwood_for_future.columns and 'roundwood_supply_m3' in roundwood_for_future.columns:
                        roundwood_for_future = roundwood_for_future.rename(columns={'roundwood_supply_m3': 'roundwood_m3'})

                # ✅ DEBUG: 检查传入 run_luc_emissions_future 的数据
                if not luc_area_for_future.empty:
                    us_luc = luc_area_for_future[luc_area_for_future['country'] == 'United States of America'] if 'country' in luc_area_for_future.columns else pd.DataFrame()
                    if not us_luc.empty:
                        for year in [2030, 2050, 2080]:
                            year_data = us_luc[us_luc['year'] == year] if 'year' in us_luc.columns else pd.DataFrame()
                            if not year_data.empty:
                                crop_ha = year_data.get('cropland_ha', pd.Series([0])).iloc[0]
                                forest_ha = year_data.get('forest_ha', pd.Series([0])).iloc[0]
                                grass_ha = year_data.get('grassland_ha', pd.Series([0])).iloc[0]
                                _log_step(f"  [DEBUG] luc_area_for_future U.S. {year}: cropland={crop_ha:,.0f}, forest={forest_ha:,.0f}, grassland={grass_ha:,.0f}")
                
                # ✅ 关键修复：计算历史 Wood harvest 排放因子（kt CO2 / m³）
                # 使用历史排放数据(luc_hist_df)和历史Roundwood产量(roundwood_supply_df)
                # 计算每个国家的排放因子，确保未来预测与历史数据一致
                historical_wood_harvest_ef: Dict[str, float] = {}
                try:
                    if not luc_hist_df.empty and isinstance(roundwood_supply_df, pd.DataFrame) and not roundwood_supply_df.empty:
                        # 从历史LUC排放中提取 Wood harvest 的2020年数据
                        wood_hist = luc_hist_df[
                            (luc_hist_df['Process'] == 'Wood harvest') & 
                            (luc_hist_df['year'] == 2020)
                        ].copy()
                        
                        # 从历史 Roundwood 产量中提取2020年数据
                        rw_hist = roundwood_supply_df[roundwood_supply_df['year'] == 2020].copy()
                        
                        if not wood_hist.empty and not rw_hist.empty:
                            # 规范化 M49 格式
                            def _norm_m49_ef(val):
                                s = str(val).strip().lstrip("'\"")
                                try:
                                    return f"'{int(float(s)):03d}"
                                except:
                                    return f"'{s}" if not s.startswith("'") else s
                            
                            wood_hist['M49_norm'] = wood_hist['M49_Country_Code'].apply(_norm_m49_ef)
                            
                            # 为 roundwood 添加 M49
                            if 'M49_Country_Code' not in rw_hist.columns:
                                rw_hist['M49_Country_Code'] = rw_hist['country'].map(universe.m49_by_country)
                            rw_hist['M49_norm'] = rw_hist['M49_Country_Code'].apply(_norm_m49_ef)
                            
                            # 聚合历史排放（kt CO2）和产量（m³）
                            wood_agg = wood_hist.groupby('M49_norm')['value'].sum().to_dict()
                            rw_agg = rw_hist.groupby('M49_norm')['roundwood_m3'].sum().to_dict()
                            
                            # 计算排放因子: EF = kt CO2 / m³
                            for m49, emis_kt in wood_agg.items():
                                if m49 in rw_agg and rw_agg[m49] > 0:
                                    ef = emis_kt / rw_agg[m49]  # kt CO2 per m³
                                    historical_wood_harvest_ef[m49] = ef
                            
                            _log_step(f"✅ 计算历史Wood harvest排放因子: {len(historical_wood_harvest_ef)} 个国家")
                            # DEBUG: 检查几个关键国家的EF
                            for test_m49 in ["'840", "'004"]:  # U.S., Afghanistan
                                if test_m49 in historical_wood_harvest_ef:
                                    _log_step(f"  [诊断] {test_m49} Wood harvest EF: {historical_wood_harvest_ef[test_m49]:.6f} kt/m³")
                        else:
                            _log_step(f"⚠️ 无法计算历史Wood harvest EF: wood_hist={len(wood_hist)}, rw_hist={len(rw_hist)}")
                    else:
                        _log_step(f"⚠️ 无法计算历史Wood harvest EF: luc_hist_df={len(luc_hist_df) if not luc_hist_df.empty else 0}, roundwood_supply_df存在={isinstance(roundwood_supply_df, pd.DataFrame)}")
                except Exception as ef_err:
                    _log_step(f"⚠️ 计算历史Wood harvest EF失败: {ef_err}")
                
                # ========== 准备森林面积数据（用于Forest碳汇计算）==========
                forest_area_for_future = pd.DataFrame()
                if isinstance(luc_period_end_df, pd.DataFrame) and not luc_period_end_df.empty:
                    # 从 luc_period_end_df 提取森林面积（期末面积）
                    forest_cols = ['M49_Country_Code', 'country', 'iso3', 'year']
                    # 检查森林面积列
                    if 'forest_ha' in luc_period_end_df.columns:
                        forest_area_for_future = luc_period_end_df[forest_cols + ['forest_ha']].copy()
                    elif 'forest_area_ha' in luc_period_end_df.columns:
                        forest_area_for_future = luc_period_end_df[forest_cols + ['forest_area_ha']].copy()
                        forest_area_for_future = forest_area_for_future.rename(columns={'forest_area_ha': 'forest_ha'})
                    
                    if not forest_area_for_future.empty:
                        # 仅保留未来年份
                        forest_area_for_future = forest_area_for_future[forest_area_for_future['year'] > 2020]
                        _log_step(f"✅ 准备Forest碳汇面积数据: {len(forest_area_for_future)} 行")
                        # DEBUG: 检查几个关键年份的全球森林面积
                        for yr in [2030, 2050, 2080]:
                            yr_data = forest_area_for_future[forest_area_for_future['year'] == yr]
                            if not yr_data.empty:
                                total_forest = yr_data['forest_ha'].sum()
                                _log_step(f"  [诊断] {yr}年全球森林面积: {total_forest:,.0f} ha")
                else:
                    _log_step("⚠️ luc_period_end_df 为空，无法计算Forest碳汇")
                
                # ========== 计算历史Forest碳汇EF（kt CO2/ha）==========
                # ======= Forest EF加载 - 开关控制 =======
                # paths.use_precomputed_forest_ef: True=从Excel直接读取, False=从历史数据反算
                historical_forest_sink_ef = {}
                
                if paths.use_precomputed_forest_ef:
                    # 直接从LUCE_parameter.xlsx读取预计算的Forest EF
                    _log_step("[FOREST_EF] 使用预计算的Forest排放因子")
                    precomputed_ef = load_forest_ef_from_excel(paths.luce_forest_ef_xlsx, universe, sheet_name='Forest_EF')
                    if precomputed_ef:
                        # 转换为标准格式：precomputed_ef是 Dict[M49, Dict[year, EF]]
                        # 使用2020年的EF值
                        for m49, ef_by_year in precomputed_ef.items():
                            # 使用2020年EF，如果没有则用最近可用的年份
                            if 2020 in ef_by_year:
                                ef_val = ef_by_year[2020]
                            else:
                                # 用最后一个可用年份
                                latest_year = max(ef_by_year.keys())
                                ef_val = ef_by_year[latest_year]
                            
                            # m49已经是标准'xxx格式（来自load_forest_ef_from_excel）
                            # 单位转换：tCO2/ha/yr → kt CO2/ha/yr (除以1000)
                            historical_forest_sink_ef[m49] = ef_val / 1000.0
                        
                        _log_step(f"✅ 从Excel加载Forest EF: {len(historical_forest_sink_ef)} 个国家")
                        if historical_forest_sink_ef:
                            sample_efs = list(historical_forest_sink_ef.items())[:5]
                            _log_step(f"  [DEBUG] Forest EF样本: {sample_efs}")
                    else:
                        _log_step("⚠️ 预计算Forest EF为空，将回退到反算模式")
                        paths.use_precomputed_forest_ef = False  # 回退
                
                if not paths.use_precomputed_forest_ef:
                    # 从历史数据反算Forest EF
                    _log_step("[FOREST_EF] 从历史数据反算Forest排放因子")
                    try:
                        if not luc_hist_df.empty and isinstance(base_area_df, pd.DataFrame) and not base_area_df.empty:
                            # 从历史LUC排放提取Forest过程 - 只用2020年数据计算EF
                            # 注意：不能用多年sum，因为EF是单位面积的年排放率
                            forest_hist = luc_hist_df[
                                (luc_hist_df['Process'] == 'Forest') & 
                                (luc_hist_df['year'] == 2020)
                            ].copy()
                            
                            # ✅ 关键修复：过滤掉World汇总行（M49='000'）
                            forest_hist = forest_hist[~forest_hist['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"").isin(['0', '00', '000'])]
                            
                            # ✅ 修复：检查forest列名（可能是forest, forest_ha或forest_area_ha）
                            forest_col_name = None
                            if 'forest' in base_area_df.columns:
                                forest_col_name = 'forest'
                            elif 'forest_ha' in base_area_df.columns:
                                forest_col_name = 'forest_ha'
                            elif 'forest_area_ha' in base_area_df.columns:
                                forest_col_name = 'forest_area_ha'
                            
                            if not forest_hist.empty and forest_col_name:
                                _log_step(f"  [DEBUG] 使用森林面积列: {forest_col_name}, Forest历史数据: {len(forest_hist)} 国家")
                                # 标准化M49
                                forest_hist['M49_norm'] = forest_hist['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"")
                                forest_hist['M49_norm'] = forest_hist['M49_norm'].apply(lambda x: f"'{int(x):03d}" if x.isdigit() else x)
                                
                                base_area_df_norm = base_area_df.copy()
                                # 也只用2020年的面积数据
                                if 'year' in base_area_df_norm.columns:
                                    base_area_df_norm = base_area_df_norm[base_area_df_norm['year'] == 2020]
                                
                                if 'M49_Country_Code' in base_area_df_norm.columns:
                                    base_area_df_norm['M49_norm'] = base_area_df_norm['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"")
                                    base_area_df_norm['M49_norm'] = base_area_df_norm['M49_norm'].apply(lambda x: f"'{int(x):03d}" if x.isdigit() else x)
                                elif 'country' in base_area_df_norm.columns:
                                    base_area_df_norm['M49_norm'] = base_area_df_norm['country'].map(universe.m49_by_country)
                                    base_area_df_norm['M49_norm'] = base_area_df_norm['M49_norm'].astype(str).str.strip().str.lstrip("'\"")
                                    base_area_df_norm['M49_norm'] = base_area_df_norm['M49_norm'].apply(lambda x: f"'{int(x):03d}" if str(x).isdigit() else x)
                                
                                # 聚合2020年排放（kt CO2）和森林面积（ha）
                                # 注意：现在用的是2020单年数据，不再sum多年
                                forest_emis_agg = forest_hist.groupby('M49_norm')['value'].sum().to_dict()
                                forest_area_agg = base_area_df_norm.groupby('M49_norm')[forest_col_name].mean().to_dict()
                                
                                # ✅ DEBUG: 打印聚合结果样本
                                _log_step(f"  [DEBUG] 历史Forest排放聚合: {len(forest_emis_agg)} 国家")
                                _log_step(f"  [DEBUG] 基准森林面积聚合: {len(forest_area_agg)} 国家")
                                
                                # 计算排放因子: EF = kt CO2/年 / ha = kt CO2/(ha·年)
                                for m49, emis_kt in forest_emis_agg.items():
                                    if m49 in forest_area_agg and forest_area_agg[m49] > 0:
                                        ef = emis_kt / forest_area_agg[m49]  # kt CO2 per ha per year (应为负值)
                                        historical_forest_sink_ef[m49] = ef
                                
                                _log_step(f"✅ 计算历史Forest碳汇排放因子: {len(historical_forest_sink_ef)} 个国家 (基于2020年数据)")
                                
                                # ✅ DEBUG: 打印EF样本
                                if historical_forest_sink_ef:
                                    sample_efs = list(historical_forest_sink_ef.items())[:5]
                                    _log_step(f"  [DEBUG] Forest EF样本: {sample_efs}")
                            else:
                                _log_step(f"⚠️ 无法计算历史Forest EF: forest_hist={len(forest_hist)}, forest_col={forest_col_name}, base_area列={list(base_area_df.columns)}")
                    except Exception as forest_ef_err:
                        _log_step(f"⚠️ 计算历史Forest碳汇EF失败: {forest_ef_err}")
                
                # ✅ 修复验证：调用LUC计算前最终检查数据完整性
                _log_step("\n" + "="*100)
                _log_step("[LUC INPUT 最终验证] 传入run_luc_emissions_future的数据:")
                _log_step("="*100)
                if not luc_area_for_future.empty:
                    _log_step(f"  行数: {len(luc_area_for_future)}")
                    _log_step(f"  列: {list(luc_area_for_future.columns)}")
                    
                    # 必需列检查
                    required_cols = ['M49_Country_Code', 'year', 'cropland_ha', 'grassland_ha', 'forest_ha']
                    missing_cols = [c for c in required_cols if c not in luc_area_for_future.columns]
                    if missing_cols:
                        _log_step(f"  ❌ [CRITICAL] 缺少必需列: {missing_cols}")
                    else:
                        _log_step(f"  ✅ 所有必需列存在")
                    
                    # grassland_ha详细统计
                    if 'grassland_ha' in luc_area_for_future.columns:
                        grass_nonzero = (luc_area_for_future['grassland_ha'] != 0).sum()
                        grass_total = luc_area_for_future['grassland_ha'].sum()
                        _log_step(f"  grassland_ha统计:")
                        _log_step(f"    非零行: {grass_nonzero}/{len(luc_area_for_future)}")
                        _log_step(f"    总计: {grass_total:,.0f} ha")
                        
                        # 2080年数据（关键年份）
                        if 'year' in luc_area_for_future.columns and 2080 in luc_area_for_future['year'].values:
                            data_2080 = luc_area_for_future[luc_area_for_future['year'] == 2080]
                            grass_2080 = data_2080['grassland_ha'].sum()
                            crop_2080 = data_2080['cropland_ha'].sum() if 'cropland_ha' in data_2080.columns else 0
                            forest_2080 = data_2080['forest_ha'].sum() if 'forest_ha' in data_2080.columns else 0
                            _log_step(f"  2080年数据汇总:")
                            _log_step(f"    grassland: {grass_2080:>15,.0f} ha")
                            _log_step(f"    cropland:  {crop_2080:>15,.0f} ha")
                            _log_step(f"    forest:    {forest_2080:>15,.0f} ha")
                            
                            if grass_2080 == 0:
                                _log_step(f"  ❌ [CRITICAL] 2080年grassland=0，De/Reforestation_pasture排放将为0！")
                            else:
                                _log_step(f"  ✅ 2080年grassland数据正常")
                    else:
                        _log_step(f"  ❌ [CRITICAL] 缺少grassland_ha列，将导致De/Reforestation_pasture排放计算失败！")
                else:
                    _log_step(f"  ❌ [CRITICAL] luc_area_for_future为空，LUC排放计算将失败！")
                _log_step("="*100 + "\n")
                
                # ✅ 关键修复：将累积delta转换为年度增量delta（避免重复累加）
                # 问题：luc_area_for_future中的cropland_ha等是相对2020年的累积变化（如2080年=1000万ha）
                # LUC模块会把每年的delta累加到碳库，导致重复计算
                # 解决：转换为年度增量（2030年增量，2050年增量，2080年增量）
                # 注意：如果只有一个年份（如2080），则累积delta就是增量delta（无需转换）
                if isinstance(luc_area_for_future, pd.DataFrame) and not luc_area_for_future.empty:
                    unique_years = sorted(luc_area_for_future['year'].unique())
                    
                    if len(unique_years) > 1:
                        # 多年份：需要转换为增量
                        _log_step(f"[LUC INCREMENT] 检测到多个年份{unique_years}，转换累积delta为年度增量...")
                        luc_area_incremental = []
                        
                        # 按国家分组，计算年度增量
                        for country in luc_area_for_future['country'].unique():
                            country_data = luc_area_for_future[luc_area_for_future['country'] == country].sort_values('year')
                            
                            prev_year = 2020
                            prev_cropland = 0.0
                            prev_grassland = 0.0
                            prev_forest = 0.0
                            
                            for _, row in country_data.iterrows():
                                year = row['year']
                                
                                # 累积delta（相对2020）
                                cumul_cropland = row.get('cropland_ha', 0.0)
                                cumul_grassland = row.get('grassland_ha', 0.0)
                                cumul_forest = row.get('forest_ha', 0.0)
                                
                                # 计算本期增量（相对上一期）
                                incr_cropland = cumul_cropland - prev_cropland
                                incr_grassland = cumul_grassland - prev_grassland
                                incr_forest = cumul_forest - prev_forest
                                
                                luc_area_incremental.append({
                                    'country': row['country'],
                                    'iso3': row.get('iso3', ''),
                                    'M49_Country_Code': row.get('M49_Country_Code', ''),
                                    'year': year,
                                    'cropland_ha': incr_cropland,
                                    'grassland_ha': incr_grassland,
                                    'forest_ha': incr_forest
                                })
                                
                                # 更新上一期
                                prev_cropland = cumul_cropland
                                prev_grassland = cumul_grassland
                                prev_forest = cumul_forest
                        
                        luc_area_for_future = pd.DataFrame(luc_area_incremental)
                        _log_step(f"[LUC INCREMENT] ✅ 已将累积delta转换为年度增量delta: {len(luc_area_for_future)} 行")
                        
                        # 诊断：打印转换后的全球年度增量
                        for yr in sorted(luc_area_for_future['year'].unique()):
                            yr_data = luc_area_for_future[luc_area_for_future['year'] == yr]
                            d_crop = yr_data['cropland_ha'].sum()
                            d_grass = yr_data['grassland_ha'].sum()
                            d_forest = yr_data['forest_ha'].sum()
                            _log_step(f"  {yr}年增量: d_cropland={d_crop:,.0f} ha, d_grassland={d_grass:,.0f} ha, d_forest={d_forest:,.0f} ha")
                    else:
                        # 单年份：累积delta即为增量delta（相对2020），无需转换
                        _log_step(f"[LUC INCREMENT] 检测到单一年份{unique_years}，累积delta（相对2020）即为增量delta，无需转换")
                        _log_step(f"  {unique_years[0]}年增量（=累积delta）: d_cropland={luc_area_for_future['cropland_ha'].sum():,.0f} ha, d_grassland={luc_area_for_future['grassland_ha'].sum():,.0f} ha")
                
                # 调用未来LUC排放计算
                luc_future_dict = run_luc_emissions_future(
                    param_excel=paths.luc_param_xlsx,
                    luc_area_df=luc_area_for_future,
                    roundwood_change_df=roundwood_for_future,
                    forest_area_df=forest_area_for_future,  # ✅ 传入森林面积（用于Forest碳汇）
                    years=active_years,
                    dict_v3_path=paths.dict_v3_path,
                    historical_wood_harvest_ef=historical_wood_harvest_ef,  # ✅ 传入历史EF
                    historical_forest_sink_ef=historical_forest_sink_ef,    # ✅ 传入历史Forest EF
                )
                luc_future_df = luc_future_dict.get('future', pd.DataFrame())
                _log_step(f"计算未来LUC排放: {len(luc_future_df)} 行")
            except Exception as e:
                _log_step(f"[WARN] 计算未来LUC排放失败: {e}")
                luc_future_df = pd.DataFrame()
            # ✅ 关键修复：统一year列类型为int
            if 'year' in luc_hist_df.columns:
                luc_hist_df['year'] = luc_hist_df['year'].astype(int)
            if 'year' in luc_future_df.columns:
                luc_future_df['year'] = luc_future_df['year'].astype(int)
            
            # 合并历史+未来LUC排放
            luc_combined_df = pd.concat([luc_hist_df, luc_future_df], ignore_index=True) if not luc_hist_df.empty or not luc_future_df.empty else pd.DataFrame()
            if not luc_combined_df.empty:
                # 确保M49_Country_Code为字符串格式
                luc_combined_df['M49_Country_Code'] = luc_combined_df['M49_Country_Code'].astype(str)
                # 确保year列为int类型
                if 'year' in luc_combined_df.columns:
                    luc_combined_df['year'] = luc_combined_df['year'].astype(int)
                # ✅ 关键修复：直接传递DataFrame，不要用列表包装
                fao_results['LUC'] = luc_combined_df
                _log_step(f"已将LUC排放(共{len(luc_combined_df)}行)添加到fao_results")
                # ✅ 关键调试：检查LUC数据中的年份分布
                luc_years = sorted(luc_combined_df['year'].unique()) if 'year' in luc_combined_df.columns else []
                luc_2080 = luc_combined_df[luc_combined_df['year'] == 2080] if 'year' in luc_combined_df.columns else pd.DataFrame()
        except Exception as e:
            _log_step(f"[WARN] LUC排放处理异常: {e}")
        # 9.6) 处理Drained Organic Soils排放（历史+未来）
        _log_step("开始处理Drained Organic Soils排放（历史+未来）")
        try:
            # 准备未来年份的Cropland和Grassland面积
            cropland_area_future = {}  # {(m49, country, year): area_ha}
            grassland_area_future = {}  # {(m49, country, year): area_ha}
            
            # ✅ 关键修复：优先从luc_period_end_df获取绝对面积（而非delta）
            source_df = None
            if isinstance(luc_period_end_df, pd.DataFrame) and not luc_period_end_df.empty:
                source_df = luc_period_end_df
                _log_step("[GSOIL INPUT] 使用 luc_period_end_df（绝对面积）构建grassland_area_future")
            elif isinstance(luc_area_df, pd.DataFrame) and not luc_area_df.empty:
                source_df = luc_area_df
                _log_step("[GSOIL INPUT] 回退使用 luc_area_df 构建grassland_area_future")
            
            if source_df is not None:
                # 确保M49_Country_Code存在
                if 'M49_Country_Code' not in source_df.columns:
                    if 'iso3' in source_df.columns:
                        iso3_to_country = {v: k for k, v in universe.iso3_by_country.items()}
                        iso3_to_m49_map = {}
                        for iso, ctry in iso3_to_country.items():
                            m49 = universe.m49_by_country.get(ctry)
                            if m49:
                                try:
                                    iso3_to_m49_map[iso] = f"'{int(float(m49)):03d}"
                                except Exception:
                                    iso3_to_m49_map[iso] = str(m49).strip()
                        source_df = source_df.copy()
                        source_df['M49_Country_Code'] = source_df['iso3'].map(iso3_to_m49_map)
                    elif 'country' in source_df.columns:
                        source_df = source_df.copy()
                        source_df['M49_Country_Code'] = source_df['country'].map(universe.m49_by_country)
                
                # 提取未来年份的面积数据
                for _, row in source_df.iterrows():
                    if pd.notna(row.get('year')) and row['year'] > 2020:
                        m49 = str(row.get('M49_Country_Code', '')).strip()
                        country = row.get('country', '')
                        year = int(row['year'])
                        
                        # Cropland面积（优先cropland_ha，回退cropland_area_ha）
                        crop_val = row.get('cropland_ha') if 'cropland_ha' in row else row.get('cropland_area_ha')
                        if pd.notna(crop_val) and crop_val > 0:
                            cropland_area_future[(m49, country, year)] = crop_val
                        
                        # Grassland面积（优先grassland_ha，回退grassland_area_ha）
                        grass_val = row.get('grassland_ha') if 'grassland_ha' in row else row.get('grassland_area_ha')
                        if pd.notna(grass_val) and grass_val > 0:
                            grassland_area_future[(m49, country, year)] = grass_val
                
                _log_step(f"[GSOIL INPUT] 从{'luc_period_end_df' if source_df is luc_period_end_df else 'luc_area_df'}提取未来面积数据")
                _log_step(f"  - cropland记录: {len(cropland_area_future)}")
                _log_step(f"  - grassland记录: {len(grassland_area_future)}")
                
                # 诊断：检查2080年数据
                grass_2080_keys = [k for k in grassland_area_future.keys() if k[2] == 2080]
                if grass_2080_keys:
                    grass_2080_total = sum(grassland_area_future[k] for k in grass_2080_keys)
                    _log_step(f"  - 2080年grassland: {len(grass_2080_keys)}个国家, 总计={grass_2080_total:,.0f} ha")
                else:
                    _log_step(f"  - ⚠️ 2080年grassland数据为空！")
            else:
                _log_step(f"[WARN] 无LUC面积数据（luc_period_end_df和luc_area_df均为空），将跳过未来Drained Organic Soils计算")
            # 调用Drained Organic Soils排放计算
            soil_emissions_path = os.path.join(get_input_base(), 'Emission', 
                                              'Emissions_Drained_Organic_Soils_E_All_Data_NOFLAG.csv')
            soil_params_path = os.path.join(get_src_base(), 'Soil_parameters.xlsx')
            if os.path.exists(soil_params_path):
                gsoil_results = run_drained_organic_soils_emissions(
                    universe=universe,
                    dict_v3_path=paths.dict_v3_path,
                    soil_params_path=soil_params_path,
                    emissions_csv_path=soil_emissions_path,
                    cropland_area_future=cropland_area_future,
                    grassland_area_future=grassland_area_future,
                    historical_years=active_years
                )
                if gsoil_results:
                    _log_step(f"Drained Organic Soils排放计算完成: {len(gsoil_results)} 个模块")
                    _log_step(f"  返回的keys: {list(gsoil_results.keys())}")
                    # 处理所有排放结果（不再跳过Historical开头的）
                    gsoil_emis_list = []
                    for process_name, emis_df in gsoil_results.items():
                        if not isinstance(emis_df, pd.DataFrame):
                            _log_step(f"  [SKIP] {process_name}: 不是DataFrame，类型={type(emis_df)}")
                            continue
                        if emis_df.empty:
                            _log_step(f"  [SKIP] {process_name}: DataFrame为空")
                            continue
                        _log_step(f"  [处理] {process_name}: {len(emis_df)}行, 列={list(emis_df.columns)}")
                        std_emis = emis_df.copy()
                        # 规范化M49代码为3位数字字符串格式
                        if 'M49_Country_Code' in std_emis.columns:
                            std_emis['M49_Country_Code'] = std_emis['M49_Country_Code'].astype(str)
                        # 确保country存在
                        if 'country' not in std_emis.columns:
                            std_emis['country'] = std_emis['M49_Country_Code'].map(universe.country_by_m49)
                        # 确保三个排放列都存在
                        if 'ch4_kt' not in std_emis.columns:
                            std_emis['ch4_kt'] = 0.0
                        if 'n2o_kt' not in std_emis.columns:
                            std_emis['n2o_kt'] = 0.0
                        if 'co2_kt' not in std_emis.columns:
                            std_emis['co2_kt'] = 0.0
                        # 重命名Year为year
                        if 'Year' in std_emis.columns and 'year' not in std_emis.columns:
                            std_emis = std_emis.rename(columns={'Year': 'year'})
                        # 添加commodity列，使用DataFrame中已有的Process列作为process
                        if 'Item' not in std_emis.columns and 'commodity' in std_emis.columns:
                            std_emis['Item'] = std_emis['commodity']
                        std_emis['commodity'] = std_emis.get('Item', '')
                        # 使用DataFrame中原有的Process列（必须存在）
                        if 'Process' in std_emis.columns:
                            std_emis['process'] = std_emis['Process']
                        elif 'process' in std_emis.columns:
                            pass  # 已有小写process列，保持不变
                        else:
                            raise ValueError(f"GSOIL DataFrame 缺少 Process/process 列！来自key: {process_name}")
                        # 选择输出列
                        output_cols = ['M49_Country_Code', 'country', 'year', 'commodity', 'process', 
                                      'ch4_kt', 'n2o_kt', 'co2_kt']
                        missing_cols = [col for col in output_cols if col not in std_emis.columns]
                        if missing_cols:
                            _log_step(f"    [SKIP] 缺失列: {missing_cols}")
                            continue
                        clean_df = std_emis[output_cols].copy()
                        gsoil_emis_list.append(clean_df)
                        _log_step(f"    添加成功: {len(clean_df)} 行")
                    _log_step(f"  处理完成，gsoil_emis_list包含 {len(gsoil_emis_list)} 个DataFrame")
                    # 将gsoil排放结果添加到fao_results
                    if gsoil_emis_list:
                        if 'GSOIL' not in fao_results or fao_results['GSOIL'] is None:
                            fao_results['GSOIL'] = []
                        if not isinstance(fao_results['GSOIL'], list):
                            fao_results['GSOIL'] = [fao_results['GSOIL']] if fao_results['GSOIL'] else []
                        fao_results['GSOIL'].extend(gsoil_emis_list)
                        total_gsoil_rows = sum(len(df) for df in fao_results['GSOIL'] if isinstance(df, pd.DataFrame))
                        _log_step(f"GSOIL已添加到fao_results: {len(gsoil_emis_list)}个DataFrame, 总计{total_gsoil_rows}行")
                    else:
                        _log_step("[WARN] gsoil_emis_list为空，未添加任何数据到fao_results")
                else:
                    _log_step("[WARN] Drained Organic Soils排放计算返回空结果")
            else:
                _log_step(f"[WARN] Soil_parameters.xlsx不存在({soil_params_path})，跳过土壤排放计算")
        except Exception as e:
            _log_step(f"[WARN] Drained Organic Soils排放处理异常: {e}")
            import traceback
            _log_step(f"  详情: {traceback.format_exc()}")
        # 10) 运行完整的畜牧业排放计算（从产量到排放的完整链路）
        _log_step("开始运行畜牧业排放完整计算")
        try:
            # ✅ 关键修复：加载Item_Emis → Item_Production_Map映射
            # 模型中的commodity是Item_Emis名称，但GLE模块期望Item_Production_Map名称
            emis_to_production_map = {}
            try:
                emis_item_df = pd.read_excel(paths.dict_v3_path, sheet_name='Emis_item')
                for _, row in emis_item_df.iterrows():
                    item_emis = row.get('Item_Emis')
                    item_prod = row.get('Item_Production_Map')
                    if pd.notna(item_emis) and pd.notna(item_prod):
                        emis_to_production_map[str(item_emis)] = str(item_prod)
                _log_step(f"✅ 加载了 {len(emis_to_production_map)} 个商品映射 (Item_Emis → Item_Production_Map)")
                # 调试：显示前5个畜牧商品映射
                livestock_items_sample = {k: v for k, v in list(emis_to_production_map.items())[:5] if 'cattle' in k.lower() or 'dairy' in k.lower()}
                if livestock_items_sample:
                    _log_step(f"  样例映射: {livestock_items_sample}")
            except Exception as map_err:
                _log_step(f"⚠️ 警告：无法加载商品映射: {map_err}")
                _log_step(f"  将直接使用Item_Emis名称，可能导致未来年份计算失败")
            # 从模型结果提取产量数据
            future_production_data = []
            # ✅ DEBUG: 诊断条件
            _log_step(f"[DEBUG] 检查future_production_data提取条件:")
            _log_step(f"  - solve={solve}, solve_status={solve_status}, gp.GRB.OPTIMAL={gp.GRB.OPTIMAL}")
            _log_step(f"  - solve and solve_status == gp.GRB.OPTIMAL: {solve and solve_status == gp.GRB.OPTIMAL}")
            if solve and solve_status == gp.GRB.OPTIMAL:
                Qs = var.get('Qs', {})
                _log_step(f"  - Qs变量数量: {len(Qs)}")
                # 统计年份分布
                years_in_qs = {}
                for (i, j, t), v in Qs.items():
                    years_in_qs[t] = years_in_qs.get(t, 0) + 1
                _log_step(f"  - Qs中的年份分布: {years_in_qs}")
                qs_with_value = 0
                for (i, j, t), v in Qs.items():
                    try:
                        # ✅ 修复：Gurobi Qs变量的单位是tonnes（吨），与Q0/FAOSTAT一致
                        # 之前错误地假设Qs单位是kg并除以1000，导致production_t被缩小1000倍
                        prod_val_t = v.X if hasattr(v, 'X') else 0.0
                        if prod_val_t > 0:
                            qs_with_value += 1
                            # ✅ 修复：M49代码标准化为 'xxx 格式（单引号+3位数字）
                            m49_raw = universe.m49_by_country.get(i)
                            if m49_raw is not None:
                                # ✅ 标准化为 'xxx 格式（单引号+3位数字）
                                m49_str = str(m49_raw).strip().lstrip("'\"")  # 去除可能的引号
                                if m49_str.isdigit():
                                    m49_code = f"'{m49_str.zfill(3)}"  # 标准格式: '004
                                else:
                                    m49_code = f"'{m49_str}"  # 保持原样但加引号
                            else:
                                m49_code = None
                            # ✅ 关键修复：将Item_Emis (j) 映射为Item_Production_Map
                            commodity_name = j  # j是Item_Emis名称（如"Cattle, dairy"）
                            if emis_to_production_map and commodity_name in emis_to_production_map:
                                commodity_name = emis_to_production_map[commodity_name]  # 映射为FAOSTAT名称（如"Raw milk of cattle"）
                            future_production_data.append({
                                'country': i,
                                'M49_Country_Code': m49_code,
                                'Commodity': commodity_name,  # 使用映射后的名称
                                'Item_Emis': j,  # 保留原始Item_Emis用于调试
                                'year': t,
                                'production_t': prod_val_t
                            })
                    except Exception:
                        continue
            else:
                pass  # 模型未求解
            # 合并历史和未来产量
            production_for_livestock = production_df.copy() if len(production_df) else pd.DataFrame()
            # 修复：统一commodity列名为Commodity（大写C）
            if len(production_for_livestock) > 0:
                if 'commodity' in production_for_livestock.columns and 'Commodity' not in production_for_livestock.columns:
                    production_for_livestock = production_for_livestock.rename(columns={'commodity': 'Commodity'})
                # 添加M49_Country_Code列（如果缺失）
                if 'M49_Country_Code' not in production_for_livestock.columns:
                    if 'country' in production_for_livestock.columns:
                        # 从country反向映射到M49
                        country_to_m49 = {v: k for k, v in universe.country_by_m49.items()}
                        production_for_livestock['M49_Country_Code'] = production_for_livestock['country'].map(country_to_m49)
                # ✅ 规范化M49格式为 'xxx（单引号+3位数字）
                if 'M49_Country_Code' in production_for_livestock.columns:
                    def _format_m49_standard(val):
                        if pd.isna(val):
                            return val
                        s = str(val).strip().lstrip("'\"")
                        try:
                            return f"'{int(s):03d}"
                        except:
                            return f"'{s}"
                    production_for_livestock['M49_Country_Code'] = production_for_livestock['M49_Country_Code'].apply(_format_m49_standard)
            # WARNING: debug: check M49 format in historical production_df
            if len(production_for_livestock) > 0 and 'M49_Country_Code' in production_for_livestock.columns:
                sample_m49 = production_for_livestock['M49_Country_Code'].head(5).tolist()
            if future_production_data:
                future_df = pd.DataFrame(future_production_data)
                # ✅ 调试：检查未来年份数据的商品名称映射
                _log_step(f"[DEBUG] future_df 行数: {len(future_df)}")
                if 'Commodity' in future_df.columns:
                    unique_commodities = future_df['Commodity'].unique()
                    _log_step(f"  - 未来年份包含 {len(unique_commodities)} 种商品")
                    _log_step(f"  - 前5种: {unique_commodities[:5].tolist() if len(unique_commodities) > 0 else '无'}")
                    # ✅ 检查畜牧商品（使用Item_Production_Map格式）
                    livestock_keywords = ['cattle', 'swine', 'pig', 'chicken', 'sheep', 'goat', 'buffalo', 'dairy', 'meat', 'milk', 'egg']
                    livestock_in_future = [c for c in unique_commodities if any(kw in str(c).lower() for kw in livestock_keywords)]
                    _log_step(f"  - 其中畜牧商品 ({len(livestock_in_future)}个): {livestock_in_future[:10]}")
                if 'Item_Emis' in future_df.columns:
                    unique_emis = future_df['Item_Emis'].unique()
                    _log_step(f"  - 对应的Item_Emis: {unique_emis[:5].tolist() if len(unique_emis) > 0 else '无'}")
                    # ✅ 检查畜牧Item_Emis
                    livestock_emis = [c for c in unique_emis if any(kw in str(c).lower() for kw in livestock_keywords)]
                    _log_step(f"  - 畜牧Item_Emis ({len(livestock_emis)}个): {livestock_emis[:10]}")
                # WARNING: debug: check M49 format in future_df
                if 'M49_Country_Code' in future_df.columns:
                    sample_m49_future = future_df['M49_Country_Code'].head(5).tolist()
                if len(production_for_livestock):
                    before_concat = len(production_for_livestock)
                    production_for_livestock = pd.concat([production_for_livestock, future_df], ignore_index=True)
                    after_concat = len(production_for_livestock)
                    # 检查未来年份是否成功合并
                    if 'year' in production_for_livestock.columns:
                        future_count = len(production_for_livestock[production_for_livestock['year'] > 2020])
                        _log_step(f"  - 合并后production_for_livestock: {after_concat} 行")
                        _log_step(f"  - 其中未来年份 (>2020): {future_count} 行")
                else:
                    production_for_livestock = future_df
            else:
                _log_step(f"[DEBUG] ⚠️ future_production_data为空! 未来年份产量数据没有提取到。")
            # WARNING: important: ensure column names are standardized
            if len(production_for_livestock) > 0:
                # 标准化列名：统一使用 'Commodity'（大写C）
                if 'commodity' in production_for_livestock.columns:
                    if 'Commodity' in production_for_livestock.columns:
                        # 两列都存在，合并并删除重复
                        production_for_livestock['Commodity'] = production_for_livestock['Commodity'].fillna(production_for_livestock['commodity'])
                        production_for_livestock = production_for_livestock.drop(columns=['commodity'])
                    else:
                        # 只有 commodity，重命名
                        production_for_livestock = production_for_livestock.rename(columns={'commodity': 'Commodity'})
                # 删除 Commodity 列中的 NaN 值
                if 'Commodity' in production_for_livestock.columns:
                    production_for_livestock = production_for_livestock.dropna(subset=['Commodity']).copy()
                # 确保必需列存在
                required_cols = ['M49_Country_Code', 'Commodity', 'year', 'production_t']
                missing_cols = [c for c in required_cols if c not in production_for_livestock.columns]
                if missing_cols:
                    _log_step(f"WARNING: production data is missing required columns: {missing_cols}")
                    _log_step(f"  现有列: {list(production_for_livestock.columns)}")
                # 筛选畜牧商品（从 dict_v3 加载）
                try:
                    emis_item_df = pd.read_excel(paths.dict_v3_path, sheet_name='Emis_item')
                    livestock_processes = [
                        'Enteric fermentation',
                        'Manure management', 
                        'Manure applied to soils',
                        'Manure left on pasture'
                    ]
                    # 筛选畜牧相关的行
                    livestock_rows = emis_item_df[emis_item_df['Process'].isin(livestock_processes)]
                    # WARNING: production data uses Item_Production_Map names (FAOSTAT), not Item_Emis
                    # 需要收集所有畜牧商品的Item_Production_Map名称
                    livestock_production_items = livestock_rows['Item_Production_Map'].dropna().unique().tolist()
                    livestock_emis_items = livestock_rows['Item_Emis'].dropna().unique().tolist()
                    _log_step(f"  dict_v3 中定义的畜牧商品 (Item_Emis): {len(livestock_emis_items)} 种")
                    _log_step(f"    前5个: {livestock_emis_items[:5]}")
                    _log_step(f"  对应的FAOSTAT生产数据名称 (Item_Production_Map): {len(livestock_production_items)} 种")
                    _log_step(f"    前5个: {livestock_production_items[:5]}")
                    # 检查实际数据中的商品（修复后应该是Item_Production_Map FAOSTAT名称）
                    actual_commodities = production_for_livestock['Commodity'].unique()
                    _log_step(f"  产量数据中的商品: {len(actual_commodities)} 种")
                    _log_step(f"    前10个: {actual_commodities[:10].tolist()}")
                    # ✅ 关键修复：现在actual_commodities是Item_Production_Map格式，直接用livestock_production_items匹配
                    matching_prod = set(actual_commodities) & set(livestock_production_items)
                    _log_step(f"  匹配的畜牧商品 (Item_Production_Map): {len(matching_prod)} 种")
                    if matching_prod:
                        _log_step(f"    匹配示例: {list(matching_prod)[:5]}")
                        livestock_items_to_use = livestock_production_items
                        matching = matching_prod
                    else:
                        # Fallback: 检查是否仍是Item_Emis格式（向后兼容）
                        matching_emis = set(actual_commodities) & set(livestock_emis_items)
                        if matching_emis:
                            _log_step(f"  WARNING: 数据使用Item_Emis格式（应该已映射为Item_Production_Map）")
                            _log_step(f"    匹配示例: {list(matching_emis)[:5]}")
                            livestock_items_to_use = livestock_emis_items
                            matching = matching_emis
                        else:
                            _log_step(f"  ❌ ERROR: 没有匹配的畜牧商品！商品名称格式可能不对")
                            _log_step(f"    actual样例: {actual_commodities[:3].tolist() if len(actual_commodities) > 0 else '无'}")
                            _log_step(f"    expected (Production_Map): {livestock_production_items[:3]}")
                            _log_step(f"    expected (Emis): {livestock_emis_items[:3]}")
                            livestock_items_to_use = []
                            matching = set()
                    # 只保留匹配的畜牧商品
                    if matching:
                        before_filter = len(production_for_livestock)
                        production_for_livestock = production_for_livestock[
                            production_for_livestock['Commodity'].isin(livestock_items_to_use)
                        ].copy()
                        after_filter = len(production_for_livestock)
                        _log_step(f"  ✅ 筛选后的畜牧产量数据: {after_filter} 行 (过滤掉 {before_filter - after_filter} 行非畜牧商品)")
                    else:
                        production_for_livestock = pd.DataFrame()  # 没有匹配，清空
                        _log_step(f"  ❌ 筛选后的畜牧产量数据: 0 行 (无匹配商品)")
                    # WARNING: filter out countries where Region_label_new == 'no'
                    if 'M49_Country_Code' in production_for_livestock.columns:
                        region_df = pd.read_excel(paths.dict_v3_path, sheet_name='region')
                        # 规范化M49代码：消除前导单引号和前导零，将可转换项转为int
                        def _norm_m49(x):
                            try:
                                if pd.isna(x):
                                    return None
                                s = str(x).strip()
                                if s.startswith("'"):
                                    s = s[1:]
                                # 移除前导零
                                s = s.lstrip('0')
                                if s == '':
                                    return None
                                return int(s)
                            except Exception:
                                # 无法转换为int时回退为原始字符串（去掉单引号）
                                s = str(x).strip()
                                if s.startswith("'"):
                                    return s[1:]
                                return s
                        region_codes_raw = region_df[region_df['Region_label_new'] != 'no']['M49_Country_Code'].unique()
                        valid_m49_set = set(v for v in (_norm_m49(c) for c in region_codes_raw) if v is not None)
                        # 对 production_for_livestock 的 M49 也做相同规范化再比较，避免格式不一致导致误删
                        production_for_livestock['_m49_norm'] = production_for_livestock['M49_Country_Code'].apply(_norm_m49)
                        rows_before = len(production_for_livestock)
                        # 检查过滤前未来年份数据
                        future_before = 0
                        if 'year' in production_for_livestock.columns:
                            future_before = len(production_for_livestock[production_for_livestock['year'] > 2020])
                        production_for_livestock = production_for_livestock[
                            production_for_livestock['_m49_norm'].isin(valid_m49_set)
                        ].copy()
                        rows_after = len(production_for_livestock)
                        # 检查过滤后未来年份数据
                        future_after = 0
                        if 'year' in production_for_livestock.columns:
                            future_after = len(production_for_livestock[production_for_livestock['year'] > 2020])
                        # 清理辅助列
                        if '_m49_norm' in production_for_livestock.columns:
                            production_for_livestock = production_for_livestock.drop(columns=['_m49_norm'])
                        if rows_before > rows_after:
                            _log_step(f"  [INFO] 过滤掉 {rows_before - rows_after} 行无效国家的产量数据 (M49格式已规范化)")
                            if future_before > future_after:
                                _log_step(f"    ⚠️ 警告：过滤掉了 {future_before - future_after} 行未来年份数据！")
                        _log_step(f"  [INFO] Region过滤后剩余: {len(production_for_livestock)} 行 (其中未来年份: {future_after} 行)")
                except Exception as e:
                    _log_step(f"WARNING: failed to map commodities using dict_v3: {e}")
                    import traceback
                    _log_step(f"  错误详情: {traceback.format_exc()}")
            # 运行畜牧业排放计算
            if len(production_for_livestock):
                _log_step(f"准备计算畜牧业排放: {len(production_for_livestock)} 行产量数据")
                _log_step(f"  - 列名: {list(production_for_livestock.columns)}")
                # 检查商品列（可能是 'commodity' 或 'Commodity'）
                commodity_col = 'Commodity' if 'Commodity' in production_for_livestock.columns else 'commodity'
                if commodity_col in production_for_livestock.columns:
                    unique_commodities = production_for_livestock[commodity_col].unique()
                    _log_step(f"  - 包含 {len(unique_commodities)} 种商品: {unique_commodities[:10].tolist()}")
                    # ✅ 诊断：检查2080年的畜牧商品数据
                    if 'year' in production_for_livestock.columns:
                        y2080_data = production_for_livestock[production_for_livestock['year'] == 2080]
                        if len(y2080_data) > 0:
                            livestock_keywords = ['cattle', 'swine', 'pig', 'chicken', 'sheep', 'goat', 'buffalo', 'dairy', 'meat', 'milk', 'egg']
                            y2080_commodities = y2080_data[commodity_col].unique()
                            y2080_livestock = [c for c in y2080_commodities if any(kw in str(c).lower() for kw in livestock_keywords)]
                            _log_step(f"  - 2080年数据: {len(y2080_data)} 行, 包含 {len(y2080_commodities)} 种商品")
                            _log_step(f"  - 2080年畜牧商品 ({len(y2080_livestock)}个): {y2080_livestock[:10]}")
                            # 检查商品格式是否是Item_Production_Map（GLE期望的格式）
                            has_meat_of = any('Meat of' in str(c) for c in y2080_livestock)
                            has_raw_milk = any('Raw milk' in str(c) for c in y2080_livestock)
                            _log_step(f"  - 商品格式检查: 包含'Meat of'={has_meat_of}, 包含'Raw milk'={has_raw_milk}")
                            if not has_meat_of and not has_raw_milk and y2080_livestock:
                                _log_step(f"  ⚠️ 警告: 2080年畜牧商品可能使用Item_Emis格式而非Item_Production_Map格式!")
                                _log_step(f"     GLE需要如 'Meat of cattle with the bone, fresh or chilled' 格式")
                                _log_step(f"     当前格式示例: {y2080_livestock[:3]}")
                        else:
                            _log_step(f"  ⚠️ 警告: production_for_livestock中没有2080年数据!")
                if 'year' in production_for_livestock.columns:
                    _log_step(f"  - 年份范围: {production_for_livestock['year'].min()}-{production_for_livestock['year'].max()}")
                livestock_results = run_livestock_emissions(
                    production_df=production_for_livestock,
                    years=active_years,
                    gle_params_path=os.path.join(get_src_base(), 'GLE_parameters.xlsx'),
                    hist_production_path=paths.production_faostat_csv,
                    hist_emissions_path=os.path.join(get_input_base(), 'Emission', 
                                                     'Emissions_livestock_dairy_split.csv'),  # ✅ 使用已拆分dairy/non-dairy的文件
                    hist_manure_stock_path=os.path.join(get_input_base(), 'Manure_Stock', 
                                                        'Environment_LivestockManure_with_ratio.csv'),
                    dict_v3_path=paths.dict_v3_path,
                    scenario_params=scenario_ctx,
                    hist_cutoff_year=2020  # 历史年份(≤2020)直接从CSV读取，未来年份才计算
                )
                # 提取排放和参数
                livestock_emissions = livestock_results.get('emissions', {}) if isinstance(livestock_results, dict) else livestock_results
                livestock_parameters = livestock_results.get('parameters', {}) if isinstance(livestock_results, dict) else {}
                
                # DEBUG: 检查livestock_parameters内容
                _log_step(f"[诊断] livestock_parameters类型: {type(livestock_parameters)}")
                if livestock_parameters:
                    _log_step(f"[诊断] livestock_parameters keys: {list(livestock_parameters.keys())}")
                    for k, v in livestock_parameters.items():
                        if isinstance(v, pd.DataFrame):
                            _log_step(f"  [{k}]: {len(v)} 行, 列={list(v.columns)[:5]}")
                            if 'year' in v.columns:
                                year_dist = v['year'].value_counts().to_dict()
                                _log_step(f"    年份分布: {year_dist}")
                                # 检查2080年数据
                                y2080 = v[v['year'] == 2080]
                                _log_step(f"    2080年数据: {len(y2080)} 行")
                                if len(y2080) > 0:
                                    _log_step(f"    2080年数据示例:")
                                    _log_step(f"    {y2080.head(3).to_string()}")
                        else:
                            _log_step(f"  [{k}]: {type(v)}")
                else:
                    _log_step(f"[诊断] ⚠️ livestock_parameters为空或False!")
                
                # 将livestock参数合并到全局参数DataFrames
                # ✅ 现在GLE导出的参数中，country列就是M49代码（如'840'）
                # 不需要转换为国家名称！直接重命名为M49_Country_Code列，然后规范化格式即可
                
                # ✅ 定义局部的M49规范化函数（避免作用域问题）
                def _local_normalize_m49(m49_val):
                    """将M49标准化为'xxx格式（单引号+3位数字，如'004）"""
                    if m49_val is None or pd.isna(m49_val):
                        return ''
                    m49_str = str(m49_val).strip().strip("'\"")
                    try:
                        m49_int = int(m49_str)
                        # ✅ 确保格式：'004（单引号前缀+3位数字）
                        return f"'{m49_int:03d}"
                    except Exception:
                        return str(m49_val)
                
                def _normalize_gle_m49(df: pd.DataFrame) -> pd.DataFrame:
                    """规范化GLE导出数据中的M49格式"""
                    if df is None or df.empty:
                        return df
                    df = df.copy()
                    
                    # GLE的'country'列实际是M49代码
                    if 'country' in df.columns and 'M49_Country_Code' not in df.columns:
                        df = df.rename(columns={'country': 'M49_Country_Code'})
                    
                    # 规范化M49格式为'xxx（3位数字）
                    if 'M49_Country_Code' in df.columns:
                        df['M49_Country_Code'] = df['M49_Country_Code'].apply(_local_normalize_m49)
                    
                    return df
                
                # ✅ 创建country_to_m49映射，供后续函数使用
                country_to_m49_local = {v: k for k, v in universe.country_by_m49.items()}
                
                if livestock_parameters:
                    # 对于slaughter/stock/carcass_yield，将GLE计算的未来年份数据与FAOSTAT历史数据合并
                    # 而不是完全替换，这样可以保留2020年等历史baseline数据
                    def _merge_gle_with_historical(hist_df: pd.DataFrame, computed_df: pd.DataFrame, 
                                                     value_col: str, future_cutoff: int = 2020) -> pd.DataFrame:
                        """将GLE计算的未来数据与FAOSTAT历史数据合并（都使用M49）"""
                        if computed_df is None or computed_df.empty:
                            return hist_df
                        if hist_df is None or hist_df.empty:
                            return _normalize_gle_m49(computed_df)
                        
                        # 1. 规范化GLE导出的M49格式
                        computed_normalized = _normalize_gle_m49(computed_df)
                        
                        # 2. 确保历史数据也有M49_Country_Code列
                        hist_df = hist_df.copy()
                        if 'M49_Country_Code' not in hist_df.columns and 'country' in hist_df.columns:
                            # 从country映射到M49（使用外层作用域的country_to_m49_local）
                            hist_df['M49_Country_Code'] = hist_df['country'].map(
                                lambda c: country_to_m49_local.get(str(c).strip(), '')
                            )
                        if 'M49_Country_Code' in hist_df.columns:
                            hist_df['M49_Country_Code'] = hist_df['M49_Country_Code'].apply(_local_normalize_m49)
                        
                        # 3. 分离历史和未来
                        hist_only = hist_df[hist_df['year'] <= future_cutoff].copy() if 'year' in hist_df.columns else hist_df.copy()
                        future_computed = computed_normalized[computed_normalized['year'] > future_cutoff].copy() if 'year' in computed_normalized.columns else computed_normalized.copy()
                        
                        # 4. 合并
                        merged = pd.concat([hist_only, future_computed], ignore_index=True)
                        return merged
                    
                    if 'stock' in livestock_parameters:
                        stock_hist_df = _merge_gle_with_historical(stock_hist_df, livestock_parameters['stock'], 'stock_head')
                        _log_step(f"  - stock_hist_df: {len(stock_hist_df)} 行 (合并GLE计算+FAOSTAT历史，使用M49)")
                    if 'slaughter' in livestock_parameters:
                        slaughter_hist_df = _merge_gle_with_historical(slaughter_hist_df, livestock_parameters['slaughter'], 'slaughter_head')
                        _log_step(f"  - slaughter_hist_df: {len(slaughter_hist_df)} 行 (合并GLE计算+FAOSTAT历史，使用M49)")
                    if 'carcass_yield' in livestock_parameters:
                        livestock_yield_df = _merge_gle_with_historical(livestock_yield_df, livestock_parameters['carcass_yield'], 'yield_t_per_head')
                        _log_step(f"  - livestock_yield_df (meat类): {len(livestock_yield_df)} 行 (合并GLE计算+FAOSTAT历史，使用M49)")
                    # ✅ 新增：合并dairy_yield (milk/egg类) 到 livestock_yield_df
                    if 'dairy_yield' in livestock_parameters:
                        dairy_yield_computed = livestock_parameters['dairy_yield']
                        if not dairy_yield_computed.empty:
                            # 规范化M49格式
                            dairy_yield_computed = _normalize_gle_m49(dairy_yield_computed)
                            # 追加到livestock_yield_df（而不是覆盖）
                            livestock_yield_df = pd.concat([livestock_yield_df, dairy_yield_computed], ignore_index=True)
                            # 去重（以M49_Country_Code, year, commodity为key，保留最后一个）
                            livestock_yield_df = livestock_yield_df.drop_duplicates(
                                subset=['M49_Country_Code', 'year', 'commodity'], keep='last'
                            )
                            _log_step(f"  - livestock_yield_df (含dairy): {len(livestock_yield_df)} 行 (合并meat+dairy yield)")
                    if 'feed_requirement' in livestock_parameters:
                        # ✅ feed_requirement直接规范化M49格式
                        feed_req_df = _normalize_gle_m49(livestock_parameters['feed_requirement'])
                        _log_step(f"  - feed_req_df: {len(feed_req_df)} 行 (已M49规范化)")
                    if 'manure_ratio' in livestock_parameters:
                        # ✅ manure_ratio直接规范化M49格式
                        manure_ratio_df = _normalize_gle_m49(livestock_parameters['manure_ratio'])
                        _log_step(f"  - manure_ratio_df: {len(manure_ratio_df)} 行 (已M49规范化)")
                # 将畜牧业排放结果转换为标准格式并添加到fao_results
                if livestock_emissions and fao_results is not None:
                    livestock_emis_list = []
                    for process_name, emis_df in livestock_emissions.items():
                        if emis_df is not None and not emis_df.empty:
                            # 转换为标准排放格式
                            std_emis = emis_df.copy()
                            # 安全检查必需的列
                            if 'M49_Country_Code' not in std_emis.columns:
                                _log_step(f"  WARNING: {process_name}: missing M49_Country_Code column; skipping")
                                continue
                            if 'Item' not in std_emis.columns:
                                _log_step(f"  WARNING: {process_name}: missing Item column; skipping")
                                continue
                            # 关键修复：规范化M49代码为3位数字字符串格式
                            # livestock_emissions 中的M49格式为 '004 (带引号)
                            # 但universe.country_by_m49 期望 "004" (3位数字字符串，无引号)
                            # 需要提取数字并格式化为3位数字字符串供映射使用
                            def _normalize_m49_for_mapping(m49_val):
                                """从 '004 格式提取数字部分，返回'xxx格式"""
                                try:
                                    import re
                                    match = re.search(r'\d+', str(m49_val))
                                    if match:
                                        # 返回'xxx格式（单引号+3位数字）
                                        return f"'{int(match.group(0)):03d}"  # ✅ 'xxx格式
                                    return None
                                except Exception:
                                    return None
                            std_emis['M49_Country_Code'] = std_emis['M49_Country_Code'].apply(_normalize_m49_for_mapping)
                            std_emis['country'] = std_emis['M49_Country_Code'].map(universe.country_by_m49)
                            std_emis['commodity'] = std_emis['Item']
                            # 重命名排放列
                            std_emis = std_emis.rename(columns={
                                'CH4_kt': 'ch4_kt',
                                'N2O_kt': 'n2o_kt',
                                'CO2_kt': 'co2_kt'
                            })
                            # 调试：检查映射后的数据
                            _log_step(f"    - M49样本: {std_emis['M49_Country_Code'].iloc[:3].tolist() if 'M49_Country_Code' in std_emis.columns else '无M49列'}")
                            _log_step(f"    - country样本: {std_emis['country'].iloc[:3].tolist() if 'country' in std_emis.columns else '无country列'}")
                            _log_step(f"    - commodity样本: {std_emis['commodity'].iloc[:3].tolist() if 'commodity' in std_emis.columns else '无commodity列'}")
                            # WARNING: debug: inspect emission values
                            _log_step(f"    - 排放值统计:")
                            for gas_col in ['ch4_kt', 'n2o_kt', 'co2_kt']:
                                if gas_col in std_emis.columns:
                                    non_zero = (std_emis[gas_col] > 0).sum()
                                    total = std_emis[gas_col].sum()
                                    _log_step(f"      {gas_col}: {non_zero}个非零值, 总计={total:.2f} kt")
                                    if non_zero > 0:
                                        _log_step(f"        样本值: {std_emis[std_emis[gas_col]>0][gas_col].iloc[:3].tolist()}")
                            # 创建新的DataFrame，只选择需要的列，并确保process列是标量
                            # 删除原始的Item列，避免与commodity重名
                            if 'Item' in std_emis.columns:
                                std_emis = std_emis.drop(columns=['Item'])
                            # 关键修复：必须包含M49_Country_Code用于汇总
                            required_cols = ['M49_Country_Code', 'country', 'year', 'commodity', 'ch4_kt', 'n2o_kt', 'co2_kt']
                            clean_df = std_emis[required_cols].copy()
                            # 添加process列为标量字符串
                            clean_df['process'] = str(process_name)
                            livestock_emis_list.append(clean_df[
                                ['M49_Country_Code', 'country', 'year', 'commodity', 'process', 
                                 'ch4_kt', 'n2o_kt', 'co2_kt']
                            ])
                            # WARNING: validate data after appending
                            _log_step(f"    - 已添加到livestock_emis_list: {len(clean_df)} 行")
                            for gas_col in ['ch4_kt', 'n2o_kt', 'co2_kt']:
                                if gas_col in clean_df.columns:
                                    non_zero = (clean_df[gas_col] > 0).sum()
                                    _log_step(f"      {gas_col}: {non_zero}个非零值")
                    # 添加到GLE结果中
                    if livestock_emis_list:
                        # create debug dir and write diagnostics before extending
                        try:
                            import datetime
                            dbg_dir = os.path.join(os.getcwd(), 'debug_outputs')
                            os.makedirs(dbg_dir, exist_ok=True)
                            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            # summary log
                            with open(os.path.join(dbg_dir, f'gle_append_summary_{ts}.txt'), 'w', encoding='utf-8') as fh:
                                fh.write(f"livestock_emis_list_count={len(livestock_emis_list)}\n")
                                # show per-process row counts and sample M49s
                                for idx, df in enumerate(livestock_emis_list):
                                    try:
                                        rows = len(df)
                                        sample_m49 = df['M49_Country_Code'].dropna().astype(str).unique()[:5].tolist() if 'M49_Country_Code' in df.columns else []
                                    except Exception:
                                        rows = 'ERR'
                                        sample_m49 = []
                                    fh.write(f"item_{idx}: rows={rows}, sample_m49={sample_m49}\n")
                            # write concatenated snapshot for quick inspection
                            try:
                                import pandas as _pd
                                concat_df = _pd.concat(livestock_emis_list, ignore_index=True)
                                # small sample and full headcount
                                concat_df.head(50).to_csv(os.path.join(dbg_dir, f'gle_livestock_snapshot_head_{ts}.csv'), index=False)
                                concat_df.to_csv(os.path.join(dbg_dir, f'gle_livestock_snapshot_full_{ts}.csv'), index=False)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        if 'GLE' not in fao_results or fao_results['GLE'] is None:
                            fao_results['GLE'] = []
                        if not isinstance(fao_results['GLE'], list):
                            fao_results['GLE'] = [fao_results['GLE']] if fao_results['GLE'] else []

                        # record length before extend
                        try:
                            before_len = len(fao_results['GLE'])
                        except Exception:
                            before_len = None

                        fao_results['GLE'].extend(livestock_emis_list)

                        # record length after extend
                        try:
                            after_len = len(fao_results['GLE'])
                        except Exception:
                            after_len = None

                        # append a short append-log
                        try:
                            import datetime
                            dbg_dir = os.path.join(os.getcwd(), 'debug_outputs')
                            ts2 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            with open(os.path.join(dbg_dir, f'gle_append_after_{ts2}.txt'), 'w', encoding='utf-8') as fh:
                                fh.write(f"before_len={before_len}\n")
                                fh.write(f"after_len={after_len}\n")
                        except Exception:
                            pass
                # 统计实际计算的排放过程数量（非空的）
                if livestock_emissions and isinstance(livestock_emissions, dict):
                    non_empty_count = sum(1 for emis_df in livestock_emissions.values() 
                                         if emis_df is not None and not emis_df.empty)
                    _log_step(f"畜牧业排放计算完成，共计算 {non_empty_count}/{len(livestock_emissions)} 个排放过程")
                    # 显示每个过程的详细信息
                    for process_name, emis_df in livestock_emissions.items():
                        if emis_df is not None and not emis_df.empty:
                            _log_step(f"  - {process_name}: {len(emis_df)} 行数据")
                        else:
                            _log_step(f"  - {process_name}: 无数据 (跳过)")
                else:
                    _log_step(f"WARNING: livestock_emissions returned an empty result: {type(livestock_emissions)}")
            else:
                _log_step("无产量数据，跳过畜牧业排放计算")
        except Exception as exc:
            import traceback
            detail = traceback.format_exc()
            _log_step(f"畜牧业排放计算失败：{exc}\n{detail}")
            # 继续执行，不中断整个流程
    # ======================== 作物排放计算 (GCE) ========================
    if not can_emit:
        _log_step("跳过作物排放计算 (GCE) - 模型未成功求解")
    else:
        _log_step("开始作物排放计算...")
    try:
        if not can_emit:
            raise RuntimeWarning("SKIP_GCE")
        # 从 dict_v3 获取 M49 国家过滤表
        dict_v3_path = paths.dict_v3_path
        if os.path.exists(dict_v3_path):
            try:
                region_df = pd.read_excel(dict_v3_path, sheet_name='region')
                # 过滤出有效的 M49 代码（Region_label_new != 'no'）
                valid_regions = region_df[region_df['Region_label_new'] != 'no']['M49_Country_Code'].tolist()
                valid_m49_set = {str(int(x)) if not pd.isna(x) else None for x in valid_regions}
                _log_step(f"从dict_v3读取有效M49代码: {len(valid_m49_set)} 个")
            except Exception as e:
                _log_step(f"警告：无法从dict_v3读取M49过滤表: {e}，将使用全部M49代码")
                valid_m49_set = None
        else:
            _log_step(f"警告：dict_v3不存在({dict_v3_path})，将使用全部M49代码")
            valid_m49_set = None
        # 准备作物产量数据（包含历史+未来年份）
        crop_production_df = pd.DataFrame()
        try:
            # 从 dict_v3 获取作物项目列表
            emis_item_df = pd.read_excel(dict_v3_path, sheet_name='Emis_item')
            crop_processes = ['Crop residues', 'Burning crop residues', 'Rice cultivation', 'Synthetic fertilizers']
            crop_items_emis = emis_item_df[emis_item_df['Process'].isin(crop_processes)]['Item_Emis'].unique()
            _log_step(f"从dict_v3读取作物项目(Item_Emis): {len(crop_items_emis)} 个")
            # ✅ 修复：优先使用优化后的产量(var['Qs'])，而不是基准产量(node.Q0)
            Qs = var.get('Qs', {}) if var else {}
            if data and data.nodes:
                node_data = []
                for node in data.nodes:
                    if node.commodity not in crop_items_emis:
                        continue
                    key = (node.country, node.commodity, node.year)
                    # 优先使用优化后的产量Qs，否则回退到Q0
                    if key in Qs:
                        qs_var = Qs[key]
                        # 获取优化变量的值
                        if hasattr(qs_var, 'X'):
                            production_t = float(qs_var.X)
                        elif hasattr(qs_var, 'val'):
                            production_t = float(qs_var.val)
                        elif isinstance(qs_var, (int, float)):
                            production_t = float(qs_var)
                        else:
                            production_t = node.Q0
                    else:
                        production_t = node.Q0
                    
                    node_data.append({
                        'country': node.country,
                        'commodity': node.commodity,
                        'year': node.year,
                        'production_t': production_t
                    })
                
                node_df = pd.DataFrame(node_data)
                if not node_df.empty:
                    # 添加 M49_Country_Code（universe.m49_by_country已经是'xxx格式）
                    node_df['M49_Country_Code'] = node_df['country'].map(universe.m49_by_country)
                    # 重命名列名以匹配函数期望
                    crop_production_df = node_df.rename(columns={'commodity': 'Item'})
                    _log_step(f"准备作物生产数据(从nodes): {len(crop_production_df)} 行, 年份范围: {crop_production_df['year'].min()}-{crop_production_df['year'].max()}")
                    # DEBUG: 检查 China Rice 2080年的产量是否使用了优化值
                    china_rice_2080 = crop_production_df[(crop_production_df['country'] == 'China') & 
                                                          (crop_production_df['Item'] == 'Rice') & 
                                                          (crop_production_df['year'] == 2080)]
                    if not china_rice_2080.empty:
                        _log_step(f"  ✓ 诊断: China Rice 2080年 production_t = {china_rice_2080['production_t'].values[0]:,.0f}")
                    # DEBUG: 检查 U.S. Barley 的 production_t (用于诊断 Crop residues)
                    us_barley = crop_production_df[(crop_production_df['country'] == 'United States of America') & 
                                                    (crop_production_df['Item'] == 'Barley')]
                    if not us_barley.empty:
                        for year in [2020, 2080]:
                            year_data = us_barley[us_barley['year'] == year]
                            if not year_data.empty:
                                _log_step(f"  ✓ 诊断: U.S. Barley {year}年 production_t = {year_data['production_t'].values[0]:,.0f}")
                    # Debug: 显示未来年份数据统计
                    future_crop = crop_production_df[crop_production_df['year'] > 2020]
                    if not future_crop.empty:
                        _log_step(f"  - 未来年份作物数据: {len(future_crop)} 行, 年份: {sorted(future_crop['year'].unique())[:5]}")
                        unique_items = future_crop['Item'].nunique()
                        _log_step(f"  - 未来年份作物种类: {unique_items} 种")
                    else:
                        _log_step("  ⚠️ 警告：未来年份无作物数据！")
                else:
                    _log_step("警告：nodes中没有作物数据")
            else:
                _log_step("警告：data.nodes不可用，回退到历史数据")
                # 回退：使用历史production_df
                if not production_df.empty:
                    crop_items_prod = emis_item_df[emis_item_df['Process'].isin(crop_processes)]['Item_Production_Map'].unique()
                    crop_production_df = production_df[production_df['commodity'].isin(crop_items_prod)].copy()
                    # 添加 M49_Country_Code（universe.m49_by_country已经是'xxx格式）
                    crop_production_df['M49_Country_Code'] = crop_production_df['country'].map(universe.m49_by_country)
                    crop_production_df = crop_production_df.rename(columns={'commodity': 'Item'})
                    _log_step(f"准备作物生产数据(历史): {len(crop_production_df)} 行")
        except Exception as e:
            _log_step(f"警告：无法获取作物项目列表: {e}")
            import traceback
            traceback.print_exc()
        # 准备收获面积数据（包含历史+未来年份）
        harvest_area_df = pd.DataFrame()
        if not crop_production_df.empty:
            try:
                # 方法：从nodes获取yield0_t_per_ha，计算面积 = production / yield
                area_data = []
                # 构建yield查找字典（从nodes）
                yield_lookup = {}
                if data and data.nodes:
                    for node in data.nodes:
                        if hasattr(node, 'yield0_t_per_ha') and node.yield0_t_per_ha and node.yield0_t_per_ha > 0:
                            key = (node.country, node.commodity, node.year)
                            yield_lookup[key] = node.yield0_t_per_ha
                    _log_step(f"从nodes提取yield数据: {len(yield_lookup)} 条记录")
                for _, row in crop_production_df.iterrows():
                    country = row.get('country')
                    year = int(row.get('year'))
                    item = row.get('Item')
                    production_t = row.get('production_t', 0)
                    if production_t <= 0:
                        continue
                    # 查找yield（优先nodes，回退历史数据）
                    yield_t_per_ha = yield_lookup.get((country, item, year))
                    if not yield_t_per_ha and not yield_hist_df.empty:
                        # 回退：查询历史yield（使用最近年份）
                        yield_match = yield_hist_df[
                            (yield_hist_df['country'] == country) &
                            (yield_hist_df['commodity'] == item) &
                            (yield_hist_df['year'] <= year)
                        ]
                        if not yield_match.empty:
                            yield_match = yield_match.sort_values('year', ascending=False).iloc[0]
                            yield_t_per_ha = yield_match.get('yield_t_per_ha')
                    # 计算面积
                    if yield_t_per_ha and yield_t_per_ha > 0:
                        area_ha = production_t / yield_t_per_ha
                    else:
                        # 最后回退：使用默认yield（3 t/ha）
                        area_ha = production_t / 3.0
                    if area_ha > 0:
                        area_data.append({
                            'M49_Country_Code': row.get('M49_Country_Code'),
                            'country': country,
                            'year': year,
                            'Item': item,
                            'harvested_area_ha': area_ha
                        })
                if area_data:
                    harvest_area_df = pd.DataFrame(area_data)
                    _log_step(f"准备收获面积数据: {len(harvest_area_df)} 行, 年份范围: {harvest_area_df['year'].min()}-{harvest_area_df['year'].max()}")
                    # Debug: 显示未来年份面积统计
                    future_area = harvest_area_df[harvest_area_df['year'] > 2020]
                    if not future_area.empty:
                        _log_step(f"  - 未来年份面积数据: {len(future_area)} 行")
                    else:
                        _log_step("  ⚠️ 警告：未来年份无面积数据！")
                else:
                    _log_step("警告：无法计算收获面积数据")
            except Exception as e:
                _log_step(f"警告：无法准备收获面积数据: {e}")
                import traceback
                traceback.print_exc()
        # 调用作物排放计算
        if not crop_production_df.empty:
            # Fertilizer efficiency路径（如果存在则用于Synthetic fertilizers历史分配）
            fert_eff_path = os.path.join(get_input_base(), 'Fertilizer', 'Fertilizer_efficiency.xlsx')
            if not os.path.exists(fert_eff_path):
                _log_step("⚠️ Fertilizer_efficiency.xlsx未找到，Synthetic fertilizers将不按Item细分")
                fert_eff_path = None
            crop_results = run_crop_emissions(
                production_df=crop_production_df,
                harvest_area_df=harvest_area_df,
                years=active_years,
                gle_params_path=os.path.join(get_src_base(), 'GCE_parameters.xlsx'),
                dict_v3_path=dict_v3_path,
                hist_emissions_crop_path=os.path.join(get_input_base(), 'Emission', 
                                                      'Emissions_crops_E_All_Data_NOFLAG.csv'),
                fertilizer_efficiency_path=fert_eff_path,
                scenario_params=scenario_ctx.get('crop_emission_factors', {})
            )
            # 调试：检查crop_results
            if crop_results:
                if isinstance(crop_results, dict):
                    for key, val in crop_results.items():
                        if isinstance(val, pd.DataFrame):
                            pass
                        else:
                            pass
                else:
                    pass
            else:
                pass
            # ✅ 关键修复：GCE返回Dict格式，直接作为dict添加到fao_results
            # S4_1_results.summarize_emissions会正确处理dict格式（line 1291-1298）
            if crop_results and isinstance(crop_results, dict):
                _log_step(f"作物排放计算返回 {len(crop_results)} 个过程")
                # 直接将dict添加到fao_results['GCE']
                if 'GCE' not in fao_results or fao_results['GCE'] is None:
                    fao_results['GCE'] = {}
                elif not isinstance(fao_results['GCE'], dict):
                    _log_step(f"WARNING: fao_results['GCE']不是dict（类型={type(fao_results['GCE'])}），重置为dict")
                    fao_results['GCE'] = {}
                # 合并crop_results到GCE
                fao_results['GCE'].update(crop_results)
                _log_step(f"作物排放已添加到fao_results['GCE']（dict格式），共 {len(fao_results['GCE'])} 个过程:")
                for proc_name in fao_results['GCE'].keys():
                    proc_df = fao_results['GCE'][proc_name]
                    if isinstance(proc_df, pd.DataFrame):
                        _log_step(f"  - {proc_name}: {len(proc_df)} 行, 列: {list(proc_df.columns)}")
                else:
                    _log_step("作物排放计算返回空结果或格式错误")
        else:
            _log_step("无作物产量数据，跳过作物排放计算")
    except Exception as exc:
        if "SKIP_GCE" in str(exc):
            pass
        else:
            import traceback
            detail = traceback.format_exc()
            _log_step(f"作物排放计算失败：{exc}\n{detail}")
    # ======================== GCE模块已作为dict添加到fao_results ========================
    price_df = load_prices(paths.prices_csv, universe) if os.path.exists(paths.prices_csv) else None
    _log_step("\u5f00\u59cb\u751f\u6210\u5e02\u573a\u6c47\u603b")
    market_sum = summarize_market(model if solve else None, var, universe, data, price_df=price_df)
    _log_step("\u5e02\u573a\u6c47\u603b\u5b8c\u6210")
    
    # 生成成本汇总（仅包含减排成本信息）
    _log_step("\u5f00\u59cb\u751f\u6210\u6210\u672c\u6c47\u603b")
    from S4_1_results import generate_cost_summary
    cost_summary_path = outdir / "cost_summary.csv"
    generate_cost_summary(
        var=var,
        unit_cost_data=unit_cost_data,
        baseline_scenario_result={},  # 占位参数，函数实际不使用
        output_path=str(cost_summary_path),
        regions=list(set(n.country for n in data.nodes)),
        commodities=list(universe.commodities),
        years=active_years
    )
    _log_step(f"\u6210\u672c\u6c47\u603b\u5b8c\u6210\uff0c\u6587\u4ef6: {cost_summary_path}")
    
    # ✅ 诊断：检查2080年的Qs是否存在
    if market_sum is not None and len(market_sum) > 0:
        if 'year' in market_sum.columns and 'Qs' in market_sum.columns:
            qs_2080 = market_sum[(market_sum['year'] == 2080) & (market_sum['Qs'].notna())]
            qs_2080_count = len(qs_2080)
            if qs_2080_count > 0:
                qs_2080_mean = qs_2080['Qs'].mean()
                qs_2080_sum = qs_2080['Qs'].sum()
                _log_step(f"✓ market_summary中2080年Qs数据: {qs_2080_count}条, 平均={qs_2080_mean:.4f}t, 总和={qs_2080_sum:.0f}t")
            else:
                total_2080 = len(market_sum[market_sum['year'] == 2080])
                _log_step(f"⚠ market_summary中2080年Qs为空: {total_2080}条记录全部NaN (市场汇总共{len(market_sum)}行)")
        else:
            _log_step(f"⚠ market_sum缺少'year'或'Qs'列，无法检查2080数据")
    else:
        _log_step(f"⚠ market_sum为空或None")
    
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
    # NOTE: GFIRE 仅用于内部合并到 emissions_detail，不再单独输出到 DS 目录
    
    # ========== [DEBUG] build_production_summary调用前诊断 ==========
    _log_step("[DEBUG] build_production_summary调用前检查livestock参数DataFrame:")
    
    def _debug_livestock_df(name, df):
        """诊断livestock参数DataFrame"""
        if df is None or df.empty:
            _log_step(f"  {name}: 空或None")
            return
        _log_step(f"  {name}: {len(df)} 行")
        _log_step(f"    列名: {list(df.columns)}")
        # 检查是否有M49_Country_Code列
        if 'M49_Country_Code' in df.columns:
            _log_step(f"    ✅ 有M49_Country_Code列")
            # 样本M49
            m49_samples = df['M49_Country_Code'].head(3).tolist()
            _log_step(f"    M49样本: {m49_samples}")
        else:
            _log_step(f"    ❌ 无M49_Country_Code列!")
            if 'country' in df.columns:
                country_samples = df['country'].head(3).tolist()
                _log_step(f"    country列样本: {country_samples}")
        # 检查年份范围
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            _log_step(f"    年份范围: {years}")
            # 检查2080年数据
            year_2080 = df[df['year'] == 2080] if 2080 in years else pd.DataFrame()
            _log_step(f"    2080年行数: {len(year_2080)}")
        # 检查商品
        if 'commodity' in df.columns:
            comms = df['commodity'].unique()[:5]
            _log_step(f"    commodity样本: {list(comms)}")
    
    _debug_livestock_df("slaughter_hist_df", slaughter_hist_df)
    _debug_livestock_df("stock_hist_df", stock_hist_df)
    _debug_livestock_df("livestock_yield_df", livestock_yield_df)
    _debug_livestock_df("feed_req_df", feed_req_df)
    _debug_livestock_df("manure_ratio_df", manure_ratio_df)
    _log_step("[DEBUG] build_production_summary调用前检查完成")
    # ========== [DEBUG] 诊断结束 ==========
    
    production_summary_df = build_production_summary(
        detailed_outputs.get('node_detail'),
        yield_df=yield_hist_df,
        area_df=area_hist_df,
        fertilizer_eff_df=fertilizer_eff_df,
        fertilizer_amt_df=fertilizer_amt_df,
        feed_req_df=feed_req_df,
        slaughter_df=slaughter_hist_df,
        livestock_yield_df=livestock_yield_df,
        stock_df=stock_hist_df,
        manure_ratio_df=manure_ratio_df,
        luc_area_df=luc_area_long if isinstance(luc_area_long, pd.DataFrame) else pd.DataFrame(),
        feed_amount_df=feed_override_df,
        grass_share_df=grass_share_df,
        feed_eff_df=feed_eff_df,
        scenario_ctx=scenario_ctx,
        m49_by_country=universe.m49_by_country,
    )
    if isinstance(production_summary_df, pd.DataFrame) and len(production_summary_df):
        detailed_outputs['production_summary'] = production_summary_df
    if isinstance(luc_emis_summary, pd.DataFrame) and len(luc_emis_summary):
        detailed_outputs['luc_emissions_summary'] = luc_emis_summary
    # LUC 土地面积相关输出
    if isinstance(luc_period_start_df, pd.DataFrame) and len(luc_period_start_df):
        detailed_outputs['luc_land_area_period_start'] = luc_period_start_df  # 各期初土地面积
    if isinstance(luc_period_end_df, pd.DataFrame) and len(luc_period_end_df):
        detailed_outputs['luc_land_area_period_end'] = luc_period_end_df  # 各期末土地面积
    if isinstance(luc_change_long, pd.DataFrame) and len(luc_change_long):
        detailed_outputs['luc_land_area_change_by_period'] = luc_change_long  # 各期面积变化量
    if isinstance(luc_transitions_hist, pd.DataFrame) and len(luc_transitions_hist):
        detailed_outputs['luc_transitions_all_records'] = luc_transitions_hist  # 所有LUC转换记录
    if isinstance(coarse_transitions_df, pd.DataFrame) and len(coarse_transitions_df):
        detailed_outputs['luc_period_transitions'] = coarse_transitions_df  # 逐期森林⇔耕地/草地转换量
    _log_step("\u8be6\u7ec6\u8f93\u51fa DataFrame \u5df2\u751f\u6210")
    emission_detail_df: Optional[pd.DataFrame] = None
    if can_emit:
        emission_detail_df = detailed_outputs.get('emissions_detail')
        _log_step(f"[MODEL\u8f93\u51fa] emission_detail_df: {len(emission_detail_df) if isinstance(emission_detail_df, pd.DataFrame) else 'None'} \u884c")
        _log_step(f"  \u8bf4\u660e: MODEL\u8f93\u51fa\u4ec5\u5305\u542bGFIRE\u548cLUC\u6392\u653e\uff0c\u4e0d\u5305\u542bGLE/GCE\u755c\u7267\u4e1a\u548c\u4f5c\u7269\u6392\u653e")
        _log_step(f"  GLE/GCE\u6392\u653e\u5c06\u5728\u540e\u7eed\u901a\u8fc7FAO\u6a21\u5757\u5355\u72ec\u8ba1\u7b97\u5e76\u5408\u5e76")
        if isinstance(emission_detail_df, pd.DataFrame) and len(emission_detail_df) > 0:
            _log_step(f"  - \u5217\u540d: {list(emission_detail_df.columns)}")
            if 'Process' in emission_detail_df.columns:
                unique_processes = emission_detail_df['Process'].unique()
                _log_step(f"  - \u5305\u542b\u6392\u653e\u8fc7\u7a0b: {len(unique_processes)} \u4e2a - {unique_processes[:5].tolist()}")
            if 'value' in emission_detail_df.columns:
                _log_step(f"  - MODEL\u603b\u6392\u653e\u91cf(GFIRE+LUC): {emission_detail_df['value'].sum():.2f} tCO2e")
            if 'M49_Country_Code' not in emission_detail_df.columns and 'M49' in emission_detail_df.columns:
                emission_detail_df = emission_detail_df.copy()
                emission_detail_df['M49_Country_Code'] = emission_detail_df['M49']
            elif 'M49_Country_Code' in emission_detail_df.columns and 'M49' in emission_detail_df.columns:
                missing_mask = emission_detail_df['M49_Country_Code'].isna() & emission_detail_df['M49'].notna()
                if missing_mask.any():
                    emission_detail_df = emission_detail_df.copy()
                    emission_detail_df.loc[missing_mask, 'M49_Country_Code'] = emission_detail_df.loc[missing_mask, 'M49']
            detailed_outputs['emissions_detail'] = emission_detail_df
        if isinstance(gfire_combined_df, pd.DataFrame) and len(gfire_combined_df):
            _log_step(f"  - GFIRE\u6392\u653e: {len(gfire_combined_df)} \u884c")
            gfire_years = sorted(gfire_combined_df['year'].unique()) if 'year' in gfire_combined_df.columns else []
            _log_step(f"    \u5e74\u4efd: {min(gfire_years)}-{max(gfire_years)} ({len(gfire_years)}\u4e2a)" if gfire_years else "")
            if isinstance(emission_detail_df, pd.DataFrame) and len(emission_detail_df):
                emission_detail_df = pd.concat([emission_detail_df, gfire_combined_df], ignore_index=True)
            else:
                emission_detail_df = gfire_combined_df.copy()
            detailed_outputs['emissions_detail'] = emission_detail_df
        if isinstance(luc_emis_detail, pd.DataFrame) and len(luc_emis_detail):
            _log_step(f"  - LUC\u6392\u653e: {len(luc_emis_detail)} \u884c")
            luc_years = sorted(luc_emis_detail['year'].unique()) if 'year' in luc_emis_detail.columns else []
            _log_step(f"    \u5e74\u4efd: {min(luc_years)}-{max(luc_years)} ({len(luc_years)}\u4e2a)" if luc_years else "")
            if not isinstance(emission_detail_df, pd.DataFrame) or emission_detail_df.empty:
                emission_detail_df = luc_emis_detail.copy()
            else:
                emission_detail_df = pd.concat([emission_detail_df, luc_emis_detail], ignore_index=True)
            detailed_outputs['emissions_detail'] = emission_detail_df
    if can_emit:
        if isinstance(emission_detail_df, pd.DataFrame) and len(emission_detail_df):
            _log_step("=" * 80)
            _log_step("[Stage 1] MODEL\u8f93\u51fa\u6392\u653e\u6c47\u603b (GFIRE + LUC)")
            _log_step("=" * 80)
            _log_step(f"emission_detail_df: {len(emission_detail_df)} \u884c")
            _log_step(f"\u5217\u540d: {list(emission_detail_df.columns)}")
            if 'year' in emission_detail_df.columns:
                years = sorted(emission_detail_df['year'].unique())
                _log_step(f"\u5e74\u4efd: {min(years)}-{max(years)} ({len(years)}\u4e2a)")
            if len(emission_detail_df) > 0 and 'value' in emission_detail_df.columns:
                _log_step(f"\u603b\u6392\u653e\u91cf: {emission_detail_df['value'].sum():.2f}")
            # 先从model输出汇总(GFIRE + LUC等)
            emis_sum = summarize_emissions_from_detail(emission_detail_df,
                                                       process_meta_map=universe.process_meta,
                                                       allowed_years=active_years,
                                                       dict_v3_path=paths.dict_v3_path)
            _log_step("[Stage 1] \u6c47\u603b\u5b8c\u6210")
            # 然后合并FAO模块的排放(GCE, GLE等)
            if fao_results is not None and isinstance(fao_results, dict):
                _log_step("=" * 80)
                _log_step("[Stage 2] FAO\u6a21\u5757\u6392\u653e\u6c47\u603b (GLE + GCE + GSOIL)")
                _log_step("=" * 80)
                _log_step(f"fao_results\u6a21\u5757: {list(fao_results.keys())}")
                for module_name in fao_results.keys():
                    module_data = fao_results[module_name]
                    if module_data is None:
                        _log_step(f"  {module_name}: None")
                    elif isinstance(module_data, pd.DataFrame):
                        _log_step(f"  {module_name}: {len(module_data)} \u884c")
                    elif isinstance(module_data, list):
                        total_rows = sum(len(df) if isinstance(df, pd.DataFrame) else 0 for df in module_data)
                        _log_step(f"  {module_name}: \u5217\u8868 {len(module_data)} \u9879, \u603b\u8ba1 {total_rows} \u884c")
                    else:
                        _log_step(f"  {module_name}: {type(module_data)}")
                fao_emis_sum = summarize_emissions(fao_results, extra_emis=None,
                                                   process_meta_map=universe.process_meta,
                                                   allowed_years=set(active_years) if active_years else None,
                                                   dict_v3_path=paths.dict_v3_path,
                                                   production_df=production_summary_df)
                _log_step(f"[Stage 2] FAO\u6c47\u603b\u5b8c\u6210")
                _log_step(f"\u6c47\u603b\u7ed3\u679c keys: {list(fao_emis_sum.keys())}")
                for key, df in fao_emis_sum.items():
                    if isinstance(df, pd.DataFrame):
                        _log_step(f"  {key}: {len(df)} \u884c")
                # 合并emis_sum和fao_emis_sum
                _log_step("\u5408\u5e76Stage 1 (MODEL)\u548cStage 2 (FAO)\u7ed3\u679c...")
                for key in fao_emis_sum:
                    if key not in emis_sum:
                        emis_sum[key] = fao_emis_sum[key]
                        _log_step(f"  + \u6dfb\u52a0: {key} ({len(emis_sum[key]) if isinstance(emis_sum[key], pd.DataFrame) else 'N/A'} \u884c)")
                    elif isinstance(emis_sum[key], pd.DataFrame) and isinstance(fao_emis_sum[key], pd.DataFrame):
                        # 合并同名的DataFrame
                        rows_before = len(emis_sum[key])
                        emis_sum[key] = pd.concat([emis_sum[key], fao_emis_sum[key]], ignore_index=True)
                        _log_step(f"  + \u5408\u5e76: {key} ({rows_before} + {len(fao_emis_sum[key])} = {len(emis_sum[key])} \u884c)")
                # [Final Validation] Check 2080 livestock data completeness
                _log_step("=" * 80)
                _log_step("[Final Validation] 2080\u5e74\u755c\u7267\u4e1a\u6392\u653e\u6570\u636e\u5b8c\u6574\u6027\u68c0\u67e5")
                _log_step("=" * 80)
                target_processes = ['Enteric fermentation', 'Manure management', 'Manure left on pasture', 'Manure applied to soils']
                if 'detail' in emis_sum and isinstance(emis_sum['detail'], pd.DataFrame):
                    detail_df = emis_sum['detail']
                    year_2080 = detail_df[detail_df['year'] == 2080] if 'year' in detail_df.columns else pd.DataFrame()
                    _log_step(f"2080\u5e74\u603b\u8bb0\u5f55\u6570: {len(year_2080)}")
                    if len(year_2080) > 0 and 'Process' in year_2080.columns:
                        for proc in target_processes:
                            proc_data = year_2080[year_2080['Process'] == proc]
                            _log_step(f"  {proc}: {len(proc_data)} \u884c")
                            if len(proc_data) > 0 and 'value' in proc_data.columns:
                                total = proc_data['value'].sum()
                                non_zero = len(proc_data[proc_data['value'] > 0])
                                _log_step(f"    \u603b\u6392\u653e: {total:.2f} tCO2e, \u975e\u96f6\u56fd\u5bb6: {non_zero}")
                else:
                    _log_step("\u6ce8\u610f: emis_sum['detail']\u4e0d\u5b58\u5728\u6216\u4e0d\u662fDataFrame")
            # 调试：显示每个汇总的行数
            _log_step("=" * 80)
            _log_step("\u6700\u7ec8\u6c47\u603b\u7ed3\u679c\u7edf\u8ba1:")
            for k, df in emis_sum.items(): 
                if isinstance(df, pd.DataFrame):
                    _log_step(f"  {k}: {len(df)} \u884c")
        elif fao_results is not None:
            _log_step("\u6392\u653e\u8be6\u7ec6\u6570\u636e\u4e3a\u7a7a\uff0c\u4f7f\u7528 FAO \u6a21\u5757\u8f93\u51fa\u6c47\u603b")
            _log_step(f"FAO 结果类型: {type(fao_results)}")
            if isinstance(fao_results, dict):
                _log_step(f"FAO 结果包含的模块: {list(fao_results.keys())}")
                for module_name, module_data in fao_results.items():
                    if module_data is None:
                        _log_step(f"  - {module_name}: None")
                    elif isinstance(module_data, pd.DataFrame):
                        _log_step(f"  - {module_name}: DataFrame with {len(module_data)} 行")
                    elif isinstance(module_data, list):
                        total_rows = sum(len(df) if isinstance(df, pd.DataFrame) else 0 for df in module_data)
                        _log_step(f"  - {module_name}: List with {len(module_data)} items, 总计 {total_rows} 行")
                    elif isinstance(module_data, dict):
                        _log_step(f"  - {module_name}: Dict with {len(module_data)} keys")
                        if module_name == 'GCE':
                            _log_step(f"    GCE包含的过程: {list(module_data.keys())}")
                            for proc_name, proc_df in module_data.items():
                                if isinstance(proc_df, pd.DataFrame):
                                    _log_step(f"      * {proc_name}: {len(proc_df)} 行, 列: {list(proc_df.columns)[:5]}...")
                    else:
                        _log_step(f"  - {module_name}: {type(module_data)}")
            if 'GCE' in fao_results and isinstance(fao_results['GCE'], dict):
                pass
            emis_sum = summarize_emissions(fao_results, extra_emis=gfire_combined_df,
                                           process_meta_map=universe.process_meta,
                                           allowed_years=set(active_years) if active_years else None,
                                           dict_v3_path=paths.dict_v3_path,
                                           production_df=production_summary_df)
            _log_step("summarize_emissions 调用完成，检查结果...")
            total_emis_rows = sum(len(df) if isinstance(df, pd.DataFrame) else 0 for df in emis_sum.values())
            _log_step(f"汇总结果总行数: {total_emis_rows}")
            for k, df in emis_sum.items():
                if isinstance(df, pd.DataFrame):
                    _log_step(f"  - {k}: {len(df)} 行")
            if active_years:
                filt_years = set(active_years)
                for k, df in list(emis_sum.items()):
                    if isinstance(df, pd.DataFrame) and 'year' in df.columns:
                        before_filter = len(df)
                        emis_sum[k] = df[df['year'].isin(filt_years)].reset_index(drop=True)
                        after_filter = len(emis_sum[k])
                        if before_filter != after_filter:
                            _log_step(f"  年份过滤 {k}: {before_filter} → {after_filter} 行")
            _log_step("\u6392\u653e\u6c47\u603b\u5b8c\u6210\uff08FAO \u8f93\u51fa\uff09")
        else:
            _log_step("\u6392\u653e\u6c47\u603b\u8df3\u8fc7\uff0c\u6ca1\u6709\u53ef\u7528\u7684\u6392\u653e\u6570\u636e")
    else:
        _log_step("\u5df2\u751f\u6210\u5e02\u573a\u548c DS \u6c47\u603b\uff08\u6392\u653e\u8df3\u8fc7\uff09")
# 10) Write outputs
    ds_dir = outdir / "DS"
    ds_dir.mkdir(parents=True, exist_ok=True)
    outdir_str = str(outdir)
    ds_dir_str = str(ds_dir)
    if can_emit:
        emis_dir = outdir / "Emis"
        emis_dir.mkdir(parents=True, exist_ok=True)
        emis_dir_str = str(emis_dir)
        # ???????Excel???sheet
        if emis_sum:
            # ????????????DataFrame
            has_data = any(isinstance(df, pd.DataFrame) and not df.empty for df in emis_sum.values())
            if has_data:
                excel_path = os.path.join(emis_dir_str, "emissions_summary.xlsx")
                _log_step(f"??????: {excel_path}")
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        sheet_name_map = {
                            'by_ctry_proc_comm': 'By_Country_Process_Item',
                            'by_ctry_proc': 'By_Country_Process',
                            'by_ctry': 'By_Country',
                            'long': 'Detail_Long'
                            # ????? 'by_year' ????????????
                        }
                        for k, df in emis_sum.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                sheet_name = sheet_name_map.get(k, k)[:31]  # Excel sheet????31???
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                                _log_step(f"  - Sheet '{sheet_name}': {len(df)} ?")
                    _log_step("????Excel???????")
                except Exception as e:
                    _log_step(f"Excel写入失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise  # 抛出异常而不是fallback到CSV
            else:
                _log_step("WARNING: agriculture emissions result is empty")
    else:
        _log_step("Gurobi ??????? Emis ?????? DS ??")
    
    # ✅ 定义_add_m49_column函数（需要在调用前定义，原本定义在调用后导致UnboundLocalError）
    iso3_to_country = {v: k for k, v in universe.iso3_by_country.items()}
    m49_by_country = universe.m49_by_country or {}
    iso3_to_m49 = {iso: m49_by_country.get(ctry) for iso, ctry in iso3_to_country.items()}
    
    def _add_m49_column(df_in: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df_in, pd.DataFrame):
            return df_in
        df_out = df_in.copy()
        if 'M49_Country_Code' not in df_out.columns:
            if 'country' in df_out.columns:
                df_out['M49_Country_Code'] = df_out['country'].map(m49_by_country)
            elif 'iso3' in df_out.columns:
                df_out['M49_Country_Code'] = df_out['iso3'].map(iso3_to_m49)
        if 'M49_Country_Code' in df_out.columns:
            def _fmt_m49(x):
                """标准化M49格式为'xxx（带前导单引号）"""
                if pd.isna(x) or x is None or x == '':
                    return ''
                try:
                    m49_int = int(float(str(x).strip()))
                    return f"'{m49_int}"
                except Exception:
                    s = str(x).strip()
                    if not s.startswith("'"):
                        return f"'{s}"
                    return s
            df_out['M49_Country_Code'] = df_out['M49_Country_Code'].apply(_fmt_m49)
            cols = df_out.columns.tolist()
            cols = ['M49_Country_Code'] + [c for c in cols if c != 'M49_Country_Code']
            df_out = df_out[cols]
        if 'iso3' in df_out.columns:
            df_out = df_out.drop(columns=['iso3'])
        return df_out
    
    if market_sum is not None and len(market_sum):
        # 添加M49_Country_Code列并重排列顺序
        market_sum = _add_m49_column(market_sum)
        # 按国家、年份、商品排序，便于检查重复
        if 'M49_Country_Code' in market_sum.columns:
            market_sum = market_sum.sort_values(['M49_Country_Code', 'year', 'commodity']).reset_index(drop=True)
        market_sum.to_csv(os.path.join(ds_dir_str, "market_summary.csv"), index=False, encoding='utf-8-sig')
    
    # 成本汇总已在前面通过generate_cost_summary直接写入文件
    # cost_summary.csv 文件路径: outdir / "cost_summary.csv"

    for name, df in (detailed_outputs or {}).items():
        if df is not None and len(df):
            # Skip saving standalone GFIRE emissions files to DS
            if str(name).startswith('gfire_emissions'):
                continue
            # emission 相关汇总已在 EMIS 目录生成，不再重复写入 DS
            if 'emission' in str(name).lower():
                continue
            fname_map = {
                'node_detail': 'detailed_node_summary.csv',
                'country_year_summary': 'country_year_summary.csv',
                'nutrition_per_capita': 'nutrition_per_capita.csv',
                'land_use_LUH2_summary': 'luc_land_area_LUH2base_start.csv',  # LUH2基准期初土地面积
                'luc_land_area_period_start': 'luc_land_area_period_start.csv',  # 各期初土地面积
                'luc_land_area_period_end': 'luc_land_area_period_end.csv',  # 各期末土地面积
                'luc_land_area_change_by_period': 'luc_land_area_change_by_period.csv',  # 各期面积变化量
                'luc_transitions_all_records': 'luc_transitions_all_records.csv',  # 所有LUC转换记录
                'luc_period_transitions': 'luc_period_transitions.csv',  # 逐期森林⇔耕地/草地转换量
                'production_summary': 'production_summary.csv',
            }
            fname = fname_map.get(name, f"{name}.csv")
            df_out = _add_m49_column(df)
            df_out.to_csv(os.path.join(ds_dir_str, fname), index=False, encoding='utf-8-sig')
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
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    print(f"\n{'='*100}")
    print(f"⏱️  情景 {scenario_id} 运行完成！")
    print(f"⏱️  总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"结果保存在: {outdir_str}")
    print(f"{'='*100}\n")
    
    return outdir_str
# -------------- CLI --------------
def main():
    if CFG['run_mode']=='scenario':
        # Batch scenarios from Scenario sheet
        cfg = ScenarioConfig()
        universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
        effects = load_scenarios(paths.scenario_config_xlsx, universe, sheet='Scenario')
        ids = sorted({e.scenario_id for e in effects}) or ['BASE']
        
        # ✅ 修改：传入effects而非提前计算的params，让run_one_pipeline在加载完数据后再计算
        for sid in ids:
            run_one_pipeline(
                paths,
                pre_macc_e0=CFG['premacc_e0'],
                scenario_id=sid,
                scenario_params=None,  # 不再提前计算
                scenario_effects=effects,  # ✅ 传入effects
                solve=CFG['solve'],
                use_fao_modules=CFG['use_fao_modules'],
                future_last_only=CFG['future_last_only'],
                use_linear=CFG['use_linear_model']
            )
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
        production_stats = load_production_statistics(paths, universe)
        # ✅ 所有默认DataFrame都包含M49_Country_Code列
        stock_hist_df = production_stats.get('stock', pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','stock_head']))
        feed_override_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','commodity','feed_t'])
        grass_share_df = pd.DataFrame(columns=['M49_Country_Code','country','iso3','year','grass_feed_share'])
        if isinstance(stock_hist_df, pd.DataFrame) and not stock_hist_df.empty:
            try:
                feed_outputs = build_feed_demand_from_stock(
                    stock_df=stock_hist_df,
                    universe=universe,
                    maps=item_maps,
                    paths=paths,
                    years=universe.years,
                    conversion_multiplier={}
                )
                if isinstance(feed_outputs.crop_feed_demand, pd.DataFrame) and not feed_outputs.crop_feed_demand.empty:
                    feed_override_df = feed_outputs.crop_feed_demand
                if isinstance(feed_outputs.species_dm_detail, pd.DataFrame) and not feed_outputs.species_dm_detail.empty:
                    detail = feed_outputs.species_dm_detail.copy()
                    if 'iso3' not in detail.columns or detail['iso3'].isna().all():
                        detail['iso3'] = detail['country'].map(universe.iso3_by_country)
                    grp = detail.groupby(['country','iso3','year'], as_index=False)[['grass_dm_kg','dm_total_kg']].sum()
                    grp['grass_feed_share'] = np.where(grp['dm_total_kg'] > 0, grp['grass_dm_kg'] / grp['dm_total_kg'], np.nan)
                    grass_share_df = grp[['country','iso3','year','grass_feed_share']]
            except Exception:
                feed_override_df = pd.DataFrame(columns=['country','iso3','year','commodity','feed_t'])
                grass_share_df = pd.DataFrame(columns=['country','iso3','year','grass_feed_share'])
        else:
            grass_share_df = pd.DataFrame(columns=['country','iso3','year','grass_feed_share'])
        trade_imports, trade_exports = load_trade_import_export(paths, universe)
        # Demand from FBS
        fbs_comp = build_demand_components_from_fbs(
            paths.fbs_csv,
            universe,
            production_lookup=prod_lookup,
            latest_hist_prod=latest_hist_prod,
            feed_override_df=feed_override_df,
        )
        apply_fbs_components_to_nodes(data.nodes, fbs_comp, feed_efficiency=cfg.feed_efficiency)
        latest_hist_demand = apply_trade_baseline_to_nodes(
            data.nodes,
            hist_end=cfg.years_hist_end,
            trade_imports=trade_imports,
            trade_exports=trade_exports,
        )
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
        run_one_pipeline(
            paths,
            pre_macc_e0=CFG['premacc_e0'],
            scenario_id=sid,
            scenario_params=scen,
            scenario_effects=None,  # single模式不使用effects
            solve=CFG['solve'],
            use_fao_modules=CFG['use_fao_modules'],
            future_last_only=CFG['future_last_only'],
            use_linear=CFG['use_linear_model']
        )
if __name__ == '__main__':
    import sys
    # 支持命令行参数覆盖配置
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].upper()
        if mode_arg in ['BASE', 'SINGLE', 'SCENARIO', 'MC']:
            CFG['run_mode'] = 'single' if mode_arg in ['BASE', 'SINGLE'] else mode_arg.lower()
            print(f"[CLI] 使用命令行参数: run_mode='{CFG['run_mode']}'")
    main()
