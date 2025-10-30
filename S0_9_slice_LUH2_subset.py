# -*- coding: utf-8 -*-
"""
S0.9 — 生成 LUH2 子集（2010-2020）
----------------------------------
从默认 LUH2 states / transitions NetCDF 中切出 2010-2020 年，
写到同目录下的新文件，便于后续快速读取。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import xarray as xr

DEFAULT_BASE = Path(r"R:\Data\Food\LUH2")
DEFAULT_STATES = DEFAULT_BASE / "LUH2_GCB2019_states.nc4"
DEFAULT_TRANSITIONS = DEFAULT_BASE / "LUH2_GCB2019_transitions.nc4"
YEAR_RANGE: Tuple[int, int] = (2010, 2020)
CHUNKS = {"time": 20}
COMPRESSION = dict(zlib=True, complevel=4)


def _slice_dataset(src: Path, dst: Path, year_range: Tuple[int, int]) -> None:
    if not src.exists():
        print(f"[WARN] 源文件不存在：{src}")
        return

    print(f"[INFO] 读取 {src} …")
    ds = xr.open_dataset(src, chunks=CHUNKS if "time" in CHUNKS else None)
    try:
        years = year_range
        if "year" in ds.coords:
            sliced = ds.sel(year=slice(years[0], years[1]))
        elif "time" in ds.coords:
            sliced = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
        else:
            raise ValueError("数据集中缺少 year/time 维度，无法切片")

        print(f"[INFO] 裁剪后形状：{sliced.dims}")
        sliced = sliced.load()

        encoding = {var: COMPRESSION for var in sliced.data_vars}
        tmp_path = dst.with_suffix(dst.suffix + ".tmp")
        print(f"[INFO] 写出 {dst} …")
        sliced.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, dst)
        print(f"[INFO] 完成：{dst}")
    finally:
        ds.close()


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="切片 LUH2 states/transitions 到 2010-2020")
    ap.add_argument("--states", type=Path, default=DEFAULT_STATES,
                    help="原始 LUH2 states 文件路径")
    ap.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS,
                    help="原始 LUH2 transitions 文件路径")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="输出目录（默认与源文件相同）")
    ap.add_argument("--start", type=int, default=YEAR_RANGE[0], help="开始年份")
    ap.add_argument("--end", type=int, default=YEAR_RANGE[1], help="结束年份（含）")
    args = ap.parse_args(list(argv) if argv is not None else None)

    year_range = (min(args.start, args.end), max(args.start, args.end))

    def target_path(src: Path, suffix: str) -> Path:
        out_dir = args.out_dir or src.parent
        name = f"{src.stem}_{year_range[0]}_{year_range[1]}{suffix}"
        return Path(out_dir) / name

    states_dst = target_path(args.states, args.states.suffix)
    transitions_dst = target_path(args.transitions, args.transitions.suffix)

    _slice_dataset(args.states, states_dst, year_range)
    _slice_dataset(args.transitions, transitions_dst, year_range)


if __name__ == "__main__":
    main()
