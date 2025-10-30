from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_input_base() -> str:
    """Return the base directory for FAOSTAT and runtime input files.

    Order of precedence:
    1) Environment variable `NZF_INPUT_DIR`
    2) Project default (Windows path provided by user)
    """
    env = os.environ.get("NZF_INPUT_DIR")
    if env:
        return str(Path(env))
    # Default input path
    return r"G:\\我的云端硬盘\\Work\\Net-zero food\\Code\\input"


def get_src_base() -> str:
    """Return the base directory for configuration/dictionary files (Code/src).

    Order of precedence:
    1) Environment variable `NZF_SRC_DIR`
    2) Project default (Windows path provided by user)
    """
    env = os.environ.get("NZF_SRC_DIR")
    if env:
        return str(Path(env))
    return r"G:\\我的云端硬盘\\Work\\Net-zero food\\Code\\src"


def get_results_base(scenario: Optional[str] = None) -> str:
    """Return the output directory, optionally for a specific scenario.

    Preference order:
      1) Environment variable `NZF_OUTPUT_DIR`
      2) Project default `G:\\我的云端硬盘\\Work\\Net-zero food\\Code\\output`
    """
    env = os.environ.get("NZF_OUTPUT_DIR")
    base = Path(env) if env else Path(r"C:\\Users\\cheng\\MyDrive\\Work\\Net-zero food\\Code\\output")
    if scenario:
        base = base / scenario
    return str(base)
