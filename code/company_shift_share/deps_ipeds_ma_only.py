"""Build ipeds_ma_only.parquet from IPEDS completions raw data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    # Allow direct script/notebook execution when repo root is not on PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IPEDS master's-only parquet for shift-share inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override raw IPEDS completions .dta path.")
    parser.add_argument("--output", type=Path, default=None, help="Override output parquet path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_cfg_section(cfg, "paths")
    ipeds_cfg = get_cfg_section(cfg, "ipeds_ma_only")

    input_path = args.input or paths.get("ipeds_completions_raw")
    output_path = args.output or paths.get("ipeds_ma_only")
    if not input_path or not output_path:
        raise ValueError("Missing input/output path; set in config or CLI args.")

    min_year = int(ipeds_cfg.get("min_year", 2002))
    min_cip = int(ipeds_cfg.get("min_cip", 10000))
    master_only = bool(ipeds_cfg.get("master_only", True))

    raw = pd.read_stata(str(input_path), convert_categoricals=False)
    clean = raw[(raw["majornum"] == 1) & (raw["ctotalt"] > 0) & (raw["year"] >= min_year)].copy()
    if master_only:
        clean = clean[clean["master"] == 1].copy()
    ma_only = clean[clean["cipcode"] >= min_cip].copy()

    ma_only["intlcat"] = np.where(
        pd.isnull(ma_only["share_intl"]) == 1,
        "null",
        np.where(
            ma_only["share_intl"] == 0,
            "No International Students",
            np.where(
                ma_only["share_intl"] >= 0.7,
                "70%+ International",
                np.where(ma_only["share_intl"] >= 0.3, "30-69% International", "1-29% International"),
            ),
        ),
    )

    cols = [
        "unitid",
        "cipcode",
        "cnralt",
        "ctotalt",
        "ptotal",
        "pmastrde",
        "share_intl",
        "intlcat",
        "cip2dig",
        "year",
    ]
    missing_cols = [c for c in cols if c not in ma_only.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in IPEDS data: {missing_cols}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ma_only[cols].to_parquet(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
