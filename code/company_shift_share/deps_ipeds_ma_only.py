"""Build degree-filtered IPEDS parquet(s) for shift-share inputs."""

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
    parser = argparse.ArgumentParser(description="Build degree-filtered IPEDS parquet for shift-share inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override raw IPEDS completions .dta path.")
    parser.add_argument("--output", type=Path, default=None, help="Override output parquet path.")
    parser.add_argument(
        "--include-bachelors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include Bachelor's in addition to Master's programs (awlevel 5 and 7).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_cfg_section(cfg, "paths")
    ipeds_cfg = get_cfg_section(cfg, "ipeds_ma_only")

    input_path = args.input or paths.get("ipeds_completions_raw")
    include_bachelors = (
        args.include_bachelors
        if args.include_bachelors is not None
        else bool(ipeds_cfg.get("include_bachelors", False))
    )
    if args.output is not None:
        output_path = args.output
    elif include_bachelors:
        output_path = paths.get("ipeds_ma_ba_only") or paths.get("ipeds_ma_only")
    else:
        output_path = paths.get("ipeds_ma_only")
    if not input_path or not output_path:
        raise ValueError("Missing input/output path; set in config or CLI args.")

    min_year = int(ipeds_cfg.get("min_year", 2002))
    min_cip = int(ipeds_cfg.get("min_cip", 10000))
    master_only = bool(ipeds_cfg.get("master_only", True))

    raw = pd.read_stata(str(input_path), convert_categoricals=False)
    clean = raw[(raw["majornum"] == 1) & (raw["ctotalt"] > 0) & (raw["year"] >= min_year)].copy()
    if include_bachelors:
        if "awlevel" in clean.columns:
            clean = clean[clean["awlevel"].isin([5, 7])].copy()
        elif {"master", "bachelor"}.issubset(set(clean.columns)):
            clean = clean[(clean["master"] == 1) | (clean["bachelor"] == 1)].copy()
        elif "master" in clean.columns:
            # Fallback: if bachelor's indicator is unavailable, keep prior master's behavior.
            clean = clean[clean["master"] == 1].copy()
    elif master_only:
        clean = clean[clean["master"] == 1].copy()
    degree_filtered = clean[clean["cipcode"] >= min_cip].copy()

    degree_filtered["intlcat"] = np.where(
        pd.isnull(degree_filtered["share_intl"]) == 1,
        "null",
        np.where(
            degree_filtered["share_intl"] == 0,
            "No International Students",
            np.where(
                degree_filtered["share_intl"] >= 0.7,
                "70%+ International",
                np.where(degree_filtered["share_intl"] >= 0.3, "30-69% International", "1-29% International"),
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
    missing_cols = [c for c in cols if c not in degree_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in IPEDS data: {missing_cols}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    degree_filtered[cols].to_parquet(output_path, index=False)
    scope = "masters+bachelors" if include_bachelors else "masters-only"
    print(f"Wrote {output_path} ({scope})")


if __name__ == "__main__":
    main()
