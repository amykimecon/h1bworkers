"""Wrapper to build foia_sevp_with_person_id.parquet for company_shift_share."""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb as ddb
import sys

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
    from f1_foia.foia_person_id_linkage import run_person_id_linkage
except ModuleNotFoundError:
    # Allow direct script execution from within subdirectories.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
    from f1_foia.foia_person_id_linkage import run_person_id_linkage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FOIA SEVP person_id linkage outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override input parquet path.")
    parser.add_argument("--crosswalk-out", type=Path, default=None, help="Override crosswalk output path.")
    parser.add_argument("--full-out", type=Path, default=None, help="Override full output path.")
    parser.add_argument(
        "--employment-corrected-out",
        type=Path,
        default=None,
        help="Override employment-corrected output path.",
    )
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Override temp directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_cfg_section(cfg, "paths")

    input_path = args.input or paths.get("foia_sevp_combined")
    crosswalk_out_path = args.crosswalk_out or paths.get("foia_person_key_year_crosswalk")
    full_out_path = args.full_out or paths.get("foia_sevp_with_person_id")
    employment_corrected_out_path = args.employment_corrected_out or paths.get("foia_sevp_with_person_id_employment_corrected")
    tmp_dir = args.tmp_dir or paths.get("foia_person_id_tmp_dir")
    if not employment_corrected_out_path and full_out_path:
        full_p = Path(full_out_path)
        employment_corrected_out_path = full_p.with_name(f"{full_p.stem}_employment_corrected{full_p.suffix}")

    if not input_path or not crosswalk_out_path or not full_out_path or not employment_corrected_out_path or not tmp_dir:
        raise ValueError("Missing required paths; set in config or CLI args.")

    con = ddb.connect()
    run_person_id_linkage(
        con=con,
        input_path=input_path,
        crosswalk_out_path=crosswalk_out_path,
        full_out_path=full_out_path,
        employment_corrected_out_path=employment_corrected_out_path,
        tmp_dir=tmp_dir,
    )


if __name__ == "__main__":
    main()
