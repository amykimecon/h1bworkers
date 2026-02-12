"""Run all dependency builders for company_shift_share."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from company_shift_share.config_loader import DEFAULT_CONFIG_PATH


def _run(script: Path, config_path: Path) -> None:
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all upstream dependencies for company_shift_share.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--skip-foia", action="store_true", help="Skip FOIA clean pipeline.")
    parser.add_argument("--skip-rsid", action="store_true", help="Skip RSID/IPEDS geoname crosswalk.")
    parser.add_argument("--skip-ipeds", action="store_true", help="Skip IPEDS master's-only parquet.")
    parser.add_argument("--skip-revelio", action="store_true", help="Skip Revelio transitions/headcounts build.")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH

    base_dir = Path(__file__).resolve().parent
    if not args.skip_foia:
        _run(base_dir / "deps_foia_clean.py", config_path)
        _run(base_dir / "deps_foia_person_id_linkage.py", config_path)
    if not args.skip_rsid:
        _run(base_dir / "deps_rsid_ipeds_cw.py", config_path)
    if not args.skip_ipeds:
        _run(base_dir / "deps_ipeds_ma_only.py", config_path)
    if not args.skip_revelio:
        _run(base_dir / "revelio_school_to_employer.py", config_path)

    print("All dependency steps completed.")


if __name__ == "__main__":
    main()
