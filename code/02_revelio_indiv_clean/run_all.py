#!/usr/bin/env python3
"""Run the 02_revelio_indiv_clean pipeline end-to-end."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import root, wrds_out  # noqa: E402



SCRIPT_ORDER = [
    "wrds_users.py",
    "wrds_positions.py",
    "rev_indiv_name2nat.py",
    "rev_indiv_nametrace.py",
    "clean_revelio_institutions.py",
    "rev_users_clean.py",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all scripts in 02_revelio_indiv_clean in the default order."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config for rev_indiv_config.py. "
            "If provided, exports REV_INDIV_CONFIG for child scripts."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these script basenames (e.g., wrds_users.py rev_users_clean.py).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="Skip these script basenames.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining scripts after a failure.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    return parser.parse_args(argv)


def resolve_plan(only: list[str] | None, skip: list[str] | None) -> list[str]:
    available = set(SCRIPT_ORDER)
    only_set = set(only or SCRIPT_ORDER)
    skip_set = set(skip or [])

    unknown = sorted((only_set | skip_set) - available)
    if unknown:
        raise ValueError(f"Unknown scripts: {', '.join(unknown)}")

    return [s for s in SCRIPT_ORDER if s in only_set and s not in skip_set]


def run_script(script: str, script_dir: Path, py: str, env: dict[str, str]) -> int:
    cmd = [py, str(script_dir / script)]
    t0 = time.time()
    print(f"\n=== Running {script} ===")
    rc = subprocess.run(cmd, env=env, cwd=str(script_dir)).returncode
    dt = round(time.time() - t0, 2)
    if rc == 0:
        print(f"=== Success: {script} ({dt}s) ===")
    else:
        print(f"=== Failed: {script} (exit={rc}, {dt}s) ===")
    return rc


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI and interactive use.

    Examples:
        main()
        main(["--config", "configs/rev_indiv_clean_test.yaml"])
        main(["--only", "wrds_users.py", "wrds_positions.py"])
    """
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent

    try:
        plan = resolve_plan(args.only, args.skip)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 2

    if not plan:
        print("No scripts selected to run.")
        return 0

    env = os.environ.copy()
    if args.config:
        cfg_path = str(Path(args.config).expanduser().resolve())
        env["REV_INDIV_CONFIG"] = cfg_path
        print(f"Using config: {cfg_path}")

    print("Run plan:")
    for i, s in enumerate(plan, start=1):
        print(f"{i}. {s}")

    failures: list[tuple[str, int]] = []
    for script in plan:
        rc = run_script(script, script_dir, args.python, env)
        if rc != 0:
            failures.append((script, rc))
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailures:")
        for script, rc in failures:
            print(f"- {script}: exit {rc}")
        return 1

    print("\nAll selected scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
