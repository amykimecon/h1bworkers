"""Build the local FOIA cleaning and person-linkage outputs for stage 01."""

from __future__ import annotations

import argparse
import os
import sys
import time
from builtins import print as _print
from functools import partial
from pathlib import Path

import duckdb as ddb

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
for _path in (STAGE_DIR, PIPELINE_ROOT, REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from src.config_loader import get_stage_config, load_config
from src.progress_tracker import mark_stage_complete
from src.pipeline_runtime import sanitize_ipykernel_argv

from raw_foia import build_combined_raw_foia
from person_linkage import run_person_id_linkage

print = partial(_print, flush=True)

STAGE_NAME = "01_f1_foia_clean"


def _configure_duckdb_runtime(con: ddb.DuckDBPyConnection, temp_dir: str | Path | None) -> None:
    con.sql(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
    con.sql("PRAGMA preserve_insertion_order=false")
    if temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        escaped_temp_path = str(temp_path).replace("'", "''")
        con.sql(f"PRAGMA temp_directory='{escaped_temp_path}'")


def _assert_parquet_has_columns(
    con: ddb.DuckDBPyConnection,
    parquet_path: str | Path,
    required_cols: list[str],
    *,
    label: str,
) -> None:
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"{label} parquet not found: {parquet_path}")
    escaped_path = str(parquet_path).replace("'", "''")
    cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')"
        ).fetchall()
    ]
    missing = [col for col in required_cols if col not in cols]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _assert_parquet_nonempty(con: ddb.DuckDBPyConnection, parquet_path: str | Path, *, label: str) -> int:
    escaped_path = str(parquet_path).replace("'", "''")
    row_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{escaped_path}')"
        ).fetchone()[0]
    )
    if row_count <= 0:
        raise ValueError(f"{label} is empty: {parquet_path}")
    return row_count


def build_clean_foia(
    config_path: str | Path | None = None,
    pipeline_cfg: dict | None = None,
    testing: bool | None = None,
) -> dict[str, str | int]:
    cfg = pipeline_cfg or load_config(config_path)
    stage_cfg = get_stage_config(cfg, STAGE_NAME)
    build_cfg = cfg.get("build", {})

    raw_foia_dir = stage_cfg.get("raw_foia_dir")
    by_year_cache_dir = stage_cfg.get("by_year_cache_dir")
    combined_raw_parquet = stage_cfg.get("combined_raw_parquet")
    person_id_crosswalk_parquet = stage_cfg.get("person_id_crosswalk_parquet")
    cleaned_person_panel_parquet = stage_cfg.get("cleaned_person_panel_parquet")
    temp_dir = stage_cfg.get("temp_dir")

    required_paths = {
        "raw_foia_dir": raw_foia_dir,
        "by_year_cache_dir": by_year_cache_dir,
        "combined_raw_parquet": combined_raw_parquet,
        "person_id_crosswalk_parquet": person_id_crosswalk_parquet,
        "cleaned_person_panel_parquet": cleaned_person_panel_parquet,
        "temp_dir": temp_dir,
    }
    missing_cfg = [name for name, value in required_paths.items() if not value]
    if missing_cfg:
        raise ValueError(f"Missing required config values for {STAGE_NAME}: {missing_cfg}")

    overwrite = bool(build_cfg.get("overwrite", False))
    required_raw_columns = list(stage_cfg.get("required_raw_columns", ["filename", "student_key", "individual_key", "year"]))
    skip_duplicate_year_dirs = list(stage_cfg.get("skip_duplicate_year_dirs", ["2009"]))
    early_year_relabels = dict(stage_cfg.get("early_year_relabels", {}))
    drop_columns = list(stage_cfg.get("drop_columns", ["birth_date"]))
    linkage_min_year = int(stage_cfg.get("linkage_min_year", 2006))
    correction_year = int(stage_cfg.get("correction_year", 2015))

    con = ddb.connect()
    _configure_duckdb_runtime(con, temp_dir)

    print(f"[{STAGE_NAME}] Building combined raw FOIA parquet")
    build_combined_raw_foia(
        con=con,
        combined_parquet_path=combined_raw_parquet,
        foia_raw_dir=raw_foia_dir,
        foia_byyear_dir=by_year_cache_dir,
        skip_duplicate_year_dirs=skip_duplicate_year_dirs,
        early_year_relabels=early_year_relabels,
        drop_columns=drop_columns,
        overwrite=overwrite,
        verbose=True,
    )

    _assert_parquet_has_columns(
        con,
        combined_raw_parquet,
        required_raw_columns,
        label="Combined raw FOIA output",
    )
    combined_row_count = _assert_parquet_nonempty(
        con,
        combined_raw_parquet,
        label="Combined raw FOIA output",
    )

    print(f"[{STAGE_NAME}] Linking people across years and correcting employer/spell mismatches")
    linkage_stats = run_person_id_linkage(
        con=con,
        input_path=combined_raw_parquet,
        crosswalk_out_path=person_id_crosswalk_parquet,
        employment_corrected_out_path=cleaned_person_panel_parquet,
        tmp_dir=temp_dir,
        min_year=linkage_min_year,
        correction_year=correction_year,
        overwrite=overwrite,
    )

    _assert_parquet_has_columns(
        con,
        person_id_crosswalk_parquet,
        ["individual_key", "year", "person_id"],
        label="Person-id crosswalk output",
    )
    person_id_count = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT person_id)
            FROM read_parquet('{str(person_id_crosswalk_parquet).replace("'", "''")}')
            WHERE person_id IS NOT NULL
            """
        ).fetchone()[0]
    )
    if person_id_count <= 0:
        raise ValueError("Person-id crosswalk did not create any non-null person_id values.")

    _assert_parquet_has_columns(
        con,
        cleaned_person_panel_parquet,
        ["person_id"],
        label="Cleaned FOIA person-panel output",
    )
    cleaned_row_count = _assert_parquet_nonempty(
        con,
        cleaned_person_panel_parquet,
        label="Cleaned FOIA person-panel output",
    )

    results = {
        "combined_raw_parquet": str(combined_raw_parquet),
        "person_id_crosswalk_parquet": str(person_id_crosswalk_parquet),
        "cleaned_person_panel_parquet": str(cleaned_person_panel_parquet),
        "combined_raw_rows": combined_row_count,
        "cleaned_person_panel_rows": cleaned_row_count,
        "person_id_count": person_id_count,
    }
    results.update(linkage_stats)
    return results


def run(
    config_path: str | Path | None = None,
    pipeline_cfg: dict | None = None,
    testing: bool | None = None,
) -> dict[str, str | int]:
    return build_clean_foia(config_path=config_path, pipeline_cfg=pipeline_cfg, testing=testing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 01_f1_foia_clean.")
    parser.add_argument("--config", type=Path, default=None)
    testing_group = parser.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    parser.set_defaults(testing=None)
    args = parser.parse_args(sanitize_ipykernel_argv())
    cfg = load_config(args.config)
    effective_testing = bool(cfg.get("testing", {}).get("enabled", False) if args.testing is None else args.testing)
    t0 = time.perf_counter()
    out = run(config_path=args.config, pipeline_cfg=cfg, testing=args.testing)
    if not effective_testing:
        mark_stage_complete(STAGE_NAME, time.perf_counter() - t0)
    print(f"[{STAGE_NAME}] Done.")
    for key, value in out.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
